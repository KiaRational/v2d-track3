#!/bin/bash
#SBATCH --job-name=hoi4d-pipe
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# hoi4d end-to-end for one run (README stages 1-11) on top of the existing pipeline outputs.
#   GPUs 0,1: vLLM server (Qwen3-VL-32B-FP8, tensor parallel 2: VLM task.json + review agent); GPU 2: everything else (agile).
# Usage: [STAGES="ingest track verify diagnose task review refine validate export"] sbatch run_pipeline.sbatch <run>
set -euo pipefail
source ~/v2d-vulcan/env.sh
RUN=$1
STAGES=${STAGES:-ingest track verify diagnose task review refine validate export}
OUT=$V2D_ROOT/runs/$RUN/hoi4d
mkdir -p $OUT
has() { [[ " $STAGES " == *" $1 "* ]]; }
AG() { docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT \
  -e V2D_VIZ=$V2D_VIZ -e CUDA_VISIBLE_DEVICES=${AG_GPU:-2} -e PYTHONPATH=/code/hoi4d/src -e V2D_CODE=/code agile "$@"; }
H() { echo "=== hoi4d $1 $RUN ($(date +%T))"; AG python -m hoi4d.cli "$1" $RUN "${@:2}"; }

VLLM_PID=
start_vllm() {
  [ -n "$VLLM_PID" ] && return
  echo "=== starting vLLM ($(date +%T))"
  docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -e CUDA_VISIBLE_DEVICES=0,1 -e HF_HUB_OFFLINE=1 \
    -e VLLM_NO_USAGE_STATS=1 -e XDG_CACHE_HOME=$SLURM_TMPDIR/xdg vllm \
    python3 -m vllm.entrypoints.openai.api_server --model $V2D_ROOT/weights/qwen3-vl-32b-fp8 \
    --served-model-name qwen3-vl-32b --port 8011 --host 127.0.0.1 --max-model-len 32768 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 24}' \
    --enable-auto-tool-choice --tool-call-parser hermes > $V2D_ROOT/logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
}

has ingest && H ingest
has track && H track
if has verify; then  # existing observation critic on the initial objects (imported by diagnose)
  echo "=== verify (refine_agent) initial"
  AG bash -c "cd /code/refine_agent && V2D_OBJ_DIR=objects_sam3 python verify.py $RUN --objects fused_all_sam3 \
    --hands \$(jq -r .initial.hands_variant $OUT/versions.json 2>/dev/null || echo mano_fit_hop_phys) \
    --tag hoi4d_fused_all_sam3 --quiet" || echo "verify failed (diagnostics continue without it)"
fi
has diagnose && H diagnose --version initial
if has task || has review; then start_vllm; fi
has task && H task
has review && H review
has refine && H refine
if has validate; then
  CUR=$(jq -r '._current // "initial"' $OUT/versions.json)
  OV=$(jq -r --arg v $CUR '.[$v].objects_variant' $OUT/versions.json)
  HV=$(jq -r --arg v $CUR '.[$v].hands_variant' $OUT/versions.json)
  H diagnose --version $CUR
  # simulator validation of the final version (MuJoCo hold test; not proof of dynamics, README 10.4)
  SIF=/scratch/kghasemz/franka/containers/python312_trixie.sif
  apptainer exec -B /scratch -B $HOME/v2d-vulcan "$SIF" bash -c ". $V2D_ROOT/venv-mj/bin/activate && \
    V2D_ROOT=$V2D_ROOT python $HOME/v2d-vulcan/stereo/physics_check.py $RUN --variant $OV --hands $HV --stability" \
    && cp $V2D_ROOT/runs/$RUN/stereo/stability_$HV.json $OUT/stability_$CUR.json || echo "stability check failed"
  # overlays + 4D replay, initial vs final
  for V in initial $CUR; do
    OVV=$(jq -r --arg v $V '.[$v].objects_variant' $OUT/versions.json)
    HVV=$(jq -r --arg v $V '.[$v].hands_variant' $OUT/versions.json)
    AG python /code/stages/render_4d_scene.py $RUN --objects $OVV --hands $HVV \
      --label "hoi4d $V" --out_name ${RUN}_hoi4d_${V}.mp4 || echo "render $V failed"
  done
fi
has export && H export
echo "=== done ($(date +%T))"
