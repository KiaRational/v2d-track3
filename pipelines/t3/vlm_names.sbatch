#!/bin/bash
#SBATCH --job-name=vlm-names
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# VLM task & object naming (Qwen3-VL via vLLM, GPUs 0,1) for the 20 scored episodes, then SAM 3 check of the VLM
# prompts vs the hand-written table against the masks the pipeline used (GPU 2).
set -uo pipefail
source ~/v2d-vulcan/env.sh
RUNS=$(cat /scratch/kghasemz/v2d/beat1/eval20.txt)
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -e CUDA_VISIBLE_DEVICES=0,1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e XDG_CACHE_HOME=$SLURM_TMPDIR/xdg vllm \
  python3 -m vllm.entrypoints.openai.api_server --model $V2D_ROOT/weights/qwen3-vl-32b-fp8 \
  --served-model-name qwen3-vl-32b --port 8011 --host 127.0.0.1 --max-model-len 24576 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 10}' \
  > $V2D_ROOT/logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VP=$!; trap 'kill $VP 2>/dev/null || true' EXIT
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e CUDA_VISIBLE_DEVICES=2 \
  -e PYTHONPATH=/code/hoi4d/src agile python /code/stages/objects_v2/vlm_objects.py ask $RUNS 2>&1 | grep -v Warning
kill $VP 2>/dev/null
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:$HOME/v2d-vulcan -e V2D_ROOT=$V2D_ROOT \
  sam3 python $HOME/v2d-vulcan/stages/objects_v2/vlm_objects.py verify $RUNS 2>&1 | grep -v Warning
