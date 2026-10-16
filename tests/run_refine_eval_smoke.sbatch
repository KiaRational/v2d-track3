#!/bin/bash
#SBATCH --job-name=v2d-refine-eval-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/kghasemz/v2d/code_tests/refine_eval_smoke_%j.out
# Smoke run on real data without touching the run dir: (1) verify old vs new (--quiet, read-only) in the
# agile image; (2) eval_objects in the host env on a sandbox copy of the run (symlinked inputs), so
# gt_world.json / the report land in code_tests/sandbox_root, not in the real run.
set -euo pipefail
RUN=${1:-public_ep000000}
T=/scratch/kghasemz/v2d/code_tests
SNAP=/scratch/kghasemz/v2d/code_snapshots/v2d-vulcan_post_codex_0930_0306
module load StdEnv/2023 apptainer/1.3.5
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$T/tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
apptainer exec --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,TMPDIR="$TMPDIR",V2D_ROOT=/scratch/kghasemz/v2d \
  -B /home/kghasemz/v2d-vulcan:/code:ro -B /scratch/kghasemz/v2d \
  /scratch/kghasemz/v2d/sif/agile.sif python /code/tests/smoke_refine_eval.py "$RUN" "$SNAP"
[ "${SMOKE_VERIFY_ONLY:-0}" = 1 ] && exit 0
SB=$T/sandbox_root
rm -rf "$SB/runs/$RUN"
mkdir -p "$SB/runs/$RUN"
ln -sfn /scratch/kghasemz/v2d/meshes_obj "$SB/meshes_obj"
for f in /scratch/kghasemz/v2d/runs/$RUN/*; do
  case $(basename "$f") in gt_world.json|eval_*.json) ;; *) ln -s "$f" "$SB/runs/$RUN/";; esac
done
source ~/v2d-vulcan/env.sh
export V2D_ROOT=$SB PYTHONDONTWRITEBYTECODE=1
cd ~/v2d-vulcan/stages
V2D_OBJ_DIR=objects_sam3 python eval_objects.py "$RUN" --variants fp fused_all_sam3 --align fp --out eval_se3.json
V2D_OBJ_DIR=objects_sam3 python eval_objects.py "$RUN" --variants fp fused_all_sam3 --align fp --out eval_sim3.json --sim3
jq -c '.align_mode, (.variants | map_values({s: .sim3_scale, inl: .align_inlier_frac, auc: .official.AUC, sp: .official.SP_SR, mp: .official.MP_SR, miss: .official.missing_frames}))' "$SB/runs/$RUN/eval_se3.json" "$SB/runs/$RUN/eval_sim3.json"
jq -c '.sim3_world_to_mocap.s, .align_mode' "$SB/runs/$RUN/gt_world.json"
python ../stereo/eval_global.py "$RUN" --variants fp fused | head -4
rm -rf "$TMPDIR"
