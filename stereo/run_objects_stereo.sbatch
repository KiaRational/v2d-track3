#!/bin/bash
#SBATCH --job-name=v2d-objstereo
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stereo-ICP object poses -> FP+ICP SE(3) fusion -> mocap evaluation, for the given runs.
# Usage: sbatch run_objects_stereo.sbatch <run> [...]   (SKIP_ICP=1 to reuse stereo_icp/)
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo
python se3_fusion.py
[ -n "${SKIP_ICP:-}" ] || python object_icp.py "$@"
python fuse_objects.py "$@"
for r in "$@"; do
  [[ $r == public_* ]] && python ../stages/eval_objects.py "$r" --variants fp icp fused_free fused fused_sym fused_hand agile agile_hc --align fp || true
done
