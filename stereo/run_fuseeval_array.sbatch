#!/bin/bash
#SBATCH --job-name=v2d-fuseeval
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Object fusion (all variants incl. hand-follow) + mocap evaluation, one run per array task.
# Usage: sbatch --array=0-N run_fuseeval_array.sbatch <runs_list.txt>
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd ~/v2d-vulcan/stereo
python fuse_objects.py "$NAME"
python ../stages/eval_objects.py "$NAME" --variants fp icp fused_free fused fused_sym fused_hand agile agile_hc --align fp
