#!/bin/bash
#SBATCH --job-name=v2d-s06darr
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Stage 6d for one run per array task.
# Usage: sbatch --array=0-N%K run_s06d_array.sbatch <runs_list.txt>
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" sam3 python /code/stereo/s06d_sam3_hands.py "$NAME" ${S06D_ARGS:-}
