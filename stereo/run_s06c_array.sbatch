#!/bin/bash
#SBATCH --job-name=v2d-s06c
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Stage 6c (sequence MANO fit to stereo-triangulated joints) in the agile image.
# Usage: sbatch --array=0-N%K run_s06c_array.sbatch <runs_list.txt> [extra s06c args...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1"); shift
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile python /code/stereo/s06c_mano_fit.py "$NAME" "$@"
