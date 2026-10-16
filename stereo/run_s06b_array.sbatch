#!/bin/bash
#SBATCH --job-name=v2d-s06b
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Stage 6b for one run per array task; keeps the previous output as hands_tri.<BACKUP_TAG>.json.
# Usage: [BACKUP_TAG=v2] sbatch --array=0-N%K run_s06b_array.sbatch <runs_list.txt>
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
F=$V2D_ROOT/runs/$NAME/stereo/hands_tri.json
[ -f "$F" ] && [ ! -f "${F%.json}.${BACKUP_TAG:-prev}.json" ] && cp "$F" "${F%.json}.${BACKUP_TAG:-prev}.json"
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile python /code/stereo/s06b_triangulate_align.py "$NAME" ${S06B_ARGS:-}
