#!/bin/bash
#SBATCH --job-name=v2d-s06b3
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6b v3 (SAM 3 identity + quality gate) in the agile image.
# Usage: [S06B3_ARGS="..."] sbatch run_s06b_v3.sbatch <run_name> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
for NAME in "$@"; do
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
    -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile \
    python /code/stereo/s06b_v3_sam3.py "$NAME" --sam3 --quality --suffix _sam3 ${S06B3_ARGS:-}
done
