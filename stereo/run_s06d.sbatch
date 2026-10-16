#!/bin/bash
#SBATCH --job-name=v2d-s06d
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6d: SAM 3 wearer-hand segmentation + identity tracking, in the `sam3` image.
# Usage: [S06D_ARGS="--overwrite"] sbatch run_s06d.sbatch <run_name> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" sam3 python /code/stereo/s06d_sam3_hands.py "$@" ${S06D_ARGS:-}
