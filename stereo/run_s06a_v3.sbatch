#!/bin/bash
#SBATCH --job-name=v2d-s06a3
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6a v3: WiLoR prompted with the Stage-6d SAM 3 hand boxes.
# Usage: [S06A3_ARGS="--overwrite"] sbatch run_s06a_v3.sbatch <run_name> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo && python s06a_v3_sam3_boxes.py "$@" ${S06A3_ARGS:-}
