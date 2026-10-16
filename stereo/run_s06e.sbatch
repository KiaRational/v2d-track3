#!/bin/bash
#SBATCH --job-name=v2d-s06e
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6e: hand_quality confidence gate applied to an existing hands_tri.json.
# Usage: [S06E_ARGS="--no_mask_iou --out hands_tri_q.json"] sbatch run_s06e.sbatch <run> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo && python s06e_hand_confidence.py "$@" ${S06E_ARGS:-}
