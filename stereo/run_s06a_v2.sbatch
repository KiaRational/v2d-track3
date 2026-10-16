#!/bin/bash
#SBATCH --job-name=v2d-s06a2
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6a v2 (CLAHE + flipped-handedness stereo WiLoR) for the given run names.
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo && python s06a_v2_stereo_wilor_clahe.py "$@"
