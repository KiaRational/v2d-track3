#!/bin/bash
#SBATCH --job-name=v2d-s06a
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6a (stereo WiLoR) for the given run names.
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo && python s06a_stereo_wilor.py "$@"
