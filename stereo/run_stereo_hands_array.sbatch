#!/bin/bash
#SBATCH --job-name=v2d-sthands
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Stereo hands for one run per array task: 6a (stereo WiLoR) -> 6a_v2 (CLAHE + flipped) -> 6b.
# Usage: sbatch --array=0-N%K run_stereo_hands_array.sbatch <runs_list.txt>
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd ~/v2d-vulcan/stereo
python s06a_stereo_wilor.py "$NAME"
python s06a_v2_stereo_wilor_clahe.py "$NAME"
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile python /code/stereo/s06b_triangulate_align.py "$NAME"
