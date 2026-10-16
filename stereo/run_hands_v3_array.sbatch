#!/bin/bash
#SBATCH --job-name=v2d-hands3
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Full SAM 3 hand chain for one run per array task: 6d -> 6a-v3 -> 6b-v3 [-> 6c].
# Usage: sbatch --array=0-N%K run_hands_v3_array.sbatch <runs_list.txt>
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd ~/v2d-vulcan/stereo
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" sam3 python /code/stereo/s06d_sam3_hands.py "$NAME" ${S06D_ARGS:-}
python s06a_v3_sam3_boxes.py "$NAME" ${S06A3_ARGS:-}
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile \
  python /code/stereo/s06b_v3_sam3.py "$NAME" --sam3 --quality --suffix _sam3 ${S06B3_ARGS:-}
python viz_hands_sam3.py "$NAME" --frames
