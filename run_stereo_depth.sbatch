#!/bin/bash
#SBATCH --job-name=v2d-stereo
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Stage 3 (stereo metric depth) for one episode per array task.
# Usage: sbatch --array=0-39%N run_stereo_depth.sbatch [--valid_iters K --hiera 0|1]
set -euo pipefail
source ~/v2d-vulcan/env.sh
mapfile -t NAMES < <(ls "$V2D_ROOT/runs")
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

docker run --rm --gpus all \
  -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_DATA="$V2D_DATA" -e V2D_VIZ="$V2D_VIZ" -e HF_HOME="$HF_HOME" \
  foundation_stereo_official python /code/stages/s03_stereo_depth.py "$NAME" "$@"

f=$V2D_VIZ/03_stereo_depth/$NAME.mp4
ffmpeg -loglevel error -y -i "$f" -c:v libx264 -crf 22 -pix_fmt yuv420p "${f%.mp4}.h264.mp4" && mv "${f%.mp4}.h264.mp4" "$f"
jq -c '{frames, mean_coverage, stereo_over_moge_median_of_medians}' "$V2D_ROOT/runs/$NAME/depth_stereo_stats.json"
