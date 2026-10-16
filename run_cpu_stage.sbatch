#!/bin/bash
#SBATCH --job-name=v2d-stage
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Usage: sbatch run_cpu_stage.sbatch stages/<script>.py <viz-subdir> [script args...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
SCRIPT=$1; VIZ_SUB=$2; shift 2
cd ~/v2d-vulcan
python "$SCRIPT" "$@"

# Re-encode OpenCV's mp4v output to H.264 so the videos play anywhere.
for f in "$V2D_VIZ/$VIZ_SUB"/*.mp4; do
  [ -e "$f" ] || continue
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f")
  [ "$codec" = "h264" ] && continue
  ffmpeg -loglevel error -y -i "$f" -c:v libx264 -crf 22 -preset medium -pix_fmt yuv420p "${f%.mp4}.h264.mp4" && mv "${f%.mp4}.h264.mp4" "$f"
done
ls -lh "$V2D_VIZ/$VIZ_SUB" | head -50
