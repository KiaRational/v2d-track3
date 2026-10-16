#!/bin/bash
#SBATCH --job-name=v2d-haptic
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# HaPTIC on the undetected hand frames (s06m_haptic.py; detected frames kept as the stereo fit) + the 4D video
# (teal hands = undetected frames filled from HaPTIC, violet = out of view, optimiser fill).
# Usage: sbatch run_haptic_fill.sbatch <run> [...]   (FILL/OUT to change variants)
set -euo pipefail
source ~/v2d-vulcan/env.sh
FILL=${FILL:-mano_fit_hop_fill}; OUT=${OUT:-mano_fit_hop_haptic}
H=$V2D_ROOT/third_party/haptic
VID=${VID:-$V2D_ROOT/viz/21_haptic_fill}; mkdir -p "$VID"
for NAME in "$@"; do
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" -e PYTHONPATH="$V2D_ROOT/pyextra/haptic" -w "$H" \
    haptic python /code/stereo/s06m_haptic.py "$NAME" --fill "$FILL" --out "$OUT"
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" \
    -e V2D_VIZ="$V2D_ROOT/viz" agile python /code/stages/render_4d_scene.py "$NAME" --hands "$OUT" \
    --out_name "${NAME}_${OUT}.mp4" --label "${LABEL:-stereo hands + HaPTIC on undetected frames}"
  cp "$V2D_ROOT/viz/09_scene4d/${NAME}_${OUT}.mp4" "$VID/"
done
