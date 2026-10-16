#!/bin/bash
#SBATCH --job-name=v2d-geneoh
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# GeneOH-Diffusion clean-up of the hands (s06k_geneoh.py: unobserved frames empty -> optimiser + GeneOH fill) and
# the 4D video of the cleaned hand-object motion (blue/violet hands = frames that were not detected).
# Usage: sbatch run_geneoh_clean.sbatch <run> [...]   (GENEOH_ARGS="--seeds 0,11,22" etc.; HANDS/OUT to change variants)
set -euo pipefail
source ~/v2d-vulcan/env.sh
HANDS=${HANDS:-mano_fit_hop_phys}; OUT=${OUT:-mano_fit_hop_geneoh}
G=$V2D_ROOT/third_party/GeneOH-Diffusion
VID=${VID:-$V2D_ROOT/viz/19_geneoh_clean}; mkdir -p "$VID"
for NAME in "$@"; do
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" -w "$G" \
    geneoh python /code/stereo/s06k_geneoh.py "$NAME" --hands "$HANDS" --out "$OUT" ${GENEOH_ARGS:-}
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" \
    -e V2D_VIZ="$V2D_ROOT/viz" agile python /code/stages/render_4d_scene.py "$NAME" --hands "$OUT" \
    --out_name "${NAME}_${OUT}.mp4" --label "${LABEL:-GeneOH-cleaned hands}"
  cp "$V2D_ROOT/viz/09_scene4d/${NAME}_${OUT}.mp4" "$VID/"
done
