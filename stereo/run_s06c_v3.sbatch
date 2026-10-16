#!/bin/bash
#SBATCH --job-name=v2d-s06c3
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6c on a chosen hands_tri variant, WITHOUT editing s06c_mano_fit.py (it hardcodes
# stereo/hands_tri.json -> stereo/mano_fit). The baseline is preserved once, under
# hands_tri.base.json / mano_fit_base, then the variant is installed and 6c re-run; the
# result is copied to mano_fit_<TAG> and the baseline is restored.
# Usage: [TRI=hands_tri_sam3.json TAG=sam3] sbatch run_s06c_v3.sbatch <run_name> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
TRI=${TRI:-hands_tri_sam3.json}
TAG=${TAG:-sam3}
for NAME in "$@"; do
  S="$V2D_ROOT/runs/$NAME/stereo"
  [ -f "$S/$TRI" ] || { echo "$NAME: no $TRI, skipping"; continue; }
  [ -f "$S/hands_tri.base.json" ] || cp "$S/hands_tri.json" "$S/hands_tri.base.json"
  [ -d "$S/mano_fit_base" ] || cp -r "$S/mano_fit" "$S/mano_fit_base"
  cp "$S/$TRI" "$S/hands_tri.json"
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
    -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile \
    python /code/stereo/s06c_mano_fit.py "$NAME" --no_video
  rm -rf "$S/mano_fit_$TAG"
  cp -r "$S/mano_fit" "$S/mano_fit_$TAG"
  cp "$S/hands_tri.base.json" "$S/hands_tri.json"
  rm -rf "$S/mano_fit" && cp -r "$S/mano_fit_base" "$S/mano_fit"
  echo "$NAME: mano_fit_$TAG written; baseline restored"
done
