#!/bin/bash
#SBATCH --job-name=v2d-scene4d
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# 4D scene video (hands + objects + scene + mocap ghost) for the given runs.
# Usage: sbatch run_render_4d.sbatch <run> [...]   (RENDER_ARGS="--objects fused_sam3" to change sources)
set -euo pipefail
source ~/v2d-vulcan/env.sh
for NAME in "$@"; do
  if [[ $NAME == public_* ]]; then   # ground truth in our world frame, for the green ghosts
    V2D_OBJ_DIR=objects_sam3 python ~/v2d-vulcan/stages/eval_objects.py "$NAME" \
      --variants fp fused_all_sam3 --align fp --out eval_sam3_render.json > /dev/null
  fi
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
    -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile python /code/stages/render_4d_scene.py "$NAME" ${RENDER_ARGS:-}
done
