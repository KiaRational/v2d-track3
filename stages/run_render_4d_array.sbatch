#!/bin/bash
#SBATCH --job-name=v2d-scene4d
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# 4D scene video for one run per array task (gt_world.json must exist for the mocap outlines).
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile python /code/stages/render_4d_scene.py "$NAME" ${RENDER_ARGS:-}
