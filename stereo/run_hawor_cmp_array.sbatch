#!/bin/bash
#SBATCH --job-name=v2d-hawor
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# HaWoR vs stereo hands (wrist-aligned) + 4D scene videos of HaWoR hands, one run per task.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python"
$D /code/stereo/compare_hawor.py "$NAME"
$D /code/stages/render_4d_scene.py "$NAME" --hands hawor_base
$D /code/stages/render_4d_scene.py "$NAME" --hands hawor_raw
