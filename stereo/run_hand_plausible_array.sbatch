#!/bin/bash
#SBATCH --job-name=v2d-handphys
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Physically plausible hands for one run per task, two versions from the stereo MANO fit:
#   mano_fit_refined      contact refinement only (no memory)
#   mano_fit_mem_refined  hand memory (grasp gaps filled from the held object) + refinement
# then the contact audit, the error/jitter table and a 4D scene video of each.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python"
$D /code/stereo/s06f_contact_refine.py "$NAME" --hands mano_fit --out mano_fit_refined
$D /code/stereo/s06g_hand_memory.py "$NAME"
$D /code/stereo/s06f_contact_refine.py "$NAME" --hands mano_fit_mem --out mano_fit_mem_refined
for H in mano_fit mano_fit_refined mano_fit_mem_refined; do $D /code/stereo/audit_contact.py "$NAME" --hands $H; done
$D /code/stereo/compare_hawor.py "$NAME" --fits mano_fit mano_fit_refined mano_fit_mem_refined --no_video
$D /code/stages/render_4d_scene.py "$NAME" --hands mano_fit_refined
$D /code/stages/render_4d_scene.py "$NAME" --hands mano_fit_mem_refined
