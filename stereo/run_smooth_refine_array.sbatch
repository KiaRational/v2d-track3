#!/bin/bash
#SBATCH --job-name=v2d-smoothref
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Smooth, conservative contact refinement of the stereo MANO fit (no memory) -> mano_fit_smooth,
# then contact audit, error/jitter vs the stereo fit, and a 4D video. One run per task.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python"
$D /code/stereo/s06f_contact_refine.py "$NAME" --hands mano_fit --out mano_fit_smooth
$D /code/stereo/audit_contact.py "$NAME" --hands mano_fit
$D /code/stereo/audit_contact.py "$NAME" --hands mano_fit_smooth
$D /code/stereo/compare_hawor.py "$NAME" --fits mano_fit mano_fit_smooth --no_video
$D /code/stages/render_4d_scene.py "$NAME" --hands mano_fit_smooth
