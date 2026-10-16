#!/bin/bash
#SBATCH --job-name=v2d-plaus
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Physically plausible grasps for one run per task: s06h (smooth corrections, contacts anchored at
# the observed places, squeeze, no object/table penetration) -> contact audit -> MuJoCo check of the
# stereo fit and of the plausible hands -> 4D video.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd ~/v2d-vulcan/stereo
[ -f "$V2D_ROOT/runs/$NAME/table_plane.json" ] || python table_plane.py "$NAME"
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python"
$D /code/stereo/s06h_grasp_plausible.py "$NAME"
$D /code/stereo/audit_contact.py "$NAME" --hands mano_fit_plaus
SIF=/scratch/kghasemz/franka/containers/python312_trixie.sif
for H in mano_fit mano_fit_plaus; do
  apptainer exec -B /scratch -B $HOME/v2d-vulcan "$SIF" bash -c \
    ". /scratch/kghasemz/v2d/venv-mj/bin/activate && V2D_ROOT=$V2D_ROOT python $HOME/v2d-vulcan/stereo/physics_check.py $NAME --hands $H"
done
$D /code/stages/render_4d_scene.py "$NAME" --hands mano_fit_plaus
