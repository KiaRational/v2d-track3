#!/bin/bash
#SBATCH --job-name=v2d-mjtrack
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# MuJoCo physics-based tracking + video, one run per task (hands: mano_fit_phys).
module load apptainer/1.4.5
source ~/v2d-vulcan/env.sh >/dev/null 2>&1
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
H=${PREP_HANDS:-mano_fit_phys}
[ -f "$V2D_ROOT/runs/$NAME/stereo/mj_prep_$H.json" ] || docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT agile python /code/stereo/mj_prep.py "$NAME" --hands $H
apptainer exec --nv -B /scratch -B $HOME/v2d-vulcan /scratch/kghasemz/franka/containers/python312_trixie.sif bash -c \
  ". /scratch/kghasemz/v2d/venv-mj/bin/activate && MUJOCO_GL=egl V2D_ROOT=$V2D_ROOT V2D_VIZ=$V2D_VIZ python $HOME/v2d-vulcan/stereo/mj_track.py $NAME --prep mj_prep_$H.json"
