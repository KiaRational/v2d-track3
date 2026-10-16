#!/bin/bash
#SBATCH --job-name=v2d-mjtrack
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# MuJoCo physics-based tracking + video for the given runs. Usage: sbatch run_mj_track.sbatch <run> [...] [-- mj_track args]
module load apptainer/1.4.5
source ~/v2d-vulcan/env.sh >/dev/null 2>&1
RUNS=(); ARGS=(); seen=0; for x in "$@"; do if [ "$x" = "--" ]; then seen=1; elif [ $seen = 0 ]; then RUNS+=("$x"); else ARGS+=("$x"); fi; done
SIF=/scratch/kghasemz/franka/containers/python312_trixie.sif
for NAME in "${RUNS[@]}"; do
  H=${PREP_HANDS:-mano_fit_phys}; [ -f "$V2D_ROOT/runs/$NAME/stereo/mj_prep_$H.json" ] || docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT agile python /code/stereo/mj_prep.py "$NAME" --hands $H
  apptainer exec --nv -B /scratch -B $HOME/v2d-vulcan "$SIF" bash -c \
    ". /scratch/kghasemz/v2d/venv-mj/bin/activate && MUJOCO_GL=egl V2D_ROOT=$V2D_ROOT V2D_VIZ=$V2D_VIZ python $HOME/v2d-vulcan/stereo/mj_track.py $NAME --prep mj_prep_$H.json ${ARGS[*]}"
done
