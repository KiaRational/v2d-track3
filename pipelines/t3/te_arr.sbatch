#!/bin/bash
#SBATCH --job-name=te-obj
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# TrackEverything object tracks, one run per task
set -uo pipefail
module load apptainer/1.4.5 2>/dev/null
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $1)
[ -f /scratch/kghasemz/v2d/runs/$RUN/objects_recon/te_tracks.npz ] && { echo "$RUN exists"; exit 0; }
apptainer exec --nv --cleanenv -B /scratch -B $HOME/v2d-vulcan:/code \
  --env PYTHONPATH=/scratch/kghasemz/v2d/third_party:/scratch/kghasemz/v2d/pyextra/te_stubs,HF_HOME=/scratch/kghasemz/v2d/cache/hf,TORCH_HOME=/scratch/kghasemz/v2d/weights/torch_home,V2D_ROOT=/scratch/kghasemz/v2d \
  /scratch/kghasemz/v2d/sif/trackeverything.sif python /code/stereo/te_obj_tracks.py $RUN 2>&1 | grep --line-buffered -v Warning | tail -3
