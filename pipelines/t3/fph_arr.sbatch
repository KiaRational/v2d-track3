#!/bin/bash
#SBATCH --job-name=fph
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# all FoundationPose candidates, one (run, object, 60-frame chunk) per task: beat1/fph_tasks.txt
set -uo pipefail
source ~/v2d-vulcan/env.sh
read -r RUN OBJ S E <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $1)"
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT v2d_foundation_pose \
  python /code/stereo/fp_hyps.py $RUN --objects $OBJ --start $S --end $E 2>&1 | grep --line-buffered -v "Warning\|INFO" | tail -3
