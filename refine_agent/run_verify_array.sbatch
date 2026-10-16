#!/bin/bash
#SBATCH --job-name=v2d-verify
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Verifier (critic) of one episode's 4D reconstruction per task. Usage: sbatch --array=.. run_verify_array.sbatch <list> [verify args]
set -euo pipefail
source ~/v2d-vulcan/env.sh
L=$1; shift
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$L")
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT agile \
  python /code/refine_agent/verify.py "$NAME" "$@"
