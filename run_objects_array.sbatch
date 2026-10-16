#!/bin/bash
#SBATCH --job-name=v2d-objects
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Stage 5 (all objects) over the public episodes, one per array task.
# Usage: sbatch --array=0-19%N run_objects_array.sbatch
set -euo pipefail
EPS=(0 2 11 12 13 18 21 22 23 24 26 27 31 39 40 41 42 43 44 45)
NAME=$(printf "public_ep%06d" "${EPS[$SLURM_ARRAY_TASK_ID]}")
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan && python stages/s05_objects.py "$NAME"
