#!/bin/bash
#SBATCH --job-name=v2d-objects
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 5: multi-object tracking for one episode run dir.
# Usage: sbatch run_objects.sbatch <run_name> [--objects a,b]
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan
python stages/s05_objects.py "$@"
