#!/bin/bash
#SBATCH --job-name=v2d-objv2
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 5 v2 (robust segmentation: validated, re-prompted SAM2) for one episode.
# Usage: sbatch run_objects_v2.sbatch <run_name> [--objects a,b]
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan
python stages/s05_objects_v2.py "$@"
