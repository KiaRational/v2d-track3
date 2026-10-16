#!/bin/bash
#SBATCH --job-name=v2d-egophi
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# EgoPHI contact/force on our reconstructions. Usage: sbatch run_egophi.sbatch <run> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code \
  -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ -e EGOPHI_ROOT=$V2D_ROOT/egophi \
  -e EGOPHI_DATA_ROOT=$V2D_ROOT/egophi_data/arctic_data -e HACO_RELEASE_ROOT=/nonexistent \
  egophi python /code/stereo/egophi_run.py "$@"
