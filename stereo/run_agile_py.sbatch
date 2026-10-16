#!/bin/bash
#SBATCH --job-name=v2d-agilepy
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Run a python script from ~/v2d-vulcan/stereo in the agile image. Usage: sbatch run_agile_py.sbatch <script.py> [args...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
S=$1; shift
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile python /code/stereo/$S "$@"
