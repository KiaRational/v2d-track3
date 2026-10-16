#!/bin/bash
#SBATCH --job-name=v2d-dump2d
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
set -euo pipefail
source ~/v2d-vulcan/env.sh
docker run --rm -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" \
  agile python /code/stereo/diag/dump_wilor2d.py "$@"
