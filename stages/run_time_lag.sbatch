#!/bin/bash
#SBATCH --job-name=v2d-timelag
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
set -euo pipefail
source ~/v2d-vulcan/env.sh
python ~/v2d-vulcan/stages/time_lag.py "$@"
