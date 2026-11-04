#!/bin/bash
#SBATCH --job-name=gifs
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
source ~/v2d-vulcan/env.sh
python /scratch/kghasemz/v2d/report/gifs/make_gifs.py 2>&1 | grep -v Warning
