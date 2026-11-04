#!/bin/bash
#SBATCH --job-name=arch-assets
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo
python /scratch/kghasemz/v2d/report/arch/make_assets2.py 2>&1 | grep -v Warning
O=/scratch/kghasemz/v2d/report/arch/assets; V=/scratch/kghasemz/v2d/viz
