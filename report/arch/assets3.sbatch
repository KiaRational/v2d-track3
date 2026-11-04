#!/bin/bash
#SBATCH --job-name=arch-hand4d
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
source ~/v2d-vulcan/env.sh; module load ffmpeg/7.1.1 2>/dev/null
docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT agile \
  python /scratch/kghasemz/v2d/report/arch/make_assets3.py dump 2>&1 | grep -v Warning | tail -3
# frames for the stacked-video input (early / middle / late)
cd ~/v2d-vulcan/stereo && python /scratch/kghasemz/v2d/report/arch/make_assets3.py 2>&1 | grep -v Warning
O=/scratch/kghasemz/v2d/report/arch/assets; v=$V2D_ROOT/runs/evaluation_ep000033/input_undistorted.mp4
for s in 2 6 10; do ffmpeg -loglevel error -y -ss $s -i $v -frames:v 1 -vf scale=676:-1 $O/video_$s.png; done; echo frames
