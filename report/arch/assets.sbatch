#!/bin/bash
#SBATCH --job-name=arch-assets
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
source ~/v2d-vulcan/env.sh; module load ffmpeg/7.1.1 2>/dev/null
cd ~/v2d-vulcan/stereo
python /scratch/kghasemz/v2d/report/arch/make_assets.py 120 2>&1 | grep -v Warning
O=/scratch/kghasemz/v2d/report/arch/assets; V=/scratch/kghasemz/v2d/viz
v=$(ls $V/09_scene4d/evaluation_ep000033_tpfsu.mp4 $V/09_scene4d/evaluation_ep000033_c3bfix.mp4 2>/dev/null | head -1)
ffmpeg -loglevel error -y -ss 6 -i $v -frames:v 1 $O/scene4d.png && echo "wrote scene4d.png from $v"
p=$V/robot_pretty/t10_e33_task2_front34.mp4
for s in 4 9 14; do ffmpeg -loglevel error -y -ss $s -i $p -frames:v 1 -vf "crop=iw*0.6:ih*0.75:iw*0.2:ih*0.1" $O/rollout_$s.png; done; echo wrote rollouts
