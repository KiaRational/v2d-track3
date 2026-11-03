#!/bin/bash
#SBATCH --job-name=figs
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
source ~/v2d-vulcan/env.sh; module load ffmpeg/7.1.1 2>/dev/null
python /scratch/kghasemz/v2d/report/make_figures.py ${FIGS:-} 2>&1 | grep -v Warning
V=/scratch/kghasemz/v2d/viz/robot; O=/scratch/kghasemz/v2d/report
for x in "t10_e38_task2_fr15:fig12a_rollout_e38_success" "t6_e30_task2s7:fig12b_rollout_e30_success" "t7_e20_task2s7:fig12c_rollout_e20_failure"; do
  v=$V/${x%%:*}.mp4; d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $v)
  ffmpeg -loglevel error -y -i $v -vf "fps=6/$d,scale=420:-1,tile=6x1:padding=6:color=0xfcfcfb" -frames:v 1 $O/${x##*:}.png && echo "wrote ${x##*:}.png"
done
cp /scratch/kghasemz/v2d/beat1/diag/frames/e33_spoon.jpg $O/fig13a_pose_overlay_e33.jpg
cp /scratch/kghasemz/v2d/beat1/diag/dustpan/e38.jpg $O/fig13b_pose_overlay_e38.jpg
cp /scratch/kghasemz/v2d/beat1/diag/fix3/e30_sweeper.jpg $O/fig13c_pose_overlay_e30.jpg
ls $O/*.png $O/*.jpg | wc -l
