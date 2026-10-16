#!/bin/bash
#SBATCH --job-name=v2d-regencmp
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Before/after the 09-30 code fixes, one episode per task -> $V2D_VIZ/13_regen/<run>_regen_compare.mp4
# (top: before = fused_all_sam3_pre0930 + mano_fit_phys_pre0930; bottom: after = regenerated; green = mocap)
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python /code/stages/render_4d_scene.py"
$D "$NAME" --objects fused_all_sam3_pre0930 --hands mano_fit_phys_pre0930 --label "BEFORE fixes" --out_name ${NAME}_pre.mp4
$D "$NAME" --objects fused_all_sam3 --hands mano_fit_phys --label "AFTER fixes" --out_name ${NAME}_post.mp4
mkdir -p $V2D_VIZ/13_regen
ffmpeg -loglevel error -y -i $V2D_VIZ/09_scene4d/${NAME}_pre.mp4 -i $V2D_VIZ/09_scene4d/${NAME}_post.mp4 \
  -filter_complex vstack=inputs=2 -c:v libx264 -crf 22 -pix_fmt yuv420p $V2D_VIZ/13_regen/${NAME}_regen_compare.mp4
rm -f $V2D_VIZ/09_scene4d/${NAME}_pre.mp4 $V2D_VIZ/09_scene4d/${NAME}_post.mp4
ls -lh $V2D_VIZ/13_regen/${NAME}_regen_compare.mp4
