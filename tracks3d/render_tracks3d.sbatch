#!/bin/bash
#SBATCH --job-name=v2d-tracks3d-viz
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# 4D video of the TAPIP3D tracks + poses -> $V2D_VIZ/15_tracks3d/<run>_tracks3d.mp4
# (top: current fusion + tracks; bottom: object poses fitted from the tracks + tracks; green outline = mocap)
# Usage: sbatch render_tracks3d.sbatch <run_name> [<run_name> ...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
mkdir -p $V2D_VIZ/15_tracks3d
D="docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ agile python /code/stages/render_4d_scene.py"
for NAME in "$@"; do
  $D "$NAME" --objects fused_all_sam3 --hands mano_fit_phys --tracks tracks3d/tapip3d.npz \
     --label "CURRENT FUSION + 3D tracks" --out_name ${NAME}_trk_fused.mp4
  $D "$NAME" --objects track_rigid --hands mano_fit_phys --tracks tracks3d/tapip3d.npz \
     --label "POSES FROM 3D TRACKS" --out_name ${NAME}_trk_rigid.mp4
  ffmpeg -loglevel error -y -i $V2D_VIZ/09_scene4d/${NAME}_trk_fused.mp4 -i $V2D_VIZ/09_scene4d/${NAME}_trk_rigid.mp4 \
    -filter_complex vstack=inputs=2 -c:v libx264 -crf 22 -pix_fmt yuv420p $V2D_VIZ/15_tracks3d/${NAME}_tracks3d.mp4
  rm -f $V2D_VIZ/09_scene4d/${NAME}_trk_fused.mp4 $V2D_VIZ/09_scene4d/${NAME}_trk_rigid.mp4
  ls -lh $V2D_VIZ/15_tracks3d/${NAME}_tracks3d.mp4
done
