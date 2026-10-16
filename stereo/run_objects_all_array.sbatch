#!/bin/bash
#SBATCH --job-name=v2d-objall
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Object stage for evaluation runs with EVERY official object (incl. wooden_spoon): SAM 3 masks -> FoundationPose
# + EKF (objects_sam3) -> stereo-ICP (icp_sam3) -> SE(3) fusion (fused*_sam3) -> 4D video with the current hands.
# Needs the core reconstruction + s06b stereo hands (fusion uses them). One run per array task.
# Usage: sbatch --array=0-N run_objects_all_array.sbatch <run_list>
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
OBJS=$(jq -r '.objects | join(",")' "$V2D_ROOT/runs/$NAME/episode.json")
echo "$NAME: objects $OBJS"
# s05 skips SAM tracking when objects_sam3/tracking_report.json exists: start clean (pre-spoon results are in
# $V2D_ROOT/backup_nospoon/<run>.tar)
rm -rf "${V2D_ROOT:?}/runs/${NAME:?}/objects_sam3"
cd ~/v2d-vulcan
python stages/s05_objects_v2.py "$NAME" --segmenter sam3 --out objects_sam3 --objects "$OBJS"
cd stereo
export V2D_OBJ_DIR=objects_sam3
[ -f "$V2D_ROOT/runs/$NAME/table_plane.json" ] || python table_plane.py "$NAME"
python object_icp.py "$NAME"
python fuse_objects.py "$NAME"
H=${HANDS:-mano_fit_hop_phys}
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" \
  agile python /code/stages/render_4d_scene.py "$NAME" --objects fused_all_sam3 --hands "$H" \
  --out_name "${NAME}_objects_spoon.mp4" --label "objects incl. wooden_spoon (fused_all_sam3)" || echo "render failed (non-fatal)"
