#!/bin/bash
#SBATCH --job-name=v2d-tracks3d
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# TAPIP3D 3D tracks (tapip3d image) -> object poses from tracks (host) -> mocap eval vs current fusion.
# Usage: sbatch run_tracks3d.sbatch <run_name> [run_tapip3d.py args...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$1; shift
docker run --rm --gpus all -e TORCH_HOME=$V2D_ROOT/cache/torch -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -w $V2D_ROOT/third_party/TAPIP3D \
  tapip3d python /code/tracks3d/run_tapip3d.py "$NAME" "$@"
python ~/v2d-vulcan/tracks3d/track_objects.py "$NAME"
cd ~/v2d-vulcan/stereo && V2D_OBJ_DIR=objects_sam3 python ../stages/eval_objects.py "$NAME" \
  --variants fp fused_all_sam3 track_rigid --align fp --out eval_tracks.json
jq -c ".variants | to_entries[] | {v: .key, AUC: .value.official.AUC, per: [.value.official.per_object[]? | {AUC, pos: .pos_err_cm_mean, rot: .rot_err_deg_mean, miss: .missing}]}" \
  $V2D_ROOT/runs/$NAME/eval_tracks.json 2>/dev/null || true
