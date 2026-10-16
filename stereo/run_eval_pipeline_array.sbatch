#!/bin/bash
#SBATCH --job-name=v2d-evalpipe
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Evaluation split (no mocap GT): SAM 3 objects -> stereo hands -> stereo-ICP -> fusion.
# Needs the core reconstruction (run_recon_eval_array.sbatch) to have finished.
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${EVAL_LIST:-/scratch/kghasemz/v2d/eval_scored_runs.txt})
[ -d "$V2D_ROOT/runs/$NAME/recon_hawor" ] || { echo "$NAME: no core reconstruction, skipping"; exit 0; }
cd ~/v2d-vulcan
# every official object (wooden_spoon: reconstructed SAM 3D mesh, meshes_obj/wooden_spoon)
OBJS=$(jq -r '.objects | join(",")' "$V2D_ROOT/runs/$NAME/episode.json")
python stages/s05_objects_v2.py "$NAME" --segmenter sam3 --out objects_sam3 --objects "$OBJS"
cd stereo
python s06a_stereo_wilor.py "$NAME"
python s06a_v2_stereo_wilor_clahe.py "$NAME"
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" \
  -e V2D_ROOT="$V2D_ROOT" -e V2D_VIZ="$V2D_VIZ" agile python /code/stereo/s06b_triangulate_align.py "$NAME"
export V2D_OBJ_DIR=objects_sam3
python object_icp.py "$NAME"
python fuse_objects.py "$NAME"
