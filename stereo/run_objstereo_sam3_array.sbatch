#!/bin/bash
#SBATCH --job-name=v2d-sam3pipe
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Stereo-ICP + fusion + evaluation on the SAM 3 segmentation, one run per array task.
# Reads objects_sam3/, writes variants suffixed _sam3; the v1 results are untouched.
set -euo pipefail
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_sam3
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd ~/v2d-vulcan/stereo
python object_icp.py "$NAME"
python fuse_objects.py "$NAME"
python ../stages/eval_objects.py "$NAME" --variants fp icp_sam3 fused_sam3 fused_sym_sam3 fused_hand_sam3 fused_grasp_sam3 fused_all_sam3 --align fp --out eval_sam3.json
