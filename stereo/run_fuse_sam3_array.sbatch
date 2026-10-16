#!/bin/bash
#SBATCH --job-name=v2d-fusesam3
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Fusion (all variants) + mocap evaluation on the SAM 3 segmentation, one run per task (stereo-ICP reused).
set -euo pipefail
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_sam3
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd ~/v2d-vulcan/stereo
python fuse_objects.py "$NAME"
python ../stages/eval_objects.py "$NAME" --variants fp icp_sam3 fused_sam3 fused_sym_sam3 fused_hand_sam3 fused_grasp_sam3 fused_all_sam3 --align fp --out eval_sam3.json
