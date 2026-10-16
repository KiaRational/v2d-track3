#!/bin/bash
#SBATCH --job-name=v2d-evalpp
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Same evaluator (SE(3) + kit-style block) on raw FP, pre-fix and post-fix fused trajectories -> eval_prepost.json
set -euo pipefail
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_sam3
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd ~/v2d-vulcan/stereo
python ../stages/eval_objects.py "$NAME" --variants fp fused_all_sam3_pre0930 fused_all_sam3 --align fp --out eval_prepost.json
