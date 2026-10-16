#!/bin/bash
#SBATCH --job-name=v2d-reconev
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Core toolkit reconstruction for the evaluation split (no mocap GT), one episode per task.
# Usage: sbatch --array=0-19%N run_recon_eval_array.sbatch
set -euo pipefail
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${EVAL_LIST:-/scratch/kghasemz/v2d/eval_scored_runs.txt})
export V2D_RECON_MODE=core
exec bash ~/v2d-vulcan/run_recon.sbatch "$NAME" hawor
