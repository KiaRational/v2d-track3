#!/bin/bash
#SBATCH --job-name=hoi4d-arr
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# hoi4d over a list of runs: sbatch --array=0-39%6 run_pipeline_array.sbatch /scratch/kghasemz/v2d/hoi4d_runs.txt
set -euo pipefail
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
exec bash ~/v2d-vulcan/hoi4d/run_pipeline.sbatch "$RUN"
