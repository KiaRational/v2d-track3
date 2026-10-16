#!/bin/bash
#SBATCH --job-name=v2d-recon
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Core toolkit reconstruction (SLAM + depth + WiLoR + SAM2 + FoundationPose + HaWoR,
# with overlays) over the public episodes, one per array task. The slow toolkit
# refinement/export steps are skipped: our own stages replace them.
# Usage: sbatch --array=0-19%N run_recon_array.sbatch
set -euo pipefail
EPS=(0 2 11 12 13 18 21 22 23 24 26 27 31 39 40 41 42 43 44 45)
NAME=$(printf "public_ep%06d" "${EPS[$SLURM_ARRAY_TASK_ID]}")
export V2D_RECON_MODE=core
exec bash ~/v2d-vulcan/run_recon.sbatch "$NAME" hawor
