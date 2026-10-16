#!/bin/bash
#SBATCH --job-name=v2d-fc
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Run a FlashCHORD command in its image (pinned revision aee95f25), from its source dir.
# Usage: [FC_SRC=<checkout>] sbatch run_fc.sbatch <script.sh>   (a bash script run inside the image, cwd = flash_chord source)
module load apptainer/1.4.5
FC=${FC_SRC:-/scratch/kghasemz/v2d/fc_src}/robotic_grounding  # FC_SRC overrides the source checkout (e.g. the v2d-rewards worktree)
OUT=/scratch/kghasemz/v2d/robot/fc_out; mkdir -p $OUT /scratch/kghasemz/v2d/cache/{jax,warp,xla}
export APPTAINERENV_V2D_ASSETS_DIR=/v2d/assets APPTAINERENV_WARP_CACHE_PATH=/scratch/kghasemz/v2d/cache/warp \
       APPTAINERENV_JAX_COMPILATION_CACHE_DIR=/scratch/kghasemz/v2d/cache/jax APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false \

apptainer exec --nv --cleanenv -B /scratch -B $FC/flash_chord:/workspace \
  -B $FC/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro \
  -B /scratch/kghasemz/v2d/robot/assets_extra:/v2d/assets_extra \
  --pwd /workspace /scratch/kghasemz/v2d/sif/flash_chord.sif bash -c "nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; bash $1"
