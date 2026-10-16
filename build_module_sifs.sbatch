#!/bin/bash
#SBATCH --job-name=v2d-sif
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%A_%a.out
# Build one toolkit module image per array task as a SIF (Docker is unavailable
# on Vulcan). Submit with:  sbatch --array=0-N build_module_sifs.sbatch [MODULE...]
set -euo pipefail
module load StdEnv/2023 apptainer/1.4.5 python/3.11.5

DEFAULT_MODULES=(v2d_anycalib v2d_geocalib v2d_droid_slam v2d_sam2 v2d_wilor v2d_hamer
                 v2d_hawor v2d_moge v2d_foundation_pose v2d_foundation_stereo
                 v2d_grounding_dino v2d_gsplat_refinement v2d_hand_alignment v2d_mediapipe)
if [ "$#" -gt 0 ]; then MODULES=("$@"); else MODULES=("${DEFAULT_MODULES[@]}"); fi
MOD=${MODULES[$SLURM_ARRAY_TASK_ID]}

SRC=/scratch/kghasemz/v2d/toolkit/reconstruction/modules
SIF_DIR=/scratch/kghasemz/v2d/sif
BASE=$SIF_DIR/base_pytorch251_cu124.sif
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/cache APPTAINER_TMPDIR=$SLURM_TMPDIR
export V2D_BUILD_JOBS=$SLURM_CPUS_PER_TASK

# Docker build context = modules/, filtered by its .dockerignore.
CTX=$SLURM_TMPDIR/ctx
rsync -a --exclude-from="$SRC/.dockerignore" "$SRC/" "$CTX/"

DEF=$SIF_DIR/$MOD.def
python ~/v2d-vulcan/dockerfile2def.py "$SRC/$MOD/docker/Dockerfile" "$CTX" "$DEF" --base-sif "$BASE"

echo "=== building $MOD ==="
apptainer build --fakeroot -F "$SIF_DIR/$MOD.sif.tmp" "$DEF"
mv "$SIF_DIR/$MOD.sif.tmp" "$SIF_DIR/$MOD.sif"
ls -lh "$SIF_DIR/$MOD.sif"
