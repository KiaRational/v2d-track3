#!/bin/bash
#SBATCH --job-name=v2d-sif-def
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Build a hand-written Apptainer definition into $V2D_SIF_DIR/<name>.sif.
# Usage: sbatch build_def_sif.sbatch defs/<name>.def
set -euo pipefail
module load StdEnv/2023 apptainer/1.4.5
DEF=$(realpath "$1"); NAME=$(basename "$DEF" .def)
SIF_DIR=/scratch/kghasemz/v2d/sif
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/cache APPTAINER_TMPDIR=$SLURM_TMPDIR
apptainer build --fakeroot -F "$SIF_DIR/$NAME.sif.tmp" "$DEF"
mv "$SIF_DIR/$NAME.sif.tmp" "$SIF_DIR/$NAME.sif"
ls -lh "$SIF_DIR/$NAME.sif"
