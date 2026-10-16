#!/bin/bash
#SBATCH --job-name=v2d-joint-tests
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/kghasemz/v2d/code_tests/joint_tests_%j.out
# Synthetic tests of stereo/s06j_joint_hoi.py (CPU, agile image; no run data, no MANO weights).
set -euo pipefail
module load StdEnv/2023 apptainer/1.3.5
export XDG_CACHE_HOME=/scratch/kghasemz/v2d/code_tests/cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/scratch/kghasemz/v2d/code_tests/tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
apptainer exec --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,TMPDIR="$TMPDIR" \
  -B /home/kghasemz/v2d-vulcan:/code:ro -B /scratch/kghasemz/v2d/code_tests \
  /scratch/kghasemz/v2d/sif/agile.sif \
  python /code/tests/test_joint_hoi.py "$@"
rm -rf "$TMPDIR"
