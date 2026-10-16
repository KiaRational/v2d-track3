#!/bin/bash
#SBATCH --job-name=v2d-refine-eval-tests
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/kghasemz/v2d/code_tests/refine_eval_tests_%j.out
set -euo pipefail
module load StdEnv/2023 apptainer/1.3.5
export XDG_CACHE_HOME=/scratch/kghasemz/v2d/code_tests/cache
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/scratch/kghasemz/v2d/code_tests/tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
KIT=/scratch/kghasemz/v2d/submission_kit/v2d_submission_kit
apptainer exec --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,TMPDIR="$TMPDIR",V2D_ROOT="$TMPDIR",V2D_DATA="$TMPDIR",V2D_KIT=/kit \
  -B /home/kghasemz/v2d-vulcan:/code:ro -B /scratch/kghasemz/v2d/code_tests -B "$KIT":/kit:ro \
  /scratch/kghasemz/v2d/sif/agile.sif \
  bash -c 'python -c "import pandas" 2>/dev/null && echo "pandas: yes" || echo "pandas: no (stubbed for the kit cross-check)"; python /code/tests/test_refine_eval.py'
rm -rf "$TMPDIR"
