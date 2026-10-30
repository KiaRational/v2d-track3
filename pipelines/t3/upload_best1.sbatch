#!/bin/bash
#SBATCH --job-name=best1-upload
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Upload the packed t3 Track 3 submission to all six Kaggle competitions and wait for the scores.
# Usage: sbatch upload_t3.sbatch
set -uo pipefail
module load python/3.11.5 arrow/24.0.0
source /scratch/kghasemz/v2d/venv-kaggle/bin/activate
cd /scratch/kghasemz/v2d/submission_kit_new/unz/v2d_submission_kit
python submit/v2d_submit.py /scratch/kghasemz/v2d/submission/best1/track3_submission.parquet --track track_3 \
  -m "best1: per-episode best of t3 / t6 (t3 12 eps, t6 8 eps), own meshes; code KiaRational/v2d-track3@0fb842f" \
  --wait
echo "upload exit $?"
ls -la /scratch/kghasemz/v2d/submission/best1/*receipt*.json 2>/dev/null
