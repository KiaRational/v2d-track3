#!/bin/bash
#SBATCH --job-name=v2d-s06f
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stage 6f contact refinement, then the contact audit on the refined hands.
# Usage: sbatch run_s06f.sbatch <runs_list.txt> [s06f args]
set -euo pipefail
source ~/v2d-vulcan/env.sh
L=$1; shift
for NAME in $(cat $L); do
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" \
    agile python /code/stereo/s06f_contact_refine.py "$NAME" "$@"
done
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" \
  agile python /code/stereo/audit_contact.py $(tr '\n' ' ' < $L) --hands mano_fit_refined
