#!/bin/bash
#SBATCH --job-name=v2d-contact
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Hand-object penetration / contact-gap audit. Usage: sbatch run_audit_contact.sbatch <runs_list.txt> [args]
set -euo pipefail
source ~/v2d-vulcan/env.sh
L=$1; shift
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" \
  agile python /code/stereo/audit_contact.py $(tr '\n' ' ' < $L) "$@"
