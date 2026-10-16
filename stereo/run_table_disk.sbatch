#!/bin/bash
#SBATCH --job-name=v2d-tabledisk
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Add the round-table disk (center/radius/thickness) to table_plane.json for every run in a list (keeps up/height).
set -euo pipefail
source ~/v2d-vulcan/env.sh
export V2D_OBJ_DIR=objects_sam3
cd ~/v2d-vulcan/stereo
python table_plane.py $(cat "$1") "${@:2}"
