#!/bin/bash
#SBATCH --job-name=v2d-spoon-find
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# SAM 3 search for wooden_spoon views in the spoon episodes (spoon_mesh.py find).
set -euo pipefail
source ~/v2d-vulcan/env.sh
docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" \
  sam3 python /code/stages/objects_v2/spoon_mesh.py find --out "$V2D_ROOT/tmp_diag/spoon" \
  --runs evaluation_ep000003 evaluation_ep000005 evaluation_ep000006 evaluation_ep000015 evaluation_ep000016 \
         evaluation_ep000032 evaluation_ep000033 --text "wooden spoon" --step 15
