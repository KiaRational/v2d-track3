#!/bin/bash
#SBATCH --job-name=v2d-spoon-build
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=rack02-06
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# SAM 3D Objects mesh of wooden_spoon from SAM 3 candidates (spoon_mesh.py build). Args: <run> <frame> <idx> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
D=$V2D_ROOT/tmp_diag/spoon
while [ $# -ge 3 ]; do
  docker run --rm --gpus all -v "$V2D_ROOT:$V2D_ROOT" -v "$HOME/v2d-vulcan:/code" -e V2D_ROOT="$V2D_ROOT" -e CONDA_PREFIX=/usr/local/cuda -e PYTHONPATH="$V2D_ROOT/pyextra/sam3d" -e TORCH_HOME="$V2D_ROOT/weights/torch_home" -e HF_HUB_OFFLINE=1 \
    sam3d_objects python /code/stages/objects_v2/spoon_mesh.py build --run "$1" --frame "$2" --cand "$D/cands.npz" \
    --idx "$3" --out "$D/mesh_c$3"
  shift 3
done
