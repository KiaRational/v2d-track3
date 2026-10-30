#!/bin/bash
#SBATCH --job-name=v2d-evidence
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
set -euo pipefail
module load StdEnv/2023 apptainer/1.3.5
export V2D_ROOT=/scratch/kghasemz/v2d
export XDG_CACHE_HOME="$V2D_ROOT/cache"
export HF_HOME="$V2D_ROOT/hf_cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$V2D_ROOT"
command=(python /code/stereo/evidence_fusion.py "$@")
if [[ "${1:-}" == --test ]]; then
    command=(python /code/tests/test_evidence_fusion.py)
elif [[ "${1:-}" == --preview ]]; then
    shift
    command=(python /code/stereo/evidence_preview.py "$@")
elif [[ "${1:-}" == --rigid ]]; then
    shift
    command=(python /code/stereo/rigid_carry_refine.py "$@")
fi
apptainer exec --cleanenv \
  --env V2D_ROOT="$V2D_ROOT",V2D_OBJ_DIR=objects_recon,V2D_MESH_DIR=meshes_recon,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,XDG_CACHE_HOME="$XDG_CACHE_HOME",HF_HOME="$HF_HOME" \
  -B /home/kghasemz/v2d-vulcan:/code:ro -B "$V2D_ROOT:$V2D_ROOT" \
  "$V2D_ROOT/sif/agile.sif" "${command[@]}"
