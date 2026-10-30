#!/bin/bash
#SBATCH --job-name=v2d-evidence-joint
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
set -euo pipefail
module load StdEnv/2023 apptainer/1.3.5
export V2D_ROOT=/scratch/kghasemz/v2d
export HF_HOME="$V2D_ROOT/hf_cache" XDG_CACHE_HOME="$V2D_ROOT/cache"
export PYTHONDONTWRITEBYTECODE=1
cd "$V2D_ROOT"
for run in "$@"; do
    if [[ -e "$V2D_ROOT/runs/$run/fused_evidence_joint" || -e "$V2D_ROOT/runs/$run/stereo/hands_evidence_joint" ]]; then
        echo "Output already exists for $run; refusing to overwrite" >&2
        exit 1
    fi
    apptainer exec --nv --cleanenv \
      --env V2D_ROOT="$V2D_ROOT",V2D_OBJ_DIR=objects_recon,V2D_MESH_DIR=meshes_recon,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=1,XDG_CACHE_HOME="$XDG_CACHE_HOME",HF_HOME="$HF_HOME" \
      -B /home/kghasemz/v2d-vulcan:/code:ro -B "$V2D_ROOT:$V2D_ROOT" \
      "$V2D_ROOT/sif/agile.sif" python /code/stereo/s06j_joint_hoi.py "$run" \
      --obj_dir objects_recon --init_objects fused_evidence_recon --evidence_variant fused_evidence_recon \
      --hands hands_c --ref_hands hands_c --segment_params '' \
      --variant fused_evidence_joint --out hands_evidence_joint --redetect refine --carry 1 --w_nohand 0
done
