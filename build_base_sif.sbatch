#!/bin/bash
#SBATCH --job-name=v2d-base-sif
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Pull the PyTorch base shared by nearly every toolkit module, once.
set -euo pipefail
module load StdEnv/2023 apptainer/1.4.5
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/cache APPTAINER_TMPDIR=$SLURM_TMPDIR
OUT=/scratch/kghasemz/v2d/sif/base_pytorch251_cu124.sif
apptainer pull -F "$OUT" docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ls -lh "$OUT"
apptainer exec "$OUT" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
