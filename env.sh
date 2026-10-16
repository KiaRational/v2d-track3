# Source this (inside a Slurm job) to get the V2D host environment on Vulcan.
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 opencv/4.11.0 arrow/25.0.0 ffmpeg/7.1.1 apptainer/1.4.5

export V2D_ROOT=/scratch/kghasemz/v2d
export V2D_TOOLKIT=$V2D_ROOT/toolkit
export V2D_DATA=$V2D_ROOT/data/track_3
export V2D_VIZ=$V2D_ROOT/viz
# Toolkit `docker` calls are translated to Apptainer by ~/v2d-vulcan/bin/docker.
export V2D_SIF_DIR=$V2D_ROOT/sif
export HF_HOME=$V2D_ROOT/cache/hf XDG_CACHE_HOME=$V2D_ROOT/cache/xdg TORCH_HOME=$V2D_ROOT/cache/torch
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

source $V2D_ROOT/venv-host/bin/activate
# docker -> apptainer shim must win over anything else on PATH.
export PATH=$HOME/v2d-vulcan/bin:$PATH
