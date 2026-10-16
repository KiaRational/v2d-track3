#!/bin/bash
#SBATCH --job-name=v2d-physics
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# MuJoCo physics check of reconstructed hand-object interaction. Usage: sbatch run_physics.sbatch <run> [...] [-- args]
module load apptainer/1.4.5
SIF=/scratch/kghasemz/franka/containers/python312_trixie.sif
apptainer exec -B /scratch -B $HOME/v2d-vulcan "$SIF" bash -c \
  ". /scratch/kghasemz/v2d/venv-mj/bin/activate && V2D_ROOT=/scratch/kghasemz/v2d python $HOME/v2d-vulcan/stereo/physics_check.py $*"
