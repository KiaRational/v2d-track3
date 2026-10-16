#!/bin/bash
#SBATCH --job-name=v2d-rgpy
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Run a small python script in one of our images (SIF=rg|flash_chord|egophi), CPU only.
module load apptainer/1.4.5
SIF=${SIF:-rg}  # rg | flash_chord | egophi
PY=python; [ "$SIF" = flash_chord ] && PY=/opt/venv/bin/python
apptainer exec --cleanenv -B /scratch -B $HOME/v2d-vulcan /scratch/kghasemz/v2d/sif/$SIF.sif $PY "$@"
