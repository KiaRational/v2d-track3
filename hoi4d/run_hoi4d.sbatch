#!/bin/bash
#SBATCH --job-name=hoi4d
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# hoi4d stages in the agile image. Usage: sbatch run_hoi4d.sbatch <run> <cmd> [cmd ...]   (cmds: ingest diagnose ...)
set -euo pipefail
source ~/v2d-vulcan/env.sh
RUN=$1; shift
for C in "$@"; do
  echo "=== $C $RUN"
  docker run --rm --gpus all -v $V2D_ROOT:$V2D_ROOT -v $HOME/v2d-vulcan:/code -e V2D_ROOT=$V2D_ROOT -e V2D_VIZ=$V2D_VIZ \
    -e PYTHONPATH=/code/hoi4d/src agile python -m hoi4d.cli $C $RUN ${HOI4D_ARGS:-}
done
