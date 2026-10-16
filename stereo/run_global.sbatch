#!/bin/bash
#SBATCH --job-name=v2d-sglobal
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Stereo global registration (FP-free) -> variant $VARIANT -> eval_global + viz.
# Usage: sbatch run_global.sbatch <run> [...]
#   env: MASK_DIR=objects/masks VARIANT=stereo_global GR_ARGS="..." NO_VIZ=1 NO_EVAL=1
#   v2 masks: MASK_DIR=objects_v2/masks VARIANT=stereo_global_v2 sbatch run_global.sbatch ...
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo
MASK_DIR=${MASK_DIR:-objects/masks}
VARIANT=${VARIANT:-stereo_global}
python global_register.py "$@" --mask_dir "$MASK_DIR" --variant "$VARIANT" ${GR_ARGS:-}
if [ -z "${NO_EVAL:-}" ]; then
  for r in "$@"; do
    [[ $r == public_* ]] && python eval_global.py "$r" || true
  done
fi
[ -n "${NO_VIZ:-}" ] || python viz_global.py "$@" --variant "$VARIANT" --mask_dir "$MASK_DIR" || true
