#!/bin/bash
#SBATCH --job-name=v2d-fglobal
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# fused_global[_sfx]: FP + stereo-ICP + FP-free global registration (fuse_objects.py --w_global),
# one process per run in parallel, then eval_global + eval_objects (all existing variants).
# Usage: GV=stereo_global W=4 sbatch run_fused_global.sbatch <run> [...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo
GV=${GV:-stereo_global}; W=${W:-4}
printf '%s\n' "$@" | xargs -P "${SLURM_CPUS_PER_TASK:-8}" -I{} sh -c \
  "python fuse_objects.py {} --w_global $W --only_global --global_variant $GV > /scratch/kghasemz/v2d/logs/fglobal_{}_$GV.log 2>&1 || echo FAILED {}"
printf '%s\n' "$@" | grep '^public_' | xargs -P "${SLURM_CPUS_PER_TASK:-8}" -I{} sh -c "python eval_global.py {} > /dev/null || echo EVAL_FAILED {}"
[ -n "${NO_EVAL_OBJECTS:-}" ] || bash run_eval_objects_global.sh "$@"
