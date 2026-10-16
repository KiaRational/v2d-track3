#!/bin/bash
# Run stages/eval_objects.py (--align fp) for public runs, with every variant that already has
# poses in the run (so earlier variants in eval_objects.json are preserved) + the stereo_global ones.
# Call inside a job after `source ~/v2d-vulcan/env.sh`.  Usage: bash run_eval_objects_global.sh <run> [...]
KNOWN="icp fused_free fused fused_sym fused_hand agile agile_hc stereo_global stereo_global_raw fused_global \
stereo_global_v2 stereo_global_v2_raw fused_global_v2 stereo_global_sam3 stereo_global_sam3_raw"
for r in "$@"; do
  [[ $r == public_* ]] || continue
  R=/scratch/kghasemz/v2d/runs/$r
  vs="fp"; for v in $KNOWN; do [ -d "$R/$v" ] && vs="$vs $v"; done
  python ~/v2d-vulcan/stages/eval_objects.py "$r" --variants $vs --align fp > /dev/null || echo "eval_objects failed: $r"
done
