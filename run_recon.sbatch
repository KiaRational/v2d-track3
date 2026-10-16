#!/bin/bash
#SBATCH --job-name=v2d-recon
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Toolkit ego reconstruction for one prepared episode (Stage 2 run dir).
# Usage: sbatch run_recon.sbatch <run_dir_name> [hawor|hamer|dynhamr] [extra pipeline args...]
set -euo pipefail
source ~/v2d-vulcan/env.sh
NAME=$1; RUN=$V2D_ROOT/runs/$NAME; TRACKER=${2:-hawor}; shift $(( $# >= 2 ? 2 : 1 ))
EP=$RUN/episode.json
prompt=$(python -c "import json;print(json.load(open('$EP'))['object_prompt'])")
mesh=$(python -c "import json;print(json.load(open('$EP'))['object_mesh'])")

# V2D_RECON_MODE=core runs SLAM + all models (with overlays) but skips the slow
# toolkit refinement/export, which our own later stages replace.
if [ "${V2D_RECON_MODE:-full}" = core ]; then
  STAGE_FLAGS="--run_droid_slam"
else
  STAGE_FLAGS="--run_droid_slam --run_gravity_alignment --run_gsplat_refinement --export_threejs_result"
fi

# Verified calibration: the pipeline skips its MoGe intrinsics estimate if present.
OUT=$RUN/recon_$TRACKER; mkdir -p "$OUT"
[ -e "$OUT/intrinsics_stable.json" ] || cp "$RUN/intrinsics_stable.json" "$OUT/"
# Three.js viewer modules come from a CDN the proxy blocks; staged from the login
# node, and the exporter skips its download when they are already present.
if [ "${V2D_RECON_MODE:-full}" != core ]; then
  for r in result result_slam result_gravity_aligned result_slam_gravity_aligned; do
    mkdir -p "$OUT/$r/threejs_scene" && cp "$V2D_ROOT"/cache/threejs/*.js "$OUT/$r/threejs_scene/"
  done
fi

# FoundationPose at half resolution: full-res registration of large objects OOMs.
export V2D_FP_TARGET_WH=${V2D_FP_TARGET_WH:-1014x760}
# Reference frame must show the hands: once WiLoR has run (cached), use the earliest
# frame with both hands detected (else one hand); frame 0 otherwise.
REF=$(python - "$OUT/wilor_raw" <<'EOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
counts = {int(f.stem): sum(h.get("score", 0) >= 0.5 for h in json.loads(f.read_text()))
          for f in sorted(d.glob("*.json"))} if d.is_dir() else {}
best = next((i for i, c in sorted(counts.items()) if c >= 2), None)
best = best if best is not None else next((i for i, c in sorted(counts.items()) if c >= 1), 0)
print(best)
EOF
)
echo "reference frame: $REF"

cd "$V2D_TOOLKIT/reconstruction"
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video "$RUN/input_undistorted.mp4" \
  --object_prompt "$prompt" --object_mesh "$mesh" --skip_object_scale_estimation \
  --output_dir "$RUN/recon_$TRACKER" --reference_frame "$REF" \
  --hand_tracking "$TRACKER" \
  $STAGE_FLAGS \
  "$@"

# Collect the pipeline's own overlay videos into the viz tree.
VIZ_OUT=$V2D_VIZ/03_recon_$TRACKER/$NAME; mkdir -p "$VIZ_OUT"
(cd "$OUT" && find . -name '*.mp4' -type f | while read -r f; do
   cp "$f" "$VIZ_OUT/$(echo "${f#./}" | tr '/' '__')"; done)
ls -lh "$VIZ_OUT"
