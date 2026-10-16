#!/bin/bash
#SBATCH --job-name=v2d-hbase
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# Baseline hand audit: audit_hands.py over the given runs + mano_fit jitter summary.
set -euo pipefail
source ~/v2d-vulcan/env.sh
cd ~/v2d-vulcan/stereo
python audit_hands.py "$@" --file "${AUDIT_FILE:-hands_tri.json}"
echo "=== mano_fit jitter / mpjpe ==="
python - "$@" <<'PY'
import json, os, sys
from pathlib import Path
R = Path(os.environ["V2D_ROOT"]) / "runs"
sub = os.environ.get("MANO_SUBDIR", "mano_fit")
print(f"{'run':22s} {'side':6s} {'frames':>6s} {'fit_jit':>8s} {'tri_jit':>8s} {'mpjpe_tri':>9s} {'reproj_px':>9s}")
for r in sys.argv[1:]:
    p = R / r / "stereo" / sub / "report.json"
    if not p.exists():
        print(f"{r:22s} (no {sub}/report.json)"); continue
    d = json.loads(p.read_text())
    for side, v in d["sides"].items():
        mf = v.get("mano_fit", {}); ts = v.get("triangulated_X_smooth", {})
        g = lambda x, k: ("%.2f" % x[k]) if x.get(k) is not None else "-"
        print(f"{r:22s} {side:6s} {mf.get('frames',0):6d} {g(mf,'jitter_all_frames_mm_per_frame2'):>8s} "
              f"{g(ts,'jitter_on_tri_frames_mm_per_frame2'):>8s} {g(mf,'mpjpe_tri_mm_mean'):>9s} "
              f"{g(mf,'reproj_wilor2d_px_mean'):>9s}")
PY
