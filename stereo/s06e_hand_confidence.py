"""Stage 6e: stand-alone false/jittery-pose gating of an existing hands_tri.json (host env, CPU).

Applies stereo/hand_quality.py to a Stage-6b output without re-triangulating, so the
confidence gate can be measured on its own (A/B against the untouched baseline) and applied
to episodes whose Stage 6b run predates it. Reads <run>/stereo/<--in>, rewrites `conf` and
`valid` (the fields Stage 6c consumes) and writes <run>/stereo/<--out>.

Usage: python s06e_hand_confidence.py <run> [...] [--in hands_tri.json] [--out hands_tri_q.json]
       [--no_mask_iou] [--set bone_soft=0.2,min_frame=0.25]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hand_quality import DEFAULTS, apply_to_tri

ROOT = Path(os.environ["V2D_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--in", dest="src", default="hands_tri.json")
    ap.add_argument("--out", default="hands_tri_q.json")
    ap.add_argument("--no_mask_iou", action="store_true")
    ap.add_argument("--set", default="", help="comma-separated key=value overrides of hand_quality.DEFAULTS")
    a = ap.parse_args()
    params = {}
    for kv in filter(None, a.set.split(",")):
        k, _, v = kv.partition("=")
        if k not in DEFAULTS:
            raise SystemExit(f"unknown hand_quality parameter {k!r}; known: {sorted(DEFAULTS)}")
        params[k] = float(v)
    for name in a.runs:
        run = ROOT / "runs" / name
        src = run / "stereo" / a.src
        if not src.exists():
            print(f"{name}: no {a.src}", flush=True)
            continue
        tri = json.loads(src.read_text())
        rep = apply_to_tri(tri, run, params=params, use_mask=not a.no_mask_iou)
        (run / "stereo" / a.out).write_text(json.dumps(tri))
        print(f"{name}: {json.dumps(rep)}", flush=True)


if __name__ == "__main__":
    main()
