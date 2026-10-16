"""Cache WiLoR 2D keypoints (MANO joints projected with WiLoR's camera) for views a,c,b.
Runs in the agile image (CPU ok). Output: <run>/stereo/wilor2d.json
 {view: [[{is_right, score, bbox[4], j2d 21x2}, ...] per frame]}"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/code/agile")
from geometry import Mano, wilor_keypoints_2d

ROOT = Path(os.environ["V2D_ROOT"])
run = ROOT / "runs" / sys.argv[1]
mano = Mano(str(ROOT / "weights/hand"), device="cpu")
dirs = {"a": run / "recon_hawor/wilor_raw", "c": run / "stereo/wilor_c", "b": run / "stereo/wilor_b"}
if len(sys.argv) > 2:  # extra suffix, e.g. _clahe -> wilor_c_clahe
    dirs = {"c": run / f"stereo/wilor_c{sys.argv[2]}", "b": run / f"stereo/wilor_b{sys.argv[2]}"}
n = len(list((run / "recon_hawor/frames").glob("*.png")))
out = {}
for v, d in dirs.items():
    out[v] = []
    for t in range(n):
        f = d / f"{t:06d}.json"
        dets = []
        for rec in (json.loads(f.read_text()) if f.exists() else []):
            _, j = mano.local(rec)
            wh = rec.get("image_size") or ([2028, 1520] if v == "a" else [1280, 800])
            bb = rec["bbox"]
            dets.append({"is_right": bool(rec["is_right"]), "score": float(rec.get("score", 1.0)),
                         "bbox": [bb["x0"], bb["y0"], bb["x1"], bb["y1"]],
                         "j2d": wilor_keypoints_2d(j.numpy(), rec, wh).tolist()})
        out[v].append(dets)
name = "wilor2d.json" if len(sys.argv) <= 2 else f"wilor2d{sys.argv[2]}.json"
(run / "stereo" / name).write_text(json.dumps(out))
print("wrote", run / "stereo" / name, {v: sum(len(x) for x in out[v]) for v in out})
