"""Overlay the stereo_global object poses on the colour frames -> H.264 1014x760 video.

Output: $V2D_VIZ/08_stereo_global/<run>.mp4 (mesh silhouette per object, tinted, with
its contour; label = hypothesis source / score, 'H' while a hand touches the object).
Usage: python viz_global.py <run> [...] [--variant stereo_global] [--frames 10 60 ...]
  --frames: also save those frames as PNG next to the video (for inspection).
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ["V2D_ROOT"])
VIZ = Path(os.environ.get("V2D_VIZ", ROOT / "viz")) / "08_stereo_global"
COLS = [(60, 220, 60), (230, 120, 30), (40, 160, 250), (220, 60, 200)]
W, H = 1014, 760


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--variant", default="stereo_global")
    ap.add_argument("--frames", nargs="*", type=int, default=[])
    ap.add_argument("--mask_dir", default="objects/masks", help="only for its objects.json (object list/colours)")
    a = ap.parse_args()
    VIZ.mkdir(parents=True, exist_ok=True)
    for r in a.runs:
        run = ROOT / "runs" / r
        k = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        sx, sy = W / k["width"], H / k["height"]
        md = run / a.mask_dir.rsplit("/", 1)[0] / "objects.json"
        objs = json.loads((md if md.exists() else run / "objects/objects.json").read_text())
        models = {o["name"]: np.load(ROOT / "cache/global_models" / f"{o['name']}.npz")["dense"] for o in objs
                  if (ROOT / "cache/global_models" / f"{o['name']}.npz").exists()}
        frames = sorted((run / "recon_hawor/frames").glob("*.png"))
        fps = json.loads((run / "episode.json").read_text()).get("fps", 20)
        sfx = "" if a.variant == "stereo_global" else "_" + a.variant
        out = VIZ / f"{r}{sfx}.mp4"
        ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                               "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                               str(out)], stdin=subprocess.PIPE)
        for fpath in frames:
            t = int(fpath.stem)
            img = cv2.resize(cv2.imread(str(fpath)), (W, H), interpolation=cv2.INTER_AREA)
            over = img.copy()
            y0 = 22
            for oi, o in enumerate(objs):
                nm = o["name"]
                p = run / a.variant / nm / "poses" / f"{t:06d}.json"
                if nm not in models or not p.exists():
                    continue
                d = json.loads(p.read_text())
                X = models[nm] @ np.array(d["R"]).T + np.array(d["T"])
                z = X[:, 2]
                ok = z > 0.05
                u = (k["fx"] * X[ok, 0] / z[ok] + k["cx"]) * sx
                v = (k["fy"] * X[ok, 1] / z[ok] + k["cy"]) * sy
                inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                m = np.zeros((H, W), np.uint8)
                m[v[inb].astype(int), u[inb].astype(int)] = 1
                m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                col = COLS[oi % len(COLS)]
                over[m > 0] = (0.45 * np.array(col) + 0.55 * over[m > 0]).astype(np.uint8)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(over, cs, -1, col, 2)
                info = run / "objects" / nm / a.variant / f"{t:06d}.json"
                lab = nm
                if info.exists():
                    di = json.loads(info.read_text())
                    if di.get("chosen") is not None:
                        h = di["hyps"][di["chosen"]]
                        lab += f"  {h['src']} s={h['score']:.2f} iou={h['iou']:.2f}"
                    else:
                        lab += "  (interp)"
                    lab += "  H" if di.get("touched") else ""
                cv2.putText(over, lab, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                y0 += 22
            cv2.putText(over, f"{r} t={t}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            ff.stdin.write(over.tobytes())
            if t in a.frames:
                cv2.imwrite(str(VIZ / f"{r}{sfx}_{t:06d}.png"), over)
        ff.stdin.close()
        ff.wait()
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
