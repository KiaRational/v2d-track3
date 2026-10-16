"""Overlay of the Stage-6d SAM 3 hand tracks (host env, CPU).

Per view, draws every accepted instance's mask tint, box, track name and metric distance,
and marks the frames around a re-association (a hand leaving and re-entering the view) so
they can be eyeballed. Writes a video per view and, with --frames, single PNGs of the
interesting frames (re-association neighbourhoods and the frames with the most rejected
background instances) to $V2D_ROOT/tmp_frames/<run>/.

Usage: python viz_hands_sam3.py <run> [...] [--views a,c] [--frames] [--fps 20]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ["V2D_ROOT"])
VIZ = Path(os.environ.get("V2D_VIZ", str(ROOT / "viz")))
COL = {"left": (255, 90, 0), "right": (0, 200, 255)}      # BGR


def frames_of(run: Path, view: str, n: int):
    if view == "a":
        for t in range(n):
            p = run / f"recon_hawor/frames/{t:06d}.png"
            yield t, (cv2.imread(str(p)) if p.exists() else None)
        return
    cap = cv2.VideoCapture(str(run / f"stereo/cam_{view}.mp4"))
    for t in range(n):
        ok, f = cap.read()
        yield t, (f if ok else None)
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--views", default="a,c")
    ap.add_argument("--frames", action="store_true", help="also dump single PNGs to tmp_frames/")
    ap.add_argument("--fps", type=float, default=0.0)
    a = ap.parse_args()
    for name in a.runs:
        run = ROOT / "runs" / name
        sm = run / "stereo/hands_sam3/summary.json"
        if not sm.exists():
            print(f"{name}: no hands_sam3", flush=True)
            continue
        S = json.loads(sm.read_text())
        n = S["n_frames"]
        fps = a.fps or json.loads((run / "episode.json").read_text())["fps"]
        interesting = sorted({t for e in S["reassociation_events"]
                              for t in range(max(0, e["frame"] - 3), min(n, e["frame"] + 4))})
        out = VIZ / "07b_sam3_hands"
        out.mkdir(parents=True, exist_ok=True)
        tf = ROOT / "tmp_frames" / name
        if a.frames:
            tf.mkdir(parents=True, exist_ok=True)
        for view in a.views.split(","):
            det_dir = run / "stereo/hands_sam3" / view
            if not det_dir.is_dir():
                continue
            wr, wrote = None, 0
            for t, img in frames_of(run, view, n):
                if img is None:
                    continue
                f = det_dir / f"{t:06d}.json"
                recs = json.loads(f.read_text()) if f.exists() else []
                over = img.copy()
                for r in recs:
                    m = cv2.imread(str(run / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
                    c = COL[r["track"]]
                    if m is not None:
                        if m.shape[:2] != img.shape[:2]:
                            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        over[m > 0] = (0.45 * np.array(c) + 0.55 * over[m > 0]).astype(np.uint8)
                    x0, y0, x1, y1 = [int(v) for v in r["bbox"]]
                    cv2.rectangle(over, (x0, y0), (x1, y1), c, 3)
                    lab = f"{r['track']} {r['depth_m']:.2f}m {r['depth_source']}" if r["depth_m"] else r["track"]
                    if r["source"] == "fill":
                        lab += " [fill]"
                    cv2.putText(over, lab, (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9 if view == "a" else 0.5, c, 2, cv2.LINE_AA)
                tag = f"{name} {view} f{t}"
                if t in interesting:
                    tag += "  RE-ASSOC"
                    cv2.rectangle(over, (0, 0), (over.shape[1] - 1, over.shape[0] - 1), (0, 0, 255), 6)
                cv2.putText(over, tag, (12, over.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9 if view == "a" else 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                if wr is None:
                    wr = cv2.VideoWriter(str(out / f"{name}_{view}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (over.shape[1], over.shape[0]))
                wr.write(over)
                wrote += 1
                if a.frames and t in interesting:
                    cv2.imwrite(str(tf / f"{view}_{t:06d}.png"),
                                cv2.resize(over, None, fx=0.5, fy=0.5) if view == "a" else over)
            if wr is not None:
                wr.release()
            print(f"{name}/{view}: {wrote} frames -> {out / f'{name}_{view}.mp4'}", flush=True)
        print(f"{name}: {len(S['reassociation_events'])} re-association events at "
              f"{[e['frame'] for e in S['reassociation_events'][:20]]}", flush=True)


if __name__ == "__main__":
    main()
