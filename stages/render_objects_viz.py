"""Render the object segmentation/pose overlay video for any objects directory.

stages/s05_objects*.py only render their viz when writing to `objects/`, so runs written
to `objects_sam3/` (or any other variant) have masks and poses but no video. This renders
one from whatever directory is given, tagging the output so variants sit side by side:
  $V2D_VIZ/05_objects/<run>[_<tag>].mp4
Mask outline per object plus its pose box, over the colour frames at 1014x760.
Usage (host env, CPU job): python render_objects_viz.py <run> [...] [--objects objects_sam3]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ["V2D_ROOT"])
VIZ = Path(os.environ["V2D_VIZ"]) / "05_objects"
EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
COLOURS = [(0, 140, 255), (255, 80, 60), (60, 220, 60), (200, 60, 200)]


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]  # toolkit Transform3d, w-first
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--objects", default="objects_sam3", help="objects directory inside the run")
    ap.add_argument("--poses", default="poses_smoothed")
    ap.add_argument("--tag", default=None, help="video suffix (default: derived from --objects)")
    a = ap.parse_args()
    tag = a.tag if a.tag is not None else ("" if a.objects == "objects" else a.objects.replace("objects_", ""))
    K = np.array(json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())["undistorted_ego_cam_a_K"])
    VIZ.mkdir(parents=True, exist_ok=True)
    for r in a.runs:
        run = ROOT / "runs" / r
        od = run / a.objects
        if not (od / "objects.json").exists():
            print(f"{r}: no {a.objects}/objects.json, skipping", flush=True)
            continue
        objs = json.loads((od / "objects.json").read_text())
        meta = json.loads((run / "episode.json").read_text())
        frames = sorted((run / "recon_hawor/frames").glob("*.png"))
        vw, vh = 1014, 760
        s = vw / 2028
        out = VIZ / (f"{r}_{tag}.mp4" if tag else f"{r}.mp4")
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (vw, vh))
        box = {}
        for o in objs:
            lo, hi = trimesh.load(o["mesh"], force="mesh").bounds
            box[o["id"]] = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        for t in range(len(frames)):
            img = cv2.resize(cv2.imread(str(frames[t])), (vw, vh))
            for i, o in enumerate(objs):
                col = COLOURS[i % len(COLOURS)]
                m = cv2.imread(str(od / f"masks/{o['id']}/{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
                if m is not None and m.any():
                    cnt, _ = cv2.findContours(cv2.resize(m, (vw, vh)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cnt, -1, col, 2)
                pf = od / o["name"] / a.poses / f"{t:06d}.json"
                if pf.exists():
                    P = box[o["id"]] @ load_T(pf)[:3, :3].T + load_T(pf)[:3, 3]
                    uv = (P[:, :2] / P[:, 2:3] @ K[:2, :2].T + K[:2, 2]) * s
                    for e in EDGES:
                        cv2.line(img, tuple(uv[e[0]].astype(int)), tuple(uv[e[1]].astype(int)), col, 2, cv2.LINE_AA)
                cv2.putText(img, f"{o['name']} (id {o['id']})", (10, 24 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(img, f"frame {t}", (vw - 150, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            wr.write(img)
        wr.release()
        print(f"{r}: wrote {out}", flush=True)


if __name__ == "__main__":
    main()
