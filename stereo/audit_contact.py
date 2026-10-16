"""Physical plausibility of the reconstructed hand-object interaction (agile image).

The policy stage (FlashCHORD) learns mostly from contact -- where and when the hand touches
the object -- so two hand errors matter: fingers INSIDE the object (reference contacts the
robot cannot physically make) and fingers HOVERING short of it while it moves (motion with
no contact to explain it). Per frame, each fitted MANO hand (stereo/<hands>) is placed in
each object's frame at its trajectory (<variant>) and queried in the object's signed
distance field (agile/geometry.SDFGrid, sign by normal vote, so open scans work).

Per episode and object:
  grasp frames  frames where that hand holds the object (fusion_report.json hand_contact)
  pen_mm        deepest hand vertex inside the object (0 if none), per frame
  gap_mm        closest hand vertex to the surface when not touching, per grasp frame
  tips_near     fingertips within 1 cm of the surface, per grasp frame
Writes <run>/stereo/contact_audit.json and prints one row per (episode, object).
Usage: python audit_contact.py <run> [...] [--variant fused_all_sam3] [--hands mano_fit]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano, SDFGrid  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
TIPS = [744, 320, 443, 554, 671]  # MANO fingertip vertices: thumb, index, middle, ring, pinky


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def stats(x):
    x = np.asarray(x, float)
    return None if len(x) == 0 else float(np.mean(x))


def frac(x, thr):
    x = np.asarray(x, float)
    return None if len(x) == 0 else float((x > thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--hands", default="mano_fit")
    a = ap.parse_args()
    dev = "cuda"
    mano = Mano(str(ROOT / "weights/hand"), dev)
    sdfs = {}
    print(f"{'run':20s} {'object':18s} {'grasp':>5s} {'pen_mm':>7s} {'pen>5':>6s} {'pen>10':>7s} "
          f"{'penAll':>7s} {'gap_mm':>7s} {'gap>5':>6s} {'tips':>5s}", flush=True)
    for r in a.runs:
        run = ROOT / "runs" / r
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        rep = json.loads((run / "fusion_report.json").read_text()) if (run / "fusion_report.json").exists() else {}
        hands = {}
        for side in ("left", "right"):
            for f in sorted((run / "stereo" / a.hands / side).glob("*.json")):
                t = int(f.stem)
                if t not in slam:
                    continue
                h = json.loads(f.read_text())
                v, _ = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                vc = v.cpu().numpy() + np.array(h["T"])
                hands.setdefault(side, {})[t] = vc @ slam[t][:3, :3].T + slam[t][:3, 3]
        out = {}
        for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
            nm = o["name"]
            if nm not in sdfs:
                sdfs[nm] = SDFGrid(trimesh.load(ROOT / "meshes_obj" / nm / f"{nm}.obj", force="mesh"), res=128,
                                   device=dev)
            pd = run / a.variant / nm / "poses"
            W = {t: slam[t] @ load_T(pd / f"{t:06d}.json") for t in slam if (pd / f"{t:06d}.json").exists()}
            grasp = {sd: set(rep.get(nm, {}).get("hand_contact", {}).get(sd, [])) for sd in ("left", "right")}
            pen_g, pen_all, gap_g, tips_g, per_frame = [], [], [], [], []
            for side, seq in hands.items():
                for t, vw in seq.items():
                    if t not in W:
                        continue
                    Wi = np.linalg.inv(W[t])
                    vl = vw @ Wi[:3, :3].T + Wi[:3, 3]
                    d = sdfs[nm](torch.tensor(vl, dtype=torch.float32, device=dev)).cpu().numpy() * 1000
                    pen = float(max(0.0, -d.min()))
                    pen_all.append(pen)
                    if t in grasp[side]:
                        pen_g.append(pen)
                        gap_g.append(float(max(0.0, d.min())))
                        tips_g.append(int((np.abs(d[TIPS]) < 10).sum()))
                        per_frame.append({"t": t, "side": side, "pen_mm": round(pen, 1),
                                          "gap_mm": round(float(max(0.0, d.min())), 1), "tips_near": tips_g[-1]})
            row = {"grasp_frames": len(pen_g), "pen_mm": stats(pen_g), "pen_gt5": frac(pen_g, 5),
                   "pen_gt10": frac(pen_g, 10), "pen_all_frames_mm": stats(pen_all), "pen_all_gt10": frac(pen_all, 10),
                   "gap_mm": stats(gap_g), "gap_gt5": frac(gap_g, 5), "tips_near": stats(tips_g),
                   "per_frame": per_frame}
            out[nm] = row
            f2 = lambda x, k=1: "   -" if x is None else f"{x:.{k}f}"  # noqa: E731
            print(f"{r:20s} {nm:18s} {row['grasp_frames']:5d} {f2(row['pen_mm']):>7s} {f2(row['pen_gt5'], 2):>6s} "
                  f"{f2(row['pen_gt10'], 2):>7s} {f2(row['pen_all_frames_mm']):>7s} {f2(row['gap_mm']):>7s} "
                  f"{f2(row['gap_gt5'], 2):>6s} {f2(row['tips_near']):>5s}", flush=True)
        (run / "stereo" / f"contact_audit_{a.hands}.json").write_text(json.dumps(
            {"variant": a.variant, "hands": a.hands, "objects": out}, indent=1))


if __name__ == "__main__":
    main()
