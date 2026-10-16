"""HaWoR vs our stereo hands, with HaWoR's wrist aligned to the stereo wrist (agile image).

HaWoR (CVPR 2025) is the toolkit's 4D hand model; its per-frame MANO hands are in
<run>/recon_hawor/hawor/{2=left,3=right}/NNNNNN.json (camera frame via its cam_t). Its 3D
placement is known to be centimetres off, so to compare what it gets right beyond placement,
every HaWoR hand is TRANSLATED so its wrist (joint 0) sits on the stereo-triangulated wrist of
that frame; what remains is its hand orientation and finger articulation.

Scored against the stereo-triangulated joints (stereo/hands_tri.json, valid joints only):
  hawor_base   HaWoR, wrist moved onto the stereo wrist (orientation + articulation error)
  hawor_raw    HaWoR as placed by HaWoR (adds its placement error)
  <fits>       our MANO sequences (default: mano_fit, mano_fit_mem_refined), as they are
Metrics per side: frames, MPJPE mm (mean/median over valid joints), fingertip error mm,
jitter (mean joint acceleration in the world frame, mm/frame^2), and HaWoR's wrist placement
error. Writes stereo/hawor_compare.json and a video
$V2D_VIZ/07_stereo_hands/<run>_hawor_cmp.mp4 (green stereo joints, orange our fit,
magenta HaWoR wrist-aligned).
Also exports HaWoR in the same schema as our fits (mano + T + joints), so the 4D scene renderer
can draw it: stereo/hawor_raw/<side>/ (HaWoR's own placement) and stereo/hawor_base/<side>/
(wrist on the stereo wrist; only frames with a valid stereo wrist).
Usage: python compare_hawor.py <run> [...] [--fits mano_fit mano_fit_mem_refined] [--no_video]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
TRACK = {"left": 2, "right": 3}
TIPS = [4, 8, 12, 16, 20]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def load_T(p):
    d = json.loads(Path(p).read_text())
    q = d["rotation"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def jitter(seq, slam):
    """Mean |second difference| of world-frame joints over consecutive frames, mm/frame^2."""
    ts = sorted(seq)
    acc = []
    for a, b, c in zip(ts, ts[1:], ts[2:]):
        if b == a + 1 and c == b + 1:
            W = [seq[t] @ slam[t][:3, :3].T + slam[t][:3, 3] for t in (a, b, c)]
            acc.append(np.linalg.norm(W[2] - 2 * W[1] + W[0], axis=1).mean())
    return float(np.mean(acc) * 1000) if acc else None


def score(seq, X, V):
    errs, tip = [], []
    for t, J in seq.items():
        if t in X:
            e = np.linalg.norm(J - X[t], axis=1)[V[t]]
            if len(e):
                errs.append(e.mean())
                te = np.linalg.norm(J[TIPS] - X[t][TIPS], axis=1)[V[t][TIPS]]
                if len(te):
                    tip.append(te.mean())
    return {"frames_scored": len(errs), "mpjpe_mm": float(np.mean(errs) * 1000) if errs else None,
            "mpjpe_median_mm": float(np.median(errs) * 1000) if errs else None,
            "tip_mm": float(np.mean(tip) * 1000) if tip else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fits", nargs="+", default=["mano_fit", "mano_fit_mem_refined"])
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    mano = Mano(str(ROOT / "weights/hand"), "cuda")
    print(f"{'run':18s} {'side':5s} {'method':22s} {'frames':>6s} {'MPJPE':>7s} {'median':>7s} {'tips':>6s} "
          f"{'jitter':>7s}", flush=True)
    for r in a.runs:
        run = ROOT / "runs" / r
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        tri = json.loads((run / "stereo/hands_tri.json").read_text())
        ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
        rep, draw = {}, {}
        for side in ("left", "right"):
            d = tri.get(side, {})
            Xraw = d.get("X_raw") or d.get("X_smooth") or {}
            X = {int(t): np.array(v) for t, v in Xraw.items() if int(t) in slam}
            vd = d.get("valid", {})
            V = {t: (np.array(vd[str(t)], bool) if str(t) in vd else np.isfinite(X[t][:, 0])) for t in X}
            V = {t: v for t, v in V.items() if v[0] and v.sum() >= 5}  # needs a valid stereo wrist
            X = {t: X[t] for t in V}
            seqs = {}
            hr, hb, place = {}, {}, []
            exp = {k: run / "stereo" / k / side for k in ("hawor_raw", "hawor_base")}
            for d_ in exp.values():
                d_.mkdir(parents=True, exist_ok=True)
            for f in sorted((run / f"recon_hawor/hawor/{TRACK[side]}").glob("*.json")):
                t = int(f.stem)
                if t not in slam:
                    continue
                rec = json.loads(f.read_text())
                _, j = mano.local(rec)
                J = j.cpu().numpy() + np.array(rec["camera"]["cam_t"])
                hr[t] = J
                base = {"mano": rec["mano"], "is_right": rec["is_right"], "source": "hawor"}
                ct = np.array(rec["camera"]["cam_t"])
                (exp["hawor_raw"] / f"{t:06d}.json").write_text(json.dumps(
                    {**base, "T": ct.tolist(), "joints": J.tolist()}))
                if t in X:
                    shift = X[t][0] - J[0]
                    hb[t] = J + shift  # wrist onto the stereo wrist
                    place.append(np.linalg.norm(shift))
                    (exp["hawor_base"] / f"{t:06d}.json").write_text(json.dumps(
                        {**base, "T": (ct + shift).tolist(), "joints": hb[t].tolist(), "wrist_shift_m": shift.tolist()}))
            seqs["hawor_base"], seqs["hawor_raw"] = hb, hr
            for fit in a.fits:
                fd = run / "stereo" / fit / side
                if fd.is_dir():
                    seqs[fit] = {int(p.stem): np.array(json.loads(p.read_text())["joints"])
                                 for p in fd.glob("*.json") if int(p.stem) in slam}
            rep[side] = {"stereo_wrist_frames": len(X), "hawor_frames": len(hr),
                         "hawor_wrist_placement_err_mm": float(np.mean(place) * 1000) if place else None}
            for k, seq in seqs.items():
                row = score(seq, X, V)
                row["frames"] = len(seq)
                row["jitter_mm"] = jitter(seq, slam)
                rep[side][k] = row
                f = lambda x: "     -" if x is None else f"{x:6.1f}"  # noqa: E731
                print(f"{r:18s} {side:5s} {k:22s} {row['frames']:6d} {f(row['mpjpe_mm']):>7s} "
                      f"{f(row['mpjpe_median_mm']):>7s} {f(row['tip_mm']):>6s} {f(row['jitter_mm']):>7s}", flush=True)
            print(f"{r:18s} {side:5s} {'(HaWoR wrist placement error)':22s} {'':6s} "
                  f"{rep[side]['hawor_wrist_placement_err_mm'] or 0:7.1f} mm", flush=True)
            draw[side] = (X, seqs.get(a.fits[-1]) or seqs.get(a.fits[0]) or {}, hb)
        (run / "stereo/hawor_compare.json").write_text(json.dumps(rep, indent=1))
        if a.no_video:
            continue
        out = Path(os.environ["V2D_VIZ"]) / "07_stereo_hands"
        out.mkdir(parents=True, exist_ok=True)
        meta = json.loads((run / "episode.json").read_text())
        wr = imageio.get_writer(str(out / f"{r}_hawor_cmp.mp4"), fps=meta["fps"], codec="libx264", quality=8,
                                macro_block_size=2)
        W, H = 1014, 760
        s = W / ki["width"]
        for t in sorted(slam):
            img = cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png"))
            if img is None:
                continue
            for side, (X, fit, hb) in draw.items():
                for seq, col, th in ((X, (60, 220, 60), 5), (fit, (0, 140, 255), 3), (hb, (220, 60, 220), 2)):
                    if t not in seq:
                        continue
                    P = seq[t]
                    uv = P[:, :2] / np.maximum(P[:, 2:3], 1e-3) @ K[:2, :2].T + K[:2, 2]
                    for i, j in BONES:
                        if np.isfinite(uv[[i, j]]).all():
                            cv2.line(img, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), col, th, cv2.LINE_AA)
            img = cv2.resize(img, (W, H))
            cv2.putText(img, f"{r} f{t}  green: stereo  orange: our fit  magenta: HaWoR (wrist on stereo wrist)",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            wr.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        wr.close()


if __name__ == "__main__":
    main()
