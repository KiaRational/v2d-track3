"""Time lag of each object-pose source (and the stereo hands) against mocap, per object (host env).

World-frame centroid trajectories p(t) = slam[t] @ T(t) are compared with the mocap trajectory in our world
frame (gt_world.json) shifted by tau frames (linear interpolation, tau in [-3, 3] step 0.05), on the
displacement from the trajectory mean. tau > 0: the estimate at frame t matches mocap at t + tau, i.e. the
estimate LEADS the video (is ahead in time). Hands: the stereo palm (hands_tri X_smooth, joints 0/5/9/13/17)
against the carried object's mocap on the hand's contact frames (fusion_report hand_contact).
Usage: python time_lag.py <run> [...] [--variants fp_raw icp fused_all_sam3 fused_joint]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ["V2D_ROOT"])


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
    elif "T" in d and np.shape(d["T"]) == (4, 4):  # stereo_icp: full 4x4
        T = np.array(d["T"], float)
    elif "rotation" not in d:
        return None  # e.g. a failed stereo-ICP frame
    else:
        q = d["rotation"]; T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = d["translation"]
    return T


def best_lag(ts, P, G, taus=np.arange(-3, 3.001, 0.05)):
    """ts: frames with estimate; P (len(ts),3) estimate; G (n,3) mocap with NaNs. Returns (tau, rms_at_tau, rms_at_0)."""
    n = len(G)
    ok = np.isfinite(G[:, 0])
    fr = np.arange(n)
    res = []
    for tau in taus:
        q = ts + tau
        m = (q >= 0) & (q <= n - 1)
        if m.sum() < 20:
            res.append(np.inf); continue
        g = np.stack([np.interp(q[m], fr[ok], G[ok, k]) for k in range(3)], 1)
        e = P[m] - P[m].mean(0) - (g - g.mean(0))
        res.append(np.sqrt((e ** 2).sum(1).mean()))
    res = np.array(res); i = int(np.argmin(res)); i0 = int(np.argmin(np.abs(taus)))
    return float(taus[i]), float(res[i] * 100), float(res[i0] * 100)


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--variants", nargs="+", default=["fp_raw", "fp_smoothed", "icp", "fused_all_sam3", "fused_joint"])
a = ap.parse_args()
for r in a.runs:
    run = ROOT / "runs" / r
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    gt = json.loads((run / "gt_world.json").read_text())["objects"]
    rep = json.loads((run / "fusion_report.json").read_text())
    n = len(slam)
    print(f"\n{r}  (tau > 0: estimate leads the video; rms cm at best tau / at tau 0)")
    for nm, g in gt.items():
        G = np.full((n, 3), np.nan)
        for t, M in g.items():
            if int(t) < n:
                G[int(t)] = np.array(M)[:3, 3]
        travel = np.nansum(np.linalg.norm(np.diff(G, axis=0), axis=1)) * 100
        for v in a.variants:
            d = {"fp_raw": run / "objects_sam3" / nm / "poses", "fp_smoothed": run / "objects_sam3" / nm / "poses_smoothed",
                 "icp": run / "objects_sam3" / nm / "stereo_icp"}.get(v, run / v / nm / "poses")
            fs = sorted(d.glob("*.json")) if d.exists() else []
            ts = np.array([int(f.stem) for f in fs if int(f.stem) in slam and load_T(f) is not None])
            if len(ts) < 20:
                continue
            P = np.array([(slam[t] @ load_T(d / f"{t:06d}.json"))[:3, 3] for t in ts])
            tau, rb, r0 = best_lag(ts, P, G)
            print(f"  {nm:16s} {v:16s} frames {len(ts):4d}  lag {tau:+.2f} fr  rms {rb:.2f} / {r0:.2f} cm  (travel {travel:.0f} cm)")
        tri = json.loads((run / "stereo/hands_tri.json").read_text())
        for side, frames in rep.get(nm, {}).get("hand_contact", {}).items():
            X = tri.get(side, {}).get("X_smooth", {})
            ts = np.array([t for t in frames if str(t) in X and t in slam])
            if len(ts) < 20:
                continue
            P = np.array([slam[t][:3, :3] @ np.array(X[str(t)])[[0, 5, 9, 13, 17]].mean(0) + slam[t][:3, 3] for t in ts])
            tau, rb, r0 = best_lag(ts, P, G)
            print(f"  {nm:16s} {'hand_' + side:16s} frames {len(ts):4d}  lag {tau:+.2f} fr  rms {rb:.2f} / {r0:.2f} cm  (stereo palm vs carried object)")

# ---- image space, VIDEO time: SAM 3 mask centroid (colour frames) vs each source's projected centroid ----
import cv2  # noqa: E402

print("\nimage space vs the SAM 3 masks (video time); tau > 0: the source LEADS the video; rms px")
for r in a.runs:
    run = ROOT / "runs" / r
    K = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    gt = json.loads((run / "gt_world.json").read_text())["objects"]
    n = len(slam)
    proj = lambda X: np.array([K["fx"] * X[0] / X[2] + K["cx"], K["fy"] * X[1] / X[2] + K["cy"]])
    for o in json.loads((run / "objects_sam3/objects.json").read_text()):
        nm, md = o["name"], run / "objects_sam3/masks" / str(o["id"])
        M = np.full((n, 3), np.nan)  # mask centroid u, v (third column unused)
        for f in sorted(md.glob("*.png")):
            t = int(f.stem)
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if t < n and m is not None and (m > 0).sum() > 200:
                ys, xs = np.nonzero(m)
                M[t] = [xs.mean(), ys.mean(), 0]
        # the mask centroid is the reference here, so each source is "P" and the mask is "G"
        srcs = {"mocap": {int(t): np.linalg.inv(slam[int(t)]) @ np.array(W) for t, W in gt.get(nm, {}).items() if int(t) in slam}}
        for v in a.variants:
            d = {"fp_raw": run / "objects_sam3" / nm / "poses", "icp": run / "objects_sam3" / nm / "stereo_icp"}.get(v, run / v / nm / "poses")
            if d.exists():
                srcs[v] = {int(f.stem): T for f in d.glob("*.json") if int(f.stem) < n and (T := load_T(f)) is not None}
        for v, Ts in srcs.items():
            ts = np.array(sorted(t for t in Ts if np.isfinite(M[t, 0]) and Ts[t][2, 3] > 0.05))
            if len(ts) < 20:
                continue
            P = np.array([np.r_[proj(Ts[t][:3, 3]), 0] for t in ts])
            tau, rb, r0 = best_lag(ts, P, M)
            print(f"  {r} {nm:16s} {v:16s} frames {len(ts):4d}  lag {tau:+.2f} fr  rms {rb * 0.01:.1f} / {r0 * 0.01:.1f} px")
