"""Object poses from TAPIP3D 3D tracks (pilot measurement source).

Each object track i is embedded once in the object frame at its own query frame q_i, using the
current fused pose there: x_i = O_{q_i}^{-1} X_i(q_i). Every frame t then gets an absolute pose from
the tracks visible at t, by robust rigid fitting (RANSAC on 3-point Kabsch + IRLS Huber refinement):
    O_t = argmin_{R,p} sum_i rho(||R x_i + p - X_i(t)||),  rho Huber(1 cm)
so the fused trajectory enters only through the many query-frame embeddings (averaged), and
the frame-to-frame motion comes from the tracks. Tracks whose embedding residual stays large
(slid onto the hand / background at occlusion edges) are dropped.
Writes variant <run>/<out>/<object>/poses/*.json (object-to-camera, {"R","T"}) + <out>/report.json.
Usage (host env): python track_objects.py <run> [--base fused_all_sam3] [--out track_rigid]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/scratch/kghasemz/v2d")


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


def kabsch(x, y, w=None):
    """R, p minimising sum w ||R x + p - y||^2."""
    w = np.ones(len(x)) if w is None else w
    w = w / w.sum()
    mx, my = w @ x, w @ y
    H = ((x - mx) * w[:, None]).T @ (y - my)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, my - R @ mx


def robust_fit(x, y, rng, iters=200, thr=0.01, huber=0.01):
    """RANSAC (3-point Kabsch, inlier threshold thr) + IRLS Huber. Returns R, p, inlier mask, rms (m)."""
    n = len(x)
    best = None
    for _ in range(iters):
        s = rng.choice(n, 3, replace=False)
        R, p = kabsch(x[s], y[s])
        inl = np.linalg.norm(x @ R.T + p - y, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        best = np.ones(n, bool)
    R, p = kabsch(x[best], y[best])
    for _ in range(5):
        r = np.linalg.norm(x @ R.T + p - y, axis=1)
        w = np.where(r < huber, 1.0, huber / np.maximum(r, 1e-9))
        R, p = kabsch(x, y, w)
    r = np.linalg.norm(x @ R.T + p - y, axis=1)
    inl = r < 2 * thr
    return R, p, inl, float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tracks", default="tracks3d/tapip3d.npz")
    ap.add_argument("--base", default="fused_all_sam3", help="pose variant used to embed tracks at query frames")
    ap.add_argument("--out", default="track_rigid")
    ap.add_argument("--min_tracks", type=int, default=12)
    ap.add_argument("--embed_resid", type=float, default=0.015, help="drop tracks whose median fit residual exceeds this (m)")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    z = np.load(run / a.tracks)
    C, V, Q, L, names = z["coords"].astype(np.float64), z["visibs"].astype(bool), z["queries"], z["label"], list(z["labels"])
    T = C.shape[0]
    c2w = np.stack([load_T(run / f"recon_hawor/slam_poses/{t:06d}.json") for t in range(T)])
    rng = np.random.default_rng(0)
    report = {"tracks": a.tracks, "base": a.base, "objects": {}}
    for li, name in enumerate(names):
        if name.startswith("hand_"):
            continue
        base = run / a.base / name / "poses"
        O = {t: c2w[t] @ load_T(base / f"{t:06d}.json") for t in range(T) if (base / f"{t:06d}.json").exists()}
        idx = np.where(L == li)[0]
        qt = Q[idx, 0].astype(int)
        keep = np.array([t in O for t in qt])
        idx, qt = idx[keep], qt[keep]
        # object-frame embedding at each track's own query frame
        x = np.stack([np.linalg.inv(O[t])[:3, :3] @ Q[i, 1:] + np.linalg.inv(O[t])[:3, 3] for i, t in zip(idx, qt)])
        Y, Vis = C[:, idx], V[:, idx]
        fits = {}
        for t in range(T):
            m = Vis[t]
            if m.sum() >= a.min_tracks:
                fits[t] = robust_fit(x[m], Y[t, m], rng)
        # drop tracks that do not move rigidly with the object, refit
        res = np.full(len(idx), np.nan)
        rr = [[] for _ in idx]
        for t, (R, p, _, _) in fits.items():
            m = np.where(Vis[t])[0]
            for j, r in zip(m, np.linalg.norm(x[m] @ R.T + p - Y[t, m], axis=1)):
                rr[j].append(r)
        res = np.array([np.median(r) if r else np.nan for r in rr])
        good = np.nan_to_num(res, nan=1.0) < a.embed_resid
        out = run / a.out / name / "poses"
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.json"):
            f.unlink()
        n_ok, rms, nvis = 0, [], []
        for t in range(T):
            m = Vis[t] & good
            nvis.append(int(m.sum()))
            if m.sum() < a.min_tracks:
                continue
            R, p, inl, e = robust_fit(x[m], Y[t, m], rng)
            Mw = np.eye(4)
            Mw[:3, :3], Mw[:3, 3] = R, p
            Mc = np.linalg.inv(c2w[t]) @ Mw
            (out / f"{t:06d}.json").write_text(json.dumps({"R": Mc[:3, :3].tolist(), "T": Mc[:3, 3].tolist()}))
            n_ok += 1
            rms.append(e)
        report["objects"][name] = {"tracks": int(len(idx)), "rigid_tracks": int(good.sum()), "frames_posed": n_ok,
                                   "frames": T, "median_visible_tracks": float(np.median(nvis)),
                                   "median_fit_rms_mm": float(np.nanmedian(rms) * 1000) if rms else None}
        print(name, report["objects"][name], flush=True)
    (run / a.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
