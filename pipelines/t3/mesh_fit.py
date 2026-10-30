"""Data-driven mesh checks (no scans, no mocap).
pitcher:     horizontal scale of water_pitcher fitted to the SAM 3 masks (hand pixels excluded) over well-visible frames
             of e20 / p21 / p22 at the current poses. Only frames where the current mesh already overlaps its mask
             (IoU >= 0.5) measure the shape; also reports the mesh fraction falling OUTSIDE the mask (= too wide).
dustpan_fit: thickness scale of the dust pan about its bottom face that makes its top surface match stereo depth
             (e38, resting frames 0..29).
Usage: python mesh_fit.py pitcher|dustpan_fit"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path.home() / "v2d-vulcan/stereo"))
from object_icp import load_T  # noqa: E402

R = Path("/scratch/kghasemz/v2d")
V = "tpfsut_recon"
DS = 4


def mask(run, oid, t):
    p = run / "objects_recon/masks" / str(oid) / f"{t:06d}.png"
    return cv2.imread(str(p), 0) > 127 if p.exists() else None


def oid_of(run, nm):
    return next(o["id"] for o in json.loads((run / "objects_recon/objects.json").read_text()) if o["name"] == nm)


def pitcher():
    nm = "water_pitcher"
    mesh = trimesh.load(R / "meshes_recon" / nm / f"{nm}.obj", force="mesh")
    S = trimesh.sample.sample_surface(mesh, 3000, seed=0)[0]
    c = S.mean(0)
    up = np.array([-0.118, 0.957, -0.264]); up /= np.linalg.norm(up)
    P = (S - c) - np.outer((S - c) @ up, up)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    a1, a2 = vt[0], vt[1]
    print("mesh extents along axis1 / axis2 / up (cm):", [round(float(np.ptp((S - c) @ a)) * 100, 1) for a in (a1, a2, up)])
    grid = np.round(np.arange(0.60, 1.101, 0.025), 3)
    i1 = int(np.argmin(abs(grid - 1.0)))
    tot = np.zeros((len(grid), len(grid)))
    nfr, good, outs = 0, 0, []
    for rn in ("evaluation_ep000020", "public_ep000021", "public_ep000022"):
        run = R / "runs" / rn
        k = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        oid = oid_of(run, nm)
        cv = next(iter(sorted((run / "objects_recon" / nm).glob(".contact_vis_*.npz"))), None)
        vis = np.load(cv)["vis"] if cv else None
        n = len(list((run / V / nm / "poses").glob("*.json")))
        cand = [t for t in range(n) if vis is None or vis[t] >= 0.7]
        for t in cand[:: max(1, len(cand) // 40)]:
            m = mask(run, oid, t)
            if m is None or m.sum() < 500:
                continue
            hm = np.zeros_like(m)
            for h in (2, 3):
                x = mask(run, h, t)
                if x is not None:
                    hm |= x
            Hs, Ws = m.shape[0] // DS, m.shape[1] // DS
            mo = cv2.resize(m.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0
            care = ~(cv2.resize(hm.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0)
            T = load_T(run / V / nm / "poses" / f"{t:06d}.json")
            nfr += 1
            fi = np.zeros_like(tot)
            for i, s1 in enumerate(grid):
                for j, s2 in enumerate(grid):
                    Q = c + ((S - c) @ a1)[:, None] * s1 * a1 + ((S - c) @ a2)[:, None] * s2 * a2 + ((S - c) @ up)[:, None] * up
                    X = Q @ T[:3, :3].T + T[:3, 3]
                    u = (X[:, 0] / X[:, 2] * k["fx"] + k["cx"]) / DS
                    v = (X[:, 1] / X[:, 2] * k["fy"] + k["cy"]) / DS
                    ok = (X[:, 2] > 0.05) & (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
                    r = np.zeros((Hs, Ws), np.uint8)
                    r[v[ok].astype(int), u[ok].astype(int)] = 1
                    r = cv2.dilate(r, np.ones((3, 3), np.uint8)) > 0
                    uni = ((r | mo) & care).sum()
                    fi[i, j] = (r & mo & care).sum() / uni if uni else 0
                    if i == i1 and j == i1:
                        outs.append((r & ~mo & care).sum() / max((r & care).sum(), 1))
            if fi[i1, i1] >= 0.5:
                good += 1
                tot += fi
    print(f"frames {nfr}, pose-ok frames {good}; mesh fraction outside the mask at 1.0: median {np.median(outs):.2f}")
    tot /= max(good, 1)
    i, j = np.unravel_index(tot.argmax(), tot.shape)
    print(f"IoU at 1.00/1.00 = {tot[i1, i1]:.3f}; best {tot[i, j]:.3f} at axis1 x{grid[i]} axis2 x{grid[j]}")
    print("IoU vs axis1 (axis2 at best):", [(float(g), round(float(x), 3)) for g, x in zip(grid, tot[:, j])])
    print("IoU vs axis2 (axis1 at best):", [(float(g), round(float(x), 3)) for g, x in zip(grid, tot[i, :])])


def dustpan_fit():
    nm, rn = "mini_dust_pan", "evaluation_ep000038"
    run = R / "runs" / rn
    k = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    tp = json.loads((run / "table_plane.json").read_text())
    up, h = np.asarray(tp["up"]), float(tp["height"])
    Vm = np.asarray(trimesh.load(R / "meshes_recon" / nm / f"{nm}.obj", force="mesh").vertices)
    _, evec = np.linalg.eigh(np.cov((Vm - Vm.mean(0)).T))
    thin = evec[:, 0]
    T0 = load_T(run / "recon_hawor/slam_poses/000000.json") @ load_T(run / V / nm / "poses/000000.json")
    if (T0[:3, :3] @ thin) @ up < 0:
        thin = -thin
    d = Vm @ thin
    base = d.min()
    oid = oid_of(run, nm)
    pts = []
    for t in range(30):
        m = mask(run, oid, t)
        dp = run / "depth_stereo" / f"{t:06d}.png"
        if m is None or not dp.exists():
            continue
        m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        png = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(float)
        D = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, 0)
        vv, uu = np.nonzero(m & (D > 0.1))
        z = D[vv, uu]
        Pc = np.stack([(uu - k["cx"]) / k["fx"] * z, (vv - k["cy"]) / k["fy"] * z, z], 1)
        Tw = load_T(run / "recon_hawor/slam_poses" / f"{t:06d}.json")
        pts.append(Pc @ Tw[:3, :3].T + Tw[:3, 3])
    P = np.concatenate(pts)
    zp = P @ up - h
    X0 = Vm @ T0[:3, :3].T + T0[:3, 3]
    c = X0.mean(0)
    flat = lambda X: (X - c) - np.outer((X - c) @ up, up)   # noqa: E731
    _, _, vt = np.linalg.svd(flat(X0), full_matrices=False)
    ax = vt[0]
    sp = flat(P) @ ax
    best = None
    print("mesh thickness cm", round(float(np.ptp(d)) * 100, 2))
    for s in np.round(np.arange(0.2, 1.01, 0.05), 2):
        Xs = (Vm - np.outer((d - base) * (1 - s), thin)) @ T0[:3, :3].T + T0[:3, 3]
        zs = Xs @ up - h
        ss = flat(Xs) @ ax
        bins = np.linspace(ss.min(), ss.max(), 11)
        err = []
        for b0, b1 in zip(bins, bins[1:]):
            dsel, msel = (sp >= b0) & (sp < b1), (ss >= b0) & (ss < b1)
            if dsel.sum() > 30 and msel.any():
                err.append(abs(np.percentile(zp[dsel], 90) - zs[msel].max()))
        e = float(np.mean(err)) * 1000
        print(f"  thickness x{s}: mean |top error| {e:.1f} mm, lowest point {zs.min() * 1000:.1f} mm")
        if best is None or e < best[1]:
            best = (float(s), round(e, 1))
    print("BEST thickness scale", best)


if __name__ == "__main__":
    {"pitcher": pitcher, "dustpan_fit": dustpan_fit}[sys.argv[1]]()
