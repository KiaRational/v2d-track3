"""Metric size of a reconstructed mesh from OUR stereo depth (no ground truth): fit the mesh with a free scale to the
object's stereo point cloud in each picked evaluation view, take the median scale over views.

Per view (pick_views.py candidates: unoccluded, free-standing object): the eroded mask's stereo points (camera frame,
robust depth outliers removed) are a partial surface. The mesh (meshes_recon/<obj>/<obj>.obj) is registered to them
with scaled ICP (points -> mesh, one-sided: only the visible side is observed) from 24 axis-aligned starting rotations
around the PCA alignment; the best fit's scale is that view's estimate. The median over views (robust to a bad
view) rescales the mesh in place (a .orig copy is kept) and is recorded in PROVENANCE.json.
Usage: python stereo_scale.py <obj> [--apply]   (kaggle venv)
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ROOT = Path("/scratch/kghasemz/v2d")


def view_points(c, mask):
    run = ROOT / "runs" / c["run"]
    man = json.loads((run / "hoi4d/scene_manifest.json").read_text())
    K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
    png = np.asarray(Image.open(run / "depth_stereo" / f"{c['frame']:06d}.png")).astype(float)
    D = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, np.nan)
    m = binary_erosion(mask, iterations=6) & np.isfinite(D) & (D > 0.1) & (D < 2.0)
    v, u = np.nonzero(m)
    z = D[v, u]
    med = np.median(z)
    keep = np.abs(z - med) < 0.25            # drop background / flying pixels
    u, v, z = u[keep], v[keep], z[keep]
    P = np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]
    if len(P) > 6000:
        P = P[np.random.default_rng(0).choice(len(P), 6000, replace=False)]
    return P


def fit_scale(P, mesh, iters=40):
    """Best start: (median point-to-mesh error, scale)."""
    err, s, _, _ = min(fit_starts(P, mesh, iters), key=lambda r: r[0])
    return err, s


def fit_starts(P, mesh, iters=40):
    """Scaled ICP from 24 starting rotations: list of (median error, s, R, t) with M = s * S @ R.T + t."""
    S = trimesh.sample.sample_surface(mesh, 8000, seed=0)[0]
    pc, pcen = P - P.mean(0), P.mean(0)
    runs = []
    for R0 in Rot.from_euler("xyz", np.array(list(product([0, 90, 180, 270], [0, 90, 180, 270], [0])) [:16] + [(90, 0, 90), (90, 0, 270), (0, 90, 90), (0, 270, 90), (270, 0, 90), (0, 90, 270), (90, 90, 0), (270, 90, 0)]), degrees=True).as_matrix():
        s = np.cbrt(np.prod(np.ptp(pc, 0) + 1e-6) / np.prod(np.ptp(S, 0) + 1e-6)) if True else 1.0
        s = float(np.clip(s, 0.3, 3.0))
        R, t = R0, pcen - s * (R0 @ S.mean(0))
        for _ in range(iters):  # scaled ICP, points -> transformed mesh samples
            M = s * S @ R.T + t
            d, idx = cKDTree(M).query(P)
            inl = d < np.percentile(d, 90)
            A, B = (S[idx[inl]]), P[inl]
            ma, mb = A.mean(0), B.mean(0)
            Ac, Bc = A - ma, B - mb
            U, sig, Vt = np.linalg.svd(Bc.T @ Ac)
            Dg = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
            R = U @ Dg @ Vt
            s = float(np.clip((sig * np.diag(Dg)).sum() / (Ac ** 2).sum(), 0.3, 3.0))
            t = mb - s * R @ ma
        M = s * S @ R.T + t
        runs.append((float(np.median(cKDTree(M).query(P)[0])), s, R, t))
    return runs


def main():
    obj = sys.argv[1]
    mp = ROOT / "meshes_recon" / obj / f"{obj}.obj"
    mesh = trimesh.load(mp, force="mesh")
    V = ROOT / "recon_meshes/views" / obj
    cands = json.loads((V / "cands.json").read_text())
    masks = np.load(V / "cands.npz")["masks"]
    est = []
    for c, m in zip(cands, masks):
        P = view_points(c, m)
        if len(P) < 300:
            continue
        err, s = fit_scale(P, mesh)
        est.append({"run": c["run"], "frame": c["frame"], "scale": s, "fit_median_mm": err * 1000, "points": len(P)})
        print(f"{obj} {c['run'][-2:]} f{c['frame']}: scale {s:.3f}, fit median {err*1000:.1f} mm ({len(P)} pts)", flush=True)
    good = [e["scale"] for e in est if e["fit_median_mm"] < 5.0] or [e["scale"] for e in est]
    s = float(np.median(good))
    print(f"{obj}: median scale {s:.3f} over {len(good)} views -> extent {np.round(np.sort(mesh.extents)[::-1] * s * 100, 1).tolist()} cm", flush=True)
    if "--apply" in sys.argv:
        orig = mp.with_suffix(".orig.obj")
        if not orig.exists():
            mesh.export(orig)
        else:  # always rescale from the SAM 3D size, never compound
            mesh = trimesh.load(orig, force="mesh")
        mesh.apply_scale(s)
        mesh.export(mp)
        pp = mp.parent / "PROVENANCE.json"
        prov = json.loads(pp.read_text())
        prov["stereo_scale"] = {"factor": s, "views": est, "method": "scaled ICP of the mesh to masked stereo points "
                                "(one-sided, 24 starts), median over views with fit < 5 mm"}
        prov["extent_cm"] = (np.sort(mesh.extents)[::-1] * 100).round(1).tolist()
        pp.write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
