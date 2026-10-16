"""Identity check for same-size objects (e.g. wooden_piece_1 / wooden_piece_2) on
the host: GroundingDINO/SAM2 cannot tell identical-looking boards apart, but their
scans differ in the mocap markers / slot layout. For frames where both tracks are
valid, each mask's stereo point cloud is registered (trimmed ICP from 4 PCA
initialisations) to each mesh; the assignment with the lower residual wins.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / "v2d-vulcan/stereo"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from object_icp import icp  # noqa: E402
from geom import backproject, decode_depth, load_mask  # noqa: E402


def same_size_pairs(objs, tol=0.1):
    out = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a = np.sort(objs[i]["mesh_ext"])
            b = np.sort(objs[j]["mesh_ext"])
            if np.all(np.abs(a - b) <= tol * np.maximum(a, b)):
                out.append((objs[i], objs[j]))
    return out


def pca(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    return c, Vt.T  # columns sorted by variance, descending


def residual(scan, model_pts, model_nrm):
    """Best trimmed-mean scan->visible-model distance (mm) over PCA inits."""
    cs, Us = pca(scan)
    cm, Um = pca(model_pts)
    best = np.inf
    for s1 in (1, -1):
        for s2 in (1, -1):
            D = np.diag([s1, s2, 1.0])
            R = Us @ D @ Um.T
            if np.linalg.det(R) < 0:
                D[2, 2] = -1
                R = Us @ D @ Um.T
            T0 = np.eye(4)
            T0[:3, :3] = R
            T0[:3, 3] = cs - R @ cm
            T, _, _ = icp(model_pts, model_nrm, scan, T0)
            P = model_pts @ T[:3, :3].T + T[:3, 3]
            P = P[(model_nrm @ T[:3, :3].T * P).sum(1) < 0]
            if len(P) < 50:
                continue
            d, _ = cKDTree(P).query(scan)
            d = np.sort(d)[: int(0.9 * len(d))]
            best = min(best, float(d.mean() * 1000))
    return best


def resolve(run: Path, work: Path, objs, n, max_frames=12):
    ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
    report = []
    for oa, ob in same_size_pairs(objs):
        meshes = {}
        for o in (oa, ob):
            m = trimesh.load(o["mesh"], force="mesh")
            pts, fi = trimesh.sample.sample_surface(m, 6000, seed=0)
            meshes[o["id"]] = (pts, m.face_normals[fi])
        both = [t for t in range(n) if (work / f"masks/{oa['id']}/{t:06d}.png").exists()
                and (work / f"masks/{ob['id']}/{t:06d}.png").exists()]
        sel = [both[int(k)] for k in np.linspace(0, len(both) - 1, min(max_frames, len(both)))] if both else []
        rows = []
        for t in sel:
            depth = decode_depth(run / f"depth_stereo/{t:06d}.png")
            scans = {}
            for o in (oa, ob):
                m = load_mask(work / f"masks/{o['id']}/{t:06d}.png")
                m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                P = backproject(m, depth, K, max_pts=3000, seed=t)
                scans[o["id"]] = P if len(P) >= 300 else None
            if scans[oa["id"]] is None or scans[ob["id"]] is None:
                continue
            r = {(i, j): residual(scans[i], *meshes[j]) for i in (oa["id"], ob["id"]) for j in (oa["id"], ob["id"])}
            keep = r[(oa["id"], oa["id"])] + r[(ob["id"], ob["id"])]
            swap = r[(oa["id"], ob["id"])] + r[(ob["id"], oa["id"])]
            rows.append({"frame": t, "keep_mm": keep, "swap_mm": swap})
            print(f"identity {oa['name']}/{ob['name']} t={t}: keep {keep:.2f} mm, swap {swap:.2f} mm", flush=True)
        if not rows:
            report.append({"pair": [oa["name"], ob["name"]], "decision": "unknown", "rows": []})
            continue
        dk = np.array([x["keep_mm"] - x["swap_mm"] for x in rows])
        votes_swap = int((dk > 0).sum())
        do_swap = bool(np.median(dk) > 0 and votes_swap > len(dk) / 2)
        confident = bool(abs(np.median(dk)) > 0.1 and max(votes_swap, len(dk) - votes_swap) >= 0.7 * len(dk))
        report.append({"pair": [oa["name"], ob["name"]], "decision": "swap" if do_swap else "keep",
                       "confident": confident, "median_keep_minus_swap_mm": float(np.median(dk)),
                       "votes_swap": votes_swap, "n": len(dk), "rows": rows})
        if do_swap:
            for sub in ("masks", "masks_ekf"):
                a, b, tmp = work / sub / str(oa["id"]), work / sub / str(ob["id"]), work / sub / "_swap"
                a.rename(tmp)
                b.rename(a)
                tmp.rename(b)
    return report
