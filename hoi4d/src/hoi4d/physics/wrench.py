"""Quasi-static contact-wrench feasibility over bounded parameter ranges (README 7 "dynamic feasibility" and
9 "report feasibility over plausible ranges rather than claiming recovery of true forces").

At frame f the object needs net wrench  w = [m (a - g); I alpha ~ 0]  about its COM. Contacts: hand points within
contact distance of the surface (normal = surface normal, pointing into the object) and support contacts (table
top / lower object, normal +z). Each contact force lies in a linearised friction cone (8 edges, mu) with bounded
magnitude f_max; feasibility = bounded least squares residual of sum_k G_k lambda_k = w, normalised by |w|.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from ..geometry.assets import asset, inv, transform

G = np.array([0, 0, -9.81])


def surface_normals(name):
    a = asset(name)
    if not hasattr(a, "normals"):
        mesh = trimesh.load(a_path(name), force="mesh")
        _, _, fid = trimesh.proximity.closest_point(mesh, a.surface)
        a.normals = mesh.face_normals[fid]
    return a.normals


def a_path(name):
    from .. import io as hio
    return hio.ROOT / "meshes_obj" / name / f"{name}.obj"


def cone_edges(n: np.ndarray, mu: float, k: int = 8) -> np.ndarray:
    """Unit generators of the linearised friction cone around inward normal n."""
    n = n / np.linalg.norm(n)
    t1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    E = n[None] + mu * (np.cos(ang)[:, None] * t1 + np.sin(ang)[:, None] * t2)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def frame_contacts(tr, name, f, contact_m=0.01, support=True):
    """World contact points + inward normals for object `name` at frame f."""
    o = tr.objects[name]
    T = o["T_world_obj"][f]
    a = asset(name)
    nrm = surface_normals(name)
    tree = cKDTree(a.surface)
    pts, nors = [], []
    from ..diagnostics.contacts import hand_points
    for h, hd in tr.hands.items():
        if not hd["valid"][f]:
            continue
        P, r = hand_points(hd["joints"][f])
        Po = transform(inv(T), P)
        d, i = tree.query(Po)
        sel = d - r < contact_m
        for k in np.where(sel)[0]:
            pts.append(transform(T, a.surface[i[k]][None])[0])
            nors.append(-(T[:3, :3] @ nrm[i[k]]))  # inward
    if support:
        Pw = transform(T, a.surface)
        low = Pw[:, 2] < Pw[:, 2].min() + 0.003
        if Pw[:, 2].min() < 0.01:  # resting on the table top (z = 0)
            for p in Pw[low][:: max(1, low.sum() // 12)]:
                pts.append(p)
                nors.append(np.array([0, 0, 1.0]))
    return np.asarray(pts).reshape(-1, 3), np.asarray(nors).reshape(-1, 3)


def feasibility(tr, name, frames, mus=(0.3, 0.6, 1.0), masses=(0.05, 0.3, 0.6), f_max=15.0, tol=0.1):
    o = tr.objects[name]
    p = o["T_world_obj"][:, :3, 3]
    acc = np.gradient(np.gradient(p, tr.t_s, axis=0), tr.t_s, axis=0)
    com_o = asset(name).centroid
    res = {f"mu={mu},m={m}": [] for mu in mus for m in masses}
    for f in frames:
        if not o["valid"][f] or not np.isfinite(acc[f]).all():
            continue
        P, N = frame_contacts(tr, name, f)
        com = transform(o["T_world_obj"][f], com_o[None])[0]
        for mu in mus:
            cols = []
            for x, n in zip(P, N):
                for e in cone_edges(n, mu):
                    cols.append(np.r_[e, np.cross(x - com, e)])
            Gm = np.asarray(cols).T if cols else np.zeros((6, 0))
            for m in masses:
                w = np.r_[m * (acc[f] - G), np.zeros(3)]
                if Gm.shape[1] == 0:
                    r = 1.0
                else:
                    sol = lsq_linear(Gm, w, bounds=(0, f_max))
                    r = float(np.linalg.norm(Gm @ sol.x - w) / max(np.linalg.norm(w), 1e-6))
                res[f"mu={mu},m={m}"].append((int(f), r, len(P)))
    out = {}
    for k, rows in res.items():
        if rows:
            r = np.array([x[1] for x in rows])
            out[k] = {"frames": len(rows), "feasible_frac": float((r < tol).mean()), "median_residual": float(np.median(r)),
                      "worst_frames": [x[0] for x in sorted(rows, key=lambda x: -x[1])[:5]]}
    return {"object": name, "tolerance": tol, "f_max_N": f_max, "settings": out,
            "note": "quasi-static (rotational inertia ignored); feasibility over ranges, not recovered forces"}
