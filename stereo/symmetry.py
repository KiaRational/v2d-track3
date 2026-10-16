"""Object symmetry: detect it from the mesh, and remove its effect from pose sequences.

A pose is only defined up to the object's own symmetry: rotating a cup about its axis,
or flipping a flat board by 180 degrees, produces the same shape and the same image, so
a pose estimator picks among those arbitrarily and its output spins or flips while the
object sits still. Detection is by ADD-S self-distance (mean distance from the rotated
surface samples back to the surface), which is exactly what "looks the same" means --
unlike an inertia-tensor test, it also catches near-symmetry of open, non-watertight
scans.

  mesh_symmetry(mesh_path, tol_mm) -> {"cont_axis": axis|None, "group": [3x3, ...]}
    cont_axis: continuous rotational symmetry (cup, pot body) -- any spin about it is
               unobservable; represented in `group` by a fine sampling.
    group:     discrete rotations (180 deg flips of a board) including identity.

  canonicalise(rots, group, ref=None) -> (rots_out, changed)
    Sequentially replaces each rotation R by R @ g, g in the group, choosing the variant
    closest to the running reference. Consecutive frames then describe the same physical
    orientation with the same matrix, so flips and free spin stop appearing as motion.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

CONT_STEPS = 72  # 5 degree sampling of a continuous axis


def _axes(P):
    _, vec = np.linalg.eigh(np.cov(P.T))
    return vec.T


def mesh_symmetry(mesh_path, tol_mm: float = 5.0, n: int = 4000, seed: int = 0):
    """Symmetries of the mesh, as rotations in the object frame (mesh centroid at origin)."""
    m = trimesh.load(mesh_path, force="mesh")
    P = trimesh.sample.sample_surface(m, n, seed=seed)[0] - m.centroid
    tree = cKDTree(P)

    def adds(R):
        return float(tree.query(P @ R.T)[0].mean() * 1000)

    cont_axis, group = None, [np.eye(3)]
    for ax in _axes(P):
        # continuous symmetry: invariant at several unrelated angles
        if max(adds(Rotation.from_rotvec(np.deg2rad(a) * ax).as_matrix()) for a in (37, 90, 143)) < tol_mm:
            cont_axis = ax
            group = [Rotation.from_rotvec(2 * np.pi * k / CONT_STEPS * ax).as_matrix() for k in range(CONT_STEPS)]
            break
    if cont_axis is None:
        for ax in _axes(P):
            R = Rotation.from_rotvec(np.pi * ax).as_matrix()
            if adds(R) < tol_mm:
                group.append(R)
        # products of two independent 180 deg flips are themselves symmetries
        for i in range(1, len(group)):
            for j in range(i + 1, len(group)):
                G = group[i] @ group[j]
                if not any(np.allclose(G, g, atol=1e-6) for g in group):
                    group.append(G)
    return {"cont_axis": None if cont_axis is None else cont_axis, "group": group, "center": np.asarray(m.centroid),
            "kind": "continuous" if cont_axis is not None else ("discrete" if len(group) > 1 else "none")}


def canonicalise(rots, group, ref=None):
    """Map each rotation to the symmetry-equivalent variant closest to the running reference.

    rots: dict {frame: 3x3} or list; returns the same type plus the number changed."""
    if len(group) <= 1:
        return rots, 0
    G = np.asarray(group)
    items = sorted(rots.items()) if isinstance(rots, dict) else list(enumerate(rots))
    out, changed = {}, 0
    for t, R in items:
        R = np.asarray(R)
        if ref is None:
            ref = R
        cand = R @ G                                   # (g,3,3)
        tr = np.einsum("ij,gij->g", ref, cand)         # trace(ref^T cand) -- larger is closer
        k = int(tr.argmax())
        if k:
            changed += 1
        ref = cand[k]
        out[t] = cand[k]
    if not isinstance(rots, dict):
        return [out[i] for i in range(len(items))], changed
    return out, changed
