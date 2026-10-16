"""Make resting objects physically resting: flat on a stable face, bottom exactly on the table.

Our object poses are visually right but carry ~1-3 deg / few-mm errors. For a tall narrow object (the
mini sweeper: COM 62 mm high, base margin 2-3 mm) that is enough to put it on an edge with its COM
outside the footprint, or 3-5 mm into the table, and the simulator topples it. Here, in the
gravity-aligned table frame (table top z = 0, +z up), every *rest segment* -- object static, low and
over the table disk -- gets the smallest tilt that puts a stable convex-hull face flat on the table
(yaw and the COM's xy are kept), and is lifted/lowered so its lowest point is at z = 0. The correction
is ramped in/out over a few frames on both sides of the segment so the trajectory stays continuous.

Uses only our reconstruction, the object mesh and our stereo table estimate (no mocap / ground truth).
The COM is the mesh's area centroid (= the simulator's inertial origin: the URDF origin of the
recentered collision mesh)."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

DEFAULTS = dict(max_speed=0.02, max_ang_speed_deg=15.0, max_height=0.015, min_frames=5, max_tilt_deg=8.0,
                min_margin=0.0003, good_margin=0.003, ramp=4, support_tol=0.001)


def stable_faces(verts, com, support_tol=0.001):
    """Stable resting faces of a rigid body: [(outward normal n (object frame), COM margin m)], m > 0 means the
    COM projects inside the support polygon (the hull vertices within support_tol of the face plane)."""
    hull = trimesh.convex.convex_hull(verts)
    normals = []
    for n in hull.face_normals[np.argsort(-hull.area_faces)]:  # big faces first; cluster within ~1 deg
        if not normals or np.max(np.asarray(normals) @ n) < np.cos(np.radians(1.0)):
            normals.append(n)
    V = np.asarray(hull.vertices)
    out = []
    for n in normals:
        d = V @ n
        sup = V[d > d.max() - support_tol]
        if len(sup) < 3:
            continue
        a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        P2 = np.c_[sup @ a, sup @ b]
        try:
            h = ConvexHull(P2)
        except Exception:
            continue
        c2 = np.array([com @ a, com @ b])
        m = -(h.equations[:, :2] @ c2 + h.equations[:, 2]).max()
        out.append((n, float(m)))
    return out


def rest_segments(o2w, verts, disk=None, max_speed=0.02, max_ang_speed_deg=15.0, max_height=0.015, min_frames=5,
                  fps=20.0, **_):
    """Frame runs [a, b) where the object is static (translation < max_speed m/s, rotation < max_ang_speed
    deg/s), its lowest point within max_height of the table top and its centre over the table disk."""
    n = len(o2w)
    p = o2w[:, :3, 3]
    v = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1)] * fps
    R = Rotation.from_matrix(o2w[:, :3, :3])
    w = np.r_[0.0, np.degrees((R[1:] * R[:-1].inv()).magnitude())] * fps
    v, w = np.maximum(v, np.r_[v[1:], 0]), np.maximum(w, np.r_[w[1:], 0])  # a frame is static if both sides are
    zmin = np.array([(verts @ o2w[t, :3, :3].T)[:, 2].min() + p[t, 2] for t in range(n)])
    ok = (v < max_speed) & (w < max_ang_speed_deg) & (np.abs(zmin) < max_height)
    if disk is not None:
        c, r = np.asarray(disk["center"], float), float(disk["radius"])
        ok &= np.linalg.norm(p[:, :2] - c[:2], axis=1) < r
    segs, t = [], 0
    while t < n:
        if ok[t]:
            s = t
            while t < n and ok[t]:
                t += 1
            if t - s >= min_frames:
                segs.append((s, t))
        else:
            t += 1
    return segs, zmin


def settle(o2w, verts, com, disk=None, fps=20.0, **kw):
    """Returns (settled o2w copy, report). o2w: (N,4,4) object-to-table-frame poses (table top z = 0, +z up);
    verts: (V,3) collision-mesh vertices and com: (3,) centre of mass, both in the object frame."""
    cfg = {**DEFAULTS, **kw}
    out = np.array(o2w, dtype=np.float64, copy=True)
    n = len(out)
    segs, zmin = rest_segments(out, verts, disk=disk, fps=fps, **cfg)
    faces = stable_faces(verts, com, cfg["support_tol"])
    rot = np.zeros((n, 3))   # correction rotation vector (world frame, applied about the COM)
    dz = np.zeros(n)
    wgt = np.zeros(n)        # ramp weight of the correction (1 inside segments)
    report = []
    for a, b in segs:
        Rs = Rotation.from_matrix(out[a:b, :3, :3])
        down = Rs.inv().apply(np.tile([0, 0, -1.0], (b - a, 1))).mean(0)  # mean table normal in the object frame
        down /= np.linalg.norm(down)
        cand = [(np.degrees(np.arccos(np.clip(f @ down, -1, 1))), f, m) for f, m in faces if m >= cfg["min_margin"]]
        cand = [c for c in cand if c[0] <= cfg["max_tilt_deg"]]
        if not cand:
            report.append({"frames": [a, b], "applied": False,
                           "reason": f"no stable face within {cfg['max_tilt_deg']} deg"})
            continue
        # the nearest face that is comfortably stable; else the most stable one (e.g. a scanned rounded base whose
        # best rigid face holds the COM < 1 mm from the edge) -- the same face for every segment on that base
        good = [c for c in cand if c[2] >= cfg["good_margin"]]
        tilt, f, margin = min(good, key=lambda c: c[0]) if good else max(cand, key=lambda c: c[2])
        for t in range(a, b):
            Rt = Rotation.from_matrix(out[t, :3, :3])
            d = Rotation.align_vectors([[0, 0, -1.0]], [Rt.apply(f)])[0]  # minimal tilt: horizontal axis, yaw kept
            rot[t] = d.as_rotvec()
            c = Rt.apply(com) + out[t, :3, 3]
            Rn = d * Rt
            low = (verts @ Rn.as_matrix().T)[:, 2].min() + (c - Rn.apply(com))[2]
            dz[t] = -low
            wgt[t] = 1.0
        report.append({"frames": [a, b], "applied": True, "tilt_deg": round(float(tilt), 2),
                       "com_margin_mm": round(margin * 1000, 1),
                       "dz_mm": [round(float(dz[a:b].min()) * 1000, 1), round(float(dz[a:b].max()) * 1000, 1)],
                       "zmin_before_mm": [round(float(zmin[a:b].min()) * 1000, 1),
                                          round(float(zmin[a:b].max()) * 1000, 1)]})
    k = int(cfg["ramp"])
    # short gaps (<= 2 ramps) between settled segments: carry the correction across instead of fading it out and in
    done = [(r["frames"][0], r["frames"][1]) for r in report if r["applied"]]
    for (_, b1), (a2, _) in zip(done, done[1:]):
        g = a2 - b1
        if 0 < g <= 2 * k:
            for t in range(b1, a2):
                s = (t - b1 + 1) / (g + 1)
                rot[t], dz[t], wgt[t] = (1 - s) * rot[b1 - 1] + s * rot[a2], (1 - s) * dz[b1 - 1] + s * dz[a2], 1.0
    # ramp the boundary correction out over cfg["ramp"] frames on each side (not into another segment)
    for a, b in [(r["frames"][0], r["frames"][1]) for r in report if r["applied"]]:
        for j in range(1, k + 1):
            s = 1.0 - j / (k + 1)
            for t, src in ((a - j, a), (b - 1 + j, b - 1)):
                if 0 <= t < n and wgt[t] < s:
                    rot[t], dz[t], wgt[t] = rot[src] * s, dz[src] * s, s
    for t in np.where(wgt > 0)[0]:
        Rt = Rotation.from_matrix(out[t, :3, :3])
        d = Rotation.from_rotvec(rot[t])
        c = Rt.apply(com) + out[t, :3, 3]                     # rotate about the COM: its xy stays put
        Rn = d * Rt
        out[t, :3, :3] = Rn.as_matrix()
        out[t, :3, 3] = c - Rn.apply(com) + [0, 0, dz[t]]
    return out, report


def mesh_verts_com(obj_path):
    m = trimesh.load(str(obj_path), force="mesh")
    return np.array(m.vertices, float), np.array(m.centroid, float)  # copies: trimesh caches are read-only
