"""Remove physically impossible object-in-table poses from our reconstructed object trajectories.

Two failure modes seen in the plank episodes (ep23/ep24, table frame: z = 0 table top, +z up):
1. A static object whose estimated pose drifts while it is occluded: plank 1 stands upright at the table centre
   (centre still to ~1 cm/s) while its in-plane rotation drifts ~20 deg, so a corner goes up to 4 cm into the
   table for ~4 s, until rest_settle snaps it back. A real object cannot do that.
   -> hold_static: a slow block of frames that go > tol into the table (over the disk) is replaced by a slerp
   between the valid frames just before and after it (or a hold of the one valid end).
   Frames that are physically possible are never touched.
2. A moving (slid / carried) object a few mm to ~2 cm into the table.
   -> lift_out_of_table: each frame is lifted by exactly its depth below z = 0 (only vertices over the table
   disk count, so a part overhanging the rim is free to go lower), with a max-filter + box-filter of equal width
   so the lift is continuous and never below the per-frame requirement.
Run hold_static before rest_settle (so the held run settles as one rest segment) and lift_out_of_table after it.
Uses only our reconstruction, the object mesh and our stereo table estimate."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp


def hull_verts(verts):
    return np.asarray(verts, float)[ConvexHull(verts).vertices]


def table_depth(o2w, hv, disk):
    """(N,) depth (m, >= 0) of the object's lowest hull vertex over the table disk below the table top."""
    W = np.einsum("vj,nij->nvi", hv, o2w[:, :3, :3]) + o2w[:, None, :3, 3]
    over = np.ones(W.shape[:2], bool)
    if disk is not None:
        over = np.linalg.norm(W[..., :2] - np.asarray(disk["center"], float)[:2], axis=-1) < float(disk["radius"])
    z = np.where(over, W[..., 2], np.inf).min(1)
    return np.clip(-z, 0.0, None)


def hold_static(o2w, verts, com, disk, fps, tol=0.003, max_speed=0.03, max_gap_disp=0.04, max_len_s=8.0):
    """Bridge every slow block of frames that go > tol into the table: the pose is slerped (rotation) / lerped
    (COM) between the valid frames just before and after the block. Slow: the COM's median speed over the block
    is < max_speed and the two valid ends are < max_gap_disp apart (else lift_out_of_table handles it). A block
    at the start/end of the sequence holds its one valid end."""
    out = np.array(o2w, np.float64, copy=True)
    n = len(out)
    depth = table_depth(out, hull_verts(verts), disk)
    c = np.einsum("nij,j->ni", out[:, :3, :3], com) + out[:, :3, 3]
    v = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)] * fps
    report, t = [], 0
    while t < n:
        if depth[t] <= tol:
            t += 1
            continue
        s = t
        while t < n and depth[t] > tol:
            t += 1
        l, r = (s - 1 if s > 0 else None), (t if t < n else None)
        rec = {"frames": [s, t], "max_depth_mm_before": round(float(depth[s:t].max()) * 1000, 1),
               "median_speed": round(float(np.median(v[s:t])), 3)}
        ends = [e for e in (l, r) if e is not None]
        gap = float(np.linalg.norm(c[l] - c[r])) if len(ends) == 2 else 0.0
        if not ends or np.median(v[s:t]) >= max_speed or gap >= max_gap_disp or (t - s) > max_len_s * fps:
            report.append({**rec, "applied": False, "reason": f"moving or long (gap {gap * 100:.1f} cm)"})
            continue
        if len(ends) == 2:
            Rs = Slerp([l, r], Rotation.from_matrix(out[[l, r], :3, :3]))(np.arange(s, t)).as_matrix()
            w = ((np.arange(s, t) - l) / (r - l))[:, None]
            out[s:t, :3, :3] = Rs
            out[s:t, :3, 3] = (1 - w) * c[l] + w * c[r] - Rs @ com
        else:
            out[s:t] = out[ends[0]]
        report.append({**rec, "applied": True, "replaced_frames": t - s,
                       "max_depth_mm_after": round(float(table_depth(out[s:t], hull_verts(verts), disk).max()) * 1000, 1)})
    return out, report

def lift_out_of_table(o2w, verts, disk, half_width=3):
    out = np.array(o2w, np.float64, copy=True)
    depth = table_depth(out, hull_verts(verts), disk)
    w = 2 * half_width + 1
    dz = uniform_filter1d(maximum_filter1d(depth, w, mode="nearest"), w, mode="nearest")
    dz = np.maximum(dz, depth)  # exact at the sequence ends too
    out[:, 2, 3] += dz
    return out, {"lifted_frames": int((dz > 5e-4).sum()), "max_lift_mm": round(float(dz.max()) * 1000, 1),
                 "frames_over_5mm_before": int((depth > 0.005).sum())}


def load_parts(path):
    """Convex collision parts (object frame) written by collision_parts.py, or None."""
    try:
        with np.load(path) as d:
            return [np.asarray(d[f"v{k}"], float) for k in range(int(d["n"]))]
    except FileNotFoundError:
        return None


def _world_eqs(parts, T):
    return [ConvexHull(v @ T[:3, :3].T + T[:3, 3]).equations for v in parts]


def _inside(P, eqs, dt=None):
    m = np.zeros(len(P), bool)
    for e in eqs:
        off = e[:, 3] if dt is None else e[:, 3] - e[:, :3] @ dt  # the hull moved by dt
        m |= (P @ e[:, :3].T + off <= 0).all(1)
    return m


def snap_assembly(o2w, parts, fps, max_shift=0.06, accept_cm3=0.3, near=0.02, ramp=10, n_mc=300000, seed=0):
    """Objects that end up assembled (e.g. two slotted pieces in an X) but overlap in the simulator's collision
    geometry at the last frame: the object that moved last gets the smallest translation (search over shells up to
    max_shift, then a 1 mm local refinement) that brings the overlap below accept_cm3, applied from the frame
    the pair comes within `near` (AABB gap, original poses) to the end, ramped in over `ramp` frames before it.
    o2w: list of (N,4,4) table-frame poses; parts: list of convex part lists (object frame) or None.
    Returns (per-object (N,3) world translations, report)."""
    nb, n = len(o2w), len(o2w[0])
    deltas = [np.zeros((n, 3)) for _ in range(nb)]
    report = []
    rng = np.random.default_rng(seed)

    def last_move(i):
        c = o2w[i][:, :3, 3]
        v = uniform_filter1d(np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)] * fps, 5, mode="nearest")
        idx = np.where(v > 0.03)[0]
        return int(idx[-1]) if len(idx) else -1

    for i in range(nb):
        for j in range(i + 1, nb):
            if parts[i] is None or parts[j] is None:
                continue
            mover, still = (i, j) if last_move(i) >= last_move(j) else (j, i)
            Ts, Tm = o2w[still][-1], o2w[mover][-1]
            Vs = np.concatenate([v @ Ts[:3, :3].T + Ts[:3, 3] for v in parts[still]])
            lo, hi = Vs.min(0), Vs.max(0)
            P = rng.uniform(lo, hi, (n_mc, 3))
            cell = float(np.prod(hi - lo)) / n_mc * 1e6  # cm3 per sample
            P = P[_inside(P, _world_eqs(parts[still], Ts))]
            Em = _world_eqs(parts[mover], Tm)
            ov0 = _inside(P, Em).sum() * cell
            if ov0 < accept_cm3:
                continue
            # sideways first (a slot seats by a horizontal shift; a vertical lift would rest one piece on the other),
            # then any direction; each shell's best is refined in 1 mm steps before the next, larger shell
            horiz = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
            full = [(np.array(d, float) - 1) / np.linalg.norm(np.array(d, float) - 1)
                    for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)] + horiz
            steps = [np.array(x, float) - 1 for x in np.ndindex(3, 3, 3) if x != (1, 1, 1)]

            def refine(b):
                for _ in range(4):
                    for d in steps:
                        cand = b[1] + 0.001 * d
                        ov = _inside(P, Em, cand).sum() * cell
                        if ov < b[0] - 1e-9:
                            b = (ov, cand)
                return b

            best = (ov0, np.zeros(3))
            for dirs in (horiz, full):
                for r in np.arange(0.002, max_shift + 1e-9, 0.002):
                    sb = min(((_inside(P, Em, r * d).sum() * cell, r * d) for d in dirs), key=lambda x: x[0])
                    if sb[0] < 3 * accept_cm3:
                        sb = refine(sb)
                    if sb[0] < best[0]:
                        best = sb
                    if best[0] < accept_cm3:
                        break
                if best[0] < accept_cm3:
                    break
            rec = {"pair": [i, j], "mover": mover, "overlap_cm3_before": round(ov0, 2),
                   "overlap_cm3_after": round(best[0], 2), "shift_cm": np.round(best[1] * 100, 2).tolist()}
            if best[0] >= accept_cm3:
                report.append({**rec, "applied": False, "reason": f"no shift <= {max_shift * 100:.0f} cm seats it"})
                continue
            # window: from the frame the pair comes within `near` (AABB gap) to the end, original poses
            Vm0 = [np.concatenate([v for v in parts[k]]) for k in (still, mover)]
            s = n - 1
            while s > 0:
                A = Vm0[0] @ o2w[still][s - 1, :3, :3].T + o2w[still][s - 1, :3, 3]
                B = Vm0[1] @ o2w[mover][s - 1, :3, :3].T + o2w[mover][s - 1, :3, 3]
                gap = np.max(np.maximum(B.min(0) - A.max(0), A.min(0) - B.max(0)))
                if gap > near:
                    break
                s -= 1
            w = np.clip((np.arange(n) - (s - ramp)) / ramp, 0.0, 1.0)
            deltas[mover] += w[:, None] * best[1]
            report.append({**rec, "applied": True, "frames": [max(s - ramp, 0), s, n]})
    return deltas, report
