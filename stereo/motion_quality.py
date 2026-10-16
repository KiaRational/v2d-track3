"""Small numerical helpers for motion refinement; no models or dataset I/O."""
import logging

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import BSpline


def bundle_timing(bundle, n):
    if "fps" not in bundle:
        raise ValueError("Bundle has no source fps; re-export with explicit timing")
    fps = float(bundle["fps"])
    if not np.isfinite(fps) or fps <= 0 or n < 2:
        raise ValueError("Invalid source fps or frame count")
    t = np.asarray(bundle.get("timestamps", np.arange(n) / fps), float)
    if t.shape != (n,) or not np.isfinite(t).all() or not np.allclose(np.diff(t), 1 / fps, rtol=1e-3, atol=1e-6):
        raise ValueError("Bundle timestamps must match source fps; resample the whole scene together")
    return fps, t


def interpolate_short_gaps(values, valid, timestamps, max_gap_seconds=0.15, rotations=False):
    """Fill bounded gaps only. Rotations are axis-angle (..., 3), interpolated on SO(3).

    Leading/trailing gaps and long occlusions need a reconstruction, not extrapolation.
    Observed values are never changed. Returns filled values and an imputation mask.
    """
    x = np.asarray(values, float).copy()
    valid = np.asarray(valid, bool)
    t = np.asarray(timestamps, float)
    if valid.shape != (len(x),) or t.shape != valid.shape or len(t) < 2:
        raise ValueError("Motion needs matching values, validity and at least two timestamps")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if not np.isfinite(x[valid]).all():
        raise ValueError("Non-finite observed motion")
    missing = ~valid
    if missing.any() and (not valid[0] or not valid[-1]):
        raise ValueError("Unbounded hand gap: reconstruct or explicitly crop the whole scene")
    for a, b in contiguous_segments(np.flatnonzero(missing)):
        left, right = a - 1, b + 1
        if t[right] - t[left] > max_gap_seconds + 1e-9:
            raise ValueError(f"Hand gap {a}:{b} exceeds {max_gap_seconds:g} s; reconstruct it first")
        alpha = (t[a:b + 1] - t[left]) / (t[right] - t[left])
        if rotations:
            if x.shape[-1] != 3:
                raise ValueError("Axis-angle rotations need a final dimension of three")
            flat = x.reshape(len(x), -1, 3)
            for j in range(flat.shape[1]):
                flat[a:b + 1, j] = Slerp(t[[left, right]], Rotation.from_rotvec(flat[[left, right], j]))(t[a:b + 1]).as_rotvec()
        else:
            w = alpha.reshape((-1,) + (1,) * (x.ndim - 1))
            x[a:b + 1] = (1 - w) * x[left] + w * x[right]
    return x, missing


def contiguous_segments(frames):
    frames = np.asarray(frames, int)
    if not len(frames):
        return []
    return [(int(s[0]), int(s[-1])) for s in np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)]


SPLINE_RIDGE = 0.1  # weight of the control-point curvature ridge, relative to one frame's residual


def debounce_phases(phases, min_frames=3):
    """Contact phases without flicker: runs shorter than min_frames are merged into their
    longer neighbour (a 1-2 frame hole inside a grasp is filled, a 1-2 frame touch dropped)."""
    out = list(phases)
    while True:
        cuts = [0] + [i for i in range(1, len(out)) if out[i] != out[i - 1]] + [len(out)]
        runs = list(zip(cuts[:-1], cuts[1:]))
        short = [k for k, (s, e) in enumerate(runs) if e - s < min_frames]
        if len(runs) <= 1 or not short:
            return out
        k = min(short, key=lambda k: runs[k][1] - runs[k][0])
        j = max([j for j in (k - 1, k + 1) if 0 <= j < len(runs)], key=lambda j: runs[j][1] - runs[j][0])
        s, e = runs[k]
        out[s:e] = [out[runs[j][0]]] * (e - s)


def spline_bases(frames, fps, phases=None, knot_seconds=0.15, max_gap_seconds=0.15):
    """Clamped cubic B-spline bases per span, split at contact changes and gaps.

    Returns [(indices, B (len, m), D (m-2, m))]: D is the second divided difference of the
    control points at their Greville abscissae, scaled by knot_seconds^2 (uniform knots:
    [1, -2, 1]), so constant-velocity motion is in its null space. Spans with fewer frames
    than basis functions are omitted (no spline term there, rather than an interpolant).
    """
    frames = np.asarray(frames)
    if not np.isfinite(fps) or fps <= 0 or knot_seconds <= 0:
        raise ValueError("Positive fps and spline knot spacing required")
    if len(frames) < 4:
        return []
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Frame indices must increase")
    phases = [None] * len(frames) if phases is None else phases
    if len(phases) != len(frames):
        raise ValueError("One contact phase per frame required")
    cuts = [i for i in range(1, len(frames))
            if phases[i] != phases[i - 1] or (frames[i] - frames[i - 1]) / fps > max_gap_seconds]
    out = []
    for idx in np.split(np.arange(len(frames)), cuts):
        if len(idx) < 5:
            continue
        t = (frames[idx] - frames[idx[0]]) / fps
        inside = np.arange(knot_seconds, t[-1] - 1e-9, knot_seconds)
        if len(inside) and t[-1] - inside[-1] < 0.5 * knot_seconds:
            inside = inside[:-1]  # merge a tiny last interval: near-duplicate knots are ill-conditioned
        knots = np.r_[np.repeat(t[0], 4), inside, np.repeat(t[-1], 4)]
        n_basis = len(knots) - 4
        if n_basis >= len(idx):
            continue  # insufficient evidence to smooth this short phase
        B = BSpline(knots, np.eye(n_basis), 3)(t)
        g = np.array([knots[j + 1:j + 4].mean() for j in range(n_basis)])  # Greville abscissae
        D = np.zeros((max(n_basis - 2, 0), n_basis))
        for j in range(1, n_basis - 1):
            a, b, c = g[j - 1], g[j], g[j + 1]
            D[j - 1, j - 1:j + 2] = 2 * knot_seconds ** 2 * np.array(
                [1 / ((b - a) * (c - a)), -1 / ((b - a) * (c - b)), 1 / ((c - b) * (c - a))])
        out.append((idx, B, D))
    return out


def spline_projectors(frames, fps, phases=None, knot_seconds=0.15, max_gap_seconds=0.15, ridge=SPLINE_RIDGE):
    """Least-squares cubic B-spline smoothers, split at contact changes and gaps.

    Operates on Euclidean positions or local rotation tangent vectors, never raw
    quaternion components. Each result is (indices, B (B'B + ridge D'D)^-1 B'); the small
    curvature ridge keeps clamped ends from interpolating their last noisy frame.
    """
    return [(idx, B @ np.linalg.solve(B.T @ B + ridge * D.T @ D, B.T))
            for idx, B, D in spline_bases(frames, fps, phases, knot_seconds, max_gap_seconds)]


def spline_loss(points, projectors):
    """Differentiable distance of complete world trajectories to a cubic spline.

    This is a soft prior: contact and data terms can override it. Optimizing MANO
    parameters rather than directly replacing joints preserves hand kinematics.
    """
    import torch
    total = points.sum() * 0
    count = 0
    for idx, projection in projectors:
        x = points[idx].reshape(len(idx), -1)
        P = torch.as_tensor(projection, dtype=points.dtype, device=points.device)
        total = total + ((x - P @ x) ** 2).sum()
        count += x.numel()
    return total / max(count, 1)


def chordal_mean(Rs, w=None):
    """Weighted chordal L2 mean of rotations (n, 3, 3)."""
    M = np.einsum("n,nij->ij", np.ones(len(Rs)) if w is None else np.asarray(w, float), np.asarray(Rs))
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def unwrap_rotvecs(rv):
    """Sequentially replace each axis-angle v by the equivalent v (1 + 2 pi k / |v|), k in
    {-1, 0, 1}, nearest the previous one, so a path through angle pi stays continuous."""
    out = np.array(rv, float, copy=True)
    for i in range(1, len(out)):
        a = np.linalg.norm(out[i])
        if a > 1e-9:
            cand = out[i][None] * (1 + 2 * np.pi * np.array([-1., 0., 1.]) / a)[:, None]
            out[i] = cand[np.argmin(np.linalg.norm(cand - out[i - 1], axis=1))]
    return out


def spline_pose_prior(poses, frames, fps, phases=None, knot_seconds=0.15):
    """Cubic pose prior; rotations are fitted in local SO(3) tangent charts.

    The chart is centred at the span's chordal mean and sequentially unwrapped, so a
    rotation passing through pi stays continuous. A span whose frame-to-frame rotation is
    too large to unwrap (> 90 deg) keeps its rotations, and this is logged.
    """
    poses = np.asarray(poses)
    out = poses.copy()
    for idx, P in spline_projectors(frames, fps, phases, knot_seconds):
        out[idx, :3, 3] = P @ poses[idx, :3, 3]
        anchor = chordal_mean(poses[idx, :3, :3])
        rv = unwrap_rotvecs(Rotation.from_matrix(anchor.T @ poses[idx, :3, :3]).as_rotvec())
        if np.max(np.linalg.norm(np.diff(rv, axis=0), axis=1), initial=0) < np.pi / 2:
            out[idx, :3, :3] = anchor @ Rotation.from_rotvec(P @ rv).as_matrix()
        else:
            logging.getLogger(__name__).warning(
                "spline_pose_prior: frames %d-%d rotate > 90 deg per frame; rotation smoothing skipped",
                int(np.asarray(frames)[idx[0]]), int(np.asarray(frames)[idx[-1]]))
    return out


def sticking_windows(frames, points, distances, contact_m=0.015, slip_m=0.01, min_frames=3):
    """Only anchor continuously observed, near-surface patches with little slip.

    Split long drifting contacts rather than pinning a sliding finger for a whole grasp.
    Returned lists index the input arrays.
    """
    frames, points, distances = np.asarray(frames), np.asarray(points), np.asarray(distances)
    windows, cur = [], []
    for i in range(len(frames)):
        near = np.isfinite(points[i]).all() and np.isfinite(distances[i]) and abs(distances[i]) < contact_m
        split = cur and (frames[i] != frames[cur[-1]] + 1 or
                         np.max(np.linalg.norm(points[cur + [i]] - np.median(points[cur + [i]], axis=0), axis=1)) > slip_m)
        if not near or split:
            if len(cur) >= min_frames:
                windows.append(cur)
            cur = []
        if near:
            cur.append(i)
    if len(cur) >= min_frames:
        windows.append(cur)
    return windows


def acceleration_loss(points, frames, fps, max_gap_seconds=0.15):
    """Differentiable robust world acceleration, accounting for actual frame spacing.

    Normalized to a 20 Hz second difference to retain existing loss-weight units.
    Large accelerations at intentional transitions get a robust, not quadratic, cost.
    """
    import torch
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if len(points) < 3:
        return points.sum() * 0
    dt = torch.as_tensor(np.diff(frames) / fps, dtype=points.dtype, device=points.device)
    if (dt <= 0).any():
        raise ValueError("Frame indices must increase")
    shape = (-1,) + (1,) * (points.ndim - 1)
    vel = (points[1:] - points[:-1]) / dt.reshape(shape)
    acc = 2 * (vel[1:] - vel[:-1]) / (dt[1:] + dt[:-1]).reshape(shape)
    keep = (dt[1:] <= max_gap_seconds) & (dt[:-1] <= max_gap_seconds)
    if not keep.any():
        return points.sum() * 0
    sq = (acc[keep] ** 2).sum(-1)
    delta = 5.0  # m/s^2; robust transition scale, not an actuator limit
    return (2 * delta ** 2 * (torch.sqrt(1 + sq / delta ** 2) - 1)).mean() / 20.0 ** 4


def apply_symmetry(pose, symmetry, center):
    """Compose a symmetry about the mesh centroid, preserving its world position."""
    out = np.array(pose, float, copy=True)
    out[:3, :3] = pose[:3, :3] @ symmetry
    out[:3, 3] += pose[:3, :3] @ (np.asarray(center) - symmetry @ center)
    return out


def table_slab_depth(points, plane, soft=0.01, side="nearest"):
    """Differentiable tabletop penetration depth (torch, metres) of points (..., 3).

    plane {"up", "height"} (+ "center", "radius", "thickness" from table_plane.py): the top is a
    solid disk slab; a point is penetrating only when its horizontal distance to the centre is
    within the radius (linear soft margin +-soft) AND its height h = (p - center) . up lies in
    (-thickness, 0); depth = min(-h, h + thickness), the distance to the nearer slab face, so a
    hand under the table is not pushed up through it. side="auto" (points (..., V, 3), one body per
    leading index, e.g. a hand's vertices per frame): the body's side is its mean height (above if
    > -thickness/2), and its vertices are pushed out through THAT face only (depth -h above, h +
    thickness below), so a fingertip pressed deeper than half the slab is not pushed through the
    top. Without the disk keys: the legacy infinite plane relu(height - p . up)."""
    import torch
    up = torch.as_tensor(plane["up"], dtype=points.dtype, device=points.device)
    if "center" not in plane or "radius" not in plane:
        return torch.relu(float(plane["height"]) - points @ up)
    d = points - torch.as_tensor(plane["center"], dtype=points.dtype, device=points.device)
    h = d @ up
    rad = (d - h[..., None] * up).norm(dim=-1)
    inside = ((float(plane["radius"]) + soft - rad) / (2 * soft)).clamp(0, 1)
    th = float(plane.get("thickness", 0.03))
    in_slab = (h < 0) & (h > -th)
    if side == "auto":
        above = (h.mean(-1, keepdim=True) > -th / 2).expand_as(h)
        return torch.where(above, torch.relu(-h), torch.relu(h + th) * in_slab) * inside
    return torch.relu(torch.minimum(-h, h + th)) * inside


def table_disk_inside(points, plane, margin=0.0):
    """Horizontal inside-the-disk test (numpy or torch, (..., 3)); all True without disk keys."""
    if "center" not in plane or "radius" not in plane:
        return points[..., 0] == points[..., 0]
    up = np.asarray(plane["up"], float)
    c = np.asarray(plane["center"], float)
    if not isinstance(points, np.ndarray):
        import torch
        up, c = (torch.as_tensor(x, dtype=points.dtype, device=points.device) for x in (up, c))
    d = points - c
    h = d @ up
    return (d - h[..., None] * up).norm(dim=-1) <= plane["radius"] + margin if not isinstance(points, np.ndarray) \
        else np.linalg.norm(d - h[..., None] * up, axis=-1) <= plane["radius"] + margin
