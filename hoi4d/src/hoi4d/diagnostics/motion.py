"""README stage 7, motion checks: discontinuities and derivatives with real timestamps, rotation-space
(geodesic) distances, symmetry-aware (a visually ambiguous symmetric-object rotation is not a jump).
All functions are pure numpy on hoi4d.io.Trajectories arrays; NaN (missing) frames break windows."""
from __future__ import annotations

import numpy as np


def geodesic(Ra: np.ndarray, Rb: np.ndarray, sym: list[np.ndarray] | None = None) -> np.ndarray:
    """Angle (rad) between rotations, minimised over the object's symmetry group (applied in the object frame)."""
    syms = sym if sym else [np.eye(3)]
    best = np.full(len(Ra), np.inf)
    for S in syms:
        M = np.einsum("fji,fjk->fik", Ra, Rb @ S)  # Ra^T Rb S
        c = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
        best = np.minimum(best, np.arccos(c))
    return best


def finite_diff(x: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
    """order-th time derivative (central where possible) with real timestamps; NaN-propagating."""
    y = x.astype(float)
    tt = t.astype(float)
    for _ in range(order):
        dt = np.gradient(tt)
        y = np.gradient(y, axis=0) / dt.reshape(-1, *([1] * (y.ndim - 1)))
    return y


def runs(mask: np.ndarray, min_len: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Inclusive [a, b] index intervals where mask is True (gaps <= merge_gap merged)."""
    out = []
    i, F = 0, len(mask)
    while i < F:
        if mask[i]:
            j = i
            while j + 1 < F and mask[j + 1]:
                j += 1
            if out and i - out[-1][1] - 1 <= merge_gap:
                out[-1] = (out[-1][0], j)
            else:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return [(a, b) for a, b in out if b - a + 1 >= min_len]


def object_motion(T: np.ndarray, valid: np.ndarray, t_s: np.ndarray, sym=None, sigma_t=None) -> dict:
    """Per-frame step sizes, velocities, accelerations, jerks and their prediction residual.

    jump_resid_m: |p_t - (2 p_{t-1} - p_{t-2})| (constant-velocity prediction error): a real fast motion has a
    small residual, a tracking jump a large one -- velocity alone would flag legitimate fast transport."""
    F = len(t_s)
    p = T[:, :3, 3]
    R = T[:, :3, :3]
    step_t = np.full(F, np.nan)
    step_r = np.full(F, np.nan)
    ok = valid[1:] & valid[:-1]
    step_t[1:][ok] = np.linalg.norm(p[1:][ok] - p[:-1][ok], axis=1)
    step_r[1:][ok] = geodesic(R[1:][ok], R[:-1][ok], sym)
    pred_res = np.full(F, np.nan)
    ok2 = valid[2:] & valid[1:-1] & valid[:-2]
    pred = 2 * p[1:-1] - p[:-2]
    pred_res[2:][ok2] = np.linalg.norm(p[2:][ok2] - pred[ok2], axis=1)
    rot_pred_res = np.full(F, np.nan)
    if ok2.any():
        # constant angular velocity prediction: R_pred = R_{t-1} (R_{t-2}^T R_{t-1})
        Rd = np.einsum("fji,fjk->fik", R[:-2], R[1:-1])
        Rp = np.einsum("fij,fjk->fik", R[1:-1], Rd)
        rot_pred_res[2:][ok2] = geodesic(R[2:][ok2], Rp[ok2], sym)
    v = finite_diff(p, t_s, 1)
    a = finite_diff(p, t_s, 2)
    j = finite_diff(p, t_s, 3)
    return {"step_t_m": step_t, "step_r_rad": step_r, "jump_resid_m": pred_res, "rot_jump_resid_rad": rot_pred_res,
            "speed_mps": np.linalg.norm(v, axis=1), "acc_mps2": np.linalg.norm(a, axis=1),
            "jerk_mps3": np.linalg.norm(j, axis=1)}


def motion_anomalies(name: str, m: dict, t_s: np.ndarray, frame: np.ndarray, th: dict, sigma_t=None) -> list[dict]:
    """Candidates only; thresholds widen with the per-frame uncertainty when it is known."""
    out = []
    unc = np.nan_to_num(np.asarray(sigma_t, float), nan=0.0) if sigma_t is not None else 0.0
    checks = [("translation_jump", m["jump_resid_m"], th["jump_trans_m"] + 3 * unc, "m"),
              ("rotation_jump", m["rot_jump_resid_rad"], np.deg2rad(th["jump_rot_deg"]), "rad"),
              ("high_acceleration", m["acc_mps2"], th["acc_mps2"], "m/s^2"),
              ("high_jerk", m["jerk_mps3"], th["jerk_mps3"], "m/s^3")]
    for typ, x, thr, unit in checks:
        bad = np.nan_to_num(x) > thr
        for a, b in runs(bad, merge_gap=2):
            seg = x[a:b + 1]
            out.append({"type": typ, "entities": [name], "t_start_s": float(t_s[a]), "t_end_s": float(t_s[b]),
                        "frames": [int(frame[a]), int(frame[b])],
                        "measurements": {"max": float(np.nanmax(seg)), "unit": unit,
                                         "frames_over": int(b - a + 1)},
                        "thresholds": {"value": float(np.max(np.atleast_1d(thr)[a:b + 1] if np.ndim(thr) else thr)),
                                       "unit": unit},
                        "severity": float(min(1.0, np.nanmax(seg) / (np.max(np.atleast_1d(thr)) * 3)))})
    return out


HAND_FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]


def hand_anatomy(joints: np.ndarray, valid: np.ndarray, hand_pose: np.ndarray) -> dict:
    """Anatomy checks on MANO output (bone lengths are rigid by construction except the skinned fingertip
    vertices, so they are not tested): per-joint rotation magnitude (bound 2.0 rad) and finger interpenetration
    (twisted fingers): min distance between segments of different fingers, distal halves only (MCPs sit close)."""
    F = len(joints)
    ang = np.linalg.norm(hand_pose.reshape(F, 15, 3), axis=-1)
    seg = []
    for fj in HAND_FINGERS:
        pts = [joints[:, fj[k]] * (1 - s) + joints[:, fj[k + 1]] * s for k in (1, 2) for s in (0, 0.5)] + [joints[:, fj[3]]]
        seg.append(np.stack(pts, 1))  # (F, 5, 3) PIP..tip
    dmin = np.full(F, np.inf)
    for a in range(5):
        for b in range(a + 1, 5):
            d = np.linalg.norm(seg[a][:, :, None] - seg[b][:, None], axis=-1).min(axis=(1, 2))
            dmin = np.minimum(dmin, d)
    dmin[~valid] = np.nan
    return {"joint_angle_max_rad": np.nanmax(ang, axis=1), "finger_min_dist_m": dmin}
