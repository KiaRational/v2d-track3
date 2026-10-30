"""Robust SE(3) trajectory fusion of several noisy per-frame pose measurements.

Poses are fused in a common (world) frame. Unknowns per frame t: a rotation
correction delta_t (so R_t = exp(delta_t) R0_t, with R0_t the initial guess) and a
translation p_t. Cost (all residuals normalised by their sigma, soft-L1 robust loss,
sparse Jacobian, scipy.optimize.least_squares):

  unary       [ sqrt(w_t) (p_t - p_m) / sigma_t ,  sqrt(w_r) rot_err(R_t, R_m) / sigma_r ]
              a measurement's weight is a scalar w, or a pair (w_t, w_r) when translation
              and rotation deserve different trust -- under occlusion a tracker's rotation
              degrades long before its position does
              for every measurement m = (t, T, w) of every source
  hand prior  the same form for hand-predicted poses (hand_pred[t] = {"T", "w"})
  smoothness  second differences: (p_{t+1} - 2 p_t + p_{t-1}) / sigma_acc and
              (log(R_t^T R_{t+1}) - log(R_{t-1}^T R_t)) / sigma_racc
  static      sqrt(s_t) * [ (p_{t+1} - p_t) / sigma_vel , log(R_t^T R_{t+1}) / sigma_rvel ]
              for transitions with s_t > 0 (params["static_w"], length n-1): an object
              that no hand touches must not move in the world
  follow      sqrt(w) * [ (p_{t+1} - (D p_t + d)) / sigma_ft , log((D_R R_t)^T R_{t+1}) / sigma_fr ]
              for (t, D, w) in params["follow"]: D = H_{t+1} H_t^-1 is the world motion of the
              hand holding the object, so the object stays rigidly fixed in the hand frame
  grasp       while a hand holds the object, the object is rigid in the hand's frame: one
              unknown grasp transform G per grasp segment (6 dof, appended to the state),
              residuals sqrt(w) * [ (p_t - (H_t R_G t_G + H_t p_H)) / sigma_gt ,
              log((H_t R_G)^T R_t) / sigma_gr ] for every frame of the segment. Unlike a
              frame-to-frame follow term this is an absolute anchor: G is fixed by the
              frames where the object is clearly seen, and it then carries the pose through
              the frames where a hand hides it, instead of letting it drift.
              params["grasp"] = [{"frames": [...], "H": {t: 4x4 world<-palm}, "w": float}, ...]
  spline      (params["spline_weight"] w > 0) smoothness of the CURRENT estimate, not a pull
              toward the initial guess: per span of params["contact_phases"] (split at phase
              changes; spans shorter than the basis get no term), a clamped cubic B-spline with
              its own control points (appended to the state): sqrt(w) (p_t - B_t c) / sigma_t,
              sqrt(w) log(S_t^T R_t) / sigma_r with S_t the cumulative SO(3) B-spline of control
              rotations (local, no log-chart branch cut), plus a small curvature ridge
              sqrt(w ridge) D c on the control points. Minimising over c this is ||(I - P) x||^2
              with P the span's spline smoother, but it stays sparse (frame t + 4 controls).
  anchor      tiny pull of delta_t to 0 (gauge / unobserved symmetric rotation)

rot_err is the full rotation log when params["sym_axis"] is None, otherwise the
chordal difference of the symmetry axis (R_t a - R_m a), which ignores the unobservable
spin of rotationally symmetric objects (cups, bowls).

fuse_trajectory(measurements, n_frames, hand_pred=None, params=None) -> (poses, info)
  measurements: {source: [(t, 4x4, weight), ...]}
  poses: (n_frames, 4, 4) fused; info: per-measurement normalised residuals and
  outlier flags (> params["outlier_sigma"]), the solver status, etc.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp
from motion_quality import SPLINE_RIDGE, spline_bases

DEFAULTS = dict(sigma_t=0.01, sigma_r=np.deg2rad(5), sigma_acc=0.004, sigma_racc=np.deg2rad(2.0),
                sigma_anchor=np.deg2rad(60), loss="soft_l1", f_scale=1.0, sym_axis=None,
                outlier_sigma=3.0, max_nfev=200, static_w=None, sigma_vel=0.001,
                sigma_rvel=np.deg2rad(0.3), follow=None, sigma_ft=0.004, sigma_fr=np.deg2rad(3.0),
                grasp=None, sigma_gt=0.006, sigma_gr=np.deg2rad(4.0),
                spline_weight=0.0, spline_knot_seconds=0.15, spline_ridge=SPLINE_RIDGE, contact_phases=None,
                sigma_spline_t=None, sigma_spline_r=None)  # spline sigmas default to sigma_t / sigma_r


def _log(R):
    return Rotation.from_matrix(R).as_rotvec()


def _init(meas, n, rel_floor=0.25):
    """Per frame: weighted median translation and chordal-mean rotation; gaps
    interpolated (SLERP / linear), ends held constant.

    Measurements weighted far below the sequence's best are ignored here even when they
    are the only ones at their frame: starting a frame from a measurement we hardly trust
    (a tracker under an occluding hand) can strand the robust loss in a local minimum
    around that value. Such frames are interpolated from trusted ones instead."""
    by_t: dict[int, list] = {}
    raw = []
    for src in meas.values():
        for t, T, w in src:
            wt, wr = (w, w) if np.isscalar(w) else (w[0], w[1])
            if max(wt, wr) > 0 and 0 <= t < n:
                raw.append((t, np.asarray(T, float), wt, wr))
    if raw:
        floor = rel_floor * max(max(wt, wr) for _, _, wt, wr in raw)
        for t, T, wt, wr in raw:
            if max(wt, wr) >= floor:
                by_t.setdefault(t, []).append((T, wt, wr))
    if not by_t:
        raise ValueError("no positive-weight measurement")
    ts = np.array(sorted(by_t))
    P, Rs, tr_ = [], [], []
    for t in ts:
        Ts, ws, wr = (np.array(x) if i else x for i, x in enumerate(zip(*by_t[t])))
        p = np.array([T[:3, 3] for T in Ts])
        order = np.argsort(p, axis=0)
        med = []
        for k in range(3):
            c = np.cumsum(ws[order[:, k]])
            med.append(p[order[np.searchsorted(c, c[-1] / 2), k], k])
        P.append(med)
        if np.sum(wr) <= 0:   # rotation gated at this frame (translation only): interpolate its rotation
            continue
        M = sum(w * T[:3, :3] for T, w in zip(Ts, wr))
        U, _, Vt = np.linalg.svd(M)
        Rs.append(U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt); tr_.append(t)
    P = np.array(P)
    frames = np.arange(n)
    tc = np.clip(frames, ts[0], ts[-1])
    p0 = np.stack([np.interp(tc, ts, P[:, k]) for k in range(3)], 1)
    tr_ = np.array(tr_) if tr_ else ts[:1]
    if not Rs:
        Rs = [np.eye(3)]
    tcr = np.clip(frames, tr_[0], tr_[-1])
    R0 = Slerp(tr_, Rotation.from_matrix(np.array(Rs)))(tcr).as_matrix() if len(tr_) > 1 else np.repeat(np.array(Rs)[:1], n, 0)
    return p0, R0


_FLIPS = [np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])]


def _ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def _grasp_rotation(R_trust, fr, H, R0, n):
    """Object-in-hand rotation for one grasp segment, continuity-aware. The trusted-frame mean alone can sit on the
    wrong side of a FoundationPose flip; the object then flips by ~160 deg exactly when it is grabbed or released (e30
    sweeper, e33 spoon). Candidates: that mean, the rotations that keep the object where it was just before the grasp
    and just after it, and each of these with a 180 deg flip about an object axis. Score: median per-frame
    disagreement with the measurements (capped at 90 deg) + the rotation jumps the grasp leaves at its two ends."""
    base = [R_trust]
    if fr[0] > 0:
        base.append(H[0][:3, :3].T @ R0[fr[0] - 1])
    if fr[-1] < n - 1:
        base.append(H[-1][:3, :3].T @ R0[fr[-1] + 1])
    cands = base + [C @ F for C in base for F in _FLIPS]

    def score(Rg):
        Ro = H[:, :3, :3] @ Rg
        data = np.median([min(_ang(Ro[i], R0[t]), 90.0) for i, t in enumerate(fr)])
        jump = (_ang(Ro[0], R0[fr[0] - 1]) if fr[0] > 0 else 0.0) + (_ang(Ro[-1], R0[fr[-1] + 1]) if fr[-1] < n - 1 else 0.0)
        return data + jump
    return min(cands, key=score)


def fuse_trajectory(measurements: Mapping[str, Sequence[tuple]], n_frames: int,
                    hand_pred: Sequence[Mapping[str, Any] | None] | None = None,
                    params: Mapping[str, Any] | None = None):
    prm = {**DEFAULTS, **(params or {})}
    fps = float(prm.get("fps", 20.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    # Legacy sigmas are per-frame values at 20 Hz. Preserve physical smoothing
    # strength when frame rate changes instead of penalizing identical motion differently.
    for key in ("sigma_acc", "sigma_racc"):
        prm[key] *= (20.0 / fps) ** 2
    for key in ("sigma_vel", "sigma_rvel"):
        prm[key] *= 20.0 / fps
    n = n_frames
    def _w(w):
        return (float(w), float(w)) if np.isscalar(w) else (float(w[0]), float(w[1]))

    meas = {k: [(int(t), np.asarray(T, float), _w(w)) for t, T, w in v] for k, v in measurements.items()}
    if hand_pred is not None:
        meas["_hand"] = [(t, np.asarray(h["T"], float), _w(h["w"])) for t, h in enumerate(hand_pred)
                         if h is not None and np.max(h["w"]) > 0]
    p0, R0 = _init({k: v for k, v in meas.items() if k != "_hand"} or meas, n)
    axis = None if prm["sym_axis"] is None else np.asarray(prm["sym_axis"], float) / np.linalg.norm(prm["sym_axis"])
    flat = [(src, t, T, w) for src, v in meas.items() for t, T, w in v if max(w) > 0 and 0 <= t < n]
    n_un = len(flat)
    sw_static = np.zeros(max(n - 1, 0)) if prm["static_w"] is None else np.asarray(prm["static_w"], float)
    st_idx = np.nonzero(sw_static > 0)[0]
    follow = [(int(t), np.asarray(D, float), float(w)) for t, D, w in (prm["follow"] or []) if w > 0 and 0 <= t < n - 1]
    grasp = []
    for g in (prm["grasp"] or []):
        fr = [int(t) for t in g["frames"] if 0 <= int(t) < n and int(t) in g["H"]]
        if len(fr) >= 2 and float(g.get("w", 1.0)) > 0:
            grasp.append({"frames": fr, "H": np.array([np.asarray(g["H"][t], float) for t in fr]),
                          "w": float(g.get("w", 1.0)),
                          "trust": np.array([float((g.get("trust") or {}).get(t, 0.0)) for t in fr])})
    n_g = sum(len(g["frames"]) for g in grasp)
    # spline spans: control points [c_t (m,3), rotation deltas on C0 (m,3)] appended after the grasps
    spans, off = [], 6 * n + 6 * len(grasp)
    w_sp, lam = float(prm["spline_weight"] or 0), float(prm["spline_ridge"])
    sg_t = prm["sigma_spline_t"] or prm["sigma_t"]
    sg_r = prm["sigma_spline_r"] or prm["sigma_r"]
    if w_sp > 0:
        for idx, B, D in spline_bases(np.arange(n), fps, prm["contact_phases"], prm["spline_knot_seconds"]):
            m = B.shape[1]
            Bt = np.cumsum(B[:, ::-1], 1)[:, ::-1]           # cumulative basis: S_t = C_b prod exp(Bt w_j)
            base = np.argmax(B > 1e-12, axis=1)               # first active control of each frame
            jj = base[:, None] + np.arange(1, 4)
            cf = np.where(jj < m, Bt[np.arange(len(idx))[:, None], np.minimum(jj, m - 1)], 0.0)
            c_t0 = np.linalg.solve(B.T @ B + lam * D.T @ D, B.T @ p0[idx])
            spans.append(dict(idx=idx, B=B, D=D, m=m, off=off, base=base, jj=np.minimum(jj, m - 1), cf=cf,
                              C0=R0[idx[np.argmax(B, axis=0)]], c_t0=c_t0))
            off += 6 * m
    n_sp = sum(6 * len(s["idx"]) + 6 * len(s["D"]) for s in spans)
    n_res = 6 * n_un + (6 * (n - 2) if n > 2 else 0) + 6 * len(st_idx) + 6 * len(follow) + 6 * n_g + n_sp + 3 * n
    n_var = off
    # Batch the repeated measurement/follow residuals. Their ordering is unchanged
    # (translation then rotation per frame), preserving the sparse Jacobian layout.
    mt = np.array([r[1] for r in flat], dtype=int)
    mT = np.array([r[2] for r in flat])
    mw = np.sqrt(np.array([r[3] for r in flat]))
    ft = np.array([r[0] for r in follow], dtype=int)
    fT = np.array([r[1] for r in follow])
    fw = np.sqrt(np.array([r[2] for r in follow]))[:, None]

    def unpack(x):
        xf = x[:6 * n].reshape(n, 6)
        R = Rotation.from_rotvec(xf[:, :3]).as_matrix() @ R0
        return R, xf[:, 3:]

    def unpack_grasp(x, k):
        """Grasp transform k: rotation delta on its initial guess, plus translation."""
        xg = x[6 * n + 6 * k: 6 * n + 6 * k + 6]
        return Rotation.from_rotvec(xg[:3]).as_matrix() @ grasp[k]["R0"], xg[3:] + grasp[k]["t0"]

    def residuals(x):
        R, p = unpack(x)
        out = np.empty(n_res)
        i = 0
        if n_un:
            trans = mw[:, :1] * (p[mt] - mT[:, :3, 3]) / prm["sigma_t"]
            if axis is None:
                rot = _log(np.swapaxes(mT[:, :3, :3], 1, 2) @ R[mt])
            else:
                rot = R[mt] @ axis - mT[:, :3, :3] @ axis
            out[:6*n_un] = np.c_[trans, mw[:, 1:] * rot / prm["sigma_r"]].ravel()
            i = 6*n_un
        if n > 2:
            acc = (p[2:] - 2 * p[1:-1] + p[:-2]) / prm["sigma_acc"]
            om = Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:])).as_rotvec()
            racc = (om[1:] - om[:-1]) / prm["sigma_racc"]
            out[i:i + 6 * (n - 2)] = np.c_[acc, racc].ravel()
            i += 6 * (n - 2)
        if len(st_idx):
            ss = np.sqrt(sw_static[st_idx])[:, None]
            vel = (p[st_idx + 1] - p[st_idx]) / prm["sigma_vel"]
            rv = Rotation.from_matrix(np.einsum("nji,njk->nik", R[st_idx], R[st_idx + 1])).as_rotvec() / prm["sigma_rvel"]
            out[i:i + 6 * len(st_idx)] = (ss * np.c_[vel, rv]).ravel()
            i += 6 * len(st_idx)
        for k, g in enumerate(grasp):
            Rg, tg = unpack_grasp(x, k)
            sw = np.sqrt(g["w"])
            H, fr = g["H"], g["frames"]
            pred_p = np.einsum("fij,j->fi", H[:, :3, :3], tg) + H[:, :3, 3]
            pred_R = H[:, :3, :3] @ Rg
            m = len(fr)
            out[i:i + 3 * m].reshape(m, 3)[:] = sw * (p[fr] - pred_p) / prm["sigma_gt"]
            out[i + 3 * m:i + 6 * m].reshape(m, 3)[:] = sw * Rotation.from_matrix(
                np.einsum("fji,fjk->fik", pred_R, R[fr])).as_rotvec() / prm["sigma_gr"]
            i += 6 * m
        if len(follow):
            pred_p = np.einsum("nij,nj->ni", fT[:, :3, :3], p[ft]) + fT[:, :3, 3]
            pred_R = fT[:, :3, :3] @ R[ft]
            trans = fw * (p[ft+1] - pred_p) / prm["sigma_ft"]
            rot = fw * _log(np.swapaxes(pred_R, 1, 2) @ R[ft+1]) / prm["sigma_fr"]
            out[i:i+6*len(follow)] = np.c_[trans, rot].ravel()
            i += 6*len(follow)
        for s in spans:   # spline smoothness of the current estimate (see module docstring)
            idx, m, o = s["idx"], s["m"], s["off"]
            L, k = len(idx), len(s["D"])
            ct = x[o:o + 3 * m].reshape(m, 3)
            C = Rotation.from_rotvec(x[o + 3 * m:o + 6 * m].reshape(m, 3)).as_matrix() @ s["C0"]
            rel = np.zeros((m, 3))
            rel[1:] = Rotation.from_matrix(np.einsum("mji,mjk->mik", C[:-1], C[1:])).as_rotvec()
            S_ = C[s["base"]]
            for q in range(3):
                S_ = S_ @ Rotation.from_rotvec(s["cf"][:, q, None] * rel[s["jj"][:, q]]).as_matrix()
            sw = np.sqrt(w_sp)
            out[i:i + 3 * L] = (sw * (p[idx] - s["B"] @ ct) / sg_t).ravel()
            out[i + 3 * L:i + 6 * L] = (sw * Rotation.from_matrix(
                np.einsum("lji,ljk->lik", S_, R[idx])).as_rotvec() / sg_r).ravel()
            i += 6 * L
            sr = np.sqrt(w_sp * lam)
            out[i:i + 3 * k] = (sr * (s["D"] @ ct) / sg_t).ravel()
            out[i + 3 * k:i + 6 * k] = (sr * (s["D"] @ np.cumsum(rel, 0)) / sg_r).ravel()
            i += 6 * k
        out[i:i + 3 * n] = x[:6 * n].reshape(n, 6)[:, :3].ravel() / prm["sigma_anchor"]
        return out

    S = lil_matrix((n_res, n_var), dtype=np.int8)
    i = 0
    for _, t, _, _ in flat:
        S[i:i + 6, 6 * t:6 * t + 6] = 1
        i += 6
    for t in range(1, n - 1):
        S[i:i + 6, 6 * (t - 1):6 * (t + 2)] = 1
        i += 6
    for t in st_idx:
        S[i:i + 6, 6 * t:6 * t + 12] = 1
        i += 6
    for k, g in enumerate(grasp):
        m = len(g["frames"])
        for r, t in enumerate(g["frames"]):
            S[i + 3 * r:i + 3 * r + 3, 6 * t + 3:6 * t + 6] = 1            # translation rows
            S[i + 3 * m + 3 * r:i + 3 * m + 3 * r + 3, 6 * t:6 * t + 3] = 1  # rotation rows
        S[i:i + 3 * m, 6 * n + 6 * k + 3: 6 * n + 6 * k + 6] = 1
        S[i + 3 * m:i + 6 * m, 6 * n + 6 * k: 6 * n + 6 * k + 3] = 1
        i += 6 * m
    for t, _, _ in follow:
        S[i:i + 6, 6 * t:6 * t + 12] = 1
        i += 6
    for s in spans:
        idx, m, o = s["idx"], s["m"], s["off"]
        L = len(idx)
        for r, t in enumerate(idx):
            b0, b1 = s["base"][r], min(s["base"][r] + 4, m)
            S[i + 3 * r:i + 3 * r + 3, 6 * t + 3:6 * t + 6] = 1            # translation rows
            S[i + 3 * r:i + 3 * r + 3, o + 3 * b0:o + 3 * b1] = 1
            S[i + 3 * L + 3 * r:i + 3 * L + 3 * r + 3, 6 * t:6 * t + 3] = 1  # rotation rows
            S[i + 3 * L + 3 * r:i + 3 * L + 3 * r + 3, o + 3 * m + 3 * b0:o + 3 * m + 3 * b1] = 1
        i += 6 * L
        k = len(s["D"])
        for j in range(k):   # ridge row j couples controls j..j+2
            S[i + 3 * j:i + 3 * j + 3, o + 3 * j:o + 3 * j + 9] = 1
            S[i + 3 * k + 3 * j:i + 3 * k + 3 * j + 3, o + 3 * m + 3 * j:o + 3 * m + 3 * j + 9] = 1
        i += 6 * k
    for t in range(n):
        S[i:i + 3, 6 * t:6 * t + 3] = 1
        i += 3

    for g in grasp:   # initialise G from the segment's most trusted frames (median of H^-1 O)
        fr, H = g["frames"], g["H"]
        Gs = np.linalg.inv(H) @ np.stack([np.r_[np.c_[R0[t], p0[t]], [[0, 0, 0, 1.0]]] for t in fr])
        tr = g["trust"] >= 0.5   # clearly seen frames (caller's trust: visibility x stereo-ICP); else all frames
        if tr.sum() >= 3:
            Gs = Gs[tr]
        g["t0"] = np.median(Gs[:, :3, 3], axis=0)
        U, _, Vt = np.linalg.svd(Gs[:, :3, :3].sum(0))
        g["R0"] = _grasp_rotation(U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt, fr, H, R0, n)
    x0 = np.concatenate([np.c_[np.zeros((n, 3)), p0].ravel(), np.zeros(6 * len(grasp))] +
                        [np.r_[s["c_t0"].ravel(), np.zeros(3 * s["m"])] for s in spans])
    sol = least_squares(residuals, x0, jac_sparsity=S.tocsr(), loss=prm["loss"], f_scale=prm["f_scale"],
                        method="trf", x_scale="jac", max_nfev=prm["max_nfev"])
    R, p = unpack(sol.x)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = R, p

    r = residuals(sol.x)[:6 * n_un].reshape(-1, 6)
    per = []
    for k, (src, t, T, w) in enumerate(flat):
        e_t = float(np.linalg.norm(p[t] - T[:3, 3]))
        e_r = float(np.linalg.norm(_log(T[:3, :3].T @ R[t]))) if axis is None else \
            float(np.arccos(np.clip((R[t] @ axis) @ (T[:3, :3] @ axis), -1, 1)))
        z = float(np.linalg.norm(r[k] / np.sqrt(np.r_[np.full(3, max(w[0], 1e-9)),
                                                            np.full(3, max(w[1], 1e-9))]))) / np.sqrt(2)
        per.append({"source": src, "frame": t, "weight": list(w), "trans_err_m": e_t, "rot_err_rad": e_r,
                    "z": z, "outlier": bool(z > prm["outlier_sigma"])})
    stats = {}
    for src in meas:
        rows = [q for q in per if q["source"] == src]
        if rows:
            stats[src] = {"n": len(rows), "outlier_frac": float(np.mean([q["outlier"] for q in rows])),
                          "median_trans_err_mm": float(np.median([q["trans_err_m"] for q in rows]) * 1000),
                          "median_rot_err_deg": float(np.rad2deg(np.median([q["rot_err_rad"] for q in rows])))}
    return poses, {"status": int(sol.status), "cost": float(sol.cost), "nfev": int(sol.nfev),
                   "per_measurement": per, "sources": stats, "spline_spans": len(spans)}


def _self_test():
    rng = np.random.default_rng(0)
    n = 120
    tt = np.arange(n) / 20.0
    gt = np.tile(np.eye(4), (n, 1, 1))
    gt[:, :3, 3] = np.c_[0.1 * np.sin(tt), 0.05 * tt, 0.5 + 0.02 * np.cos(2 * tt)]
    gt[:, :3, :3] = Rotation.from_rotvec(np.c_[0.3 * np.sin(tt), 0.2 * tt, np.zeros(n)]).as_matrix()

    def noisy(T, st, sr):
        M = T.copy()
        M[:3, 3] += rng.normal(0, st, 3)
        M[:3, :3] = Rotation.from_rotvec(rng.normal(0, sr, 3)).as_matrix() @ M[:3, :3]
        return M

    fp = [(t, noisy(gt[t], 0.006, np.deg2rad(3)), 1.0) for t in range(n)]
    for t in rng.choice(n, 12, replace=False):  # gross FP failures (occlusion)
        fp[t] = (t, noisy(gt[t], 0.08, np.deg2rad(60)), 1.0)
    icp = [(t, noisy(gt[t], 0.002, np.deg2rad(1.5)), 3.0) for t in range(n) if not 40 <= t < 70]  # gap
    poses, info = fuse_trajectory({"fp": fp, "icp": icp}, n)

    def err(P):
        et = np.linalg.norm(P[:, :3, 3] - gt[:, :3, 3], axis=1) * 1000
        er = np.rad2deg(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", P[:, :3, :3], gt[:, :3, :3]))
                                       .as_rotvec(), axis=1))
        return et.mean(), er.mean()

    raw = np.array([T for _, T, _ in fp])
    print("FP raw   : %.1f mm  %.2f deg" % err(raw))
    print("fused    : %.1f mm  %.2f deg" % err(poses))
    flagged = {q["frame"] for q in info["per_measurement"] if q["source"] == "fp" and q["outlier"]}
    print("fp outliers flagged:", len(flagged), "status", info["status"], info["sources"])
    assert err(poses)[0] < 3.0 and err(poses)[1] < 2.0
    axis_test, _ = fuse_trajectory({"fp": fp, "icp": icp}, n, params={"sym_axis": [0, 0, 1]})
    a = np.einsum("nij,j->ni", axis_test[:, :3, :3], [0, 0, 1])
    g = np.einsum("nij,j->ni", gt[:, :3, :3], [0, 0, 1])
    print("sym-axis mode: axis err %.2f deg" % np.rad2deg(np.arccos(np.clip((a * g).sum(1), -1, 1))).mean())
    still = np.tile(gt[0], (n, 1, 1))
    fp_s = [(t, noisy(still[t], 0.006, np.deg2rad(3)), 1.0) for t in range(n)]
    for t in range(30, 70):  # long occlusion: FP consistently wrong
        M = still[t].copy()
        M[0, 3] += 0.04
        fp_s[t] = (t, noisy(M, 0.006, np.deg2rad(3)), 1.0)
    P_free, _ = fuse_trajectory({"fp": fp_s}, n)
    P_stat, _ = fuse_trajectory({"fp": fp_s}, n, params={"static_w": np.ones(n - 1)})
    e = lambda P: np.linalg.norm(P[30:70, :3, 3] - still[30:70, :3, 3], axis=1).mean() * 1000
    print("static object, 40-frame FP failure: no static term %.1f mm, static term %.1f mm" % (e(P_free), e(P_stat)))
    assert e(P_stat) < e(P_free)
    # hand-held object, FP lost for 40 frames: the hand's motion carries it through
    offset = np.eye(4)
    offset[:3, 3] = [0.0, 0.05, 0.02]
    H = np.array([noisy(gt[t] @ np.linalg.inv(offset), 0.001, np.deg2rad(0.5)) for t in range(n)])
    fp_h = [(t, noisy(gt[t], 0.006, np.deg2rad(3)), 1.0) for t in range(n) if not 40 <= t < 80]
    fol = [(t, H[t + 1] @ np.linalg.inv(H[t]), 1.0) for t in range(n - 1)]
    P_nf, _ = fuse_trajectory({"fp": fp_h}, n)
    P_f, _ = fuse_trajectory({"fp": fp_h}, n, params={"follow": fol})
    e2 = lambda P: np.linalg.norm(P[40:80, :3, 3] - gt[40:80, :3, 3], axis=1).mean() * 1000
    print("held object, 40-frame FP gap: no follow %.1f mm, follow %.1f mm" % (e2(P_nf), e2(P_f)))
    assert e2(P_f) < e2(P_nf)
    # grasped object, hidden by the hand for 40 frames: the grasp transform must carry it
    off = np.eye(4); off[:3, 3] = [0.0, 0.05, 0.02]
    Rg0 = Rotation.from_rotvec([0.2, -0.1, 0.4]).as_matrix()
    off[:3, :3] = Rg0
    Hs = {t: noisy(gt[t] @ np.linalg.inv(off), 0.001, np.deg2rad(0.5)) for t in range(n)}
    fp_g = []
    for t in range(n):
        M = gt[t].copy()
        if 40 <= t < 80:          # hidden: rotation collapses, position degrades
            M[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.deg2rad(45), 3)).as_matrix() @ M[:3, :3]
            M[:3, 3] += rng.normal(0, 0.02, 3)
            fp_g.append((t, M, (0.15, 0.02)))
        else:
            fp_g.append((t, noisy(M, 0.006, np.deg2rad(3)), 1.0))
    seg = [{"frames": list(range(n)), "H": Hs, "w": 1.0}]
    def rerr(P):
        return np.rad2deg(np.linalg.norm(Rotation.from_matrix(
            np.einsum("nji,njk->nik", P[40:80, :3, :3], gt[40:80, :3, :3])).as_rotvec(), axis=1)).mean()
    def terr(P):
        return np.linalg.norm(P[40:80, :3, 3] - gt[40:80, :3, 3], axis=1).mean() * 1000
    P_no, _ = fuse_trajectory({"fp": fp_g}, n)
    P_fol, _ = fuse_trajectory({"fp": fp_g}, n,
                               params={"follow": [(t, Hs[t + 1] @ np.linalg.inv(Hs[t]), 1.0) for t in range(n - 1)]})
    P_gr, info_g = fuse_trajectory({"fp": fp_g}, n, params={"grasp": seg})
    print("grasped + hidden 40 frames:")
    for tag, P in (("no hand term", P_no), ("frame-to-frame follow", P_fol), ("grasp transform", P_gr)):
        print("  %-22s rot %5.1f deg  trans %5.1f mm" % (tag, rerr(P), terr(P)))
    assert rerr(P_gr) < 0.5 * rerr(P_fol) and terr(P_gr) < terr(P_fol)
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
