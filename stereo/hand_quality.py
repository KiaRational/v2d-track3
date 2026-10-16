"""Per-frame / per-joint plausibility of a triangulated or regressed hand (numpy + cv2 only).

Stage 6b hands Stage 6c a target (X_raw) with a per-joint weight (conf) and a per-joint
`valid` flag. A WiLoR hand that is simply wrong -- a mislabelled border detection, a
hallucinated hand on a forearm, a thumb extrapolated outside a truncated view -- still
triangulates to *something*, enters the fit at full weight and drags the MANO sequence
around; the smoothness terms then spread the error over neighbouring frames (jitter).

This module scores each joint of each frame with independent, model-free signals and
returns a multiplicative weight in [0, 1] plus a per-frame verdict. Nothing here needs a
hand prior: every threshold is relative to *that hand's own* statistics over the episode.

    bone      each bone's length vs the median length of that bone over the episode
              (one hand does not change size); the deviation invalidates the child joint.
    angle     finger hyperextension. The palm normal n = (J5-J0) x (J17-J0) gives a signed
              bend s = <(b_parent x b_child), n> at every interior finger joint. A real hand
              curls all fingers the same way, so the sign of s is consistent; the reference
              sign is taken from the hand's own distribution and a joint bending the other
              way by more than --ang_tol degrees is implausible.
    accel     Hampel test on the second difference of each joint's 3D position, scaled by
              that joint's own MAD -- catches single-frame teleports the temporal smoother
              would otherwise turn into two frames of jitter.
    reproj    the triangulation's own per-frame pixel RMS.
    mask      IoU between the hand silhouette rendered from the 3D joints and the SAM 3 hand
              mask of that track (Stage 6d). Low IoU = the pose sits somewhere the hand is
              not, which no purely temporal test can see.

A frame whose combined score falls below --min_frame is rejected outright (valid=False on
every joint), so Stage 6c sees no 3D target there and falls back to its 2D / smoothness
terms instead of fitting to garbage.

Used by stereo/s06b_v3_sam3.py and by the stand-alone stereo/s06e_hand_confidence.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
PALM = [0, 5, 9, 13, 17]

DEFAULTS = dict(bone_abs_mm=5.0, bone_soft=0.18, ang_tol=25.0, ang_tol_thumb=50.0, ang_soft=25.0,
                acc_z=3.0, acc_soft=3.0, acc_floor_mm=3.0, rms_soft=8.0, iou_lo=0.15, iou_hi=0.50,
                min_frame=0.22, min_joint=0.05)


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


# --------------------------------------------------------------------------- signals
def bone_weights(X, ok, p):
    """(n,21) weight from each bone's deviation from its own episode median."""
    n = len(X)
    L = np.full((n, len(BONES)), np.nan)
    for k, (a, b) in enumerate(BONES):
        good = ok[:, a] & ok[:, b]
        d = np.linalg.norm(X[:, a] - X[:, b], axis=1)
        L[:, k] = np.where(good & np.isfinite(d), d, np.nan)
    med = np.nanmedian(L, 0)
    # tolerate an absolute triangulation error as well as a relative one: the distal
    # segments are only ~20 mm long, so a pure percentage test punishes ordinary noise
    dev = np.clip(np.abs(L - med) - p["bone_abs_mm"] / 1000.0, 0, None) / np.maximum(med, 1e-3)
    W = np.ones((n, 21))
    for k, (_, b) in enumerate(BONES):
        w = np.exp(-(np.nan_to_num(dev[:, k], nan=0.0) / p["bone_soft"]) ** 2)
        W[:, b] = np.minimum(W[:, b], np.where(np.isnan(dev[:, k]), 1.0, w))
    # the wrist is the parent of five bones: only blame it if most of them are off
    wrist_bad = np.nanmean(np.stack([np.nan_to_num(dev[:, k], nan=0.0)
                                     for k, (a, _) in enumerate(BONES) if a == 0], 1), 1)
    W[:, 0] = np.minimum(W[:, 0], np.exp(-(wrist_bad / p["bone_soft"]) ** 2))
    return W, med, dev


def bend_signed(X, ok):
    """(n, 15) signed flexion s = sin(bend) about the joint's own flexion axis, the unsigned
    bend angle (deg), and the joint index of each column.

    A finger flexes *towards the palm*: the rotation carrying the parent bone onto the child
    bone is about an axis lying in the palm plane, perpendicular to the bone -- not about the
    palm normal itself. So the axis is built per joint as n x u_parent, and s is the component
    of u_parent x u_child along it. All of a real hand's finger joints flex the same way, so
    the sign of s is constant; a joint with the opposite sign is bending backwards.
    """
    p5, p17 = X[:, 5] - X[:, 0], X[:, 17] - X[:, 0]
    cr = np.cross(p5, p17)
    # a degenerate palm (the two reference vectors nearly parallel, e.g. a badly truncated
    # hand) gives a noise-dominated normal: mark the frame and skip the test there
    span = np.linalg.norm(cr, axis=1) / np.maximum(np.linalg.norm(p5, axis=1) * np.linalg.norm(p17, axis=1), 1e-9)
    nrm = np.where((span > 0.25)[:, None], _unit(cr), np.nan)
    cols, sgn, ang = [], [], []
    for ch in CHAINS:
        for i in range(1, 4):
            p, j, q = ch[i - 1], ch[i], ch[i + 1]
            u1, u2 = _unit(X[:, j] - X[:, p]), _unit(X[:, q] - X[:, j])
            axis = _unit(np.cross(nrm, u1))
            s = (np.cross(u1, u2) * axis).sum(1)
            a = np.degrees(np.arccos(np.clip((u1 * u2).sum(1), -1, 1)))
            good = ok[:, p] & ok[:, j] & ok[:, q]
            cols.append(j)
            sgn.append(np.where(good, s, np.nan))
            ang.append(np.where(good, a, np.nan))
    return np.array(cols), np.stack(sgn, 1), np.stack(ang, 1)


def angle_weights(X, ok, p):
    """(n,21) weight; a finger joint bending against the hand's own curl direction is
    implausible and takes its whole distal sub-chain down with it."""
    cols, S, A = bend_signed(X, ok)
    is_thumb = np.isin(cols, CHAINS[0][1:4])
    ref = np.sign(np.nanmedian(np.where(A[:, ~is_thumb] > 12, S[:, ~is_thumb], np.nan)))
    if not np.isfinite(ref) or ref == 0:
        ref = 1.0
    W = np.ones((len(X), 21))
    tol = np.where(is_thumb, p["ang_tol_thumb"], p["ang_tol"])
    # The bend must be a real flexion, not measurement noise on a nearly straight finger:
    # |s| = sin(bend) * cos(axis, palm normal), so a noise-driven out-of-plane wobble on a
    # straight joint has a small |s| even when the raw angle crosses the tolerance.
    s_min = 0.5 * np.sin(np.radians(tol))
    bad = (np.sign(S) != ref) & (A > tol[None]) & (np.abs(S) > s_min[None]) & np.isfinite(S)
    excess = np.where(bad, A - tol[None], 0.0)
    w = np.exp(-(excess / p["ang_soft"]) ** 2)
    for k, j in enumerate(cols):
        # a broken joint makes everything distal to it meaningless
        chain = next(c for c in CHAINS if j in c)
        for d in chain[chain.index(j):]:
            W[:, d] = np.minimum(W[:, d], w[:, k])
    return W, float(ref), bad


def accel_weights(X, ok, p):
    """(n,21) Hampel weight on the per-joint second difference (mm), self-scaled."""
    n = len(X)
    W = np.ones((n, 21))
    if n < 3:
        return W, np.zeros((n, 21))
    R = np.full((n, 21), np.nan)
    good = ok & np.isfinite(X[:, :, 0])
    for t in range(1, n - 1):
        m = good[t] & good[t - 1] & good[t + 1]
        if not m.any():
            continue
        r = np.linalg.norm(X[t] - 0.5 * (X[t - 1] + X[t + 1]), axis=1)
        R[t, m] = r[m]
    med = np.nanmedian(R, 0)
    mad = 1.4826 * np.nanmedian(np.abs(R - med), 0)
    scale = np.maximum(np.nan_to_num(mad, nan=0.0), p["acc_floor_mm"] / 1000.0)
    Z = np.nan_to_num((R - med) / scale, nan=0.0)
    over = np.clip(Z - p["acc_z"], 0, None)
    W = np.exp(-(over / p["acc_soft"]) ** 2)
    return W, Z


def project(X, K):
    return X[..., :2] / np.maximum(X[..., 2:3], 1e-6) @ np.asarray(K)[:2, :2].T + np.asarray(K)[:2, 2]


def render_hand(uv, ok, shape, radius_px, scale=0.25):
    """Coarse hand silhouette from projected joints: thick bones + a filled palm."""
    h, w = int(shape[0] * scale), int(shape[1] * scale)
    img = np.zeros((h, w), np.uint8)
    q = (uv * scale).astype(np.int32)
    r = max(1, int(radius_px * scale))
    for a, b in BONES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(q[a]), tuple(q[b]), 1, r * 2)
    pal = [q[j] for j in PALM if ok[j]]
    if len(pal) >= 3:
        cv2.fillConvexPoly(img, cv2.convexHull(np.array(pal, np.int32)), 1)
    return img > 0


def mask_iou_series(X, ok, K, shape, mask_for, scale=0.25):
    """IoU per frame between the rendered hand and mask_for(t) (bool, full resolution)."""
    out = np.full(len(X), np.nan)
    if cv2 is None:
        return out
    for t in range(len(X)):
        if not np.isfinite(X[t, 0, 0]) or ok[t].sum() < 6:
            continue
        m = mask_for(t)
        if m is None or not m.any():
            continue
        uv = project(X[t], K)
        z = float(np.nanmedian(X[t][ok[t], 2]))
        rad = 0.018 * float(K[0][0]) / max(z, 0.1)
        r = render_hand(uv, ok[t], shape, rad, scale)
        ms = cv2.resize(m.astype(np.uint8), (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        u = np.logical_or(r, ms).sum()
        out[t] = float(np.logical_and(r, ms).sum() / u) if u else np.nan
    return out


# --------------------------------------------------------------------------- driver
def score(X, valid, conf, rms_px=None, iou=None, params=None):
    """X (n,21,3) NaN where absent; valid/conf (n,21). -> (conf2, valid2, diag)."""
    p = dict(DEFAULTS, **(params or {}))
    X = np.asarray(X, float)
    valid = np.asarray(valid, bool)
    conf = np.asarray(conf, float)
    n = len(X)
    present = np.isfinite(X[:, :, 0])
    ok = valid & present
    Wb, med, dev = bone_weights(X, ok, p)
    Wa, ref, bad_ang = angle_weights(X, ok, p)
    Wc, Z = accel_weights(X, ok, p)
    W = Wb * Wa * Wc
    frame = np.ones(n)
    if rms_px is not None:
        r = np.asarray(rms_px, float)
        frame = frame * np.where(np.isfinite(r), np.exp(-(np.nan_to_num(r, nan=0.0) / p["rms_soft"]) ** 2), 1.0)
    if iou is not None:
        i = np.asarray(iou, float)
        wi = np.clip((i - p["iou_lo"]) / max(p["iou_hi"] - p["iou_lo"], 1e-6), 0.12, 1.0)
        frame = frame * np.where(np.isfinite(i), wi, 1.0)
    Wtot = W * frame[:, None]
    # per-frame verdict: the median weight over the joints that were measured
    fs = np.array([np.median(Wtot[t][ok[t]]) if ok[t].any() else 0.0 for t in range(n)])
    reject = present.any(1) & (fs < p["min_frame"])
    conf2 = np.clip(conf * Wtot, 0.0, 1.0)
    valid2 = ok & (Wtot > p["min_joint"]) & ~reject[:, None]
    diag = {"bone_median_m": [round(float(x), 5) for x in np.nan_to_num(med)],
            "curl_sign": ref,
            "n_joint_downweighted": int(((Wtot < 0.5) & ok).sum()),
            "n_joint_dropped": int((ok & ~valid2).sum()),
            "n_frame_rejected": int(reject.sum()),
            "frame_score": [round(float(x), 3) for x in fs],
            "reject_frames": [int(t) for t in np.flatnonzero(reject)],
            "mean_weight": float(Wtot[ok].mean()) if ok.any() else None,
            "n_hyperextended": int(bad_ang.sum()),
            "n_accel_outlier": int(((Z > p["acc_z"]) & ok).sum()),
            "median_mask_iou": (None if iou is None or not np.isfinite(iou).any()
                                else round(float(np.nanmedian(iou)), 3))}
    return conf2, valid2, diag


def sam3_mask_loader(run: Path, view: str, track: str):
    """t -> bool mask from <run>/stereo/hands_sam3/masks/<view>/<track>/NNNNNN.png, or None."""
    d = Path(run) / "stereo/hands_sam3/masks" / view / track
    if cv2 is None or not d.is_dir():
        return None

    def get(t):
        p = d / f"{t:06d}.png"
        if not p.exists():
            return None
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return None if m is None else m > 0

    return get


def apply_to_tri(tri: dict, run: Path, sides=("left", "right"), params=None, use_mask=True):
    """Rewrite `conf`/`valid` of a hands_tri.json dict in place. Returns the diagnostics."""
    ia = json.loads((Path(run) / "recon_hawor/intrinsics_stable.json").read_text())
    K = [[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.0]]
    shape = (ia["height"], ia["width"])
    n = 0
    for side in sides:
        for k in tri.get(side, {}).get("X_raw", {}):
            n = max(n, int(k) + 1)
    report = {}
    for side in sides:
        S = tri.get(side)
        if not S or "X_raw" not in S:
            continue
        X = np.full((n, 21, 3), np.nan)
        V = np.zeros((n, 21), bool)
        C = np.zeros((n, 21))
        for k, v in S["X_raw"].items():
            X[int(k)] = np.array(v, float)
        for k, v in S.get("valid", {}).items():
            V[int(k)] = np.array(v, bool)
        for k, v in S.get("conf", {}).items():
            C[int(k)] = np.array(v, float)
        rms = np.full(n, np.nan)
        for st in S.get("stats", []):
            if st.get("rms_px") is not None:
                rms[st["frame"]] = st["rms_px"]
        iou = None
        if use_mask:
            ld = sam3_mask_loader(run, "a", side)
            if ld is not None:
                iou = mask_iou_series(X, V, K, shape, ld)
        C2, V2, diag = score(X, V, C, rms_px=rms, iou=iou, params=params)
        for k in list(S["X_raw"]):
            t = int(k)
            S.setdefault("conf", {})[k] = [round(float(x), 3) for x in C2[t]]
            S.setdefault("valid", {})[k] = [bool(x) for x in V2[t]]
        S["quality"] = diag
        report[side] = {k: v for k, v in diag.items() if k not in ("frame_score",)}
    tri["hand_quality"] = {"params": dict(DEFAULTS, **(params or {})), "sides": report}
    return report
