"""Distorted multiview hand triangulation, similarity fitting, and smoothing.

Written by GPT-6-Astra (Codex) to our spec; saved verbatim from its reply.
"""

from typing import Mapping, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

Array = np.ndarray
Similarity = Tuple[float, Array, Array]


def undistort_points(uv: Array, cam: Mapping) -> Array:
    """Convert raw distorted pixels (..., 2) to normalized camera coordinates."""
    uv = np.asarray(uv, dtype=float)
    if uv.ndim < 1 or uv.shape[-1] != 2:
        raise ValueError("uv must have shape (..., 2)")
    flat = uv.reshape(-1, 2)
    out = np.full_like(flat, np.nan)
    good = np.isfinite(flat).all(axis=1)
    if good.any():
        out[good] = cv2.undistortPoints(
            flat[good].reshape(-1, 1, 2),
            np.asarray(cam["K"], dtype=float),
            np.asarray(cam["dist"], dtype=float),
        ).reshape(-1, 2)
    return out.reshape(uv.shape)


def _transforms(calib: Mapping, ref: str) -> dict:
    """Return transforms mapping reference-camera points into each camera."""
    a_to_c = np.asarray(calib["T_a_to_c"], dtype=float)
    c_to_b = np.asarray(calib["T_left_to_right"], dtype=float)
    for transform in (a_to_c, c_to_b):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Extrinsics must be finite 4x4 rigid transforms")
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("Extrinsic rotations must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0):
            raise ValueError("Extrinsic rotations must be proper rotations")
    from_a = {"a": np.eye(4), "c": a_to_c, "b": c_to_b @ a_to_c}
    if ref not in from_a:
        raise ValueError("ref must be 'a', 'b', or 'c'")
    ref_to_a = np.linalg.inv(from_a[ref])
    return {name: transform @ ref_to_a for name, transform in from_a.items()}


def _project(points: Array, transform: Array, cam: Mapping) -> Array:
    """Project reference-frame points into a raw distorted image."""
    rvec = cv2.Rodrigues(transform[:3, :3])[0]
    return cv2.projectPoints(
        np.asarray(points, dtype=float).reshape(-1, 3),
        rvec,
        transform[:3, 3],
        np.asarray(cam["K"], dtype=float),
        np.asarray(cam["dist"], dtype=float),
    )[0].reshape(-1, 2)


def triangulate_joints(
    obs: Mapping,
    calib: Mapping,
    ref: str = "a",
    *,
    thr: float = 0.3,
    max_px: float = 5.0,
    depth_range: Tuple[float, float] = (0.1, 1.5),
) -> Tuple[Array, Array, Array, Array]:
    """Return XYZ metres, weighted pixel RMS, view counts, and validity."""
    if not np.isfinite([thr, max_px, *depth_range]).all():
        raise ValueError("Thresholds must be finite")
    if thr < 0 or max_px <= 0 or not 0 < depth_range[0] < depth_range[1]:
        raise ValueError("Invalid confidence, RMS, or depth thresholds")
    transforms = _transforms(calib, ref)
    cameras = calib["cameras"]
    prepared = {}
    for name, (uv, conf) in obs.items():
        if name not in transforms or name not in cameras:
            raise ValueError(f"Unknown or uncalibrated camera: {name}")
        uv, conf = np.asarray(uv, float), np.asarray(conf, float)
        if uv.shape != (21, 2) or conf.shape != (21,):
            raise ValueError("Observations must have shapes (21,2) and (21,)")
        good = np.isfinite(uv).all(axis=1) & np.isfinite(conf)
        good &= (conf >= thr) & (conf > 0)
        normalized = np.full((21, 2), np.nan)
        normalized[good] = undistort_points(uv[good], cameras[name])
        good &= np.isfinite(normalized).all(axis=1)
        prepared[name] = (uv, conf, normalized, good)

    X = np.full((21, 3), np.nan)
    rms = np.full(21, np.nan)
    counts = np.zeros(21, dtype=int)
    valid = np.zeros(21, dtype=bool)
    for joint in range(21):
        names = [name for name, data in prepared.items() if data[3][joint]]
        counts[joint] = len(names)
        if len(names) < 2:
            continue
        weights = np.array([prepared[name][1][joint] for name in names])
        weights /= weights.max()
        rows = []
        for name, weight in zip(names, weights):
            u, v = prepared[name][2][joint]
            P = transforms[name][:3]
            rows.extend(np.sqrt(weight) * np.array([u * P[2] - P[0],
                                                   v * P[2] - P[1]]))
        try:
            _, singular, vt = np.linalg.svd(np.asarray(rows))
            h = vt[-1]
            if singular[-2] <= singular[0] * 1e-12 or abs(h[3]) < 1e-12:
                continue
            initial = h[:3] / h[3]

            def residual(point: Array) -> Array:
                errors = [
                    _project(point, transforms[name], cameras[name])[0]
                    - prepared[name][0][joint]
                    for name in names
                ]
                return (np.asarray(errors) * np.sqrt(weights[:, None])).ravel()

            fit = least_squares(residual, initial, method="lm", max_nfev=200)
            if not np.isfinite(fit.x).all() or not np.isfinite(fit.fun).all():
                continue
            X[joint] = fit.x
            rms[joint] = np.sqrt(np.sum(fit.fun ** 2) / weights.sum())
            depths = [
                (transforms[name][:3, :3] @ fit.x
                 + transforms[name][:3, 3])[2]
                for name in names
            ]
            valid[joint] = (
                fit.success
                and np.all(np.asarray(depths) > 0)
                and rms[joint] < max_px
                and depth_range[0] <= fit.x[2] <= depth_range[1]
            )
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
    return X, rms, counts, valid


def umeyama(
    src: Array,
    dst: Array,
    weights: Optional[Array] = None,
    with_scale: bool = True,
) -> Similarity:
    """Fit dst ~= s * (src @ R.T) + t using a proper rotation."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.ndim != 2 or src.shape[1] != 3 or dst.shape != src.shape:
        raise ValueError("src and dst must both have shape (N,3)")
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(src),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite, nonnegative, and shape (N,)")
    keep = w > 0
    if keep.sum() < 3:
        raise ValueError("At least three positive-weight correspondences required")
    src, dst, w = src[keep], dst[keep], w[keep]
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Positive-weight correspondences must be finite")
    w = w / w.max()
    w /= w.sum()
    src_mean, dst_mean = w @ src, w @ dst
    a, b = src - src_mean, dst - dst_mean
    covariance = (b * w[:, None]).T @ a
    U, singular, Vt = np.linalg.svd(covariance)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-10:
        raise ValueError("Degenerate alignment: points must not be collinear")
    signs = np.ones(3)
    signs[-1] = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    R = (U * signs) @ Vt
    variance = np.sum(w * np.sum(a * a, axis=1))
    s = float(np.dot(singular, signs) / variance) if with_scale else 1.0
    t = dst_mean - s * (R @ src_mean)
    return s, R, t


def align_hand_to_triangulation(
    pred_joints: Array,
    X: Array,
    valid: Array,
    weights: Optional[Array] = None,
    *,
    with_scale: bool = True,
    huber_delta_mm: float = 10.0,
    iterations: int = 8,
) -> Tuple[float, Array, Array, Array, Array]:
    """Robust similarity fit; residuals are NaN outside usable target joints."""
    pred, target = np.asarray(pred_joints, float), np.asarray(X, float)
    valid = np.asarray(valid, bool)
    base = np.ones(21) if weights is None else np.asarray(weights, float)
    if pred.shape != (21, 3) or target.shape != (21, 3):
        raise ValueError("Joint arrays must have shape (21,3)")
    if valid.shape != (21,) or base.shape != (21,):
        raise ValueError("valid and weights must have shape (21,)")
    if not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("weights must be finite and nonnegative")
    if iterations < 1 or not np.isfinite(huber_delta_mm) or huber_delta_mm <= 0:
        raise ValueError("iterations and Huber threshold must be positive")
    usable = valid & np.isfinite(pred).all(axis=1)
    usable &= np.isfinite(target).all(axis=1) & (base > 0)
    if usable.sum() < 3:
        raise ValueError("At least three usable joints required for alignment")
    source, dest = pred[usable], target[usable]
    base = base[usable] / base[usable].max()
    robust = np.ones(len(source))
    delta = huber_delta_mm / 1000.0
    for _ in range(iterations):
        s, R, t = umeyama(source, dest, base * robust, with_scale)
        distances = np.linalg.norm(s * (source @ R.T) + t - dest, axis=1)
        updated = np.minimum(1.0, delta / np.maximum(distances, 1e-15))
        if np.max(np.abs(updated - robust)) < 1e-4:
            robust = updated
            break
        robust = updated
    s, R, t = umeyama(source, dest, base * robust, with_scale)
    aligned = s * (pred @ R.T) + t
    residuals = np.full(21, np.nan)
    residuals[usable] = 1000.0 * np.linalg.norm(aligned[usable] - dest, axis=1)
    return s, R, t, aligned, residuals


def temporal_smooth_joints(
    X_seq: Array,
    valid_seq: Array,
    fps: float,
    *,
    min_cutoff: float = 1.0,
    beta: float = 0.02,
    d_cutoff: float = 1.0,
) -> Array:
    """One-Euro coordinate filtering; hold gaps, use elapsed time on resumption."""
    X, valid = np.asarray(X_seq, float), np.asarray(valid_seq, bool)
    if X.ndim != 3 or X.shape[1:] != (21, 3):
        raise ValueError("X_seq must have shape (T,21,3)")
    if valid.shape != X.shape[:2]:
        raise ValueError("valid_seq must have shape (T,21)")
    if not np.isfinite([fps, min_cutoff, beta, d_cutoff]).all():
        raise ValueError("Filter parameters must be finite")
    if fps <= 0 or min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
        raise ValueError("Invalid filter parameters")
    result = np.full_like(X, np.nan)
    filtered = np.full((21, 3), np.nan)
    previous = np.full((21, 3), np.nan)
    derivative = np.zeros((21, 3))
    last_seen = np.full(21, -1, dtype=int)

    def alpha(cutoff: Array, dt: float) -> Array:
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))

    for frame in range(len(X)):
        for joint in range(21):
            if valid[frame, joint] and np.isfinite(X[frame, joint]).all():
                value = X[frame, joint]
                if last_seen[joint] < 0:
                    filtered[joint] = value
                else:
                    dt = (frame - last_seen[joint]) / fps
                    velocity = (value - previous[joint]) / dt
                    ad = alpha(np.asarray(d_cutoff), dt)
                    derivative[joint] += ad * (velocity - derivative[joint])
                    cutoff = min_cutoff + beta * np.abs(derivative[joint])
                    filtered[joint] += alpha(cutoff, dt) * (value - filtered[joint])
                previous[joint] = value
                last_seen[joint] = frame
            result[frame, joint] = filtered[joint]
    return result


def _self_test() -> None:
    """Reproducible synthetic smoke test; run only on a compute node."""
    rng = np.random.default_rng(17)
    cameras = {}
    for name, size, focal in [
        ("a", [2028, 1520], 1150.0),
        ("c", [1280, 800], 790.0),
        ("b", [1280, 800], 795.0),
    ]:
        w, h = size
        cameras[name] = {
            "K": np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]]),
            "dist": np.array([
                -0.12, 0.035, 0.0005, -0.0004, -0.006, 0.008, -0.002,
                0.0003, 0.0001, -0.00005, 0.00008, -0.00004, 0.001, -0.001,
            ]),
            "size": size,
        }
    c_to_b = np.eye(4)
    c_to_b[0, 3] = -0.075
    a_to_c = np.eye(4)
    a_to_c[:3, :3] = cv2.Rodrigues(np.array([0.008, -0.015, 0.003]))[0]
    a_to_c[:3, 3] = [0.0375, 0.018, 0.004]
    calib = {"cameras": cameras, "T_left_to_right": c_to_b, "T_a_to_c": a_to_c}
    truth = np.zeros((21, 3))
    truth[0] = [0, 0.045, 0.45]
    for finger, (x, length) in enumerate(zip(
        [-0.04, -0.023, 0.0, 0.021, 0.039], [0.065, 0.095, 0.105, 0.098, 0.08]
    )):
        for segment in range(4):
            fraction = (segment + 1) / 4
            truth[1 + 4 * finger + segment] = [
                x * (0.65 + 0.35 * fraction),
                0.025 - length * fraction,
                0.45 + 0.012 * np.sin(fraction * np.pi),
            ]
    truth += rng.normal(0, 0.001, truth.shape)
    obs = {}
    for name, transform in _transforms(calib, "a").items():
        uv = _project(truth, transform, cameras[name])
        uv += rng.normal(0, 1.5, uv.shape)
        confidence = rng.uniform(0.7, 1.0, 21)
        confidence[rng.random(21) < 0.12] = 0
        obs[name] = (uv, confidence)
    obs["a"][1][0] = 0  # Guarantee at least one dropped observation.
    two = triangulate_joints({name: obs[name] for name in ("c", "b")}, calib)
    three = triangulate_joints(obs, calib)
    common = two[3] & three[3] & (three[2] == 3)
    for label, result in [("2-view c/b", two), ("up-to-3-view a/c/b", three)]:
        X, _, counts, valid = result
        error = 1000 * np.linalg.norm(X - truth, axis=1)
        mean = error[valid].mean() if valid.any() else float("nan")
        matched = error[common].mean() if common.any() else float("nan")
        print(f"{label}: valid={valid.sum()}/21, mean={mean:.2f} mm; "
              f"matched subset ({common.sum()} joints)={matched:.2f} mm")
    rotation = cv2.Rodrigues(rng.normal(0, 0.5, 3))[0]
    scale = rng.uniform(0.7, 1.4)
    prediction = scale * (truth @ rotation.T) + rng.normal(0, 0.1, 3)
    prediction += rng.normal(0, 0.003, prediction.shape)
    X, _, counts, valid = three
    _, _, _, aligned, residuals = align_hand_to_triangulation(
        prediction, X, valid, counts.astype(float)
    )
    print(f"Alignment mean residual to triangulation: "
          f"{np.nanmean(residuals):.2f} mm")
    print(f"Aligned mean error to truth: "
          f"{1000 * np.linalg.norm(aligned - truth, axis=1).mean():.2f} mm")


if __name__ == "__main__":
    _self_test()
