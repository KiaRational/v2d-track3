"""Geometric mask validation with metric stereo depth (numpy + cv2/PIL only, so it
also runs inside the SAM2 / GroundingDINO containers).

A segmentation mask of a known rigid object must back-project (stereo depth) to a
point cloud whose robust extent is compatible with the scanned mesh's bounding box:
a mask that covers the table is far too large, a speck is too small. Depth
discontinuities split the mask into surface-connected components so that a mask
leaking onto the background can be trimmed back to the object.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # containers without OpenCV
    cv2 = None


def decode_depth(path):
    """depth_m = 65535/png - 1; png 0/65535 invalid -> 0."""
    if cv2 is not None:
        png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        from PIL import Image
        png = np.asarray(Image.open(str(path)))
    if png is None:
        return None
    png = png.astype(np.float32)
    d = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, 0)
    return np.where(np.isfinite(d) & (d > 0.05) & (d < 5.0), d, 0).astype(np.float32)


def load_mask(path):
    if cv2 is not None:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        from PIL import Image
        try:
            m = np.asarray(Image.open(str(path)).convert("L"))
        except FileNotFoundError:
            m = None
    return None if m is None else m > 0


def backproject(mask, depth, K, max_pts=4000, seed=0):
    v, u = np.nonzero(mask & (depth > 0))
    if len(u) > max_pts:
        sel = np.random.default_rng(seed).choice(len(u), max_pts, replace=False)
        u, v = u[sel], v[sel]
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def robust_extent(P, lo=5, hi=95):
    """Sorted (descending) robust extents along the principal axes."""
    if len(P) < 30:
        return None
    Q = P - np.median(P, 0)
    _, _, Vt = np.linalg.svd(Q[:: max(1, len(Q) // 1500)], full_matrices=False)
    R = Q @ Vt.T
    e = np.percentile(R, hi, 0) - np.percentile(R, lo, 0)
    return np.sort(e)[::-1]


def depth_components(mask, depth, rel=0.03, abs_m=0.015):
    """Label mask pixels into surface-connected components (4-neighbours whose
    depth differs by < max(abs_m, rel*z)); pixels without depth get label 0."""
    valid = mask & (depth > 0)
    thr = np.maximum(abs_m, rel * depth)
    dx = np.abs(np.diff(depth, axis=1)) < thr[:, :-1]
    dy = np.abs(np.diff(depth, axis=0)) < thr[:-1, :]
    # cut edges: a pixel is a "border" if it disconnects from its right/down neighbour
    border = np.zeros_like(valid)
    border[:, :-1] |= ~dx & valid[:, :-1] & valid[:, 1:]
    border[:-1, :] |= ~dy & valid[:-1, :] & valid[1:, :]
    core = valid & ~border
    n, lab = cv2.connectedComponents(core.astype(np.uint8), connectivity=4)
    return n, lab


def clean_mask(mask, depth, K, mesh_ext, min_px=150):
    """Keep the depth-connected components that plausibly belong to one object:
    the largest component plus any other within ~one object diagonal of it.
    Returns the cleaned mask (pixels without depth are kept if inside the
    retained components' dilated hull)."""
    if cv2 is None or mask.sum() < min_px:
        return mask
    n, lab = depth_components(mask, depth)
    if n <= 2:
        return mask
    sizes = np.bincount(lab.ravel(), minlength=n)
    sizes[0] = 0
    order = np.argsort(-sizes)
    diag = float(np.linalg.norm(mesh_ext))
    main = order[0]
    c0 = np.median(backproject(lab == main, depth, K, 800), 0)
    keep = np.zeros(n, bool)
    keep[main] = True
    for k in order[1:]:
        if sizes[k] < 30:
            break
        c = np.median(backproject(lab == k, depth, K, 300), 0)
        if np.linalg.norm(c - c0) < 0.6 * diag:
            keep[k] = True
    kept = keep[lab] & (lab > 0)
    # re-admit depth-less / border pixels of the original mask adjacent to kept parts
    near = cv2.dilate(kept.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    out = kept | (mask & near & ~((lab > 0) & ~keep[lab]))
    return out


def geometry_check(mask, depth, K, mesh_ext, min_px=150):
    """Compare the mask's robust 3D extent with the mesh's sorted extents.
    Returns dict(ok, reason, ext, ratio, n_px, n_depth, center)."""
    n_px = int(mask.sum())
    rec = {"n_px": n_px, "ok": False, "reason": "", "ext": None, "ratio": None, "center": None, "fit": 0.0}
    if n_px < min_px:
        rec["reason"] = "empty"
        return rec
    me = np.sort(np.asarray(mesh_ext))[::-1]
    diag = float(np.linalg.norm(me))
    # Silhouette pixels carry mixed foreground/background stereo depth: erode first,
    # then drop points far (> 1 diag) from the median point before measuring.
    core = mask
    if cv2 is not None and n_px > 4 * min_px:
        core = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        if core.sum() < min_px:
            core = mask
    P = backproject(core, depth, K)
    rec["n_depth"] = int(len(P))
    if len(P) < 50:
        rec["reason"] = "no_depth"
        rec["ok"], rec["fit"] = True, 0.5  # cannot judge; do not reject
        return rec
    c = np.median(P, 0)
    P = P[np.linalg.norm(P - c, axis=1) < 1.0 * diag]
    e = robust_extent(P) if len(P) >= 30 else np.zeros(3)
    ratio = e / me
    rec.update(ext=e.tolist(), ratio=ratio.tolist(), center=c.tolist())
    if e[0] > max(1.35 * me[0], 0.9 * diag) or e[1] > 1.5 * me[1] + 0.02:
        rec["reason"] = "too_large"
    elif e[0] < 0.15 * me[0] and n_px < 3000:
        rec["reason"] = "too_small"
    elif ratio[0] < 0.45 and ratio[1] < 0.5:
        # far too small on BOTH major axes: this is a different, smaller object
        # (occlusion shortens some axes, it does not shrink the whole silhouette)
        rec["reason"] = "too_small"
    else:
        rec["ok"] = True
    rec["fit"] = fit_score(ratio)
    return rec


def fit_score(ratio):
    """1 for extents matching the mesh, decaying for over-size and under-size.

    Under-size has to bite: a blue-cup mask measures about a third of the water
    pitcher on every axis (ratio ~[0.35, 0.37, 0.48]) yet the old weak penalty still
    scored it 0.66 as a "pitcher", which is how the cup got adopted as the pitcher's
    reference in ep21/ep22. Occlusion only ever shrinks the mask along some axes, so
    the penalty uses the two largest ratios together and tolerates ratios >= ~0.75.
    """
    r = np.asarray(ratio, float)
    over = np.clip(r[:2] - 1.15, 0, None)
    under = np.clip(0.75 - r[:2], 0, None)
    return float(np.exp(-(over ** 2).sum() / 0.05 - (under ** 2).sum() / 0.06))


def mask_iou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def containment(a, b):
    """Fraction of a inside b."""
    s = a.sum()
    return float(np.logical_and(a, b).sum() / s) if s else 0.0
