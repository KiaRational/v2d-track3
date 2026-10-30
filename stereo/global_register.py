"""Object 6D poses from stereo 3D reconstruction, independent of FoundationPose.

Pipeline per object (all geometry metric; FoundationStereo depth in colour-camera pixels):
 1. Segments -> 3D.  The 2D object mask (--mask_dir, default objects/masks; SAM 3 masks
    will live in objects/masks_sam3) is eroded, hand / other-object pixels (dilated) are
    removed, depth edges (local depth range > 1.5 cm + 2 % z) are dropped, the rest is
    back-projected, statistical-outlier filtered, reduced to its largest depth-connected
    cluster(s), and the support plane (RANSAC on a ring of pixels around the mask) is cut
    away.  Clouds are moved to the SLAM world with the stereo-consistent SLAM scale
    (estimated here by reprojecting background depth between frames).
 2. 3D location.  Visible-points centroid c; the camera only sees the front surface, so
    for every orientation hypothesis R the translation is t = c - R c_vis(R, view) where
    c_vis is the centroid of the mesh's camera-visible surface (z-buffer, precomputed on a
    view sphere).  A rotation-free anchor (c pushed along the viewing ray by the mean
    visible-centroid depth offset) is saved as the pure "location" estimate.
 3. Orientation from the mesh.  ~3000 uniform SO(3) hypotheses anchored as above are
    scored in a vectorised coarse pass (truncated scan->model distance-field residual,
    free-space violations and silhouette containment of the projected visible surface);
    the best (pose-NMS) ones are refined by symmetric trimmed ICP (scan->mesh +
    visible-mesh->scan) and re-scored: inlier fitness, visibility-aware explanation of the
    visible model surface by the depth map, silhouette IoU of the rendered mesh against the
    mask (occluded pixels ignored), depth agreement and support-plane penetration.
    Rotationally symmetric meshes (auto-detected) compare hypotheses by symmetry axis only.
 4. Multi-frame + pose.  Frames where no hand is near the object and whose world centroid
    is stable form static segments; their clouds are fused in the world (3 mm voxels) and
    registered once against all segment views, the result is injected as a hypothesis in
    every frame of the segment.  A Viterbi pass over frames (unary = 1 - score, pairwise =
    world pose change, loose while a hand touches the object) picks one hypothesis per
    frame; gaps are interpolated in the world.

Outputs:
  <run>/objects/<name>/<variant>/NNNNNN.json  {"hyps": [{"T","fitness","rmse_mm","iou",
      "vis_expl","depth_agree","pen","score","src"}...], "chosen", "loc_raw", "loc_anchor",
      "n_scan", "touched", "segment"}
  <run>/stereo_global/<name>/poses/NNNNNN.json  {"R","T"} camera-from-object (eval layout),
      Viterbi path robustly smoothed with se3_fusion (static while untouched);
  <run>/stereo_global_raw/<name>/poses/  the Viterbi path itself (gaps interpolated)
  <run>/objects/<name>/<variant>_report.json   (variant default stereo_global; e.g. --mask_dir
      objects_v2/masks --variant stereo_global_v2, whose ids come from objects_v2/objects.json)
FoundationPose is not used, except with --fp_hyp (adds its pose as ONE extra hypothesis).
Usage (host env, CPU job): python global_register.py <run> [...] [--mask_dir objects/masks]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
import trimesh
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from object_icp import HAND_IDS, decode_depth, kabsch, load_T
from se3_fusion import fuse_trajectory

ROOT = Path(os.environ["V2D_ROOT"])
CACHE = ROOT / "cache" / "global_models"
N_MODEL, N_DENSE, N_VIEWS, N_VIS_SUB = 4000, 30000, 600, 150
N_ROT, N_REFINE, TOPK, N_SEEDS = 2000, 64, 8, 6
DF_VOX, DF_TRUNC, DF_PAD = 0.0025, 0.02, 0.05  # cost truncation 2 cm; field (and gradient) to 5 cm
cv2.setNumThreads(1)


# ----------------------------------------------------------------------------- model
def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def detect_symmetry(P, c):
    """Continuous rotational symmetry about a principal axis through the centroid:
    median NN distance of the cloud to itself rotated by 60/120/180 deg < 4 mm (median:
    robust to the mocap markers, which break exact symmetry)."""
    tree = cKDTree(P)
    ev, vec = np.linalg.eigh(np.cov((P - c).T))
    best = (np.inf, None)
    for k in range(3):
        a = vec[:, k]
        err = max(np.median(tree.query((P - c) @ Rotation.from_rotvec(a * np.deg2rad(g)).as_matrix().T + c)[0])
                  for g in (60, 120, 180))
        if err < best[0]:
            best = (err, a)
    return (best[1], float(best[0])) if best[0] < 0.004 else (None, float(best[0]))


def build_model(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.exists():
        return
    mesh = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj", force="mesh")
    dense, fi = trimesh.sample.sample_surface(mesh, N_DENSE, seed=1)
    dn = mesh.face_normals[fi]
    rng = np.random.default_rng(0)
    sel = rng.choice(N_DENSE, N_MODEL, replace=False)
    pts, nrm = dense[sel], dn[sel]
    centre = (mesh.bounds[0] + mesh.bounds[1]) / 2
    # visibility per view direction (v points from object towards the camera), orthographic z-buffer
    views = fib_sphere(N_VIEWS)
    cell = max(0.003, 2.0 * np.sqrt(mesh.area / N_DENSE))
    vis = np.zeros((N_VIEWS, N_MODEL), bool)
    for j, v in enumerate(views):
        a = np.cross(v, [1, 0, 0] if abs(v[0]) < 0.9 else [0, 1, 0])
        a /= np.linalg.norm(a)
        b = np.cross(v, a)
        key = lambda X: np.floor(X @ np.stack([a, b], 1) / cell).astype(np.int64)
        kd = key(dense)
        lo = kd.min(0)
        kd -= lo
        W = kd[:, 1].max() + 3
        idd = kd[:, 0] * W + kd[:, 1]
        zmax = np.full(idd.max() + 1, -np.inf)
        np.maximum.at(zmax, idd, dense @ v)
        km = np.clip(key(pts) - lo, 0, None)
        im = np.clip(km[:, 0] * W + km[:, 1], 0, len(zmax) - 1)
        vis[j] = (pts @ v >= zmax[im] - 0.006) & (nrm @ v > -0.2)
    cvis = np.stack([pts[vis[j]].mean(0) if vis[j].any() else centre for j in range(N_VIEWS)])
    vis_sub = np.stack([rng.choice(np.nonzero(vis[j])[0] if vis[j].any() else np.arange(N_MODEL), N_VIS_SUB)
                        for j in range(N_VIEWS)])
    # translation seeds per view: the visible centroid + k-means centres of the visible surface
    # (a partial scan's centroid matches the centroid of SOME visible patch, not of all of it)
    seeds = np.zeros((N_VIEWS, 1 + N_SEEDS, 3))
    for j in range(N_VIEWS):
        Vp = pts[vis[j]] if vis[j].sum() >= N_SEEDS else pts
        seeds[j, 0] = Vp.mean(0)
        seeds[j, 1:] = kmeans2(Vp, N_SEEDS, seed=j, minit="++")[0]
    # truncated distance field of the surface
    lo = dense.min(0) - DF_PAD
    hi = dense.max(0) + DF_PAD
    dims = np.ceil((hi - lo) / DF_VOX).astype(int) + 1
    g = np.stack(np.meshgrid(*[lo[i] + DF_VOX * np.arange(dims[i]) for i in range(3)], indexing="ij"), -1)
    df = cKDTree(dense).query(g.reshape(-1, 3), distance_upper_bound=DF_PAD)[0]
    df = np.minimum(df, DF_PAD).reshape(dims).astype(np.float32)
    dfg = np.stack(np.gradient(df, DF_VOX), -1).astype(np.float32)  # unit-ish direction away from surface
    axis, sym_err = detect_symmetry(dense[:6000], dense.mean(0))
    # rotation-free location offset: mean depth of the full centre behind the visible centroid
    depth_off = float(np.mean([(cvis[j] - centre) @ views[j] for j in range(N_VIEWS)]))
    tmp = f.with_name(f"{name}.{os.getpid()}.tmp.npz")  # atomic: concurrent jobs may build too
    np.savez_compressed(tmp, pts=pts, nrm=nrm, dense=dense.astype(np.float32), centre=centre, views=views,
                        vis=vis, cvis=cvis, seeds=seeds, vis_sub=vis_sub, df=df, dfg=dfg, df_lo=lo, axis=np.zeros(3) if axis is None else axis,
                        has_axis=axis is not None, sym_err=sym_err, depth_off=depth_off,
                        diam=np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    os.replace(tmp, f)


@lru_cache(maxsize=4)
def get_model(name):
    d = dict(np.load(CACHE / f"{name}.npz"))
    d["tree"] = cKDTree(d["pts"])
    d["view_tree"] = cKDTree(d["views"])
    d["axis"] = d["axis"] if bool(d["has_axis"]) else None
    return d


def df_lookup(M, X, grad=False):
    """Distance of object-frame points X (..., 3) to the surface (truncated at DF_PAD);
    with grad=True also the distance-field gradient (points away from the surface)."""
    idx = np.rint((X - M["df_lo"]) / DF_VOX).astype(np.int64)
    dims = np.array(M["df"].shape)
    ok = np.all((idx >= 0) & (idx < dims), -1)
    idx = np.clip(idx, 0, dims - 1)
    d = np.where(ok, M["df"][idx[..., 0], idx[..., 1], idx[..., 2]], DF_PAD)
    if not grad:
        return d
    g = M["dfg"][idx[..., 0], idx[..., 1], idx[..., 2]] * ok[..., None]
    return d, g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)


# ----------------------------------------------------------------------------- run io
class Run:
    def __init__(self, run, mask_dir):
        self.dir = Path(run)
        k = json.loads((self.dir / "recon_hawor/intrinsics_stable.json").read_text())
        self.K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
        self.shape = (k["height"], k["width"])
        self.mask_dir = self.dir / mask_dir
        oj = self.mask_dir.parent / "objects.json"  # e.g. objects_v2/objects.json (ids may differ from v1)
        self.objs = json.loads((oj if oj.exists() else self.dir / "objects/objects.json").read_text())
        self.slam = {int(p.stem): load_T(p) for p in sorted((self.dir / "recon_hawor/slam_poses").glob("*.json"))}
        self.n = max(self.slam) + 1 if self.slam else len(list((self.dir / "depth_stereo").glob("*.png")))

    def mask(self, oid, t):
        for base in (self.mask_dir, self.dir / "objects/masks"):
            p = base / str(oid) / f"{t:06d}.png"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    return m > 0
            if base == self.mask_dir and oid not in HAND_IDS:
                break  # objects: only the chosen mask source; hands may fall back
        return np.zeros(self.shape, bool)

    def depth(self, t):
        p = self.dir / f"depth_stereo/{t:06d}.png"
        return decode_depth(p) if p.exists() else None

    def world(self, t, scale):
        T = self.slam[t].copy()
        T[:3, 3] *= scale
        return T


def backproject(K, u, v, z):
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def estimate_slam_scale(run: Run, n_pairs=40, gap=6):
    """Scale s on the SLAM translations that makes stereo depth consistent between frames:
    background pixels of frame t reprojected into t+gap should match its depth."""
    ts = [t for t in sorted(run.slam) if t + gap in run.slam and (run.dir / f"depth_stereo/{t:06d}.png").exists()]
    if len(ts) < 5:
        return 1.0, {}
    ts = ts[:: max(1, len(ts) // n_pairs)][:n_pairs]
    scales = np.arange(0.6, 1.401, 0.01)
    res = np.zeros((len(ts), len(scales)))
    motion = []
    rng = np.random.default_rng(0)
    H, W = run.shape
    for i, t in enumerate(ts):
        d0, d1 = run.depth(t), run.depth(t + gap)
        fg = np.zeros(run.shape, bool)
        for o in run.objs:
            fg |= run.mask(o["id"], t)
        for h in HAND_IDS:
            fg |= run.mask(h, t)
        v, u = np.nonzero((d0 > 0.2) & (d0 < 3.0) & ~fg)
        if len(u) < 500:
            res[i] = np.nan
            continue
        s_ = rng.choice(len(u), 6000)
        p = backproject(run.K, u[s_], v[s_], d0[v[s_], u[s_]])
        A, B = run.slam[t], run.slam[t + gap]
        motion.append(np.linalg.norm(A[:3, 3] - B[:3, 3]))
        Rrel = B[:3, :3].T @ A[:3, :3]
        dt = B[:3, :3].T @ (A[:3, 3] - B[:3, 3])
        for j, s in enumerate(scales):
            q = p @ Rrel.T + s * dt
            z = q[:, 2]
            ok = z > 0.1
            uu = np.rint(run.K[0, 0] * q[ok, 0] / z[ok] + run.K[0, 2]).astype(int)
            vv = np.rint(run.K[1, 1] * q[ok, 1] / z[ok] + run.K[1, 2]).astype(int)
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            zo = d1[vv[inb], uu[inb]]
            good = zo > 0
            e = np.abs(zo[good] - z[ok][inb][good]) / zo[good]
            res[i, j] = np.mean(np.minimum(e, 0.05)) if len(e) > 200 else np.nan
    cost = np.nanmean(res, 0)
    j = int(np.nanargmin(cost))
    info = {"cost_at_1": float(cost[np.argmin(np.abs(scales - 1))]), "cost_best": float(cost[j]),
            "mean_cam_motion_m": float(np.mean(motion)) if motion else 0.0, "scale_grid_best": float(scales[j])}
    # accept only when clearly better than scale 1 (translation too small -> unobservable)
    s = float(scales[j]) if info["cost_at_1"] - info["cost_best"] > 0.0005 else 1.0
    return s, info


# ----------------------------------------------------------------------------- step 1
def ransac_plane(P, iters=150, thr=0.005, rng=None):
    rng = rng or np.random.default_rng(0)
    best = (0, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - a) @ n) < thr).sum())
        if k > best[0]:
            best = (k, (n, -n @ a))
    if best[1] is None:
        return None, 0.0
    n, d = best[1]
    inl = np.abs(P @ n + d) < thr  # least-squares refit on inliers
    Q = P[inl]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    if n @ c > 0:  # orient the normal towards the camera (origin)
        n = -n
    return (n, -n @ c), float(inl.mean())


def segment(run: Run, oid, other_ids, t, depth=None, max_pts=4000, erode=7):
    """Masked stereo points of one object in camera coordinates (cleaned) + diagnostics."""
    depth = run.depth(t) if depth is None else depth
    mo = run.mask(oid, t)
    occ = np.zeros(run.shape, bool)
    for h in HAND_IDS + tuple(other_ids):
        occ |= run.mask(h, t)
    out = {"n_mask": int(mo.sum()), "occ": occ, "obj": mo, "depth": depth}
    if depth is None or mo.sum() < 100:
        return None, out
    occ_d = cv2.dilate(occ.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    m = cv2.erode((mo & ~occ_d).astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0
    k5 = np.ones((5, 5), np.uint8)
    dz = cv2.dilate(depth, k5) - cv2.erode(np.where(depth > 0, depth, 1e3).astype(np.float32), k5)
    m &= (depth > 0.1) & (depth < 3.0) & (dz < 0.015 + 0.02 * depth)
    v, u = np.nonzero(m)
    if len(u) < 150:
        return None, out
    rng = np.random.default_rng(t)
    if len(u) > max_pts:
        s_ = rng.choice(len(u), max_pts, replace=False)
        u, v = u[s_], v[s_]
    P = backproject(run.K, u, v, depth[v, u])
    # statistical outlier removal
    tree = cKDTree(P)
    dk = tree.query(P, 9)[0][:, 1:].mean(1)
    keep = dk < np.median(dk) + 3 * 1.4826 * np.median(np.abs(dk - np.median(dk))) + 1e-4
    P = P[keep]
    # depth-connected clusters: keep the largest and big clusters near it
    r = max(0.006, 3 * np.median(dk))
    pairs = cKDTree(P).query_pairs(r, output_type="ndarray")
    if len(pairs):
        nc, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                                  shape=(len(P), len(P))), directed=False)
        cnt = np.bincount(lab)
        big = np.argmax(cnt)
        cb = P[lab == big].mean(0)
        ok = [c for c in range(nc) if cnt[c] >= 0.3 * cnt[big] and np.linalg.norm(P[lab == c].mean(0) - cb) < 0.15]
        P = P[np.isin(lab, ok)]
    # support plane from a ring of pixels around the object
    ring = (cv2.dilate(mo.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & \
           ~(cv2.dilate(mo.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~occ_d & (depth > 0.1) & (depth < 3.0)
    rv, ru = np.nonzero(ring)
    plane = None
    if len(ru) > 300:
        s_ = rng.choice(len(ru), min(3000, len(ru)), replace=False)
        Q = backproject(run.K, ru[s_], rv[s_], depth[rv[s_], ru[s_]])
        pl, frac = ransac_plane(Q, rng=rng)
        if pl is not None and frac > 0.5:
            h = P @ pl[0] + pl[1]
            keep = h > 0.007
            if keep.sum() >= 0.5 * len(P):  # plane plausible as the support: cut it away
                P = P[keep]
                plane = pl
    out["plane"] = plane
    if len(P) < 100:
        return None, out
    return P, out


def touch_count(run: Run, t, P, depth, radius=0.03):
    hm = run.mask(HAND_IDS[0], t) | run.mask(HAND_IDS[1], t)
    v, u = np.nonzero(hm & (depth > 0.1))
    if len(u) < 20 or P is None:
        return 0
    if len(u) > 4000:
        s_ = np.random.default_rng(t).choice(len(u), 4000, replace=False)
        u, v = u[s_], v[s_]
    Hp = backproject(run.K, u, v, depth[v, u])
    d = cKDTree(P).query(Hp, distance_upper_bound=radius)[0]
    return int(np.isfinite(d).sum())


# ----------------------------------------------------------------------------- steps 2-3
def rot_dist(M, Ra, Rb):
    if M["axis"] is not None:
        return np.degrees(np.arccos(np.clip((Ra @ M["axis"]) @ (Rb @ M["axis"]), -1, 1)))
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def centre_of(M, T):
    return T[:3, :3] @ M["centre"] + T[:3, 3]


def nms(M, Ts, order, k, dt=0.015, dr=15.0):
    keep = []
    for i in order:
        if all(np.linalg.norm(centre_of(M, Ts[i]) - centre_of(M, Ts[j])) > dt or
               rot_dist(M, Ts[i][:3, :3], Ts[j][:3, :3]) > dr for j in keep):
            keep.append(i)
            if len(keep) >= k:
                break
    return keep


def view_dirs(views, c):
    """Unit vectors (ref frame) from the scan centroid towards each camera centre."""
    out = []
    for V in views:
        cc = np.linalg.inv(V["T"])[:3, 3]
        d = cc - c
        out.append(d / np.linalg.norm(d))
    return np.array(out)


def vis_rows(M, Rs, U):
    """Nearest precomputed view index per hypothesis and camera: (H, V)."""
    vo = np.einsum("hji,vj->hvi", Rs, U)  # R^T u
    return M["view_tree"].query(vo.reshape(-1, 3))[1].reshape(len(Rs), len(U))


def project(K, T, X):
    q = X @ T[:3, :3].T + T[:3, 3]
    z = q[..., 2]
    zs = np.where(z > 1e-3, z, 1e-3)
    return K[0, 0] * q[..., 0] / zs + K[0, 2], K[1, 1] * q[..., 1] / zs + K[1, 2], z


def proj_check(V, K, X, tol=0.015):
    """For ref-frame points X (..., 3): (valid, freespace-violation, inside-silhouette, agree)."""
    H, W = V["depth"].shape
    u, v, z = project(K, V["T"], X)
    ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z > 0.05)
    ui, vi = np.clip(ui, 0, W - 1), np.clip(vi, 0, H - 1)
    zo = V["depth"][vi, ui]
    occ = V["occ_d"][vi, ui]
    # inside the (dilated) mask/occluder silhouette, or outside it but depth-consistent
    inside = (V["sil"][vi, ui] | ((zo > 0) & (np.abs(z - zo) < tol))) & inb
    valid = inb & (zo > 0) & ~occ
    viol = valid & (z < zo - tol)
    agree = valid & (np.abs(z - zo) < 0.01)
    return inb, viol, inside, agree, valid, occ, zo, z


def coarse(M, scan, views, K, rots, seed=0, t_iters=6):
    """Vectorised scoring of rotation x translation-seed hypotheses -> (T (H,4,4), cost (H,)).
    Per rotation R the scan centroid c is matched to each translation seed p of the
    camera-visible mesh surface (visible centroid, k-means patch centres, midpoints):
    t = c - R p; then `t_iters` translation-only steps along the distance-field gradient
    (scan -> mesh, trimmed) absorb the remaining offset. Cost = truncated scan->mesh
    distance + free-space violations + silhouette/depth containment of the visible mesh."""
    rng = np.random.default_rng(seed)
    S = scan[rng.choice(len(scan), min(400, len(scan)), replace=False)]
    St = S[:200]
    c = scan.mean(0)
    U = view_dirs(views, c)
    J = vis_rows(M, rots, U)
    sd = M["seeds"][J[:, 0]]  # (H0, 1+k, 3) of the first view
    if len(views) > 1:  # union of the visible surfaces of all views for the centroid seed
        un = M["vis"][J].any(1).astype(np.float32)
        sd[:, 0] = (un @ M["pts"]) / np.maximum(un.sum(1, keepdims=True), 1)
    sd = np.concatenate([sd, 0.5 * (sd[:, :1] + sd[:, 1:])], 1)  # + midpoints
    ns = sd.shape[1]
    Rs = np.repeat(rots, ns, 0)
    Jr = np.repeat(J, ns, 0)
    t = c - np.einsum("hij,hj->hi", Rs, sd.reshape(-1, 3))
    cost = np.zeros(len(Rs))
    for a in range(0, len(Rs), 1000):
        b = slice(a, a + 1000)
        for _ in range(t_iters):
            So = np.einsum("hni,hij->hnj", St[None] - t[b, None], Rs[b])
            d, gr = df_lookup(M, So, grad=True)
            w = (d <= np.median(d, 1, keepdims=True) * 2 + 1e-4)[..., None]
            delta = d[..., None] * gr  # So - closest surface point
            t[b] += np.einsum("hij,hj->hi", Rs[b], (delta * w).sum(1) / np.maximum(w.sum(1), 1))
        So = np.einsum("hni,hij->hnj", S[None] - t[b, None], Rs[b])  # R^T (s - t)
        cost[b] = np.minimum(df_lookup(M, So), DF_TRUNC).mean(1) / 0.01
        fs, ins = 0, 0
        for iv, V in enumerate(views):
            P = M["pts"][M["vis_sub"][Jr[b, iv]]]  # (h, 150, 3)
            X = np.einsum("hij,hnj->hni", Rs[b], P) + t[b, None]
            inb, viol, inside, *_ = proj_check(V, K, X)
            fs = fs + viol.mean(1)
            ins = ins + inside.mean(1)
        cost[b] += 1.5 * fs / len(views) + 1.0 * (1 - ins / len(views))
    T = np.tile(np.eye(4), (len(Rs), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = Rs, t
    return T, cost


def icp_refine(M, scan, scan_tree, views, T0, K, iters=20):
    """Symmetric trimmed ICP. scan->mesh uses all scan points; mesh->scan only the visible
    mesh points that project into a view's object mask with valid depth (points outside a
    partial/occluded mask have no scan counterpart and would drag the pose)."""
    U = view_dirs(views, scan.mean(0))
    T = T0.copy()
    for _ in range(iters):
        R, t = T[:3, :3], T[:3, 3]
        So = (scan - t) @ R
        d1, i1 = M["tree"].query(So)
        k1 = np.argsort(d1)[: int(0.8 * len(d1))]
        k1 = k1[d1[k1] < max(3 * np.median(d1), 0.01)]
        J = vis_rows(M, R[None], U)[0]
        keep = np.zeros(len(M["pts"]), bool)
        for iv, V in enumerate(views):
            idx = np.nonzero(M["vis"][J[iv]])[0]
            X = M["pts"][idx] @ R.T + t
            u, v, z = project(K, V["T"], X)
            H, W = V["depth"].shape
            ui, vi = np.clip(np.rint(u).astype(int), 0, W - 1), np.clip(np.rint(v).astype(int), 0, H - 1)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & V["objm"][vi, ui] & (V["depth"][vi, ui] > 0)
            keep[idx[ok]] = True
        src, dst = [M["pts"][i1[k1]] @ R.T + t], [scan[k1]]
        if keep.sum() >= 30:
            Pv = M["pts"][keep] @ R.T + t
            d2, i2 = scan_tree.query(Pv)
            k2 = np.argsort(d2)[: int(0.7 * len(d2))]
            k2 = k2[d2[k2] < min(3 * np.median(d2), 0.01)]
            if len(k2) > 0.3 * len(k1):  # secondary term: never dominates scan->mesh
                k2 = k2[:: int(np.ceil(len(k2) / (0.3 * len(k1) + 1)))]
            src.append(Pv[k2])
            dst.append(scan[i2[k2]])
        src, dst = np.concatenate(src), np.concatenate(dst)
        if len(src) < 10:
            break
        dR, dt = kabsch(src, dst)
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = dR, dt
        T = D @ T
        if np.linalg.norm(dt) < 2e-5 and np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)) < 2e-4:
            break
    return T


def render_sil(M, K, T, shape, f=4):
    """Silhouette and z-buffer (inf = empty) of the mesh at 1/f resolution."""
    Kf = K.copy()
    Kf[:2] /= f
    u, v, z = project(Kf, T, M["dense"][::2])
    H, W = shape[0] // f, shape[1] // f
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.05)
    zb = np.full((H, W), np.inf, np.float32)
    np.minimum.at(zb, (v[ok].astype(int), u[ok].astype(int)), z[ok].astype(np.float32))
    zb = -cv2.dilate(-np.where(np.isfinite(zb), zb, 1e3).astype(np.float32), np.ones((3, 3), np.uint8))
    img = cv2.morphologyEx((zb < 1e3).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    zb = np.where(img & (zb >= 1e3), cv2.erode(zb, np.ones((5, 5), np.uint8)), zb)
    return img, zb


def full_score(M, scan, views, K, T):
    """Hypothesis score. iou = silhouette IoU where rendered pixels outside the mask count
    as object when the stereo depth there agrees with the rendered depth (robust to masks
    that miss object parts, e.g. a handle); occluder pixels are ignored."""
    R, t = T[:3, :3], T[:3, 3]
    d = M["tree"].query((scan - t) @ R)[0]
    inl = d < 0.005
    U = view_dirs(views, scan.mean(0))
    J = vis_rows(M, R[None], U)[0]
    ious, ious_m, expl, agr, pens, sils = [], [], [], [], [], []
    for iv, V in enumerate(views):
        X = M["pts"][M["vis"][J[iv]]] @ R.T + t
        inb, viol, inside, agree, valid, occ, zo, z = proj_check(V, K, X)
        nv = int(valid.sum())
        expl.append(1 - viol.sum() / max(nv, 1) if nv >= 20 else 0.5)
        agr.append(float(agree.sum() / max(nv, 1)) if nv >= 20 else 0.5)
        A, zb = render_sil(M, K, V["T"] @ T, V["depth"].shape)
        Mm, O, D4 = V["objm4"], V["occ4"], V["depth4"]
        sup = A & (D4 > 0) & (np.abs(zb - D4) < 0.015)
        Ms = Mm | sup
        ious.append(float((A & Ms & ~O).sum() / max(((A | Ms) & ~O).sum(), 1)))
        ious_m.append(float((A & Mm & ~O).sum() / max(((A | Mm) & ~O).sum(), 1)))
        # silhouette chamfer (1/4 res), normalised by the mask scale: rendered pixels far outside
        # the (depth-supported) mask and mask pixels far from the rendering are penalised
        if Ms.any() and A.any():
            dt_out = cv2.distanceTransform((~Ms).astype(np.uint8), cv2.DIST_L2, 3)
            dt_in = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, 3)
            a_, m_ = A & ~O, Mm & ~O
            e = 0.5 * ((dt_out[a_].mean() if a_.any() else 0) + (dt_in[m_].mean() if m_.any() else 0))
            sils.append(float(np.exp(-e / np.sqrt(max(Ms.sum(), 1)) / 0.03)))
        else:
            sils.append(0.0)
        if V.get("plane") is not None:  # plane in this view's camera frame
            n, d0 = V["plane"]
            Xc = M["pts"] @ (V["T"][:3, :3] @ R).T + V["T"][:3, :3] @ t + V["T"][:3, 3]
            pens.append(float(((Xc @ n + d0) < -0.015).mean()))
    rec = {"fitness": float(inl.mean()), "rmse_mm": float(np.sqrt((d[inl] ** 2).mean()) * 1000) if inl.any() else None,
           "iou": float(np.mean(ious)), "iou_mask": float(np.mean(ious_m)), "vis_expl": float(np.mean(expl)),
           "depth_agree": float(np.mean(agr)), "sil": float(np.mean(sils)),
           "pen": float(np.mean(pens)) if pens else 0.0}
    rec["score"] = (0.2 * rec["fitness"] + 0.15 * rec["vis_expl"] + 0.25 * rec["iou"] + 0.15 * rec["depth_agree"]
                    + 0.25 * rec["sil"]
                    - 0.5 * rec["pen"])
    return rec


def make_view(T, depth, obj, occ, plane=None):
    occ_d = cv2.dilate(occ.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    sil = cv2.dilate((obj | occ).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    H, W = depth.shape
    small = lambda m: cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST) > 0
    return {"T": T, "depth": depth, "occ_d": occ_d, "sil": sil, "objm": obj, "objm4": small(obj & ~occ),
            "occ4": small(occ_d), "plane": plane,
            "depth4": cv2.resize(depth, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)}


@lru_cache(maxsize=1)
def rotations():
    return Rotation.random(N_ROT, random_state=7).as_matrix()


def register(M, scan, views, K, extra=(), k=TOPK, n_final=12):
    """Global registration of the mesh to `scan` (ref frame; views[i]["T"] = cam_i-from-ref):
    coarse SO(3) scoring -> pose-NMS -> short ICP on a scan subsample for N_REFINE
    candidates -> score -> long ICP on the full scan for the best n_final -> re-score."""
    Ts, cost = coarse(M, scan, views, K, rotations())
    order = np.argsort(cost)
    cand = [Ts[i] for i in nms(M, Ts, order[:800], N_REFINE, dt=0.02, dr=20.0)]
    cand += [np.asarray(e[1]) for e in extra]
    srcs = ["global"] * (len(cand) - len(extra)) + [e[0] for e in extra]
    sub = scan[np.random.default_rng(1).choice(len(scan), min(1000, len(scan)), replace=False)]
    tsub = cKDTree(sub)
    ref = [icp_refine(M, sub, tsub, views, T, K, iters=12) for T in cand]
    sc = [full_score(M, sub, views, K, T)["score"] for T in ref]
    top = sorted(range(len(ref)), key=lambda i: -sc[i])[:n_final]
    top += [i for i in range(len(ref)) if srcs[i] != "global" and i not in top]
    tree = cKDTree(scan)
    pre = [ref[i] for i in top]
    ref = [icp_refine(M, scan, tree, views, T, K, iters=25) for T in pre]
    srcs = [srcs[i] for i in top]
    sc = [full_score(M, scan, views, K, T) for T in ref]
    for i, T in enumerate(pre):  # ICP must not make a hypothesis worse
        s0 = full_score(M, scan, views, K, T)
        if s0["score"] > sc[i]["score"]:
            ref[i], sc[i] = T, s0
    for s_, src in zip(sc, srcs):
        s_["src"] = src
    order = sorted(range(len(ref)), key=lambda i: -sc[i]["score"])
    keep = nms(M, ref, order, k)
    return [dict(T=ref[i], **sc[i]) for i in keep]


# ----------------------------------------------------------------------------- per-frame worker
def frame_task(args):
    run_dir, mask_dir, name, oid, others, frames, fp_hyp = args
    run = Run(run_dir, mask_dir)
    M = get_model(name)
    out = {}
    for t in frames:
        depth = run.depth(t)
        P, info = segment(run, oid, others, t, depth)
        rec = {"frame": t, "n_scan": 0 if P is None else len(P), "n_mask": info["n_mask"], "hyps": []}
        if P is not None:
            c = P.mean(0)
            rec["loc_raw"] = c.tolist()
            rec["loc_anchor"] = (c + c / np.linalg.norm(c) * M["depth_off"]).tolist()
            rec["touch_pts"] = touch_count(run, t, P, depth)
            V = make_view(np.eye(4), depth, info["obj"], info["occ"], info.get("plane"))
            extra = []
            fp = run.dir / "objects" / name / "poses_smoothed" / f"{t:06d}.json"
            if fp_hyp and fp.exists():
                extra = [("fp", load_T(fp))]
            hyps = register(M, P, [V], run.K, extra)
            rec["hyps"] = [{**{k: v for k, v in h.items() if k != "T"}, "T": h["T"].tolist()} for h in hyps]
            rec["plane"] = None if info.get("plane") is None else [*info["plane"][0].tolist(), float(info["plane"][1])]
            sub = P[np.random.default_rng(t).choice(len(P), min(1500, len(P)), replace=False)]
            rec["_pts"] = sub.astype(np.float32)
        out[t] = rec
    return name, out


def segment_task(args):
    """Register a static segment's world-fused cloud against several of its views."""
    run_dir, mask_dir, name, oid, others, frames, pts_w, scale = args
    run = Run(run_dir, mask_dir)
    M = get_model(name)
    P = pts_w
    # voxel downsample 3 mm + outlier/cluster cleaning
    _, first = np.unique(np.floor(P / 0.003).astype(np.int64), axis=0, return_index=True)
    P = P[first]
    dk = cKDTree(P).query(P, 9)[0][:, 1:].mean(1)
    P = P[dk < np.median(dk) + 3 * 1.4826 * np.median(np.abs(dk - np.median(dk))) + 1e-4]
    pairs = cKDTree(P).query_pairs(0.008, output_type="ndarray")
    if len(pairs):
        _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                                 shape=(len(P), len(P))), directed=False)
        P = P[lab == np.argmax(np.bincount(lab))]
    if len(P) > 6000:
        P = P[np.random.default_rng(0).choice(len(P), 6000, replace=False)]
    views = []
    for t in np.array(frames)[np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)]:
        t = int(t)
        _, info = segment(run, oid, others, t)
        Tcw = np.linalg.inv(run.world(t, scale))
        pl = info.get("plane")
        views.append(make_view(Tcw, info["depth"], info["obj"], info["occ"], pl))
    hyps = register(M, P, views, run.K, k=3)
    return name, frames, hyps, len(P)


# ----------------------------------------------------------------------------- step 4
def static_segments(ts, cw, touched, min_len=8, tol=0.03):
    segs, cur = [], []
    for t in ts:
        if touched[t]:
            if len(cur) >= min_len:
                segs.append(cur)
            cur = []
            continue
        if cur and (t - cur[-1] > 3 or np.linalg.norm(cw[t] - np.median([cw[s] for s in cur[-15:]], 0)) > tol):
            if len(cur) >= min_len:
                segs.append(cur)
            cur = []
        cur.append(t)
    if len(cur) >= min_len:
        segs.append(cur)
    return segs


def viterbi(M, states, touched, w_unary=20.0, cap_move=3.0, cap_static=8.0):
    """states: {t: [(T_world, score), ...]} -> {t: chosen index}. Unary w_unary*(1-score);
    pairwise = world centre change / sigma_t + rotation change / sigma_r (symmetry-aware),
    truncated at cap_move / cap_static so a sustained better unary can leave a wrong mode."""
    ts = sorted(t for t in states if states[t])
    if not ts:
        return {}
    cost = [np.array([w_unary * (1 - s) for _, s in states[ts[0]]])]
    back = []
    for a, b in zip(ts[:-1], ts[1:]):
        gap = b - a
        mv = any(touched.get(x, False) for x in range(a, b + 1))
        st, sr = (0.05, 30.0) if mv else (0.01, 10.0)
        st, sr = st * np.sqrt(gap), sr * np.sqrt(gap)
        Pa, Pb = states[a], states[b]
        pair = np.minimum(np.array([[np.linalg.norm(centre_of(M, Ta) - centre_of(M, Tb)) / st +
                          rot_dist(M, Ta[:3, :3], Tb[:3, :3]) / sr for Tb, _ in Pb] for Ta, _ in Pa]),
                          cap_move if mv else cap_static)
        tot = cost[-1][:, None] + pair
        back.append(tot.argmin(0))
        cost.append(tot.min(0) + np.array([w_unary * (1 - s) for _, s in Pb]))
    idx = [int(cost[-1].argmin())]
    for bp in back[::-1]:
        idx.append(int(bp[idx[-1]]))
    return dict(zip(ts, idx[::-1]))


def interp_world(chosen_w, n):
    """Fill frames without a pose: linear translation / SLERP rotation, ends held."""
    ts = np.array(sorted(chosen_w))
    out = {}
    if len(ts) == 0:
        return out
    Rs = Rotation.from_matrix([chosen_w[t][:3, :3] for t in ts])
    P = np.array([chosen_w[t][:3, 3] for t in ts])
    sl = Slerp(ts, Rs) if len(ts) > 1 else None
    for t in range(n):
        tc = np.clip(t, ts[0], ts[-1])
        T = np.eye(4)
        T[:3, :3] = sl([tc]).as_matrix()[0] if sl else Rs.as_matrix()[0]
        T[:3, 3] = [np.interp(tc, ts, P[:, i]) for i in range(3)]
        out[t] = T
    return out


def write_pose(d, t, T):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{t:06d}.json").write_text(json.dumps({"R": T[:3, :3].tolist(), "T": T[:3, 3].tolist()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--mask_dir", default="objects/masks", help="relative to the run (objects/masks_sam3 later)")
    ap.add_argument("--variant", default="stereo_global")
    ap.add_argument("--fp_hyp", type=int, default=0, help="1: add the FoundationPose pose as one extra hypothesis")
    ap.add_argument("--max_frames", type=int, default=0, help="debug: only the first N frames")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--no_multi", action="store_true", help="disable static-segment multi-frame registration")
    a = ap.parse_args()
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    runs = [Run(ROOT / "runs" / r, a.mask_dir) for r in a.runs]
    names = sorted({o["name"] for r in runs for o in r.objs})
    with ProcessPoolExecutor(min(ncpu, len(names))) as ex:
        list(ex.map(build_model, names))
    scales = {}
    with ProcessPoolExecutor(ncpu) as ex:
        for r, (s, info) in zip(runs, ex.map(estimate_slam_scale, runs)):
            scales[r.dir.name] = s
            print(f"{r.dir.name}: stereo-consistent SLAM scale {s:.3f} {info}", flush=True)
    tasks, meta = [], []
    for r in runs:
        n = r.n if not a.max_frames else min(r.n, a.max_frames)
        for o in r.objs:
            if a.objects and o["name"] not in a.objects:
                continue
            others = [x["id"] for x in r.objs if x["id"] != o["id"]]
            meta.append((r, o, others, n))
            for c in np.array_split(np.arange(n), max(1, n // 6)):
                tasks.append((str(r.dir), a.mask_dir, o["name"], o["id"], others, c.tolist(), a.fp_hyp))
    results = {}
    with ProcessPoolExecutor(ncpu) as ex:
        for (rd, _, name, *_), (_, out) in zip(tasks, ex.map(frame_task, tasks, chunksize=1)):
            results.setdefault((rd, name), {}).update(out)
    print("per-frame registration done", flush=True)
    # static segments -> fused multi-frame registration
    seg_tasks, seg_info = [], {}
    for r, o, others, n in meta:
        key, s = (str(r.dir), o["name"]), scales[r.dir.name]
        res = results[key]
        ts = [t for t in range(n) if res[t]["hyps"] and t in r.slam]
        cw = {t: (r.world(t, s) @ np.r_[res[t]["loc_anchor"], 1])[:3] for t in ts}
        touch = {t: res[t].get("touch_pts", 0) >= 20 for t in range(n)}
        near = {t: any(touch.get(x, False) for x in range(t - 3, t + 4)) for t in range(n)}
        segs = [] if a.no_multi else static_segments(ts, cw, near)
        seg_info[key] = (near, segs)
        for seg in segs:
            Pw = np.concatenate([res[t]["_pts"] @ r.world(t, s)[:3, :3].T + r.world(t, s)[:3, 3] for t in seg])
            seg_tasks.append((str(r.dir), a.mask_dir, o["name"], o["id"], others, seg, Pw, s))
    seg_res = {}
    with ProcessPoolExecutor(ncpu) as ex:
        for (rd, *_), (name, frames, hyps, npts) in zip(seg_tasks, ex.map(segment_task, seg_tasks, chunksize=1)):
            seg_res.setdefault((rd, name), []).append((frames, hyps, npts))
    # Viterbi + outputs
    for r, o, others, n in meta:
        key, s, name = (str(r.dir), o["name"]), scales[r.dir.name], o["name"]
        M = get_model(name)
        res = results[key]
        near, segs = seg_info[key]
        seg_of = {}
        for si, (frames, hyps, npts) in enumerate(seg_res.get(key, [])):
            for t in frames:
                seg_of[t] = (si, hyps[0])
        # re-score the segment pose in each of its frames with the per-frame score (comparable unary)
        for t, (si, h) in seg_of.items():
            depth = r.depth(t)
            P, info = segment(r, o["id"], others, t, depth)
            if P is None:
                continue
            Tc = np.linalg.inv(r.world(t, s)) @ h["T"]
            V = make_view(np.eye(4), depth, info["obj"], info["occ"], info.get("plane"))
            sc = full_score(M, P, [V], r.K, Tc)
            sc.update(src=f"segment{si}", seg_score=h["score"], score=sc["score"] + 0.05)
            res[t]["hyps"].insert(0, {**sc, "T": Tc.tolist()})
        states = {t: [(r.world(t, s) @ np.array(h["T"]), h["score"]) for h in res[t]["hyps"]]
                  for t in range(n) if res[t]["hyps"] and t in r.slam}
        ch = viterbi(M, states, near)
        chosen_w = {t: states[t][i][0] for t, i in ch.items()}
        full = interp_world(chosen_w, n)
        # robust SE(3) smoothing of the Viterbi path (static while untouched, soft-L1 unaries)
        smooth, status = None, None
        if len(chosen_w) >= 3:
            meas = [(t, chosen_w[t], max(states[t][ch[t]][1], 0.05) ** 2) for t in ch]
            static_w = np.array([0.0 if near.get(t, False) or near.get(t + 1, False) else 1.0 for t in range(n - 1)])
            smooth, info_s = fuse_trajectory({"global": meas}, n, params={"static_w": static_w, "sym_axis": M["axis"]})
            status = info_s["status"]
        odir = r.dir / "objects" / name / a.variant
        odir.mkdir(parents=True, exist_ok=True)
        vdir, vraw = r.dir / a.variant / name / "poses", r.dir / f"{a.variant}_raw" / name / "poses"
        for t in range(n):
            rec = {k: v for k, v in res[t].items() if k != "_pts"}
            rec.update(chosen=ch.get(t), touched=bool(near.get(t, False)),
                       segment=seg_of[t][0] if t in seg_of else None)
            (odir / f"{t:06d}.json").write_text(json.dumps(rec))
            if t in full and t in r.slam:
                Tinv = np.linalg.inv(r.world(t, s))
                write_pose(vraw, t, Tinv @ full[t])
                write_pose(vdir, t, Tinv @ (smooth[t] if smooth is not None else full[t]))
        rep = {"slam_scale": s, "mask_dir": a.mask_dir, "n_frames": n, "n_with_hyps": len(states),
               "sym_axis": None if M["axis"] is None else M["axis"].tolist(), "sym_err_mm": float(M["sym_err"]) * 1000,
               "segments": [{"frames": [f[0], f[-1]], "n": len(f), "n_pts": npts, "score": h[0]["score"] if h else None,
                             "iou": h[0]["iou"] if h else None} for f, h, npts in seg_res.get(key, [])],
               "chosen_src": {src: sum(1 for t, i in ch.items() if res[t]["hyps"][i]["src"] == src)
                              for src in {res[t]["hyps"][i]["src"] for t, i in ch.items()}},
               "smooth_status": status, "mean_chosen_score": float(np.mean([res[t]["hyps"][i]["score"] for t, i in ch.items()])) if ch else None}
        (odir.parent / f"{a.variant}_report.json").write_text(json.dumps(rep, indent=1))
        print(r.dir.name, name, json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
