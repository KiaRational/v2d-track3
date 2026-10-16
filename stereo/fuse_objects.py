"""Fuse FoundationPose + stereo-ICP object poses into one stable trajectory per object.

Both measurement streams are moved to the SLAM world frame (the ego camera moves; the
objects mostly do not), fused with se3_fusion.fuse_trajectory, and written back in
the camera frame, in the layout eval_objects.py reads for non-FP variants:
  <run>/<variant>/<name>/poses/NNNNNN.json  {"R": 3x3, "T": 3}
Object symmetry (symmetry.py) is removed from the measurements first: a flat board is
unchanged by a 180 deg flip and a cup by any spin about its axis, so estimators flip and
spin them at random while they sit still; each measurement (FP, ICP, global, merged and
time-sorted) is mapped to the symmetry-equivalent rotation closest to one shared running
reference before anything else runs.
Variants: icp (gated raw stereo-ICP); fused_free (FP + ICP + smoothness); fused (+ the
hand gate: zero world velocity while no hand touches the object); fused_sym (fused,
spin about the mesh's symmetry axis left to smoothness, if the mesh is near-symmetric);
fused_grasp (fused + a persistent grasp transform: over each stretch where one hand holds
the object, the object's pose in that hand's frame is a single unknown, fixed by the frames
where the object is clearly visible and carried through the frames a hand hides it);
fused_hand (fused + hand-follow: while a stereo-triangulated hand (Stage 6b) touches the
object, the object moves rigidly with that hand's palm, i.e. stays fixed in the hand frame).
Contact is 3D: stereo points of the hand masks within 2 cm of the object at its FP pose.
FP weight falls with occlusion (--vis_weight, on by default): (w_t, w_r) = (vis, vis^2),
vis = the object's visible mask area over its 90th-percentile area. Under a hand the tracker
loses rotation first, so its rotation is discounted faster than its position; the object's
orientation is then carried by smoothness and, while held, by the hand-follow term.
Stereo-ICP weight: w_icp * fitness^2, zero if fitness < 0.5, rmse > 4 mm, < 300 scan
points or the ICP moved > 5 cm from its FP initialisation.
fused_global (--w_global W): fused + the FP-free stereo global registration (global_register.py).
Usage (host env, CPU): python fuse_objects.py <run> [...] [--w_icp 4]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import trimesh

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from object_icp import HAND_IDS, decode_depth, load_T
from se3_fusion import fuse_trajectory
from symmetry import canonicalise, mesh_symmetry
from motion_quality import apply_symmetry, debounce_phases, table_disk_inside

ROOT = Path(os.environ["V2D_ROOT"])
OBJ = os.environ.get("V2D_OBJ_DIR", "objects")  # segmentation output to read
# variants get a suffix per segmentation source, so v1 and SAM 3 results coexist
SUF = "" if OBJ == "objects" else "_" + OBJ.replace("objects_", "")


def _rot_w(w):
    return float(w if np.isscalar(w) else w[-1])


def canon_measurements(meas, sym, ref_floor=0.1):
    """Remove symmetry-induced flips/spin from measurements [(t, T, w), ...] or
    {source: [(t, T, w), ...]} -> same layout, number changed (per source for a dict).

    All sources are canonicalised in ONE time-sorted pass against a shared running
    reference, per measurement (duplicates at one frame each keep their own group element),
    so FP, ICP and global cannot settle on different symmetry branches. Within a frame the
    most rotation-trusted measurement goes first; only measurements with rotation weight
    >= ref_floor * the median move the reference (an occluded, spinning FP must not drag it)."""
    single = not isinstance(meas, dict)
    srcs = {"_": meas} if single else meas
    group = sym["group"]
    if len(group) <= 1:
        return (meas, 0) if single else (meas, {k: 0 for k in srcs})
    flat = [(k, i, t, T, w) for k, v in srcs.items() for i, (t, T, w) in enumerate(v)]
    flat.sort(key=lambda q: (q[2], -_rot_w(q[4])))
    floor = ref_floor * np.median([_rot_w(q[4]) for q in flat]) if flat else 0.0
    out = {k: list(v) for k, v in srcs.items()}
    ch, ref = {k: 0 for k in srcs}, None
    for k, i, t, T, w in flat:
        R = canonicalise([T[:3, :3]], group, ref=T[:3, :3] if ref is None else ref)[0][0]
        if ref is None or _rot_w(w) >= floor:
            ref = R
        g = T[:3, :3].T @ R
        ch[k] += int(not np.allclose(g, np.eye(3), atol=1e-9))
        out[k][i] = (t, apply_symmetry(T, g, sym.get("center", np.zeros(3))), w)
    return (out["_"], ch["_"]) if single else (out, ch)


def damping_weights(near, vis, held, segs, observed, fps, verts=None, plane=None, cont_axis=None,
                    still_speed=0.004, w_damp=1.0, w_still=1.0, window=2):
    """Per-transition world-velocity damping (length n-1) -> (static_w, still, rest).

    free  (no hand near either frame) -- gated by `rest`: both frames supported by the table
          (plane {"up", "height"}, mesh verts at the observed pose within 1 cm) and slow
          observed motion, the median over +-window transitions of symmetry-canonicalised
          FP poses `observed` {t: world 4x4} with vis > 0.2 (continuous symmetry: axis
          tilt only). No visible hand alone is not evidence that the object is at rest.
    occl  w_damp * (1 - vis) while not held: the tracker is blind, acceleration-only
          smoothness would let a resting object drift/spin. Not gated.
    still w_still * exp(-(v/still_speed)^2) while held by a still hand (lifted or not). Not gated.
    No plane: warn and leave free ungated (the pre-support-test behaviour)."""
    n = len(vis)
    free = (~near[:-1] & ~near[1:]).astype(float)
    occl = (1.0 - np.minimum(vis[:-1], vis[1:])) * (~held[:-1] & ~held[1:])
    still = np.zeros(n - 1)
    for g in segs:
        fr = g["frames"]
        P = np.array([g["H"][t][:3, 3] for t in fr])
        for i in range(len(fr) - 1):
            lo, hi = max(0, i - 2), min(len(fr) - 1, i + 3)
            v = np.linalg.norm(P[hi] - P[lo]) / max(fr[hi] - fr[lo], 1)  # smoothed speed, m/frame
            if fr[i + 1] == fr[i] + 1:
                still[fr[i]] = np.exp(-(v / still_speed) ** 2)
    if plane is None or verts is None:
        warnings.warn("no table plane: free-object damping is not gated by a support/rest test")
        rest = np.ones(n - 1)
    else:
        up, height = np.asarray(plane["up"], float), float(plane["height"])
        supported = np.zeros(n, bool)
        for t, W in observed.items():
            if 0 <= t < n:
                Vw = verts @ W[:3, :3].T + W[:3, 3]  # on the top: lowest point at the plane AND over the (round) table
                supported[t] = abs(np.min(Vw @ up) - height) < 0.01 and bool(table_disk_inside(Vw.mean(0), plane))
        speed, angle = np.full(n - 1, np.nan), np.full(n - 1, np.nan)
        for t in range(n - 1):
            if t in observed and t + 1 in observed and vis[t] > 0.2 and vis[t + 1] > 0.2:
                W0, W1 = observed[t], observed[t + 1]
                speed[t] = np.linalg.norm(W1[:3, 3] - W0[:3, 3]) * fps
                c = (W0[:3, :3] @ cont_axis) @ (W1[:3, :3] @ cont_axis) if cont_axis is not None \
                    else (np.trace(W0[:3, :3].T @ W1[:3, :3]) - 1) / 2
                angle[t] = np.arccos(np.clip(c, -1, 1)) * fps
        rest = np.zeros(n - 1)
        for t in range(n - 1):
            win = slice(max(0, t - window), t + window + 1)
            if supported[t] and supported[t + 1] and np.isfinite(speed[win]).any():
                rest[t] = np.exp(-(np.nanmedian(speed[win]) / 0.04) ** 2 - (np.nanmedian(angle[win]) / 0.5) ** 2)
    static_w = np.maximum.reduce([free * rest, w_damp * occl, w_still * still])
    return static_w, still, rest


def contact_and_visibility(run, obj_id, name, n, K, fp_dir, radius=0.02, min_pts=15, window=2):
    """Per frame: 3D hand-object contact (>= min_pts stereo-depth points of a SAM2 hand
    mask lie within `radius` of the object surface placed at its FP pose) and the
    object's visible-area fraction (mask area / its 90th-percentile area)."""
    mesh = trimesh.load(ROOT / "meshes_obj" / name / f"{name}.obj", force="mesh")
    pts = trimesh.sample.sample_surface(mesh, 3000, seed=0)[0]
    touch, area = np.zeros(n, bool), np.zeros(n)
    rng = np.random.default_rng(0)
    for t in range(n):
        mo = cv2.imread(str(run / f"{OBJ}/masks/{obj_id}/{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mo is not None:
            area[t] = (mo > 0).sum()
        pf = fp_dir / f"{t:06d}.json"
        dp = run / f"depth_stereo/{t:06d}.png"
        if not pf.exists() or not dp.exists():
            continue
        depth = decode_depth(dp)
        hm = np.zeros(depth.shape, bool)
        for hid in HAND_IDS:
            m = cv2.imread(str(run / f"{OBJ}/masks/{hid}/{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                hm |= m > 0
        v, u = np.nonzero(hm & (depth > 0))
        if len(u) < min_pts:
            continue
        if len(u) > 5000:
            sel = rng.choice(len(u), 5000, replace=False)
            u, v = u[sel], v[sel]
        z = depth[v, u]
        H = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        T = load_T(pf)
        d, _ = cKDTree(pts @ T[:3, :3].T + T[:3, 3]).query(H, distance_upper_bound=radius)
        touch[t] = np.isfinite(d).sum() >= min_pts
    near = np.convolve(touch.astype(float), np.ones(2 * window + 1), "same") > 0
    vis = np.clip(area / max(np.percentile(area[area > 0], 90), 1) if (area > 0).any() else area, 0, 1)
    return near, vis, touch


PALM = [0, 5, 9, 13, 17]  # wrist + finger MCPs: the (nearly) rigid part of the hand


def stereo_hands(run, n):
    """Stereo-triangulated hands (Stage 6b): per side joints (n,21,3) camera frame,
    per-joint confidence (n,21) and the palm's rigid pose H (n,4,4) camera<-palm, with
    the palm template = the median-shaped frame of the sequence."""
    f = run / "stereo/hands_tri.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    out = {}
    for side in ("left", "right"):
        J, C = np.full((n, 21, 3), np.nan), np.zeros((n, 21))
        for t, X in d.get(side, {}).get("X_smooth", {}).items():
            if int(t) < n:
                J[int(t)] = X
                c = d[side].get("conf", {}).get(t)
                C[int(t)] = c if c is not None else 1.0
        ok = np.isfinite(J[:, 0, 0])
        if ok.sum() < 5:
            continue
        P = J[ok][:, PALM] - J[ok][:, PALM].mean(1, keepdims=True)
        tmpl = np.median(P, 0)
        H = np.full((n, 4, 4), np.nan)
        for t in np.nonzero(ok)[0]:
            Q = J[t, PALM]
            U, _, Vt = np.linalg.svd(tmpl.T @ (Q - Q.mean(0)))
            R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            H[t] = np.eye(4)
            H[t, :3, :3], H[t, :3, 3] = R, Q.mean(0)
        out[side] = {"J": J, "C": C, "H": H}
    return out


def hand_follow(hands, fp_dir, name, slam, n, radius=0.025, min_joints=2):
    """Per frame the hand whose joints touch the object (at its FP pose), and follow
    transitions (t, D = H_{t+1} H_t^-1 in the world, weight = palm confidence)."""
    mesh = trimesh.load(ROOT / "meshes_obj" / name / f"{name}.obj", force="mesh")
    tree_pts = trimesh.sample.sample_surface(mesh, 3000, seed=0)[0]
    side_at = [tuple() for _ in range(n)]
    for t in range(n):
        pf = fp_dir / f"{t:06d}.json"
        if not pf.exists() or not hands:
            continue
        T = load_T(pf)
        tree = cKDTree(tree_pts @ T[:3, :3].T + T[:3, 3])
        touching = []
        for side, h in hands.items():
            if not np.isfinite(h["J"][t, 0, 0]):
                continue
            d, _ = tree.query(h["J"][t])
            k = int(((d < radius) & (h["C"][t] > 0.25)).sum())
            if k >= min_joints:
                touching.append(side)
        side_at[t] = tuple(touching)
    follow = []
    for t in range(n - 1):
        if t not in slam or t + 1 not in slam:
            continue
        shared = set(side_at[t]) & set(side_at[t + 1])
        for s in sorted(shared):
            H = hands[s]["H"]
            if not (np.isfinite(H[t, 0, 0]) and np.isfinite(H[t + 1, 0, 0])):
                continue
            Hw0, Hw1 = slam[t] @ H[t], slam[t + 1] @ H[t + 1]
            w = float(min(hands[s]["C"][t, PALM].mean(), hands[s]["C"][t + 1, PALM].mean())) / len(shared)
            follow.append((t, Hw1 @ np.linalg.inv(Hw0), w))
    return side_at, follow


def grasp_segments(hands, side_at, slam, n, min_len=4):
    """Contiguous stretches held by one hand -> a grasp segment with that hand's world palm
    poses. Each segment gets one rigid object-in-hand transform in the fuser."""
    segs = []
    for sd in hands:
        t = 0
        while t < n:
            if sd not in side_at[t]:
                t += 1
                continue
            u = t
            while u + 1 < n and sd in side_at[u + 1]:
                u += 1
            H = hands[sd]["H"]
            fr = [k for k in range(t, u + 1) if k in slam and np.isfinite(H[k, 0, 0])]
            if len(fr) >= min_len:
                segs.append({"frames": fr, "H": {k: slam[k] @ H[k] for k in fr}, "side": sd,
                             "w": float(np.median([hands[sd]["C"][k, PALM].mean() / len(side_at[k]) for k in fr]))})
            t = u + 1
    return segs


def observed_positions(meas, n, win=1):
    """Per-frame robust world position of the object from its own observations (canonical FP / ICP /
    global centres, weight-median per frame, then a +-win median over time); NaN where unobserved."""
    P = np.full((n, 3), np.nan)
    by = {}
    for src in meas.values():
        for t, W, w in src:
            if 0 <= t < n:
                by.setdefault(t, []).append((W[:3, 3], max(w) if isinstance(w, tuple) else w))
    for t, v in by.items():
        X, w = np.array([x for x, _ in v]), np.array([x for _, x in v])
        if w.sum() > 0:
            P[t] = [X[np.argsort(X[:, k])][np.searchsorted(np.cumsum(w[np.argsort(X[:, k])]), w.sum() / 2), k] for k in range(3)]
    Q = P.copy()
    for t in range(n):
        s = P[max(0, t - win):t + win + 1]
        s = s[np.isfinite(s[:, 0])]
        if len(s) and np.isfinite(P[t, 0]):
            Q[t] = np.median(s, 0)
    return Q





def carry_segments(segs, P, move=0.01, lead=1, ctx=5):
    """Contact is not carrying: a holding hand moves (grip closing, lifting) before the object leaves its
    support and after it is put down, and a rigid object-in-hand transform over the whole contact drags the
    object with the palm too early / too late. Trim each grasp segment to where the object's OWN observations
    have left the pose they had at the start (onset) / reach the pose they end at (release), with `lead`
    frames of margin. P: one (n,3) track (observed_positions) or {source: track}; with several sources a
    frame counts as displaced only if EVERY source observed there (and with a rest pose) is displaced > `move`
    from its own rest pose -- consensus, because a hand covering the object biases a single source (stereo-ICP
    reads the pitcher 1-2 cm high once fingers touch it). Unobserved frames never decide; a segment whose
    observations never show the object moving is dropped (held, not carried); one without observations is
    kept unchanged (cannot tell)."""
    tracks = P if isinstance(P, dict) else {"all": P}
    out, log = [], []
    for g in segs:
        fr = g["frames"]
        s0, s1 = fr[0], fr[-1]
        rest0, rest1 = {}, {}
        for k, X in tracks.items():
            pre, post = X[max(0, s0 - ctx):s0 + 2], X[max(0, s1 - 1):s1 + ctx + 1]
            pre, post = pre[np.isfinite(pre[:, 0])], post[np.isfinite(post[:, 0])]
            if len(pre):
                rest0[k] = np.median(pre, 0)
            if len(post):
                rest1[k] = np.median(post, 0)

        def displaced(t, rest):
            d = [np.linalg.norm(tracks[k][t] - r) > move for k, r in rest.items() if np.isfinite(tracks[k][t, 0])]
            return None if not d else all(d)
        on_s = [displaced(t, rest0) for t in fr]
        off_s = [displaced(t, rest1) for t in fr]
        if all(x is None for x in on_s) or all(x is None for x in off_s):
            out.append(g); log.append({"side": g["side"], "from": s0, "to": s1, "carry": [s0, s1], "decided": False})
            continue
        on = next((t for t, x in zip(fr, on_s) if x), None)
        off = next((t for t, x in zip(reversed(fr), reversed(off_s)) if x), None)
        if on is None or off is None or off < on:
            log.append({"side": g["side"], "from": s0, "to": s1, "carry": None, "decided": True})
            continue
        a_, b_ = max(s0, on - lead), min(s1, off + lead)
        keep = [t for t in fr if a_ <= t <= b_]
        if len(keep) >= 2:
            out.append({**g, "frames": keep, "H": {t: g["H"][t] for t in keep}})
        log.append({"side": g["side"], "from": s0, "to": s1, "carry": [a_, b_] if len(keep) >= 2 else None, "decided": True})
    return out, log


def retime(meas_list, shift, slam):
    """Resample stereo measurements to colour time: the stereo stream at colour frame t shows the scene at
    t + shift frames, so the colour-time pose at t is the measurement at t - shift, interpolated (lerp /
    slerp in the world) between the two neighbouring measured frames; weight = the smaller of the two."""
    if not shift:
        return meas_list
    by = {t: (W, w) for t, W, w in meas_list}
    out = []
    for t in sorted(slam):
        q = t - shift
        t0 = int(np.floor(q))
        if t0 not in by or t0 + 1 not in by:
            # sequence ends / a missing neighbour: the nearer existing measurement (< 1 frame away) at half
            # weight -- dropping it would e.g. leave frame 0 (the leaderboard's alignment frame) without ICP
            k = min((x for x in (t0, t0 + 1) if x in by), key=lambda x: abs(x - q), default=None)
            if k is not None and abs(k - q) < 1:
                W_, w_ = by[k]
                out.append((t, W_, tuple(0.5 * x for x in w_) if isinstance(w_, tuple) else 0.5 * w_))
            continue
        al = q - t0
        (W0, w0), (W1, w1) = by[t0], by[t0 + 1]
        W = np.eye(4)
        W[:3, 3] = (1 - al) * W0[:3, 3] + al * W1[:3, 3]
        W[:3, :3] = Slerp([0, 1], Rotation.from_matrix([W0[:3, :3], W1[:3, :3]]))(al).as_matrix()
        w = tuple(min(x, y) for x, y in zip(w0, w1)) if isinstance(w0, tuple) else min(w0, w1)
        out.append((t, W, w))
    return out


def write(dirp, t, T):
    dirp.mkdir(parents=True, exist_ok=True)
    (dirp / f"{t:06d}.json").write_text(json.dumps({"R": T[:3, :3].tolist(), "T": T[:3, 3].tolist()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--w_icp", type=float, default=4.0)
    ap.add_argument("--icp_time_shift", default="0",  # off by default: no net gain (09-30 test)
                    help="frames the stereo stream (depth -> stereo-ICP) leads the colour video; auto = -sync_offset_frac "
                         "of stereo/hands_tri.json (Stage 6b), 0 disables")
    ap.add_argument("--carry_move_mm", type=float, default=0.0,  # off by default: trimming grasps lost occlusion support (09-30 test)
                    help="a grasp only moves the object once its own observations moved this far (0: whole contact)")
    ap.add_argument("--spline_weight", type=float, default=0.2,
                    help="Soft cubic pose prior; 0 disables it. Raw observations and grasp constraints stay active")
    ap.add_argument("--spline_knot_seconds", type=float, default=0.15)
    ap.add_argument("--fp_source", default="poses_smoothed", help="poses (raw FP) or poses_smoothed (EKF)")
    ap.add_argument("--still_speed", type=float, default=0.004,
                    help="palm speed (m/frame, smoothed) below which a holding hand counts as still")
    ap.add_argument("--w_still", type=float, default=1.0, help="damping of a held object while its hand is still")
    ap.add_argument("--w_damp", type=float, default=1.0,
                    help="strength of velocity damping while the object is occluded but not held")
    ap.add_argument("--vis_weight", type=int, default=1,
                    help="1: FP weight falls with occlusion -- (w_t, w_r) = (vis, vis^2). A hand over the "
                         "object costs the tracker its rotation long before its position, and that is "
                         "exactly when it spins a resting object.")
    ap.add_argument("--w_global", type=float, default=0.0,
                    help=">0: add stereo global-registration poses (global_register.py, the Viterbi-chosen "
                         "hypothesis per frame, score >= 0.75 and IoU >= 0.6, weight w_global*score^2) as a "
                         "third source -> extra variant fused_global (fused + this source)")
    ap.add_argument("--global_variant", default="stereo_global",
                    help="with --w_global: which global_register.py variant to read (e.g. stereo_global_v2 -> fused_global_v2)")
    ap.add_argument("--overrides", default=None,
                    help="refinement-agent overrides JSON {object: {fp_mult: [[a, b, m]], icp_mult: [[a, b, m]], "
                         "icp_extra: <dir under the object dir, e.g. stereo_icp_agent>, static: [[a, b, w]]}}")
    ap.add_argument("--out_variant", default=None, help="write only fused_all, to this variant dir (agent trials)")
    ap.add_argument("--only_objects", nargs="*", default=None)
    ap.add_argument("--only_global", action="store_true",
                    help="with --w_global: only compute fused_global (existing variants/icp/report untouched)")
    a = ap.parse_args()
    for r in a.runs:
        run = ROOT / "runs" / r
        fps = float(json.loads((run / "episode.json").read_text())["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source fps")
        slam = {}
        for p in sorted((run / "recon_hawor/slam_poses").glob("*.json")):
            slam[int(p.stem)] = load_T(p)
        n = max(slam) + 1
        ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
        hands = stereo_hands(run, n)
        if a.icp_time_shift == "auto":
            ht = run / "stereo/hands_tri.json"
            icp_shift = -float(json.loads(ht.read_text()).get("sync_offset_frac") or 0.0) if ht.exists() else 0.0
        else:
            icp_shift = float(a.icp_time_shift)
        report = {}
        ovr = json.loads(Path(a.overrides).read_text()) if a.overrides else {}
        for o in json.loads((run / OBJ / "objects.json").read_text()):
            name = o["name"]
            if a.only_objects and name not in a.only_objects:
                continue
            ov = ovr.get(name, {})
            od = run / OBJ / name
            cache = od / f".contact_vis_{a.fp_source}.npz"  # expensive (reads every depth frame), trial-independent
            if cache.exists() and cache.stat().st_mtime > (run / OBJ / "objects.json").stat().st_mtime:
                c_ = np.load(cache)
                near, vis, touch = c_["near"], c_["vis"], c_["touch"]
            else:
                near, vis, touch = contact_and_visibility(run, o["id"], name, n, K, od / a.fp_source)
                np.savez(cache, near=near, vis=vis, touch=touch)
            side_at, follow = hand_follow(hands, od / a.fp_source, name, slam, n)
            touch_h = np.array([bool(x) for x in side_at])
            near = near | (np.convolve(touch_h.astype(float), np.ones(5), "same") > 0)
            segs = grasp_segments(hands, side_at, slam, n)
            held = np.zeros(n, bool)
            for g in segs:
                held[g["frames"]] = True
            fp, icp = [], []
            for p in sorted((od / a.fp_source).glob("*.json")):
                t = int(p.stem)
                if t in slam and vis[t] > 0.05:
                    v = float(vis[t])
                    fp.append((t, slam[t] @ load_T(p), (v, v ** 2) if a.vis_weight else 1.0))
            for p in sorted((od / "stereo_icp").glob("*.json")):
                t, d = int(p.stem), json.loads(p.read_text())
                if t not in slam or "T" not in d:
                    continue
                ok = (d["fitness"] >= 0.5 and d["rmse_mm"] is not None and d["rmse_mm"] <= 4.0
                      and d["n_scan"] >= 300 and d["moved_from_init_mm"] <= 50)
                if ok:
                    icp.append((t, slam[t] @ np.array(d["T"]), a.w_icp * d["fitness"] ** 2))
            extra = ov.get("icp_extra") or []
            for ed in ([extra] if isinstance(extra, str) else extra):  # agent: extra stereo-ICP from the current trajectory
                for p in sorted((od / ed).glob("*.json")):
                    t, d = int(p.stem), json.loads(p.read_text())
                    if t in slam and "T" in d and d["fitness"] >= 0.4 and d["rmse_mm"] is not None \
                            and d["rmse_mm"] <= 5.0 and d["n_scan"] >= 200:
                        icp.append((t, slam[t] @ np.array(d["T"]), a.w_icp * d["fitness"] ** 2))

            def mult(meas_list, rules):
                out = []
                for t, T_, w_ in meas_list:
                    m_ = 1.0
                    for s0, s1, f_ in rules:
                        if s0 <= t <= s1:
                            m_ *= f_
                    w2 = tuple(x * m_ for x in w_) if isinstance(w_, tuple) else w_ * m_
                    out.append((t, T_, w2))
                return out
            fp, icp = mult(fp, ov.get("fp_mult", [])), mult(icp, ov.get("icp_mult", []))
            icp = retime(icp, icp_shift, slam)  # stereo time -> colour time (FP / masks / SLAM are colour)
            fp = [m_ for m_ in fp if max(m_[2]) > 0] if fp and isinstance(fp[0][2], tuple) else [m_ for m_ in fp if m_[2] > 0]
            if not fp and not icp:
                continue
            for t, W, _ in ([] if (a.only_global or a.out_variant) else icp):
                write(run / ("icp" + SUF) / name / "poses", t, np.linalg.inv(slam[t]) @ W)
            meas = {"fp": fp, "icp": icp}
            if a.w_global > 0:
                sg = []
                for p in sorted((od / a.global_variant).glob("*.json")):
                    t, d = int(p.stem), json.loads(p.read_text())
                    if t not in slam or d.get("chosen") is None:
                        continue
                    h = d["hyps"][d["chosen"]]
                    if h["score"] >= 0.75 and h["iou"] >= 0.6:
                        sg.append((t, slam[t] @ np.array(h["T"]), a.w_global * h["score"] ** 2))
                meas["global"] = sg
            sym = mesh_symmetry(ROOT / "meshes_obj" / name / f"{name}.obj")
            # one merged, time-sorted pass with a shared reference: all sources on one branch
            meas, n_ch = canon_measurements(meas, sym)
            fp, icp = meas["fp"], meas["icp"]
            # contact is not carrying: the grasp transform (and hand-follow) only once the object itself moves
            carry_log = []
            if a.carry_move_mm > 0 and segs:
                segs, carry_log = carry_segments(segs, {k: observed_positions({k: v}, n) for k, v in (("fp", fp), ("icp", icp)) if v}, a.carry_move_mm / 1000)
                held = np.zeros(n, bool)
                for g in segs:
                    held[g["frames"]] = True
                carried = {(g["side"], t) for g in segs for t in g["frames"]}
                follow = [f_ for f_ in follow if any((sd, f_[0]) in carried and (sd, f_[0] + 1) in carried for sd in hands)]
            axis = sym["cont_axis"]
            # Velocity damping: `free` where no hand is near, gated by table support + slow
            # observed (canonical FP) motion -- no visible hand alone is not evidence of rest;
            # `occl` graded by occlusion while not held and `still` while a holding hand is
            # still, both ungated: exactly when the tracker is blind they must hold the object.
            plane = run / "table_plane.json"
            tp = json.loads(plane.read_text()) if plane.exists() else None
            if tp is None:
                print(f"WARNING {r} {name}: no table_plane.json, free damping not support-gated", flush=True)
            verts = trimesh.load(ROOT / "meshes_obj" / name / f"{name}.obj", force="mesh").vertices
            static_w, still, rest = damping_weights(
                near, vis, held, segs, {t: T_ for t, T_, _ in fp}, fps, verts, tp, axis,
                a.still_speed, a.w_damp, a.w_still)
            for s0, s1, w_ in ov.get("static", []):  # agent: hold the object still in these windows
                static_w[s0:s1] = np.maximum(static_w[s0:s1], w_)
            phases = debounce_phases(side_at, 3)  # spline spans: contact changes without flicker
            report[name] = {"n_fp": len(fp), "n_icp": len(icp), "symmetry": sym["kind"], "sym_group": len(sym["group"]),
                            "canonicalised": n_ch,
                            "sym_axis": None if axis is None else axis.tolist(),
                            "static_frac": float((static_w > 0.5).mean()), "still_frac": float((still > 0.5).mean()), "damp_mean": float(static_w.mean()),
                            "rest_frac": float((rest > 0.5).mean()), "table_plane": tp is not None,
                            "contact_frames": np.nonzero(touch)[0].tolist(),
                            "hand_contact": {sd: [t for t, x in enumerate(side_at) if sd in x] for sd in hands},
                            "n_follow": len(follow),
                            "hand_carry": {sd: sorted(t for g in segs if g["side"] == sd for t in g["frames"]) for sd in hands},
                            "carry_trim": carry_log, "icp_time_shift": icp_shift,
                            "grasp_segments": [{"side": g["side"], "n": len(g["frames"]),
                                                "from": g["frames"][0], "to": g["frames"][-1]} for g in segs]}
            variants = [("fused_free", {}), ("fused", {"static_w": static_w}),
                        ("fused_sym", {"static_w": static_w, "sym_axis": axis}),
                        ("fused_hand", {"static_w": static_w, "follow": follow}),
                        ("fused_grasp", {"static_w": static_w, "grasp": segs}),
                        # everything that constrains the pose at once: symmetry-canonicalised
                        # measurements, occlusion-graded damping and the grasp memory
                        ("fused_all", {"static_w": static_w, "sym_axis": axis, "grasp": segs})]
            if a.w_global > 0:
                report[name]["n_global"] = len(meas["global"])
                gname = "fused_global" + a.global_variant[len("stereo_global"):]
                variants = ([] if a.only_global else variants) + [(gname, {"static_w": static_w})]
            if a.out_variant:  # agent trial: only the full model, to its own directory
                variants = [(a.out_variant, {"static_w": static_w, "sym_axis": axis, "grasp": segs})]
            for variant, prm in variants:
                if (variant == "fused_sym" and axis is None) or \
                        (variant in ("fused_hand", "fused_grasp", "fused_all") and not hands):
                    continue
                poses, info = fuse_trajectory(meas if variant.startswith("fused_global") else {"fp": fp, "icp": icp},
                                              n, params={**prm, "fps": fps, "spline_weight": a.spline_weight,
                                                         "spline_knot_seconds": a.spline_knot_seconds,
                                                         "contact_phases": phases})
                vdir = run / variant / name / "poses" if a.out_variant else run / (variant + SUF) / name / "poses"
                for t in slam:
                    write(vdir, t, np.linalg.inv(slam[t]) @ poses[t])
                report[name][variant] = {"status": info["status"], "sources": info["sources"]}
            print(r, name, json.dumps(report[name]), flush=True)
        if not a.out_variant:  # agent trials never overwrite the pipeline's report (hand contacts live there)
            (run / (f"fusion_{a.global_variant}_report.json" if a.only_global else "fusion_report.json")).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
