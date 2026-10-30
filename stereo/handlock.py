"""Hand-locked object trajectories: a held object moves WITH the hand and never spins on its own; a hand-covered
object's (wrong) pose estimates are ignored. Runs after the fusion (+ branchfix), replaces physfix.

Phases per object (fusion_report grasp_segments / hand_contact):
  carried  grasp segment with a valid palm: object(t) = palm_s(t) @ G(t). palm_s = palm pose (hands variant, world)
           lightly smoothed (SIG_PALM frames, removes per-frame palm jitter without lag). G(t) = object-in-palm offset,
           a ROBUST WINDOWED estimate (Gaussian window SIG_G frames x per-frame reliability, Cauchy IRLS) over the
           relative poses palm^-1 @ measurement of FoundationPose (raw), gated stereo-ICP and, weakly, the source
           trajectory. Reliability = visibility (mask area / its usual area) -- frames where the hand covers the object
           (vis < VIS_MIN) get no weight, the window carries G through them. So the object can only turn relative to
           the hand as fast as G changes (slowly; e.g. a spoon re-gripped while stirring), never spin by itself.
           Dev check vs public mocap (beat1/diag/grasp_rigidity.py): ONE constant G per grasp is too rigid (our palm
           frame drifts ~2 cm vs the object over long grasps), a +-10..20-frame window is as accurate as the fusion.
  rest     no hand contact, object not moving (source centre within FREE_MOVE_M): ONE pose for the whole run --
           robust estimate from its well-visible frames (rotation: weighted medoid + IRLS chordal mean; position:
           weighted median), canonicalised to the source's symmetry branch.
  other    contact without a valid grasp (pushing, regrasp transitions), or a free run that moves: source poses.
Phase boundaries are blended over BLEND frames. Measurements are mapped to the source's branch of the mesh symmetry
group first (continuous cups: the spin about the axis is unobservable and follows the source).
Writes <run>/<out>/<obj>/poses (camera<-object) + handlock_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python handlock.py <run>... [--src fused_bfix_recon]
       [--out fused_hlock_recon] [--sig_g 15] [--hands hands_c]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fuse_objects import stereo_hands
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
VIS_MIN, SIG_PALM, BLEND, MIN_SEG = 0.4, 1.5, 4, 4
VIS_FLOOR = 0.1      # hand-covered frames keep a little weight in src mode (0.01 vs 1 for a fully visible frame)
FREE_MOVE_M = 0.03
AIR_M, HAND_MAX_M, PRE_LIFT = 0.02, 0.06, 8   # airborne height, max hand-object distance, pre-lift grasp frames
STILL_M, STILL_DEG = 0.02, 10.0     # a grasp whose source trajectory stays within these is not a carry
W_SRC = 0.15            # weight of the source trajectory's own relative pose (fills frames with no measurement)
ROT_SCALE, TR_SCALE = 12.0, 0.015   # Cauchy scales (deg, m)


def ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def angs(Rs, R):
    return np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", Rs, R) - 1) / 2, -1, 1)))


def proj(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def wmedoid(Rs, w, cap=30.0):
    """Weighted medoid with capped angles: the rotation most measurements agree with (a minority of flipped
    FoundationPose frames cannot move it)."""
    tr = np.einsum("aij,bij->ab", Rs, Rs)
    A = np.minimum(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))), cap)
    return Rs[np.argmin(A @ w)]


def robust_pose(Rs, ts, w, R0=None, iters=6):
    """IRLS (Cauchy) rotation chordal mean + translation weighted mean, started from the weighted medoid."""
    w = np.asarray(w, float)
    if w.sum() <= 0:
        return None
    R = wmedoid(Rs, w) if R0 is None else R0
    t = np.average(ts, 0, w)
    t = ts[np.argmin(np.linalg.norm(ts - t, axis=1))]
    for _ in range(iters):
        wr = w / (1 + (angs(Rs, R) / ROT_SCALE) ** 2)
        wt = w / (1 + (np.linalg.norm(ts - t, axis=1) / TR_SCALE) ** 2)
        if wr.sum() <= 1e-9 or wt.sum() <= 1e-9:
            break
        R = proj(np.einsum("n,nij->ij", wr, Rs))
        t = np.average(ts, 0, wt)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def smooth_poses(P, sig):
    """Gaussian smoothing of a pose sequence (dict t -> 4x4, consecutive or not): local chordal mean / mean."""
    ts = np.array(sorted(P))
    A = np.array([P[t] for t in ts])
    out = {}
    for i, t in enumerate(ts):
        w = np.exp(-0.5 * ((ts - t) / sig) ** 2)
        k = w > 1e-3
        M = np.eye(4)
        M[:3, :3] = proj(np.einsum("n,nij->ij", w[k], A[k, :3, :3]))
        M[:3, 3] = np.average(A[k, :3, 3], 0, w[k])
        out[int(t)] = M
    return out


def runs_of(lab, n):
    out, s = [], 0
    for t in range(1, n + 1):
        if t == n or lab[t] != lab[s]:
            out.append((lab[s], s, t))
            s = t
    return out


def interp_poses(P, n):
    """Fill missing frames of {t: 4x4} by slerp / lerp between the nearest valid ones (held at the ends)."""
    ts = sorted(P)
    out = {}
    for t in range(n):
        if t in P:
            out[t] = P[t]
            continue
        lo = max((u for u in ts if u < t), default=None)
        hi = min((u for u in ts if u > t), default=None)
        if lo is None or hi is None:
            out[t] = P[lo if hi is None else hi]
            continue
        al = (t - lo) / (hi - lo)
        M = np.eye(4)
        M[:3, :3] = Slerp([0, 1], Rotation.from_matrix([P[lo][:3, :3], P[hi][:3, :3]]))(al).as_matrix()
        M[:3, 3] = (1 - al) * P[lo][:3, 3] + al * P[hi][:3, 3]
        out[t] = M
    return out


def air_lock(final, F, vis, hands, slam, n, tp, mesh_dir, nm):
    """User rule: an object in the AIR is rigidly in the hand, with the grasp it had just before lift-off.
    Airborne = lowest mesh point of the source pose > AIR_M above the table (median-filtered). Per airborne run the
    holding hand = the one whose joints are closest to the object surface (median over the run, < HAND_MAX_M) -- no
    reliance on the contact detector, which misses grasps (e03 pot: 16 of its carried frames). The grasp G (object in
    smoothed palm frame) = robust vis-weighted mean of palm^-1 @ pose over the PRE_LIFT frames before lift-off (plus the
    first airborne frames), then object(t) = palm(t) @ G for the whole run; blended over BLEND frames at both ends.
    Modifies `final` in place; returns the runs [from, to, side, deg_change_at_liftoff]."""
    import trimesh
    from scipy.ndimage import median_filter
    from scipy.spatial import cKDTree
    up, h = np.asarray(tp["up"], float), float(tp["height"])
    V = np.asarray(trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh").vertices)
    V = V[:: max(1, len(V) // 1500)]
    low = np.array([float(((V @ T[:3, :3].T + T[:3, 3]) @ up).min() - h) for T in F])
    air = median_filter((low > AIR_M).astype(int), size=5, mode="nearest").astype(bool)
    runs = []
    for kind, s, e in runs_of(list(air), n):
        if not kind or e - s < 3:
            continue
        if max(np.linalg.norm(F[t][:3, 3] - F[s][:3, 3]) for t in range(s, e)) < STILL_M:
            continue    # off the table but not moving: resting on another object (lid on pot), not carried
        best = None
        for side, hd in hands.items():
            ds = []
            for t in range(s, e):
                if np.isfinite(hd["J"][t]).all():
                    Jw = hd["J"][t] @ slam[t][:3, :3].T + slam[t][:3, 3]
                    ds.append(cKDTree(V @ F[t][:3, :3].T + F[t][:3, 3]).query(Jw)[0].min())
            if len(ds) >= max(2, (e - s) // 3) and (best is None or np.median(ds) < best[1]):
                best = (side, float(np.median(ds)))
        if best is None or best[1] > HAND_MAX_M:
            runs.append([s, e - 1, None, None])
            continue
        side = best[0]
        H = hands[side]["H"]
        lo_, hi_ = max(0, s - PRE_LIFT), min(n, s + 3)
        valid = {t: slam[t] @ H[t] for t in range(max(0, s - PRE_LIFT - 5), min(n, e + 5)) if np.isfinite(H[t]).all()}
        if len(valid) < 2:
            runs.append([s, e - 1, side, None])
            continue
        Hw = interp_poses(smooth_poses(valid, SIG_PALM), n)
        fr = [t for t in range(lo_, hi_) if t in valid]
        if not fr:
            fr = [t for t in range(s, e) if t in valid][:5]
        Gs = np.array([np.linalg.inv(Hw[t]) @ final[t] for t in fr])
        w = np.array([max(float(vis[t]), VIS_FLOOR) ** 2 for t in fr])
        G = robust_pose(Gs[:, :3, :3], Gs[:, :3, 3], w)
        before = final[s].copy()
        locked = {t: Hw[t] @ G for t in range(s, e)}
        orig = {t: final[t].copy() for t in range(max(0, s - BLEND), min(n, e + BLEND))}
        for t, M in locked.items():
            final[t] = M
        for t0, t1, toward in ((max(0, s - BLEND), s, "in"), (e, min(n, e + BLEND), "out")):
            for t in range(t0, t1):
                w_ = (t - t0 + 1) / (t1 - t0 + 1)
                A, B = (orig[t], locked[s]) if toward == "in" else (locked[e - 1], orig[t])
                M = np.eye(4)
                M[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(w_).as_matrix()
                M[:3, 3] = (1 - w_) * A[:3, 3] + w_ * B[:3, 3]
                final[t] = M
        runs.append([s, e - 1, side, round(ang(before[:3, :3], final[s][:3, :3]), 1)])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_bfix_recon")
    ap.add_argument("--out", default="fused_hlock_recon")
    ap.add_argument("--hands", default=os.environ.get("V2D_FUSE_HANDS", "hands_c"))
    ap.add_argument("--sig_g", type=float, default=15.0, help="frames: Gaussian window of the object-in-palm offset")
    ap.add_argument("--meas", choices=["src", "raw"], default="src",
                    help="src: smooth the source trajectory in the palm frame (vis-weighted); raw: re-estimate from raw "
                         "FoundationPose + stereo-ICP (dev: raw FP still carries flips the repair removed, e.g. the lid)")
    ap.add_argument("--no_air_lock", action="store_true", help="do not hard-lock airborne objects to the holding hand")
    ap.add_argument("--no_rest",action="store_true", help="keep the source poses on untouched runs")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        rep = json.loads((run / f"fusion_report{suf}.json").read_text())
        hands = stereo_hands(run, n, a.hands)
        tpp = run / "table_plane.json"
        tp = json.loads(tpp.read_text()) if tpp.exists() else None
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if nm not in rep or not isinstance(rep[nm], dict) or not src.is_dir():
                continue
            if any(not (src / f"{t:06d}.json").exists() for t in range(n)) or any(t not in slam for t in range(n)):
                print(f"{rn} {nm}: source incomplete, skipped", flush=True)
                continue
            F = [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]
            od = run / obj_dir / nm
            cv = next(iter(sorted(od.glob(".contact_vis_*.npz"))), None)
            vis = np.load(cv)["vis"][:n] if cv is not None else np.ones(n)
            group = [np.asarray(S) for S in mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")["group"]]
            # measurements (world), weighted by visibility; mapped to the source's symmetry branch
            meas = [[] for _ in range(n)]
            if a.meas == "src":   # the repaired trajectory itself, weighted by how well the object is seen
                for t in range(n):
                    meas[t].append((F[t], max(float(vis[t]), VIS_FLOOR) ** 2))
            for d, kind in ([] if a.meas == "src" else ((od / "poses", "fp"), (run / ("icp" + suf) / nm / "poses", "icp"))):
                for t in range(n):
                    p = d / f"{t:06d}.json"
                    if not p.exists() or vis[t] < VIS_MIN:
                        continue
                    M = slam[t] @ load_T(p)
                    if len(group) > 1:
                        S = min(group, key=lambda S: ang(M[:3, :3] @ S, F[t][:3, :3]))
                        M[:3, :3] = M[:3, :3] @ S
                    meas[t].append((M, float(vis[t]) ** 2 * (1.5 if kind == "icp" else 1.0)))
            contact = {t for v in rep[nm].get("hand_contact", {}).values() for t in v}
            lab = ["other" if t in contact else "rest" for t in range(n)]
            for g in rep[nm].get("grasp_segments", []):
                H = hands.get(g["side"], {}).get("H")
                if H is None:
                    continue
                for t in range(g["from"], min(g["to"] + 1, n)):
                    if np.isfinite(H[t]).all():
                        lab[t] = "carry:" + g["side"]
            # a grasp that does not move the object (hand resting on it on the table): the object rests
            for kind, s, e in runs_of(lab, n):
                if kind.startswith("carry:"):
                    pc = F[s][:3, 3]
                    if max(np.linalg.norm(F[t][:3, 3] - pc) for t in range(s, e)) < STILL_M and \
                            max(ang(F[t][:3, :3], F[s][:3, :3]) for t in range(s, e)) < STILL_DEG:
                        lab[s:e] = ["rest"] * (e - s)
            out = [T.copy() for T in F]
            info = {"carry_frames": 0, "rest_frames": 0, "rest_runs": [], "carry_runs": []}
            for kind, s, e in runs_of(lab, n):
                fr = list(range(s, e))
                if e - s < MIN_SEG:
                    continue
                if kind == "rest" and not a.no_rest:
                    pc = np.median([F[t][:3, 3] for t in fr], 0)
                    if max(np.linalg.norm(F[t][:3, 3] - pc) for t in fr) > FREE_MOVE_M:
                        continue        # moves without detected contact (missed contact): keep the source
                    Ms = [(M, w) for t in fr for M, w in meas[t]]
                    if len(Ms) < 3:
                        continue
                    Rs = np.array([M[:3, :3] for M, _ in Ms]); ts = np.array([M[:3, 3] for M, _ in Ms])
                    w = np.array([w for _, w in Ms])
                    if a.meas != "src":   # source rotations vote too (weakly): keeps the branch the global flip repair chose
                        Rs = np.concatenate([Rs, np.array([F[t][:3, :3] for t in fr])])
                        ts = np.concatenate([ts, np.array([F[t][:3, 3] for t in fr])])
                        w = np.concatenate([w, np.full(len(fr), W_SRC)])
                    P = robust_pose(Rs, ts, w)
                    for t in fr:
                        out[t] = P.copy()
                    info["rest_frames"] += len(fr)
                    info["rest_runs"].append([s, e - 1, round(ang(P[:3, :3], F[s][:3, :3]), 1)])
                elif kind.startswith("carry:"):
                    side = kind.split(":")[1]
                    Hw = smooth_poses({t: slam[t] @ hands[side]["H"][t] for t in fr}, SIG_PALM)
                    Hi = {t: np.linalg.inv(Hw[t]) for t in fr}
                    rel = []    # (t, G, w)
                    for t in fr:
                        if a.meas != "src":
                            rel.append((t, Hi[t] @ F[t], W_SRC))
                        rel += [(t, Hi[t] @ M, w) for M, w in meas[t]]
                    tt = np.array([x[0] for x in rel]); GG = np.array([x[1] for x in rel]); ww = np.array([x[2] for x in rel])
                    # segment-wide robust offset first: anchors every window on the majority branch
                    G0 = robust_pose(GG[:, :3, :3], GG[:, :3, 3], ww)
                    for t in fr:
                        k = np.abs(tt - t) <= 3 * a.sig_g
                        wk = ww[k] * np.exp(-0.5 * ((tt[k] - t) / a.sig_g) ** 2)
                        # a window full of hand-covered frames leans on the segment-wide offset
                        Gt = robust_pose(np.concatenate([GG[k, :3, :3], G0[None, :3, :3]]),
                                         np.concatenate([GG[k, :3, 3], G0[None, :3, 3]]),
                                         np.concatenate([wk, [0.5]]))
                        out[t] = Hw[t] @ Gt
                    info["carry_frames"] += len(fr)
                    info["carry_runs"].append([s, e - 1, side])
            # blend every phase boundary (no teleport): slerp/lerp between the poses BLEND frames either side
            final = [T.copy() for T in out]
            for _, s, _ in runs_of(lab, n)[1:]:
                lo, hi = max(0, s - BLEND), min(n - 1, s + BLEND - 1)
                A, B = out[max(lo - 1, 0)], out[min(hi + 1, n - 1)]
                if ang(A[:3, :3], B[:3, :3]) < 0.5 and np.linalg.norm(A[:3, 3] - B[:3, 3]) < 0.001:
                    continue
                sl = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))
                for t in range(lo, hi + 1):
                    al = (t - lo + 1) / (hi - lo + 2)
                    M = np.eye(4)
                    M[:3, :3] = sl(al).as_matrix()
                    M[:3, 3] = (1 - al) * A[:3, 3] + al * B[:3, 3]
                    final[t] = M
            if tp is not None and not a.no_air_lock:
                info["air_runs"] = air_lock(final, F, vis, hands, slam, n, tp, mesh_dir, nm)
            od_ = run / a.out / nm / "poses"
            od_.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ final[t]
                (od_ / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st = [ang(final[t + 1][:3, :3], final[t][:3, :3]) for t in range(n - 1)]
            info.update(max_step_deg=round(max(st), 1), jumps10=int(sum(x > 10 for x in st)),
                        src_jumps10=int(sum(ang(F[t + 1][:3, :3], F[t][:3, :3]) > 10 for t in range(n - 1))),
                        occluded_frames=int((vis < VIS_MIN).sum()))
            report[nm] = info
            print(rn, nm, json.dumps({k: v for k, v in info.items() if not k.endswith("_runs")}), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "handlock_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
