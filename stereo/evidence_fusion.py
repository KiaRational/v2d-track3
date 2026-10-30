"""Recover gross pose errors from masked stereo depth, then fuse with palm motion.

No public meshes, mocap, object-specific orientation overrides, or GT inputs.
Writes a separate camera-frame pose variant; mesh body coordinates are unchanged.
Run with run_evidence_fusion.sbatch, never on a login node.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from object_icp import decode_depth, load_T, icp
from fuse_objects import stereo_hands
from se3_fusion import fuse_trajectory


def centroid_pose(T, center):
    out = T.copy()
    out[..., :3, 3] += out[..., :3, :3] @ center
    return out


def body_pose(T, center):
    out = T.copy()
    out[..., :3, 3] -= out[..., :3, :3] @ center
    return out


def angle(A, B):
    return np.linalg.norm(Rotation.from_matrix(np.swapaxes(A, -1, -2) @ B).as_rotvec(), axis=-1)


def robust_pose(T, weights):
    """Medoid-seeded SO(3) estimate; a minority of half-turns cannot cancel rotations."""
    weights = np.asarray(weights, float)
    if not len(T) or not np.isfinite(T).all() or weights.sum() <= 0:
        raise ValueError("Need finite poses and positive total confidence")
    R, p = T[:, :3, :3], T[:, :3, 3]
    cost = np.array([np.sum(weights * np.minimum(angle(r, R), 0.5)) for r in R])
    r = R[np.argmin(cost)].copy()
    q = np.median(p, axis=0)
    for _ in range(12):
        dr = Rotation.from_matrix(r.T @ R).as_rotvec()
        wr = weights / (1 + (np.linalg.norm(dr, axis=1) / 0.2) ** 2)
        wt = weights / (1 + (np.linalg.norm(p - q, axis=1) / 0.015) ** 2)
        r = r @ Rotation.from_rotvec(np.average(dr, axis=0, weights=wr)).as_matrix()
        q = np.average(p, axis=0, weights=wt)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, q
    return out


def hypotheses(T, center, axes):
    """24 proper principal-axis orientations about the mesh centroid, not its origin.

    These are hypotheses to score against observations, NEVER assumed symmetries.
    """
    c = T[:3, :3] @ center + T[:3, 3]
    out = [T.copy()]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            G = np.eye(3)[:, perm] * signs
            if np.linalg.det(G) < 0:
                continue
            H = T.copy()
            H[:3, :3] = T[:3, :3] @ axes @ G @ axes.T
            H[:3, 3] = c - H[:3, :3] @ center
            if not any(angle(H[:3, :3], h[:3, :3]) < 1e-6 for h in out):
                out.append(H)
    return out


def geometry_score(T, pts, normals, scan, scan_tree=None):
    """Bidirectional clipped surface distance. Units metres; all scan points count."""
    P = pts @ T[:3, :3].T + T[:3, 3]
    facing = ((normals @ T[:3, :3].T) * P).sum(1) < 0
    P = P[facing]
    if len(P) < 20 or not len(scan):
        return 1.
    d = cKDTree(P).query(scan)[0]
    e = (scan_tree if scan_tree is not None else cKDTree(scan)).query(P)[0]
    # Scan-to-model dominates: hidden model surfaces need not appear in the mask.
    return float(np.minimum(d, .04).mean() + .25 * np.minimum(e, .04).mean())


def recover_frame(T, alternatives, pts, normals, scan, center, axes):
    candidates = hypotheses(T, center, axes) + alternatives
    tree = cKDTree(scan)
    before = geometry_score(T, pts, normals, scan, tree)
    scores = [geometry_score(C, pts, normals, scan, tree) for C in candidates]
    refined = [(T, before)]
    for k in np.argsort(scores)[:4]:
        C, _, _ = icp(pts, normals, scan, candidates[k], max_iters=12)
        # Registration cannot move the object outside its observed neighbourhood.
        if np.linalg.norm((C[:3, :3] @ center + C[:3, 3]) - (T[:3, :3] @ center + T[:3, 3])) < .12:
            refined.append((C, geometry_score(C, pts, normals, scan, tree)))
    refined.sort(key=lambda x: x[1])
    best, score = refined[0]
    # Refuse large corrections without a material independent geometry improvement.
    changed = before - score > max(.001, .12 * before)
    return (best if changed else T), {"before_mm": before * 1000,
        "after_mm": (score if changed else before) * 1000, "accepted": bool(changed),
        "rotation_deg": float(np.degrees(angle(T[:3, :3], best[:3, :3]))) if changed else 0.}


def contiguous(mask, min_length=4):
    ix = np.flatnonzero(mask)
    return [p for p in np.split(ix, np.flatnonzero(np.diff(ix) != 1) + 1) if len(p) >= min_length]


def hand_segments(W, points, hands, slam, plane, reported, fps):
    """Recover missed carries using proximity plus airborne/co-motion evidence.

    Missing hand frames split segments. Touch alone does not imply a rigid grasp.
    Both hands may constrain a bimanual carry, with divided weights.
    """
    n = len(W)
    c = W[:, :3, 3]
    supported, airborne = np.zeros(n, bool), np.zeros(n, bool)
    if plane:
        up = np.asarray(plane["up"])
        vw = points @ np.swapaxes(W[:, :3, :3], 1, 2) + c[:, None]
        low = (vw @ up).min(1) - plane["height"]
        supported = np.abs(low) < .015
        airborne = low > .02
        if "center" in plane and "radius" in plane:
            d = c - np.array(plane["center"])
            supported &= np.linalg.norm(d - (d @ up)[:, None] * up, axis=1) < plane["radius"]
    out = []
    for side, hd in hands.items():
        H = slam @ hd["H"]
        valid = np.isfinite(H).all((1, 2))
        distance = np.full(n, np.inf)
        for t in np.flatnonzero(valid):
            joints = hd["J"][t] @ slam[t, :3, :3].T + slam[t, :3, 3]
            distance[t] = cKDTree(points @ W[t, :3, :3].T + c[t]).query(joints)[0].min()
        comove = np.zeros(n, bool)
        half = max(2, round(.15 * fps))
        for t in range(half, n - half):
            a, b = t - half, t + half
            if valid[a:b+1].all():
                vo, vh = c[b] - c[a], H[b, :3, 3] - H[a, :3, 3]
                comove[t] = np.linalg.norm(vo) > .015 and np.linalg.norm(vo-vh) < .5 * np.linalg.norm(vo)
        report = np.zeros(n, bool)
        report[[t for t in reported.get(side, []) if 0 <= t < n]] = True
        # A nearby stationary hand can hold an airborne object still; motion is not required.
        travel = np.array([np.max(np.linalg.norm(c[max(0,t-10):t+11]-c[t],axis=1)) for t in range(n)])
        mask = valid & (distance < .09) & (report | (airborne & (travel>.02)) | comove)
        mask = binary_closing(mask, structure=np.ones(3)) & valid & (distance < .10)
        for fr in contiguous(mask):
            out.append({"side": side, "frames": fr, "H": H})
    return out, supported


def refine(W, observations, segments, supported, fps, max_nfev=120, hand_weight=.5):
    """Robust world-frame SE(3) solve, with rigid relative-hand anchors and rest gating."""
    n = len(W)
    held = np.zeros(n, int)
    for g in segments:
        held[g["frames"]] += 1
    spacing = max(1., float(np.median(np.diff([t for t, _, _ in observations])))) if len(observations)>1 else 1.
    data = {"visual": [(t,T,w*spacing) for t,T,w in observations],
            "initial": [(t, T, (.08, .03)) for t, T in enumerate(W)]}
    predictions = []
    follow = []
    segment_reports = []
    by_frame = {t: (T, w) for t, T, w in observations}
    for g in segments:
        fr, H = g["frames"], g["H"]
        trusted = [t for t in fr if t in by_frame and by_frame[t][1] > .15]
        if len(trusted) < 3:
            # No independent anchor: preserve the source rather than inventing a grasp orientation.
            continue
        G = robust_pose(np.array([np.linalg.inv(H[t]) @ by_frame[t][0] for t in trusted]),
                        [by_frame[t][1] for t in trusted])
        for i, t in enumerate(fr):
            ramp = min(1., (i+1)/4, (len(fr)-i)/4)
            predictions.append((int(t), H[t] @ G, hand_weight * ramp / held[t]))
        for a, b in zip(fr[:-1], fr[1:]):
            follow.append((int(a), H[b] @ np.linalg.inv(H[a]), hand_weight / max(held[a], held[b])))
        segment_reports.append({"side": g["side"], "from": int(fr[0]), "to": int(fr[-1]), "anchors": len(trusted)})
    data["hand_anchor"] = predictions
    # Static damping needs support AND low observed motion, never merely an absent hand.
    step = np.linalg.norm(np.diff(W[:, :3, 3], axis=0), axis=1) * fps
    still = np.array([np.median(step[max(0,t-3):t+4]) < .04 for t in range(n-1)])
    static = supported[:-1] & supported[1:] & (held[:-1] == 0) & (held[1:] == 0) & still
    result, info = fuse_trajectory(data, n, params={"fps": fps, "max_nfev": max_nfev,
        "static_w": static.astype(float), "follow": follow, "sigma_ft": .003,
        "sigma_fr": np.deg2rad(1.5), "sigma_r": np.deg2rad(8.), "sigma_acc": .004,
        "sigma_racc": np.deg2rad(2.), "sym_axis": None})
    return result, {"solver": info, "segments": segment_reports, "static_edges": int(static.sum())}


def scan_frame(run, obj_id, other_ids, t, K):
    path = run / "depth_stereo" / f"{t:06d}.png"
    mpath = run / "objects_recon/masks" / str(obj_id) / path.name
    if not path.exists() or not mpath.exists():
        return None
    depth = decode_depth(path)
    mask = cv2.imread(str(mpath), 0) > 0
    for oid in other_ids:
        p = mpath.parent.parent / str(oid) / path.name
        if p.exists():
            mask &= cv2.imread(str(p), 0) == 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5,5), np.uint8)) > 0
    v, u = np.nonzero(mask & (depth > .05) & (depth < 3.))
    if len(u) < 100:
        return None
    sel = np.random.default_rng(t).choice(len(u), min(1200, len(u)), replace=False)
    u, v = u[sel], v[sel]
    z = depth[v, u]
    return np.stack([(u-K["cx"])/K["fx"]*z, (v-K["cy"])/K["fy"]*z, z], 1)


def metrics(W, center, fps):
    c = centroid_pose(W, center)[:, :3, 3]
    deg = np.degrees(angle(W[:-1, :3, :3], W[1:, :3, :3]))
    return {"max_rotation_step_deg": float(deg.max()), "p95_rotation_step_deg": float(np.percentile(deg,95)),
            "steps_over_25deg": int((deg>25).sum()),
            "p95_acceleration_m_s2": float(np.percentile(np.linalg.norm(np.diff(c,n=2,axis=0)*fps**2,axis=1),95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_bfix_recon")
    ap.add_argument("--out", default="fused_evidence_recon")
    ap.add_argument("--hands", default="hands_c")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max_nfev", type=int, default=120)
    ap.add_argument("--hand_weight", type=float, default=.5)
    ap.add_argument("--reuse_evidence", help="Reuse recorded observations with verified mesh hash; still re-solve and revalidate")
    ap.add_argument("--objects", nargs="*")
    a = ap.parse_args()
    if a.stride < 1 or a.max_nfev < 1 or a.out == a.src or Path(a.out).name != a.out:
        raise ValueError("Use a fresh output variant, positive stride/evaluation budget")
    root = Path(os.environ["V2D_ROOT"])
    for rn in a.runs:
        run = root / "runs" / rn
        dest = run / a.out
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite {dest}; choose a new --out")
        slam_map = {int(p.stem): load_T(p) for p in (run/"recon_hawor/slam_poses").glob("[0-9]*.json")}
        n = max(slam_map)+1
        if set(slam_map) != set(range(n)):
            raise ValueError("Camera gaps must be reconstructed before object refinement")
        slam = np.array([slam_map[t] for t in range(n)])
        fps = float(json.loads((run/"episode.json").read_text())["fps"])
        K = json.loads((run/"recon_hawor/intrinsics_stable.json").read_text())
        hands = stereo_hands(run, n, a.hands)
        plane = json.loads((run/"table_plane.json").read_text()) if (run/"table_plane.json").exists() else None
        report_path = run/"fusion_report_recon.json"
        reports = json.loads(report_path.read_text()) if report_path.exists() else {}
        objects = json.loads((run/"objects_recon/objects.json").read_text())
        reuse = json.loads((run/a.reuse_evidence/"evidence_report.json").read_text()) if a.reuse_evidence else None
        report = {"args": vars(a), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "fps": fps, "mesh_root": "meshes_recon", "objects": {}, "complete": False}
        for obj in objects:
            name = obj["name"]
            if a.objects and name not in a.objects:
                continue
            mesh_path = root/"meshes_recon"/name/f"{name}.obj"
            mesh = trimesh.load(mesh_path, force="mesh")
            center = np.asarray(mesh.centroid)
            pts, fi = trimesh.sample.sample_surface(mesh, 2400, seed=0)
            normals = mesh.face_normals[fi]
            _, axes = np.linalg.eigh(np.cov((pts-center).T))
            source = np.array([load_T(run/a.src/name/"poses"/f"{t:06d}.json") for t in range(n)])
            W = slam @ source
            if not np.isfinite(W).all():
                raise ValueError(f"Non-finite input poses: {name}")
            obs, evidence, scans = [], {}, {}
            print(f"{rn}/{name}: recovering masked-depth observations", flush=True)
            if reuse:
                rec = reuse["objects"][name]
                if (rec["mesh_sha256"] != hashlib.sha256(mesh_path.read_bytes()).hexdigest()
                        or reuse["fps"] != fps or reuse["args"]["src"] != a.src):
                    raise ValueError("Cached evidence mesh, timing or source mismatch")
                obs = [(r["frame"],centroid_pose(np.array(r["pose_world"]),center),r["weight"]) for r in rec["observations"]]
                evidence = {int(t): v for t,v in rec["evidence"].items()}
            for t in ([] if reuse else sorted(set(range(0,n,a.stride)) | {n-1})):
                scan = scan_frame(run,obj["id"],[2,3]+[o["id"] for o in objects if o["id"]!=obj["id"]],t,K)
                if scan is None:
                    continue
                scans[t] = scan
                alt = []
                for sub in ("poses", "poses_smoothed"):
                    p = run/"objects_recon"/name/sub/f"{t:06d}.json"
                    if p.exists():
                        alt.append(load_T(p))
                T, diagnostic = recover_frame(source[t],alt,pts,normals,scan,center,axes)
                quality = float(np.exp(-(diagnostic["after_mm"]/12.)**2))
                obs.append((t,centroid_pose(slam[t] @ T,center),quality))
                evidence[t] = diagnostic
            if not obs:
                raise ValueError(f"No valid masked stereo evidence for {name}")
            C = centroid_pose(W,center)
            segs, support = hand_segments(C,pts-center,hands,slam,plane,
                reports.get(name,{}).get("hand_carry", reports.get(name,{}).get("hand_contact",{})),fps)
            print(f"{rn}/{name}: optimizing {n} poses, {len(obs)} observations, {len(segs)} hand segments", flush=True)
            result, detail = refine(C,obs,segs,support,fps,a.max_nfev,a.hand_weight)
            result = body_pose(result,center)
            # Independent after-solve evidence gate: report failures, never promote silently.
            before, after = [], []
            for t in evidence:
                scan = scans.get(t)
                if scan is None:
                    scan = scan_frame(run,obj["id"],[2,3]+[o["id"] for o in objects if o["id"]!=obj["id"]],t,K)
                if scan is None:
                    raise ValueError(f"Cached evidence no longer has valid depth/masks: {name}, frame {t}")
                before.append(geometry_score(source[t],pts,normals,scan))
                after.append(geometry_score(np.linalg.inv(slam[t])@result[t],pts,normals,scan))
            detail.update(before=metrics(W,center,fps),after=metrics(result,center,fps),evidence=evidence,
                geometry_before_mm=float(np.mean(before)*1000),geometry_after_mm=float(np.mean(after)*1000),
                geometry_pass=bool(np.mean(after) <= np.mean(before)+.001),
                observations=[{"frame":t,"pose_world":body_pose(T,center).tolist(),"weight":w} for t,T,w in obs],
                mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest())
            od = dest/name/"poses"
            od.mkdir(parents=True)
            for t,T in enumerate(np.linalg.inv(slam)@result):
                (od/f"{t:06d}.json").write_text(json.dumps({"R":T[:3,:3].tolist(),"T":T[:3,3].tolist()}))
            report["objects"][name] = detail
            print(rn,name,json.dumps({k:v for k,v in detail.items() if k not in ("solver","evidence","observations")}),flush=True)
            (dest/"evidence_report.json").write_text(json.dumps(report,indent=2))
        report["complete"] = True
        (dest/"evidence_report.json").write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
