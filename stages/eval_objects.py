"""Evaluate object trajectories against Vicon mocap (public split).

For each variant (FoundationPose tracking, AGILE, AGILE + hand coupling, ...):
world pose = DROID-SLAM camera-to-world x object-in-camera pose. There is no
camera-to-mocap extrinsic, so one rigid transform (SE(3), s = 1: SLAM + stereo is metric;
--sim3 frees the scale) per episode maps our world frame onto the mocap frame, fitted robustly
(inlier-reweighted) on all objects' POSES (object-attached points, so orientation residuals enter;
see robust_align), so one failed object cannot distort the frame its neighbours are scored in.
A free-scale fit on positions only can collapse (s ~ 0.4 seen) and leaks alignment error into the
rotation error. gt_world.json (validate_critic, table_plane, render_4d_scene) always comes from the
robust SE(3) fit of the --align variant. Mocap poses are in each scan's own frame,
exactly like our meshes, so orientations compare directly.

Metrics per object: translation error (cm), rotation error (deg), ADD (mean model-
vertex distance, cm), jitter (mean |2nd difference| of position, mm/frame^2) vs the
mocap's own, and interaction-onset error vs mocap (first frame displaced > 1 cm).
"official" per variant: the kit's Track-3 protocol (metric_code/track_3/AUC.py, SP-SR.py, MP-SR.py):
rigid alignment of the first scored pose of object slot 0 only, ADD with fixed points (here: 2000
seeded mesh vertices; the kit's own samples are withheld), AUC over 1-9 cm, SP-SR, MP-SR; frames
with no estimate count as failures. Slot order = the parquet's observation.objects order.
Usage: python eval_objects.py <run_name> [--variants fp agile agile_hc ...] [--sim3]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ["V2D_ROOT"])
DATA = Path(os.environ["V2D_DATA"])
OBJ = os.environ.get("V2D_OBJ_DIR", "objects")  # segmentation output to read


def T_from_json(p, key_style):
    d = json.loads(Path(p).read_text())
    M = np.eye(4)
    if key_style == "agile":
        M[:3, :3] = np.array(d["R"]) * 1.0
        M[:3, 3] = d["T"]
        return M, np.array(d.get("scale", [1, 1, 1]))
    q = d["rotation"]  # toolkit Transform3d: w-first
    M[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    M[:3, 3] = d["translation"]
    return M, np.ones(3)


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum() if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


KP_RADIUS = 0.05  # object-attached alignment points: origin + 6 points 5 cm along +-x/y/z (the kit's MPPE points)
_KP = np.vstack([np.zeros(3), np.eye(3), -np.eye(3)]) * KP_RADIUS


def pose_points(M, sym="none", axis=None):
    """Object-attached points of a world pose (4x4) that are well defined under the object's symmetry:
    all 7 for an asymmetric object, origin +- r * axis for a continuous (rotational) symmetry, the
    origin only otherwise (discrete / unknown symmetry: the spin is ambiguous)."""
    if sym == "none" or sym is None:
        K = _KP
    elif sym == "continuous" and axis is not None:
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        K = np.vstack([np.zeros(3), a * KP_RADIUS, -a * KP_RADIUS])
    else:
        K = _KP[:1]
    return K @ M[:3, :3].T + M[:3, 3]


def robust_align(E, G, with_scale=False, syms=None, iters=5):
    """World -> mocap alignment fitted on full POSES, not positions only: each (estimate, mocap) pose
    pair contributes its object-attached points (pose_points), so orientation residuals enter the fit
    and position-only degeneracies (few, nearly collinear object positions) cannot rotate the frame.
    SE(3) by default (s = 1: SLAM + stereo is metric); with_scale=True gives Sim(3). Robust: pose pairs
    whose mean point residual exceeds max(3 x median, 2 cm) are dropped and the fit repeated, so one
    failed object cannot distort the frame its neighbours are scored in.
    E, G: sequences of 4x4 world poses; syms: per pair (kind, axis) or None. -> s, R, t, inlier mask."""
    syms = syms or [("none", None)] * len(E)
    A = [pose_points(e, *sy) for e, sy in zip(E, syms)]
    B = [pose_points(g, *sy) for g, sy in zip(G, syms)]
    keep = np.ones(len(A), bool)
    for _ in range(iters + 1):
        s, R, tr = umeyama(np.concatenate([a for a, k in zip(A, keep) if k]),
                           np.concatenate([b for b, k in zip(B, keep) if k]), with_scale)
        res = np.array([np.linalg.norm(s * a @ R.T + tr - b, axis=1).mean() for a, b in zip(A, B)])
        new = res <= max(np.median(res) * 3.0, 0.02)
        if new.sum() < max(10, 0.3 * len(res)) or (new == keep).all():
            break
        keep = new
    return s, R, tr, keep


# ---- official Track-3 protocol, replicated from the kit's metric_code/track_3/AUC.py (identical in
# SP-SR.py / MP-SR.py but for _METRIC): _warp_to_reference_start, _object_add, _add_auc,
# _spider_success, _maniptrans_success
KIT_ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
KIT_SP_POS_M, KIT_SP_ROT_RAD = 0.1, 0.5
KIT_MP_POS_M, KIT_MP_ROT_DEG = 0.03, 30.0


def kit_add_auc(add):
    """_add_auc for one object: fraction of frames with ADD < thr (strict) over 1..9 cm, trapezoid on [0, 1]."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    add = np.asarray(add, float)
    return float(trapezoid(np.array([np.mean(add < t) for t in KIT_ADD_THRESHOLDS_M]),
                           x=np.linspace(0.0, 1.0, KIT_ADD_THRESHOLDS_M.size)))


def kit_style_metrics(est, gt, verts):
    """Official-style scores of one episode (one evaluation world). est, gt: per object slot (slot 0
    first) arrays (n, 4, 4) of world poses, NaN where missing; verts: per slot (V, 3) fixed points.
    One rigid transform aligns the FIRST scored pose of slot 0 (first frame with mocap for slot 0) to
    the reference, then every object and frame with mocap is scored. A frame with no estimate is a
    failure (ADD / position error = inf; the real kit rejects a submission with missing rows). If the
    estimate is missing at slot 0's first scored frame, the first frame where both exist anchors the
    alignment (flagged)."""
    fin = lambda M: np.isfinite(M[:, 0, 0])
    g0, e0 = fin(gt[0]), fin(est[0])
    if not g0.any():
        return {"error": "slot 0 has no mocap"}
    t0 = int(np.argmax(g0))
    fallback = not e0[t0]
    if fallback:
        both = g0 & e0
        if not both.any():
            return {"error": "slot 0 has no frame with both estimate and mocap"}
        t0 = int(np.argmax(both))
    Ra = gt[0][t0, :3, :3] @ est[0][t0, :3, :3].T
    ta = gt[0][t0, :3, 3] - Ra @ est[0][t0, :3, 3]
    per, aucs, pos_m, rot_rad, pos_av, rot_av, miss = [], [], [], [], [], [], 0
    for E, G, V in zip(est, gt, verts):
        ts = np.nonzero(fin(G))[0]
        add, pe, re = np.full(len(ts), np.inf), np.full(len(ts), np.inf), np.full(len(ts), np.inf)
        for i, t in enumerate(ts):
            if not np.isfinite(E[t, 0, 0]):
                continue
            Re, pe_ = Ra @ E[t, :3, :3], Ra @ E[t, :3, 3] + ta
            dR, dp = Re - G[t, :3, :3], pe_ - G[t, :3, 3]
            add[i] = np.linalg.norm(V @ dR.T + dp, axis=1).mean()
            pe[i] = np.linalg.norm(dp)
            re[i] = np.arccos(np.clip((np.trace(G[t, :3, :3].T @ Re) - 1) / 2, -1, 1))
        ok = np.isfinite(add)
        miss += int((~ok).sum())
        aucs.append(kit_add_auc(add) if len(ts) else None)
        pos_m.append(float(pe.mean()) if len(ts) else np.inf)
        rot_rad.append(float(re.mean()) if len(ts) else np.inf)
        pos_av.append(float(pe[ok].mean()) if ok.any() else np.inf)
        rot_av.append(float(re[ok].mean()) if ok.any() else np.inf)
        per.append({"frames": int(len(ts)), "missing": int((~ok).sum()), "AUC": aucs[-1],
                    "ADD_cm_median": float(np.median(add) * 100) if len(ts) and np.isfinite(np.median(add)) else None,
                    "pos_err_cm_mean": float(pe[ok].mean() * 100) if ok.any() else None,
                    "rot_err_deg_mean": float(np.degrees(re[ok].mean())) if ok.any() else None})
    au = [x for x in aucs if x is not None]
    sp = lambda P, Q: bool(np.mean(P) <= KIT_SP_POS_M and np.mean(Q) <= KIT_SP_ROT_RAD)
    mp = lambda P, Q: bool(all(p < KIT_MP_POS_M and np.degrees(q) < KIT_MP_ROT_DEG for p, q in zip(P, Q)))
    return {"AUC": float(np.mean(au)) if au else None,
            "SP_SR": sp(pos_m, rot_rad), "MP_SR": mp(pos_m, rot_rad),           # missing frames = failure
            "SP_SR_available": sp(pos_av, rot_av), "MP_SR_available": mp(pos_av, rot_av),  # scored frames only
            "align_frame": t0, "align_fallback": fallback, "missing_frames": miss, "per_object": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variants", nargs="+", default=["fp", "agile", "agile_hc"])
    ap.add_argument("--out", default="eval_objects.json", help="report filename inside the run dir")
    ap.add_argument("--align", default="fp",
                    help="variant whose world->mocap SE(3) is used for ALL variants ('self' = per variant). "
                         "A shared alignment keeps one variant's errors from distorting another's frame.")
    ap.add_argument("--sim3", action="store_true", help="free the alignment scale (default SE(3), s = 1)")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    meta = json.loads((run / "episode.json").read_text())
    import pyarrow.parquet as pq  # host env only (the module's helpers import without it)
    table = pq.read_table(DATA / f"public/data/chunk-000/episode_{meta['episode']:06d}.parquet")
    frames = table.column("observation.objects").to_pylist()
    names = [o["name"] for o in frames[0]]
    n = len(frames)
    gt = {}
    for i, name in enumerate(names):
        Ms = np.full((n, 4, 4), np.nan)
        for t, f in enumerate(frames):
            if f[i]["visible"]:
                x, y, z, qw, qx, qy, qz = f[i]["pose"]
                Ms[t] = np.eye(4)
                Ms[t, :3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                Ms[t, :3, 3] = (x, y, z)
        gt[name] = Ms
    cams = {}
    for t in range(n):
        p = run / f"recon_hawor/slam_poses/{t:06d}.json"
        if p.exists():
            cams[t] = T_from_json(p, "toolkit")[0]

    report = {"episode": args.run, "task": meta["task"], "objects": names, "variants": {}, "align": args.align}
    # per-object symmetry (fusion_report.json): the spin of a symmetric object is not observable, so
    # only its well-defined points enter the alignment
    fr = json.loads((run / "fusion_report.json").read_text()) if (run / "fusion_report.json").exists() else {}
    syms = {nm: (fr[nm].get("symmetry", "none"), fr[nm].get("sym_axis")) for nm in names if isinstance(fr.get(nm), dict)}
    report["align_mode"] = "sim3_robust_pose" if args.sim3 else "se3_robust_pose"
    shared = None
    if args.align != "self":  # evaluate the alignment reference first
        args.variants = [args.align] + [v for v in args.variants if v != args.align]
    for variant in args.variants:
        est = {}
        for name in names:
            d = run / OBJ / name / "poses_smoothed" if variant == "fp" else run / variant / name / "poses"
            style = "toolkit" if variant == "fp" else "agile"
            Ms, sc = np.full((n, 4, 4), np.nan), np.ones(3)
            for t in range(n):
                p = d / f"{t:06d}.json"
                if p.exists() and t in cams:
                    M, sc = T_from_json(p, style)
                    Ms[t] = cams[t] @ M
            est[name] = (Ms, sc)
        idx = [(nm, t) for nm in names for t in range(n)
               if np.isfinite(est[nm][0][t, 0, 0]) and np.isfinite(gt[nm][t, 0, 0])]
        if len(idx) < 10:
            report["variants"][variant] = {"error": "no overlapping frames"}
            continue
        # Robust pose alignment (SE(3) unless --sim3): a single badly tracked object (a lost mask, a
        # track on the wrong object) would otherwise drag the whole episode's alignment and inflate
        # the error of its well-tracked neighbours. Inlier pose pairs only after a first pass.
        E = [est[nm][0][t] for nm, t in idx]
        G = [gt[nm][t] for nm, t in idx]
        sy = [syms.get(nm, ("none", None)) for nm, _ in idx]
        s, R, tr, keep = robust_align(E, G, with_scale=args.sim3, syms=sy)
        inl = float(keep.mean())
        if args.align != "self":
            if variant == args.align:
                shared = (s, R, tr, inl)
                # mocap ground truth expressed in OUR world frame (inverse of the robust SE(3), also
                # under --sim3: the reconstruction is metric), so visualisations / the critic's
                # validation / the table check can draw and compare it next to the reconstruction
                s_, R_, tr_ = (s, R, tr) if not args.sim3 else robust_align(E, G, with_scale=False, syms=sy)[:3]
                gtw = {}
                for nm in names:
                    gtw[nm] = {}
                    for t in range(n):
                        if np.isfinite(gt[nm][t, 0, 0]):
                            M = np.eye(4)
                            M[:3, :3] = R_.T @ gt[nm][t, :3, :3]
                            M[:3, 3] = R_.T @ (gt[nm][t, :3, 3] - tr_) / s_
                            gtw[nm][t] = M.round(6).tolist()
                (run / "gt_world.json").write_text(json.dumps(
                    {"sim3_world_to_mocap": {"s": float(s_), "R": R_.tolist(), "t": tr_.tolist()},
                     "align_mode": "se3_robust_pose", "align_variant": variant, "obj_dir": OBJ, "objects": gtw}))
            elif shared is not None:
                s, R, tr, inl = shared
        per_obj, Vs = {}, {}
        for nm in names:
            Ms, sc = est[nm]
            mesh = trimesh.load(ROOT / "meshes_obj" / nm / f"{nm}.obj", force="mesh")
            V = mesh.vertices[np.random.default_rng(0).choice(len(mesh.vertices), min(2000, len(mesh.vertices)), False)]
            Vs[nm] = V
            te, re, add, adds, pos_est, pos_gt = [], [], [], [], [], []
            for t in range(n):
                if not (np.isfinite(Ms[t, 0, 0]) and np.isfinite(gt[nm][t, 0, 0])):
                    continue
                Ra, pa = R @ Ms[t, :3, :3], s * R @ Ms[t, :3, 3] + tr
                Rg, pg = gt[nm][t, :3, :3], gt[nm][t, :3, 3]
                te.append(np.linalg.norm(pa - pg))
                re.append(np.degrees(np.arccos(np.clip((np.trace(Rg.T @ Ra) - 1) / 2, -1, 1))))
                Pe, Pg = (V * sc) @ Ra.T + pa, V @ Rg.T + pg
                add.append(np.linalg.norm(Pe - Pg, axis=1).mean())
                adds.append(cKDTree(Pg).query(Pe)[0].mean())  # ADD-S: symmetry-invariant (cups)
                pos_est.append(pa)
                pos_gt.append(pg)
            jit = lambda P: float(np.linalg.norm(np.diff(np.array(P), 2, axis=0), axis=1).mean() * 1000) if len(P) > 3 else None
            per_obj[nm] = {"frames": len(te), "trans_err_cm_mean": float(np.mean(te) * 100),
                           "trans_err_cm_median": float(np.median(te) * 100), "rot_err_deg_mean": float(np.mean(re)),
                           "rot_err_deg_median": float(np.median(re)), "ADD_cm_mean": float(np.mean(add) * 100),
                           "ADDS_cm_mean": float(np.mean(adds) * 100),
                           "jitter_mm": jit(pos_est), "gt_jitter_mm": jit(pos_gt),
                           "per_frame_trans_err_cm": [round(float(x) * 100, 2) for x in te]}
            if variant != "fp":
                pj = run / "agile" / nm / "prepare.json"
                if pj.exists():
                    iof = json.loads(pj.read_text())["iof"]
                    P = gt[nm][:, :3, 3]
                    ref = P[np.isfinite(P[:, 0])][0]
                    disp = np.linalg.norm(P - ref, axis=1)
                    onset = next((t for t in range(n) if np.isfinite(disp[t]) and disp[t] > 0.01), None)
                    per_obj[nm].update(iof=iof, mocap_onset=onset)
        # the kit's protocol: its own rigid alignment (first scored pose of slot 0), no mesh scale
        off = kit_style_metrics([est[nm][0] for nm in names], [gt[nm] for nm in names], [Vs[nm] for nm in names])
        off.update(slot_order=names, fixed_points="2000 mesh vertices (seed 0); kit samples withheld")
        report["variants"][variant] = {"sim3_scale": float(s), "align_mode": report["align_mode"],
                                       "align_inlier_frac": inl, "objects": per_obj, "official": off}
    out = run / args.out
    out.write_text(json.dumps(report, indent=2))
    def fmt(x, w=6, d=2):
        return f"{x:{w}.{d}f}" if isinstance(x, (int, float)) else f"{'n/a':>{w}s}"

    for v, r in report["variants"].items():
        for nm, m in r.get("objects", {}).items():
            print(f"{v:18s} {nm:18s} trans {fmt(m['trans_err_cm_mean'])} cm  "
                  f"rot {fmt(m['rot_err_deg_mean'])} deg  ADD {fmt(m['ADD_cm_mean'])} cm  "
                  f"ADD-S {fmt(m['ADDS_cm_mean'], 5)} cm  "
                  f"jitter {fmt(m['jitter_mm'], 4)} (gt {fmt(m['gt_jitter_mm'], 4)}) mm  "
                  f"iof {m.get('iof')} vs mocap {m.get('mocap_onset')}")
        o = r.get("official", {})
        if "AUC" in o:
            print(f"{v:18s} official-style: AUC {fmt(o['AUC'], 5, 3)}  SP-SR {int(o['SP_SR'])}  MP-SR {int(o['MP_SR'])}  "
                  f"(missing {o['missing_frames']} fr, align frame {o['align_frame']})  align s {fmt(r['sim3_scale'], 5, 3)}")


if __name__ == "__main__":
    main()
