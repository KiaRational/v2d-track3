"""Our reconstruction -> the toolkit's portable result bundle (result/result.npz + mesh), host env.

This is the input of the official robot pipeline (reconstruction/modules/v2d_task_library_loader
ego_recon_loader -> ManoSharpaData -> robotic_grounding ego_recon_to_sharpa retarget -> FlashCHORD),
so everything downstream is NVIDIA's own code. Fields exactly as v2d_common/result_bundle.py:
  camera_to_world_transform   DROID-SLAM camera-to-world per frame
  object_*                    our fused object trajectory (camera frame -> world), rigid, scale 1
  hand_{left,right}_*         our hands (MANO betas / global_orient / 45-dof hand_pose, translation as
                              the wrist trans in the camera frame; left hands in the mirrored-right
                              convention the loader expects and corrects itself)
  camera_intrinsics           fx, fy, cx, cy
Objects: --objects_list all (default) | A,B exports the episode's objects in the REFERENCE order
(object slot of the pose metrics): evaluation episodes from the submission kit's
data/track_3_evaluation_objects.json, public episodes from the struct order of observation.objects in
the episode parquet (read with the kit's own v2dlb.trajectory.read_episode), else objects.json order.
One object -> the single-object bundle above, unchanged (also for --object NAME). Several objects -> a
multi-object bundle (read by the patched ego_recon loader, patches/ego_recon_loader.py):
  object_names                (B,) str, slot order;  object_count  B
  object_<i>_to_camera_transform / object_<i>_to_world_transform (N,4,4), object_<i>_is_valid (N,),
  object_<i>_scale            per slot i; object_* (no index) = slot 0, for single-object readers
  mesh_<i>.obj (+ mesh_<i>.mtl, mesh_<i>_<texture>) per slot; mesh.obj (+ material) = slot 0
  manifest.json "objects" lists slot, name, mesh and keys; "object_order" the order and its source.
Usage: python export_result_bundle.py <run> [--object NAME | --objects_list all|A,B] [--objects fused_agent]
       [--hands mano_fit_phys] [--out <dir>] [--strict] [--max_gap_s 0.15] [--no_fill_long]
       (default <run>/robot/<object or A__B>/outputs/result)
Hands: once a side has any observation no frame is written as zeros (the upstream loader ignores
is_valid): interior gaps of any length are filled in the world frame (wrist-joint lerp, wrist and
finger slerp), leading/trailing gaps hold the first/last observed world pose; all filled frames are
is_valid True and hand_*_is_interpolated True (--no_fill_long: fill only gaps <= --max_gap_s). A side
without observations stays zeros, is_valid False (loud warning). fps and timestamps are
written to result.npz and manifest.json (the patched loader's get_fps reads them).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rest_settle import mesh_verts_com, settle  # noqa: E402
from depenetrate import hold_static, lift_out_of_table, load_parts, snap_assembly  # noqa: E402

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
KIT = ROOT / "submission_kit/v2d_submission_kit"


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


MIRROR = np.array([1.0, -1.0, -1.0])  # left hands: mirrored-right params <-> native left (loader convention)
HAND_FOLLOW_M = 0.08  # wrist to object surface: a hand this close follows the object's export displacement
ASSEMBLY_OBJECTS = {"wooden_piece_1", "wooden_piece_2"}   # slotted pieces: assembly snap


def gap_runs(valid):
    """(first, last, kind) of each run of invalid frames; kind is 'interior', 'lead' or 'trail'.
    Empty when no frame is valid (nothing to fill from)."""
    out, t, n = [], 0, len(valid)
    if not np.any(valid):
        return out
    while t < n:
        if valid[t]:
            t += 1
            continue
        s = t
        while t < n and not valid[t]:
            t += 1
        out.append((s, t - 1, "lead" if s == 0 else "trail" if t == n else "interior"))
    return out


def interior_gaps(valid, max_missing):
    """(first, last) of each interior gap (valid frames on both sides) at most max_missing long."""
    return [(s, e) for s, e, k in gap_runs(valid) if k == "interior" and e - s + 1 <= max_missing]


def fill_gaps(orient, trans, finger, j0, valid, c2w, timestamps, max_missing, left=False, fill_long=True):
    """Fill hand gaps in the WORLD frame, then re-express them in each frame's camera frame.

    Interior gaps: the wrist joint p = R_c2w (J0 + T) + t_c2w is lerped (J0 = rest wrist, from the
    records' joints, so T is MANO transl with center_idx=None); the world wrist rotation
    R_c2w R(orient) and every finger axis-angle are slerped. Leading/trailing gaps hold the first/
    last observed WORLD pose. A moving camera therefore never moves the hand. Gaps longer than
    max_missing frames, and leading/trailing gaps, are only filled with fill_long.
    Returns (filled mask, counts); arrays are modified in place."""
    n = len(valid)
    filled = np.zeros(n, bool)
    counts = {"n_interpolated_short": 0, "n_interpolated_long": 0, "n_held_lead": 0, "n_held_trail": 0}
    C, tc = c2w[:, :3, :3].astype(np.float64), c2w[:, :3, 3].astype(np.float64)
    mir = MIRROR if left else 1.0

    def world(k):  # observed frame k -> (world wrist joint, world wrist rotation (native side))
        return C[k] @ (j0[k] + trans[k]) + tc[k], C[k] @ Rotation.from_rotvec(orient[k] * mir).as_matrix()

    def put(idx, p, Rg, J, fing):
        orient[idx] = Rotation.from_matrix(np.transpose(C[idx], (0, 2, 1)) @ Rg).as_rotvec() * mir
        trans[idx] = np.einsum("nji,nj->ni", C[idx], p - tc[idx]) - J
        j0[idx], finger[idx] = J, fing
        filled[idx] = True

    for s, e, kind in gap_runs(valid):
        idx = np.arange(s, e + 1)
        m = len(idx)
        short = kind == "interior" and m <= max_missing
        if not (short or fill_long):
            continue
        if kind == "interior":
            a, b = s - 1, e + 1
            al = (timestamps[idx] - timestamps[a]) / (timestamps[b] - timestamps[a])
            (pa, Ra), (pb, Rb) = world(a), world(b)
            Rg = Slerp([0, 1], Rotation.from_matrix(np.stack([Ra, Rb])))(al).as_matrix()
            p = (1 - al)[:, None] * pa + al[:, None] * pb
            J = (1 - al)[:, None] * j0[a] + al[:, None] * j0[b]
            fing = np.stack([Slerp([0, 1], Rotation.from_rotvec(finger[[a, b], k]))(al).as_rotvec()
                             for k in range(15)], 1)
            counts["n_interpolated_short" if short else "n_interpolated_long"] += m
        else:
            k = e + 1 if kind == "lead" else s - 1
            pk, Rk = world(k)
            p, Rg, J = np.tile(pk, (m, 1)), np.tile(Rk, (m, 1, 1)), np.tile(j0[k], (m, 1))
            fing = np.tile(finger[k], (m, 1, 1))
            counts["n_held_lead" if kind == "lead" else "n_held_trail"] += m
        put(idx, p, Rg, J, fing)
    return filled, counts


def spurious_hand_frames(run, hands, n, min_sep=0.04, max_isolated=3, mask_win=2):
    """Frames whose detection of a side is spurious -> {side: set(frames)}. Two cases, both seen as the
    one hand detected twice (ep24 frame 1: 'right' 1 cm from the left wrist, no right SAM 3 mask until
    frame 28, and the gap fill then dragged the right hand from the left hand's spot for 1.2 s):
      - duplicate: both wrists within min_sep (two hands cannot occupy the same place) -> drop the side
        without a SAM 3 mask near t (if both / neither have one, the side in the shorter observed run);
      - isolated: a run of <= max_isolated consecutive observations with no SAM 3 mask of that side
        within +-mask_win frames anywhere in the run."""
    wrist, mask = {}, {}
    for side in ("left", "right"):
        wrist[side] = {}
        for f in (run / "stereo" / hands / side).glob("*.json"):
            t = int(f.stem)
            if 0 <= t < n:
                j = json.loads(f.read_text()).get("joints")
                if j is not None:
                    wrist[side][t] = np.asarray(j[0], float)
        md = run / "stereo/hands_sam3/masks/a" / side
        has = np.zeros(n, bool)
        for t in range(n):
            has[t] = (md / f"{t:06d}.png").exists()
        mask[side] = np.array([has[max(0, t - mask_win):t + mask_win + 1].any() for t in range(n)])

    def run_len(side, t):
        a = b = t
        while a - 1 in wrist[side]:
            a -= 1
        while b + 1 in wrist[side]:
            b += 1
        return b - a + 1

    out = {"left": set(), "right": set()}
    for t in sorted(set(wrist["left"]) & set(wrist["right"])):
        if np.linalg.norm(wrist["left"][t] - wrist["right"][t]) >= min_sep:
            continue
        ml, mr = mask["left"][t], mask["right"][t]
        out["right" if ml and not mr else "left" if mr and not ml else
            min(("left", "right"), key=lambda s: run_len(s, t))].add(t)
    for side in ("left", "right"):
        ts = sorted(wrist[side])
        i = 0
        while i < len(ts):
            j = i
            while j + 1 < len(ts) and ts[j + 1] == ts[j] + 1:
                j += 1
            seg = ts[i:j + 1]
            if len(seg) <= max_isolated and not any(mask[side][t] for t in seg):
                out[side].update(seg)
            i = j + 1
    return out


def hand_arrays(run, hands, side, n, c2w, timestamps, max_missing, fill_long=True, reject=()):
    """Hand arrays in the schema of NVIDIA's writer (v2d_common/result_bundle.py). The upstream
    loader ignores is_valid, so once a side has any observation no frame is left as zeros (a
    zero-pose hand at the camera): gaps are filled in the world frame (fill_gaps) and marked in
    hand_*_is_interpolated. A side without observations stays zeros with is_valid False."""
    betas = np.zeros(10, np.float32)
    orient, trans = np.zeros((n, 3), np.float64), np.zeros((n, 3), np.float64)
    finger = np.zeros((n, 15, 3), np.float64)
    j0 = np.zeros((n, 3), np.float64)
    valid = np.zeros(n, bool)
    got = False
    for f in sorted((run / "stereo" / hands / side).glob("*.json")):
        t = int(f.stem)
        if not 0 <= t < n or t in reject:
            continue
        r = json.loads(f.read_text())
        if not got:
            betas[:] = r["mano"]["betas"]
            got = True
        orient[t] = r["mano"]["global_orient"]
        finger[t] = np.asarray(r["mano"]["hand_pose"]).reshape(15, 3)
        trans[t] = r["T"]  # MANO with center_idx=None: verts = layer(pose, betas) + T, i.e. the bundle's cam_t
        if "joints" in r:  # rest wrist J0 = wrist joint - transl (camera frame)
            j0[t] = np.asarray(r["joints"][0]) - trans[t]
        valid[t] = bool(np.isfinite(orient[t]).all() and np.isfinite(finger[t]).all() and np.isfinite(trans[t]).all()
                        and np.isfinite(j0[t]).all())
    for x in (orient, trans, finger, j0):
        x[~valid] = 0.0
    filled, counts = fill_gaps(orient, trans, finger, j0, valid, c2w, timestamps, max_missing,
                               left=side == "left", fill_long=fill_long)
    counts.update(n_observed=int(valid.sum()), no_observations=bool(not valid.any()))
    valid = valid | filled
    counts["n_invalid"] = int((~valid).sum())
    w2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    if valid.any():
        w2c[valid, :3, :3] = Rotation.from_rotvec(orient[valid]).as_matrix()
        w2c[valid, :3, 3] = trans[valid]
    return (betas, orient.astype(np.float32), trans.astype(np.float32), finger.astype(np.float32),
            valid, w2c, filled, counts)


def reference_object_order(meta, ours):
    """(names, source): the episode's objects in the reference (metric object slot) order."""
    split, ep = meta.get("split"), meta.get("episode")
    if split == "evaluation":
        f = KIT / "data/track_3_evaluation_objects.json"
        return list(json.loads(f.read_text())["episodes"][str(int(ep))]), str(f)
    if split == "public":
        sys.path.insert(0, str(KIT))
        from v2dlb.trajectory import episode_parquet, read_episode  # the scorer's reader: slot = struct order
        p = episode_parquet(ROOT / "data/track_3/public", int(ep))
        return list(read_episode(p, int(ep)).object_names), f"{p}:observation.objects"
    return list(ours), "objects.json"


def select_objects(spec, meta, ours):
    """Objects to export, in reference order: 'all' = every reference object (all must be reconstructed),
    'A,B' = that subset (reordered to the reference order). Returns (names, order source)."""
    order, source = reference_object_order(meta, ours)
    if spec == "all":
        missing = [o for o in order if o not in ours]
        if missing:
            raise ValueError(f"Reference objects {missing} have no reconstruction (ours: {ours}); "
                             "pass --objects_list with the reconstructed subset")
        return order, source
    want = [o.strip() for o in spec.split(",") if o.strip()]
    bad = [o for o in want if o not in ours]
    if bad or len(set(want)) != len(want):
        raise ValueError(f"--objects_list {spec!r}: unknown or repeated objects (ours: {ours})")
    names = sorted(want, key=lambda o: order.index(o) if o in order else len(order) + want.index(o))
    if names != want:
        print(f"note: objects reordered to the reference order {names} ({source})", flush=True)
    return names, source


def object_poses(run, variant, nm, n):
    """Object-to-camera poses of one object for all n frames; refuses incomplete trajectories."""
    o2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    ok = np.zeros(n, bool)
    for f in (run / variant / nm / "poses").glob("*.json"):
        t = int(f.stem)
        if 0 <= t < n:
            o2c[t], ok[t] = load_T(f), True
    if not ok.all() or not np.isfinite(o2c).all():
        raise ValueError(f"Incomplete/non-finite {nm} trajectory ({ok.sum()}/{n} frames); "
                         "reconstruct missing frames first")
    return o2c, ok


def copy_mesh(nm, out, stem):
    """The scanned OBJ of nm as <out>/<stem>.obj with its material and textures renamed <stem>.mtl /
    <stem>_<texture>, so several objects (all scans use material.mtl + material_0.png) share one dir."""
    mesh_dir = ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm
    lines = []
    for ln in (mesh_dir / f"{nm}.obj").read_text().splitlines():
        if ln.split()[:1] == ["mtllib"]:
            mtl = mesh_dir / " ".join(ln.split()[1:])
            ln = f"mtllib {stem}.mtl"
            mlines = []
            for m in mtl.read_text().splitlines():
                p = m.split()
                if p and p[0].startswith("map_") and len(p) >= 2:
                    shutil.copy(mesh_dir / p[-1], out / f"{stem}_{Path(p[-1]).name}")
                    m = " ".join(p[:-1] + [f"{stem}_{Path(p[-1]).name}"])
                mlines.append(m)
            (out / f"{stem}.mtl").write_text("\n".join(mlines) + "\n")
        lines.append(ln)
    (out / f"{stem}.obj").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--object", default=None, help="export this one object (single-object bundle)")
    ap.add_argument("--objects_list", default="all",
                    help="'all' (every object of the episode, reference order) or a comma list; ignored with --object")
    ap.add_argument("--objects", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default=os.environ.get("V2D_OBJ_DIR", "objects_sam3"))
    ap.add_argument("--hands", default="mano_fit_phys")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_gap_s", type=float, default=0.15,
                    help="bookkeeping threshold: interior gaps up to this long count as short (n_interpolated_short)")
    ap.add_argument("--no_fill_long", action="store_true",
                    help="only fill short interior gaps; longer and leading/trailing gaps stay zeros, is_valid False")
    ap.add_argument("--no_gap_fill", action="store_true",
                    help="refuse to invent hand frames: every frame of an observed side must come from --hands "
                         "(e.g. s06k_geneoh, which fills unobserved frames by optimisation), else error")
    ap.add_argument("--no_reject_spurious_hands", action="store_true",
                    help="keep hand detections that duplicate the other hand or are short runs without a SAM 3 mask")
    ap.add_argument("--no_rest_settle", action="store_true",
                    help="keep raw resting poses (default: rest_settle.py puts resting objects flat on a stable face, "
                         "bottom on the table)")
    ap.add_argument("--no_depenetrate", action="store_true",
                    help="keep object poses that go into the table (default: depenetrate.py holds static objects at a valid "
                         "pose and lifts moving ones out of the table)")
    ap.add_argument("--strict", action="store_true",
                    help="refuse export when the hand refinement's geometric quality check failed or is missing")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    # Require provenance instead of silently mixing independently refined motions.
    hand_report = json.loads((run / "stereo" / a.hands / "report.json").read_text())
    hand_variant = hand_report.get("args", {}).get("variant")
    if hand_variant != a.objects:
        raise ValueError(f"Hands must be refined against {a.objects!r}; report says {hand_variant!r}. "
                         "Run the final grasp refinement against the chosen object variant first.")
    quality_passed = hand_report.get("quality_passed") is True
    if not quality_passed:
        msg = (f"Hand geometry failed or lacks the final penetration/finite-value checks "
               f"(quality_passed={hand_report.get('quality_passed')!r}, {a.hands}/report.json)")
        if a.strict:
            raise ValueError(msg + "; refine before export")
        print(f"\n!!!!! WARNING: {msg}. Exporting anyway; manifest quality.passed = false. "
              "Use --strict to refuse. !!!!!\n", file=sys.stderr, flush=True)
    meta = json.loads((run / "episode.json").read_text())
    fps = float(meta["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be positive and finite")
    objs = [o["name"] for o in json.loads((run / a.obj_dir / "objects.json").read_text())]
    if a.object:
        names, order_source = [a.object], "--object"
    else:
        names, order_source = select_objects(a.objects_list, meta, objs)
    nm, multi = names[0], len(names) > 1  # one object -> the single-object bundle, unchanged
    out = Path(a.out) if a.out else run / "robot" / "__".join(names) / "outputs" / "result"
    out.mkdir(parents=True, exist_ok=True)
    n = len(list((run / "recon_hawor/frames").glob("*.png")))
    if n < 2:
        raise ValueError("At least two frames are required")
    timestamps = np.asarray(meta.get("timestamps", np.arange(n) / fps), dtype=np.float64)
    if timestamps.shape != (n,) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Invalid source timestamps")
    if not np.allclose(np.diff(timestamps), 1 / fps, rtol=1e-3, atol=1e-6):
        raise ValueError("Nonuniform timestamps: resample all scene streams together before export")
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    cam_ok = np.zeros(n, bool)
    for t, M in slam.items():
        if 0 <= t < n:
            c2w[t], cam_ok[t] = M, True
    if not cam_ok.all() or not np.isfinite(c2w).all():
        raise ValueError("Incomplete/non-finite camera trajectory; refusing identity-pose placeholders")
    # gravity alignment: world rotated so the measured table normal is +z and the table is at z = 0
    # (the loader's own OBB guess takes the object's thinnest axis as vertical -- wrong for cups).
    # Only camera-to-world changes: objects and hands are stored relative to the camera.
    tp = json.loads((run / "table_plane.json").read_text())
    up = np.array(tp["up"], float)
    A = np.eye(4)
    A[:3, :3] = Rotation.align_vectors([[0, 0, 1.0]], [up])[0].as_matrix()
    A[2, 3] = -float(tp["height"])  # p.up = h on the table -> z = 0 after the rotation
    c2w = np.einsum("ij,njk->nik", A, c2w).astype(np.float32)
    poses = [object_poses(run, a.objects, o, n) for o in names]  # complete per frame, else refused
    # rest settle (table frame: z = 0 is the table top): resting objects flat on a stable face, bottom at z = 0
    disk = None
    if tp.get("disk_fit", {}).get("ok") and "center" in tp and "radius" in tp:
        disk = {"center": (A[:3, :3] @ np.array(tp["center"], float) + A[:3, 3]).tolist(), "radius": tp["radius"]}
    settle_report, depen_report, frame0_min_z, o2w_all = {}, {}, [], []
    o2w_orig, coms, vsub = [], [], []   # before hold/settle/lift/snap: the hands follow the object's world displacement
    w2c_all = np.linalg.inv(c2w.astype(np.float64))
    for i, o in enumerate(names):
        verts, com = mesh_verts_com(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / o / f"{o}.obj")
        o2w_i = np.einsum("nij,njk->nik", c2w.astype(np.float64), poses[i][0].astype(np.float64))
        o2w_orig.append(o2w_i.copy()); coms.append(np.asarray(com, float)); vsub.append(verts[:: max(1, len(verts) // 300)])
        if disk is not None and not a.no_depenetrate:  # static run drifting into the table -> valid neighbours
            o2w_i, depen_report[o] = hold_static(o2w_i, verts, com, disk, fps)
        if not a.no_rest_settle:
            o2w_i, settle_report[o] = settle(o2w_i, verts, com, disk=disk, fps=fps)
        if disk is not None and not a.no_depenetrate:  # what is left (moving objects): lift out of the table
            o2w_i, lift = lift_out_of_table(o2w_i, verts, disk)
            depen_report[o] = {"hold_static": depen_report.get(o, []), "lift": lift}
            held = [r for r in depen_report[o]["hold_static"] if r["applied"]]
            print(f"depenetrate {o}: held {sum(r['replaced_frames'] for r in held)} static frames "
                  f"({[(r['frames'], r['max_depth_mm_before'], r['max_depth_mm_after']) for r in held]}), lift {lift}",
                  flush=True)
        if o in settle_report or o in depen_report:
            poses[i] = (np.einsum("nij,njk->nik", w2c_all, o2w_i).astype(np.float32), poses[i][1])
        if o in settle_report:
            done = [r for r in settle_report[o] if r["applied"]]
            print(f"rest settle {o}: {len(done)}/{len(settle_report[o])} rest segments settled "
                  + ", ".join(f"{r['frames']} tilt {r['tilt_deg']} deg dz {r['dz_mm']} mm" for r in done), flush=True)
        o2w_all.append(o2w_i)
        frame0_min_z.append(float((verts @ o2w_i[0, :3, :3].T)[:, 2].min() + o2w_i[0, 2, 3]))
    # assembled objects (slotted wooden pieces) overlapping in the simulator's collision parts at the end: shift the
    # last-moved one (and, below, the hand holding it) into place; parts from collision_parts.py
    snap_report = []
    if multi and not a.no_depenetrate:
        # provided collision meshes only when allowed; the compliant tree (V2D_PROVIDED_COLLISION=none) uses the
        # CoACD parts of OUR meshes (collision_parts.py --visual -> collision_parts_<mesh dir>)
        mdir = os.environ.get("V2D_MESH_DIR", "meshes_obj")
        pdir = ROOT / "robot/assets_extra" / ("collision_parts" if os.environ.get("V2D_PROVIDED_COLLISION", "all") != "none"
                                              and mdir == "meshes_obj" else f"collision_parts_{mdir}")
        # only slotted assemblies (the wooden pieces) are snapped; every other pair is skipped (parts None), as
        # before our meshes had parts for all objects -- running it on e.g. two cups took hours and has no purpose
        parts = [load_parts(pdir / f"{o}.npz") if o in ASSEMBLY_OBJECTS else None for o in names]
        print(f"assembly snap collision parts: {pdir}", flush=True)
        deltas, snap_report = snap_assembly(o2w_all, parts, fps)
        for r in snap_report:
            print(f"assembly snap {names[r['mover']]} (vs {names[sum(r['pair']) - r['mover']]}): overlap "
                  f"{r['overlap_cm3_before']} -> {r['overlap_cm3_after']} cm3, shift {r['shift_cm']} cm, "
                  f"{'frames ' + str(r['frames']) if r['applied'] else 'NOT applied: ' + r['reason']}", flush=True)
        for i, dl in enumerate(deltas):
            if np.abs(dl).max() > 0:
                o2w_all[i][:, :3, 3] += dl
                poses[i] = (np.einsum("nij,njk->nik", w2c_all, o2w_all[i]).astype(np.float32), poses[i][1])
                frame0_min_z[i] = float((mesh_verts_com(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / names[i] / f"{names[i]}.obj")[0]
                                         @ o2w_all[i][0, :3, :3].T)[:, 2].min() + o2w_all[i][0, 2, 3])
    # The loader puts the lowest frame-0 object point (all bodies) at z = ground_z_offset; passing this value
    # makes its shift zero, so the sim ground stays at our table top (run_convert.sbatch reads it).
    loader_ground_z_offset = min(frame0_min_z)
    o2c, obj_ok = poses[0]
    o2w = np.einsum("nij,njk->nik", c2w, o2c)
    ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    arr = dict(fps=np.float64(fps), timestamps=timestamps,
               camera_to_world_transform=c2w, camera_is_valid=cam_ok,
               camera_intrinsics=np.array([ki["fx"], ki["fy"], ki["cx"], ki["cy"]], np.float32),
               object_to_camera_transform=o2c, object_to_world_transform=o2w, object_scale=np.float32(1.0),
               object_is_valid=obj_ok)
    if multi:  # per-slot keys; the unindexed object_* above are slot 0 (for single-object readers)
        arr.update(object_names=np.array(names), object_count=np.int64(len(names)))
        for i, (o2c_i, ok_i) in enumerate(poses):
            arr.update({f"object_{i}_to_camera_transform": o2c_i,
                        f"object_{i}_to_world_transform": np.einsum("nij,njk->nik", c2w, o2c_i),
                        f"object_{i}_scale": np.float32(1.0), f"object_{i}_is_valid": ok_i})
    max_missing = int(np.floor(a.max_gap_s * fps + 1e-6))  # 0.15 s at 20 fps -> 3 frames
    gaps = {}
    spurious = {"left": set(), "right": set()} if a.no_reject_spurious_hands else spurious_hand_frames(run, a.hands, n)
    for side in ("left", "right"):
        if spurious[side]:
            print(f"rejected {len(spurious[side])} spurious {side}-hand frames: {sorted(spurious[side])}", flush=True)
        b, o, tr, fp, ok, w2c, filled, counts = hand_arrays(run, a.hands, side, n, c2w, timestamps, max_missing,
                                                            fill_long=not a.no_fill_long, reject=spurious[side])
        counts["rejected_spurious"] = sorted(int(t) for t in spurious[side])
        if a.no_gap_fill and not counts["no_observations"] and filled.any():
            raise ValueError(f"--no_gap_fill: {side} hand has {int(filled.sum())} frames missing from {a.hands} "
                             f"({counts}); fill them in the hand stage, not by export interpolation/hold")
        # A hand at an object follows that object's export displacement (hold_static / settle / lift / assembly snap):
        # moving the object alone left grasping hands 2-6 cm off it. Per frame, the nearest object whose centre
        # (original pose) has its surface within HAND_FOLLOW_M of the wrist and that moved by 1 mm .. 6 cm; translation only.
        wrist_w = np.einsum("nij,nj->ni", c2w[:, :3, :3].astype(np.float64), w2c[:, :3, 3]) + c2w[:, :3, 3]
        shift, best = np.zeros((n, 3)), np.full(n, np.inf)
        for k in range(len(names)):
            c0 = np.einsum("nij,j->ni", o2w_orig[k][:, :3, :3], coms[k]) + o2w_orig[k][:, :3, 3]
            c1 = np.einsum("nij,j->ni", o2w_all[k][:, :3, :3], coms[k]) + o2w_all[k][:, :3, 3]
            surf = np.einsum("nij,pj->npi", o2w_orig[k][:, :3, :3], vsub[k]) + o2w_orig[k][:, None, :3, 3]
            dk, dist = c1 - c0, np.linalg.norm(surf - wrist_w[:, None], axis=2).min(1)   # wrist to object SURFACE
            mag = np.linalg.norm(dk, axis=1)
            sel = ok & (dist < HAND_FOLLOW_M) & (mag > 1e-3) & (mag < 0.06) & (dist < best)
            shift[sel], best[sel] = dk[sel], dist[sel]
        moved = np.linalg.norm(shift, axis=1) > 0
        counts["follow_object_frames"] = int(moved.sum())
        counts["follow_object_shift_mm_max"] = round(float(np.linalg.norm(shift, axis=1).max()) * 1000, 1)
        if moved.any():
            d_cam = np.einsum("nji,nj->ni", c2w[:, :3, :3].astype(np.float64), shift)  # R_c2w^T d
            tr = (tr + d_cam).astype(np.float32)
            w2c[:, :3, 3] = tr
            print(f"hand follow: {side} hand shifted with its object on {int(moved.sum())} frames "
                  f"(max {counts['follow_object_shift_mm_max']} mm)", flush=True)
        arr.update({f"hand_{side}_betas": b, f"hand_{side}_wrist_orient_in_camera": o,
                    f"hand_{side}_wrist_trans_in_camera": tr, f"hand_{side}_wrist_to_camera_transform": w2c,
                    f"hand_{side}_wrist_to_world_transform": np.einsum("nij,njk->nik", c2w, w2c),
                    f"hand_{side}_finger_pose": fp, f"hand_{side}_scale": np.ones(n, np.float32),
                    f"hand_{side}_is_valid": ok, f"hand_{side}_is_interpolated": filled})
        gaps[side] = {"valid_frames": int(ok.sum()), "interpolated_frames": int(filled.sum()), **counts}
        if counts["no_observations"]:
            print(f"\n!!!!! WARNING: {a.run}: no {side}-hand observations in {a.hands}; the {side} hand is "
                  "written as zeros with is_valid False, and the upstream loader ignores is_valid (a zero-pose "
                  f"hand at the camera). Manifest hand_gaps.{side}.no_observations = true. !!!!!\n",
                  file=sys.stderr, flush=True)
    np.savez(out / "result.npz", **arr)
    # mesh: the scanned OBJ with its material and textures
    mesh_dir = ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm
    shutil.copy(mesh_dir / f"{nm}.obj", out / "mesh.obj")
    for f in mesh_dir.iterdir():
        if f.suffix.lower() in (".mtl", ".png", ".jpg", ".jpeg"):
            shutil.copy(f, out / f.name)
    for i, o in enumerate(names if multi else []):
        copy_mesh(o, out, f"mesh_{i}")
    manifest = {"schema_version": 2, "n_frames": n, "fps": fps, "timestamps": timestamps.tolist(),
                "result_npz": "result.npz", "camera_pose_convention": "camera_to_world",
                "hand_gaps": {"max_gap_s": a.max_gap_s, "max_missing_frames": max_missing,
                              "fill": ("interior: world-frame lerp (wrist joint) / slerp (wrist, fingers); "
                                       "leading/trailing: hold first/last observed world pose"),
                              "fill_long": not a.no_fill_long,
                              "unfilled": "zeros, is_valid false (only sides without observations, or --no_fill_long)",
                              **gaps},
                "quality": {"passed": quality_passed, "strict": a.strict,
                            "hand_report": {k: hand_report.get(k) for k in ("quality_passed", "quality_summary")}},
                "mesh": {"path": "mesh.obj", "vertex_scale": [1.0, 1.0, 1.0]},
                "rest_settle": {"enabled": not a.no_rest_settle, "table_disk": disk, "segments": settle_report},
                "depenetrate": {"enabled": disk is not None and not a.no_depenetrate, "objects": depen_report,
                                "assembly_snap": snap_report},
                # round table top in the exported (table) world frame, top face at z = 0; write_table_support.py
                # turns it into the FlashCHORD support surface (sim table) after the loader's xy recentring
                "table": None if disk is None else {**disk, "thickness": float(tp.get("thickness", 0.03)),
                                                    "disk_fit": tp.get("disk_fit")},
                "loader_ground_z_offset": loader_ground_z_offset,
                "gravity_alignment": {"method": "stereo_table_plane", "up_in_slam_world": up.tolist(),
                                      "table_height": tp["height"], "transform": A.tolist()},
                "sources": {"run": a.run, "object": nm, "objects_variant": a.objects, "hands": a.hands,
                            "object_transform_convention": "rigid_pose_no_scale", "mesh_source": str(mesh_dir)},
                "keys": sorted(arr)}
    if multi:
        manifest["sources"]["object"] = "__".join(names)
        manifest["object_order"] = {"names": names, "source": order_source,
                                    "convention": "slot i = metric object slot i = loader body i"}
        manifest["objects"] = [{"slot": i, "name": o, "mesh": f"mesh_{i}.obj", "key_prefix": f"object_{i}_",
                                "valid_frames": int(poses[i][1].sum()),
                                "mesh_source": str(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / o)}
                               for i, o in enumerate(names)]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    valid = [int(p[1].sum()) for p in poses] if multi else obj_ok.sum()
    print(f"{a.run}/{'+'.join(names)}: {n} frames, object valid {valid}, left {arr['hand_left_is_valid'].sum()}, "
          f"right {arr['hand_right_is_valid'].sum()} (interpolated left {gaps['left']['interpolated_frames']}, "
          f"right {gaps['right']['interpolated_frames']}), quality passed {quality_passed} -> {out}", flush=True)


if __name__ == "__main__":
    main()
