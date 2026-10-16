"""Verifier of a 4D hand-object sequence -- the critic of the refinement agent (agile image).

Checks every frame of an episode's reconstruction against the OBSERVATIONS (stereo depth, SAM 3
masks, 2D hand detections, stereo triangulation) and against PHYSICS. It never uses mocap, so it
runs on the evaluation split too. Per-frame scores are turned into flagged windows ("issues") with
evidence, which the planner/LLM agent acts on.

Objects (per object, per frame)
  depth_agree   fraction of the posed mesh's camera-facing surface points whose depth matches the
                stereo depth within 1 cm (pixels under a hand excluded)
  free_space    fraction of those points lying > 2 cm IN FRONT of the observed surface: the object
                would occlude something the camera sees -- a strong, occlusion-proof error signal
  mask_iou      IoU of the posed mesh's image footprint with the SAM 3 mask (hand pixels excluded)
  acc           world-frame acceleration of the object (mm / frame^2) and angular (deg / frame^2)
  table_pen     how far the mesh sinks below the measured table plane (mm)
  unsupported   off the table, no hand within 1 cm, not resting on another object
  moved_alone   moving (> 5 mm/frame in the world) with no hand within 1 cm
                (--no_hands: "hand" = a stereo joint within 2.5 cm or a fusion_report hand contact +-2 fr)
Hands (per side, per frame)
  tri_err       mean distance to the valid stereo-triangulated joints (mm)
  reproj        mean distance of the projected joints to the colour WiLoR 2D keypoints (px)
  acc           world-frame joint acceleration (mm / frame^2)
  pen           deepest hand vertex inside any object (mm)
Writes <run>/refine/verify_<tag>.json and prints a summary.
Usage: python verify.py <run> [...] [--hands mano_fit_phys] [--objects fused_all_sam3] [--tag base]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano, SDFGrid, wilor_keypoints_2d  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
HAND_IDS = {"left": 2, "right": 3}

# issue thresholds: (metric, comparison, value, issue type, suggested corrector)
OBJ_RULES = [("free_space", ">", 0.15, "object_in_free_space", "object_refit"),
             ("depth_agree", "<", 0.35, "object_depth_mismatch", "object_refit"),
             ("mask_iou", "<", 0.45, "object_mask_mismatch", "object_refit"),
             ("acc_mm", ">", 12.0, "object_jitter", "object_smooth"),
             ("acc_deg", ">", 8.0, "object_rot_jitter", "object_smooth"),
             ("table_pen", ">", 8.0, "object_in_table", "object_refit"),
             ("unsupported", ">", 0.5, "object_floating", "object_refit"),
             ("moved_alone", ">", 0.5, "object_moves_untouched", "object_hold_still")]
HAND_RULES = [("tri_err", ">", 25.0, "hand_off_stereo", "hand_refit"),
              ("reproj", ">", 45.0, "hand_off_image", "hand_refit"),
              ("acc_mm", ">", 15.0, "hand_jitter", "hand_smooth"),
              ("pen", ">", 8.0, "hand_penetration", "hand_contact")]


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


def decode_depth(p):
    png = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    png = png.astype(np.float32)
    return np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, 0).astype(np.float32)


def read_mask(p, shape):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return np.zeros(shape, bool) if m is None else m > 0


JOINT_NEAR = 0.025  # stereo hand joint -> object surface distance counted as "hand on it" (joints sit
                    # inside the finger / palm and are triangulated with ~1 cm noise; MANO vertices use 1 cm)
CONTACT_PAD = 2     # frames of slack around a fusion_report hand_contact frame


def contact_frames(obj_report, pad=CONTACT_PAD):
    """fusion_report.json entry of one object -> set of frames with a hand in contact (+-pad), any side."""
    out = set()
    for frames in (obj_report or {}).get("hand_contact", {}).values():
        for t in frames:
            out |= set(range(int(t) - pad, int(t) + pad + 1))
    return out


def hand_near_joints(t, Wc, tree, joints, contacts=(), thr=JOINT_NEAR):
    """Is a hand on the object at frame t without MANO? True if any stereo-triangulated joint (camera
    frame, {side: {t: 21x3}}) is within thr of the object surface (tree of object-frame surface points,
    object at camera-frame pose Wc), or the fusion contact labels put a hand on it at t (+-pad)."""
    if t in contacts:
        return True
    Wi = None
    for seq in joints.values():
        X = seq.get(t)
        if X is None:
            continue
        X = np.asarray(X, float).reshape(-1, 3)
        X = X[np.isfinite(X).all(1)]
        if not len(X):
            continue
        Wi = np.linalg.inv(Wc) if Wi is None else Wi
        if tree.query(X @ Wi[:3, :3].T + Wi[:3, 3])[0].min() < thr:
            return True
    return False


def windows(flags, gap=3, min_len=2):
    """Frame indices where a flag holds -> merged [start, end] windows."""
    out, cur = [], []
    for t in sorted(flags):
        if cur and t - cur[-1] > gap:
            if len(cur) >= min_len:
                out.append([cur[0], cur[-1]])
            cur = []
        cur.append(t)
    if len(cur) >= min_len:
        out.append([cur[0], cur[-1]])
    return out


def flag_issues(series, rules, target):
    issues = []
    for metric, cmp_, thr, kind, fix in rules:
        s = series.get(metric, {})
        bad = [t for t, v in s.items() if v is not None and ((v > thr) if cmp_ == ">" else (v < thr))]
        for a, b in windows(bad):
            vals = [s[t] for t in range(a, b + 1) if s.get(t) is not None]
            issues.append({"type": kind, "target": target, "frames": [a, b], "metric": metric,
                           "threshold": thr, "value_mean": round(float(np.mean(vals)), 3),
                           "severity": round(float(np.mean(vals) / thr if cmp_ == ">" else thr / max(np.mean(vals), 1e-3)), 2),
                           "suggested": fix})
    return issues


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", default="mano_fit_phys")
    ap.add_argument("--objects", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--only_objects", nargs="*", default=None)
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    ap.add_argument("--no_hands", action="store_true", help="object checks only (no GPU needed)")
    ap.add_argument("--quiet", action="store_true")
    return ap


_MANO = {}


def main(argv=None):
    a = parser().parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.no_hands:
        mano = None
    else:
        mano = _MANO.get(dev) or _MANO.setdefault(dev, Mano(str(ROOT / "weights/hand"), dev))
    reps = {}
    for r in a.runs:
        run = ROOT / "runs" / r
        ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
        Hh, Ww = ki["height"], ki["width"]
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        ts = sorted(slam)[::a.stride]
        if a.frames:
            ts = [t for t in ts if a.frames[0] <= t <= a.frames[1]]
        tp = json.loads((run / "table_plane.json").read_text())
        up, table_h = np.array(tp["up"]), float(tp["height"])
        tri = json.loads((run / "stereo/hands_tri.json").read_text()) if (run / "stereo/hands_tri.json").exists() else {}
        # without MANO (--no_hands, as the agent runs it) the hand-proximity checks use the stereo joints
        # (camera frame) and the fusion's hand-contact labels; with an empty hand dict every carried
        # frame would count as 'moved_alone' and every lifted one as 'unsupported'
        HJ, fus = {}, {}
        if mano is None:
            for side in ("left", "right"):
                HJ[side] = {int(t): X for t, X in (tri.get(side, {}).get("X_smooth") or {}).items() if X is not None}
            if (run / "fusion_report.json").exists():
                fus = json.loads((run / "fusion_report.json").read_text())
        # hands: camera-frame vertices and joints
        H = {}
        for side in ("left", "right"):
            for f in (run / "stereo" / a.hands / side).glob("*.json"):
                t = int(f.stem)
                if t in slam and mano is not None and (not a.frames or a.frames[0] - 1 <= t <= a.frames[1] + 1):
                    h = json.loads(f.read_text())
                    v, j = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                    T = np.array(h["T"])
                    H.setdefault(side, {})[t] = (v.cpu().numpy() + T, j.cpu().numpy() + T)
        objs = []
        for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
            if a.only_objects and o["name"] not in a.only_objects:
                continue
            pd = run / a.objects / o["name"] / "poses"
            W = {t: load_T(pd / f"{t:06d}.json") for t in slam if (pd / f"{t:06d}.json").exists()}
            mesh = trimesh.load(ROOT / "meshes_obj" / o["name"] / f"{o['name']}.obj", force="mesh")
            P, fi = trimesh.sample.sample_surface(mesh, 3000, seed=0)
            objs.append({"name": o["name"], "id": o["id"], "W": W, "P": P, "N": mesh.face_normals[fi],
                         "V": np.asarray(mesh.vertices), "sdf": None if a.no_hands else SDFGrid(mesh, res=128, device=dev),
                         "tree": cKDTree(P), "contacts": contact_frames(fus.get(o["name"]))})
        rep = {"run": r, "hands": a.hands, "objects": a.objects, "frames": len(ts), "object": {}, "hand": {}, "issues": []}
        for o in objs:
            ser = {k: {} for k in ("depth_agree", "free_space", "mask_iou", "acc_mm", "acc_deg", "table_pen",
                                   "unsupported", "moved_alone")}
            for t in ts:
                if t not in o["W"]:
                    continue
                Wc = o["W"][t]
                pc = o["P"] @ Wc[:3, :3].T + Wc[:3, 3]
                nc = o["N"] @ Wc[:3, :3].T
                facing = (nc * -pc).sum(1) > 0
                d = decode_depth(run / f"depth_stereo/{t:06d}.png")
                hand_px = np.zeros((Hh, Ww), bool)
                for s_, hid in HAND_IDS.items():
                    hand_px |= read_mask(run / a.obj_dir / f"masks/{hid}/{t:06d}.png", (Hh, Ww))
                hand_px = cv2.dilate(hand_px.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                q = pc[facing]
                uv = (q[:, :2] / q[:, 2:3] @ K[:2, :2].T + K[:2, 2]).round().astype(int)
                ok = (uv[:, 0] >= 0) & (uv[:, 0] < Ww) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh) & (q[:, 2] > 0.05)
                uv, q = uv[ok], q[ok]
                if d is not None and len(q):
                    dz = d[uv[:, 1], uv[:, 0]]
                    valid = (dz > 0) & ~hand_px[uv[:, 1], uv[:, 0]]
                    if valid.sum() >= 30:
                        res = q[valid, 2] - dz[valid]
                        ser["depth_agree"][t] = float((np.abs(res) < 0.01).mean())
                        ser["free_space"][t] = float((res < -0.02).mean())
                # silhouette footprint vs mask (quarter resolution)
                m = read_mask(run / a.obj_dir / f"masks/{o['id']}/{t:06d}.png", (Hh, Ww))
                if m.any() and len(uv):
                    s4 = 4
                    fp = np.zeros((Hh // s4 + 1, Ww // s4 + 1), np.uint8)
                    fp[uv[:, 1] // s4, uv[:, 0] // s4] = 1
                    fp = cv2.dilate(fp, np.ones((5, 5), np.uint8)) > 0
                    mm = cv2.resize(m.astype(np.uint8), (fp.shape[1], fp.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    hh = cv2.resize(hand_px.astype(np.uint8), (fp.shape[1], fp.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    fp, mm = fp & ~hh, mm & ~hh
                    u = (fp | mm).sum()
                    if u > 20:
                        ser["mask_iou"][t] = float((fp & mm).sum() / u)
                # world-frame physics
                Wt = slam[t] @ Wc
                vw = o["V"] @ Wt[:3, :3].T + Wt[:3, 3]
                ser["table_pen"][t] = float(max(0.0, table_h - (vw @ up).min()) * 1000)
                near_hand = False
                for s_, seq in H.items():
                    if t in seq:
                        Wi = np.linalg.inv(Wc)
                        hl = seq[t][0] @ Wi[:3, :3].T + Wi[:3, 3]
                        if o["tree"].query(hl)[0].min() < 0.01:
                            near_hand = True
                if mano is None:
                    near_hand = hand_near_joints(t, Wc, o["tree"], HJ, o["contacts"])
                lifted = (vw @ up).min() - table_h > 0.01
                on_other = False
                for o2 in objs:
                    if o2 is o or t not in o2["W"]:
                        continue
                    W2 = slam[t] @ o2["W"][t]
                    rel = np.linalg.inv(W2) @ Wt
                    if o2["tree"].query(o["P"][::10] @ rel[:3, :3].T + rel[:3, 3])[0].min() < 0.01:
                        on_other = True
                ser["unsupported"][t] = float(lifted and not near_hand and not on_other)
                tp_, tn_ = t - 1, t + 1
                if tp_ in o["W"] and tn_ in o["W"] and tp_ in slam and tn_ in slam:
                    A, B = slam[tp_] @ o["W"][tp_], slam[tn_] @ o["W"][tn_]
                    ser["acc_mm"][t] = float(np.linalg.norm(A[:3, 3] - 2 * Wt[:3, 3] + B[:3, 3]) * 1000)
                    w1 = Rotation.from_matrix(Wt[:3, :3] @ A[:3, :3].T).as_rotvec()
                    w2 = Rotation.from_matrix(B[:3, :3] @ Wt[:3, :3].T).as_rotvec()
                    ser["acc_deg"][t] = float(np.degrees(np.linalg.norm(w2 - w1)))
                    speed = np.linalg.norm(B[:3, 3] - A[:3, 3]) / 2
                    ser["moved_alone"][t] = float(speed > 0.005 and not near_hand)
            rep["object"][o["name"]] = {k: {str(t): v for t, v in s.items()} for k, s in ser.items()}
            rep["issues"] += flag_issues(ser, OBJ_RULES, o["name"])
        for side, seq in H.items():
            ser = {k: {} for k in ("tri_err", "reproj", "acc_mm", "pen")}
            d3 = tri.get(side, {})
            Xr, Vd = d3.get("X_raw") or {}, d3.get("valid") or {}
            for t in ts:
                if t not in seq:
                    continue
                v, j = seq[t]
                if str(t) in Xr and str(t) in Vd:
                    X, ok = np.array(Xr[str(t)]), np.array(Vd[str(t)], bool)
                    if ok.sum() >= 5:
                        ser["tri_err"][t] = float(np.linalg.norm(j[ok] - X[ok], axis=1).mean() * 1000)
                wf = run / f"recon_hawor/wilor_raw/{t:06d}.json"
                if wf.exists():
                    uv = j[:, :2] / j[:, 2:3] @ K[:2, :2].T + K[:2, 2]
                    best = None
                    for rec in json.loads(wf.read_text()):
                        if rec.get("score", 0) < 0.3:
                            continue
                        _, jl = mano.local(rec)
                        k2 = wilor_keypoints_2d(jl.cpu().numpy(), rec, (Ww, Hh))
                        e = float(np.linalg.norm(k2 - uv, axis=1).mean())
                        best = e if best is None else min(best, e)
                    if best is not None and best < 150:  # a detection of this hand (not the other one)
                        ser["reproj"][t] = best
                if t - 1 in seq and t + 1 in seq:
                    Jw = [seq[u][1] @ slam[u][:3, :3].T + slam[u][:3, 3] for u in (t - 1, t, t + 1)]
                    ser["acc_mm"][t] = float(np.linalg.norm(Jw[2] - 2 * Jw[1] + Jw[0], axis=1).mean() * 1000)
                pen = 0.0
                for o in objs:
                    if t in o["W"]:
                        Wi = np.linalg.inv(o["W"][t])
                        vl = torch.tensor(v @ Wi[:3, :3].T + Wi[:3, 3], dtype=torch.float32, device=dev)
                        pen = max(pen, float(torch.relu(-o["sdf"](vl)).max()) * 1000)
                ser["pen"][t] = pen
            rep["hand"][side] = {k: {str(t): v for t, v in s.items()} for k, s in ser.items()}
            rep["issues"] += flag_issues(ser, HAND_RULES, side)
        # summary scores (lower is better except agreement / iou)
        summ = {}
        for nm, s in rep["object"].items():
            summ[nm] = {k: (round(float(np.mean(list(v.values()))), 4) if v else None) for k, v in s.items()}
        for sd, s in rep["hand"].items():
            summ[sd] = {k: (round(float(np.mean(list(v.values()))), 3) if v else None) for k, v in s.items()}
        rep["summary"] = summ
        reps[r] = rep
        if a.quiet:
            continue
        out = run / "refine"
        out.mkdir(exist_ok=True)
        (out / f"verify_{a.tag}.json").write_text(json.dumps(rep))
        by = {}
        for i in rep["issues"]:
            by.setdefault(f"{i['target']}:{i['type']}", []).append(i["frames"][1] - i["frames"][0] + 1)
        print(f"{r}: " + json.dumps(summ), flush=True)
        print(f"  issues: " + ", ".join(f"{k} x{len(v)} ({sum(v)} fr)" for k, v in sorted(by.items())), flush=True)
    return reps


if __name__ == "__main__":
    main()
