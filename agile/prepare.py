"""AGILE phase A (host env, GPU job): interaction onset frame + FoundationPose init.

IOF (paper Appendix A, Eq. 7): the first frame i with
    r_i = ||(M_o^{i+1} - M_o^i) * (1 - M_h^{i+1}) * (1 - M_h^i)||_1 / (||M_o^i||_1 + eps) > tau
(tau = 0.025, eps = 1e-8) whose object mask does not touch the image border.
Egocentric adaptation (paper Appendix C: the IOF heuristic is the only part that
assumes a static camera). Two detectors, both logged:
  mask_ratio   Eq. 7 with frame i's masks warped into frame i+1 by the metric depth and
               the DROID-SLAM relative camera motion. On ep 11 it fired on mask flicker
               (static cup at frame 27, moving cup at frame 1 vs. mocap onset 25).
  world_motion (default) first frame the object's world position (SLAM x FoundationPose)
               leaves its resting position by > 2 cm for 3 frames, object fully in view.
An object that never moves (e.g. a dish rack) falls back to the frame where it is most
visible (largest border-free mask).

Then FoundationPose (toolkit module) is registered at the IOF with the metric mesh
(paper: "RGB frame, metric depth map, mesh pre-scaled by s_o").
Writes <run>/agile/<object>/prepare.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses

ROOT = Path(os.environ["V2D_ROOT"])
TAU, EPS = 0.025, 1e-8
HAND_IDS = (2, 3)


def decode_depth(path):
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = np.where(png > 0, 65535.0 / np.maximum(png, 1) - 1.0, 0)
    return np.where(np.isfinite(d) & (d > 1e-3), d, 0)


def load_pose(path):
    p = json.loads(Path(path).read_text())
    q = p["rotation"]  # w-first
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = p["translation"]
    return T


def mask(path, shape):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 0) if m is not None else np.zeros(shape, bool)


def warp_masks(masks, depth, K, T_next_from_cur, shape):
    """Forward-warp binary masks of frame i into frame i+1 (splat + small closing)."""
    h, w = shape
    union = np.zeros(shape, bool)
    for m in masks:
        union |= m
    v, u = np.nonzero(union & (depth > 0))
    out = [np.zeros(shape, np.uint8) for _ in masks]
    if len(u) == 0:
        return [o.astype(bool) for o in out]
    z = depth[v, u]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    P = P @ T_next_from_cur[:3, :3].T + T_next_from_cur[:3, 3]
    ok = P[:, 2] > 1e-3
    uu = np.round(P[ok, 0] / P[ok, 2] * K[0, 0] + K[0, 2]).astype(int)
    vv = np.round(P[ok, 1] / P[ok, 2] * K[1, 1] + K[1, 2]).astype(int)
    inb = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    src_u, src_v = u[ok][inb], v[ok][inb]
    for k, m in enumerate(masks):
        sel = m[src_v, src_u]
        out[k][vv[inb][sel], uu[inb][sel]] = 1
        out[k] = cv2.morphologyEx(out[k], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return [o.astype(bool) for o in out]


def interaction_onset(run: Path, obj_id: int, depth_dir: Path, K, n: int):
    masks_dir = run / "objects/masks"
    slam = run / "recon_hawor/slam_poses"
    shape = cv2.imread(str(run / "recon_hawor/frames/000000.png"), cv2.IMREAD_GRAYSCALE).shape
    ratios, border_ok, areas = [], [], []
    prev = None
    for i in range(n):
        mo = mask(masks_dir / f"{obj_id}/{i:06d}.png", shape)
        mh = np.zeros(shape, bool)
        for hid in HAND_IDS:
            mh |= mask(masks_dir / f"{hid}/{i:06d}.png", shape)
        touches = mo[0].any() or mo[-1].any() or mo[:, 0].any() or mo[:, -1].any()
        border_ok.append(bool(mo.any() and not touches))
        areas.append(int(mo.sum()))
        if prev is not None:
            pmo, pmh, pdepth, pT = prev
            T_next_from_cur = np.linalg.inv(load_pose(slam / f"{i:06d}.json")) @ pT  # c_{i} <- w <- c_{i-1}
            wmo, wmh = warp_masks([pmo, pmh], pdepth, K, T_next_from_cur, shape)
            num = np.abs(mo.astype(np.int8) - wmo.astype(np.int8)) * (~mh) * (~wmh)
            ratios.append(float(num.sum() / (wmo.sum() + EPS)))
        prev = (mo, mh, decode_depth(depth_dir / f"{i:06d}.png"), load_pose(slam / f"{i:06d}.json"))
    # r_i compares frame i -> i+1; onset is frame i (first satisfying both criteria).
    iof = next((i for i, r in enumerate(ratios) if r > TAU and border_ok[i] and border_ok[i + 1]), None)
    return iof, ratios, border_ok, areas


def hand_world_tracks(run: Path, depth_dir: Path, K, n: int, shape):
    """Per hand: world-frame 3D centroid (median of depth-unprojected SAM2 hand pixels,
    via SLAM camera-to-world) and the dilated mask used for contact tests."""
    slam = run / "recon_hawor/slam_poses"
    tracks = {}
    for hid in HAND_IDS:
        pos = np.full((n, 3), np.nan)
        for t in range(n):
            m = mask(run / f"objects/masks/{hid}/{t:06d}.png", shape)
            if m.sum() < 500:
                continue
            d = decode_depth(depth_dir / f"{t:06d}.png")
            v, u = np.nonzero(m & (d > 0))
            if len(u) < 200:
                continue
            z = d[v, u]
            p = np.median(np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1), 0)
            pos[t] = (load_pose(slam / f"{t:06d}.json") @ np.r_[p, 1])[:3]
        tracks[hid] = pos
    return tracks


def world_motion_onset(run: Path, obj: dict, n: int, border_ok: list, hand_tracks: dict, shape,
                       thresh=0.02, persist=3, hand_thresh=0.015):
    """Egocentric IOF (our replacement per paper App. C). Object world position =
    SLAM camera-to-world x FoundationPose object-in-camera pose. Onset = first
    border-free frame where the object has left its resting position by > `thresh`
    for `persist` frames AND a hand touching it (dilated hand mask overlaps the object
    mask) moves > `hand_thresh` over the same window: objects only move when a
    contacting hand moves, which rejects FoundationPose jumps under occlusion (e.g.
    the static blue cup in ep 11 'moving' when another cup was stacked on it)."""
    poses_dir = run / "objects" / obj["name"] / "poses_smoothed"
    slam = run / "recon_hawor/slam_poses"
    pw = np.full((n, 3), np.nan)
    for t in range(n):
        pf, sf = poses_dir / f"{t:06d}.json", slam / f"{t:06d}.json"
        if pf.exists() and sf.exists():
            pw[t] = (load_pose(sf) @ load_pose(pf))[:3, 3]
    valid = np.nonzero(np.isfinite(pw[:, 0]))[0]
    if len(valid) < persist + 5:
        return None, pw
    rest = np.median(pw[valid[:5]], 0)
    moved = np.linalg.norm(pw - rest, axis=1) > thresh
    kernel = np.ones((31, 31), np.uint8)
    for t in valid:
        if not (border_ok[t] and all(t + k < n and moved[t + k] for k in range(persist))):
            continue
        mo = mask(run / f"objects/masks/{obj['id']}/{t:06d}.png", shape)
        lo, hi = max(0, t - persist), min(n - 1, t + persist)
        d_obj = pw[hi] - pw[lo] if np.isfinite(pw[[lo, hi], 0]).all() else None
        for hid, pos in hand_tracks.items():
            mh = mask(run / f"objects/masks/{hid}/{t:06d}.png", shape)
            touching = (cv2.dilate(mh.astype(np.uint8), kernel) > 0) & mo
            if not touching.any() or d_obj is None or not np.isfinite(pos[[lo, hi], 0]).all():
                continue
            d_hand = pos[hi] - pos[lo]
            # A grasped object moves *with* the touching hand: the hand must move and the
            # object's displacement must agree with it (rejects occlusion jumps while the
            # hand merely holds the object still).
            if np.linalg.norm(d_hand) > hand_thresh and \
                    np.linalg.norm(d_obj - d_hand) < 0.5 * max(np.linalg.norm(d_obj), np.linalg.norm(d_hand)):
                return int(t), pw
    return None, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--iof", choices=("world_motion", "mask_ratio"), default="world_motion",
                    help="world_motion: our egocentric replacement (App. C); mask_ratio: paper Eq. 7 "
                         "with ego-motion-compensated masks")
    ap.add_argument("--depth", choices=("moge", "stereo"), default="moge",
                    help="paper uses MoGe-2 metric depth; 'stereo' = our FoundationStereo depth")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    objs = json.loads((run / "objects/objects.json").read_text())
    depth_dir = run / ("recon_hawor/depth" if args.depth == "moge" else "depth_stereo")
    intr = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    n = len(list((run / "recon_hawor/frames").glob("*.png")))
    shape = cv2.imread(str(run / "recon_hawor/frames/000000.png"), cv2.IMREAD_GRAYSCALE).shape
    hand_tracks = hand_world_tracks(run, depth_dir, K, n, shape)
    for o in objs:
        out = run / "agile" / o["name"]
        out.mkdir(parents=True, exist_ok=True)
        iof_mask, ratios, border_ok, areas = interaction_onset(run, o["id"], depth_dir, K, n)
        iof_world, world_traj = world_motion_onset(run, o, n, border_ok, hand_tracks, shape)
        iof = iof_world if args.iof == "world_motion" else iof_mask
        how = args.iof
        if iof is None:  # never moves (e.g. a dish rack): most visible border-free frame
            iof = int(np.argmax([a if ok else 0 for a, ok in zip(areas, border_ok)]))
            how = "static_fallback_max_visible"
        print(f"{o['name']}: IOF = frame {iof} ({how}); world_motion={iof_world} mask_ratio={iof_mask}", flush=True)
        fp_dir = out / f"fp_ref{iof:06d}"
        if not (fp_dir / f"{iof:06d}.json").exists():
            run_video_to_poses(video_path=str(run / "input_undistorted.mp4"), depth_folder=str(depth_dir),
                               masks_folder=str(run / f"objects/masks/{o['id']}"),
                               camera_intrinsics_path=str(run / "recon_hawor/intrinsics_stable.json"),
                               mesh_path=o["mesh"], poses_dir=str(fp_dir),
                               weights_dir=str(ROOT / "weights/foundation_pose"), reference_frame=iof,
                               mask_depth=True, target_width=1014, target_height=760, dev=False)
        (out / "prepare.json").write_text(json.dumps({
            "object": o["name"], "object_id": o["id"], "mesh": o["mesh"], "iof": iof, "iof_method": how,
            "iof_world_motion": iof_world, "iof_mask_ratio": iof_mask, "iof_ratios": ratios, "tau": TAU,
            "depth": args.depth,
            "fp_pose_at_iof": str(fp_dir / f"{iof:06d}.json")}, indent=2))


if __name__ == "__main__":
    main()
