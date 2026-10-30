"""README stage 6.2, second backend: TrackEverything (github.com/ayushjain1144/trackeverything @ f0c18dd,
weights katefgroup/TrackEverything, CC BY-NC 4.0) -- 3D point tracking that consumes OUR geometry (stereo depth,
calibrated K, SLAM camera poses) instead of estimating its own, so tracks live directly in the hoi4d world.

Two steps, two images:
  prepare()  (agile image)              queries -> hoi4d/te_queries.npz
      objects: eroded SAM 3 mask samples every reseed_every frames (same seeding as CoTracker);
      hands:   every DETECTED hand joint (source == observed) every hand_reseed frames, projected into camera a,
               kept only where the hand's own mask covers the pixel and stereo depth there is within
               hand_depth_tol of the joint depth (so a joint occluded by the object in front is never seeded on
               the object). Per query, the joint-minus-surface offset at the query frame is stored.
  infer()    (trackeverything image)    -> hoi4d/tracks_te.npz, same schema as tracks.npz (uv, vis, P_world,
      queries, owner, names) + joint (-1 for object points, MANO/OpenPose joint index for hand points) and
      joint_offset, so agreement(), diagnostics and the optimizer's track term take either backend.

Frame contract (demo_hf._preprocess, supplied geometry): with geometry_backend != "vggt_omega" the model
re-expresses the scene in camera-0 coordinates (extrinsics @ inv(E0)) and divides by a median-depth norm_factor;
query_coords come back in that frame, so P_world = T_world_cam[0] @ (coords * norm_factor). The backend name
only selects that branch: the Pi3 estimator itself is lazy and never loaded when geometry is supplied.
Queries are tracked forward from their own frame (they enter the window that contains it), so frames before a
query's frame are marked not visible; reseeding covers the clip.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/scratch/kghasemz/v2d")


def _decode_depth(png):
    d = 65535.0 / np.maximum(png.astype(np.float64), 1) - 1.0
    d[(png == 0) | (png >= 65535) | ~np.isfinite(d)] = 0.0  # TrackEverything: depth > 0 is the valid mask
    return d


def _erode(m, r=6):
    er = m.copy()
    for ax in (0, 1):
        for s in (r, -r):
            er &= np.roll(m, s, ax)
    return er


def prepare(run_name: str, man: dict, tr, cfg: dict) -> dict:
    """Agile image. tr: the initial Trajectories (world frame, hands with source codes)."""
    from .. import io as hio
    from ..conventions import SOURCE_IDS
    run = ROOT / "runs" / run_name
    F = man["frame_count"]
    W, H = man["cameras"]["ego_cam_a"]["size"]
    K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
    T_wc = tr.extra["T_world_cam"]
    rng = np.random.default_rng(0)
    ents = [e for e in man["entities"] if e.get("mask_id") is not None]
    names = [e["id"] for e in ents]
    Q, owner, joint, offs = [], [], [], []
    every = cfg.get("reseed_every", 10)
    for k, e in enumerate(ents):
        if e["id"].startswith("hand_"):
            continue  # hands are seeded on their joints below
        for f in range(0, F, every):
            p = run / hio.obj_dir() / "masks" / str(e["mask_id"]) / f"{f:06d}.png"
            if not p.exists():
                continue
            ys, xs = np.nonzero(_erode(np.asarray(Image.open(p)) > 0))
            if len(xs) < 30:
                continue
            idx = rng.choice(len(xs), min(cfg.get("points_per_object", 64), len(xs)), replace=False)
            Q += [(f, xs[i], ys[i]) for i in idx]
            owner += [k] * len(idx); joint += [-1] * len(idx); offs += [(0.0, 0.0, 0.0)] * len(idx)
    n_hand = {}
    obs = SOURCE_IDS["observed"]
    for h, hd in tr.hands.items():
        if h not in names:
            continue
        k = names.index(h)
        mid = ents[k]["mask_id"]
        n_hand[h] = 0
        for f in range(0, F, cfg.get("hand_reseed", 5)):
            if hd["source"][f] != obs or not np.isfinite(T_wc[f]).all():
                continue
            dp = run / "depth_stereo" / f"{f:06d}.png"
            if not dp.exists():
                continue
            Jc = (hd["joints"][f] - T_wc[f][:3, 3]) @ T_wc[f][:3, :3]  # world -> camera a
            # hand mask for THIS hand at this frame: SAM 3's left/right ids are not reliable (eval ep38 has them
            # swapped for the whole clip), so take the hand mask (2 or 3) that contains most of the projected joints
            best, m = -1, None
            for cand in (2, 3):
                mp = run / hio.obj_dir() / "masks" / str(cand) / f"{f:06d}.png"
                if not mp.exists():
                    continue
                mm = np.asarray(Image.open(mp)) > 0
                uv = np.c_[K[0, 0] * Jc[:, 0] / Jc[:, 2] + K[0, 2], K[1, 1] * Jc[:, 1] / Jc[:, 2] + K[1, 2]].round().astype(int)
                ok = (Jc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                hits = int(mm[uv[ok, 1], uv[ok, 0]].sum())
                if hits > best:
                    best, m = hits, mm
            if m is None or best < 3:
                continue
            D = _decode_depth(np.asarray(Image.open(dp)))
            for j, X in enumerate(Jc):
                if not np.isfinite(X).all() or X[2] <= 0:
                    continue
                u, v = K[0, 0] * X[0] / X[2] + K[0, 2], K[1, 1] * X[1] / X[2] + K[1, 2]
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < W and 0 <= vi < H) or not m[vi, ui]:
                    continue
                z = D[vi, ui]
                if z <= 0 or abs(z - X[2]) > cfg.get("hand_depth_tol", 0.03):
                    continue
                S = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])  # surface point
                Q.append((f, u, v)); owner.append(k); joint.append(j)
                offs.append(tuple(T_wc[f][:3, :3] @ (X - S)))  # joint - surface, world frame, at the query frame
                n_hand[h] += 1
    out = hio.out_dir(run_name) / "te_queries.npz"
    np.savez_compressed(out, queries=np.asarray(Q, np.float32), owner=np.asarray(owner, np.int32),
                        joint=np.asarray(joint, np.int32), joint_offset=np.asarray(offs, np.float32),
                        names=np.array(names), K=K, T_world_cam=T_wc, size=np.array([W, H]), frame_count=F)
    per = {n: int((np.asarray(owner) == i).sum()) for i, n in enumerate(names)}
    return {"queries": len(Q), "per_entity": per, "hand_joint_queries": n_hand, "file": str(out)}


def infer(run_name: str, cfg: dict, device="cuda") -> dict:
    """TrackEverything image (torch + the repo on PYTHONPATH). Reads te_queries.npz, writes tracks_te.npz."""
    import torch
    from trackeverything.demo_hf import TrackEverything
    run = ROOT / "runs" / run_name
    out_dir = run / os.environ.get("HOI4D_DIR", "hoi4d")
    q = dict(np.load(out_dir / "te_queries.npz", allow_pickle=False))
    F, (W, H) = int(q["frame_count"]), q["size"].astype(int)
    K, T_wc = q["K"], q["T_world_cam"].copy()
    # TrackEverything needs a pose for every frame: SLAM gaps (rare) take the nearest valid pose (input only)
    ok = np.array([np.isfinite(T).all() for T in T_wc])
    idx = np.where(ok)[0]
    for f in np.where(~ok)[0]:
        T_wc[f] = T_wc[idx[np.argmin(abs(idx - f))]]
    s = cfg.get("input_scale", 0.5)  # model resizes to its own image_size anyway; halve the RAM footprint
    w, h = int(W * s) // 4 * 4, int(H * s) // 4 * 4
    sx, sy = w / W, h / H
    video = np.stack([np.asarray(Image.open(run / "recon_hawor/frames" / f"{i:06d}.png").convert("RGB")
                                 .resize((w, h), Image.BILINEAR)) for i in range(F)])
    depths = np.zeros((F, h, w), np.float32)
    for i in range(F):
        dp = run / "depth_stereo" / f"{i:06d}.png"
        if dp.exists():
            D = _decode_depth(np.asarray(Image.open(dp)))
            depths[i] = np.asarray(Image.fromarray(D.astype(np.float32)).resize((w, h), Image.NEAREST))
    Ks = K.copy(); Ks[0] *= sx; Ks[1] *= sy
    E = np.linalg.inv(T_wc).astype(np.float32)  # world -> camera
    Qf = q["queries"].astype(np.float32).copy(); Qf[:, 1] *= sx; Qf[:, 2] *= sy
    model = TrackEverything.from_pretrained(cfg.get("checkpoint", "/scratch/kghasemz/v2d/weights/trackeverything/model.pth"), device=device,
                                            geometry_backend="pi3")  # backend never loaded: selects cam0+norm branch
    out = model.infer(video, depths=depths, intrinsics=np.repeat(Ks[None], F, 0).astype(np.float32),
                      extrinsics=E, query_points_2d=Qf, mode="sparse", num_iters=cfg.get("num_iters", 3))
    C = out.query_coords[0].float().cpu().numpy()  # (T, N, 3) cam0-normalised frame
    Vl = out.query_visibs[0].float().cpu().numpy()
    Vp = Vl if (Vl.min() >= 0 and Vl.max() <= 1) else 1 / (1 + np.exp(-Vl))
    Pw = (C * out.norm_factor) @ T_wc[0][:3, :3].T + T_wc[0][:3, 3]
    N = Pw.shape[1]
    t0 = q["queries"][:, 0].astype(int)
    vis = (Vp > cfg.get("vis_threshold", 0.5)) & (np.arange(F)[:, None] >= t0[None])
    Pw[np.arange(F)[:, None] < t0[None]] = np.nan  # before its own frame a query has no track
    Pc = np.einsum("fij,fnj->fni", np.linalg.inv(T_wc)[:, :3, :3], Pw) + np.linalg.inv(T_wc)[:, None, :3, 3]
    uv = np.stack([K[0, 0] * Pc[..., 0] / Pc[..., 2] + K[0, 2], K[1, 1] * Pc[..., 1] / Pc[..., 2] + K[1, 2]], -1)
    # frame check: each query's track at its own frame vs our depth unprojection there
    Xq = np.array([[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0] for _, u, v in q["queries"]])
    zq, cache = np.zeros(N), {}
    for i, (f, u, v) in enumerate(q["queries"]):
        f = int(f)
        if f not in cache:
            dp = run / "depth_stereo" / f"{f:06d}.png"
            cache[f] = _decode_depth(np.asarray(Image.open(dp))) if dp.exists() else None
        if cache[f] is not None:
            zq[i] = cache[f][min(H - 1, int(round(v))), min(W - 1, int(round(u)))]
    Pq = np.einsum("nij,nj->ni", T_wc[t0][:, :3, :3], Xq * zq[:, None]) + T_wc[t0][:, :3, 3]
    err = np.linalg.norm(Pw[t0, np.arange(N)] - Pq, axis=1)
    err = err[np.isfinite(err) & (zq > 0)]
    np.savez_compressed(out_dir / "tracks_te.npz", uv=uv.astype(np.float32), vis=vis, P_world=Pw.astype(np.float32),
                        queries=q["queries"], owner=q["owner"], names=q["names"], joint=q["joint"],
                        joint_offset=q["joint_offset"], norm_factor=np.array(out.norm_factor),
                        checkpoint=np.array(cfg.get("checkpoint", "/scratch/kghasemz/v2d/weights/trackeverything/model.pth")),
                        license=np.array("TrackEverything code + weights: CC-BY-NC-4.0"))
    rep = {"points": int(N), "frames": int(F), "norm_factor": float(out.norm_factor),
           "query_frame_err_mm": {"median": float(np.median(err) * 1000) if len(err) else None,
                                  "p95": float(np.percentile(err, 95) * 1000) if len(err) else None, "n": int(len(err))},
           "visible_frac": float(vis.mean()), "file": str(out_dir / "tracks_te.npz")}
    (out_dir / "tracks_te_report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":  # trackeverything image: python -m hoi4d.tracking.trackeverything <run>
    import sys
    print(json.dumps(infer(sys.argv[1], {}), indent=1))
