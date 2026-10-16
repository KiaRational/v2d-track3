"""README stage 6.2: CoTracker3 temporal point correspondences, lifted to 3D by stereo depth.

track():      points seeded inside every entity's SAM 3 mask (eroded, every reseed_every frames) are tracked
              through the whole clip (CoTracker3 offline, both directions); per point: 2D track, visibility,
              depth-lifted world position (NaN where occluded / no valid depth). -> hoi4d/tracks.npz
agreement():  for a trajectory version, each object's tracked points are mapped into its object frame with that
              version's poses; a rigid object's surface points must stay put, so the per-frame residual
              (median drift from the seed position, robust over points) is pose-error evidence independent of
              FoundationPose and of the stereo ICP. Used by diagnostics and by the optimizer's visual term.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from .. import io as hio
from ..geometry.assets import inv, transform


def decode_depth(png: np.ndarray) -> np.ndarray:
    d = 65535.0 / np.maximum(png.astype(np.float64), 1) - 1.0
    d[(png == 0) | (png >= 65535)] = np.nan
    return d


def seeds(run, man, F, per_entity, reseed_every, rng):
    """(t, x, y) queries in full-resolution pixels + entity index per query."""
    ents = [e for e in man["entities"] if e.get("mask_id") is not None]
    Q, owner = [], []
    for k, e in enumerate(ents):
        for f in range(0, F, reseed_every):
            # nearest frame with a usable mask at/after f
            for g in range(f, min(F, f + reseed_every)):
                p = run / "objects_sam3/masks" / str(e["mask_id"]) / f"{g:06d}.png"
                if not p.exists():
                    continue
                m = np.asarray(Image.open(p)) > 0
                # erode 6 px so seeds are not on the boundary (mask leakage / other surfaces)
                er = m.copy()
                for ax in (0, 1):
                    for s in (6, -6):
                        er &= np.roll(m, s, ax)
                ys, xs = np.nonzero(er)
                if len(xs) < 30:
                    continue
                idx = rng.choice(len(xs), min(per_entity, len(xs)), replace=False)
                Q += [(g, xs[i], ys[i]) for i in idx]
                owner += [k] * len(idx)
                break
    return np.asarray(Q, float), np.asarray(owner, int), [e["id"] for e in ents]


def track(run_name: str, man: dict, T_wc: np.ndarray, cfg: dict, device="cuda") -> dict:
    import torch
    sys.path.insert(0, cfg["repo"])
    from cotracker.predictor import CoTrackerPredictor
    run = hio.ROOT / "runs" / run_name
    F = man["frame_count"]
    W, H = man["cameras"]["ego_cam_a"]["size"]
    K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
    s = 960 / W  # working resolution
    w, h = int(round(W * s)) // 8 * 8, int(round(H * s)) // 8 * 8
    frames = np.stack([np.asarray(Image.open(run / "recon_hawor/frames" / f"{i:06d}.png").convert("RGB")
                                  .resize((w, h), Image.BILINEAR)) for i in range(F)])
    rng = np.random.default_rng(0)
    Q, owner, names = seeds(run, man, F, cfg["grid_per_object"], cfg["reseed_every"], rng)
    sx, sy = w / W, h / H
    model = CoTrackerPredictor(checkpoint=cfg["checkpoint"], offline=True, window_len=cfg["window"]).to(device)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    q = torch.from_numpy(np.c_[Q[:, 0], Q[:, 1] * sx, Q[:, 2] * sy]).float()[None].to(device)
    tracks, vis = [], []
    with torch.no_grad():
        for c in range(0, len(Q), 512):  # chunk queries for memory
            tr, vi = model(video, queries=q[:, c:c + 512], backward_tracking=True)
            tracks.append(tr[0].cpu().numpy())
            vis.append(vi[0].cpu().numpy())
    uv = np.concatenate(tracks, 1) / [sx, sy]  # (F, N, 2) full-res pixels
    vis = np.concatenate(vis, 1).astype(bool)   # (F, N)
    N = uv.shape[1]
    Pw = np.full((F, N, 3), np.nan)
    for f in range(F):
        dp = run / "depth_stereo" / f"{f:06d}.png"
        if not dp.exists() or not np.isfinite(T_wc[f]).all():
            continue
        D = decode_depth(np.asarray(Image.open(dp)))
        u = np.round(uv[f, :, 0]).astype(int)
        v = np.round(uv[f, :, 1]).astype(int)
        ok = vis[f] & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        z = np.full(N, np.nan)
        z[ok] = D[v[ok], u[ok]]
        X = np.c_[(uv[f, :, 0] - K[0, 2]) / K[0, 0] * z, (uv[f, :, 1] - K[1, 2]) / K[1, 1] * z, z]
        Pw[f] = transform(T_wc[f], X)
    out = hio.out_dir(run_name) / "tracks.npz"
    np.savez_compressed(out, uv=uv.astype(np.float32), vis=vis, P_world=Pw.astype(np.float32), queries=Q,
                        owner=owner, names=np.array(names), checkpoint=np.array(cfg["checkpoint"]),
                        license=np.array("CoTracker3 weights: CC-BY-NC-4.0"))
    return {"points": N, "per_entity": {n: int((owner == k).sum()) for k, n in enumerate(names)}, "file": str(out)}


def load_tracks(run_name):
    p = hio.out_dir(run_name) / "tracks.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


def sym_axis(run_name, name):
    """Continuous-symmetry axis (object frame) + centre from fusion_report.json, or None."""
    import json
    rep = hio.ROOT / "runs" / run_name / "fusion_report.json"
    if not rep.exists():
        return None
    d = json.loads(rep.read_text()).get(name, {})
    if d.get("symmetry") != "continuous" or d.get("sym_axis") is None:
        return None
    from ..geometry.assets import asset
    ax = np.asarray(d["sym_axis"], float)
    return ax / np.linalg.norm(ax), asset(name).centroid


def sym_coords(Po, axc):
    """Object-frame points -> symmetry-invariant coordinates (radius from the axis, height along it, 0)."""
    ax, c = axc
    q = Po - c
    h = q @ ax
    r = np.linalg.norm(q - h[..., None] * ax, axis=-1)
    return np.stack([r, h, np.zeros_like(r)], -1)


def agreement(tr, tracks: dict, min_pts: int = 8, run_name: str | None = None) -> dict:
    """Per object, per frame: robust drift (m) of its tracked surface points in the object frame (for a
    continuously symmetric object only radius + height along the axis: its spin is unobservable)."""
    names = [str(x) for x in tracks["names"]]
    Q, owner, Pw = tracks["queries"], tracks["owner"], tracks["P_world"].astype(float)
    F = tr.F
    out = {}
    for k, n in enumerate(names):
        if n not in tr.objects:
            continue
        o = tr.objects[n]
        idx = np.where(owner == k)[0]
        if not len(idx):
            continue
        Po = np.full((F, len(idx), 3), np.nan)
        for f in np.where(o["valid"])[0]:
            Po[f] = transform(inv(o["T_world_obj"][f]), Pw[f, idx])
        axc = sym_axis(run_name, n) if run_name else None
        if axc is not None:
            Po = sym_coords(Po, axc)
        q_t = Q[idx, 0].astype(int)
        ref = np.array([Po[q_t[j], j] for j in range(len(idx))])  # each point at its seed frame
        drift = np.linalg.norm(Po - ref[None], axis=2)  # (F, P)
        # robust: median over points, ignoring points that are mostly far from the object (mis-seeded)
        bad_pt = np.nanmedian(drift, axis=0) > 0.05
        drift[:, bad_pt] = np.nan
        cnt = np.isfinite(drift).sum(1)
        res = np.where(cnt >= min_pts, np.nanmedian(np.where(np.isfinite(drift), drift, np.nan), axis=1), np.nan)
        out[n] = {"rigid_resid_m": res, "n_points": cnt, "threshold_m": 0.015}
    return out
