"""TrackEverything 3D point tracks on OBJECT surfaces only, in the SLAM world (for track_pose.py).

Queries per object every --every frames: --per points stratified over the object's SAM 3 mask (objects_recon masks),
eroded, minus the (dilated) hand masks, minus stereo-depth edges (depth range in a 7x7 window > 1.5 cm + 2 % z) and
pixels without depth -- seeds sit on clearly visible, depth-reliable object surface, never on a finger.
Inference exactly as hoi4d.tracking.trackeverything.infer (supplied geometry: our stereo depth, K, camera poses;
geometry_backend "pi3" selects the cam0+norm branch) but with the SLAM camera poses, so P_world is in the same world as
recon_hawor/slam_poses. A query is tracked forward from its own frame (NaN before it).
Writes <run>/<obj_dir>/te_tracks.npz {P_world (F,N,3), vis (F,N), queries (N,3) f,u,v, owner (N,) index into names,
names}, + te_tracks_report.json (per object: seeds, seed-frame error vs depth).
Runs in the trackeverything image. Usage: python te_obj_tracks.py <run> [--every 5] [--per 64]"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))


def decode_depth(png):
    d = 65535.0 / np.maximum(png.astype(np.float64), 1) - 1.0
    d[(png == 0) | (png >= 65535) | ~np.isfinite(d)] = 0.0
    return d


def shift_and(m, r):
    out = m.copy()
    for ax in (0, 1):
        for s in range(1, r + 1):
            out &= np.roll(m, s, ax) & np.roll(m, -s, ax)
    return out


def shift_or(m, r):
    out = m.copy()
    for ax in (0, 1):
        for s in range(1, r + 1):
            out |= np.roll(m, s, ax) | np.roll(m, -s, ax)
    return out


def depth_edges(D, r=3):
    big = np.where(D > 0, D, np.nan)
    mx, mn = np.full_like(D, -np.inf), np.full_like(D, np.inf)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            s = np.roll(np.roll(big, dy, 0), dx, 1)
            mx = np.fmax(mx, s)
            mn = np.fmin(mn, s)
    return (mx - mn) > (0.015 + 0.02 * np.nan_to_num(D))


def load_T(p):
    """Our variant files {R, T} or toolkit Transform3d {rotation (w-first quaternion), translation} (SLAM poses)."""
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.asarray(d["R"]), d["T"]
        return T
    w, x, y, z = np.asarray(d["rotation"], float) / np.linalg.norm(d["rotation"])
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = d["translation"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--obj_dir", default="objects_recon")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--per", type=int, default=64)
    ap.add_argument("--input_scale", type=float, default=0.5)
    ap.add_argument("--num_iters", type=int, default=3)
    ap.add_argument("--checkpoint", default=str(ROOT / "weights/trackeverything/model.pth"))
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    k = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    F = len(list((run / "recon_hawor/frames").glob("*.png")))
    T_wc = np.full((F, 4, 4), np.nan)
    for p in (run / "recon_hawor/slam_poses").glob("*.json"):
        if int(p.stem) < F:
            T_wc[int(p.stem)] = load_T(p)
    ok = np.array([np.isfinite(T).all() for T in T_wc])
    idx = np.where(ok)[0]
    for f in np.where(~ok)[0]:
        T_wc[f] = T_wc[idx[np.argmin(abs(idx - f))]]
    W, H = Image.open(run / "recon_hawor/frames/000000.png").size
    objs = json.loads((run / a.obj_dir / "objects.json").read_text())
    names = [o["name"] for o in objs]
    rng = np.random.default_rng(0)
    Q, owner, rep = [], [], {}
    for f in range(0, F, a.every):
        dp = run / "depth_stereo" / f"{f:06d}.png"
        if not dp.exists():
            continue
        D = decode_depth(np.asarray(Image.open(dp)))
        edge = depth_edges(D)
        hm = np.zeros((H, W), bool)
        for hid in (2, 3):
            p = run / a.obj_dir / "masks" / str(hid) / f"{f:06d}.png"
            if p.exists():
                hm |= np.asarray(Image.open(p)) > 0
        hm = shift_or(hm, 8)
        for k_, o in enumerate(objs):
            p = run / a.obj_dir / "masks" / str(o["id"]) / f"{f:06d}.png"
            if not p.exists():
                continue
            m = shift_and(np.asarray(Image.open(p)) > 0, 6) & ~hm & ~edge & (D > 0.1) & (D < 2.5)
            ys, xs = np.nonzero(m)
            if len(xs) < 40:
                continue
            # stratified: one random pixel per occupied cell of a grid sized for ~per cells
            cell = max(4, int(np.sqrt(len(xs) / a.per)))
            key = (ys // cell) * (W // cell + 1) + xs // cell
            order = rng.permutation(len(xs))
            _, first = np.unique(key[order], return_index=True)
            pick = order[first]
            if len(pick) > a.per:
                pick = rng.choice(pick, a.per, replace=False)
            Q += [(f, float(xs[i]), float(ys[i])) for i in pick]
            owner += [k_] * len(pick)
            rep.setdefault(o["name"], {"seeds": 0, "seed_frames": 0})
            rep[o["name"]]["seeds"] += len(pick)
            rep[o["name"]]["seed_frames"] += 1
    Q = np.asarray(Q, np.float32)
    owner = np.asarray(owner, np.int32)
    print(a.run, "queries", len(Q), json.dumps(rep), flush=True)

    import torch  # noqa: F401
    from trackeverything.demo_hf import TrackEverything
    s = a.input_scale
    w, h = int(W * s) // 4 * 4, int(H * s) // 4 * 4
    sx, sy = w / W, h / H
    video = np.stack([np.asarray(Image.open(run / "recon_hawor/frames" / f"{i:06d}.png").convert("RGB").resize((w, h), Image.BILINEAR))
                      for i in range(F)])
    depths = np.zeros((F, h, w), np.float32)
    for i in range(F):
        dp = run / "depth_stereo" / f"{i:06d}.png"
        if dp.exists():
            depths[i] = np.asarray(Image.fromarray(decode_depth(np.asarray(Image.open(dp))).astype(np.float32)).resize((w, h), Image.NEAREST))
    Ks = K.copy()
    Ks[0] *= sx
    Ks[1] *= sy
    E = np.linalg.inv(T_wc).astype(np.float32)
    Qs = Q.copy()
    Qs[:, 1] *= sx
    Qs[:, 2] *= sy
    model = TrackEverything.from_pretrained(a.checkpoint, device="cuda", geometry_backend="pi3")
    out = model.infer(video, depths=depths, intrinsics=np.repeat(Ks[None], F, 0).astype(np.float32), extrinsics=E,
                      query_points_2d=Qs, mode="sparse", num_iters=a.num_iters)
    C = out.query_coords[0].float().cpu().numpy()
    Vl = out.query_visibs[0].float().cpu().numpy()
    Vp = Vl if (Vl.min() >= 0 and Vl.max() <= 1) else 1 / (1 + np.exp(-Vl))
    Pw = (C * out.norm_factor) @ T_wc[0][:3, :3].T + T_wc[0][:3, 3]
    t0 = Q[:, 0].astype(int)
    before = np.arange(F)[:, None] < t0[None]
    Pw[before] = np.nan
    vis = (Vp > 0.5) & ~before
    # seed-frame check vs our depth
    err, dcache = {}, {}
    for i, (f, u, v) in enumerate(Q):
        f = int(f)
        if f not in dcache:
            dcache[f] = decode_depth(np.asarray(Image.open(run / "depth_stereo" / f"{f:06d}.png")))
        D = dcache[f]
        z = D[int(round(v)), int(round(u))]
        X = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        err.setdefault(names[owner[i]], []).append(np.linalg.norm(T_wc[f][:3, :3] @ X + T_wc[f][:3, 3] - Pw[f, i]))
    for nm, e in err.items():
        rep[nm]["seed_err_mm_median"] = round(float(np.median(e)) * 1000, 2)
        rep[nm]["visible_frac"] = round(float(vis[:, owner == names.index(nm)].mean()), 3)
    np.savez_compressed(run / a.obj_dir / "te_tracks.npz", P_world=Pw.astype(np.float32), vis=vis, queries=Q,
                        owner=owner, names=np.array(names), license=np.array("TrackEverything: CC-BY-NC-4.0"))
    (run / a.obj_dir / "te_tracks_report.json").write_text(json.dumps(rep, indent=1))
    print(a.run, json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
