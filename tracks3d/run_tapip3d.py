"""TAPIP3D 3D point tracks for every object and hand of an episode, in our SLAM world frame.

Inputs (all ours, no mocap): undistorted colour frames (recon_hawor/frames), metric stereo depth in
colour-camera pixels (depth_stereo), intrinsics (intrinsics_stable.json), DROID-SLAM camera-to-world
poses (recon_hawor/slam_poses) -> TAPIP3D RGB-D mode (extrinsics = world-to-camera; MegaSaM not used).
Queries: K points sampled inside the (eroded) SAM 3 mask of each target -- objects_sam3/masks/<id> and
stereo/hands_sam3/masks/a/{left,right} -- every --stride frames, lifted to world with the stereo depth.
TAPIP3D tracks each query forward and backward over the whole clip.
Output <run>/tracks3d/tapip3d.npz: coords (T,N,3) world, visibs (T,N), queries (N,4) [t,x,y,z],
query_px (N,2) full-res pixels, label (N,) index into labels (names: object names, hand_left, hand_right).
Usage (in tapip3d.sif, cwd = TAPIP3D repo): python run_tapip3d.py <run> [--stride 15] [--k 48]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path("/scratch/kghasemz/v2d")
sys.path.insert(0, str(ROOT / "third_party/TAPIP3D"))
from datasets.data_ops import _filter_one_depth  # noqa: E402
from utils.inference_utils import inference, load_model, resize_depth_bilinear  # noqa: E402


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


def decode_depth(path):
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    png = png.astype(np.float32)
    d = np.where(png > 0, 65535.0 / np.maximum(png, 1) - 1.0, 0)
    return np.where(np.isfinite(d) & (d > 1e-3), d, 0).astype(np.float32)


def targets(run):
    out = [(o["name"], run / "objects_sam3/masks" / str(o["id"]))
           for o in json.loads((run / "objects_sam3/objects.json").read_text())]
    out += [(f"hand_{s}", run / "stereo/hands_sam3/masks/a" / s) for s in ("left", "right")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stride", type=int, default=15, help="query frames every N frames")
    ap.add_argument("--k", type=int, default=48, help="points per target per query frame")
    ap.add_argument("--erode", type=int, default=7, help="mask erosion (px, full res) to avoid edge points")
    ap.add_argument("--resolution_factor", type=float, default=2.0)
    ap.add_argument("--num_iters", type=int, default=6)
    ap.add_argument("--support_grid_size", type=int, default=16)
    ap.add_argument("--max_frames", type=int, default=0, help="pilot: only the first N frames")
    ap.add_argument("--out", default="tracks3d/tapip3d.npz")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    frames = sorted((run / "recon_hawor/frames").glob("*.png"))
    if a.max_frames:
        frames = frames[:a.max_frames]
    T = len(frames)
    ki = json.loads((run / "intrinsics_stable.json").read_text())
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1]], np.float64)
    c2w = np.stack([load_T(run / f"recon_hawor/slam_poses/{t:06d}.json") for t in range(T)])
    video = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in frames])
    H, W = video.shape[1:3]
    depths = []
    for t in range(T):
        d = decode_depth(run / f"depth_stereo/{t:06d}.png")
        depths.append(np.zeros((H, W), np.float32) if d is None else d)
    depths = np.stack(depths)
    print(f"{a.run}: {T} frames {W}x{H}, depth coverage {np.mean(depths > 0):.2f}", flush=True)

    # queries: points inside each target's eroded mask with valid depth, lifted to world
    rng = np.random.default_rng(0)
    names = [n for n, _ in targets(run)]
    Q, QPX, LAB = [], [], []
    ker = np.ones((2 * a.erode + 1, 2 * a.erode + 1), np.uint8)
    Kinv = np.linalg.inv(K)
    for li, (name, mdir) in enumerate(targets(run)):
        nq = 0
        for t in range(0, T, a.stride):
            m = cv2.imread(str(mdir / f"{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.erode((m > 0).astype(np.uint8), ker) > 0
            ys, xs = np.nonzero(m & (depths[t] > 0))
            if len(xs) < a.k:
                continue
            sel = rng.choice(len(xs), a.k, replace=False)
            u, v = xs[sel].astype(np.float64), ys[sel].astype(np.float64)
            z = depths[t][ys[sel], xs[sel]].astype(np.float64)
            pc = (Kinv @ np.stack([u, v, np.ones_like(u)])) * z
            pw = (c2w[t][:3, :3] @ pc + c2w[t][:3, 3:4]).T
            Q.append(np.c_[np.full(a.k, t), pw]); QPX.append(np.c_[u, v]); LAB.append(np.full(a.k, li))
            nq += a.k
        print(f"  {name}: {nq} queries", flush=True)
    Q, QPX, LAB = np.concatenate(Q), np.concatenate(QPX), np.concatenate(LAB)

    # TAPIP3D preprocessing (as its inference.py: resize, scale intrinsics, filter depth edges)
    dev = "cuda"
    model = load_model(str(ROOT / "third_party/TAPIP3D/checkpoints/tapip3d_final.pth")).to(dev)
    res = (int(model.image_size[0] * np.sqrt(a.resolution_factor)), int(model.image_size[1] * np.sqrt(a.resolution_factor)))
    model.set_image_size(res)
    intr = np.repeat(K[None], T, 0).astype(np.float32)
    intr[:, 0, :] *= (res[1] - 1) / (W - 1)
    intr[:, 1, :] *= (res[0] - 1) / (H - 1)
    vid = np.stack([cv2.resize(f, (res[1], res[0]), interpolation=cv2.INTER_LINEAR) for f in video])
    dep = np.stack([resize_depth_bilinear(d, (res[1], res[0])) for d in depths])
    dep = np.stack([_filter_one_depth(d, 0.08, 15, k) for d, k in zip(dep, intr)])
    del video, depths
    w2c = np.linalg.inv(c2w).astype(np.float32)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        coords, visibs = inference(
            model=model,
            video=(torch.from_numpy(vid).permute(0, 3, 1, 2).float() / 255.0).to(dev),
            depths=torch.from_numpy(dep).float().to(dev),
            intrinsics=torch.from_numpy(intr).to(dev),
            extrinsics=torch.from_numpy(w2c).to(dev),
            query_point=torch.from_numpy(Q).float().to(dev),
            num_iters=a.num_iters, grid_size=a.support_grid_size)
    print(f"tracked {len(Q)} points over {T} frames at {res} in {time.time() - t0:.0f} s, "
          f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    out = run / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, coords=coords.float().cpu().numpy(), visibs=visibs.cpu().numpy(), queries=Q,
                        query_px=QPX, label=LAB, labels=np.array(names), inference_res=np.array(res),
                        params=json.dumps(vars(a)))
    print("saved", out)


if __name__ == "__main__":
    main()
