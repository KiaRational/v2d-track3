"""Stage 3: metric depth for the colour camera from the ego stereo pair.

FoundationStereo (CVPR'25, ViT-L "23-51-11", hierarchical full-res inference) on the
rectified mono pair (left=ego_cam_c, right=ego_cam_b); disparity -> metric depth; the
3D points are reprojected into the *undistorted* ego_cam_a frame used by the hand
pipeline (z-buffered splat + small-hole fill). Runs inside the
foundation_stereo_official image.

Outputs per episode run dir ($V2D_ROOT/runs/<name>/):
  depth_stereo/NNNNNN.png   toolkit DepthImage encoding: 65535/(depth_m+1), 65535 (=0 m) = invalid
  depth_stereo_stats.json   coverage, depth range, median stereo/MoGe ratio per frame
Viz: $V2D_VIZ/03_stereo_depth/<name>.mp4  (colour | stereo depth | MoGe depth)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/opt/FoundationStereo")
from core.foundation_stereo import FoundationStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
DATA = Path(os.environ["V2D_DATA"])
VIZ = Path(os.environ["V2D_VIZ"]) / "03_stereo_depth"
CKPT = ROOT / "weights/foundation_stereo_official/23-51-11/model_best_bp2.pth"
Z_MIN, Z_MAX = 0.1, 6.0  # metres; ego workspace plus room


def load_model(valid_iters):
    cfg = OmegaConf.load(CKPT.parent / "cfg.yaml")
    cfg.setdefault("vit_size", "vitl")
    cfg["valid_iters"] = valid_iters
    model = FoundationStereo(OmegaConf.create(cfg))
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    return model.cuda().eval()


@torch.no_grad()
def disparity(model, left, right, iters, hiera):
    t = lambda im: torch.as_tensor(np.repeat(im[..., None], 3, axis=2)).cuda().float()[None].permute(0, 3, 1, 2)
    a, b = t(left), t(right)
    padder = InputPadder(a.shape, divis_by=32, force_square=False)
    a, b = padder.pad(a, b)
    with torch.autocast("cuda", enabled=True):
        if hiera:
            d = model.run_hierachical(a, b, iters=iters, test_mode=True, small_ratio=0.5)
        else:
            d = model.forward(a, b, iters=iters, test_mode=True)
    return padder.unpad(d.float()).cpu().numpy()[0, 0]


def to_cam_a(disp, P1, R1, baseline, T_c2a, Ka, size_a):
    """Rectified-left disparity -> depth map in the undistorted ego_cam_a image."""
    h, w = disp.shape
    fx, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    v, u = np.mgrid[0:h, 0:w]
    valid = (disp > 0.5) & (u - disp >= 0)  # visible in both views
    z = np.where(valid, fx * baseline / np.maximum(disp, 1e-6), 0)
    valid &= (z > Z_MIN) & (z < Z_MAX)
    pts = np.stack([(u - cx) / fx * z, (v - cy) / fx * z, z], -1)[valid]  # rectified-left frame
    pts = pts @ R1  # R1 maps cam_c -> rectified, so R1^T back to cam_c (row-vector form)
    pts = pts @ T_c2a[:3, :3].T + T_c2a[:3, 3]
    pts = pts[pts[:, 2] > Z_MIN]
    wa, ha = size_a
    uv = pts[:, :2] / pts[:, 2:3] @ Ka[:2, :2].T + Ka[:2, 2]
    depth = np.full(ha * wa, np.inf, np.float32)
    # 2x2 splat: the colour image is denser than the stereo pair in the overlap.
    for du in (0, 1):
        for dv in (0, 1):
            x, y = np.floor(uv[:, 0]).astype(int) + du, np.floor(uv[:, 1]).astype(int) + dv
            ok = (x >= 0) & (x < wa) & (y >= 0) & (y < ha)
            np.minimum.at(depth, y[ok] * wa + x[ok], pts[ok, 2].astype(np.float32))
    depth = depth.reshape(ha, wa)
    # Fill only small holes with the nearest surface (min filter), keep measured values.
    holes = ~np.isfinite(depth)
    filled = cv2.erode(np.where(holes, 1e6, depth).astype(np.float32), np.ones((5, 5), np.uint8))
    depth = np.where(holes & (filled < 1e5), filled, depth)
    depth[~np.isfinite(depth)] = 0
    return depth


def encode(depth):
    # Invalid -> 65535, which the toolkit decodes to exactly 0 m; FoundationPose
    # ignores depth < 1 mm. (Pixel 0 would decode to +inf and poison its xyz maps.)
    enc = np.where(depth > 0, 65535.0 / (depth + 1.0), 65535)
    return enc.clip(1, 65535).astype(np.uint16)


def decode(png):
    """Toolkit DepthImage decoding; returns 0 for invalid (0 m or +inf)."""
    png = png.astype(np.float32)
    depth = np.where(png > 0, 65535.0 / np.maximum(png, 1) - 1.0, 0)
    return np.where(np.isfinite(depth) & (depth > 1e-3), depth, 0)


def colorize(depth, lo=0.2, hi=2.5):
    norm = np.clip((depth - lo) / (hi - lo), 0, 1)
    col = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[depth <= 0] = 0
    return col


def process(name, model, iters, hiera):
    run = ROOT / "runs" / name
    meta = json.loads((run / "episode.json").read_text())
    calib = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
    st, cams = calib["stereo"], calib["cameras"]
    size = tuple(cams["ego_cam_c"]["size"])
    maps = {}
    for cam, R, P in (("ego_cam_c", st["R1"], st["P1"]), ("ego_cam_b", st["R2"], st["P2"])):
        c = cams[cam]
        maps[cam] = cv2.initUndistortRectifyMap(np.array(c["K"]), np.array(c["dist"]), np.array(R),
                                                np.array(P), size, cv2.CV_32FC1)
    P1, R1 = np.array(st["P1"]), np.array(st["R1"])
    T_c2a = np.linalg.inv(np.array(calib["T_ego_cam_a_to_ego_cam_c"]))
    Ka, size_a = np.array(calib["undistorted_ego_cam_a_K"]), tuple(cams["ego_cam_a"]["size"])
    baseline = st["baseline_m"]

    split, ep = meta["split"], meta["episode"]
    vid = lambda k: cv2.VideoCapture(str(DATA / f"{split}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4"))
    cap_l, cap_r, cap_a = vid("ego_cam_c"), vid("ego_cam_b"), cv2.VideoCapture(str(run / "input_undistorted.mp4"))
    out_dir = run / "depth_stereo"
    out_dir.mkdir(exist_ok=True)
    moge_dir = run / "recon_hawor/depth"
    VIZ.mkdir(parents=True, exist_ok=True)
    vw, vh = 512, 384
    writer = cv2.VideoWriter(str(VIZ / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (3 * vw, vh))
    stats, t = [], 0
    while True:
        okl, fl = cap_l.read()
        okr, fr = cap_r.read()
        oka, fa = cap_a.read()
        if not (okl and okr and oka):
            break
        gl = cv2.remap(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), *maps["ego_cam_c"], cv2.INTER_LINEAR)
        gr = cv2.remap(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), *maps["ego_cam_b"], cv2.INTER_LINEAR)
        disp = disparity(model, gl, gr, iters, hiera)
        depth = to_cam_a(disp, P1, R1, baseline, T_c2a, Ka, size_a)
        cv2.imwrite(str(out_dir / f"{t:06d}.png"), encode(depth))

        s = {"frame": t, "coverage": float((depth > 0).mean()),
             "median_depth_m": float(np.median(depth[depth > 0])) if (depth > 0).any() else None}
        moge_png = moge_dir / f"{t:06d}.png"
        moge = decode(cv2.imread(str(moge_png), cv2.IMREAD_UNCHANGED)) if moge_png.exists() else None
        if moge is not None:
            both = (depth > 0) & (moge > 0) & (depth < 2.5)
            if both.sum() > 1000:
                s["stereo_over_moge_median"] = float(np.median(depth[both] / moge[both]))
        stats.append(s)
        panels = [cv2.resize(fa, (vw, vh)), cv2.resize(colorize(depth), (vw, vh), interpolation=cv2.INTER_NEAREST),
                  cv2.resize(colorize(moge), (vw, vh), interpolation=cv2.INTER_NEAREST) if moge is not None
                  else np.zeros((vh, vw, 3), np.uint8)]
        row = np.hstack(panels)
        for i, txt in enumerate(("ego_cam_a (undistorted)", "FoundationStereo metric depth", "MoGe (toolkit baseline)")):
            cv2.putText(row, txt, (i * vw + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(row, f"frame {t}  coverage {s['coverage']:.0%}  0.2m (red) .. 2.5m (blue)", (vw + 8, vh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        writer.write(row)
        t += 1
    writer.release()
    ratios = [s["stereo_over_moge_median"] for s in stats if "stereo_over_moge_median" in s]
    summary = {"frames": t, "mean_coverage": float(np.mean([s["coverage"] for s in stats])),
               "stereo_over_moge_median_of_medians": float(np.median(ratios)) if ratios else None,
               "per_frame": stats, "model": "FoundationStereo 23-51-11 (ViT-L)", "valid_iters": iters, "hiera": hiera}
    (run / "depth_stereo_stats.json").write_text(json.dumps(summary, indent=2))
    return f"{name}: {t} frames, coverage {summary['mean_coverage']:.0%}, stereo/MoGe {summary['stereo_over_moge_median_of_medians']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="run dir names, e.g. public_ep000011")
    ap.add_argument("--valid_iters", type=int, default=32)
    ap.add_argument("--hiera", type=int, default=1)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model = load_model(args.valid_iters)
    for name in args.names:
        print(process(name, model, args.valid_iters, args.hiera), flush=True)


if __name__ == "__main__":
    main()
