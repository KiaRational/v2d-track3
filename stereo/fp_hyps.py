"""Dump ALL FoundationPose pose hypotheses per frame (Hand2Robot-style multi-hypothesis input, arXiv 2610.15769).
Per frame and object: FoundationPose.register from scratch (viewpoint hypotheses -> refiner -> scorer) and save every
refined candidate with its ScoreNet score, instead of keeping only the argmax. Same inputs as s05_objects_v2's
FoundationPose call (input_undistorted.mp4, depth_stereo, objects_recon masks, 1014x760, depth masked to the object).
Writes <run>/<obj_dir>/<obj>/fp_hyps/NNNNNN.npz {poses (N,4,4) camera<-object at full-res K, scores (N,)}.
Runs in the v2d_foundation_pose image. Usage: python fp_hyps.py <run> [--objects a,b] [--stride 1] [--iteration 5]"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
from v2d.mesh.lib.mesh import Mesh

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
TW, TH = 1014, 760


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--obj_dir", default="objects_recon")
    ap.add_argument("--objects", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9, help="exclusive")
    ap.add_argument("--iteration", type=int, default=5)
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    K = CameraIntrinsics.load(str(run / "recon_hawor/intrinsics_stable.json")).to_matrix()
    cap = cv2.VideoCapture(str(run / "input_undistorted.mp4"))
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sx, sy = TW / W0, TH / H0
    Ks = K.copy()
    Ks[0] *= sx
    Ks[1] *= sy
    intr = CameraIntrinsics(fx=float(Ks[0, 0]), fy=float(Ks[1, 1]), cx=float(Ks[0, 2]), cy=float(Ks[1, 2]), width=TW, height=TH)
    frames = []
    for t in range(n):   # decode sequentially, keep only this chunk
        ok, f = cap.read()
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (TW, TH)) if ok and a.start <= t < a.end else None)
    objs = json.loads((run / a.obj_dir / "objects.json").read_text())
    want = set(a.objects.split(",")) if a.objects else None
    for o in objs:
        if want and o["name"] not in want:
            continue
        tr = FoundationPoseTracker(Mesh.load(o["mesh"]), str(ROOT / "weights/foundation_pose"))
        est = tr._est
        od = run / a.obj_dir / o["name"] / "fp_hyps"
        od.mkdir(parents=True, exist_ok=True)
        done = 0
        for t in range(max(0, a.start), min(n, a.end), a.stride):
            mp = run / a.obj_dir / "masks" / str(o["id"]) / f"{t:06d}.png"
            dp = run / "depth_stereo" / f"{t:06d}.png"
            if frames[t] is None or not mp.exists() or not dp.exists() or (od / f"{t:06d}.npz").exists():
                continue
            m = cv2.resize(Mask.load(str(mp)).mask.astype(np.float32), (TW, TH), interpolation=cv2.INTER_NEAREST) > 0.5
            if m.sum() < 30:
                continue
            d = cv2.resize(DepthImage.load(str(dp)).depth, (TW, TH), interpolation=cv2.INTER_NEAREST)
            d = np.where(m, d, 0.0).astype(np.float32)
            with torch.no_grad():
                est.register(K=Ks, rgb=frames[t], depth=d, ob_mask=m, iteration=a.iteration)
            P = (est.poses @ est.get_tf_to_centered_mesh()).data.cpu().numpy()   # same convention as the saved best pose
            S = torch.as_tensor(est.scores).float().cpu().numpy()
            np.savez_compressed(od / f"{t:06d}.npz", poses=P.astype(np.float32), scores=S.astype(np.float32))
            done += 1
        print(a.run, o["name"], "frames", done, flush=True)


if __name__ == "__main__":
    main()
