"""Stereo-ICP object poses: an independent geometric measurement per frame.

For every object and frame: the object's metric point cloud from FoundationStereo
depth (colour-camera frame, SAM2 object mask minus hand/other-object pixels) is
registered rigidly to the camera-facing surface of the known scanned mesh with
symmetric trimmed ICP (Kabsch updates), initialised from FoundationPose. The inlier
fitness and residual become the measurement's confidence for the trajectory fuser.

Outputs <run>/objects/<name>/stereo_icp/NNNNNN.json:
  {"T": 4x4 camera-from-object, "fitness": frac of scan pts within 5 mm,
   "rmse_mm": inlier RMSE, "n_scan": points used, "moved_from_init_mm": ...}
Usage (host env, CPU job): python object_icp.py <run_name> [--source poses_smoothed]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ["V2D_ROOT"])
OBJ = os.environ.get("V2D_OBJ_DIR", "objects")  # segmentation output to read
HAND_IDS = (2, 3)
INLIER_M, KEEP, ITERS = 0.005, 0.8, 40


def load_T(path):
    d = json.loads(Path(path).read_text())
    if "R" in d:  # our variant files: {"R": 3x3, "T": 3}
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]  # toolkit Transform3d, w-first
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def decode_depth(path):
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = np.where(png > 0, 65535.0 / np.maximum(png, 1) - 1.0, 0)
    return np.where(np.isfinite(d) & (d > 1e-3), d, 0)


def mask(path, shape):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 0) if m is not None else np.zeros(shape, bool)


def kabsch(P, Q):
    """Rigid R, t minimising ||R P + t - Q|| (rows are points)."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def icp(model_pts, model_nrm, scan, T0, max_iters=ITERS):
    """Symmetric trimmed ICP of the camera-facing model surface onto the scan."""
    tree_s = cKDTree(scan)
    T = T0.copy()
    for _ in range(max_iters):
        P = model_pts @ T[:3, :3].T + T[:3, 3]
        N = model_nrm @ T[:3, :3].T
        vis = (N * P).sum(1) < 0  # faces the camera (camera at origin)
        Pv = P[vis]
        if len(Pv) < 50:
            break
        d1, i1 = tree_s.query(Pv)
        k1 = np.argsort(d1)[: int(KEEP * len(d1))]
        d2, i2 = cKDTree(Pv).query(scan)
        k2 = np.argsort(d2)[: int(KEEP * len(d2))]
        src = np.r_[Pv[k1], Pv[i2[k2]]]
        dst = np.r_[scan[i1[k1]], scan[k2]]
        R, t = kabsch(src, dst)
        dT = np.eye(4)
        dT[:3, :3], dT[:3, 3] = R, t
        T = dT @ T
        if np.linalg.norm(t) < 1e-5 and np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) < 1e-4:
            break
    P = model_pts @ T[:3, :3].T + T[:3, 3]
    P = P[(model_nrm @ T[:3, :3].T * P).sum(1) < 0]
    d, _ = cKDTree(P).query(scan) if len(P) else (np.full(len(scan), np.inf), None)
    inl = d < INLIER_M
    return T, float(inl.mean()), float(np.sqrt((d[inl] ** 2).mean()) * 1000) if inl.any() else None


def process(args):
    run, name, obj_id, other_ids, source, init_variant, out_name, frames = args
    run = Path(run)
    K = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    shape = (K["height"], K["width"])
    mesh = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj", force="mesh")
    pts, fidx = trimesh.sample.sample_surface(mesh, 8000, seed=0)
    nrm = mesh.face_normals[fidx]
    out = run / OBJ / name / (out_name or "stereo_icp")
    out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    src = (run / init_variant / name / "poses") if init_variant else (run / OBJ / name / source)
    for pose_file in sorted(src.glob("*.json")):
        t = int(pose_file.stem)
        if frames and not (frames[0] <= t <= frames[1]):
            continue
        mo = mask(run / f"{OBJ}/masks/{obj_id}/{t:06d}.png", shape)
        occ = np.zeros(shape, bool)
        for oid in HAND_IDS + tuple(other_ids):
            occ |= mask(run / f"{OBJ}/masks/{oid}/{t:06d}.png", shape)
        # Erode away mask boundaries (mixed depth at silhouettes) and occluders.
        m = cv2.erode((mo & ~occ).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        depth = decode_depth(run / f"depth_stereo/{t:06d}.png")
        v, u = np.nonzero(m & (depth > 0))
        rec = {"frame": t, "n_scan": int(len(u))}
        if len(u) >= 150:
            if len(u) > 6000:
                sel = np.random.default_rng(t).choice(len(u), 6000, replace=False)
                u, v = u[sel], v[sel]
            z = depth[v, u]
            scan = np.stack([(u - Km[0, 2]) / Km[0, 0] * z, (v - Km[1, 2]) / Km[1, 1] * z, z], 1)
            T0 = load_T(pose_file)
            T, fit, rmse = icp(pts, nrm, scan, T0)
            rec.update(T=T.tolist(), fitness=fit, rmse_mm=rmse,
                       moved_from_init_mm=float(np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 1000))
            n_ok += 1
        (out / f"{t:06d}.json").write_text(json.dumps(rec))
    return f"{run.name}/{name}: {n_ok} frames registered"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--source", default="poses_smoothed", help="FoundationPose pose dir used for initialisation")
    ap.add_argument("--init_variant", default=None, help="initialise from this trajectory variant instead (agent)")
    ap.add_argument("--out_name", default=None, help="output dir under the object dir (default stereo_icp)")
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    ap.add_argument("--only_objects", nargs="*", default=None)
    a = ap.parse_args()
    jobs = []
    for r in a.runs:
        run = ROOT / "runs" / r
        objs = json.loads((run / OBJ / "objects.json").read_text())
        for o in objs:
            others = [x["id"] for x in objs if x["id"] != o["id"]]
            if a.only_objects and o["name"] not in a.only_objects:
                continue
            jobs.append((str(run), o["name"], o["id"], others, a.source, a.init_variant, a.out_name, a.frames))
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as ex:
        for msg in ex.map(process, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
