"""Remove the unobservable twist of rotationally symmetric objects (cups) about their symmetry axis.

A continuous-symmetric mesh looks the same at every angle about its axis, so the pose estimate's twist is arbitrary;
the earlier steps snapped it to a sampled symmetry group (5 deg in pose_filter, 15 deg in spin_guard) -> a few degrees
of random twist per frame. The simulator reward is not symmetry-aware, so the robot must reproduce that twist and loses
the cup (eval ep32: cup twisting in the fingers, early termination). Here every frame gets the EXACT angle phi about the
axis (through the centroid) that keeps the pose closest to the previous output frame: R'_t = R_t Rot(a, phi_t),
phi_t = argmax trace(R'_{t-1}^T R_t Rot(a, phi)) (closed form). Centroid kept. Tilt (the observable part) unchanged.
Non-symmetric objects are copied unchanged. Writes <run>/<out>/<obj>/poses + twist_unwrap_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python twist_unwrap.py <run>... [--src tpfs_recon] [--out tpfsu_recon]"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from handlock import ang
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])


def rot_axis(a, phi):
    return Rotation.from_rotvec(a * phi).as_matrix()


def best_phi(Rt, Rprev, a):
    """argmax_phi trace(Rprev^T Rt Rot(a, phi)) = trace(M Rot(a, phi)), M = Rprev^T Rt.
    Rot = c I + s [a]x + (1 - c) a a^T  ->  trace(M Rot) = c (tr M - a^T M a) + s tr(M [a]x) + a^T M a."""
    M = Rprev.T @ Rt
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    A = np.trace(M) - a @ M @ a
    B = np.trace(M @ K)
    return float(np.arctan2(B, A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="tpfs_recon")
    ap.add_argument("--out", default="tpfsu_recon")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if not src.is_dir():
                continue
            od = run / a.out / nm / "poses"
            mp = ROOT / mesh_dir / nm / f"{nm}.obj"
            sym = mesh_symmetry(mp)
            if sym["cont_axis"] is None:
                shutil.rmtree(od, ignore_errors=True)
                shutil.copytree(src, od)
                report[nm] = {"symmetry": sym["kind"], "changed": False}
                continue
            ax = np.asarray(sym["cont_axis"], float)
            ax /= np.linalg.norm(ax)
            c = np.asarray(trimesh.load(mp, force="mesh").vertices).mean(0)
            X = [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]
            out = [X[0].copy()]
            twist = []
            for t in range(1, n):
                phi = best_phi(X[t][:3, :3], out[-1][:3, :3], ax)
                R = X[t][:3, :3] @ rot_axis(ax, phi)
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = X[t][:3, :3] @ c + X[t][:3, 3] - R @ c   # same centroid
                out.append(M)
                twist.append(abs(np.degrees(phi)))
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ out[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st0 = [ang(X[t + 1][:3, :3], X[t][:3, :3]) for t in range(n - 1)]
            st = [ang(out[t + 1][:3, :3], out[t][:3, :3]) for t in range(n - 1)]
            report[nm] = {"symmetry": sym["kind"], "changed": True,
                          "total_rot_before_deg": round(sum(st0)), "total_rot_after_deg": round(sum(st)),
                          "max_step_before": round(max(st0), 1), "max_step_after": round(max(st), 1),
                          "twist_removed_med_deg": round(float(np.median(twist)), 1)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "twist_unwrap_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
