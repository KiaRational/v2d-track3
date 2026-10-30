"""Spin guard: an object rotates only as the hand holding it rotates (+ a little slip); with no grasp it barely rotates.
Final step after rest_hold (tpfr_recon -> tpfs_recon).

Why: pose_filter bridges dropped frames by slerp, so a 180-deg branch change between two kept frames became a smooth
artificial spin at its speed limit (eval ep10 blue cup: ~170 deg in 12 frames at ~19.5 deg/frame while the hand only
held it; ep19 sweeper ~550 deg over 60 frames with no grasp).

Forward pass over frames, state rotation R_t (position from the source, so the object's centroid is unchanged):
  held by hand h (fusion_report grasp_segments / grasp_vlm.json intervals, side with the closest finger joint):
      predicted = dR_hand(t) @ R_{t-1}   (the palm's world rotation change, palm smoothed over 1.5 frames)
      step toward the target <= SLIP_DEG
  not held: predicted = R_{t-1}, step toward the target <= FREE_DEG
  target = the source rotation composed with the closest element of the mesh symmetry group (a cup turning about its
           own axis is no motion). Forward pass only, from the source's first-frame pose. Writes <run>/<out>/<obj>/poses + spin_guard_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python spin_guard.py <run>... [--src tpfr_recon] [--out tpfs_recon]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from fuse_objects import stereo_hands
from handlock import ang, interp_poses, smooth_poses
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
SLIP_DEG, FREE_DEG = 3.0, 1.0
FINGER_J = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]


def step_toward(R, T, max_deg):
    d = ang(R, T)
    if d <= max_deg or d < 1e-6:
        return T.copy()
    return Slerp([0, 1], Rotation.from_matrix([R, T]))(max_deg / d).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="tpfr_recon")
    ap.add_argument("--out", default="tpfs_recon")
    ap.add_argument("--hands", default="hands_c")
    ap.add_argument("--slip", type=float, default=SLIP_DEG)
    ap.add_argument("--free", type=float, default=FREE_DEG)
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        hands = stereo_hands(run, n, a.hands)
        Hr, Jw = {}, {}
        for sd, hd in hands.items():
            P = {t: slam[t] @ hd["H"][t] for t in range(n) if np.isfinite(hd["H"][t]).all()}
            if len(P) >= 2:
                Hr[sd] = interp_poses(smooth_poses(P, 1.5), n)
                Jw[sd] = np.array([hd["J"][t][FINGER_J] @ slam[t][:3, :3].T + slam[t][:3, 3] for t in range(n)])
        rp = run / f"fusion_report{suf}.json"
        rep = json.loads(rp.read_text()) if rp.exists() else {}
        gv = json.loads((run / "grasp_vlm.json").read_text()) if (run / "grasp_vlm.json").exists() else {}
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if not src.is_dir():
                continue
            X = [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]
            mesh = trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh")
            V = np.asarray(mesh.vertices)
            c = V.mean(0)
            _, _, ax = np.linalg.svd(V - c, full_matrices=False)
            S = trimesh.sample.sample_surface(mesh, 1500, seed=0)[0]
            # equivalents in the object frame about the centroid: flips x symmetry group (rotation part only; the
            # centroid is kept from the source below, so conjugation by the centroid is implicit)
            flips = [np.eye(3)] + [2 * np.outer(ax[k], ax[k]) - np.eye(3) for k in range(3)]
            group = [np.asarray(G) for G in mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")["group"]]
            if len(group) > 24:
                group = group[:: len(group) // 24]
            eq = group   # true symmetries only: a flipped equivalent could let the object sit on the wrong branch
            # who holds it
            held = [None] * n
            segs = [(g["side"], g["from"], g["to"]) for g in (rep.get(nm) if isinstance(rep.get(nm), dict) else {}).get("grasp_segments", [])]
            segs += [(sd, s, e) for sd, ivs in gv.get(nm, {}).get("intervals", {}).items() for s, e in ivs]
            for sd, s, e in segs:
                for t in range(max(0, s), min(n, e + 1)):
                    if sd not in Hr:
                        continue
                    if held[t] is None:
                        held[t] = sd
                    elif held[t] != sd:   # both: the hand whose fingers are closer
                        tree = cKDTree(S @ X[t][:3, :3].T + X[t][:3, 3])
                        dist = {h: tree.query(Jw[h][t])[0].min() if np.isfinite(Jw[h][t]).all() else np.inf for h in (held[t], sd)}
                        held[t] = min(dist, key=dist.get)

            def best_target(R_pred, t):
                Rs = X[t][:3, :3]
                return min((Rs @ E for E in eq), key=lambda T: ang(R_pred, T))

            def run_pass(order):
                R = {}
                prev = None
                for t in order:
                    if prev is None:
                        R[t] = X[t][:3, :3].copy()
                    else:
                        h = held[t] if held[t] is not None and held[t] == held[prev] else None
                        if h is not None:
                            dH = Hr[h][t][:3, :3] @ Hr[h][prev][:3, :3].T
                            pred, cap = dH @ R[prev], a.slip
                        else:
                            pred, cap = R[prev], a.free
                        R[t] = step_toward(pred, best_target(pred, t), cap)
                    prev = t
                return R
            Rf = run_pass(range(n))   # forward only, from the source's first-frame pose (one branch, no pass switching)
            out = []
            for t in range(n):
                Rt = Rf[t]
                M = np.eye(4)
                M[:3, :3] = Rt
                M[:3, 3] = X[t][:3, :3] @ c + X[t][:3, 3] - Rt @ c   # keep the source centroid
                out.append(M)
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ out[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st0 = [ang(X[t + 1][:3, :3], X[t][:3, :3]) for t in range(n - 1)]
            st = [ang(out[t + 1][:3, :3], out[t][:3, :3]) for t in range(n - 1)]
            dev = [min(ang(out[t][:3, :3], X[t][:3, :3] @ E) for E in eq) for t in range(n)]
            report[nm] = {"held_frames": sum(h is not None for h in held),
                          "total_rot_before_deg": round(sum(st0)), "total_rot_after_deg": round(sum(st)),
                          "max_step_before": round(max(st0), 1), "max_step_after": round(max(st), 1),
                          "dev_from_src_equiv_med_deg": round(float(np.median(dev)), 1),
                          "dev_from_src_equiv_p95_deg": round(float(np.percentile(dev, 95)), 1)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "spin_guard_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
