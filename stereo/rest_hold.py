"""Rest hold: an object no hand is touching does not move -- no turning on the table (eval ep10 cup), no spinning in
place (eval ep19 sweeper). Final step after pose_filter (tpf_recon -> tpfr_recon).

Touched at t = a grasp covers t (grasp_vlm.json intervals if present, + fusion_report grasp_segments) OR a finger joint of
either hand (hands variant, thumb + finger joints) is within TOUCH_M of the object surface at the source pose; dilated
by PAD frames. Every untouched stretch of >= MIN_LEN frames whose source centre stays within MOVE_M of its median (an
untouched object that really travels -- a lid riding on a carried pot, a missed grasp -- is left alone) is set to ONE
pose: visibility-weighted robust rotation (IRLS from the weighted medoid) and median position of the stretch. Edges
blended over BLEND frames. Writes <run>/<out>/<obj>/poses (camera<-object) + rest_hold_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python rest_hold.py <run>... [--src tpf_recon] [--out tpfr_recon]"""
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
from handlock import ang, robust_pose
from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])
TOUCH_M, PAD, MIN_LEN, MOVE_M, BLEND = 0.02, 3, 4, 0.02, 3
FINGER_J = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="tpf_recon")
    ap.add_argument("--out", default="tpfr_recon")
    ap.add_argument("--hands", default="hands_c")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        hands = stereo_hands(run, n, a.hands)
        Jw = {sd: np.array([hd["J"][t][FINGER_J] @ slam[t][:3, :3].T + slam[t][:3, 3] for t in range(n)]) for sd, hd in hands.items()}
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
            S = trimesh.sample.sample_surface(trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh"), 2000, seed=0)[0]
            cv = next(iter(sorted((run / obj_dir / nm).glob(".contact_vis_*.npz"))), None)
            vis = np.load(cv)["vis"][:n] if cv is not None else np.ones(n)
            touched = np.zeros(n, bool)
            for g in (rep.get(nm, {}) if isinstance(rep.get(nm), dict) else {}).get("grasp_segments", []):
                touched[max(0, g["from"]):min(n, g["to"] + 1)] = True
            for ivs in gv.get(nm, {}).get("intervals", {}).values():
                for s, e in ivs:
                    touched[max(0, s):min(n, e + 1)] = True
            for t in range(n):
                if touched[t]:
                    continue
                tree = cKDTree(S @ X[t][:3, :3].T + X[t][:3, 3])
                for J in Jw.values():
                    if np.isfinite(J[t]).all() and tree.query(J[t])[0].min() < TOUCH_M:
                        touched[t] = True
                        break
            touched = np.convolve(touched.astype(float), np.ones(2 * PAD + 1), "same") > 0
            out = [T.copy() for T in X]
            held, frozen, moving = [], 0, 0
            t = 0
            while t < n:
                if touched[t]:
                    t += 1
                    continue
                s = t
                while t < n and not touched[t]:
                    t += 1
                e = t
                if e - s < MIN_LEN:
                    continue
                c = np.array([X[u][:3, :3] @ S.mean(0) + X[u][:3, 3] for u in range(s, e)])
                if np.linalg.norm(c - np.median(c, 0), axis=1).max() > MOVE_M:
                    moving += e - s
                    continue
                w = np.array([max(float(vis[u]), 0.05) ** 2 for u in range(s, e)])
                P = robust_pose(np.array([X[u][:3, :3] for u in range(s, e)]), np.array([X[u][:3, 3] for u in range(s, e)]), w)
                spin = max(ang(X[u][:3, :3], P[:3, :3]) for u in range(s, e))
                for u in range(s, e):
                    out[u] = P.copy()
                for t0, t1, toward in ((max(0, s - BLEND), s, "in"), (e, min(n, e + BLEND), "out")):
                    for u in range(t0, t1):
                        f = (u - t0 + 1) / (t1 - t0 + 1)
                        A, B = (X[u], P) if toward == "in" else (P, X[u])
                        M = np.eye(4)
                        M[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(f).as_matrix()
                        M[:3, 3] = (1 - f) * A[:3, 3] + f * B[:3, 3]
                        out[u] = M
                frozen += e - s
                held.append([s, e - 1, round(spin, 1)])
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for u in range(n):
                C = np.linalg.inv(slam[u]) @ out[u]
                (od / f"{u:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            report[nm] = {"touched_frames": int(touched.sum()), "frozen_frames": frozen, "untouched_moving_frames": moving,
                          "frozen_stretches": held,
                          "max_removed_deg": max([h[2] for h in held], default=0.0)}
            print(rn, nm, json.dumps({k: v for k, v in report[nm].items() if k != "frozen_stretches"}), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "rest_hold_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
