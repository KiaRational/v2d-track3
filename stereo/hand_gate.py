"""Hand-gated rotation filter (user rule): an object only turns when the hand holding it turns.

Source = c14 (track_pose tp_recon). Start = robust average of the source rotation over the first <= START_N frames before
the first grasp (it rests there). Then per frame:
  held by hand h at t-1 and t (fusion_report grasp_segments, grasp_vlm.json intervals, or a finger joint within TOUCH_M
  of the surface; both hands -> the one whose
  finger joints are closer):  predicted R_t = dR_hand(t) @ R_{t-1}  (wrist/palm world rotation change, palm smoothed)
  not held:                    predicted R_t = R_{t-1}               (it does not turn)
  observation = the source rotation at t, as its closest mesh-symmetry equivalent to the prediction (a cup turning about
  its own axis is no turn).
  held: if the source's frame-to-frame turn agrees with the hand's turn within GATE_DEG, the object takes the
  source's turn (and is pulled 10 % toward the source's absolute pose when within RESYNC_DEG -- never across a flip);
  otherwise it takes the hand's turn. Not held: no turn.
Position = the source centroid at t (unchanged). No dropping, interpolating or flipping of frames.
Writes <run>/<out>/<obj>/poses (camera<-object) + hand_gate_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python hand_gate.py <run>... [--src tp_recon] [--out tpg_recon]"""
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
from handlock import ang, interp_poses, robust_pose, smooth_poses
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
GATE_DEG, TOUCH_M, START_N = 4.0, 0.02, 10
RESYNC_DEG, RESYNC_A = 25.0, 0.1
FINGER_J = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="tp_recon")
    ap.add_argument("--out", default="tpg_recon")
    ap.add_argument("--hands", default="hands_c")
    ap.add_argument("--gate", type=float, default=GATE_DEG)
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
            c = np.asarray(mesh.vertices).mean(0)
            S = trimesh.sample.sample_surface(mesh, 1500, seed=0)[0]
            group = [np.asarray(G) for G in mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")["group"]]
            held = [None] * n
            segs = [(g["side"], g["from"], g["to"]) for g in (rep.get(nm) if isinstance(rep.get(nm), dict) else {}).get("grasp_segments", [])]
            segs += [(sd, s, e) for sd, ivs in gv.get(nm, {}).get("intervals", {}).items() for s, e in ivs]
            for sd, s, e in segs:
                if sd not in Hr:
                    continue
                for t in range(max(0, s), min(n, e + 1)):
                    if held[t] is None or held[t] == sd:
                        held[t] = sd
                    else:
                        tree = cKDTree(S @ X[t][:3, :3].T + X[t][:3, 3])
                        d = {h: tree.query(Jw[h][t])[0].min() if np.isfinite(Jw[h][t]).all() else np.inf for h in (held[t], sd)}
                        held[t] = min(d, key=d.get)
            for t in range(n):   # missed grasps: fingers on the surface also count as holding
                if held[t] is not None:
                    continue
                tree = cKDTree(S @ X[t][:3, :3].T + X[t][:3, 3])
                d = {h: tree.query(Jw[h][t])[0].min() for h in Jw if np.isfinite(Jw[h][t]).all()}
                if d and min(d.values()) < TOUCH_M:
                    held[t] = min(d, key=d.get)
            # start: robust average of the source rotation over the first <= START_N frames before the first grasp
            # (each frame's closest symmetry equivalent to frame 0), not frame 0 alone
            f0 = []
            for t in range(min(START_N, n)):
                if held[t] is not None:
                    break
                f0.append(t)
            f0 = f0 or [0]
            Rs0 = np.array([min((X[t][:3, :3] @ G for G in group), key=lambda M: ang(M, X[0][:3, :3])) for t in f0])
            R0 = robust_pose(Rs0, np.zeros((len(f0), 3)), np.ones(len(f0)))[:3, :3]
            R = [R0]
            for t in f0[1:]:   # those frames keep the averaged pose (the object rests)
                R.append(R0.copy())
            first = len(R)
            # source rotations unwrapped along time (each frame's symmetry equivalent closest to the previous one), so a
            # frame-to-frame source change is a real change, not a relabelling
            O = [X[0][:3, :3]]
            for t in range(1, n):
                O.append(min((X[t][:3, :3] @ G for G in group), key=lambda M: ang(M, O[-1])))
            accepted = rejected = still = 0
            for t in range(first, n):
                h = held[t] if held[t] is not None and held[t] == held[t - 1] else None
                if h is None:
                    R.append(R[-1].copy())   # nobody turns it: it does not turn
                    still += 1
                    continue
                dH = Hr[h][t][:3, :3] @ Hr[h][t - 1][:3, :3].T     # how the hand turned (world)
                dO = O[t] @ O[t - 1].T                              # how the source says the object turned
                if ang(dO, dH) <= a.gate:
                    Rt = dO @ R[-1]
                    accepted += 1
                    # re-sync: pull gently toward the source's absolute pose when it is close (never across a flip)
                    Ot = min((X[t][:3, :3] @ G for G in group), key=lambda M: ang(M, Rt))
                    if ang(Ot, Rt) < RESYNC_DEG:
                        Rt = Slerp([0, 1], Rotation.from_matrix([Rt, Ot]))(RESYNC_A).as_matrix()
                else:
                    Rt = dH @ R[-1]   # the source's turn disagrees with the hand: the object turns with the hand
                    rejected += 1
                R.append(Rt)
            out = []
            for t in range(n):
                M = np.eye(4)
                M[:3, :3] = R[t]
                M[:3, 3] = X[t][:3, :3] @ c + X[t][:3, 3] - R[t] @ c   # source centroid kept
                out.append(M)
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ out[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st0 = [ang(X[t + 1][:3, :3], X[t][:3, :3]) for t in range(n - 1)]
            st = [ang(R[t + 1], R[t]) for t in range(n - 1)]
            report[nm] = {"held_frames": sum(x is not None for x in held), "still_frames": still,
                          "accepted": accepted, "rejected": rejected,
                          "total_rot_src_deg": round(sum(st0)), "total_rot_out_deg": round(sum(st)),
                          "max_step_src": round(max(st0), 1), "max_step_out": round(max(st), 1)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "hand_gate_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
