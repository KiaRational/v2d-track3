"""MANO hands -> joint-space targets for the articulated MuJoCo hand (agile image).

The MuJoCo hand (mj_track.py) is MANO's kinematic tree: a wrist body and 15 finger bodies at
MANO's rest joint positions (this subject's betas), each with 3 hinges (x, y, z). MANO's pose is
per-joint axis-angle in the parent frame with rest axes aligned to the hand frame, so each local
rotation is decomposed into intrinsic XYZ Euler angles = the hinge targets. Left hands are MANO's
right model mirrored in x: rest joints F J and rotations F R F (F = diag(-1, 1, 1)).

Self-check: joint positions rebuilt by forward kinematics from these angles are compared with
MANO's own joints; the finger ordering of MANO's pose vector is chosen by that check, so a wrong
mapping cannot pass silently.
Writes stereo/mj_prep_<hands>.json: per side rest joints (hand frame), per frame the world wrist
pose, 45 hinge targets, and the reference world joints.
Usage: python mj_prep.py <run> [--hands mano_fit]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
F = np.diag([-1.0, 1.0, 1.0])
# OpenPose-ordered joint (1..20, tips excluded) for each of MANO's 15 pose joints
ORDERS = {
    "index_middle_pinky_ring_thumb": [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3],
    "index_middle_ring_pinky_thumb": [5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 1, 2, 3],
}
PARENT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7, 9: 0, 10: 9, 11: 10, 12: 11,
          13: 0, 14: 13, 15: 14, 16: 15, 17: 0, 18: 17, 19: 18, 20: 19}


def load_T(p):
    d = json.loads(Path(p).read_text())
    q = d["rotation"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def fk(rest, Rroot, p0, Rloc):
    """Joint positions (21,3) from rest joints, root rotation/position, local rotations {j: R}."""
    J = np.zeros((21, 3))
    Rc = {0: Rroot}
    J[0] = p0
    for j in range(1, 21):
        pa = PARENT[j]
        J[j] = J[pa] + Rc[pa] @ (rest[j] - rest[pa])
        Rc[j] = Rc[pa] @ Rloc.get(j, np.eye(3))
    return J


# MANO's 16 skinning joints (wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3) -> our body index
MANO_TO_BODY = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]


def export_hand_parts(mano, betas, is_right, rest, outdir):
    """This subject's MANO mesh, rest pose, split into 16 rigid pieces by skinning weight (argmax bone),
    each expressed in its body's frame (rest joint at the origin, hand-aligned axes). Writes
    part_<body>.obj (the real MANO surface of that piece, for rendering) and hull_<body>.obj (its convex
    hull, for collision: a tight wrapper with no gaps). Left hands are the mirrored right model."""
    import trimesh
    z = {"mano": {"betas": betas, "global_orient": [0, 0, 0], "hand_pose": [0] * 45}, "is_right": True}
    v, _ = mano.local(z)
    V = v.cpu().numpy()
    Fc = mano.faces(True)
    Wt = getattr(mano.layer, "th_weights", None)
    if Wt is None:
        import pickle
        with open(Path(mano.layer.mano_assets_root) / "models/MANO_RIGHT.pkl", "rb") as f:
            Wt = pickle.load(f, encoding="latin1")["weights"]
    Wt = np.asarray(Wt.cpu() if hasattr(Wt, "cpu") else Wt)
    if not is_right:  # mirror: vertices in x, faces reversed to keep outward normals
        V = V @ F
        Fc = Fc[:, [0, 2, 1]]
    lab = Wt.argmax(1)
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {}
    for k, body in enumerate(MANO_TO_BODY):
        # a piece = its own vertices plus neighbours it strongly influences, so neighbouring hulls overlap
        sel = (lab == k) | (Wt[:, k] > 0.35)
        if sel.sum() < 4:
            continue
        faces = Fc[sel[Fc].all(1)]
        idx = np.unique(faces)
        remap = -np.ones(len(V), int)
        remap[idx] = np.arange(len(idx))
        local = V[idx] - rest[body]
        vis = trimesh.Trimesh(local, remap[faces], process=False)
        vis.export(outdir / f"part_{body}.obj")
        trimesh.Trimesh(V[sel] - rest[body]).convex_hull.export(outdir / f"hull_{body}.obj")
        parts[str(body)] = {"vis": str(outdir / f"part_{body}.obj"), "hull": str(outdir / f"hull_{body}.obj"),
                            "n_verts": int(sel.sum())}
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", default="mano_fit")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    mano = Mano(str(ROOT / "weights/hand"), "cuda")
    out = {"hands": a.hands, "sides": {}}
    for side in ("left", "right"):
        recs = {int(f.stem): json.loads(f.read_text()) for f in (run / "stereo" / a.hands / side).glob("*.json")
                if int(f.stem) in slam}
        if not recs:
            continue
        t0 = min(recs)
        is_right = recs[t0]["is_right"]
        betas = recs[t0]["mano"]["betas"]
        z = {"mano": {"betas": betas, "global_orient": [0, 0, 0], "hand_pose": [0] * 45}, "is_right": True}
        _, jr = mano.local(z)
        rest = jr.cpu().numpy()
        if not is_right:
            rest = rest @ F
        M = np.eye(3) if is_right else F
        frames, errs = {}, {k: [] for k in ORDERS}
        best = None
        for t in sorted(recs):
            r = recs[t]
            Rgo = M @ Rotation.from_rotvec(r["mano"]["global_orient"]).as_matrix() @ M
            hp = np.array(r["mano"]["hand_pose"]).reshape(15, 3)
            _, jm = mano.local(r)
            Jref = jm.cpu().numpy() + np.array(r["T"])  # camera frame
            p0 = rest[0] + np.array(r["T"])
            if best is None:  # choose the pose-vector ordering by forward-kinematics agreement
                for k, order in ORDERS.items():
                    Rloc = {j: M @ Rotation.from_rotvec(hp[i]).as_matrix() @ M for i, j in enumerate(order)}
                    errs[k].append(float(np.linalg.norm(fk(rest, Rgo, p0, Rloc) - Jref, axis=1).mean()))
                if len(errs[next(iter(ORDERS))]) >= 5:
                    best = min(ORDERS, key=lambda k: np.mean(errs[k]))
                    print(f"{side}: pose ordering {best}, FK error "
                          f"{ {k: round(1000 * np.mean(v), 2) for k, v in errs.items()} } mm", flush=True)
            order = ORDERS[best or "index_middle_pinky_ring_thumb"]
            Rloc = {j: M @ Rotation.from_rotvec(hp[i]).as_matrix() @ M for i, j in enumerate(order)}
            S = slam[t]
            frames[t] = {"root_R": (S[:3, :3] @ Rgo).tolist(), "root_p": (S[:3, :3] @ p0 + S[:3, 3]).tolist(),
                         "hinge": {str(j): Rotation.from_matrix(Rloc[j]).as_euler("XYZ").tolist() for j in Rloc},
                         "joints_world": (Jref @ S[:3, :3].T + S[:3, 3]).tolist()}
        # verify the chosen ordering over the whole sequence with the Euler round trip
        e = []
        for t, fr in frames.items():
            Rloc = {int(j): Rotation.from_euler("XYZ", v).as_matrix() for j, v in fr["hinge"].items()}
            J = fk(rest, np.array(fr["root_R"]), np.array(fr["root_p"]), Rloc)
            e.append(np.linalg.norm(J - np.array(fr["joints_world"]), axis=1).mean())
        print(f"{side}: {len(frames)} frames, FK round-trip error {1000 * np.mean(e):.2f} mm (max {1000 * np.max(e):.2f})",
              flush=True)
        parts = export_hand_parts(mano, betas, is_right, rest, run / "stereo" / f"mj_hand_{a.hands}" / side)
        print(f"{side}: MANO mesh hand, {len(parts)} rigid pieces", flush=True)
        out["sides"][side] = {"is_right": is_right, "betas": betas, "pose_order": best, "rest_joints": rest.tolist(),
                              "fk_error_mm": float(1000 * np.mean(e)), "frames": frames, "mesh_parts": parts}
    (run / "stereo" / f"mj_prep_{a.hands}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
