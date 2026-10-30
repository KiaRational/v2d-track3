"""Carve a container's collision so an object that really sat inside it is not stuck (eval ep30: the sweeper stands in a
slot of the dish rack; our reconstructed rack mesh / its convex pieces fill that slot, so in the simulator the sweeper
starts inside solid rack and cannot be lifted out).

From our reconstruction (same body frames as the simulator meshes -- checked: identical bounds) take every frame where
the intruder's centroid is inside the container's bounding box (expanded by PAD_M), map the intruder's surface points
into the container frame, and drop every convex collision piece of the container that contains any of them (margin
MARGIN_M). The real container is empty there -- the object occupied that space. The URDF is rewritten without those
<collision> entries (backup <name>_rigid.urdf.precarve); the piece files stay. URDFs are shared per object name in a
tree, so this applies to every sequence of the tree (physically correct: it is open space in the real rack).
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python carve_collision.py <tree_fine_dir> <run> <container> <intruder>
       [--objects tpfsut_recon]"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path.home() / "v2d-vulcan/stereo"))
from object_icp import load_T  # noqa: E402

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
PAD_M, MARGIN_M = 0.02, 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("run")
    ap.add_argument("container")
    ap.add_argument("intruder")
    ap.add_argument("--objects", default="tpfsut_recon")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    run = ROOT / "runs" / a.run
    proc = Path(a.tree) / "ego_recon/processed"
    urdf = proc / f"{a.container}_rigid.urdf"
    if not (proc / f"{a.container}_rigid.urdf.precarve").exists():
        shutil.copy(urdf, proc / f"{a.container}_rigid.urdf.precarve")
    txt = urdf.read_text()   # cumulative: several episodes may carve the same (shared) container
    blocks = re.findall(r"\s*<collision>.*?</collision>", txt, re.S)
    parts = []
    for b in blocks:
        f = re.search(r'filename="([^"]+)"', b).group(1)
        p = Path(f) if Path(f).is_absolute() else (proc / f).resolve()
        h = trimesh.load(p, force="mesh").convex_hull
        parts.append((b, np.asarray(h.face_normals), np.einsum("ij,ij->i", h.face_normals, h.triangles[:, 0])))
    C = trimesh.load(ROOT / mesh_dir / a.container / f"{a.container}.obj", force="mesh")
    lo, hi = C.bounds[0] - PAD_M, C.bounds[1] + PAD_M
    S = trimesh.sample.sample_surface(trimesh.load(ROOT / mesh_dir / a.intruder / f"{a.intruder}.obj", force="mesh"), 4000, seed=0)[0]
    n = len(list((run / a.objects / a.container / "poses").glob("*.json")))
    pts, used = [], []
    for t in range(n):
        Tc = load_T(run / a.objects / a.container / "poses" / f"{t:06d}.json")
        Ti = load_T(run / a.objects / a.intruder / "poses" / f"{t:06d}.json")
        M = np.linalg.inv(Tc) @ Ti                          # container <- intruder (camera frame cancels)
        Q = S @ M[:3, :3].T + M[:3, 3]
        c = Q.mean(0)
        if (c > lo).all() and (c < hi).all():
            pts.append(Q[::4])
            used.append(t)
        elif used:   # only the FIRST stay (resting in the slot + lifting out); later sweeping inside the rack touches
            break    # its floor for real and a pose error there must not delete the floor
    if not pts:
        print(f"{a.run}: {a.intruder} never inside {a.container}'s box -- nothing carved")
        return
    P = np.concatenate(pts)
    keep, drop = [], []
    for b, An, bb in parts:
        inside = ((P @ An.T - bb[None]) < MARGIN_M).all(1)
        (drop if inside.any() else keep).append(b)
    new = txt
    for b in drop:
        new = new.replace(b, "", 1)
    urdf.write_text(new)
    print(json.dumps({"run": a.run, "container": a.container, "intruder": a.intruder, "frames_inside": len(used),
                      "frame_range": [used[0], used[-1]], "pieces": len(parts), "dropped": len(drop)}))


if __name__ == "__main__":
    main()
