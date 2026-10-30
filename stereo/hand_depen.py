"""Hand depenetration: the reconstructed hand must not start (or stay) inside an object. The simulator resets the robot
hand to the reference pose; fingers inside an object's collision there make the physics shove them apart at t=0 and
kick the object (eval ep32: the grasping hand's fingers inside the blue cup at the first frame).

Per frame and hand: sample points along every finger bone (joints + 3 per bone, OpenPose order) and test them against
every object's surface at the given object poses (inside = behind the outward normal of the closest dense surface
sample). Where some points are inside, the WHOLE hand gets the smallest translation that brings them back to the
surface + MARGIN (iterative mean of the needed pushes). The per-frame shift is then smoothed over time (Gaussian
SIGMA frames, max-preserving so short overlaps are still cleared) -- the object never moves, the hand never jitters.
Writes stereo/<out>/<side>/NNNNNN.json (hands format, joints and MANO T shifted) + report.json (per object: frames with
overlap, max depth before/after, start-frame depth).
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python hand_depen.py <run>... [--objects tpfsu_recon]
       [--hands hands_c] [--out hands_dp]"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial import cKDTree

from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
MARGIN, SIGMA, MAX_SHIFT = 0.002, 2.0, 0.06
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def hand_points(J):
    pts = [J]
    for a, b in BONES:
        for f in (0.25, 0.5, 0.75):
            pts.append((J[a] + f * (J[b] - J[a]))[None])
    return np.concatenate(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--objects", default="tpfsu_recon")
    ap.add_argument("--hands", default="hands_c")
    ap.add_argument("--out", default="hands_dp")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        objs = []
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            pd = run / a.objects / nm / "poses"
            if not pd.is_dir():
                continue
            mesh = trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh")
            S, fi = trimesh.sample.sample_surface(mesh, 20000, seed=0)
            N = mesh.face_normals[fi]
            hull = None
            if mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")["cont_axis"] is not None:
                # cups: SOLID (convex hull, incl. the inside of the cup) -- fingers that went through a thin wall into
                # the cup are "outside" the surface test but inside the cup; a hand grasping a cup is outside it
                h = mesh.convex_hull
                hull = (np.asarray(h.face_normals), np.einsum("ij,ij->i", h.face_normals, h.triangles[:, 0]))
            objs.append({"name": nm, "S": S, "N": N, "tree": cKDTree(S), "hull": hull,
                         "Tc": [load_T(pd / f"{t:06d}.json") for t in range(n)]})   # camera <- object
        report = {}
        for side in ("left", "right"):
            sdir = run / "stereo" / a.hands / side
            if not sdir.is_dir():
                continue
            recs = {int(p.stem): json.loads(p.read_text()) for p in sdir.glob("[0-9]*.json")}
            shift = np.zeros((n, 3))          # camera frame of each frame
            depth0 = {}
            for t, r in recs.items():
                if t >= n or r.get("joints") is None:
                    continue
                P = hand_points(np.asarray(r["joints"], float))      # camera frame
                d = np.zeros(3)
                for _ in range(4):
                    need = []
                    for ob in objs:
                        T = ob["Tc"][t]
                        q = (P + d - T[:3, 3]) @ T[:3, :3]            # object frame
                        if ob["hull"] is not None:   # solid convex hull: push out through the nearest face plane
                            An, b = ob["hull"]
                            s = q @ An.T - b                                      # > 0 outside that face
                            inside = s.max(1) < 0
                            j = s.argmax(1)
                            dist = -s.max(1)
                            pushv = (dist[:, None] + MARGIN) * An[j]
                        else:
                            dist, k = ob["tree"].query(q)
                            inside = ((q - ob["S"][k]) * ob["N"][k]).sum(1) < 0
                            inside &= dist < 0.05                         # far points: not a surface penetration
                            pushv = (ob["S"][k] - q) + MARGIN * ob["N"][k]
                        if inside.any():
                            push = pushv[inside]
                            need.append(push @ T[:3, :3].T)           # back to camera frame
                        if _ == 0:
                            dep = float(dist[inside].max()) if inside.any() else 0.0
                            depth0.setdefault(ob["name"], {})[t] = dep
                    if not need:
                        break
                    allp = np.concatenate(need)
                    d = d + allp[np.argsort(-np.linalg.norm(allp, axis=1))[:max(3, len(allp) // 4)]].mean(0)
                    if np.linalg.norm(d) > MAX_SHIFT:
                        d *= MAX_SHIFT / np.linalg.norm(d)
                shift[t] = d
            # smooth in time without losing short pushes: max-preserving envelope per axis, then Gaussian
            mag = np.linalg.norm(shift, axis=1)
            env = maximum_filter1d(mag, size=5)
            sm = gaussian_filter1d(shift, SIGMA, axis=0)
            smag = np.linalg.norm(sm, axis=1)
            scale = np.where(smag > 1e-9, np.maximum(1.0, env / np.maximum(smag, 1e-9)), 1.0)
            sm = sm * np.minimum(scale, 3.0)[:, None]
            odir = run / "stereo" / a.out / side
            shutil.rmtree(odir, ignore_errors=True)
            odir.mkdir(parents=True)
            for t, r in recs.items():
                if t < n and np.linalg.norm(sm[t]) > 1e-5:
                    if r.get("joints") is not None:
                        r["joints"] = (np.asarray(r["joints"]) + sm[t]).tolist()
                    r["T"] = (np.asarray(r["T"]) + sm[t]).tolist()
                    r["depen_shift_mm"] = round(float(np.linalg.norm(sm[t])) * 1000, 1)
                (odir / f"{t:06d}.json").write_text(json.dumps(r))
            first = min(recs) if recs else None
            report[side] = {"frames_shifted": int((np.linalg.norm(sm, axis=1) > 1e-4).sum()),
                            "max_shift_mm": round(float(np.linalg.norm(sm, axis=1).max()) * 1000, 1),
                            "first_frame": first,
                            "objects": {nm: {"frames_overlap": int(sum(v > 0 for v in dd.values())),
                                             "max_depth_mm": round(max(dd.values(), default=0) * 1000, 1),
                                             "start_depth_mm": round(dd.get(first, 0.0) * 1000, 1)}
                                        for nm, dd in depth0.items()}}
            print(rn, side, json.dumps(report[side]), flush=True)
        for extra in ("report.json",):
            src = run / "stereo" / a.hands / extra
            if src.exists():
                shutil.copy(src, run / "stereo" / a.out / extra)
        (run / "stereo" / a.out / "depen_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
