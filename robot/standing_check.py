"""Is a resting object physically able to stand where the reference puts it? For each body, per frame: lowest
collision-mesh point (vs ground z=0), tilt, whether the COM projects inside the ground footprint, and the
closest distance to every other body (leaning?). Usage: python standing_check.py <parquet_dir>"""
import glob
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.spatial.transform import Rotation

t = pq.read_table(glob.glob(sys.argv[1] + "/*.parquet")[0]).to_pylist()[0]
A = lambda k: np.asarray(t[k], float)
P, Q = A("object_body_position"), A("object_body_wxyz")  # [T, B, 3], [T, B, 4]
T, B = P.shape[:2]
meshes, coms, names = [], [], []
for b, urdf in enumerate(t["object_urdf_paths"]):
    txt = open(urdf).read()
    stl = urdf.rsplit("/", 1)[0] + "/" + txt.split("<collision>")[1].split('filename="')[1].split('"')[0].lstrip("./")
    m = trimesh.load(stl, force="mesh")
    meshes.append(np.asarray(m.vertices)[:: max(1, len(m.vertices) // 6000)])
    o = re.search(r"<inertial>.*?<origin[^>]*xyz=\"([^\"]+)\"", txt, re.S)
    coms.append(np.array(o.group(1).split(), float) if o else m.center_mass)
    names.append(urdf.rsplit("/", 1)[1].replace("_rigid.urdf", ""))
    print(f"{names[-1]}: extents {np.ptp(m.vertices, 0).round(3)} m, COM(local) {coms[-1].round(4)}, "
          f"mesh centroid {m.centroid.round(4)}")
hands = [A(f"robot_{s}_frames")[..., :3] for s in ("left", "right")]
for b in range(B):
    R = Rotation.from_quat(Q[:, b][:, [1, 2, 3, 0]])
    V = [R[i].apply(meshes[b]) + P[i, b] for i in range(T)]
    zmin = np.array([v[:, 2].min() for v in V])
    com = R.apply(coms[b]) + P[:, b]
    speed = np.r_[0, np.linalg.norm(np.diff(P[:, b], axis=0), axis=1)] * 20 * 100  # cm/s (20 fps)
    hand_d = np.min([cKDTree(V[i]).query(np.concatenate([h[i] for h in hands]))[0].min() for i in range(T)] and
                    [[cKDTree(V[i]).query(np.concatenate([h[i] for h in hands]))[0].min() for i in range(T)]], 0)
    print(f"\n== {names[b]}  (frames {T})")
    print(" frame  zmin_mm  speed_cm/s  hand_mm  COM_in_footprint  margin_mm  tilt_deg  nearest_other_mm")
    for i in range(0, T, 8):
        v = V[i]
        foot = v[v[:, 2] < zmin[i] + 0.004][:, :2]
        inside, margin = "n/a", np.nan
        if len(foot) >= 4:
            try:
                hull = ConvexHull(foot)
                eq = hull.equations
                margin = -(eq[:, :2] @ com[i, :2] + eq[:, 2]).max() * 1000  # >0 inside
                inside = "yes" if margin > 0 else "NO"
            except Exception:
                pass
        up = R[i].apply(np.eye(3))[:, 2]
        tilt = np.degrees(np.arccos(np.clip(np.abs(up), 0, 1))).min()
        other = min([cKDTree(V[i]).query(Rotation.from_quat(Q[i, o][[1, 2, 3, 0]]).apply(meshes[o]) + P[i, o])[0].min()
                     for o in range(B) if o != b] or [np.nan]) * 1000
        print(f" {i:5d}  {zmin[i]*1000:7.1f}  {speed[i]:9.1f}  {hand_d[i]*1000:7.0f}  {inside:>16}  {margin:8.1f}"
              f"  {tilt:7.1f}  {other:9.1f}")
