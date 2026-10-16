"""Is the simulated object where the reference says it is? Compares the URDF collision mesh (STL, what FlashCHORD
simulates) with the OBJ the parquet poses/contacts refer to, and measures reference Sharpa fingertip / MANO
contact distances to each mesh placed at the parquet object pose. Usage: python check_object_frame.py <parquet_dir>"""
import glob
import sys

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

t = pq.read_table(glob.glob(sys.argv[1] + "/*.parquet")[0]).to_pylist()[0]
A = lambda k: np.asarray(t[k], float)
obj = trimesh.load(t["object_mesh_paths"][0], force="mesh")
urdf = t["object_urdf_paths"][0]
stl_path = urdf.rsplit("/", 1)[0] + "/" + open(urdf).read().split('<collision>')[1].split('filename="')[1].split('"')[0].lstrip("./")
stl = trimesh.load(stl_path, force="mesh")
print("urdf:", urdf, "\ncollision mesh:", stl_path)
for n, m in (("obj", obj), ("stl", stl)):
    print(f"{n}: verts {len(m.vertices)} bounds min {m.bounds[0].round(4)} max {m.bounds[1].round(4)} centroid {m.centroid.round(4)}")
print("bounds difference stl-obj (m):", (stl.bounds - obj.bounds).round(4).tolist())
d = cKDTree(obj.vertices).query(stl.vertices[:: max(1, len(stl.vertices) // 5000)])[0]
print(f"stl->obj vertex distance: median {np.median(d)*1000:.2f} mm, max {d.max()*1000:.2f} mm")
op, oq = A("object_body_position")[:, 0], A("object_body_wxyz")[:, 0]
R = Rotation.from_quat(oq[:, [1, 2, 3, 0]])
names = list(t["right_robot_frame_names"])
tips = [k for k, n in enumerate(names) if n.endswith("_tip_site")]
trees = {n: cKDTree(m.vertices) for n, m in (("obj", obj), ("stl", stl))}
for s in ("right", "left"):
    c = A(f"mano_{s}_object_contact_positions"); m = np.abs(c).sum(-1) > 0
    fr = np.where(m.any(1))[0]
    if not len(fr):
        continue
    dt, dc = {k: [] for k in trees}, {k: [] for k in trees}
    for i in fr:
        tp = R[i].inv().apply(A(f"robot_{s}_frames")[i][tips, :3] - op[i])
        cp = R[i].inv().apply(c[i][m[i]] - op[i])
        for k, tr in trees.items():
            dt[k].append(tr.query(tp)[0].min()); dc[k].append(np.median(tr.query(cp)[0]))
    print(f"{s}: {len(fr)} contact frames | nearest Sharpa tip to mesh median (mm): " +
          ", ".join(f"{k} {np.median(v)*1000:.1f}" for k, v in dt.items()) +
          " | MANO contacts to mesh median (mm): " + ", ".join(f"{k} {np.median(v)*1000:.1f}" for k, v in dc.items()))
print("object z at frame 0 (world):", op[0].round(4), " min mesh z at frame 0:",
      (R[0].apply(stl.vertices) + op[0])[:, 2].min().round(4))
