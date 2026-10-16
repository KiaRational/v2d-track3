"""Why does a standing object fall in sim? Compare stability at the reference pose using (a) the URDF COM,
(b) the volumetric COM of the mesh, with footprints from (1) the raw collision STL and (2) the CoACD hulls
FlashCHORD actually simulates. Usage: python standing_model.py <parquet_dir> <body_index> <frame>..."""
import glob
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from flash_chord.assets.mesh import convex_decompose

t = pq.read_table(glob.glob(sys.argv[1] + "/*.parquet")[0]).to_pylist()[0]
b = int(sys.argv[2])
urdf = t["object_urdf_paths"][b]
txt = open(urdf).read()
stl = urdf.rsplit("/", 1)[0] + "/" + txt.split("<collision>")[1].split('filename="')[1].split('"')[0].lstrip("./")
m = trimesh.load(stl, force="mesh")
print("watertight", m.is_watertight, "volume", m.volume, "faces", len(m.faces))
o = re.search(r"<inertial>.*?<origin[^>]*xyz=\"([^\"]+)\"", txt, re.S)
com_urdf = np.array(o.group(1).split(), float) if o else np.zeros(3)
com_vol = m.center_mass
print("COM urdf", com_urdf.round(4), " volumetric", com_vol.round(4), " surface centroid", m.centroid.round(4))
hulls = convex_decompose(np.asarray(m.vertices), np.asarray(m.faces))  # default params, as in scene build
hv = np.concatenate([h.vertices for h in hulls])
print("CoACD hulls:", len(hulls))
P, Q = np.asarray(t["object_body_position"], float)[:, b], np.asarray(t["object_body_wxyz"], float)[:, b]


def margin(V, com, tol=0.004):
    foot = V[V[:, 2] < V[:, 2].min() + tol][:, :2]
    if len(foot) < 3:
        return np.nan, 0.0
    try:
        h = ConvexHull(foot)
    except Exception:
        return np.nan, 0.0
    return -(h.equations[:, :2] @ com[:2] + h.equations[:, 2]).max() * 1000, h.volume * 1e4


print("frame | stl: margin(urdfCOM) margin(volCOM) area_cm2 | hulls: margin(urdfCOM) margin(volCOM) area_cm2 | COM height mm")
for f in map(int, sys.argv[3:]):
    R = Rotation.from_quat(Q[f][[1, 2, 3, 0]])
    Vs, Vh = R.apply(m.vertices) + P[f], R.apply(hv) + P[f]
    cu, cv = R.apply(com_urdf) + P[f], R.apply(com_vol) + P[f]
    a, ar = margin(Vs, cu); b2, _ = margin(Vs, cv); c, arh = margin(Vh, cu); d, _ = margin(Vh, cv)
    print(f"{f:5d} | {a:7.1f} {b2:7.1f} {ar:6.2f} | {c:7.1f} {d:7.1f} {arh:6.2f} | {(cu[2]-Vs[:,2].min())*1000:.0f} / vol {(cv[2]-Vs[:,2].min())*1000:.0f}")
# stability if perfectly upright on the face it stands on (tilt removed)
up = R.inv().apply([0, 0, 1])
print("trimesh stable poses (prob):", np.round(trimesh.poses.compute_stable_poses(m, center_mass=com_urdf)[1][:5], 3))
