"""Contact sheet of meshes: 3 orthographic views each (top / side / end), shaded triangles via matplotlib.
Usage: python render_mesh_views.py out.png mesh1.obj [mesh2.obj ...]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

out, paths = sys.argv[1], sys.argv[2:]
views = [("top", 90, -90), ("side", 0, -90), ("3/4", 30, -60)]
fig = plt.figure(figsize=(4 * len(views), 3.2 * len(paths)))
for i, p in enumerate(paths):
    m = trimesh.load(p, force="mesh", process=False)
    if len(m.faces) > 20000:  # subsample faces for drawing speed
        m = trimesh.Trimesh(m.vertices, m.faces[np.random.default_rng(0).choice(len(m.faces), 20000, replace=False)], process=False)
    v = m.vertices - m.vertices.mean(0)
    _, _, Vt = np.linalg.svd(v, full_matrices=False)  # long axis -> x
    v = v @ Vt.T
    tri = v[m.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.abs(n @ np.array([0.3, 0.4, 0.86]))
    col = np.c_[0.85 * shade, 0.65 * shade, 0.45 * shade, np.ones_like(shade)]
    r = np.abs(v).max()
    for j, (name, el, az) in enumerate(views):
        ax = fig.add_subplot(len(paths), len(views), i * len(views) + j + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=col, linewidths=0))
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        ax.view_init(el, az); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ext = (v.max(0) - v.min(0)) * 100
        ax.set_title(f"{p.split('/')[-2]}/{p.split('/')[-1][:-4]} {name}\n{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} cm", fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=90)
print("wrote", out)
