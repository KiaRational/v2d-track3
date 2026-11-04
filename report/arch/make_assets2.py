"""Two 3D renders for the architecture figure (eval ep33), world frame aligned with the table (+z up, table top z=0):
slam.png     DROID-SLAM camera trajectory (coloured by time, frustums) over a stereo point cloud of the scene
scene4d.png  4D HOI reconstruction at frame T: table disk, object meshes at their final poses (tpfsut_recon), both
             Dyn-HaMR MANO hand meshes (hands_dp, from hand4d.npz), spoon and cup paths over the episode."""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path.home() / "v2d-vulcan/stereo"))
from object_icp import load_T  # noqa: E402

R = Path("/scratch/kghasemz/v2d")
RUN = R / "runs/evaluation_ep000033"
OUT = R / "report/arch/assets"
T = 120
K = json.loads((RUN / "recon_hawor/intrinsics_stable.json").read_text())
tp = json.loads((RUN / "table_plane.json").read_text())
A = np.eye(4)
A[:3, :3] = Rotation.align_vectors([[0, 0, 1.0]], [np.asarray(tp["up"], float)])[0].as_matrix()
A[2, 3] = -float(tp["height"])
ctr = A[:3, :3] @ np.asarray(tp["center"], float) + A[:3, 3]
rad = float(tp["radius"])
n = len(list((RUN / "recon_hawor/slam_poses").glob("*.json")))
C2W = {t: A @ load_T(RUN / f"recon_hawor/slam_poses/{t:06d}.json") for t in range(n)}
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def frame(t):
    cap = cv2.VideoCapture(str(RUN / "input_undistorted.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)


def cloud(t, step=9, zmax=1.6):
    png = cv2.imread(str(RUN / f"depth_stereo/{t:06d}.png"), cv2.IMREAD_UNCHANGED).astype(float)
    D = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, 0)
    img = frame(t)
    v, u = np.mgrid[0:D.shape[0]:step, 0:D.shape[1]:step]
    z = D[v, u]
    ok = (z > 0.15) & (z < zmax)
    X = np.stack([(u[ok] - K["cx"]) / K["fx"] * z[ok], (v[ok] - K["cy"]) / K["fy"] * z[ok], z[ok]], 1)
    W = X @ C2W[t][:3, :3].T + C2W[t][:3, 3]
    return W, img[v[ok], u[ok]] / 255.0


def disk(ax, z=0.0, col=(0.62, 0.42, 0.25), alpha=0.85):
    th = np.linspace(0, 2 * np.pi, 80)
    P = np.c_[ctr[0] + rad * np.cos(th), ctr[1] + rad * np.sin(th), np.full_like(th, z)]
    ax.add_collection3d(Poly3DCollection([P], facecolors=[col], edgecolors=[(0.45, 0.30, 0.18)], linewidths=1.2, alpha=alpha))


def style(ax, c, r, elev, azim):
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r * 0.55, c[2] + r * 0.9)
    ax.set_box_aspect((1, 1, 0.72)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()


# ---------------- SLAM ----------------
fig = plt.figure(figsize=(6, 4.6), dpi=170)
ax = fig.add_subplot(projection="3d", computed_zorder=False)
for t in (0, n // 2, n - 1):
    W, cc = cloud(t)
    keep = (np.linalg.norm(W[:, :2] - ctr[:2], axis=1) < 0.42) & (W[:, 2] > -0.03)
    ax.scatter(W[keep, 0], W[keep, 1], W[keep, 2], c=cc[keep], s=0.6, depthshade=False, zorder=1)
cam = np.array([C2W[t][:3, 3] for t in range(n)])
seg = np.stack([cam[:-1], cam[1:]], 1)
lc = Line3DCollection(seg, colors=plt.get_cmap("viridis")(np.linspace(0, 1, n - 1)), linewidths=5, zorder=5)
ax.add_collection3d(lc)
fr = []
for t in np.linspace(0, n - 1, 6).astype(int):
    M = C2W[t]; s = 0.10
    cn = np.array([[-0.6, -0.45, 1], [0.6, -0.45, 1], [0.6, 0.45, 1], [-0.6, 0.45, 1]]) * s
    cw = cn @ M[:3, :3].T + M[:3, 3]; o = M[:3, 3]
    col = plt.get_cmap("viridis")(t / (n - 1))
    for i in range(4):
        ax.plot(*np.c_[o, cw[i]], color=col, lw=2.0, zorder=6)
        ax.plot(*np.c_[cw[i], cw[(i + 1) % 4]], color=col, lw=2.0, zorder=6)
c = np.r_[(ctr[:2] + cam[:, :2].mean(0)) / 2, 0.22]
style(ax, c, 0.42, elev=22, azim=-115)
fig.savefig(OUT / "slam.png", transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)
print("wrote slam.png")

# ---------------- 4D scene ----------------
def mesh_world(nm, t, var="tpfsut_recon", nf=60000):
    m = trimesh.load(R / "meshes_recon" / nm / f"{nm}.obj", force="mesh")
    if len(m.faces) > nf:
        keep = np.random.default_rng(0).choice(len(m.faces), nf, replace=False)
        m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    Mw = C2W[t] @ load_T(RUN / var / nm / "poses" / f"{t:06d}.json")
    m.apply_transform(Mw)
    return m


def path(nm, var="tpfsut_recon"):
    m = trimesh.load(R / "meshes_recon" / nm / f"{nm}.obj", force="mesh")
    c0 = m.vertices.mean(0)
    return np.array([(C2W[t] @ load_T(RUN / var / nm / "poses" / f"{t:06d}.json") @ np.r_[c0, 1])[:3] for t in range(n)])


fig = plt.figure(figsize=(6, 4.6), dpi=170)
ax = fig.add_subplot(projection="3d", computed_zorder=False)
disk(ax)
light = np.array([0.3, -0.4, 0.85]); light /= np.linalg.norm(light)
for nm, col in (("blue_cup", (0.30, 0.48, 0.82)), ("wooden_spoon", (0.86, 0.60, 0.36))):
    m = mesh_world(nm, T)
    sh = 0.45 + 0.55 * np.clip(m.face_normals @ light, 0, 1)
    ax.add_collection3d(Poly3DCollection(m.vertices[m.faces], facecolors=np.clip(np.array(col)[None] * sh[:, None], 0, 1),
                                         edgecolors=np.clip(np.array(col)[None] * sh[:, None], 0, 1), linewidths=0.3, zorder=4))
    P = path(nm)
    segs = np.stack([P[:-1], P[1:]], 1)
    ax.add_collection3d(Line3DCollection(segs, colors=[(*col, 0.85)], linewidths=1.6, linestyles="--", zorder=3))
HZ = np.load(OUT / "hand4d.npz")   # Dyn-HaMR MANO meshes in this world frame (make_assets3.py dump)
for side, col in (("left", np.array([0.40, 0.33, 0.85])), ("right", np.array([0.93, 0.64, 0.12]))):
    if f"{side}_{T}_v" not in HZ:
        continue
    tri = HZ[f"{side}_{T}_v"][HZ[f"{side}_{T}_f"]]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    base = col * 0.35 + np.array([0.94, 0.80, 0.70]) * 0.65
    fc = np.clip(base[None] * (0.5 + 0.5 * np.abs(nrm @ light))[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors=fc, linewidths=0.2, zorder=6))
c = np.r_[ctr[:2], 0.10]
style(ax, c, 0.27, elev=28, azim=-100)
fig.savefig(OUT / "scene4d_render.png", transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)
print("wrote scene4d_render.png")
