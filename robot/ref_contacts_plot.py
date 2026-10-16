"""Where are the reference (human) contacts on the object? Plots the object mesh with the retargeted
parquet's MANO object-contact points and the right/left MANO joints, all in the object frame, for a
few frames of the lift. Two stages because no image has both pyarrow and matplotlib:
  python ref_contacts_plot.py <parquet_dir> <out.png>   in flash_chord (writes <out>.npz), then again in egophi."""
import glob
import os
import sys

import numpy as np

KEYS = ["object_body_position", "object_body_wxyz", "robot_right_frames", "robot_left_frames"] + [f"mano_{s}_{k}" for s in ("right", "left")
                                                        for k in ("joints", "object_contact_positions")]
cache = os.path.splitext(sys.argv[2])[0] + ".npz"
try:
    import pyarrow.parquet as pq
    t = pq.read_table(glob.glob(sys.argv[1] + "/*.parquet")[0]).to_pylist()[0]
    np.savez(cache, mesh=t["object_mesh_paths"][0], rnames=np.array(t["right_robot_frame_names"]), lnames=np.array(t["left_robot_frame_names"]), **{k: np.asarray(t[k], float) for k in KEYS})
    print("wrote", cache)
    sys.exit(0)
except ImportError:
    d = np.load(cache)
    t = {k: d[k] for k in KEYS}
    t["object_mesh_paths"] = [str(d["mesh"])]
    names = {"right": [str(x) for x in d["rnames"]], "left": [str(x) for x in d["lnames"]]}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

A = lambda k: np.asarray(t[k], float)
V = np.array([l.split()[1:4] for l in open(t["object_mesh_paths"][0]) if l.startswith("v ")], float)
V = V[np.random.default_rng(0).choice(len(V), min(len(V), 6000), replace=False)]
op, oq = A("object_body_position")[:, 0], A("object_body_wxyz")[:, 0]
R = Rotation.from_quat(oq[:, [1, 2, 3, 0]])
to_obj = lambda P, i: R[i].inv().apply(P - op[i])
# the mesh's own long axis (handle direction) for the side views
c = V - V.mean(0)
ax_ = np.linalg.svd(c, full_matrices=False)[2]
views = [(0, 1, "top (principal axes 1-2)"), (0, 2, "side (1-3)")]
P = c @ ax_.T
frames = [60, 110, 160, 210, 260]
fig, axs = plt.subplots(2, len(frames), figsize=(4.2 * len(frames), 8))
cnt = {}
for s in ("right", "left"):
    cp = A(f"mano_{s}_object_contact_positions")
    cnt[s] = (np.abs(cp).sum(-1) > 0)
for j, i in enumerate(frames):
    for r, (a, b, title) in enumerate(views):
        ax = axs[r, j]
        ax.scatter(P[:, a], P[:, b], s=0.3, c="0.7")
        for s, col in (("right", "r"), ("left", "b")):
            J = (to_obj(A(f"mano_{s}_joints")[i], i) - V.mean(0)) @ ax_.T
            for ch in ([0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]):
                ax.plot(J[ch, a], J[ch, b], col + "-", lw=1)
            fr = A(f"robot_{s}_frames")[i]  # (n_frames, 7): position, wxyz
            tips = [k for k, n in enumerate(names[s]) if n.endswith("_tip_site")]
            T = (to_obj(fr[tips, :3], i) - V.mean(0)) @ ax_.T
            ax.scatter(T[:, a], T[:, b], s=30, facecolors="none", edgecolors="m" if s == "right" else "c", lw=1.5)
            m = cnt[s][i]
            if m.any():
                C = (to_obj(A(f"mano_{s}_object_contact_positions")[i][m], i) - V.mean(0)) @ ax_.T
                ax.scatter(C[:, a], C[:, b], s=25, c=col, marker="x")
        ax.set_aspect("equal")
        ax.set_xlim(P[:, a].min() - 0.1, P[:, a].max() + 0.1)
        ax.set_ylim(P[:, b].min() - 0.1, P[:, b].max() + 0.1)
        ax.set_title(f"frame {i} {title}\nR contacts {cnt['right'][i].sum()} L {cnt['left'][i].sum()}", fontsize=9)
fig.suptitle("reference: object mesh (grey), MANO right (red) / left (blue) joints, x = MANO contacts, o = Sharpa fingertips (magenta R / cyan L); object frame")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=90)
# all-frame contact footprint along the long axis
allc = []
for s in ("right", "left"):
    cp = A(f"mano_{s}_object_contact_positions")
    for i in np.where(cnt[s].any(1))[0]:
        allc.append((to_obj(cp[i][cnt[s][i]], i) - V.mean(0)) @ ax_.T)
allc = np.concatenate(allc)
tipd = []
for i in np.where(cnt["right"].any(1))[0]:
    fr = A("robot_right_frames")[i]
    tips = [k for k, n in enumerate(names["right"]) if n.endswith("_tip_site")]
    mt = to_obj(A("mano_right_joints")[i][[4, 8, 12, 16, 20]], i)
    tipd.append(np.linalg.norm(to_obj(fr[tips, :3], i) - mt, axis=1))
print("right contact frames: Sharpa tip vs MANO tip distance (m), median per finger:", np.median(tipd, 0).round(3).tolist())
print("mesh extent along long axis:", P[:, 0].min().round(3), P[:, 0].max().round(3))
print("contact positions along long axis: pct 5/50/95", np.percentile(allc[:, 0], [5, 50, 95]).round(3))
h = np.histogram(P[:, 0], bins=10)
print("mesh vertex histogram along long axis (bin edges, counts):", h[1].round(3).tolist(), h[0].tolist())
print("contact histogram, same bins:", np.histogram(allc[:, 0], bins=h[1])[0].tolist())
