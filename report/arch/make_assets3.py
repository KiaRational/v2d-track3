"""4D MANO hand trajectory render for the architecture figure (agile image: manotorch). Eval ep33, hands_dp (Dyn-HaMR
MANO, metric), world frame aligned with the table. Hands at several instants, older ones faded; the wrist paths dashed.
Writes report/arch/assets/hand4d.png (transparent)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DUMP = len(sys.argv) > 1 and sys.argv[1] == "dump"
sys.path.insert(0, "/code/stereo" if DUMP else str(Path.home() / "v2d-vulcan/stereo"))
from object_icp import load_T  # noqa: E402
if DUMP:
    sys.path.insert(0, "/code/agile")
    from geometry import Mano  # noqa: E402
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

R = Path("/scratch/kghasemz/v2d")
RUN = R / "runs/evaluation_ep000033"
OUT = R / "report/arch/assets"
tp = json.loads((RUN / "table_plane.json").read_text())
A = np.eye(4)
A[:3, :3] = Rotation.align_vectors([[0, 0, 1.0]], [np.asarray(tp["up"], float)])[0].as_matrix()
A[2, 3] = -float(tp["height"])
ctr = A[:3, :3] @ np.asarray(tp["center"], float) + A[:3, 3]
rad = float(tp["radius"])
n = len(list((RUN / "recon_hawor/slam_poses").glob("*.json")))
mano = Mano(str(R / "weights/hand"), "cuda") if DUMP else None
FR = [60, 80, 100, 120]
if DUMP:   # MANO meshes + wrist paths -> npz (agile image)
    out = {}
    for side in ("left", "right"):
        wr = []
        for t in range(40, 140, 2):
            f = RUN / f"stereo/hands_dp/{side}/{t:06d}.json"
            if f.exists():
                J = np.asarray(json.loads(f.read_text())["joints"])[0]
                C2W = A @ load_T(RUN / f"recon_hawor/slam_poses/{t:06d}.json")
                wr.append(C2W[:3, :3] @ J + C2W[:3, 3])
        out[f"{side}_wrist"] = np.array(wr)
        for t in FR:
            f = RUN / f"stereo/hands_dp/{side}/{t:06d}.json"
            if f.exists():
                h = json.loads(f.read_text())
                v, _ = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                v = v.cpu().numpy() + np.array(h["T"])
                C2W = A @ load_T(RUN / f"recon_hawor/slam_poses/{t:06d}.json")
                out[f"{side}_{t}_v"] = v @ C2W[:3, :3].T + C2W[:3, 3]
                out[f"{side}_{t}_f"] = np.asarray(mano.faces(h["is_right"]))
    np.savez(OUT / "hand4d.npz", **out)
    print("wrote hand4d.npz")
    sys.exit(0)
Z = np.load(OUT / "hand4d.npz")
COL = {"left": np.array([0.40, 0.33, 0.85]), "right": np.array([0.93, 0.64, 0.12])}
light = np.array([0.3, -0.4, 0.85]); light /= np.linalg.norm(light)

fig = plt.figure(figsize=(6, 4.6), dpi=170)
ax = fig.add_subplot(projection="3d", computed_zorder=False)
th = np.linspace(0, 2 * np.pi, 80)
ax.add_collection3d(Poly3DCollection([np.c_[ctr[0] + rad * np.cos(th), ctr[1] + rad * np.sin(th), np.zeros_like(th)]],
                                     facecolors=[(0.62, 0.42, 0.25)], edgecolors=[(0.45, 0.30, 0.18)], linewidths=1.2,
                                     alpha=0.55, zorder=0))
allv = []
for side in ("left", "right"):
    wr = Z[f"{side}_wrist"]
    if len(wr) > 1:
        wr = np.array(wr)
        ax.add_collection3d(Line3DCollection(np.stack([wr[:-1], wr[1:]], 1), colors=[(*COL[side], 0.9)], linewidths=1.6,
                                             linestyles="--", zorder=1))
    for i, t in enumerate(FR):
        if f"{side}_{t}_v" not in Z:
            continue
        vw, F = Z[f"{side}_{t}_v"], Z[f"{side}_{t}_f"]
        tri = vw[F]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        sh = 0.5 + 0.5 * np.abs(nrm @ light)
        a = (0.10, 0.22, 0.45, 1.0)[i]   # older instants fade out
        base = COL[side] * 0.35 + np.array([0.94, 0.80, 0.70]) * 0.65 if i == len(FR) - 1 else COL[side]
        fc = np.c_[np.clip(base[None] * sh[:, None], 0, 1), np.full(len(F), a)]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors=fc if a == 1.0 else "none", linewidths=0.2 if a == 1.0 else 0, zorder=2 + i))
        allv.append(vw)
V = np.concatenate(allv)
c = np.r_[V[:, :2].mean(0), max(0.08, V[:, 2].mean())]
r = max(np.ptp(V[:, 0]), np.ptp(V[:, 1])) * 0.62
ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r * 0.6, c[2] + r * 0.6)
ax.set_box_aspect((1, 1, 0.6)); ax.view_init(elev=28, azim=-100); ax.set_axis_off()
fig.savefig(OUT / "hand4d.png", transparent=True, bbox_inches="tight", pad_inches=0)
print("wrote hand4d.png")
