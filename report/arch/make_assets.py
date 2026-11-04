"""Image assets for the architecture figure, all from our own data (eval ep33: blue cup + wooden spoon).
Writes report/arch/assets/*.png (crops around the action, white or transparent background)."""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

R = Path("/scratch/kghasemz/v2d")
RUN = R / "runs/evaluation_ep000033"
OUT = R / "report/arch/assets"
OUT.mkdir(parents=True, exist_ok=True)
T = int(sys.argv[1]) if len(sys.argv) > 1 else 120
K = json.loads((RUN / "recon_hawor/intrinsics_stable.json").read_text())
sys.path.insert(0, str(Path.home() / "v2d-vulcan/stereo"))
from object_icp import load_T  # noqa: E402

OBJS = {o["name"]: o["id"] for o in json.loads((RUN / "objects_recon/objects.json").read_text())}
COL = {"wooden_spoon": (235, 104, 52), "blue_cup": (42, 120, 214)}


def frame(t):
    cap = cv2.VideoCapture(str(RUN / "input_undistorted.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)


def mask(oid, t):
    m = cv2.imread(str(RUN / f"objects_recon/masks/{oid}/{t:06d}.png"), 0)
    return None if m is None else m > 127


def proj(X):
    return np.c_[X[:, 0] / X[:, 2] * K["fx"] + K["cx"], X[:, 1] / X[:, 2] * K["fy"] + K["cy"]]


img = frame(T)
H, W = img.shape[:2]
ms = {nm: mask(oid, T) for nm, oid in OBJS.items()}
hand = np.zeros((H, W), bool)
for hid in (2, 3):
    m = mask(hid, T)
    if m is not None:
        hand |= m
U = np.zeros((H, W), bool)
for m in ms.values():
    if m is not None:
        U |= m
U |= hand
ys, xs = np.nonzero(U)
cx, cy = int(np.median(xs)), int(np.median(ys))
half = int(max(np.ptp(xs), np.ptp(ys)) * 0.75) + 60
x0, x1 = max(0, cx - half), min(W, cx + half)
y0, y1 = max(0, cy - int(half * 0.75)), min(H, cy + int(half * 0.75))


def crop(a):
    return a[y0:y1, x0:x1]


def save(a, name):
    cv2.imwrite(str(OUT / name), cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if a.ndim == 3 and a.shape[2] == 3 else a)
    print("wrote", name, a.shape)


save(cv2.resize(img, (W // 3, H // 3)), "raw_frame_full.png")
save(crop(img), "raw_frame.png")

# depth
png = cv2.imread(str(RUN / f"depth_stereo/{T:06d}.png"), cv2.IMREAD_UNCHANGED).astype(float)
D = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, np.nan)
lo, hi = np.nanpercentile(D, 2), np.nanpercentile(D, 70)
dn = np.clip((D - lo) / (hi - lo), 0, 1)
dc = (plt.get_cmap("magma_r")(np.nan_to_num(dn, nan=1.0))[..., :3] * 255).astype(np.uint8)
save(crop(dc), "depth.png")

# masks
mv = (img * 0.45 + 255 * 0.55).astype(np.uint8)
for nm, m in ms.items():
    if m is not None:
        mv[m] = (0.35 * img[m] + 0.65 * np.array(COL.get(nm, (27, 175, 122)))).astype(np.uint8)
mv[hand] = (0.5 * img[hand] + 0.5 * np.array((237, 161, 0))).astype(np.uint8)
save(crop(mv), "masks.png")

# FoundationPose hypotheses (spoon): top candidates as outlines
mesh_s = trimesh.load(R / "meshes_recon/wooden_spoon/wooden_spoon.obj", force="mesh")
S = trimesh.sample.sample_surface(mesh_s, 3000, seed=0)[0]
z = np.load(RUN / f"objects_recon/wooden_spoon/fp_hyps/{T:06d}.npz")
P, sc = z["poses"], z["scores"]
hv = (img * 0.55 + 255 * 0.45).astype(np.uint8)
cmap = plt.get_cmap("viridis")
for r, k in enumerate(np.argsort(-sc)[:8][::-1]):
    uv = proj(S @ P[k][:3, :3].T + P[k][:3, 3]).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    sil = np.zeros((H, W), np.uint8)
    sil[uv[ok, 1], uv[ok, 0]] = 255
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = (np.array(cmap(r / 7.0)[:3]) * 255).astype(int).tolist()
    cv2.drawContours(hv, cs, -1, c, 3)
save(crop(hv), "fp_hypotheses.png")

# TrackEverything 3D tracks on the objects, projected with 15-frame trails
tz = np.load(RUN / "objects_recon/te_tracks.npz")
Pw, Vis = tz["P_world"], tz["vis"]
slam = {t: load_T(RUN / f"recon_hawor/slam_poses/{t:06d}.json") for t in range(max(0, T - 15), T + 1)}
tv = (img * 0.6 + 255 * 0.4).astype(np.uint8)
sel = np.nonzero(Vis[T] & np.isfinite(Pw[T][:, 0]))[0]
sel = sel[np.random.default_rng(0).permutation(len(sel))[:220]]
tcm = plt.get_cmap("turbo")
for j, n in enumerate(sel):
    pts = []
    for t in range(max(0, T - 15), T + 1):
        if np.isfinite(Pw[t, n, 0]):
            Tc = np.linalg.inv(slam[t])
            X = Tc[:3, :3] @ Pw[t, n] + Tc[:3, 3]
            pts.append(proj(X[None])[0])
    if len(pts) > 1:
        c = (np.array(tcm(j / len(sel))[:3]) * 255).astype(int).tolist()
        cv2.polylines(tv, [np.array(pts, np.int32)], False, c, 2, cv2.LINE_AA)
        cv2.circle(tv, tuple(np.int32(pts[-1])), 4, c, -1, cv2.LINE_AA)
save(crop(tv), "tracks.png")

# hand: MANO joints skeleton (hands_dp, camera frame)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
hv2 = (img * 0.55 + 255 * 0.45).astype(np.uint8)
for side, c in (("left", (98, 80, 214)), ("right", (237, 161, 0))):
    f = RUN / f"stereo/hands_dp/{side}/{T:06d}.json"
    if f.exists():
        J = np.asarray(json.loads(f.read_text())["joints"])
        uv = proj(J).astype(int)
        for a, b in BONES:
            cv2.line(hv2, tuple(uv[a]), tuple(uv[b]), c, 4, cv2.LINE_AA)
        for p in uv:
            cv2.circle(hv2, tuple(p), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(hv2, tuple(p), 6, c, 2, cv2.LINE_AA)
save(crop(hv2), "hand.png")


def dec(m, n=60000):
    if len(m.faces) <= n:
        return m
    keep = np.random.default_rng(0).choice(len(m.faces), n, replace=False)
    return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)


def render_mesh(meshes, name, elev=22, azim=-60, size=4):
    fig = plt.figure(figsize=(size, size), dpi=150)
    ax = fig.add_subplot(projection="3d")
    allv = []
    light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
    for m, col in meshes:
        tri = m.vertices[m.faces]
        n = m.face_normals
        shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
        fc = np.clip(np.array(col)[None] * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors=fc, linewidths=0.3))
        allv.append(m.vertices)
    V = np.concatenate(allv)
    c, r = V.mean(0), np.ptp(V, 0).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    fig.savefig(OUT / name, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("wrote", name)


ms_ = mesh_s.copy(); ms_.apply_translation(-ms_.centroid)
mc = trimesh.load(R / "meshes_recon/blue_cup/blue_cup.obj", force="mesh"); mc.apply_translation(-mc.centroid + [0.0, 0.16, 0.0])
render_mesh([(dec(ms_), (0.80, 0.55, 0.32)),
             (dec(mc), (0.25, 0.45, 0.80))],
            "meshes.png")

# Sharpa Wave right hand from its URDF (zero joint configuration)
U_ = Path("/scratch/kghasemz/v2d/fc_src_rewards/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/sharpawave/right_sharpa_wave.urdf")
root = ET.parse(U_).getroot()


def origin(el):
    o = el.find("origin")
    M = np.eye(4)
    if o is not None:
        M[:3, :3] = Rotation.from_euler("xyz", [float(x) for x in o.get("rpy", "0 0 0").split()]).as_matrix()
        M[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
    return M


parent = {j.find("child").get("link"): (j.find("parent").get("link"), origin(j)) for j in root.findall("joint")}


def world(link):
    M = np.eye(4)
    while link in parent:
        p, Mo = parent[link]
        M = Mo @ M
        link = p
    return M


parts = []
for link in root.findall("link"):
    for vis in link.findall("visual"):
        g = vis.find("geometry/mesh")
        if g is None:
            continue
        fn = g.get("filename").replace("package://", "")
        cand = [U_.parent / fn, U_.parent / Path(fn).name] + list(U_.parent.rglob(Path(fn).name))
        f = next((c for c in cand if c.exists()), None)
        if f is None:
            continue
        m = trimesh.load(f, force="mesh")
        if g.get("scale"):
            m.apply_scale([float(s) for s in g.get("scale").split()])
        m.apply_transform(world(link.get("name")) @ origin(vis))
        m = dec(m, 8000)
        parts.append((m, (0.86, 0.87, 0.90)))
print("sharpa parts", len(parts))
if parts:
    render_mesh(parts, "sharpa_hand.png", elev=10, azim=-120, size=5)
