"""System architecture figure (style after the reference: coloured stage panels, white rounded method boxes, inline
thumbnails from our own data, colour-coded flow arrows). Output: arch.pdf / arch.svg / arch.png.
Assets: report/arch/assets (make_assets.py). Coordinates: 200 x 100 units."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.image as mpimg

D = Path("/scratch/kghasemz/v2d/report/arch")
A = D / "assets"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"})

C = {  # panel fill, border/accent, title
    "green": ("#e6f0dc", "#6aa84f", "#3f7a2a"),
    "yellow": ("#fbf0c9", "#d8a200", "#9a7000"),
    "grey": ("#ececea", "#9a9a95", "#555550"),
    "blue": ("#dce7f7", "#3b6fc4", "#214e94"),
    "red": ("#f7dcd8", "#d2463f", "#a8302a"),
    "robot": ("#e9e9e7", "#8d8d88", "#444440"),
}

fig = plt.figure(figsize=(20, 10.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200); ax.set_ylim(0, 104); ax.set_aspect("equal"); ax.axis("off")


def panel(x, y, w, h, key, title=None, title_y=None, fs=15):
    fc, ec, tc = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec="none", zorder=0))
    if title:
        ax.text(x + w / 2, (title_y if title_y is not None else y + h - 3.2), title, ha="center", va="center",
                fontsize=fs, color=tc, zorder=5)


def box(x, y, w, h, text, key, italic=False, fs=12, fill="white", lw=2.2, tcol=None, bold=False):
    fc, ec, tc = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fill, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tcol or tc,
            style="italic" if italic else "normal", weight="bold" if bold else "normal", zorder=4, linespacing=1.15)
    return (x, y, w, h)


def label(x, y, w, h, text, key, fs=11.5):
    fc, ec, tc = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=ec, ec="none", alpha=0.18, zorder=2))
    ax.text(x + 1.2, y + h / 2, text, ha="left", va="center", fontsize=fs, color=tc, zorder=4, linespacing=1.15)
    return (x, y, w, h)


def img(name, x, y, w, h, crop=None):
    a = mpimg.imread(A / name)
    if crop:
        r0, r1, c0, c1 = crop
        a = a[int(r0 * a.shape[0]):int(r1 * a.shape[0]), int(c0 * a.shape[1]):int(c1 * a.shape[1])]
    ax.imshow(a, extent=(x, x + w, y, y + h), zorder=4, aspect="auto")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.4", fc="none", ec="white", lw=1.2, zorder=5))


def arrow(p0, p1, col, rad=0.0, lw=2.2, ms=16, z=6, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=col, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1, shrinkB=1))


def R(b):   # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def T(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def B(b):
    return (b[0] + b[2] / 2, b[1])


def math(x, y, s, fs=14, col="#222", ha="center"):
    ax.text(x, y, s, fontsize=fs, color=col, ha=ha, va="center", zorder=7)


# ---------------- column 1: input ----------------
panel(2, 44, 36, 58, "green", "Egocentric Episode", title_y=98.5)
img("raw_frame_full.png", 5, 64, 30, 22.5)
box(8.5, 50, 23, 7.5, "colour camera +\nmono stereo pair", "green", fs=10.5)
panel(2, 2, 36, 39, "green", "Task & Objects via VLM", title_y=37.4)
vlm = box(5, 26, 30, 7, "Qwen3-VL (local)", "green", italic=True, fs=13)
ax.text(20, 17.8, "“The person holds the blue cup and\nstirs it with a wooden spoon,\nthen puts both down.”",
        ha="center", va="center", fontsize=10.5, style="italic", color="#3f7a2a")
obj = box(5, 4.5, 30, 7, "“wooden spoon”, “blue cup”", "green", fs=12.5, bold=True)
arrow(B(vlm), (20, 22.3), C["green"][1], lw=2)
arrow((20, 13.4), T(obj), C["green"][1], lw=2)

# ---------------- column 2: estimators ----------------
panel(42, 76, 82, 26, "yellow", "Hand Estimator", title_y=98.6)
dh = box(44.5, 82, 15, 9, "Dyn-HaMR", "yellow", italic=True, fs=13)
mf = box(64, 82, 21, 9, "metric MANO\ntrajectory", "yellow", fs=11.5, fill="#fff8e0")
wf = box(64, 77.3, 21, 3.6, "WiLoR stereo fallback (<2%)", "yellow", italic=True, fs=8.5)
hp = box(89.5, 79, 14, 15, "hands out\nof objects", "yellow", fs=10.5, fill="#fff8e0")
img("hand.png", 106, 78.5, 16, 16, crop=(0.25, 0.7, 0.62, 1.0))
math(114, 96.8, r"$\mathcal{M}^h_t$", 15)
arrow(R(dh), L(mf), C["yellow"][1]); arrow(R(mf), (89.5, 86.5), C["yellow"][1]); arrow((103.5, 86.5), (106, 86.5), C["yellow"][1])

panel(42, 51, 82, 22, "grey", None)
fs_ = box(44.5, 62, 21, 8, "FoundationStereo", "grey", italic=True, fs=12)
ds = box(44.5, 53, 21, 7, "DROID-SLAM", "grey", italic=True, fs=12)
img("depth.png", 69, 53, 17, 17)
math(77.5, 61.5, r"$D_t$", 17, col="white")
tp = box(89.5, 62, 15, 8, "table plane", "grey", italic=True, fs=11.5)
label(89.5, 53, 33, 7, r"camera $T_t$, gravity $\mathbf{g}$", "grey", fs=12)
arrow(R(fs_), (69, 66), C["grey"][1]); arrow((86, 66), L(tp), C["grey"][1]); arrow(R(ds), (89.5, 56.5), C["grey"][1])
label(106.5, 62, 16, 8, "metric\ngeometry", "grey", fs=11)

panel(42, 2, 82, 46, "blue", "Object Estimator", title_y=4.6)
s3 = box(44.5, 36, 21, 9, "SAM 3\n(text-prompted)", "blue", italic=True, fs=12)
img("masks.png", 69, 33, 17, 13)
math(84, 35, r"$\mathcal{S}^o_t$", 16)
s3d = box(44.5, 23.5, 21, 8, "SAM 3D Objects\n+ stereo scale", "blue", italic=True, fs=11)
img("meshes.png", 69, 20.5, 17, 11.5, crop=(0.15, 0.85, 0.1, 0.9))
math(84, 22.5, r"$\mathcal{M}^o$", 16)
fp = box(89.5, 36, 15, 9, "FoundationPose\nall ~252 hyp.", "blue", italic=True, fs=10)
img("fp_hypotheses.png", 107, 33, 15, 13, crop=(0.15, 0.65, 0.5, 1.0))
math(114.5, 47.0 - 0.6, r"$\{X^k_t\}$", 14)
te = box(89.5, 21, 15, 9, "TrackEverything\n3D tracks", "blue", italic=True, fs=10)
img("tracks.png", 107, 17.5, 15, 13, crop=(0.2, 0.7, 0.28, 0.95))
math(114.5, 15.6, r"$p_n(t)$", 14)
label(44.5, 8, 59, 8, "own meshes only - no provided scans, no mocap", "blue", fs=11)
arrow(R(s3), (69, 39.5), C["blue"][1]); arrow((86, 39.5), L(fp), C["blue"][1])
arrow((60, 36), (55, 31.5), C["blue"][1], rad=0.2); arrow(R(s3d), (69, 27), C["blue"][1])
arrow((86, 37), (97, 30), C["blue"][1], rad=-0.2); arrow((104.5, 40.5), (107, 40.5), C["blue"][1])
arrow((104.5, 25.5), (107, 25.5), C["blue"][1])

# inputs -> estimators
arrow((35, 80), L(dh), C["yellow"][1], rad=-0.15, lw=2.4)
arrow((35, 72), L(fs_), C["grey"][1], rad=0.0, lw=2.4)
arrow((35, 66), L(ds), C["grey"][1], rad=0.15, lw=2.4)
arrow((35, 60), (44.5, 42), C["blue"][1], rad=0.2, lw=2.4)
arrow(R(obj), (44.5, 38), C["green"][1], rad=0.35, lw=2.6)
arrow((77.5, 53), (55, 45), C["grey"][1], rad=0.15, lw=1.6)     # depth -> SAM3 scale / tracks
arrow((80, 53), (97, 45), C["grey"][1], rad=-0.2, lw=1.6)

# ---------------- column 3: SelectTrack ----------------
panel(127, 24, 36, 78, "red", "SelectTrack:\ninteraction-aware refinement", title_y=97.2, fs=14)
r1 = box(129.5, 76, 31, 14, "", "red", fs=11)
ax.text(145, 86.8, "Whole-sequence selection", ha="center", fontsize=12, color=C["red"][2], zorder=6)
math(145, 80.4, r"$\min_{k_{1:T}}\sum_t u_t(\mathcal{S}^o_t,X^{k_t}_t,p_n)+\lambda\,m_t(p_n)$", 11.5)
r2 = box(129.5, 61, 31, 11, "", "red")
ax.text(145, 68.8, "Speed-limit flip filter", ha="center", fontsize=12, color=C["red"][2], zorder=6)
ax.text(145, 64, "keep / flip / skip, ≤20° & 6 cm / frame", ha="center", fontsize=10, color="#333", zorder=6)
r3 = box(129.5, 44, 31, 13, "", "red")
ax.text(145, 53.6, "State-aware guards", ha="center", fontsize=12, color=C["red"][2], zorder=6)
math(145, 49.4, r"$s_t\in\{\mathrm{resting},\,\mathrm{held}\}$", 12)
ax.text(145, 46.0, "rest hold · spin guard · twist unwrap", ha="center", fontsize=9.5, color="#333", zorder=6)
r4 = box(129.5, 27, 31, 13, "", "red")
ax.text(145, 36.6, "Contact sanity", ha="center", fontsize=12, color=C["red"][2], zorder=6)
ax.text(145, 31.6, "objects rest on the table\nhands pushed out of objects", ha="center", fontsize=9.5, color="#333", zorder=6, linespacing=1.2)
for a_, b_ in ((r1, r2), (r2, r3), (r3, r4)):
    arrow(B(a_), T(b_), "#e48f89", lw=7, ms=22, style="-|>")
arrow((122, 39.5), (129.5, 83), C["blue"][1], rad=-0.25)
arrow((122, 25.5), (129.5, 79), C["blue"][1], rad=-0.3)
arrow((122, 86.5), (129.5, 50), C["yellow"][1], rad=0.25)
arrow((122, 84), (129.5, 33), C["yellow"][1], rad=0.3)

panel(127, 2, 36, 19, "red", None)
ax.text(133, 17.2, "4D HOI reconstruction", fontsize=12.5, color=C["red"][2], ha="left", zorder=6)
img("scene4d.png", 129.5, 3.5, 17, 12, crop=(0.08, 0.95, 0.69, 1.0))
math(154.5, 9.5, r"$\{\mathcal{M}^h_t,\ X^o_t,\ \mathcal{M}^o\}$", 12)
arrow(B(r4), (145, 21), "#e48f89", lw=7, ms=22)

# ---------------- column 4: robot ----------------
panel(166, 2, 32, 100, "robot", "Robot Learning", title_y=4.7, fs=15)
g = box(168.5, 9, 27, 15, "Simulator grounding\nsettled starts · carved\ncollision · cup bases", "robot", fs=10.5, lw=2.4)
rt = box(168.5, 29, 27, 19, "", "robot", lw=2.4)
ax.text(170.5, 42.5, "Retarget\nto Sharpa\nhands", fontsize=11.5, color="#333", ha="left", va="center", zorder=6, linespacing=1.15)
img("sharpa_hand.png", 182, 29.5, 13, 18, crop=(0.1, 0.9, 0.3, 0.75))
sac = box(168.5, 53, 27, 15, "FlashSAC per episode\ntask + contact reward\nfriction 1.0 / 1.5, 2 seeds", "robot", fs=10.5, lw=2.4,
          fill="#f7f7f5")
ev = box(168.5, 72, 27, 9, "official evaluator →\nper-episode best run", "robot", fs=10.5, lw=2.4)
img("rollout_4.png", 168.5, 84, 8.6, 13); img("rollout_9.png", 177.7, 84, 8.6, 13); img("rollout_14.png", 186.9, 84, 8.6, 13)
ax.text(182, 99.3, "Kaggle rollouts + meshes", fontsize=11, color="#444", ha="center", zorder=6)
for a_, b_ in ((g, rt), (rt, sac), (sac, ev)):
    arrow(T(a_), B(b_), "#a9a9a5", lw=7, ms=22)
arrow(T(ev), (182, 84), "#a9a9a5", lw=7, ms=22)
arrow((163, 11.5), (168.5, 16), "#e48f89", lw=7, ms=22)

for ext in ("pdf", "svg", "png"):
    fig.savefig(D / f"arch.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
print("wrote arch.pdf/svg/png")
