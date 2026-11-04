"""System architecture figure (style after the reference: coloured stage panels, white rounded method boxes, inline
thumbnails from our own data, colour-coded flow arrows). Output: arch.pdf / arch.svg / arch.png.
Assets: report/arch/assets (make_assets.py, make_assets2.py). Coordinates: 200 x 92 units."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
PINK, GREYA = "#d2463f", "#8d8d88"

fig = plt.figure(figsize=(20, 9.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200); ax.set_ylim(0, 92); ax.set_aspect("equal"); ax.axis("off")


def panel(x, y, w, h, key, title=None, title_y=None, fs=15.5):
    fc, ec, tc = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec="none", zorder=0))
    if title:
        ax.text(x + w / 2, (title_y if title_y is not None else y + h - 3.2), title, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold", zorder=5, linespacing=1.1)


def box(x, y, w, h, text, key, italic=False, fs=12, fill="white", lw=2.2, bold=False):
    fc, ec, tc = C[key]
    fs += 0.8
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fill, ec=ec, lw=lw, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                style="italic" if italic else "normal", weight="bold", zorder=4, linespacing=1.15)
    return (x, y, w, h)


def label(x, y, w, h, text, key, fs=11.5):
    fc, ec, tc = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=ec, ec="none", alpha=0.18, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4, linespacing=1.15)
    return (x, y, w, h)


def img(name, x, y, w, h, crop=None, frame=True):
    a = mpimg.imread(A / name)
    if crop:
        r0, r1, c0, c1 = crop
        a = a[int(r0 * a.shape[0]):int(r1 * a.shape[0]), int(c0 * a.shape[1]):int(c1 * a.shape[1])]
    ax.imshow(a, extent=(x, x + w, y, y + h), zorder=4, aspect="auto")
    if frame:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.3", fc="none", ec="white", lw=1.4, zorder=5))


def arrow(p0, p1, col, rad=0.0, lw=2.4, ms=16, z=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=1))


def elbow(pts, col, lw=2.4, z=6, halo=False):
    """Orthogonal polyline through pts, arrow head on the last segment."""
    xs, ys = zip(*pts[:-1])
    line, = ax.plot(list(xs) + [pts[-2][0]], list(ys) + [pts[-2][1]], color=col, lw=lw, zorder=z, solid_capstyle="round",
                    solid_joinstyle="round")
    if halo:
        line.set_path_effects([pe.Stroke(linewidth=lw + 3, foreground="white"), pe.Normal()])
    arrow(pts[-2], pts[-1], col, lw=lw, z=z)


def fat(p0, p1, col):
    arrow(p0, p1, col, lw=2.4, ms=16)


def R(b): return (b[0] + b[2], b[1] + b[3] / 2)
def L(b): return (b[0], b[1] + b[3] / 2)
def T(b): return (b[0] + b[2] / 2, b[1] + b[3])
def B(b): return (b[0] + b[2] / 2, b[1])


def math(x, y, s, fs=14, col="#222", ha="center"):
    ax.text(x, y, s, fontsize=fs, color=col, ha=ha, va="center", zorder=7)


def head(x, y, s, key, fs=13.2):
    ax.text(x, y, s, ha="center", fontsize=fs, color=C[key][2], weight="bold", zorder=6)


# ---------------- column 1: input ----------------
panel(2, 36, 36, 54, "green", "Egocentric Episode", title_y=86.6)
for k, (nm, dx, dy) in enumerate((("video_2.png", 0, 6), ("video_6.png", 2.2, 3), ("video_10.png", 4.4, 0))):   # stacked frames
    ax.add_patch(FancyBboxPatch((5.6 + dx, 51.2 + dy), 25.5, 19.1, boxstyle="round,pad=0,rounding_size=0.4", fc="#3a3a38",
                                ec="none", alpha=0.25, zorder=3))
    img(nm, 5 + dx, 52 + dy, 25.5, 19.1, frame=False)
label(9.95, 78.6, 20, 4.2, "raw video", "green", fs=11)
box(8.5, 40, 23, 7.5, "colour camera +\nmono stereo pair", "green", fs=10.5)
panel(2, 2, 36, 31, "green", "Task & Objects via VLM", title_y=29.8)
vlm = box(5, 20.5, 30, 6, "Qwen3-VL (local)", "green", italic=True, fs=13)
ax.text(20, 14.3, "“The person picks up a blue cup and a\nwooden spoon, stirs the contents of the\ncup with the spoon, "
        "and then places\nboth items back on the table.”", ha="center", va="center", fontsize=9.5, style="italic",
        color="#3f7a2a", linespacing=1.15)
obj = box(5, 3, 30, 6, "“wooden spoon”, “blue cup”", "green", fs=11.2)
arrow(B(vlm), (20, 18.0), C["green"][1], lw=2)
arrow((20, 10.6), T(obj), C["green"][1], lw=2)

# ---------------- column 2: estimators ----------------
YH = 78.75
panel(42, 70, 82, 20, "yellow", "Hand Estimator", title_y=87.4)
dh = box(44.5, YH - 4.25, 15, 8.5, "Dyn-HaMR", "yellow", italic=True, fs=13)
mf = box(63, YH - 3.75, 21, 7.5, "metric MANO\ntrajectory", "yellow", fs=11.5, fill="#fff8e0")
box(63, 71.2, 21, 3.1, "WiLoR stereo fallback (<2%)", "yellow", italic=True, fs=7.8)
hp = box(88, YH - 6.5, 15, 13, "hands out\nof objects", "yellow", fs=10.5, fill="#fff8e0")
img("hand4d.png", 104.5, 70.3, 19, 17, crop=(0.3, 0.8, 0.15, 0.9), frame=False)
math(114, 87.6, r"$\mathcal{M}^h_t$", 15)
arrow(R(dh), L(mf), C["yellow"][1]); arrow(R(mf), L(hp), C["yellow"][1]); arrow((103, YH), (106, YH), C["yellow"][1])

panel(42, 44, 82, 23, "grey", "Metric Geometry", title_y=64.6, fs=13.5)
fs_ = box(44.5, 54.5, 19.5, 6.5, "FoundationStereo", "grey", italic=True, fs=11)
ds = box(44.5, 46, 19.5, 6.5, "DROID-SLAM", "grey", italic=True, fs=11.5)
img("depth.png", 66, 52.3, 13.5, 10.5)
math(72.7, 57.6, r"$D_t$", 17, col="white")
img("slam.png", 82, 44.3, 18, 15.5, crop=(0.3, 0.85, 0.15, 0.85), frame=False)
math(99.5, 56, r"$T_t$", 16)
tpb = box(103, 53.5, 19, 6.5, "table plane", "grey", italic=True, fs=11)
gr = label(103, 45.5, 19, 6, r"gravity $\mathbf{g}$", "grey", fs=12)
arrow(R(fs_), (66, 57.75), C["grey"][1]); arrow(R(ds), (82.5, 49.25), C["grey"][1])
elbow([(79.5, 62), (112.5, 62), (112.5, 60)], C["grey"][1], lw=1.6)
arrow(B(tpb), T(gr), C["grey"][1], lw=1.6)

panel(42, 2, 82, 39, "blue", "Object Estimator", title_y=4.6)
s3 = box(44.5, 29, 21, 8.5, "SAM 3\n(text-prompted)", "blue", italic=True, fs=12)
img("masks.png", 69, 27, 17, 12.5)
math(84, 28.8, r"$\mathcal{S}^o_t$", 16)
s3d = box(44.5, 14, 21, 8, "SAM 3D Objects\n+ stereo scale", "blue", italic=True, fs=11)
img("meshes.png", 69, 10, 17, 12.5, crop=(0.15, 0.85, 0.1, 0.9))
math(84, 11.8, r"$\mathcal{M}^o$", 16)
fp = box(89.5, 29, 15, 8.5, "FoundationPose\nall ~252 hyp.", "blue", italic=True, fs=9.6)
img("fp_hypotheses.png", 107, 27, 15, 12.5, crop=(0.15, 0.65, 0.5, 1.0))
math(114.5, 40.3, r"$\{X^k_t\}$", 12)
te = box(89.5, 14, 15, 8, "TrackEverything\n3D tracks", "blue", italic=True, fs=9.6)
img("tracks.png", 107, 10, 15, 12.5, crop=(0.2, 0.7, 0.28, 0.95))
math(114.5, 8.3, r"$p_n(t)$", 13)
arrow(R(s3), (69, 33.25), C["blue"][1]); arrow((86, 33.25), L(fp), C["blue"][1])
arrow(B(s3), T(s3d), C["blue"][1]); arrow(R(s3d), (69, 18), C["blue"][1])
arrow((86, 18), L(te), C["blue"][1]); arrow((86, 21), (93, 29), C["blue"][1], lw=1.8)
arrow((104.5, 33.25), (107, 33.25), C["blue"][1]); arrow((104.5, 18), (107, 18), C["blue"][1])
elbow([(91, 44), (91, 42), (96.5, 37.5)], C["grey"][1], lw=2.2)
ax.text(92.3, 41.9, r"$D_t,\ T_t$", fontsize=11, color=C["grey"][2], zorder=7)

# inputs -> estimators: one orthogonal trunk in the gap between the columns
IN = "#5f8f45"
ax.plot([34.9, 40], [64.5, 64.5], color=IN, lw=2.4, zorder=6, solid_capstyle="round")
ax.plot([40, 40], [35.5, YH], color=IN, lw=2.4, zorder=6, solid_capstyle="round")
ax.add_patch(plt.Circle((40, 64.5), 0.55, color=IN, zorder=7))
for y_ in (YH, 57.75, 49.25, 35.5):
    elbow([(40, y_), (44.5, y_)], IN)
elbow([(35, 6), (40, 6), (40, 31), (44.5, 31)], C["green"][1])   # VLM object names -> SAM 3 prompts

# ---------------- column 3: SelectTrack ----------------
panel(127, 24, 36, 66, "red", "SelectTrack:\ninteraction-aware refinement", title_y=85.4, fs=14)
r1 = box(129.5, 66, 31, 12, "", "red")
head(145, 74.6, "Whole-sequence selection", "red")
math(145, 69.6, r"$\min_{k_{1:T}}\sum_t u_t(\mathcal{S}^o_t,X^{k_t}_t,p_n)+\lambda\,m_t(p_n)$", 11.5)
r2 = box(129.5, 54.5, 31, 7, "Speed-limit flip filter", "red", fs=12.4)
r3 = box(129.5, 39, 31, 11, "", "red")
head(145, 46.4, "State-aware guards", "red")
math(145, 42.4, r"$s_t\in\{\mathrm{resting},\,\mathrm{held}\}$", 12)
r4 = box(129.5, 26.5, 31, 7, "Contact sanity", "red", fs=12.4)
for a_, b_ in ((r1, r2), (r2, r3), (r3, r4)):
    fat(B(a_), T(b_), PINK)
# estimators -> refinement: orthogonal, bundled
elbow([(122, YH), (124.0, YH), (124.0, 44.5), (129.5, 44.5)], C["yellow"][1])
elbow([(124.0, 44.5), (124.0, 30), (129.5, 30)], C["yellow"][1])
ax.add_patch(plt.Circle((124.0, 44.5), 0.55, color=C["yellow"][1], zorder=7))
elbow([(122, 33.25), (125.6, 33.25), (125.6, 72), (129.5, 72)], C["blue"][1], halo=True)
ax.plot([122, 125.6, 125.6], [18, 18, 33.25], color=C["blue"][1], lw=2.4, zorder=6, solid_joinstyle="round")
ax.add_patch(plt.Circle((125.6, 33.25), 0.55, color=C["blue"][1], zorder=7))

panel(127, 2, 36, 19.5, "red", "4D HOI Reconstruction", title_y=18.6, fs=13)
img("scene4d_render.png", 128.5, 2.5, 19, 13.5, crop=(0.28, 0.82, 0.18, 0.88), frame=False)
math(155.5, 9.5, r"$\{\mathcal{M}^h_t,\ X^o_t,\ \mathcal{M}^o\}$", 12)
fat(B(r4), (145, 21.5), PINK)

# ---------------- column 4: robot ----------------
panel(166, 2, 32, 88, "robot", "Robot Learning", title_y=84.6, fs=15.5)
g = box(168.5, 8, 27, 8, "Simulator grounding", "robot", lw=2.4, fs=11.7)
rt = box(168.5, 21, 27, 17, "", "robot", lw=2.4)
ax.text(171, 29.5, "Retarget", fontsize=13, color=C["robot"][2], weight="bold", ha="left", va="center", zorder=6)
img("sharpa_hand.png", 183, 21.5, 12, 16, crop=(0.1, 0.9, 0.3, 0.75), frame=False)
sac = box(168.5, 43, 27, 7, "SAC", "robot", lw=2.4, fs=12.5)
pi = box(168.5, 55, 27, 7, r"Robot policy $\pi$", "robot", lw=2.4, fs=12.5, fill="#f7f7f5")
img("rollout_4.png", 168.5, 68.5, 13.2, 9.3); img("rollout_14.png", 182.3, 68.5, 13.2, 9.3)
for a_, b_ in ((g, rt), (rt, sac), (sac, pi)):
    fat(T(a_), B(b_), GREYA)
fat(T(pi), (182, 68.5), GREYA)
fat((163, 12), (168.5, 12), PINK)

for ext in ("pdf", "svg", "png"):
    fig.savefig(D / f"arch.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
print("wrote arch.pdf/svg/png")
