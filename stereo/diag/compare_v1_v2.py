"""Compare v1 vs v2 hands_tri.json against colour WiLoR 2D (same metric for both) + overlay frames.
Host venv. args: run frames_to_draw..."""
import json, os, sys
from itertools import product
from pathlib import Path
import cv2, numpy as np
ROOT = Path(os.environ["V2D_ROOT"]); run = ROOT / "runs" / sys.argv[1]
ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
Ka = np.array([[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.]])
A = [[np.array(d["j2d"]) for d in fr if d["score"] >= 0.3] for fr in json.loads((run / "stereo/wilor2d.json").read_text())["a"]]
V = {"v1": json.loads((run / "stereo/hands_tri.v1.json").read_text()), "v2": json.loads((run / "stereo/hands_tri.json").read_text())}
SIDES = ("left", "right")
proj = lambda X: X[:, :2] / X[:, 2:3] @ Ka[:2, :2].T + Ka[:2, 2]
def hands(v, t):
    out = {}
    for s in SIDES:
        x = V[v][s]["X_smooth"].get(str(t))
        if x is not None:
            X = np.array(x, float); m = np.isfinite(X).all(1)
            if m.sum() >= 6: out[s] = (proj(np.where(m[:, None], X, 1)), m)
    return out
def match(h, dets):
    cost = {(s, i): float(np.linalg.norm(uv[m] - d[m], axis=1).mean()) for s, (uv, m) in h.items() for i, d in enumerate(dets)}
    best, bc = {}, np.inf
    for combo in product(*[[None] + list(range(len(dets))) for _ in h]):
        ids = [i for i in combo if i is not None]
        if len(ids) != len(set(ids)): continue
        c = sum(cost[(s, i)] if i is not None else 150 for s, i in zip(h, combo))
        if c < bc: bc, best = c, dict(zip(h, combo))
    return {s: cost[(s, i)] for s, i in best.items() if i is not None}
n = len(A); res = {v: {s: {} for s in SIDES} for v in V}
for v in V:
    for t in range(n):
        for s, d in match(hands(v, t), A[t]).items(): res[v][s][t] = d
for s in SIDES:
    f1, f2 = set(V["v1"][s]["frames"]), set(V["v2"][s]["frames"])
    common = sorted(f1 & set(res["v1"][s]) & set(res["v2"][s]))
    print(f"{s}: frames v1 {len(f1)} v2 {len(f2)} | matched-to-colour v1 {len(res['v1'][s])} v2 {len(res['v2'][s])}")
    for v in V:
        allv = list(res[v][s].values()); cm = [res[v][s][t] for t in common]
        print(f"   {v}: median px (all matched) {np.median(allv):.1f}, on {len(common)} common frames {np.median(cm):.1f} "
              f"(p90 {np.percentile(cm, 90):.1f})")
# overlay: v1 red, v2 green, colour WiLoR yellow
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
for t in map(int, sys.argv[2:]):
    img = cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png"))
    for d in A[t]:
        for p in d: cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
    for v, col in (("v1", (0, 0, 255)), ("v2", (0, 255, 0))):
        for s, (uv, m) in hands(v, t).items():
            for a, b in BONES:
                if m[a] and m[b]: cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), col, 4, cv2.LINE_AA)
    cv2.putText(img, f"f{t} red=v1 green=v2 yellow=colour WiLoR", (20, 50), 0, 1.5, (255, 255, 255), 3)
    cv2.imwrite(str(ROOT / f"tmp_frames/cmp_f{t:03d}.jpg"), cv2.resize(img, (1014, 760)))
