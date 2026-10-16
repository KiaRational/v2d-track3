"""Stage-6b loss diagnosis (host venv). Needs <run>/stereo/wilor2d.json (dump_wilor2d.py)."""
import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/v2d-vulcan/stereo"))
from hand_triangulation import triangulate_joints

ROOT = Path(os.environ["V2D_ROOT"]); run = ROOT / "runs" / sys.argv[1]
K_OFF = int(sys.argv[2]) if len(sys.argv) > 2 else -1
W = json.loads((run / "stereo/wilor2d.json").read_text())
cr = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
si = json.loads((run / "stereo/intrinsics.json").read_text())
ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
Ka = np.array([[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.]])
Kc, Kb = np.array(si["ego_cam_c"]["K"]), np.array(si["ego_cam_b"]["K"])
z = np.zeros(14)
calib = {"cameras": {"a": {"K": Ka, "dist": z}, "c": {"K": Kc, "dist": z}, "b": {"K": Kb, "dist": z}},
         "T_left_to_right": np.array(cr["stereo"]["T_left_to_right"]), "T_a_to_c": np.array(cr["T_ego_cam_a_to_ego_cam_c"])}
T_ac = calib["T_a_to_c"]; T_cb = calib["T_left_to_right"]
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def Fmat(T, K1, K2):  # x2^T F x1 = 0, X2 = R X1 + t
    R, t = T[:3, :3], T[:3, 3]; return np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)
F = {"cb": Fmat(T_cb, Kc, Kb), "ac": Fmat(T_ac, Ka, Kc), "ab": Fmat(T_cb @ T_ac, Ka, Kb)}
def epi(Fm, x1, x2):
    h1 = np.c_[x1, np.ones(len(x1))]; h2 = np.c_[x2, np.ones(len(x2))]
    l = h1 @ Fm.T; return np.abs((l * h2).sum(1)) / np.hypot(l[:, 0], l[:, 1])
n = len(W["c"])
D = {v: [[d for d in W[v][t] if d["score"] >= 0.3] for t in range(n)] for v in "acb"}
for d in (x for v in D for t in D[v] for x in t): d["j2d"] = np.array(d["j2d"])
side = lambda d: "right" if d["is_right"] else "left"
project = lambda X, K: X[:, :2] / X[:, 2:3] @ K[:2, :2].T + K[:2, 2]

print("== counts per view (score>=0.3), frames with >=1 det of side")
for v in "acb":
    print(v, {s: sum(any(side(d) == s for d in D[v][t]) for t in range(n)) for s in ("left", "right")},
          "frames with 0 dets:", sum(len(D[v][t]) == 0 for t in range(n)))

print("== c-b sync check: median over frames of best-pair median epipolar px, b frame offset d")
for dd in (-2, -1, 0, 1, 2):
    e = []
    for t in range(n):
        if not 0 <= t + dd < n: continue
        es = [np.median(epi(F["cb"], c["j2d"], b["j2d"])) for c in D["c"][t] for b in D["b"][t + dd]]
        es = sorted(es)[:2]
        e += es
    print(f"  d={dd:+d}: median {np.median(e):.2f} px  (n={len(e)})")

print("== a-c / a-b epipolar sync check vs stereo offset k (same-label pairs, best per frame/side)")
for k in range(-3, 4):
    row = []
    for key, v in (("ac", "c"), ("ab", "b")):
        e = []
        for t in range(n):
            if not 0 <= t + k < n: continue
            for a in D["a"][t]:
                es = [np.median(epi(F[key], a["j2d"], x["j2d"])) for x in D[v][t + k]]
                if es: e.append(min(es))
        row.append(f"{key} {np.median(e):.2f}")
    print(f"  k={k:+d}:", ", ".join(row))

print("== stereo pairs: triangulation at max_px 8 (per pair: n_ok joints, median rms, median epi)")
npairs, npass, okhist, rms_all, epi_all, per_frame_pass = 0, 0, Counter(), [], [], []
cand = []
for t in range(n):
    cs = []
    for ci, c in enumerate(D["c"][t]):
        for bi, b in enumerate(D["b"][t]):
            obs = {"c": (c["j2d"], np.full(21, c["score"])), "b": (b["j2d"], np.full(21, b["score"]))}
            Xt, rms, _, ok = triangulate_joints(obs, calib, ref="a", max_px=8.0)
            e = np.median(epi(F["cb"], c["j2d"], b["j2d"]))
            npairs += 1; okhist[int(ok.sum()) // 3 * 3] += 1
            same = c["is_right"] == b["is_right"]
            if e < 15: rms_all.append(np.nanmedian(rms)); epi_all.append(e)
            if ok.sum() >= 10:
                npass += 1; cs.append((Xt, ok, ci, bi, same, e))
    cand.append(cs); per_frame_pass.append(len(cs))
print(f"  pairs {npairs}, pass(>=10 ok) {npass}; ok-count histogram (bins of 3): {sorted(okhist.items())}")
print(f"  for pairs with epi<15px: median rms {np.median(rms_all):.2f}, median epi {np.median(epi_all):.2f}")
print("  frames by #passing pairs:", sorted(Counter(per_frame_pass).items()))

print(f"== colour match at k={K_OFF}: per side loss reasons")
for s in ("left", "right"):
    reasons = Counter(); pxs = []
    for t in range(n):
        if not 0 <= t + K_OFF < n: reasons["out_of_range"] += 1; continue
        da = [d for d in D["a"][t] if side(d) == s]
        cs = cand[t + K_OFF]
        if not cs: reasons["no_stereo_cand"] += 1; continue
        if not da: reasons["no_a_det(stereo ok)"] += 1; continue
        ref = max(da, key=lambda d: d["score"])["j2d"]
        best = min(np.linalg.norm(project(X[ok], Ka) - ref[ok], axis=1).mean() for X, ok, *_ in cs)
        pxs.append(best)
        reasons["pass" if best < 40 else "px_gate_fail"] += 1
    print(" ", s, dict(reasons), "px_to_a pct 25/50/75/90:", np.percentile(pxs, [25, 50, 75, 90]).round(1))
# per-frame table
print("== per frame: t | a L/R | c L/R | b L/R | pass pairs(t+k) | best px L/R")
for t in range(n):
    cnt = lambda v, tt: "%d/%d" % tuple(sum(side(d) == s for d in D[v][tt]) for s in ("left", "right")) if 0 <= tt < n else "-"
    tk = t + K_OFF; cs = cand[tk] if 0 <= tk < n else []
    bp = []
    for s in ("left", "right"):
        da = [d for d in D["a"][t] if side(d) == s]
        if da and cs:
            ref = max(da, key=lambda d: d["score"])["j2d"]
            bp.append("%.0f" % min(np.linalg.norm(project(X[ok], Ka) - ref[ok], axis=1).mean() for X, ok, *_ in cs))
        else: bp.append("-")
    print(f"{t:3d} | {cnt('a', t)} | {cnt('c', tk)} | {cnt('b', tk)} | {len(cs)} | {'/'.join(bp)}")
