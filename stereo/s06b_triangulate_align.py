"""Stage 6b (v2): metric 3D hand joints from stereo + alignment of hand models to them
(runs in the `agile` image: manotorch, cv2, scipy).

Pipeline (per episode; stereo frame index s, colour frame index t):
 1. 2D joints per detection = WiLoR's MANO joints projected with WiLoR's own camera —
    colour 'a' (toolkit wilor_raw) and undistorted stereo 'c'/'b' (Stage 6a; plus, when
    present, Stage 6a-v2's CLAHE detections merged by bbox IoU and its flipped-handedness
    re-inference of every stereo box, used as a "twin" detection sharing the box id).
 2. Candidates per stereo frame: every c x b pair with the same hand model (a detection or its
    flipped twin; mixed-label pairs only without twins) is triangulated (c/b are hardware-
    synced; confidence-weighted DLT+LM, colour-camera frame); implausible single joints
    (over-long bones, far from the hand) are dropped; the pair is kept if enough joints remain
    and the hand size is plausible. In pass 2 an 'a' detection paired with a single stereo view
    (a+c / a+b, nearest colour frame) is added as a fallback for hands one stereo camera misses.
 3. Hand identity: Viterbi over frames choosing (left cand | none, right cand | none): a
    (v2.1: candidates beyond arm's reach (--max_reach, other people in the background) are
    dropped and softly penalised from --soft_reach; a left palm right of the right palm is
    penalised (swap guard); jumps > 15 cm/frame cost steeply (teleport = identity error).)
    candidate may only take the side of its hand model; costs = detector handedness votes of
    all contributing views (+ the colour detection it projects onto, pass 2), triangulation
    quality, a weak image-side prior and 3D temporal continuity. Colour detections are NOT
    required any more (v1 dropped every frame colour WiLoR missed or mislabelled).
 4. Joints that fail triangulation are completed from the WiLoR hand of that view
    (similarity-aligned to the triangulated joints) with low confidence; per-frame joint
    outliers are down-weighted with a temporal Hampel test.
 5. Sub-frame sync: colour frame t <-> stereo time t + tau (tau fractional) is estimated by
    minimising the median distance between the interpolated stereo hands projected into 'a'
    and the colour WiLoR 2D joints. Hands are resampled to colour time by interpolation.
 6. Short gaps (<= --max_gap frames) are filled by interpolation; zero-phase confidence-
    weighted local-quadratic smoothing (v1's causal One-Euro filter lagged fast motion by
    ~100 px in the colour image).
 7. Model alignment (wilor/hawor/agile) to the triangulated joints, as in v1.
Outputs: <run>/stereo/hands_tri.json (v1 schema + extra fields), <run>/stereo/aligned_<model>/
<side>/NNNNNN.json, viz $V2D_VIZ/07_stereo_hands/<run>.mp4 (green = stereo hand, cyan = gap-
filled, magenta = aligned model, yellow dots = colour WiLoR 2D joints).

hands_tri.json: top-level sync_offset (int, = round(sync_offset_frac)), sync_offset_frac,
offset_scores, summary, stereo_debug; per side: frames (colour frame ids with a hand), stats
(per frame: frame, px_to_a, px_to_a_raw, a_label_agrees, a_used_in_triangulation, rms_px,
n_valid, source in {cb, ac, ab, interp}, conf_mean), X_smooth{t: 21x3} (metres, colour-camera
frame, colour time t; all 21 joints finite), X_raw{t}, conf{t: 21}, valid{t: 21} (measured
joints; completed / gap-filled joints have valid=False and conf < 0.25 — weight by conf).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import cv2
import imageio
import numpy as np

sys.path.insert(0, "/code/agile")
sys.path.insert(0, "/code/stereo")
from geometry import Mano, wilor_keypoints_2d  # noqa: E402
from hand_triangulation import align_hand_to_triangulation, triangulate_joints, umeyama  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
SIDES = ("left", "right")
SGN = {"left": -1.0, "right": 1.0}
HAND_TRACK = {"left": 2, "right": 3}
PALM = [0, 5, 9, 13, 17]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


# ----------------------------------------------------------------------------- detections
def iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    return inter / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter, 1e-9)


def detections(groups, mano: Mano, wh, min_score):
    """WiLoR records of one frame. groups = [(dets_json, flipped_json | None), ...]; later groups only
    add boxes that do not overlap earlier ones. Each kept box gets an id; if a flipped-handedness
    re-inference of the box exists (Stage 6a-v2), it is added as a twin with the same id.
    Per det: right = hand model of its joints, det_right = the detector's own label."""
    out, nbox = [], 0

    def make(rec, box, bid, det_right, score, extra, twin):
        _, j = mano.local(rec)
        j = j.cpu().numpy()
        return {"rec": rec, "box": box, "jl": j, "j2d": wilor_keypoints_2d(j, rec, wh), "score": score,
                "right": bool(rec["is_right"]), "det_right": det_right, "id": bid, "extra": extra, "twin": twin}

    for k, (path, fpath) in enumerate(groups):
        if not path.exists():
            continue
        flips = json.loads(fpath.read_text()) if fpath is not None and fpath.exists() else []
        for rec in json.loads(path.read_text()):
            if rec.get("score", 0) < min_score:
                continue
            bb = rec["bbox"]
            box = [bb["x0"], bb["y0"], bb["x1"], bb["y1"]]
            if k > 0 and any(iou(box, d["box"]) > 0.5 for d in out):
                continue
            score, dr = float(rec.get("score", 1.0)), bool(rec["is_right"])
            out.append(make(rec, box, nbox, dr, score, k > 0, False))
            for fr in flips:
                fb = fr["bbox"]
                if fr["is_right"] != dr and iou(box, [fb["x0"], fb["y0"], fb["x1"], fb["y1"]]) > 0.8:
                    out.append(make(fr, box, nbox, dr, score, k > 0, True))
                    break
            nbox += 1
    return out


def project(X, K):
    return X[..., :2] / X[..., 2:3] @ K[:2, :2].T + K[:2, 2]


def palm_centre(X, ok):
    idx = [j for j in PALM if ok[j]]
    return X[idx].mean(0) if len(idx) >= 2 else X[ok].mean(0)


def clean_joints(X, ok):
    """Drop individual implausible joints (e.g. a thumb extrapolated outside a truncated view):
    over-long bones (wrist->MCP > 13 cm, finger segment > 7 cm) invalidate the child joint;
    joints > 18 cm from the hand's median point are dropped."""
    ok = ok.copy()
    long_ = [(a, b) for a, b in BONES if ok[a] and ok[b] and np.linalg.norm(X[a] - X[b]) > (0.13 if a == 0 else 0.07)]
    if sum(a == 0 for a, _ in long_) >= 3:  # the wrist itself is the outlier
        ok[0] = False
        long_ = [(a, b) for a, b in long_ if a != 0]
    for _, b in long_:
        ok[b] = False
    if ok.any():
        med = np.median(X[ok], 0)
        ok &= ~(np.linalg.norm(X - med, axis=1) > 0.18)
    return ok


def plausible(X, ok):
    if ok[0] and ok[9]:
        d = np.linalg.norm(X[0] - X[9])
        if not 0.05 <= d <= 0.15:
            return False
    P = X[ok]
    return float(np.linalg.norm(P - P.mean(0), axis=1).max()) < 0.2


def complete(X, ok, jl):
    """Fill joints that failed triangulation from the WiLoR hand (similarity-aligned)."""
    X = X.copy()
    if (~ok).any():
        try:
            s, R, t = umeyama(jl[ok], X[ok])
            X[~ok] = s * jl[~ok] @ R.T + t
        except ValueError:
            return None
    return X


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--min_score", type=float, default=0.3)
    ap.add_argument("--max_px", type=float, default=10.0, help="per-joint triangulation reprojection gate")
    ap.add_argument("--min_joints", type=int, default=8, help="min triangulated joints for a candidate")
    ap.add_argument("--max_offset", type=float, default=3.0, help="colour/stereo sync search range (frames)")
    ap.add_argument("--max_gap", type=int, default=5, help="max gap (frames) filled by interpolation")
    ap.add_argument("--no_clahe", action="store_true", help="ignore Stage 6a-v2 CLAHE detections")
    ap.add_argument("--no_flip", action="store_true", help="ignore Stage 6a-v2 flipped-handedness re-inference")
    ap.add_argument("--no_viz", action="store_true")
    ap.add_argument("--max_reach", type=float, default=0.9,
                    help="drop hands whose palm is farther than this from the head camera (other people)")
    ap.add_argument("--soft_reach", type=float, default=0.65, help="soft reach penalty starts here (m)")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    meta = json.loads((run / "episode.json").read_text())
    calib_r = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
    sint = json.loads((run / "stereo/intrinsics.json").read_text())
    ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    Ka = np.array([[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.0]])
    z14 = np.zeros(14)
    calib = {"cameras": {"a": {"K": Ka, "dist": z14, "size": [ia["width"], ia["height"]]},
                         "c": {"K": np.array(sint["ego_cam_c"]["K"]), "dist": z14, "size": sint["ego_cam_c"]["size"]},
                         "b": {"K": np.array(sint["ego_cam_b"]["K"]), "dist": z14, "size": sint["ego_cam_b"]["size"]}},
             "T_left_to_right": np.array(calib_r["stereo"]["T_left_to_right"]),
             "T_a_to_c": np.array(calib_r["T_ego_cam_a_to_ego_cam_c"])}
    mano = Mano(str(ROOT / "weights/hand"))
    n = len(list((run / "recon_hawor/frames").glob("*.png")))
    wh = {v: tuple(calib["cameras"][v]["size"]) for v in "acb"}
    fps = meta["fps"]

    use_flip = not args.no_flip and all((run / f"stereo/wilor_{v}_flip").is_dir() for v in "cb")

    def vdirs(v):
        names = [f"wilor_{v}"] + ([f"wilor_{v}_clahe"] if not args.no_clahe else [])
        return [(run / "stereo" / nm, (run / "stereo" / f"{nm}_flip") if use_flip else None)
                for nm in names if (run / "stereo" / nm).is_dir()]

    D = {"a": [detections([(run / f"recon_hawor/wilor_raw/{t:06d}.json", None)], mano, wh["a"], args.min_score)
               for t in range(n)]}
    for v in "cb":
        D[v] = [detections([(d / f"{t:06d}.json", f / f"{t:06d}.json" if f else None) for d, f in vdirs(v)],
                           mano, wh[v], args.min_score) for t in range(n)]
    print("detections (frames with >=1 detector box labelled L/R):",
          {v: [sum(any(d["det_right"] == (s == "right") and not d["twin"] for d in D[v][t]) for t in range(n))
               for s in SIDES] for v in D},
          "| extra CLAHE boxes:", {v: sum(d["extra"] and not d["twin"] for x in D[v] for d in x) for v in "cb"},
          "| flipped twins:", {v: sum(d["twin"] for x in D[v] for d in x) for v in "cb"}, flush=True)

    def tri(obs):
        return triangulate_joints({k: (d["j2d"], np.full(21, d["score"])) for k, d in obs.items()}, calib, ref="a",
                                  max_px=args.max_px)

    def make_cand(obs, uses, src):
        X, rms, _, ok = tri(obs)
        ok = clean_joints(X, ok)
        if ok.sum() < args.min_joints or not plausible(X, ok):
            return None
        # egocentric gate: the wearer's hands are within arm's reach of the head camera;
        # hands of people in the background triangulate just as well but are far away
        reach = float(np.linalg.norm(palm_centre(X, ok)))
        if reach > args.max_reach or palm_centre(X, ok)[2] < 0.05:
            return None
        jl_det = max((d for k, d in obs.items() if k != "a"), key=lambda d: d["score"])
        Xc = complete(X, ok, jl_det["jl"])
        if Xc is None:
            return None
        vote = sum((1.0 if d["det_right"] else -1.0) * d["score"] for d in obs.values())
        labels = {d["right"] for d in obs.values()}
        # with flipped twins available, joints of all views come from the same hand model and the
        # candidate can only be that hand; otherwise (legacy) mixed-label pairs are allowed.
        hand = ("right" if labels == {True} else "left") if len(labels) == 1 and use_flip else None
        return {"X": Xc, "ok": ok, "rms": rms, "q": float(np.median(rms[ok])), "src": src, "uses": uses,
                "vote": vote, "palm": palm_centre(X, ok), "hand": hand, "reach": reach}

    # stereo c x b candidates (pass-independent)
    cand_cb = []
    for s in range(n):
        cs = []
        for c, b in product(D["c"][s], D["b"][s]):
            if use_flip and c["right"] != b["right"]:
                continue
            cd = make_cand({"c": c, "b": b}, {("c", c["id"]), ("b", b["id"])}, "cb")
            if cd is not None:
                cs.append(cd)
        cand_cb.append(cs)
    print("stereo frames with c x b candidates:", sum(bool(c) for c in cand_cb), "/", n,
          "| total candidates:", sum(len(c) for c in cand_cb), flush=True)

    def colour_frame(s, tau):
        t = int(round(s - tau))
        return t if 0 <= t < n else None

    def a_vote(cd, t):
        """Label vote of the colour detection the candidate projects onto (nearest, < 60 px)."""
        if t is None or not D["a"][t]:
            return 0.0, None
        uv = project(cd["X"][cd["ok"]], Ka)
        best = min(D["a"][t], key=lambda d: np.linalg.norm(uv - d["j2d"][cd["ok"]], axis=1).mean())
        dist = float(np.linalg.norm(uv - best["j2d"][cd["ok"]], axis=1).mean())
        return ((1.0 if best["right"] else -1.0) * best["score"] if dist < 60 else 0.0), dist

    def build_candidates(tau):
        cands = []
        for s in range(n):
            cs = [dict(c) for c in cand_cb[s]]
            t = None if tau is None else colour_frame(s, tau)
            if t is not None:
                for cd in cs:
                    cd["vote"] = cd["vote"] + a_vote(cd, t)[0]
                # a + single stereo view fallback: only stereo detections without a c x b partner;
                # best candidate per colour detection
                used = set().union(*[c["uses"] for c in cand_cb[s]]) if cand_cb[s] else set()
                for a in D["a"][t]:
                    fb = []
                    for v in "cb":
                        for dv in D[v][s]:
                            if (v, dv["id"]) in used or (use_flip and dv["right"] != a["right"]):
                                continue
                            cd = make_cand({"a": a, v: dv}, {("a", a["id"]), (v, dv["id"])}, "a" + v)
                            if cd is not None:
                                fb.append(cd)
                    if fb:
                        cs.append(min(fb, key=lambda c: c["q"]))
            cands.append(cs)
        return cands

    M_MISS, BIRTH, W_T, W_X, SRC_PEN = 4.0, 0.5, 0.5, 1.0, 1.5
    W_REACH, CROSS_PEN, CROSS_M = 2.0, 3.0, 0.05

    def assign_cost(cd, side):
        sg = SGN[side]
        return (-sg * cd["vote"] + min(cd["q"] / 4.0, 2.0) + (SRC_PEN if cd["src"] != "cb" else 0.0)
                - sg * W_X * float(np.clip(cd["palm"][0] / 0.15, -1, 1))
                + W_REACH * max(0.0, (cd["reach"] - args.soft_reach) / 0.1) ** 2)

    def viterbi(cands):
        states, unary = [], []
        for s in range(n):
            cs, st, un = cands[s], [], []
            for iL, iR in product(range(-1, len(cs)), repeat=2):
                if (iL >= 0 and cs[iL]["hand"] == "right") or (iR >= 0 and cs[iR]["hand"] == "left"):
                    continue
                if iL >= 0 and iR >= 0:
                    if iL == iR or cs[iL]["uses"] & cs[iR]["uses"] or \
                            np.linalg.norm(cs[iL]["palm"] - cs[iR]["palm"]) < 0.03:
                        continue
                st.append((iL, iR))
                u = sum(M_MISS if i < 0 else assign_cost(cs[i], side) for i, side in zip((iL, iR), SIDES))
                # left/right swap guard: the wearer's left hand is rarely right of the right hand
                if iL >= 0 and iR >= 0 and cs[iL]["palm"][0] > cs[iR]["palm"][0] + CROSS_M:
                    u += CROSS_PEN
                un.append(u)
            states.append(st)
            unary.append(np.array(un))

        def trans(s):
            """(P, Q) transition costs between the state lists of frames s-1 and s."""
            T = 0.0
            for k in range(2):
                i = np.array([st[k] for st in states[s - 1]])
                j = np.array([st[k] for st in states[s]])
                pa = np.array([c["palm"] for c in cands[s - 1]] + [np.full(3, np.nan)])[i]  # -1 -> NaN row
                pb = np.array([c["palm"] for c in cands[s]] + [np.full(3, np.nan)])[j]
                d = np.linalg.norm(pa[:, None] - pb[None], axis=2)
                # quadratic up to 15 cm/frame (fast ego motion), then steeper: teleports are identity errors
                c = np.where(np.isfinite(d), W_T * np.where(d < 0.15, (d / 0.05) ** 2, 9.0 + 40.0 * (d - 0.15) / 0.05),
                             0.0)
                T = T + c + BIRTH * ((i < 0)[:, None] & (j >= 0)[None])
            return T

        acc, back = [unary[0]], []
        for s in range(1, n):
            tot = acc[-1][:, None] + trans(s)
            back.append(tot.argmin(0))
            acc.append(tot.min(0) + unary[s])
        q = int(acc[-1].argmin())
        path = [q]
        for s in range(n - 1, 0, -1):
            q = int(back[s - 1][q])
            path.append(q)
        path = path[::-1]
        return {side: [(cands[s][states[s][path[s]][k]] if states[s][path[s]][k] >= 0 else None) for s in range(n)]
                for k, side in enumerate(SIDES)}

    def to_arrays(track):
        """Stereo-time arrays: X (n,21,3) completed hands, conf (n,21); NaN/0 where no hand."""
        X, W = np.full((n, 21, 3), np.nan), np.zeros((n, 21))
        for s, cd in enumerate(track):
            if cd is None:
                continue
            X[s] = cd["X"]
            w = np.where(cd["ok"], np.clip(np.exp(-(np.nan_to_num(cd["rms"], nan=99) / 4.0) ** 2), 0.05, 1.0), 0.15)
            W[s] = w * (0.7 if cd["src"] != "cb" else 1.0)
        # temporal Hampel test per joint on measured joints (window +-2)
        for s in range(n):
            if not np.isfinite(X[s, 0, 0]):
                continue
            lo, hi = max(0, s - 2), min(n, s + 3)
            for j in range(21):
                nb = X[lo:hi, j][np.isfinite(X[lo:hi, j, 0]) & (W[lo:hi, j] > 0.15)]
                if len(nb) >= 3 and np.linalg.norm(X[s, j] - np.median(nb, 0)) > 0.05:
                    W[s, j] = min(W[s, j], 0.05)
        return X, W

    def resample(X, W, tau):
        """Colour-time hands: t <-> stereo time t + tau (linear interpolation)."""
        Xo, Wo = np.full((n, 21, 3), np.nan), np.zeros((n, 21))
        for t in range(n):
            st = t + tau
            s0 = int(np.floor(st))
            f = st - s0
            h0 = 0 <= s0 < n and np.isfinite(X[s0, 0, 0])
            h1 = 0 <= s0 + 1 < n and np.isfinite(X[s0 + 1, 0, 0])
            if h0 and h1:
                Xo[t] = (1 - f) * X[s0] + f * X[s0 + 1]
                Wo[t] = np.minimum(W[s0], W[s0 + 1])
            elif h0 and f <= 0.5:
                Xo[t], Wo[t] = X[s0], W[s0] * 0.7
            elif h1 and f >= 0.5:
                Xo[t], Wo[t] = X[s0 + 1], W[s0 + 1] * 0.7
        return Xo, Wo

    def px_to_a(Xt, valid, t):
        if not D["a"][t] or valid.sum() < 3:
            return None
        uv = project(Xt[valid], Ka)
        return float(min(np.linalg.norm(uv - d["j2d"][valid], axis=1).mean() for d in D["a"][t]))

    def estimate_tau(arrays):
        grid = np.round(np.arange(-args.max_offset, args.max_offset + 1e-9, 0.05), 3)
        scores = {}
        for tau in grid:
            d = []
            for side in SIDES:
                Xo, Wo = resample(*arrays[side], tau)
                for t in range(n):
                    if np.isfinite(Xo[t, 0, 0]):
                        v = px_to_a(Xo[t], Wo[t] > 0.3, t)
                        if v is not None and v < 100:
                            d.append(v)
            scores[float(tau)] = (float(np.median(d)) if d else np.inf, len(d))
        tau = min(scores, key=lambda k: scores[k][0] - 0.01 * scores[k][1])
        return tau, scores

    # pass 1: identity from stereo only -> sync; pass 2: + colour votes / a+c, a+b fallback
    tracks = viterbi(build_candidates(None))
    arrays = {s: to_arrays(tracks[s]) for s in SIDES}
    tau, scores = estimate_tau(arrays)
    print("pass 1: stereo hands L/R", [sum(c is not None for c in tracks[s]) for s in SIDES],
          "| sync tau", tau, {k: v for k, v in scores.items() if abs(k * 2 - round(k * 2)) < 1e-6}, flush=True)
    cands2 = build_candidates(tau)
    tracks = viterbi(cands2)
    arrays = {s: to_arrays(tracks[s]) for s in SIDES}
    tau, scores = estimate_tau(arrays)
    k_int = int(round(tau))
    print("pass 2: stereo hands L/R", [sum(c is not None for c in tracks[s]) for s in SIDES],
          "sources", {s: {src: sum(c is not None and c["src"] == src for c in tracks[s]) for src in ("cb", "ac", "ab")}
                      for s in SIDES},
          "| sync tau", tau, {k: v for k, v in scores.items() if abs(k * 2 - round(k * 2)) < 1e-6}, flush=True)

    # colour-time resampling, gap filling, zero-phase smoothing
    X, valid, Xs, conf, src_t = {}, {}, {}, {}, {}
    for side in SIDES:
        Xo, Wo = resample(*arrays[side], tau)
        src = ["" for _ in range(n)]
        for t in range(n):
            if np.isfinite(Xo[t, 0, 0]):
                s0 = min(max(int(round(t + tau)), 0), n - 1)
                cd = tracks[side][s0] or tracks[side][min(s0 + 1, n - 1)] or tracks[side][max(s0 - 1, 0)]
                src[t] = cd["src"] if cd else "cb"
        have = np.isfinite(Xo[:, 0, 0])
        idx = np.flatnonzero(have)
        for a0, a1 in zip(idx[:-1], idx[1:]):
            if 1 < a1 - a0 <= args.max_gap + 1:
                for t in range(a0 + 1, a1):
                    f = (t - a0) / (a1 - a0)
                    Xo[t] = (1 - f) * Xo[a0] + f * Xo[a1]
                    Wo[t] = 0.2 * np.minimum(Wo[a0], Wo[a1])
                    src[t] = "interp"
        have = np.isfinite(Xo[:, 0, 0])
        # confidence-weighted local quadratic regression, Gaussian window (sigma 1.5 frames, +-4)
        S = np.full_like(Xo, np.nan)
        for t in np.flatnonzero(have):
            ts = np.arange(max(0, t - 4), min(n, t + 5))
            ts = ts[have[ts]]
            g = np.exp(-0.5 * ((ts - t) / 1.5) ** 2)
            deg = 2 if len(ts) >= 5 else (1 if len(ts) >= 3 else 0)
            A = np.vander(ts - t, deg + 1)
            for j in range(21):
                w = g * np.maximum(Wo[ts, j], 1e-3)
                Aw = A * w[:, None]
                coef = np.linalg.lstsq(Aw.T @ A, Aw.T @ Xo[ts, j], rcond=None)[0]
                S[t, j] = coef[-1]
        X[side], conf[side], Xs[side], src_t[side] = Xo, Wo, S, src
        valid[side] = (Wo > 0.25) & have[:, None]

    def match_a(t, key):
        """Joint (both sides) assignment of this frame's hands to colour detections, by mean px
        distance over each hand's valid joints. Returns {side: (px, a_label_right) or None}; a hand
        whose best detection is taken by the other hand gets None (e.g. colour WiLoR missed it)."""
        arr = {"smooth": Xs, "raw": X}[key]
        hands = {}
        for side in SIDES:
            if np.isfinite(arr[side][t, 0, 0]):
                meas = valid[side][t] if valid[side][t].sum() >= 3 else conf[side][t] > 0.1
                hands[side] = (project(arr[side][t], Ka), meas)
        dets = D["a"][t]
        cost = {(s, i): float(np.linalg.norm(uv[m] - d["j2d"][m], axis=1).mean())
                for s, (uv, m) in hands.items() for i, d in enumerate(dets)}
        best, best_c = {}, np.inf
        opts = {s: [None] + list(range(len(dets))) for s in hands}
        for combo in product(*opts.values()):
            asg = dict(zip(opts, combo))
            ids = [i for i in asg.values() if i is not None]
            if len(ids) != len(set(ids)):
                continue
            c = sum(cost[(s, i)] if i is not None else 150.0 for s, i in asg.items())
            if c < best_c:
                best, best_c = asg, c
        return {s: ((cost[(s, i)], dets[i]["right"]) if i is not None else None) for s, i in best.items()}

    stats = {s: [] for s in SIDES}
    m_s = [match_a(t, "smooth") for t in range(n)]
    m_r = [match_a(t, "raw") for t in range(n)]
    for side in SIDES:
        for t in np.flatnonzero(np.isfinite(Xs[side][:, 0, 0])):
            ms, mr = m_s[t].get(side), m_r[t].get(side)
            cd = tracks[side][min(max(int(round(t + tau)), 0), n - 1)]
            stats[side].append({"frame": int(t), "px_to_a": ms[0] if ms else None, "px_to_a_raw": mr[0] if mr else None,
                                "a_label_agrees": (ms[1] == (side == "right")) if ms else None,
                                "a_used_in_triangulation": src_t[side][t] in ("ac", "ab"),
                                "rms_px": float(np.nanmean(cd["rms"][cd["ok"]])) if cd else None,
                                "n_valid": int(valid[side][t].sum()), "source": src_t[side][t],
                                "conf_mean": float(conf[side][t].mean())})
    tri = {"version": 2, "sync_offset": k_int, "sync_offset_frac": tau,
           "sync_note": "X_* are in colour time: colour frame t <-> stereo time t + sync_offset_frac",
           "offset_scores": {str(k): v for k, v in scores.items()},
           "params": vars(args),
           "stereo_debug": [{"s": s, "cands": [[c["src"], c["hand"], round(c["vote"], 2), round(c["q"], 2),
                                                 [round(float(x), 3) for x in c["palm"]]] for c in cands2[s]],
                             "chosen": {sd: next((i for i, c in enumerate(cands2[s]) if c is tracks[sd][s]), None)
                                        for sd in SIDES}} for s in range(n)]}
    for side in SIDES:
        fr = [st["frame"] for st in stats[side]]
        tri[side] = {"frames": fr, "stats": stats[side],
                     "X_smooth": {t: Xs[side][t].tolist() for t in fr},
                     "X_raw": {t: X[side][t].tolist() for t in fr},
                     "conf": {t: conf[side][t].round(3).tolist() for t in fr},
                     "valid": {t: valid[side][t].tolist() for t in fr}}
    (run / "stereo/hands_tri.json").write_text(json.dumps(tri))

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.median(vals)), 2) if vals else None
    summary = {}
    for side in SIDES:
        st = stats[side]
        ind = [x for x in st if x["px_to_a"] is not None and not x["a_used_in_triangulation"]]  # independent check
        summary[side] = {"frames": f"{len(st)}/{n}",
                         "measured": sum(x["source"] != "interp" for x in st),
                         "by_source": {k: sum(x["source"] == k for x in st) for k in ("cb", "ac", "ab", "interp")},
                         "matched_to_a_det": len(ind),
                         "median_px_to_a_smooth": med([x["px_to_a"] for x in ind]),
                         "median_px_to_a_raw": med([x["px_to_a_raw"] for x in ind]),
                         "median_px_to_a_smooth_lt40": med([x["px_to_a"] for x in ind if x["px_to_a"] < 40]),
                         "frac_px_to_a_lt40": round(float(np.mean([x["px_to_a"] < 40 for x in ind])), 3) if ind else None,
                         "a_label_agreement": round(float(np.mean([bool(x["a_label_agrees"]) for x in ind])), 3) if ind else None,
                         "median_rms_px": med([x["rms_px"] for x in st])}
    tri["summary"] = summary
    (run / "stereo/hands_tri.json").write_text(json.dumps(tri))
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)

    # ---- align each model's hands to the triangulated joints
    def model_joints(model, side, t):
        if model == "wilor":
            f = run / f"recon_hawor/wilor_raw/{t:06d}.json"
            da = [d for d in json.loads(f.read_text()) if d["is_right"] == (side == "right")] if f.exists() else []
            if not da:
                return None, None
            rec = max(da, key=lambda d: d.get("score", 0))
            _, j = mano.local(rec)
            return j.cpu().numpy(), rec  # local frame; placement is WiLoR's weak-perspective camera
        if model == "hawor":
            f = run / f"recon_hawor/hawor/{HAND_TRACK[side]}/{t:06d}.json"
            if not f.exists():
                return None, None
            rec = json.loads(f.read_text())
            _, j = mano.local(rec)
            return j.cpu().numpy() + np.array(rec["camera"]["cam_t"]), rec
        if model == "agile":
            f = run / f"agile/hands/{side}/{t:06d}.json"
            if not f.exists():
                return None, None
            h = json.loads(f.read_text())
            _, j = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
            return h["s_h"] * j.cpu().numpy() + np.array(h["T"]), h
        raise ValueError(model)

    report = {}
    for model in ("wilor", "hawor", "agile"):
        report[model] = {}
        out_root = run / f"stereo/aligned_{model}"
        for side in SIDES:
            out = out_root / side
            out.mkdir(parents=True, exist_ok=True)
            for f in out.glob("*.json"):  # stale frames from previous runs
                f.unlink()
            pairs = []
            for t in tri[side]["frames"]:
                if valid[side][t].sum() < 4:
                    continue
                J, rec = model_joints(model, side, t)
                if J is not None:
                    pairs.append((t, J, rec))
            if len(pairs) < 5:
                continue
            scales, place = [], []
            for t, J, _ in pairs:
                ok = valid[side][t]
                s, _, _ = umeyama(J[ok], Xs[side][t][ok])
                scales.append(s)
                if model != "wilor":  # wilor has no metric placement in our camera
                    place.append(float(np.linalg.norm(J[ok] - Xs[side][t][ok], axis=1).mean() * 1000))
            s_seq = float(np.median(scales))
            resid = []
            for t, J, rec in pairs:
                _, R, tr, aligned, res = align_hand_to_triangulation(s_seq * J, Xs[side][t], valid[side][t],
                                                                     with_scale=False)
                resid.append(float(np.nanmean(res)))
                (out / f"{t:06d}.json").write_text(json.dumps({"scale": s_seq, "R": R.tolist(), "t": tr.tolist(),
                                                               "joints": aligned.tolist(), "resid_mm": resid[-1]}))
            report[model][side] = {"frames": len(pairs), "seq_scale": s_seq,
                                   "placement_err_mm_median": float(np.median(place)) if place else None,
                                   "articulation_resid_mm_median": float(np.median(resid))}
    (run / "stereo/hand_alignment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.no_viz:
        return

    # ---- visualisation over the colour video
    def draw(img, uv, col, th):
        for a, b in BONES:
            if np.isfinite(uv[[a, b]]).all():
                cv2.line(img, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]), col, th, cv2.LINE_AA)

    best_model = min((m for m in report if report[m]),
                     key=lambda m: np.mean([v["articulation_resid_mm_median"] for v in report[m].values()]))
    viz = Path(os.environ["V2D_VIZ"]) / "07_stereo_hands"
    viz.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(viz / f"{args.run}.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=2)
    for t in range(n):
        img = cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png"))
        for d in D["a"][t]:
            for p in d["j2d"]:
                if np.isfinite(p).all():
                    cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
        for side in SIDES:
            if np.isfinite(Xs[side][t, 0, 0]):
                draw(img, project(Xs[side][t], Ka), (255, 255, 0) if src_t[side][t] == "interp" else (0, 255, 0), 4)
            af = run / f"stereo/aligned_{best_model}/{side}/{t:06d}.json"
            if af.exists():
                draw(img, project(np.array(json.loads(af.read_text())["joints"]), Ka), (255, 0, 255), 2)
        img = cv2.resize(img, (1014, 760))
        cv2.putText(img, f"{args.run} f{t} tau={tau:+.2f}  green: stereo  cyan: gap-filled  magenta: {best_model}"
                    f"  yellow: colour WiLoR", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        wr.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    wr.close()


if __name__ == "__main__":
    main()
