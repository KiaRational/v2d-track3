"""Stage 6d: identity-stable wearer-hand segmentation/tracking with SAM 3 (runs in the `sam3` image).

Problem this solves. WiLoR is run on whole frames and decides handedness per detection; at the
image border it flips, and people in the background contribute hands that look perfectly valid
to the triangulator. Stage 6b then has to guess identity from votes. Here identity is decided
once, geometrically, with exactly two persistent tracks (the wearer's left / right hand), and
handed to 6a-v3 / 6b-v3 as a hard label.

Per episode, for the colour camera 'a' (<run>/recon_hawor/frames) and both undistorted stereo
cameras 'c'/'b' (<run>/stereo/cam_{c,b}.mp4, Stage 6a):

 1. Detection  SAM 3 promptable concept segmentation ("hand", ...) on every frame of every view
    (Sam3Model). This returns *all* hands, the wearer's and other people's.
 2. Wearer gate  each instance gets a metric distance:
       view a  median of the metric stereo depth (<run>/depth_stereo, cam-a pixels) over the
               eroded mask -> a 3D palm point in the colour-camera frame;
       c/b     instances of the two stereo views are paired (near-horizontal epipolar offset,
               positive disparity, compatible area) and the pair's centroid is triangulated
               (baseline 7.5 cm) -> 3D palm in cam-c, mapped to cam-a by T_a_to_c^-1;
       any     a monocular fallback, apparent mask area vs a hand's physical area, used only to
               reject (not to accept) when no depth/disparity is available.
    Instances beyond --max_reach (0.9 m, arm's reach) or without any usable distance are dropped:
    those are the background people. Masks touching the image border keep their (unreliable)
    area estimate but may still be claimed by a track through 3D prediction.
 3. Two persistent tracks with memory  per-view 3D estimates of the same frame are fused
    (clustered within --fuse_m), lifted to the *world* frame with the SLAM pose
    (<run>/recon_hawor/slam_poses), and assigned to two tracks by Hungarian matching against
    each track's predicted world position (last position + decayed velocity). A track that
    loses its hand is *not* killed: it keeps its last world position, velocity, appearance and
    handedness, and its gate grows with the gap, so when the hand re-enters the frame it is
    re-associated instead of a new track being born or another person's hand being stolen;
    after --stale frames the position memory expires and the track re-acquires on the side
    prior, so one bad stretch cannot lock the real hand out for the rest of the episode.
    The two tracks are the wearer's *left* and *right* hand from the first frame on: the
    association carries an egocentric side prior (the left hand is the one on the -x side of
    the head camera), without which the two tracks are interchangeable and a track seeded on
    the wrong hand stays wrong. Forward and backward passes are merged; every re-association
    after a gap >= 2 frames is reported. The signed-x and Stage-5-mask (ids 2/3) votes are
    then only a consistency check on that naming.
 4. Gap filling with the SAM 3 video tracker  per view and track, the accepted masks seed a
    Sam3TrackerVideoModel session (same one-session-per-object pattern as
    stages/objects_v2/seg_sam3.py); propagated masks are used only on frames the detector
    missed, and only if they pass the same distance gate and land near the predicted position.

Outputs under <run>/stereo/hands_sam3/:
  {a,c,b}/NNNNNN.json      [{track, bbox[x0,y0,x1,y1], mask_path, score, depth_m, X_cam_a,
                             source: det|fill, border, area_px}]
  masks/<view>/<track>/NNNNNN.png   binary mask, view resolution
  summary.json             per-track coverage, re-association events, rejected instances

Usage (GPU job): python s06d_sam3_hands.py <run_name> [...] [--views acb] [--no_fill]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(os.environ["V2D_ROOT"])
TRACKS = ("left", "right")
HAND_AREA_M2 = 0.014          # physical area of a hand silhouette (palm + fingers), m^2
MASK_IDS = {"left": 2, "right": 3}


# --------------------------------------------------------------------------- io helpers
def decode_depth(path):
    """depth_m = 65535/png - 1; png 0/65535 invalid."""
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    png = png.astype(np.float32)
    d = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, 0.0)
    return np.where(np.isfinite(d) & (d > 0.05) & (d < 5.0), d, 0.0).astype(np.float32)


def load_T(path):
    """Toolkit Transform3d json (w-first quaternion) -> 4x4."""
    d = json.loads(Path(path).read_text())
    w, x, y, z = d["rotation"]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    T = np.eye(4)
    T[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, 3] = d["translation"]
    return T


def read_video(path, n=None):
    cap = cv2.VideoCapture(str(path))
    out = []
    while n is None or len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def apply_T(T, X):
    return np.asarray(X) @ T[:3, :3].T + T[:3, 3]


def mask_bbox(m):
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


# --------------------------------------------------------------------------- SAM 3
class Sam3Hands:
    def __init__(self, weights, device="cuda", thr=0.35):
        from transformers import (Sam3Model, Sam3Processor, Sam3TrackerVideoModel,
                                  Sam3TrackerVideoProcessor)
        wd = str(weights)
        if not Path(wd, "config.json").exists():
            raise FileNotFoundError(f"SAM3 weights not found in {wd}")
        self.device, self.thr = device, thr
        self.model = Sam3Model.from_pretrained(wd, local_files_only=True).to(device).eval()
        self.proc = Sam3Processor.from_pretrained(wd, local_files_only=True)
        self._tm = self._tp = None
        self._wd = wd

    def tracker(self):
        if self._tm is None:
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
            self._tm = Sam3TrackerVideoModel.from_pretrained(
                self._wd, local_files_only=True).to(self.device, dtype=torch.bfloat16).eval()
            self._tp = Sam3TrackerVideoProcessor.from_pretrained(self._wd, local_files_only=True)
        return self._tm, self._tp

    @torch.inference_mode()
    def detect(self, rgb, texts, max_inst=8):
        """-> [(score, mask bool HxW)] deduplicated across prompts."""
        img = Image.fromarray(rgb)
        found = []
        for text in texts:
            inp = self.proc(images=img, text=text, return_tensors="pt").to(self.device)
            res = self.proc.post_process_instance_segmentation(
                self.model(**inp), threshold=self.thr, mask_threshold=0.5,
                target_sizes=inp.get("original_sizes").tolist())[0]
            for m, s in zip(res["masks"], res["scores"]):
                mm = m.cpu().numpy().astype(bool)
                if mm.sum() < 60:
                    continue
                found.append((float(s), mm))
        found.sort(key=lambda x: -x[0])
        keep = []
        for s, m in found:
            if all(iou(m, k[1]) < 0.6 for k in keep):
                keep.append((s, m))
            if len(keep) >= max_inst:
                break
        return keep

    @torch.inference_mode()
    def propagate(self, video, prompts, device="cuda"):
        """prompts [(t, mask bool)] for ONE object -> {t: mask bool}; forward+backward."""
        tm, tp = self.tracker()
        sess = tp.init_video_session(video=video, inference_device=device,
                                     video_storage_device="cpu", dtype=torch.bfloat16)
        H, W = video[0].shape[:2]
        for t, m in sorted(prompts, key=lambda x: x[0]):
            tp.add_inputs_to_inference_session(inference_session=sess, frame_idx=t, obj_ids=1,
                                               input_masks=m.astype(np.uint8))
            tm(inference_session=sess, frame_idx=t)
        first = min(t for t, _ in prompts)
        last = max(t for t, _ in prompts)
        res = {}
        for reverse in (True, False):
            for out in tm.propagate_in_video_iterator(inference_session=sess,
                                                      start_frame_idx=last if reverse else first,
                                                      reverse=reverse):
                t = out.frame_idx
                if reverse or t >= first or t not in res:
                    mk = tp.post_process_masks([out.pred_masks], original_sizes=[[H, W]],
                                               binarize=True)[0]
                    res[t] = mk[0, 0].cpu().numpy().astype(bool)
        del sess
        torch.cuda.empty_cache()
        return res


# --------------------------------------------------------------------------- geometry
def backproject_mask(m, depth, K, erode=5):
    """Median 3D point of the mask's depth samples, and the depth sample count."""
    core = m
    if erode and m.sum() > 400:
        e = cv2.erode(m.astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0
        if e.sum() > 60:
            core = e
    v, u = np.nonzero(core & (depth > 0))
    if len(u) < 15:
        v, u = np.nonzero(m & (depth > 0))     # small / thin mask: do not erode
    if len(u) < 15:
        return None, 0
    z = depth[v, u]
    lo, hi = np.percentile(z, [10, 90])
    sel = (z >= lo) & (z <= hi)
    u, v, z = u[sel], v[sel], z[sel]
    if len(u) < 10:
        return None, 0
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return np.median(P, 0), int(len(u))


def mono_depth_estimate(area_px, K):
    """Rough distance from apparent hand area: area_px ~ f^2 * A / z^2."""
    if area_px < 40:
        return np.inf
    return float(np.sqrt(K[0, 0] * K[1, 1] * HAND_AREA_M2 / area_px))


def touches_border(m, pad=2):
    return bool(m[:pad].any() or m[-pad:].any() or m[:, :pad].any() or m[:, -pad:].any())


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def pair_stereo(inst_c, inst_b, Kc, Kb, base, max_dv=45.0, min_d=8.0, max_d=400.0):
    """Match c<->b instances of one stereo frame; returns [(i, j, z_c, cost)] (greedy, 1:1)."""
    out = []
    for i, ic in enumerate(inst_c):
        for j, ib in enumerate(inst_b):
            pc, pb = ic["uv"], ib["uv"]
            dv = abs(pc[1] - pb[1])
            d = pc[0] - pb[0]                      # c is the left camera -> positive disparity
            if dv > max_dv or not (min_d <= d <= max_d):
                continue
            ar = ic["area"] / max(ib["area"], 1.0)
            if not 0.45 <= ar <= 2.2:
                continue
            z = Kc[0, 0] * base / d
            out.append((i, j, float(z), float(dv / max_dv + abs(np.log(ar)))))
    out.sort(key=lambda r: r[3])
    used_i, used_j, keep = set(), set(), []
    for i, j, z, c in out:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        keep.append((i, j, z, c))
    return keep


# --------------------------------------------------------------------------- tracking
def fuse_observations(per_view, fuse_m):
    """per_view: {view: [inst...]} of ONE frame, each with 'X' (cam-a frame) or None.
    Clusters the 3D estimates into observations; instances without 3D stay separate (X=None)."""
    items = [(v, k, d) for v, L in per_view.items() for k, d in enumerate(L) if d["X"] is not None]
    obs, used = [], set()
    order = sorted(range(len(items)), key=lambda i: -items[i][2]["score"])
    for a in order:
        if a in used:
            continue
        grp = [a]
        used.add(a)
        # at most ONE instance per other view, and the *nearest* one: taking every instance
        # within fuse_m merges the two hands into a single observation when they come close
        for view in {items[b][0] for b in order if b not in used}:
            cand = [b for b in order if b not in used and items[b][0] == view
                    and np.linalg.norm(items[a][2]["X"] - items[b][2]["X"]) < fuse_m]
            if cand:
                b = min(cand, key=lambda i: np.linalg.norm(items[a][2]["X"] - items[i][2]["X"]))
                grp.append(b)
                used.add(b)
        members = {items[g][0]: items[g][1] for g in grp}
        # the monocular area estimate is only a fallback: never let it pollute a
        # depth/stereo measurement of the same hand
        good = [g for g in grp if items[g][2]["how"] in ("depth", "stereo")] or grp
        X = np.mean([items[g][2]["X"] for g in good], 0)
        obs.append({"X": X, "members": members, "X_cam": X,
                    "score": float(np.mean([items[g][2]["score"] for g in grp])),
                    "app": np.mean([items[g][2]["app"] for g in grp], 0),
                    "n_views": len(grp)})
    return obs


SIDE_SGN = {"left": -1.0, "right": 1.0}


def track_pass(frames_obs, Tw, order, gate0, gate_grow, gate_max, w_app, w_side=0.08, stale=10):
    """Two persistent tracks -- the wearer's left and right hand -- with world-frame memory.

    frames_obs[k] = fused observations of frame k (3D in the colour-camera frame). Two things
    keep the identity stable across a hand leaving the view:
      * memory. A track that loses its hand keeps its last world position, velocity and
        appearance; the association gate grows with the gap instead of the track dying, so the
        hand is re-associated on re-entry rather than a new identity being created.
      * an egocentric side prior. From the very first frame the wearer's left hand is the one
        on the -x side of the head camera. Without it the two tracks are interchangeable, and a
        track seeded on the wrong hand (or on a background person before the reach gate has
        been informative) stays wrong for the rest of the episode and locks the real hand out.
    A track that has been missing longer than `stale` frames drops its position memory
    altogether and re-acquires on the side prior alone, so one bad stretch cannot kill it.

    Returns assign[k] = {track: obs_idx} and the re-association events."""
    assign = [dict() for _ in frames_obs]
    st = {tr: None for tr in TRACKS}      # {Xw, v, last_k, app}
    events = []

    def side_cost(tr, o):
        return -w_side * float(np.clip(SIDE_SGN[tr] * o["X_cam"][0] / 0.12, -1, 1))

    for k in order:
        obs = frames_obs[k]
        if not obs:
            continue
        Tk = Tw.get(k)
        Xw = [apply_T(Tk, o["X"]) if Tk is not None else o["X"] for o in obs]
        C = np.full((len(TRACKS), len(obs)), np.inf)
        for a, tr in enumerate(TRACKS):
            s = st[tr]
            dk = abs(k - s["last_k"]) if s else None
            expired = s is None or dk > stale
            for b in range(len(obs)):
                if expired:                      # position memory gone: the side prior decides
                    C[a, b] = 0.20 + side_cost(tr, obs[b]) - 0.02 * obs[b]["n_views"]
                    continue
                pred = s["Xw"] + s["v"] * np.clip(k - s["last_k"], -3, 3)
                d = float(np.linalg.norm(Xw[b] - pred))
                if d > min(gate0 + gate_grow * dk, gate_max):
                    continue
                C[a, b] = (d + side_cost(tr, obs[b])
                           + w_app * float(np.linalg.norm(obs[b]["app"] - s["app"])) / 255.0)
        fin = np.isfinite(C)
        if not fin.any():
            continue
        ri, ci = linear_sum_assignment(np.where(fin, C, 1e6))
        for a, b in zip(ri, ci):
            if not fin[a, b]:
                continue
            tr = st_tr = TRACKS[a]
            s = st[tr]
            if s is not None and abs(k - s["last_k"]) >= 2:
                events.append({"track": tr, "frame": int(k), "gap": int(abs(k - s["last_k"])),
                               "dist_to_prediction_m": round(float(np.linalg.norm(
                                   Xw[b] - s["Xw"])), 4),
                               "expired": bool(abs(k - s["last_k"]) > stale)})
            if s is None:
                st[st_tr] = {"Xw": Xw[b], "v": np.zeros(3), "last_k": k, "app": obs[b]["app"].copy()}
            else:
                v = (Xw[b] - s["Xw"]) / max(k - s["last_k"], 1) if k > s["last_k"] else s["v"]
                st[st_tr] = {"Xw": Xw[b], "v": 0.6 * v + 0.4 * s["v"], "last_k": k,
                             "app": 0.8 * s["app"] + 0.2 * obs[b]["app"]}
            assign[k][tr] = b
    return assign, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", default=str(ROOT / "weights/sam3"))
    ap.add_argument("--views", default="acb")
    ap.add_argument("--prompts", default="hand,human hand")
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--max_reach", type=float, default=0.9, help="arm's reach gate (m)")
    ap.add_argument("--mono_reach", type=float, default=1.5,
                    help="reject when only the monocular area estimate is available and it exceeds this (m)")
    ap.add_argument("--scale_a", type=float, default=0.5, help="processing scale of the colour view")
    ap.add_argument("--fuse_m", type=float, default=0.10)
    ap.add_argument("--gate0", type=float, default=0.13)
    ap.add_argument("--gate_grow", type=float, default=0.05)
    ap.add_argument("--gate_max", type=float, default=0.55)
    ap.add_argument("--w_app", type=float, default=0.02)
    ap.add_argument("--w_side", type=float, default=0.08, help="egocentric left/right prior weight (m-equivalent)")
    ap.add_argument("--stale", type=int, default=10, help="frames after which a track drops its position memory")
    ap.add_argument("--fill_gap", type=int, default=8, help="max detector gap filled by the SAM3 tracker")
    ap.add_argument("--no_fill", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    calib = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
    T_a_to_c = np.array(calib["T_ego_cam_a_to_ego_cam_c"])
    T_c_to_a = np.linalg.inv(T_a_to_c)
    base = float(abs(np.array(calib["stereo"]["T_left_to_right"])[0, 3]))
    det = Sam3Hands(args.weights, thr=args.score)
    texts = [t.strip() for t in args.prompts.split(",") if t.strip()]

    for name in args.runs:
        run = ROOT / "runs" / name
        out = run / "stereo/hands_sam3"
        if out.exists() and (out / "summary.json").exists() and not args.overwrite:
            print(f"{name}: hands_sam3 exists, skipping (use --overwrite)", flush=True)
            continue
        frames_dir = run / "recon_hawor/frames"
        n = len(list(frames_dir.glob("*.png")))
        ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        Ka_full = np.array([[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.0]])
        sint = json.loads((run / "stereo/intrinsics.json").read_text())
        K = {"c": np.array(sint["ego_cam_c"]["K"]), "b": np.array(sint["ego_cam_b"]["K"])}
        sa = args.scale_a
        Ka = Ka_full.copy()
        Ka[:2] *= sa
        K["a"] = Ka
        # image -> cam-a-frame transforms for the 3D estimate of each view
        to_a = {"a": np.eye(4), "c": T_c_to_a, "b": T_c_to_a @ np.linalg.inv(np.array(calib["stereo"]["T_left_to_right"]))}

        # ---- frames
        video = {}
        if "a" in args.views:
            video["a"] = []
            for t in range(n):
                im = cv2.cvtColor(cv2.imread(str(frames_dir / f"{t:06d}.png")), cv2.COLOR_BGR2RGB)
                video["a"].append(cv2.resize(im, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
                                  if sa != 1.0 else im)
        for v in "cb":
            if v in args.views:
                video[v] = read_video(run / f"stereo/cam_{v}.mp4", n)
        views = [v for v in "acb" if v in video and len(video[v])]
        nf = {v: len(video[v]) for v in views}
        print(f"{name}: n={n} frames per view {nf}", flush=True)

        # ---- 1. detection
        inst = {v: [[] for _ in range(n)] for v in views}
        for v in views:
            for t in range(min(n, nf[v])):
                for s, m in det.detect(video[v][t], texts):
                    bb = mask_bbox(m)
                    inst[v][t].append({"score": s, "m": m, "uv": centroid(m), "area": float(m.sum()),
                                       "bbox": bb, "border": touches_border(m),
                                       "app": np.array([video[v][t][m].mean(0)]).reshape(-1)[:3]
                                       if video[v][t][m].size else np.zeros(3),
                                       "X": None, "z": None, "how": None})
            print(f"  {v}: {sum(len(x) for x in inst[v])} raw instances "
                  f"({sum(bool(x) for x in inst[v])}/{min(n, nf[v])} frames)", flush=True)

        # ---- 2. distance / wearer gate
        n_far, n_nodepth = 0, 0
        if "a" in views:
            for t in range(min(n, nf["a"])):
                dp = decode_depth(run / f"depth_stereo/{t:06d}.png")
                if dp is None:
                    continue
                if sa != 1.0:
                    dp = cv2.resize(dp, (video["a"][t].shape[1], video["a"][t].shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                for d in inst["a"][t]:
                    X, ns = backproject_mask(d["m"], dp, Ka)
                    if X is not None:
                        d["X"], d["z"], d["how"] = X, float(X[2]), "depth"
        if "c" in views and "b" in views:
            for t in range(min(nf["c"], nf["b"])):
                for i, j, z, _ in pair_stereo(inst["c"][t], inst["b"][t], K["c"], K["b"], base):
                    for v, k in (("c", i), ("b", j)):
                        d = inst[v][t][k]
                        uv = d["uv"]
                        Xv = np.array([(uv[0] - K[v][0, 2]) / K[v][0, 0], (uv[1] - K[v][1, 2]) / K[v][1, 1], 1.0]) * z
                        d["X"], d["z"], d["how"] = apply_T(to_a[v], Xv), float(z), "stereo"
        drops = {v: {"far": 0, "nodist": 0} for v in views}
        for v in views:
            for t in range(n):
                kept = []
                for d in inst[v][t]:
                    zm = mono_depth_estimate(d["area"], K[v])
                    d["z_mono"] = None if not np.isfinite(zm) else round(float(zm), 3)
                    if d["X"] is not None:
                        if float(np.linalg.norm(d["X"])) > args.max_reach:
                            n_far += 1
                            drops[v]["far"] += 1
                            continue
                    else:
                        # No depth / no stereo partner. The apparent-size estimate is the only
                        # distance left, and a mask cut off by the image border (exactly the
                        # near-hand case) under-reports its area, so it gets a laxer gate; the
                        # track's own 3D prediction still has to claim it.
                        lim = args.mono_reach * (2.0 if d["border"] else 1.0)
                        if not np.isfinite(zm) or zm > lim:
                            n_nodepth += 1
                            drops[v]["nodist"] += 1
                            continue
                        uv = d["uv"]
                        Xv = np.array([(uv[0] - K[v][0, 2]) / K[v][0, 0],
                                       (uv[1] - K[v][1, 2]) / K[v][1, 1], 1.0]) * zm
                        d["X"], d["z"], d["how"] = apply_T(to_a[v], Xv), float(zm), "mono"
                    kept.append(d)
                inst[v][t] = kept
        print(f"  gated: {n_far} beyond {args.max_reach} m, {n_nodepth} without usable distance; per view {drops}", flush=True)

        # ---- 3. two persistent tracks with world-frame memory
        Tw = {}
        for t in range(n):
            p = run / f"recon_hawor/slam_poses/{t:06d}.json"
            if p.exists():
                try:
                    Tw[t] = load_T(p)
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        frames_obs = [fuse_observations({v: inst[v][t] for v in views}, args.fuse_m) for t in range(n)]
        fw, ev_f = track_pass(frames_obs, Tw, list(range(n)), args.gate0, args.gate_grow,
                              args.gate_max, args.w_app, args.w_side, args.stale)
        bw, ev_b = track_pass(frames_obs, Tw, list(range(n - 1, -1, -1)), args.gate0, args.gate_grow,
                              args.gate_max, args.w_app, args.w_side, args.stale)
        # merge: forward wins; backward fills frames the forward pass left empty for a track
        assign = [dict(a) for a in fw]
        n_bw = 0
        for t in range(n):
            for tr in TRACKS:
                if tr in assign[t] or tr not in bw[t]:
                    continue
                if bw[t][tr] in assign[t].values():
                    continue
                assign[t][tr] = bw[t][tr]
                n_bw += 1
        events = sorted(ev_f + [dict(e, pass_="backward") for e in ev_b], key=lambda e: e["frame"])

        # ---- left/right naming (global): signed x of the palm in the camera frame + Stage-5 masks
        vote = {tr: 0.0 for tr in TRACKS}
        for t in range(n):
            for tr, b in assign[t].items():
                vote[tr] += float(np.sign(frames_obs[t][b]["X"][0]))
        mdirs = [run / d / "masks" for d in ("objects_sam3", "objects") if (run / d / "masks").is_dir()]
        m_vote = {tr: 0.0 for tr in TRACKS}
        if mdirs and "a" in views:
            for t in range(0, n, max(1, n // 60)):
                ref = {}
                for side, oid in MASK_IDS.items():
                    p = mdirs[0] / str(oid) / f"{t:06d}.png"
                    mm = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
                    if mm is not None and mm.any():
                        ref[side] = cv2.resize(mm, (video["a"][t].shape[1], video["a"][t].shape[0]),
                                               interpolation=cv2.INTER_NEAREST) > 0
                for tr, b in assign[t].items():
                    ai = frames_obs[t][b]["members"].get("a")
                    if ai is None:
                        continue
                    m = inst["a"][t][ai]["m"]
                    for side, rm in ref.items():
                        m_vote[tr] += iou(m, rm) * (1.0 if side == "right" else -1.0)
        # The tracks are already semantic (the side prior drives the association), so the votes
        # are a check, not the decision. Only an unambiguous contradiction from *both* the
        # geometry and the Stage-5 hand masks flips them.
        rename = {tr: tr for tr in TRACKS}
        nz = [tr for tr in TRACKS if abs(vote[tr]) > 0]
        swap = (len(nz) == 2 and vote["left"] > 0 > vote["right"]
                and m_vote["left"] > 0.5 > -0.5 > m_vote["right"])
        if swap:
            rename = {"left": "right", "right": "left"}
        print(f"  naming: {rename} (swap={swap}) from x-votes {vote} mask-votes "
              f"{ {k: round(v, 2) for k, v in m_vote.items()} }", flush=True)

        # ---- per-view, per-track accepted masks
        acc = {v: {tr: {} for tr in TRACKS} for v in views}      # view -> name -> {t: inst dict}
        for t in range(n):
            for tr, b in assign[t].items():
                nm = rename[tr]
                for v, k in frames_obs[t][b]["members"].items():
                    acc[v][nm][t] = dict(inst[v][t][k], source="det")

        # ---- 4. gap filling with the SAM 3 video tracker
        n_fill = 0
        if not args.no_fill:
            for v in views:
                for nm in TRACKS:
                    have = sorted(acc[v][nm])
                    if len(have) < 3:
                        continue
                    gaps = [(a, b) for a, b in zip(have[:-1], have[1:]) if 1 < b - a <= args.fill_gap + 1]
                    if not gaps:
                        continue
                    step = max(1, len(have) // 24)
                    prompts = [(t, acc[v][nm][t]["m"]) for t in have[::step]]
                    for a, b in gaps:                    # anchor each gap on both sides
                        for t in (a, b):
                            if all(p[0] != t for p in prompts):
                                prompts.append((t, acc[v][nm][t]["m"]))
                    try:
                        prop = det.propagate(video[v], prompts)
                    except Exception as e:               # noqa: BLE001 - tracker OOM etc.
                        print(f"  fill {v}/{nm}: tracker failed ({e})", flush=True)
                        continue
                    other = acc[v]["right" if nm == "left" else "left"]
                    for a, b in gaps:
                        for t in range(a + 1, b):
                            m = prop.get(t)
                            if m is None or m.sum() < 80:
                                continue
                            if t in other and iou(m, other[t]["m"]) > 0.3:
                                continue
                            zm = mono_depth_estimate(float(m.sum()), K[v])
                            if not np.isfinite(zm) or zm > args.mono_reach:
                                continue
                            uv = centroid(m)
                            # must land near the interpolated position of the accepted neighbours
                            f = (t - a) / (b - a)
                            pu = (1 - f) * acc[v][nm][a]["uv"] + f * acc[v][nm][b]["uv"]
                            if np.linalg.norm(uv - pu) > 0.35 * max(K[v][0, 0], 200):
                                continue
                            acc[v][nm][t] = {"score": 0.5, "m": m, "uv": uv, "area": float(m.sum()),
                                             "bbox": mask_bbox(m), "border": touches_border(m),
                                             "X": None, "z": float(zm), "how": "mono",
                                             "z_mono": round(float(zm), 3), "source": "fill"}
                            n_fill += 1

        # ---- write
        if out.exists() and args.overwrite:
            import shutil
            shutil.rmtree(out)
        for v in views:
            (out / v).mkdir(parents=True, exist_ok=True)
            for nm in TRACKS:
                (out / "masks" / v / nm).mkdir(parents=True, exist_ok=True)
        full = {"a": (ia["width"], ia["height"]), "c": tuple(sint["ego_cam_c"]["size"]),
                "b": tuple(sint["ego_cam_b"]["size"])}
        cov = {v: {nm: 0 for nm in TRACKS} for v in views}
        for v in views:
            for t in range(n):
                recs = []
                for nm in TRACKS:
                    d = acc[v][nm].get(t)
                    if d is None:
                        continue
                    m = d["m"]
                    scl = full[v][0] / m.shape[1]
                    if abs(scl - 1.0) > 1e-6:
                        m = cv2.resize(m.astype(np.uint8), full[v], interpolation=cv2.INTER_NEAREST) > 0
                    mp = out / "masks" / v / nm / f"{t:06d}.png"
                    cv2.imwrite(str(mp), (m.astype(np.uint8) * 255))
                    recs.append({"track": nm, "bbox": [round(x * scl, 2) for x in d["bbox"]],
                                 "mask_path": str(mp.relative_to(run)), "score": round(float(d["score"]), 4),
                                 "depth_m": None if d["z"] is None else round(float(d["z"]), 4),
                                 "depth_source": d["how"], "z_mono_m": d["z_mono"],
                                 "X_cam_a": None if d["X"] is None else [round(float(x), 4) for x in d["X"]],
                                 "source": d["source"], "border": bool(d["border"]),
                                 "area_px": int(d["area"] * scl * scl)})
                    cov[v][nm] += 1
                (out / v / f"{t:06d}.json").write_text(json.dumps(recs))
        summary = {"run": name, "n_frames": n, "views": views, "params": vars(args),
                   "coverage": cov, "coverage_frac": {v: {nm: round(cov[v][nm] / max(n, 1), 3) for nm in TRACKS}
                                                      for v in views},
                   "rejected_far": n_far, "rejected_no_distance": n_nodepth, "rejected_per_view": drops,
                   "filled_by_tracker": n_fill, "backward_pass_added": n_bw,
                   "track_naming": rename, "naming_votes": {"x": vote, "mask": m_vote},
                   "reassociation_events": events,
                   "n_reassociation": len(events)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"{name}: coverage {summary['coverage_frac']} | reassoc {len(events)} | "
              f"far-rejected {n_far} | no-distance {n_nodepth} | tracker-filled {n_fill}", flush=True)


if __name__ == "__main__":
    main()
