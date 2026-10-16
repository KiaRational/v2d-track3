"""Stage 6c: temporally smooth, metric MANO hand trajectory fitted to the
stereo-triangulated 3D joints of Stage 6b (runs in the `agile` image).

Per hand side, one optimisation over the whole sequence (torch, Adam then LBFGS):
  parameters  shared betas (10); per frame a rotation delta on the initial global
              orientation (3), hand_pose (45), translation T (3). MANO at metric scale,
              no extra scale factor (hand size is absorbed by the shared betas).
  losses      robust (GMoF) 3D joint loss to hands_tri.json X_smooth (finite joints);
              robust 2D loss of the projected joints to the colour-camera WiLoR
              keypoints (low weight where triangulated, higher where not);
              prior to WiLoR's per-frame hand_pose, L2 on betas;
              smoothness: joint acceleration, hand_pose and rotation velocity.
  init        colour WiLoR record of the side (nearest record for frames without
              one), rotated/translated by a per-frame rigid alignment to the
              triangulated joints; frames without triangulation take the nearest
              triangulated frame's rotation correction and a PnP translation.
Conventions match agile/geometry.py Mano.local: joints = Mano.local(rec) + T
(left hands mirrored in x), metres, colour camera 'a' frame.
Outputs: <run>/stereo/mano_fit/<side>/NNNNNN.json, <run>/stereo/mano_fit/report.json,
video $V2D_VIZ/07_stereo_hands/<run>_manofit.mp4.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

sys.path.insert(0, "/code/agile")
sys.path.insert(0, "/code/stereo")
from geometry import Mano, pnp_translation, wilor_keypoints_2d  # noqa: E402
from hand_triangulation import umeyama  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
SIDES = ("left", "right")
DEV = "cuda"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FLIP = np.diag([-1.0, 1.0, 1.0])


def gmof(sq, sigma):
    """Geman-McClure on squared residuals: sigma^2 * e^2 / (sigma^2 + e^2)."""
    return sigma ** 2 * sq / (sigma ** 2 + sq)


def project_np(X, K):
    return X[:, :2] / X[:, 2:3] @ K[:2, :2].T + K[:2, 2]


def jitter_mm(seq: dict) -> float | None:
    """Mean joint acceleration magnitude (mm/frame^2) over consecutive frame triples."""
    acc = []
    for t in seq:
        if t - 1 in seq and t + 1 in seq:
            a = seq[t + 1] - 2 * seq[t] + seq[t - 1]
            n = np.linalg.norm(a, axis=1)
            n = n[np.isfinite(n)]
            if len(n):
                acc.append(n.mean() * 1000)
    return float(np.mean(acc)) if acc else None


class HandModel:
    """Differentiable MANO for a batch of frames with our own global rotation.

    MANO applies the global orientation about the rest root joint J0, so
    J(go) = R (J(0) - J0) + J0; we evaluate the layer with go = 0 and apply R
    (right-hand parameter space) ourselves, then mirror x for left hands."""

    def __init__(self, mano: Mano, is_right: bool):
        self.layer = mano.layer
        self.flip = torch.tensor([1.0 if is_right else -1.0, 1.0, 1.0], device=DEV)

    def __call__(self, R, hand_pose, betas, T, verts=False):
        n = hand_pose.shape[0]
        pose = torch.cat([torch.zeros(n, 3, device=DEV), hand_pose], 1)
        out = self.layer(pose, betas.expand(n, -1))
        J0 = out.joints[:, :1]
        j = torch.einsum("nij,nkj->nki", R, out.joints - J0) + J0
        j = j * self.flip + T[:, None]
        if not verts:
            return j
        v = torch.einsum("nij,nkj->nki", R, out.verts - J0) + J0
        return j, v * self.flip + T[:, None]


def load_side(run: Path, side: str, n: int, tri: dict, mano: Mano, wh, K, target="raw"):
    """recs: best colour WiLoR record of this handedness (init + pose prior).
    kp2d: colour WiLoR 2D joints; on triangulated frames the colour detection (any
    handedness label) closest to the projected triangulation, else the labelled one.
    X: X_smooth (finite joints, init/metrics/viz). tgt: t -> (21x3 target, 21 weights):
    X_raw weighted by conf on `valid` joints when hands_tri.json has them (6b v2), else
    X_smooth with unit weight on finite joints."""
    S = tri.get(side, {})
    X = {}
    for k, v in S.get("X_smooth", {}).items():
        a = np.array(v, dtype=float)
        if np.isfinite(a).all(axis=1).sum() >= 6:
            X[int(k)] = a
    tgt = {}
    use_raw = target == "raw" and "X_raw" in S and "valid" in S
    for t in X:
        if use_raw and str(t) in S["X_raw"]:
            a = np.array(S["X_raw"][str(t)], dtype=float)
            w = np.array(S["valid"][str(t)], dtype=float)
            if "conf" in S:
                w = w * np.clip(np.array(S["conf"][str(t)], dtype=float), 0, 1)
        else:
            a, w = X[t].copy(), np.ones(21)
        w = np.where(np.isfinite(a).all(1), w, 0.0)
        a = np.nan_to_num(a)
        if (w > 0).sum() >= 1:
            tgt[t] = (a, w)
    recs, kp2d = {}, {}
    for t in range(n):
        f = run / f"recon_hawor/wilor_raw/{t:06d}.json"
        if not f.exists():
            continue
        dets = []
        for d in json.loads(f.read_text()):
            if d.get("score", 0) < 0.3:
                continue
            _, j = mano.local(d)
            dets.append((d, wilor_keypoints_2d(j.cpu().numpy(), d, wh)))
        own = [z for z in dets if z[0]["is_right"] == (side == "right")]
        if own:
            recs[t] = max(own, key=lambda z: z[0].get("score", 0))[0]
        if t in X and dets:
            ok = np.isfinite(X[t]).all(1)
            uv = project_np(X[t][ok], K)
            dist = [np.linalg.norm(uv - k2[ok], axis=1).mean() for _, k2 in dets]
            if min(dist) < 80.0:
                kp2d[t] = dets[int(np.argmin(dist))][1]
        elif own:
            kp2d[t] = max(own, key=lambda z: z[0].get("score", 0))[1]
    return recs, kp2d, X, tgt, use_raw


MIRROR = np.array([1.0, -1.0, -1.0])


def apply_prior(d: Path, side: str, recs: dict) -> int:
    """Replace recs[t] by the HOPformer prediction where it has this side. hop_infer.py stores HOPformer's own
    smplx-MANO params (flat_hand_mean=False, separate left model); the articulated pose is hand_pose + hand_mean
    and a left hand maps to this mirrored right-hand convention by (x, -y, -z) per joint (checked on ep24
    against the triangulation: 3-5x lower fingertip error than the other conventions). Betas: the WiLoR
    record's where one exists (the sequence betas are fitted anyway)."""
    n = 0
    for f in sorted(d.glob("[0-9]*.json")):
        h = json.loads(f.read_text())
        if side not in h or h[side].get("hand_pose_full") is None:
            continue
        t, r = int(f.stem), h[side]
        go, hp = np.array(r["global_orient"]), np.array(r["hand_pose_full"]).reshape(15, 3)
        if side == "left":
            go, hp = go * MIRROR, hp * MIRROR
        betas = recs[t]["mano"]["betas"] if t in recs else r["betas"]
        recs[t] = {"is_right": side == "right", "score": 1.0, "source": "hopformer",
                   "mano": {"global_orient": go.tolist(), "hand_pose": hp.reshape(-1).tolist(), "betas": betas}}
        n += 1
    return n

def fit_side(side, recs, kp2d, X, tgt, K, mano: Mano, args, log):
    is_right = side == "right"
    support = sorted(set(recs) | set(X))
    t0, t1 = support[0], support[-1]
    frames = list(range(t0, t1 + 1))
    N = len(frames)
    rec_frames = np.array(sorted(recs))
    tri_frames = np.array(sorted(X))

    def nearest(arr, t):
        return int(arr[np.argmin(np.abs(arr - t))])

    betas0 = np.mean([recs[t]["mano"]["betas"] for t in recs], 0)
    model = HandModel(mano, is_right)
    Fm = np.eye(3) if is_right else FLIP

    # per-frame initial WiLoR params (nearest record where missing)
    src = [recs[nearest(rec_frames, t)] for t in frames]
    Rgo = np.stack([cv2.Rodrigues(np.array(r["mano"]["global_orient"], float))[0] for r in src])
    pose0 = np.stack([np.array(r["mano"]["hand_pose"], float) for r in src])
    with torch.no_grad():
        Jw = model(torch.tensor(Rgo, dtype=torch.float32, device=DEV),
                   torch.tensor(pose0, dtype=torch.float32, device=DEV),
                   torch.tensor(betas0, dtype=torch.float32, device=DEV)[None],
                   torch.zeros(N, 3, device=DEV)).cpu().numpy()  # (N,21,3) with WiLoR orientation, T=0

    # rigid (rotation) correction + translation on triangulated frames
    Rcorr, Tinit, scales = {}, {}, []
    for i, t in enumerate(frames):
        if t in X:
            ok = np.isfinite(X[t]).all(1)
            s, R, _ = umeyama(Jw[i][ok], X[t][ok], with_scale=True)
            scales.append(s)
            Rcorr[t] = R
            Tinit[t] = X[t][ok].mean(0) - (Jw[i][ok] @ R.T).mean(0)
    s_seq = float(np.median(scales)) if scales else 1.0
    R0 = np.zeros((N, 3, 3))
    T0 = np.zeros((N, 3))
    for i, t in enumerate(frames):
        Rc = Rcorr[t] if t in Rcorr else (Rcorr[nearest(tri_frames, t)] if len(tri_frames) else np.eye(3))
        R0[i] = (Fm @ Rc @ Fm) @ Rgo[i]  # camera-space correction -> right-hand parameter space
        if t in Tinit:
            T0[i] = Tinit[t]
        else:
            J = Jw[i] @ Rc.T
            if t in kp2d:
                T0[i] = pnp_translation(s_seq * J, kp2d[t], K)
            else:
                T0[i] = np.nan
    # interpolate translation where neither triangulation nor 2D exists
    bad = ~np.isfinite(T0).all(1)
    if bad.any():
        idx = np.arange(N)
        for c in range(3):
            T0[bad, c] = np.interp(idx[bad], idx[~bad], T0[~bad, c])
    log(f"[{side}] frames {t0}..{t1} ({N}), wilor recs {len(recs)}, triangulated {len(X)}, "
        f"init similarity scale median {s_seq:.3f}")

    # ---- tensors
    f32 = dict(dtype=torch.float32, device=DEV)
    R0_t = torch.tensor(R0, **f32)
    pose_w = torch.tensor(pose0, **f32)
    Xt = torch.zeros(N, 21, 3, **f32)
    Xm = torch.zeros(N, 21, **f32)
    P2 = torch.zeros(N, 21, 2, **f32)
    has2 = torch.zeros(N, **f32)
    has3 = torch.zeros(N, **f32)
    for i, t in enumerate(frames):
        if t in tgt:
            Xt[i] = torch.tensor(tgt[t][0], **f32)
            Xm[i] = torch.tensor(tgt[t][1], **f32)
            has3[i] = float((tgt[t][1] > 0).sum() >= 3)
        if t in kp2d:
            P2[i] = torch.tensor(kp2d[t], **f32)
            has2[i] = 1
    w2 = torch.where(has3 > 0, torch.full_like(has3, args.w2d_tri), torch.full_like(has3, args.w2d_notri)) * has2
    K_t = torch.tensor(K, **f32)

    dR = torch.zeros(N, 3, **f32, requires_grad=True)
    hp = pose_w.clone().requires_grad_(True)
    T = torch.tensor(T0, **f32).requires_grad_(True)
    betas = torch.tensor(betas0, **f32)[None].clone().requires_grad_(True)

    def losses():
        R = axis_angle_to_matrix(dR) @ R0_t
        J = model(R, hp, betas, T) * 1000.0  # mm
        e3 = ((J - Xt * 1000.0) ** 2).sum(-1)
        l3 = (gmof(e3, args.sigma3d) * Xm).sum() / Xm.sum().clamp(min=1)
        z = J[..., 2:3].clamp(min=10.0)
        uv = J[..., :2] / z * torch.stack([K_t[0, 0], K_t[1, 1]]) + K_t[:2, 2]
        e2 = ((uv - P2) ** 2).sum(-1)
        l2 = (gmof(e2, args.sigma2d).mean(-1) * w2).sum() / has2.sum().clamp(min=1)
        lpose = ((hp - pose_w) ** 2).mean(-1).mean()
        lbeta = (betas ** 2).sum()
        lacc = ((J[2:] - 2 * J[1:-1] + J[:-2]) ** 2).sum(-1).mean() if N > 2 else J.sum() * 0
        lpv = ((hp[1:] - hp[:-1]) ** 2).sum(-1).mean() if N > 1 else J.sum() * 0
        lrv = ((R[1:] - R[:-1]) ** 2).sum((-1, -2)).mean() if N > 1 else J.sum() * 0
        terms = {"3d": l3, "2d": l2, "pose": args.w_pose * lpose, "beta": args.w_beta * lbeta,
                 "acc": args.w_acc * lacc, "pvel": args.w_pvel * lpv, "rvel": args.w_rvel * lrv}
        return sum(terms.values()), terms

    def fmt(terms):
        return " ".join(f"{k}={v.item():.3f}" for k, v in terms.items())

    # stage 1: rigid + shape only, stage 2: everything
    stages = [([dR, T, betas], args.iters1), ([dR, T, betas, hp], args.iters2)]
    for si, (params, iters) in enumerate(stages):
        opt = torch.optim.Adam(params, lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=args.lr * 0.05)
        for it in range(iters):
            opt.zero_grad()
            loss, terms = losses()
            loss.backward()
            opt.step()
            sched.step()
            if it % 500 == 0 or it == iters - 1:
                log(f"[{side}] stage{si + 1} it {it:5d} loss {loss.item():.3f} | {fmt(terms)}")
    if args.lbfgs > 0:
        opt = torch.optim.LBFGS([dR, T, betas, hp], lr=1.0, max_iter=args.lbfgs, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss, _ = losses()
            loss.backward()
            return loss

        opt.step(closure)
        loss, terms = losses()
        log(f"[{side}] lbfgs loss {loss.item():.3f} | {fmt(terms)}")

    with torch.no_grad():
        R = axis_angle_to_matrix(dR) @ R0_t
        J, V = model(R, hp, betas, T, verts=True)
        go = matrix_to_axis_angle(R)
    return {"frames": frames, "support": support, "J": J.cpu().numpy(), "V": V.cpu().numpy(),
            "go": go.cpu().numpy(), "hp": hp.detach().cpu().numpy(), "T": T.detach().cpu().numpy(),
            "betas": betas.detach().cpu().numpy()[0], "init_scale": s_seq, "Jinit": Jw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sigma3d", type=float, default=20.0, help="GMoF scale, mm")
    ap.add_argument("--sigma2d", type=float, default=40.0, help="GMoF scale, px")
    ap.add_argument("--w2d_tri", type=float, default=0.02)
    ap.add_argument("--w2d_notri", type=float, default=0.2)
    ap.add_argument("--w_pose", type=float, default=5.0)
    ap.add_argument("--w_beta", type=float, default=0.5)
    ap.add_argument("--w_acc", type=float, default=0.3)
    ap.add_argument("--w_pvel", type=float, default=10.0)
    ap.add_argument("--w_rvel", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--iters1", type=int, default=500)
    ap.add_argument("--iters2", type=int, default=2500)
    ap.add_argument("--lbfgs", type=int, default=100)
    ap.add_argument("--target", choices=["raw", "smooth"], default="raw",
                    help="3D target: X_raw on valid joints (conf-weighted) if present, else X_smooth")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--tri", default="hands_tri.json", help="triangulated hands file in <run>/stereo/")
    ap.add_argument("--out", default="mano_fit", help="output dir in <run>/stereo/ (video gets the same suffix)")
    ap.add_argument("--prior", default=None,
                    help="recon_hawor/<dir> of HOPformer records (hop_infer.py): replaces the colour WiLoR record as "
                         "init + hand_pose prior wherever it has the side (WiLoR 2D keypoints are kept)")
    args = ap.parse_args()
    torch.manual_seed(0)
    run = ROOT / "runs" / args.run
    meta = json.loads((run / "episode.json").read_text())
    ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.0]])
    wh = (ia["width"], ia["height"])
    n = len(list((run / "recon_hawor/frames").glob("*.png")))
    tri = json.loads((run / "stereo" / args.tri).read_text())
    mano = Mano(str(ROOT / "weights/hand"))
    outdir = run / "stereo" / args.out
    logs = []

    def log(s):
        print(s, flush=True)
        logs.append(s)

    report = {"args": vars(args), "hands_tri_version": tri.get("version", 1), "sync_offset": tri.get("sync_offset"),
              "sync_offset_frac": tri.get("sync_offset_frac"), "sides": {}}
    fits, kp2ds, Xs = {}, {}, {}
    for side in SIDES:
        recs, kp2d, X, tgt, used_raw = load_side(run, side, n, tri, mano, wh, K, args.target)
        if args.prior:
            nrep = apply_prior(run / "recon_hawor" / args.prior, side, recs)
            log(f"[{side}] prior {args.prior}: {nrep} frames")
        if not recs and not X:
            log(f"[{side}] no data, skipped")
            continue
        if not recs:
            log(f"[{side}] triangulation but no colour WiLoR record for init, skipped")
            continue
        fit = fit_side(side, recs, kp2d, X, tgt, K, mano, args, log)
        fits[side], kp2ds[side], Xs[side] = fit, kp2d, X

        # ---- write per-frame jsons (support frames only)
        sd = outdir / side
        if sd.exists():
            shutil.rmtree(sd)
        sd.mkdir(parents=True)
        idx = {t: i for i, t in enumerate(fit["frames"])}
        maxdiff = 0.0
        for t in fit["support"]:
            i = idx[t]
            rec = {"mano": {"betas": fit["betas"].tolist(), "global_orient": fit["go"][i].tolist(),
                            "hand_pose": fit["hp"][i].tolist()},
                   "is_right": side == "right", "T": fit["T"][i].tolist(), "joints": fit["J"][i].tolist(),
                   "has_triangulation": t in X, "has_wilor_2d": t in kp2d}
            (sd / f"{t:06d}.json").write_text(json.dumps(rec))
            if t % 10 == 0:  # consistency with Mano.local convention
                _, jl = mano.local(rec)
                maxdiff = max(maxdiff, float(np.abs(jl.cpu().numpy() + fit["T"][i] - fit["J"][i]).max()))
        log(f"[{side}] Mano.local(rec)+T vs exported joints: max diff {maxdiff * 1000:.4f} mm")

        # ---- metrics
        J = {t: fit["J"][idx[t]] for t in fit["support"]}
        tri_t = sorted(X)

        def mpjpe(seq):
            e = [np.nanmean(np.linalg.norm(seq[t] - X[t], axis=1)) * 1000 for t in tri_t if t in seq]
            return (float(np.mean(e)), float(np.median(e))) if e else (None, None)

        def mpjpe_raw(seq):
            e = [np.linalg.norm(seq[t] - tgt[t][0], axis=1)[tgt[t][1] > 0].mean() * 1000
                 for t in sorted(tgt) if t in seq and (tgt[t][1] > 0).sum() >= 3]
            return (float(np.mean(e)), float(np.median(e))) if e else (None, None)

        def reproj(seq):
            e = [np.nanmean(np.linalg.norm(project_np(seq[t], K) - kp2d[t], axis=1)) for t in seq if t in kp2d]
            return (float(np.mean(e)), float(np.median(e))) if e else (None, None)

        def summarize(seq, name):
            m, md = mpjpe(seq)
            mr, mrd = mpjpe_raw(seq)
            r, rd = reproj(seq)
            on_tri = {t: seq[t] for t in tri_t if t in seq}
            return {"frames": len(seq), "mpjpe_tri_mm_mean": m, "mpjpe_tri_mm_median": md,
                    "mpjpe_target_valid_mm_mean": mr, "mpjpe_target_valid_mm_median": mrd,
                    "reproj_wilor2d_px_mean": r, "reproj_wilor2d_px_median": rd,
                    "jitter_on_tri_frames_mm_per_frame2": jitter_mm(on_tri),
                    "jitter_all_frames_mm_per_frame2": jitter_mm(seq)}

        rows = {"mano_fit": summarize(J, "fit"), "triangulated_X_smooth": {
            "frames": len(X), "jitter_on_tri_frames_mm_per_frame2": jitter_mm(X),
            "reproj_wilor2d_px_mean": reproj(X)[0], "reproj_wilor2d_px_median": reproj(X)[1]}}
        if used_raw:
            Xr = {t: np.where((w > 0)[:, None], a, np.nan) for t, (a, w) in tgt.items()}
            rows["triangulated_X_raw_valid"] = {"frames": len(Xr), "jitter_on_tri_frames_mm_per_frame2": jitter_mm(Xr),
                                                "reproj_wilor2d_px_mean": reproj(Xr)[0],
                                                "reproj_wilor2d_px_median": reproj(Xr)[1]}
        for model in ("wilor", "agile", "hawor"):
            d = run / f"stereo/aligned_{model}/{side}"
            seq = {}
            for t in tri_t:
                f = d / f"{t:06d}.json"
                if f.exists():
                    seq[t] = np.array(json.loads(f.read_text())["joints"])
            if seq:
                rows[f"{model}_aligned_6b"] = summarize(seq, model)
        # bone lengths: fitted vs triangulated
        bl = lambda A: np.array([np.linalg.norm(A[a] - A[b]) for a, b in BONES])  # noqa: E731
        bt = np.nanmedian([bl(X[t]) for t in tri_t], 0) if tri_t else None
        bf = bl(fit["J"][0])
        bi = bl(fit["Jinit"][0])
        rows["shape"] = {"betas": fit["betas"].tolist(), "init_similarity_scale_wilor_to_tri": fit["init_scale"],
                         "sum_bone_len_mm": {"fit": float(bf.sum() * 1000), "wilor_init": float(bi.sum() * 1000),
                                             "triangulated_median": float(bt.sum() * 1000) if bt is not None else None}}
        per_frame = []
        for t in fit["support"]:
            pf = {"frame": t}
            if t in X:
                pf["mpjpe_mm"] = float(np.nanmean(np.linalg.norm(J[t] - X[t], axis=1)) * 1000)
                if t in tgt and (tgt[t][1] > 0).sum() >= 3:
                    pf["mpjpe_target_valid_mm"] = float(np.linalg.norm(J[t] - tgt[t][0], axis=1)[tgt[t][1] > 0].mean() * 1000)
                if t in kp2d:
                    pf["tri_vs_wilor2d_px"] = float(np.nanmean(np.linalg.norm(project_np(X[t], K) - kp2d[t], axis=1)))
            if t in kp2d:
                pf["reproj_px"] = float(np.linalg.norm(project_np(J[t], K) - kp2d[t], axis=1).mean())
            per_frame.append(pf)
        rows["per_frame"] = per_frame
        rows["target"] = "X_raw(valid, conf-weighted)" if used_raw else "X_smooth"
        report["sides"][side] = rows
        log(f"[{side}] " + json.dumps({k: v for k, v in rows.items() if k != "per_frame"}, indent=1))
    report["log"] = logs
    (outdir / "report.json").write_text(json.dumps(report, indent=2))

    if args.no_video:
        return
    # ---- video: filled, flat-shaded MANO mesh over the colour frames (rendered at half res)
    viz = Path(os.environ["V2D_VIZ"]) / "07_stereo_hands"
    viz.mkdir(parents=True, exist_ok=True)
    Wo, Ho = 1014, 760
    Ks = K.copy()
    Ks[0] *= Wo / wh[0]
    Ks[1] *= Ho / wh[1]
    colour = {"right": np.array([60, 140, 255.0]), "left": np.array([255, 200, 60.0])}  # BGR
    light = np.array([0.0, -0.3, -1.0])
    light /= np.linalg.norm(light)
    wr = imageio.get_writer(str(viz / (f"{args.run}_manofit.mp4" if args.out == "mano_fit" else f"{args.run}_{args.out}.mp4")), fps=meta["fps"], codec="libx264", quality=8,
                            macro_block_size=2)
    for t in range(n):
        img = cv2.resize(cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png")), (Wo, Ho))
        over = img.copy()
        drawn = False
        for side, fit in fits.items():
            if t not in fit["support"]:
                continue
            i = fit["frames"].index(t)
            V = fit["V"][i]
            F = mano.faces(side == "right")
            tris = V[F]
            nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            cen = tris.mean(1)
            facing = (nrm * -cen).sum(1) > 0
            shade = 0.35 + 0.65 * np.abs(nrm @ light)
            uv = project_np(V, Ks)
            order = np.argsort(-cen[:, 2])
            for f in order:
                if not facing[f]:
                    continue
                c = tuple(int(x) for x in colour[side] * shade[f])
                cv2.fillConvexPoly(over, np.round(uv[F[f]] * 4).astype(np.int32), c, cv2.LINE_AA, shift=2)
            drawn = True
        img = cv2.addWeighted(over, 0.45, img, 0.55, 0) if drawn else img
        for side in fits:
            if t in Xs[side]:
                for p in project_np(Xs[side][t][np.isfinite(Xs[side][t]).all(1)], Ks):
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
            if t in kp2ds[side]:
                for p in kp2ds[side][t] * np.array([Wo / wh[0], Ho / wh[1]]):
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, f"{args.run} f{t}  MANO fit (orange=R, blue=L) green: triangulated, red: WiLoR 2D", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        wr.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    wr.close()
    log(f"video -> {viz / (args.run + '_manofit.mp4')}")


if __name__ == "__main__":
    main()
