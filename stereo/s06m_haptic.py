"""HaPTIC (Ye et al., 4D hand trajectory from monocular video) for the frames where the stereo hand fit is missing.

The stereo fit stays the source of truth: frames where the hand was detected are copied unchanged. HaPTIC is used
for image evidence on the undetected frames only (fingers + wrist orientation from the image, temporally coherent).
Per side, on the colour camera (recon_hawor/frames, intrinsics_stable):
  1  boxes   our hand joints of every frame (detected: stereo; undetected: the optimiser fill of <fill>, s06k
             --no_geneoh) projected into the colour camera -> per-frame hand boxes in HaPTIC's det.pkl format
             (their person detector + ViTPose is not used: it does not work on egocentric video). Frames whose
             projected hand is not inside the image are flagged out of view.
  2  HaPTIC  8-frame temporal model over the whole track -> per-frame MANO rotations (global orient, finger
             pose) and a weak-perspective camera; per-frame depth only (no chained depth: we keep stereo depth).
  3  gate    on detected frames: root-relative joint error (our betas, HaPTIC rotations) and wrist-orientation
             error vs stereo. Reported per side.
  4  fuse    undetected in-view frames: finger pose + global orient = HaPTIC rotations directly (no refit of MANO
             to joints: that twists fingers), wrist on HaPTIC's image ray at the fill's depth. Out-of-view frames
             keep the fill. A parameter-space smoothing pass over the undetected frames only (detected frames fixed)
             removes the jumps at gap borders.
Writes <run>/stereo/<out>/{left,right}/<t>.json for every frame (schema of mano_fit_*, plus "observed" and
"filled_by": null | "haptic" | "optimizer") and report.json. Runs in the haptic image, cwd = the HaPTIC repo.
Usage: python s06m_haptic.py <run> [--fill mano_fit_hop_fill] [--out mano_fit_hop_haptic]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
HAPTIC = Path(os.environ.get("HAPTIC_ROOT", ROOT / "third_party/haptic"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s06k_geneoh import DEV, F, Hand, convention, log, read_side  # noqa: E402

sys.path.insert(0, str(HAPTIC))
CKPT = os.environ.get("HAPTIC_CKPT", str(HAPTIC / "output/release/mix_all/checkpoints/last.ckpt"))


def project(K, X):
    p = X @ K.T
    return p[..., :2] / p[..., 2:3]


@torch.enable_grad()
def smooth_boxes(center, scale, iters=1000, w=0.05, w_c=0.02, w_s=10):
    """HaPTIC's own box smoothing (nnutils/det_utils.smooth_bbox; that module imports ViTPose at top level)."""
    center = torch.tensor(center, dtype=torch.float32, device=DEV)
    scale = torch.tensor(scale, dtype=torch.float32, device=DEV)
    dc = torch.nn.Parameter(torch.zeros_like(center))
    ds = torch.nn.Parameter(torch.zeros_like(scale))
    opt = torch.optim.AdamW([dc, ds], lr=1e-3)
    for _ in range(iters):
        c, s = center + dc, scale + ds
        acc_c = c[2:] - 2 * c[1:-1] + c[:-2]
        acc_s = s[2:] - 2 * s[1:-1] + s[:-2]
        loss = w_c * acc_c.norm() + w_s * acc_s.norm() + w * (w_c * dc.norm() + w_s * ds.norm())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (center + dc).detach().cpu().numpy(), (scale + ds).detach().cpu().numpy()


@torch.no_grad()
def run_haptic(model, run, side, joints_cam, K, size):
    """joints_cam (n,21,3) real camera frame -> per frame dict of HaPTIC outputs (right-model space)."""
    from haptic.datasets.seq2clip import split_to_list_dl
    from haptic.utils.renderer import cam_crop_to_full_w_pp
    from nnutils import model_utils

    n = len(joints_cam)
    W, H = size
    uv = project(K, joints_cam)
    lo, hi = uv.min(1), uv.max(1)
    in_view = (joints_cam[:, :, 2].min(1) > 0.05) & (hi[:, 0] > 0) & (lo[:, 0] < W) & (hi[:, 1] > 0) & (lo[:, 1] < H)
    wh = np.maximum(hi - lo, 20.0)
    center, scale = smooth_boxes((lo + hi) / 2, wh / 100 * 1.5)  # det_utils.parse_det_seq conventions
    frames = sorted((run / "recon_hawor/frames").glob("*.png"))[:n]
    seq = {"imgname": [f.name for f in frames], "img_dir": str(run / "recon_hawor/frames"),
           "center": center, "scale": scale, "focal": np.tile(K, (n, 1, 1))[:, None],
           "is_right": np.full(n, int(side == "right")), "valid": in_view.astype(int),
           "cTw": np.tile(np.eye(4), (n, 1, 1)), "hand_pose": np.zeros((n, 45)), "hand_tsl": np.zeros((n, 3)),
           "seq": f"{run.name}_{side}"}
    nf = model.cfg.MODEL.NUM_FRAMES
    overlap = 1 if nf > 1 else 0
    dl = split_to_list_dl(model.cfg, seq, nf, overlap, box_mode="box_size_same", rescale_factor=2)
    out = {}
    for b, bs in enumerate(dl):
        bs = model_utils.to_cuda(bs, DEV)
        pred = model(bs)
        box_center = bs["box_center"][0].clone()
        if side == "left":
            box_center[..., 0] = W - box_center[..., 0]
        cam_full = cam_crop_to_full_w_pp(pred["pred_cam"], bs["intr"][0], H, W, box_center, bs["box_size"][0])
        mp = pred["pred_mano_params"]
        go = mp["global_orient"].reshape(-1, 3, 3).cpu().numpy()
        hp = Rotation.from_matrix(mp["hand_pose"].reshape(-1, 3, 3).cpu().numpy()).as_rotvec().reshape(-1, 45)
        kp = pred["pred_keypoints_3d"].reshape(go.shape[0], -1, 3).cpu().numpy()
        betas = mp["betas"].reshape(go.shape[0], -1).cpu().numpy()
        cf = cam_full.cpu().numpy()
        names = [int(e[0]) for e in bs["name"]]
        for k, t in enumerate(names):
            if t in out:  # window overlap / tail padding: first prediction wins
                continue
            out[t] = {"go": go[k], "hp": hp[k], "kp": kp[k], "betas": betas[k], "cam": cf[k]}
    return out, in_view, center, scale


def hp_convention(hap, hand_cls, ts):
    """Offset c with (HaPTIC hand_pose + c) = our hand_pose convention, from HaPTIC's own MANO joints."""
    from manopth.manolayer import ManoLayer
    lay = ManoLayer(mano_root=str(ROOT / "third_party/GeneOH-Diffusion/manopth/mano/models"), side="right",
                    use_pca=False, ncomps=45, flat_hand_mean=False)
    mean = lay.th_hands_mean.reshape(-1).numpy()
    best = None
    for name, c in (("same", np.zeros(45)), ("plus_mean", mean), ("minus_mean", -mean)):
        errs = []
        for t in ts:
            h = hand_cls(hap[t]["betas"])
            R = torch.tensor(hap[t]["go"], dtype=torch.float32, device=DEV)[None]
            hp = torch.tensor(hap[t]["hp"] + c, dtype=torch.float32, device=DEV)[None]
            j = h(R, hp, torch.zeros(1, 3, device=DEV))[0].cpu().numpy()
            j, kp = j - j[0], hap[t]["kp"] - hap[t]["kp"][0]  # wrist first in both orders
            errs.append(np.linalg.norm(kp[:, None] - j[None], axis=-1).min(1).mean())
        e = float(np.mean(errs)) * 1000
        log(f"   HaPTIC hand_pose convention {name}: {e:.2f} mm vs its own MANO joints")
        if best is None or e < best[0]:
            best = (e, name, c)
    return best


def geo(Ra, Rb):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", Ra, Rb)).as_rotvec(), axis=-1))


def fuse(hand, R, hp, T, obs, R_t, hp_t, W_t, iters=1500, sig_rot=0.05, sig_hp=0.1, sig_w=0.01,
         sig_acc=0.002, sig_pacc=0.05):
    """Parameter-space smoothing of the undetected frames toward their targets; detected frames are constants.
    R_t/hp_t: target rotation/finger pose; W_t: target wrist (mirrored camera frame); HaPTIC values on in-view gaps, the fill elsewhere."""
    t32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=DEV)  # noqa: E731
    free = t32(~obs)[:, None]
    r0 = t32(Rotation.from_matrix(R_t).as_rotvec())
    dr = torch.nn.Parameter(torch.zeros_like(r0))
    hp_v = torch.nn.Parameter(t32(hp_t).clone())
    T_v = torch.nn.Parameter(t32(T).clone())
    R_obs, hp_obs, T_obs = t32(R), t32(hp), t32(T)
    Rt, hpt, Wt = t32(R_t), t32(hp_t), t32(W_t)
    m = t32(~obs)
    J0 = hand.J0
    opt = torch.optim.Adam([dr, hp_v, T_v], lr=3e-3)
    from s06k_geneoh import rotvec_to_R
    for it in range(iters):
        Rf = rotvec_to_R(dr) @ Rt
        Rc = torch.where(free[..., None] > 0, Rf, R_obs)
        hpc = torch.where(free > 0, hp_v, hp_obs)
        Tc = torch.where(free > 0, T_v, T_obs)
        J = hand(Rc, hpc, Tc)
        l_rot = (((Rf - Rt) ** 2).sum((1, 2)) * m).sum() / m.sum().clamp(min=1) / sig_rot ** 2
        l_hp = (((hp_v - hpt) ** 2).mean(1) * m).sum() / m.sum().clamp(min=1) / sig_hp ** 2
        l_w = (((J[:, 0] - Wt) ** 2).sum(1) * m).sum() / m.sum().clamp(min=1) / sig_w ** 2
        acc = J[2:] - 2 * J[1:-1] + J[:-2]
        pacc = hpc[2:] - 2 * hpc[1:-1] + hpc[:-2]
        l_acc = (acc ** 2).sum(-1).mean() / sig_acc ** 2
        l_pacc = (pacc ** 2).mean() / sig_pacc ** 2
        loss = l_rot + l_hp + l_w + l_acc + l_pacc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == iters - 1:
            log(f"   fuse {it}: rot {l_rot.item():.3f} hp {l_hp.item():.3f} wrist {l_w.item():.3f} "
                f"acc {l_acc.item():.3f} pacc {l_pacc.item():.3f}")
    with torch.no_grad():
        Rf = rotvec_to_R(dr) @ Rt
        Rc = torch.where(free[..., None] > 0, Rf, R_obs)
        hpc = torch.where(free > 0, hp_v, hp_obs)
        Tc = torch.where(free > 0, T_v, T_obs)
        J = hand(Rc, hpc, Tc)
    return Rc.cpu().numpy(), hpc.cpu().numpy(), Tc.cpu().numpy(), J.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fill", default="mano_fit_hop_fill", help="every-frame hands (s06k --no_geneoh)")
    ap.add_argument("--out", default="mano_fit_hop_haptic")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    sd_out = run / "stereo" / a.out
    ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
    size = (int(ki["width"]), int(ki["height"]))
    n = len(list((run / "recon_hawor/frames").glob("*.png")))
    from demo import load_haptic_model
    t0 = time.time()
    model = load_haptic_model(CKPT, DEV)
    log(f"HaPTIC loaded ({time.time() - t0:.0f} s), NUM_FRAMES {model.cfg.MODEL.NUM_FRAMES}")
    report = {"run": a.run, "fill": a.fill, "ckpt": CKPT, "sides": {}}
    for side in ("left", "right"):
        recs = read_side(run, a.fill, side, n)
        if len(recs) < n:
            log(f"[{side}] {len(recs)}/{n} frames in {a.fill}: side absent, skipped")
            continue
        obs = np.array([bool(recs[t].get("observed", True)) for t in range(n)])
        M = F if side == "left" else np.eye(3)
        off, perm, conv_name, conv_mm = convention({t: recs[t] for t in range(n) if obs[t]} or recs, side)
        betas = recs[0]["mano"]["betas"]
        hand = Hand(betas, off)
        # our params in the mirrored camera frame: joints_m = R (J - J0) + J0 + Tm
        R = np.stack([Rotation.from_rotvec(recs[t]["mano"]["global_orient"]).as_matrix() for t in range(n)])
        hp = np.stack([np.asarray(recs[t]["mano"]["hand_pose"], float) for t in range(n)])
        Tm = np.stack([M @ np.asarray(recs[t]["T"], float) for t in range(n)])
        J_real = np.stack([np.asarray(recs[t]["joints"], float) for t in range(n)])  # our joint order, real cam
        hap, in_view, center, scale = run_haptic(model, run, side, J_real, K, size)
        ts_obs = [t for t in np.flatnonzero(obs) if t in hap][:: max(1, obs.sum() // 30)]
        e_c, c_name, c = hp_convention(hap, lambda b: Hand(b, off), ts_obs)
        # HaPTIC in our parameters (mirrored camera frame)
        R_h = np.stack([hap[t]["go"] for t in range(n)])
        hp_h = np.stack([hap[t]["hp"] + c for t in range(n)])
        w_h = np.stack([hap[t]["kp"][0] + hap[t]["cam"] for t in range(n)])  # HaPTIC wrist, mirrored cam
        with torch.no_grad():
            t32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=DEV)  # noqa: E731
            J_ours = hand(t32(R), t32(hp), t32(Tm)).cpu().numpy()
            J_hap = hand(t32(R_h), t32(hp_h), t32(Tm)).cpu().numpy()
        rel = lambda J: J - J[:, :1]  # noqa: E731
        o = obs & in_view
        gate = {"root_rel_mpjpe_mm": float(np.linalg.norm(rel(J_hap)[o] - rel(J_ours)[o], axis=-1).mean() * 1000),
                "global_orient_err_deg": float(np.median(geo(R_h[o], R[o]))),
                "wrist_ray_err_deg": float(np.median(np.degrees(np.arccos(np.clip(
                    (w_h[o] / np.linalg.norm(w_h[o], axis=1, keepdims=True) *
                     J_ours[o, 0] / np.linalg.norm(J_ours[o, 0], axis=1, keepdims=True)).sum(1), -1, 1))))),
                "haptic_depth_err_m": float(np.median(np.abs(w_h[o, 2] - J_ours[o, 0, 2])))}
        log(f"[{side}] gate on {o.sum()} detected frames: {json.dumps(gate)}")
        # targets for the undetected frames
        use = ~obs & in_view
        d = w_h / np.linalg.norm(w_h, axis=1, keepdims=True)
        W_t = J_ours[:, 0].copy()
        W_t[use] = d[use] * (J_ours[use, 0, 2] / d[use, 2])[:, None]  # HaPTIC ray, fill depth
        R_t, hp_t = R.copy(), hp.copy()
        R_t[use], hp_t[use] = R_h[use], hp_h[use]
        log(f"[{side}] {obs.sum()}/{n} detected, {use.sum()} undetected in view -> HaPTIC, "
            f"{(~obs & ~in_view).sum()} undetected out of view -> fill")
        RC, hpC, TC, JC = fuse(hand, R, hp, Tm, obs, R_t, hp_t, W_t)
        sd = sd_out / side
        sd.mkdir(parents=True, exist_ok=True)
        for t in range(n):
            jc = (JC[t] @ M.T)[perm]  # layer order -> ours, real camera frame
            rec = {"mano": {"betas": betas, "global_orient": Rotation.from_matrix(RC[t]).as_rotvec().tolist(),
                            "hand_pose": hpC[t].tolist()},
                   "is_right": side == "right", "T": (M @ TC[t]).tolist(), "joints": jc.tolist(),
                   "has_triangulation": bool(recs[t].get("has_triangulation")),
                   "has_wilor_2d": bool(recs[t].get("has_wilor_2d")), "observed": bool(obs[t]),
                   "filled_by": None if obs[t] else ("haptic" if use[t] else "optimizer")}
            (sd / f"{t:06d}.json").write_text(json.dumps(rec))
        jump = lambda J: float(np.linalg.norm(J[1:] - J[:-1], axis=-1).mean(1).max() * 1000)  # noqa: E731
        report["sides"][side] = {
            "detected": int(obs.sum()), "filled_haptic": int(use.sum()), "filled_optimizer": int((~obs & ~in_view).sum()),
            "convention": conv_name, "convention_mm": conv_mm, "haptic_hp_convention": c_name,
            "haptic_hp_convention_mm": e_c, "gate": gate,
            "change_on_detected_mm": float(np.linalg.norm(JC[obs] - J_ours[obs], axis=-1).mean() * 1000),
            "fill_vs_haptic_on_filled_mm": float(np.linalg.norm(JC[use] - J_ours[use], axis=-1).mean() * 1000) if use.any() else None,
            "max_frame_jump_mm_in": jump(J_ours), "max_frame_jump_mm_out": jump(JC)}
        log(f"[{side}] done: {json.dumps(report['sides'][side])}")
    sd_out.mkdir(parents=True, exist_ok=True)
    (sd_out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {sd_out}")


if __name__ == "__main__":
    main()
