"""AGILE (Shi et al., SIGGRAPH 2026, arXiv 2602.04672) — pose initialisation (§3.2)
and contact-aware hand-object optimisation (§3.3), without the object-generation
stage (§3.1): Track 3 provides scanned, textured, metric meshes.

Runs in the `agile` image (PyTorch3D/CUDA, manotorch, DINOv3) after prepare.py.

§3.2  hand: WiLoR per-frame MANO (beta, theta, R_h fixed) -> global scale s_h by
      rotation-fixed trimmed ICP to depth-unprojected SAM2 hand pixels -> per-frame
      T_h by translation-only PnP against WiLoR's 2D joints.
      object: FoundationPose at the interaction onset frame (IOF, prepare.py).
§3.3  bidirectional online optimisation from the IOF. IOF: object anisotropic scale
      via L_mask + L_contact with rotation fixed and translation on the viewing ray,
      then frozen. Every frame: Step 1 hand translation (L_joint, <=200 it, Adam 1e-3);
      Step 2 object pose from the previous frame (5 L_mask + 10 L_dino +
      lambda L_interact, <=400 it, Adam rot 2e-3 / trans 1e-3). Stop when the moving
      average of parameter change < 1e-4.

Paper details we had to choose (not specified in the paper) are marked "CHOICE".
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from geometry import (Mano, SDFGrid, constrained_trimmed_icp_scale, pnp_translation, project,
                      rotvec_to_matrix, unproject_mask, visible_vertices, wilor_keypoints_2d)
from render import Crop, Renderer, sample_surface
from dino import Dino, sample_features, sample_maps

ROOT = Path(os.environ["V2D_ROOT"])
HANDS = {"left": 2, "right": 3}                  # SAM2 track ids
W_MASK, W_DINO, W_CONTACT = 5.0, 10.0, 5.0       # paper Appendix A
SIGMA, D_MAX = 40.0, 0.05                        # interaction gating (Appendix A)
LR_ROT, LR_TRANS, LR_HAND = 2e-3, 1e-3, 1e-3
IT_HAND, IT_OBJ = 200, 400
CONV_TOL, CONV_WIN = 1e-4, 10
CROP_RES = 256        # CHOICE: optimisation crop resolution
N_POINTS = 512        # CHOICE: N_p visible points for L_dino
RESAMPLE_EVERY = 50   # CHOICE: "periodically" re-sample points / similarity maps
MIN_MASK_PX = 150     # CHOICE: below this the object counts as unobserved in a frame


# ----------------------------------------------------------------------------- utils
def load_mask(path, shape):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 0) if m is not None else np.zeros(shape, bool)


def decode_depth(path):
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = np.where(png > 0, 65535.0 / np.maximum(png, 1) - 1.0, 0)
    return np.where(np.isfinite(d) & (d > 1e-3), d, 0)


def quat_wxyz_to_R(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def mask_loss(A, M):
    """Eq. 3, ||A_ren - M_gt||^2, normalised by the observed mask area (CHOICE: the
    paper gives no normalisation). A mean over the crop made the 5:10 mask/DINO balance
    collapse: L_dino rewards shrinking the projection (edge points move onto the high-
    similarity object interior), and on ep 11 pushed a cup from 0.54 m to 0.75 m."""
    return ((A - M) ** 2).sum() / M.sum().clamp(min=1.0)


class Converged:
    """Paper: stop when the moving average of parameter variation is < 1e-4."""

    def __init__(self, tol=CONV_TOL, win=CONV_WIN):
        self.tol, self.win, self.hist, self.prev = tol, win, [], None

    def __call__(self, params) -> bool:
        cur = torch.cat([p.detach().reshape(-1) for p in params])
        if self.prev is not None:
            self.hist.append(float((cur - self.prev).abs().max()))
        self.prev = cur.clone()
        return len(self.hist) >= self.win and np.mean(self.hist[-self.win:]) < self.tol


# ------------------------------------------------------------------- §3.2 hand init
def init_hands(ep, mano, K, n, shape, depth_dir, dev):
    """Per hand side: per-frame fixed MANO verts/joints (R_h applied), WiLoR J2D,
    global scale s_h (median of per-frame constrained ICP), per-frame PnP T_h."""
    H, W = shape
    hands = {}
    for side, hid in HANDS.items():
        recs = {}
        for t in range(n):
            dets = [d for d in json.loads((ep / f"recon_hawor/wilor_raw/{t:06d}.json").read_text())
                    if d["is_right"] == (side == "right") and d.get("score", 0) >= 0.5]
            if dets:
                recs[t] = max(dets, key=lambda d: d["score"])
        if not recs:
            continue
        frames = {}
        for t, rec in recs.items():
            v, j = mano.local(rec)
            v, j = v.cpu().numpy(), j.cpu().numpy()
            frames[t] = {"rec": rec, "v": v, "j": j, "j2d": wilor_keypoints_2d(j, rec, (W, H))}
        # global scale s_h: rotation-fixed trimmed ICP (model-to-scan) per frame, median
        scales = []
        faces = mano.faces(side == "right")
        for t, f in frames.items():
            m = load_mask(ep / f"objects/masks/{hid}/{t:06d}.png", shape)
            if m.sum() < 800:
                continue
            t0 = pnp_translation(f["j"], f["j2d"], K)
            # SAM2 hand masks include forearm/sleeve; keep only pixels inside the MANO
            # hand's own projected silhouette (dilated) so ICP fits the hand, not the arm.
            uv = f["v"] + t0
            uv = (uv[:, :2] / uv[:, 2:3] * K[[0, 1], [0, 1]] + K[:2, 2]).astype(np.int32)
            hull = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(uv), 1)
            hull = cv2.dilate(hull, np.ones((15, 15), np.uint8)) > 0
            scan = unproject_mask(decode_depth(depth_dir / f"{t:06d}.png"), m & hull, K)
            if len(scan) < 200:
                continue
            vis = visible_vertices(f["v"] + t0, faces)
            model = f["v"][vis]
            s, _, res = constrained_trimmed_icp_scale(model, scan, t0, s0=1.0)
            if 0.5 < s < 2.0:
                scales.append(s)
        s_h = float(np.median(scales)) if scales else 1.0
        for t, f in frames.items():
            f["T"] = pnp_translation(s_h * f["j"], f["j2d"], K)
        hands[side] = {"s_h": s_h, "n_scale_frames": len(scales), "frames": frames,
                       "faces": faces}
        print(f"hand {side}: {len(frames)} frames, s_h = {s_h:.3f} (from {len(scales)} frames)", flush=True)
    return hands


# ------------------------------------------------------------- §3.3 step 1 (hands)
def refine_hand_translation(f, s_h, K_t, dev):
    """Step 1: T_h only, L_joint = mean ||Pi(s_h J + T_h) - J2D||^2 (Eq. 2)."""
    J = torch.tensor(s_h * f["j"], device=dev, dtype=torch.float32)
    j2d = torch.tensor(f["j2d"], device=dev, dtype=torch.float32)
    T = torch.tensor(f["T"], device=dev, dtype=torch.float32, requires_grad=True)
    opt, conv = torch.optim.Adam([T], lr=LR_HAND), Converged()
    for it in range(IT_HAND):
        loss = ((project(J + T, K_t) - j2d) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if conv([T]):
            break
    f["T"] = T.detach().cpu().numpy()
    f["L_joint"] = float(loss)
    return f


def hand_vertices(hands, t, dev, sides=None):
    """Camera-frame vertices (N,3) of the given hand sides (default: all present at t)."""
    sides = [s for s in hands if t in hands[s]["frames"]] if sides is None else sides
    vs = [torch.tensor(hands[s]["s_h"] * hands[s]["frames"][t]["v"] + hands[s]["frames"][t]["T"],
                       device=dev, dtype=torch.float32) for s in sides]
    return torch.cat(vs) if vs else None


def common_sides(hands, t0, t1):
    """Hands observed in both frames: L_interact needs vertex correspondence."""
    return [s for s in hands if t0 in hands[s]["frames"] and t1 in hands[s]["frames"]]


# ------------------------------------------------------------ §3.3 object objective
class ObjectFrame:
    """Everything needed to evaluate L_obj at one frame."""

    def __init__(self, ep, t, obj, K_full, shape, R_prev, T_prev, scale, renderer, dino, occluder_ids, dev):
        self.t, self.dev = t, dev
        img = cv2.cvtColor(cv2.imread(str(ep / f"recon_hawor/frames/{t:06d}.png")), cv2.COLOR_BGR2RGB)
        mo = load_mask(ep / f"objects/masks/{obj['object_id']}/{t:06d}.png", shape)
        occ = np.zeros(shape, bool)
        for oid in occluder_ids:
            occ |= load_mask(ep / f"objects/masks/{oid}/{t:06d}.png", shape)
        self.observed = mo.sum() >= MIN_MASK_PX
        # Crop around the previous pose's projection (and the current mask).
        verts = (renderer.verts.cpu().numpy() * scale.cpu().numpy()) @ R_prev.T + T_prev
        uv = verts[:, :2] / np.clip(verts[:, 2:3], 1e-3, None) * K_full[[0, 1], [0, 1]] + K_full[:2, 2]
        box = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
        if self.observed:
            ys, xs = np.nonzero(mo)
            box = [min(box[0], xs.min()), min(box[1], ys.min()), max(box[2], xs.max()), max(box[3], ys.max())]
        self.crop = Crop.around(box, shape[::-1], CROP_RES)
        self.K = self.crop.K(torch.tensor(K_full, dtype=torch.float32, device=dev))
        self.mask_gt = torch.tensor(self.crop.image(mo.astype(np.float32), cv2.INTER_LINEAR), device=dev)
        self.occ = torch.tensor(self.crop.image(occ.astype(np.float32), cv2.INTER_NEAREST), device=dev)
        rgb = torch.tensor(self.crop.image(img, cv2.INTER_LINEAR), device=dev).float() / 255
        self.F_gt = dino.features(rgb) if (dino is not None and self.observed) else None


def sample_dino_points(of, renderer, dino, surf, R, T, scale):
    """Periodic re-sampling (Eq. 4): visible canonical points p_i, their features in
    the current textured render, and similarity maps S_i against the target."""
    rgb, z = renderer.textured(R, T, scale, of.K, CROP_RES)
    F_ren = dino.features(rgb)
    P = (surf * scale) @ R.T + T
    uv = project(P, of.K)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < CROP_RES - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < CROP_RES - 1)
    iu, iv = uv[:, 0].long().clamp(0, CROP_RES - 1), uv[:, 1].long().clamp(0, CROP_RES - 1)
    visible = inb & (P[:, 2] <= z[iv, iu] + 5e-3)
    idx = torch.nonzero(visible).view(-1)
    if len(idx) == 0:
        return None
    idx = idx[torch.randperm(len(idx), device=idx.device)[:N_POINTS]]
    f = sample_features(F_ren, uv[idx], CROP_RES)                     # (Np,C)
    S = torch.einsum("nc,hwc->nhw", f, of.F_gt)                        # cosine similarity maps
    return surf[idx], S


def object_loss(of, renderer, R, T, scale, dino_state, hand_now, hand_prev_local, w_int, sdf, lam):
    terms = {}
    loss = torch.zeros((), device=of.dev)
    if of.observed:
        A = renderer.silhouette(R, T, scale, of.K, CROP_RES)
        terms["mask"] = mask_loss(A, of.mask_gt)                          # Eq. 3
        loss = loss + W_MASK * terms["mask"]
        if dino_state is not None:
            pts, S = dino_state
            u = project((pts * scale) @ R.T + T, of.K)                  # u_i = Pi(R(s*p)+T)
            iu = u[:, 0].detach().long().clamp(0, CROP_RES - 1)
            iv = u[:, 1].detach().long().clamp(0, CROP_RES - 1)
            m_occ = 1.0 - of.occ[iv, iu]                                 # 0 where occluded
            terms["dino"] = -(m_occ * sample_maps(S, u, CROP_RES)).sum() / max(len(u), 1)   # Eq. 4
            loss = loss + W_DINO * terms["dino"]
    if hand_now is not None and hand_prev_local is not None and w_int is not None:
        local_now = (hand_now - T) @ R                                   # Eq. 5: R^T (v - T)
        terms["interact"] = (w_int * (local_now - hand_prev_local).norm(dim=-1)).sum() / len(w_int)  # Eq. 6
        loss = loss + lam * terms["interact"]
    return loss, terms


def interaction_weights(hand_prev, R_prev, T_prev, sdf, scale):
    """w_i = 1 - tanh(sigma * max(0, SDF(v~_{t-1}))), zero beyond D_MAX (Appendix A)."""
    local_prev = (hand_prev - T_prev) @ R_prev
    d = sdf(local_prev, scale).clamp(min=0)
    w = 1.0 - torch.tanh(SIGMA * d)
    return local_prev, torch.where(d <= D_MAX, w, torch.zeros_like(w))


def contact_loss(hand_now, R, T, scale, sdf):
    """CHOICE (the paper names a 'ray-traced contact loss' without defining it):
    penetration depth of hand vertices inside the object (sign from ray-based
    containment in the SDF) plus attraction of the near-contact vertices (< D_MAX)."""
    d = sdf((hand_now - T) @ R, scale)
    pen = (-d).clamp(min=0)
    near = (d > 0) & (d < D_MAX)
    attract = d[near].mean() if near.any() else d.new_zeros(())
    return pen.mean() + attract


def optimise_object(of, renderer, dino, surf, R0, T0, scale, hands_now, prev_local, w_int, sdf, lam, prior=None):
    """Step 2: object pose (delta rotation vector + translation), fixed hands.
    `prior` (hand coupling): dict(R=, T=, w=) adds w * prior_loss to L_obj."""
    w = torch.zeros(3, device=of.dev, requires_grad=True)
    T = torch.tensor(T0, device=of.dev, dtype=torch.float32, requires_grad=True)
    R0_t = torch.tensor(R0, device=of.dev, dtype=torch.float32)
    opt = torch.optim.Adam([{"params": [w], "lr": LR_ROT}, {"params": [T], "lr": LR_TRANS}])
    conv, dino_state, terms = Converged(), None, {}
    if not of.observed and w_int is None and prior is None:
        return R0, T0, {"skipped": "unobserved, no contact"}
    for it in range(IT_OBJ):
        R = rotvec_to_matrix(w) @ R0_t
        if of.F_gt is not None and it % RESAMPLE_EVERY == 0:
            dino_state = sample_dino_points(of, renderer, dino, surf, R.detach(), T.detach(), scale)
        loss, terms = object_loss(of, renderer, R, T, scale, dino_state, hands_now, prev_local, w_int, sdf, lam)
        if prior is not None:
            terms["prior"] = prior_loss(R, T, prior["R"], prior["T"])
            loss = loss + prior["w"] * terms["prior"]
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()
        if conv([w, T]):
            break
    R = (rotvec_to_matrix(w) @ R0_t).detach().cpu().numpy()
    return R, T.detach().cpu().numpy(), {k: float(v) for k, v in terms.items()} | {"iters": it + 1}


def optimise_scale_at_iof(of, renderer, R0, T0, hands_now, sdf, dev):
    """IOF only: anisotropic scale with rotation fixed and translation constrained to
    the viewing ray through the object centre; L = 5 L_mask + 5 L_contact."""
    R = torch.tensor(R0, device=dev, dtype=torch.float32)
    ray = torch.tensor(T0 / np.linalg.norm(T0), device=dev, dtype=torch.float32)
    depth = torch.tensor(float(np.linalg.norm(T0)), device=dev, requires_grad=True)
    log_s = torch.zeros(3, device=dev, requires_grad=True)
    opt, conv = torch.optim.Adam([{"params": [log_s], "lr": LR_TRANS}, {"params": [depth], "lr": LR_TRANS}]), Converged()
    for it in range(IT_OBJ):
        s, T = log_s.exp(), depth * ray
        A = renderer.silhouette(R, T, s, of.K, CROP_RES)
        loss = W_MASK * mask_loss(A, of.mask_gt)
        if hands_now is not None:
            loss = loss + W_CONTACT * contact_loss(hands_now, R, T, s, sdf)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if conv([log_s, depth]):
            break
    return log_s.exp().detach(), (depth * ray).detach().cpu().numpy(), it + 1


# ---------------------------------------------- hand coupling (our extension, --hand_coupling)
# Objects only move when a touching hand moves. Under hand occlusion the visual terms
# (and FoundationPose-style evidence) produce spurious object motion, so each frame's
# object pose gets a prior predicted from the closest touching hand's 6-DoF motion in
# the world frame (SLAM), plus a filter rejecting drastic unexplained jumps.
CONTACT_D = 0.015                               # hand within 1.5 cm of the surface = touching
HOLD_ROT, HOLD_TRANS = np.radians(5.0), 0.01    # hand "not moving" between frames
SIGMA_R, SIGMA_T = np.radians(5.0), 0.01        # prior scales
W_PRIOR, ALPHA_OCC = 1.0, 9.0                   # weight x (1 + alpha * occluded fraction)
REJECT_T, REJECT_R, REJECT_VIS = 0.03, np.radians(20.0), 0.5


def se3(R, T):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, T
    return M


def hand_pose_world(hands, side, t, cams):
    """Wrist 6-DoF in the world frame: rotation = MANO global orientation (mirrored
    for left hands), position = s_h * root joint + T_h; camera -> world via SLAM."""
    from scipy.spatial.transform import Rotation
    f = hands[side]["frames"][t]
    R = Rotation.from_rotvec(f["rec"]["mano"]["global_orient"]).as_matrix()
    if side == "left":
        M = np.diag([-1.0, 1.0, 1.0])
        R = M @ R @ M
    return cams[t] @ se3(R, hands[side]["s_h"] * f["j"][0] + f["T"])


def predict_object_pose(hands, t, p, R_p, T_p, cams, sdf, scale, dev):
    """Object pose at frame t predicted from frame p via the touching hand."""
    P_w = cams[p] @ se3(R_p, T_p)
    best = None
    for side in hands:
        if p in hands[side]["frames"]:
            v = torch.tensor(hands[side]["s_h"] * hands[side]["frames"][p]["v"] + hands[side]["frames"][p]["T"],
                             device=dev, dtype=torch.float32)
            local = (v - torch.tensor(T_p, device=dev, dtype=torch.float32)) @ torch.tensor(R_p, device=dev,
                                                                                             dtype=torch.float32)
            d = float(sdf(local, scale).min())
            if best is None or d < best[1]:
                best = (side, d)
    mode, P_w_pred = "static", P_w
    if best is not None and best[1] < CONTACT_D and t in hands[best[0]]["frames"]:
        Hp, Ht = hand_pose_world(hands, best[0], p, cams), hand_pose_world(hands, best[0], t, cams)
        dH = Ht @ np.linalg.inv(Hp)  # hand's world motion p -> t (rigid attachment)
        ang = np.arccos(np.clip((np.trace(dH[:3, :3]) - 1) / 2, -1, 1))
        wrist_move = np.linalg.norm(Ht[:3, 3] - Hp[:3, 3])
        if ang < HOLD_ROT and wrist_move < HOLD_TRANS:
            mode = f"hold({best[0]})"
        else:
            mode, P_w_pred = f"follow({best[0]})", dH @ P_w
    P_c = np.linalg.inv(cams[t]) @ P_w_pred
    return P_c[:3, :3], P_c[:3, 3], mode, (best[1] if best else None)


def visible_fraction(of, renderer, R, T, scale):
    """Observed share of the object's predicted silhouette (low = occluded)."""
    if not of.observed:
        return 0.0
    with torch.no_grad():
        sil = renderer.silhouette(torch.tensor(R, device=of.dev, dtype=torch.float32),
                                  torch.tensor(T, device=of.dev, dtype=torch.float32), scale, of.K, CROP_RES) > 0.5
    return float((sil & (of.mask_gt > 0.5)).sum() / sil.sum().clamp(min=1))


def prior_loss(R, T, R_pred, T_pred):
    """Chordal rotation distance (~2 theta^2) and translation, in prior units."""
    return ((R - R_pred) ** 2).sum() / (2 * SIGMA_R ** 2) + ((T - T_pred) ** 2).sum() / SIGMA_T ** 2


# ------------------------------------------------------------------------ pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--objects", default=None)
    ap.add_argument("--depth", choices=("moge", "stereo"), default="moge")
    ap.add_argument("--lambda_interact", type=float, default=400.0, help="paper: 400 HO3D / 200 DexYCB")
    # The paper optimises an anisotropic object scale at the IOF only because its
    # generated meshes have no metric scale. Our scans are metric (scale = 1), and on
    # ep 11 the IOF scale search slid along the scale/depth ambiguity (cup pushed from
    # ~0.5 m to 0.82 m with scale [0.97, 0.72, 1.12]), breaking hand contact. Off by default.
    ap.add_argument("--optimise_scale", action="store_true",
                    help="paper behaviour: anisotropic scale search at the IOF (for non-metric meshes)")
    ap.add_argument("--frame_window", type=int, default=None, help="only optimise IOF +- N frames (smoke tests)")
    ap.add_argument("--hand_coupling", action="store_true",
                    help="our extension: hand-motion prior + drastic-jump filter on object poses")
    ap.add_argument("--dino", default="auto",
                    help="weights dir; 'auto' = DINOv3 (paper) if downloaded, else DINOv2-L as interim")
    ap.add_argument("--out_tag", default=None, help="default: 'agile', or 'agile_dinov2' with the interim backbone")
    args = ap.parse_args()
    if args.dino == "auto":
        v3 = ROOT / "weights/dinov3/dinov3-vitl16-pretrain-lvd1689m"
        args.dino = str(v3 if (v3 / "config.json").exists() else ROOT / "weights/dinov2/dinov2-large")
    backbone = json.loads((Path(args.dino) / "config.json").read_text()).get("model_type", "?") \
        if (Path(args.dino) / "config.json").exists() else None
    if args.out_tag is None:
        args.out_tag = "agile" if backbone in ("dinov3_vit", "dinov3") else f"agile_{backbone or 'nodino'}"
        args.out_tag += ("_hc" if args.hand_coupling else "") + ("_optscale" if args.optimise_scale else "") + \
            ("_stereo" if args.depth == "stereo" else "")
    print(f"L_dino backbone: {backbone} ({args.dino}); outputs -> {args.out_tag}/", flush=True)
    dev = "cuda"
    torch.manual_seed(0)
    ep = ROOT / "runs" / args.run
    intr = json.loads((ep / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    K_t = torch.tensor(K, dtype=torch.float32, device=dev)
    shape = (intr["height"], intr["width"])
    n = len(list((ep / "recon_hawor/frames").glob("*.png")))
    depth_dir = ep / ("recon_hawor/depth" if args.depth == "moge" else "depth_stereo")
    out_root = ep / args.out_tag
    t_start = time.time()
    cams = {}  # DROID-SLAM camera-to-world per frame (hand coupling works in the world frame)
    for t in range(n):
        p = json.loads((ep / f"recon_hawor/slam_poses/{t:06d}.json").read_text())
        cams[t] = np.eye(4)
        cams[t][:3, :3], cams[t][:3, 3] = quat_wxyz_to_R(p["rotation"]), p["translation"]

    # §3.2 + §3.3 step 1 for hands (independent of the objects, so done once).
    mano = Mano(str(ROOT / "weights/hand"), dev)
    hands = init_hands(ep, mano, K, n, shape, depth_dir, dev)
    for side, h in hands.items():
        for t in sorted(h["frames"]):
            refine_hand_translation(h["frames"][t], h["s_h"], K_t, dev)
    dino = Dino(args.dino, dev) if Path(args.dino).exists() else None
    if dino is None:
        print(f"WARNING: DINOv3 weights not found at {args.dino}; L_dino disabled", flush=True)

    objs = [json.loads(p.read_text()) for p in sorted((ep / "agile").glob("*/prepare.json"))]
    if args.objects:
        objs = [o for o in objs if o["object"] in args.objects.split(",")]
    all_ids = [o["object_id"] for o in objs]
    results = {"hands": {s: {"s_h": h["s_h"], "n_scale_frames": h["n_scale_frames"]} for s, h in hands.items()}}
    for obj in objs:
        renderer = Renderer(obj["mesh"], dev)
        surf = torch.tensor(sample_surface(obj["mesh"]), device=dev)
        sdf = SDFGrid(trimesh.load(obj["mesh"], force="mesh"), device=dev)
        occluders = list(HANDS.values()) + [i for i in all_ids if i != obj["object_id"]]
        iof = obj["iof"]
        fp = json.loads(Path(obj["fp_pose_at_iof"]).read_text())
        R_iof, T_iof = quat_wxyz_to_R(fp["rotation"]), np.array(fp["translation"])
        scale = torch.ones(3, device=dev)
        poses, logs = {}, {}

        # IOF: scale (then frozen) and object pose.
        of = ObjectFrame(ep, iof, obj, K, shape, R_iof, T_iof, scale, renderer, dino, occluders, dev)
        h_iof = hand_vertices(hands, iof, dev)
        if args.optimise_scale and of.observed:
            scale, T_iof, it = optimise_scale_at_iof(of, renderer, R_iof, T_iof, h_iof, sdf, dev)
            print(f"{obj['object']}: IOF {iof} scale = {scale.cpu().numpy().round(3)} ({it} it)", flush=True)
        of = ObjectFrame(ep, iof, obj, K, shape, R_iof, T_iof, scale, renderer, dino, occluders, dev)
        R, T, logs[iof] = optimise_object(of, renderer, dino, surf, R_iof, T_iof, scale, None, None, None, sdf,
                                          args.lambda_interact)
        poses[iof] = (R, T)

        # Bidirectional propagation from the IOF.
        hi = n if args.frame_window is None else min(n, iof + args.frame_window + 1)
        lo = -1 if args.frame_window is None else max(-1, iof - args.frame_window - 1)
        for order in (range(iof + 1, hi), range(iof - 1, lo, -1)):
            prev_t = iof
            for t in order:
                R_p, T_p = poses[prev_t]
                both = common_sides(hands, prev_t, t)
                h_prev = hand_vertices(hands, prev_t, dev, both) if both else None
                h_now = hand_vertices(hands, t, dev, both) if both else None
                prev_local = w_int = None
                if h_prev is not None:
                    prev_local, w_int = interaction_weights(
                        h_prev, torch.tensor(R_p, device=dev, dtype=torch.float32),
                        torch.tensor(T_p, device=dev, dtype=torch.float32), sdf, scale)
                    if float(w_int.sum()) == 0:
                        prev_local = w_int = None
                prior, R0, T0 = None, R_p, T_p          # paper: initialise from the previous frame
                if args.hand_coupling:
                    R0, T0, mode, d_hand = predict_object_pose(hands, t, prev_t, R_p, T_p, cams, sdf, scale, dev)
                of = ObjectFrame(ep, t, obj, K, shape, R0, T0, scale, renderer, dino, occluders, dev)
                if args.hand_coupling:
                    vis = visible_fraction(of, renderer, R0, T0, scale)
                    prior = {"R": torch.tensor(R0, device=dev, dtype=torch.float32),
                             "T": torch.tensor(T0, device=dev, dtype=torch.float32),
                             "w": W_PRIOR * (1 + ALPHA_OCC * (1 - vis))}
                R, T, logs[t] = optimise_object(of, renderer, dino, surf, R0, T0, scale, h_now, prev_local,
                                                w_int, sdf, args.lambda_interact, prior)
                if args.hand_coupling:
                    dT = float(np.linalg.norm(T - T0))
                    dR = float(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)))
                    logs[t].update(mode=mode, d_hand=d_hand, visible=vis, prior_w=prior["w"], dT_pred=dT, dR_pred=dR)
                    if (dT > REJECT_T or dR > REJECT_R) and vis < REJECT_VIS:
                        R, T = R0, T0   # drastic jump the hand does not explain, object mostly unseen
                        logs[t]["rejected"] = True
                poses[t] = (R, T)
                prev_t = t
                if t % 25 == 0:
                    print(f"  {obj['object']} frame {t}: {logs[t]}", flush=True)

        od = out_root / obj["object"]
        (od / "poses").mkdir(parents=True, exist_ok=True)
        for t, (R, T) in poses.items():
            (od / f"poses/{t:06d}.json").write_text(json.dumps(
                {"R": R.tolist(), "T": T.tolist(), "scale": scale.cpu().numpy().tolist(), "log": logs[t]}))
        results[obj["object"]] = {"iof": iof, "iof_method": obj["iof_method"], "scale": scale.cpu().numpy().tolist(),
                                  "frames": len(poses)}

    hd = out_root / "hands"
    for side, h in hands.items():
        (hd / side).mkdir(parents=True, exist_ok=True)
        for t, f in h["frames"].items():
            (hd / side / f"{t:06d}.json").write_text(json.dumps(
                {"mano": f["rec"]["mano"], "is_right": side == "right", "s_h": h["s_h"], "T": f["T"].tolist(),
                 "L_joint": f.get("L_joint")}))
    results.update(depth=args.depth, lambda_interact=args.lambda_interact, optimise_scale=args.optimise_scale,
                   hand_coupling=args.hand_coupling,
                   dino_backbone=backbone if dino is not None else None, runtime_s=time.time() - t_start)
    (out_root / "summary.json").write_text(json.dumps(results, indent=2))
    # Per-job tag file (concurrent variants of one episode must not share it).
    Path(os.environ.get("AGILE_TAG_FILE", ep / "agile_last_tag.txt")).write_text(args.out_tag + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
