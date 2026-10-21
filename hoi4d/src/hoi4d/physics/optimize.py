"""README stage 9: physics- and contact-aware optimisation of one trajectory version -> a new version.

    L = w_visual L_visual + w_depth L_depth + w_temporal L_temporal + w_anatomy L_anatomy + w_contact L_contact
      + w_collision L_collision + w_dynamics L_dynamics + w_task L_task + w_prior L_prior

Variables (all temporally smooth: cubic B-spline knots every `knot_every` frames -- user ruling, no per-frame
corrections): per object a world-frame SE(3) correction about its centroid; per hand a world-frame rigid wrist
correction plus a MANO finger-pose correction. Corrections start at zero (the input version).

Terms
  visual     object: projected surface samples outside the (object | hand) mask region (distance transform),
             CoTracker rigidity (tracked surface points fixed in the object frame); hand: observed joints stay at
             their observed positions (detected frames kept close to the observations)
  depth      visible object surface vs FoundationStereo depth inside the object mask (robust)
  temporal   acceleration of the corrected trajectories (Huber: large real impacts are not smoothed away)
  anatomy    MANO joint-rotation bound, finger-finger interpenetration
  contact    selected grasp contacts: contact points on the surface and fixed in the object frame (no slip);
             contact_off proposals push apart; support: resting objects touch the table
  collision  cavity-aware: hand verts / other objects' surface into convex parts; objects into the table top
  dynamics   no hand + no support -> ballistic (a = g); supported + not held -> static (v = 0)
  task       weak: VLM 'place'/'release' phase ends with the moved object supported (only if already near)
  prior      correction magnitudes (sigma 1 cm / 3 deg objects, 5 mm / 3 deg wrists, 0.1 rad fingers)
Staged solve: geometric (visual, depth, temporal, anatomy, prior) -> contact selection (keep a hypothesised
contact only if the geometric solution supports it) -> physical (all terms). Unilateral contact = hinge losses;
static equilibrium only in supported-and-not-held phases. Mass/friction are not optimised here (they are not
identifiable from video); feasibility over ranges is reported by physics.wrench.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .. import conventions as C
from .. import io as hio
from ..diagnostics import run as drun
from ..export import versions
from ..geometry.assets import asset
from ..tracking import cotracker

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DS = 4  # image downsampling for masks / depth


# ------------------------------------------------------------------ small torch SE(3) helpers
def rodrigues(r):
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = r / th
    K = torch.zeros(*r.shape[:-1], 3, 3, device=r.device, dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=r.device, dtype=r.dtype) + s * K + (1 - c) * (K @ K)


def bspline_basis(F, every):
    """(F, K) cubic B-spline basis with knots every `every` frames (clamped at the ends)."""
    K = int(np.ceil((F - 1) / every)) + 3
    u = np.arange(F) / every
    B = np.zeros((F, K))
    for i in range(F):
        j = int(np.floor(u[i]))
        t = u[i] - j
        w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
        for k in range(4):
            B[i, min(j + k, K - 1)] += w[k]
    return torch.tensor(B, dtype=torch.float32, device=DEV)


def huber(x, d):
    a = x.abs()
    return torch.where(a < d, 0.5 * a ** 2 / d, a - 0.5 * d)


# ------------------------------------------------------------------ evidence loading
class Evidence:
    def __init__(self, run, man, tr):
        R = hio.ROOT / "runs" / run
        self.K = torch.tensor(np.asarray(man["cameras"]["ego_cam_a"]["K"], float) / [[DS, 1, DS], [1, DS, DS], [1, 1, 1]],
                              dtype=torch.float32, device=DEV)
        W, H = man["cameras"]["ego_cam_a"]["size"]
        self.w, self.h = W // DS, H // DS
        F = tr.F
        ents = {e["id"]: e for e in man["entities"]}
        self.allowed_dt, self.depth, self.objmask = {}, None, {}
        hand_m = np.zeros((F, self.h, self.w), bool)
        for side, mid in (("left", 2), ("right", 3)):
            for f in range(F):
                p = R / "objects_sam3/masks" / str(mid) / f"{f:06d}.png"
                if p.exists():
                    hand_m[f] |= self._load_mask(p)
        self.hand_mask = torch.tensor(hand_m, device=DEV)
        for n in tr.objects:
            mid = ents[n].get("mask_id")
            m = np.zeros((F, self.h, self.w), bool)
            have = np.zeros(F, bool)
            for f in range(F):
                p = R / "objects_sam3/masks" / str(mid) / f"{f:06d}.png"
                if mid is not None and p.exists():
                    m[f] = self._load_mask(p)
                    have[f] = m[f].sum() > 20
            dt = np.zeros((F, self.h, self.w), np.float32)
            for f in np.where(have)[0]:
                dt[f] = distance_transform_edt(~(m[f] | hand_m[f]))  # px outside the allowed region
            self.allowed_dt[n] = (torch.tensor(dt, device=DEV), torch.tensor(have, device=DEV))
            self.objmask[n] = torch.tensor(m, device=DEV)
        d = np.zeros((F, self.h, self.w), np.float32)
        for f in range(F):
            p = R / "depth_stereo" / f"{f:06d}.png"
            if p.exists():
                z = cotracker.decode_depth(np.asarray(Image.open(p)))[::DS, ::DS][: self.h, : self.w]
                d[f] = np.nan_to_num(z, nan=0.0)
        self.depth = torch.tensor(d, device=DEV)

    def _load_mask(self, p):
        return (np.asarray(Image.open(p))[::DS, ::DS] > 0)[: self.h, : self.w]

    def sample(self, img, f_idx, uv):
        """Bilinear lookup of per-frame images (F,h,w) at uv (N,P,2) for frames f_idx (N,)."""
        g = torch.stack([uv[..., 0] / (self.w - 1) * 2 - 1, uv[..., 1] / (self.h - 1) * 2 - 1], -1)
        return torch.nn.functional.grid_sample(img[f_idx][:, None].float(), g[:, :, None], align_corners=True)[:, 0, :, 0]


# ------------------------------------------------------------------ MANO
def load_mano():
    sys.path.insert(0, "/code/agile")
    from geometry import Mano
    return Mano(str(hio.ROOT / "weights/hand"), DEV)


def hand_forward(mano, go, hp, betas, T, is_right):
    """Camera-frame verts/joints, pipeline convention (left = mirrored right-hand params)."""
    out = mano.layer(torch.cat([go, hp], 1), betas.expand(len(go), -1))
    v, j = out.verts, out.joints
    if not is_right:
        flip = torch.tensor([-1.0, 1.0, 1.0], device=DEV)
        v, j = v * flip, j * flip
    return v + T[:, None], j + T[:, None]


# ------------------------------------------------------------------ main
def run(run_name, version, cfg, proposals, name=None, accept_registry=True, iters=(300, 0, 400)):
    reg = versions.load(run_name)
    tr = hio.Trajectories.load(Path(reg[version]["npz"]))
    man = json.loads((hio.out_dir(run_name) / "scene_manifest.json").read_text())
    W = cfg["physics"]["weights"]
    th = cfg["diagnostics"]
    F = tr.F
    t = torch.tensor(tr.t_s, dtype=torch.float32, device=DEV)
    Twc_np = tr.extra["T_world_cam"].copy()
    Twc_np[~np.isfinite(Twc_np).all(axis=(1, 2))] = np.eye(4)  # masked placeholder (never written out)
    T_wc = torch.tensor(Twc_np, dtype=torch.float32, device=DEV)
    T_cw = torch.linalg.inv(T_wc)
    B = bspline_basis(F, cfg["physics"]["knot_every"])
    ev = Evidence(run_name, man, tr)
    tracks = cotracker.load_tracks(run_name)
    diag = drun.run(run_name, version, tr, cfg, reg[version]["objects_variant"], reg[version]["hands_variant"])
    task_p = hio.out_dir(run_name) / "task.json"
    task = json.loads(task_p.read_text()) if task_p.exists() else None

    # optimisation window (proposals) or whole sequence; knots outside are frozen at zero
    win = np.ones(F, bool)
    if proposals:
        win[:] = False
        for p in proposals:
            a, b = p["frames"]
            win[max(0, a - 6): min(F, b + 7)] = True
    knot_on = (B[torch.tensor(win, device=DEV)].sum(0) > 1e-6).float()[:, None]

    # ---------------- objects
    objs = list(tr.objects)
    A = {n: asset(n) for n in objs}
    T0 = {}
    for n in objs:
        x = tr.objects[n]["T_world_obj"].copy()
        x[~tr.objects[n]["valid"]] = np.eye(4)  # masked placeholder (invalid frames are restored to NaN on output)
        T0[n] = torch.tensor(x, dtype=torch.float32, device=DEV)
    ov = {n: torch.tensor(tr.objects[n]["valid"], device=DEV) for n in objs}
    surf = {n: torch.tensor(A[n].surface[::2], dtype=torch.float32, device=DEV) for n in objs}
    cen = {n: torch.tensor(A[n].centroid, dtype=torch.float32, device=DEV) for n in objs}
    parts = {n: [(torch.tensor(a, dtype=torch.float32, device=DEV), torch.tensor(b, dtype=torch.float32, device=DEV))
                 for a, b in A[n].parts] for n in objs}
    dO = {n: torch.zeros(B.shape[1], 6, device=DEV, requires_grad=True) for n in objs}

    def obj_T(n):
        d = B @ (dO[n] * knot_on)
        Rd = rodrigues(d[:, :3])
        T = T0[n].clone()
        c = (T[:, :3, :3] @ cen[n]) + T[:, :3, 3]
        R = Rd @ T[:, :3, :3]
        p = torch.einsum("fij,fj->fi", Rd, T[:, :3, 3] - c) + c + d[:, 3:]
        out = torch.zeros_like(T)
        out[:, :3, :3], out[:, :3, 3], out[:, 3, 3] = R, p, 1
        return out

    # ---------------- hands
    mano = load_mano() if tr.hands else None
    H = {}
    spin_rep = {}
    for h, hd in tr.hands.items():
        valid = torch.tensor(hd["valid"], device=DEV)
        # fit flips (fast wrist rotation the hand's tracked surface points do not show): re-initialise those frames
        # and let the optimizer solve them from tracks + smoothness instead of pinning them to the flipped fit
        ht = hand_track_points(tracks, h)
        rmask, rlog = spin_repair(hd, ht, 1.0 / float(np.median(np.diff(tr.t_s))))
        go_np, hp_np, T_np = spin_reinit(hd, rmask) if rmask.any() else (hd["global_orient"], hd["hand_pose"], hd["transl"])
        spin_rep[h] = {"repaired_frames": int(rmask.sum()), "suspect_pairs": rlog[:40], "track_points": 0 if ht is None else len(ht["joint"])}
        H[h] = {"valid": valid, "is_right": h == "hand_right", "repaired": rmask,
                "go": torch.tensor(np.nan_to_num(go_np), dtype=torch.float32, device=DEV),
                "hp": torch.tensor(np.nan_to_num(hp_np), dtype=torch.float32, device=DEV),
                "betas": torch.tensor(hd["betas"], dtype=torch.float32, device=DEV)[None],
                "T": torch.tensor(np.nan_to_num(T_np), dtype=torch.float32, device=DEV),
                "obs": torch.tensor((hd["source"] == C.SOURCE_IDS["observed"]) & ~rmask, device=DEV),
                "dR": torch.zeros(B.shape[1], 6, device=DEV, requires_grad=True),
                "dP": torch.zeros(B.shape[1], 45, device=DEV, requires_grad=True)}
        if ht is not None:  # TrackEverything hand-joint tracks (offset-free short-window displacement term)
            H[h]["tP"] = torch.tensor(np.nan_to_num(ht["P"]), dtype=torch.float32, device=DEV)
            H[h]["tV"] = torch.tensor(ht["V"], device=DEV)
            H[h]["tJ"] = torch.tensor(ht["joint"], device=DEV)
        with torch.no_grad():
            v, j = hand_forward(mano, H[h]["go"], H[h]["hp"], H[h]["betas"], H[h]["T"], H[h]["is_right"])
            H[h]["J0w"] = torch.einsum("fij,fpj->fpi", T_wc[:, :3, :3], j) + T_wc[:, None, :3, 3]

    def hand_world(h):
        d = H[h]
        dp = B @ (d["dP"] * knot_on)
        v, j = hand_forward(mano, d["go"], d["hp"] + dp, d["betas"], d["T"], d["is_right"])
        vw = torch.einsum("fij,fpj->fpi", T_wc[:, :3, :3], v) + T_wc[:, None, :3, 3]
        jw = torch.einsum("fij,fpj->fpi", T_wc[:, :3, :3], j) + T_wc[:, None, :3, 3]
        r = B @ (d["dR"] * knot_on)
        Rd = rodrigues(r[:, :3])
        w = jw[:, :1]
        jw = torch.einsum("fij,fpj->fpi", Rd, jw - w) + w + r[:, None, 3:]
        vw = torch.einsum("fij,fpj->fpi", Rd, vw - w) + w + r[:, None, 3:]
        return vw, jw, dp

    # ---------------- contact hypotheses (stage 2 selects among these)
    hyp = []
    for c in diag["contacts"]:
        if c["mode"] in ("grasp", "support"):
            hyp.append({"a": c["entity_a"], "b": c["entity_b"], "mode": c["mode"], "frames": c["frames"],
                        "source": "geometry (diagnostics)"})
    for p in proposals or []:
        for c in p.get("constraints", []):
            hyp.append({"a": c["entity_a"], "b": c.get("entity_b"), "mode": c["kind"], "frames": p["frames"],
                        "source": "review agent proposal: " + p.get("evidence", "")[:200]})
    if task:
        for c in task.get("contact_hypotheses", []):
            fa = int(np.argmin(np.abs(tr.t_s - c["t_start_s"])))
            fb = int(np.argmin(np.abs(tr.t_s - c["t_end_s"])))
            hyp.append({"a": c["entity_a"], "b": c["entity_b"], "mode": c["mode"], "frames": [fa, fb],
                        "source": "VLM (soft)", "soft": True})
    active = []

    tr_idx = None
    if tracks is not None:
        names = [str(x) for x in tracks["names"]]
        tr_idx = {}
        for k, n in enumerate(names):
            if n in objs:
                idx = np.where(tracks["owner"] == k)[0]
                Pw_np = tracks["P_world"][:, idx].astype(np.float32)
                fin = torch.tensor(np.isfinite(Pw_np).all(-1), device=DEV)
                Pw = torch.tensor(np.nan_to_num(Pw_np, nan=0.0), dtype=torch.float32, device=DEV)
                q = torch.tensor(tracks["queries"][idx, 0].astype(int), device=DEV)
                axc = cotracker.sym_axis(run_name, n)
                if axc is not None:
                    axc = (torch.tensor(axc[0], dtype=torch.float32, device=DEV), torch.tensor(axc[1], dtype=torch.float32, device=DEV))
                tr_idx[n] = (Pw, q, axc, fin)

    def terms(stage):
        L = {k: torch.zeros((), device=DEV) for k in C.LOSS_TERMS}
        Tn = {n: obj_T(n) for n in objs}
        for n in objs:
            T = Tn[n]
            v = ov[n]
            P = torch.einsum("fij,pj->fpi", T[:, :3, :3], surf[n]) + T[:, None, :3, 3]
            Pc = torch.einsum("fij,fpj->fpi", T_cw[:, :3, :3], P) + T_cw[:, None, :3, 3]
            z = Pc[..., 2].clamp(min=0.05)
            uv = Pc[..., :2] / z[..., None] * torch.stack([ev.K[0, 0], ev.K[1, 1]]) + torch.stack([ev.K[0, 2], ev.K[1, 2]])
            dt, have = ev.allowed_dt[n]
            fr = torch.where(v & have)[0]
            if len(fr):
                inimg = (uv[fr, :, 0] > 0) & (uv[fr, :, 0] < ev.w - 1) & (uv[fr, :, 1] > 0) & (uv[fr, :, 1] < ev.h - 1)
                dpx = ev.sample(dt, fr, uv[fr])
                L["visual"] = L["visual"] + (huber(dpx, 3.0) * inimg).sum() / inimg.sum().clamp(min=1) / 10
                # depth: facing-camera samples inside the object mask with valid stereo depth
                nrm = torch.einsum("fij,pj->fpi", T[:, :3, :3], torch.tensor(asset_normals(n), device=DEV)[::2])
                facing = (nrm * (T_wc[:, None, :3, 3] - P)).sum(-1) > 0
                zo = ev.sample(ev.depth, fr, uv[fr])
                mk = ev.sample(ev.objmask[n].float(), fr, uv[fr]) > 0.5
                ok = inimg & facing[fr] & mk & (zo > 0.1)
                L["depth"] = L["depth"] + (huber(z[fr] - zo, 0.01) * ok).sum() / ok.sum().clamp(min=1) / 0.01
            if tr_idx is not None and n in tr_idx:
                Pw, q, axc, fin = tr_idx[n]
                Ti = torch.linalg.inv(T)
                Po = torch.einsum("fij,fpj->fpi", Ti[:, :3, :3], Pw) + Ti[:, None, :3, 3]
                if axc is not None:  # continuous symmetry: spin unobservable -> radius + height only
                    qq = Po - axc[1]
                    hh = qq @ axc[0]
                    rr = (qq - hh[..., None] * axc[0]).norm(dim=-1)
                    Po = torch.stack([rr, hh, torch.zeros_like(rr)], -1)
                ref = Po[q, torch.arange(len(q), device=DEV)]
                dd = (Po - ref[None]).norm(dim=-1)
                ar = torch.arange(len(q), device=DEV)
                okp = fin & fin[q, ar][None] & v[:, None] & v[q][None] & (dd.detach() < 0.05)
                dd = torch.where(okp, dd, torch.zeros_like(dd))
                L["visual"] = L["visual"] + (huber(dd, 0.005) * okp).sum() / okp.sum().clamp(min=1) / 0.005 * 0.5
            # temporal (Huber on acceleration: real impacts are not smoothed away)
            p = T[:, :3, 3]
            acc = (p[2:] - 2 * p[1:-1] + p[:-2]) / ((t[2:] - t[:-2]) / 2)[:, None] ** 2
            okv = v[2:] & v[1:-1] & v[:-2]
            L["temporal"] = L["temporal"] + (huber(acc.norm(dim=-1), 2.0) * okv).sum() / okv.sum().clamp(min=1)
            # table: no point below the top over the disk
            inside = P[..., :2].norm(dim=-1) < 0.2375
            L["collision"] = L["collision"] + ((torch.relu(-P[..., 2]) * inside).amax(1) * v).sum() / 0.002 / F
            d = B @ (dO[n] * knot_on)
            L["prior"] = L["prior"] + ((d[:, 3:] / 0.01) ** 2).sum(-1).mean() + ((d[:, :3] / np.deg2rad(3)) ** 2).sum(-1).mean()
        # object-object collision (cavity-aware parts)
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                for x, y in ((a, b), (b, a)):
                    Tx, Ty = Tn[x], Tn[y]
                    rel = torch.linalg.inv(Ty) @ Tx
                    P = torch.einsum("fij,pj->fpi", rel[:, :3, :3], surf[x][::3]) + rel[:, None, :3, 3]
                    L["collision"] = L["collision"] + (pen_parts(P, parts[y]).amax(1) * ov[x] * ov[y]).sum() / 0.002 / F
        # hands
        HW = {}
        for h in H:
            vw, jw, dp = hand_world(h)
            HW[h] = (vw, jw)
            d = H[h]
            okh = d["valid"]
            L["visual"] = L["visual"] + (huber((jw - d["J0w"]).norm(dim=-1), 0.005) * (d["obs"] & okh)[:, None]).sum() \
                / ((d["obs"] & okh).sum() * 21).clamp(min=1) / 0.005
            # anatomy / temporal over VALID frames only (invalid frames hold a zero placeholder at the camera origin)
            vf = okh.float()
            ang = (d["hp"] + dp).view(F, 15, 3).norm(dim=-1)
            L["anatomy"] = L["anatomy"] + ((torch.relu(ang - 2.0) ** 2).sum(-1) * vf).sum() / vf.sum().clamp(min=1) \
                + (finger_pen(jw) * vf).sum() / vf.sum().clamp(min=1) / 0.002
            w = jw[:, 0]
            acc = (w[2:] - 2 * w[1:-1] + w[:-2]) / ((t[2:] - t[:-2]) / 2)[:, None] ** 2
            ok3 = (okh[2:] & okh[1:-1] & okh[:-2]).float()
            L["temporal"] = L["temporal"] + (huber(acc.norm(dim=-1), 5.0) * ok3).sum() / ok3.sum().clamp(min=1) * 0.2
            # wrist rotation speed (joint frame): soft cap at 8 rad/s -- no spinning hands
            xa = jw[:, 5] - jw[:, 17]
            xa = xa / xa.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            za = torch.cross(xa, jw[:, 9] - jw[:, 0], dim=-1)
            za = za / za.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            Rh = torch.stack([xa, torch.cross(za, xa, dim=-1), za], -1)
            cs = ((Rh[:-1] * Rh[1:]).sum((-1, -2)) - 1) / 2
            spd = torch.acos(cs.clamp(-1 + 1e-6, 1 - 1e-6)) / (t[1:] - t[:-1])
            ok2 = (okh[1:] & okh[:-1]).float()
            L["temporal"] = L["temporal"] + ((torch.relu(spd - 8.0) ** 2) * ok2).sum() / ok2.sum().clamp(min=1) * 0.05
            # TrackEverything hand tracks: each joint moves like its tracked surface point over 1-3 frames
            # (offset-free: the joint-to-surface offset barely rotates within 3 frames; longer lags drift ~1 cm)
            if "tP" in d:
                jt = jw[:, d["tJ"]]  # (F, N, 3)
                for lag in (1, 2, 3):
                    okl = d["tV"][lag:] & d["tV"][:-lag] & (okh[lag:] & okh[:-lag])[:, None]
                    res = ((jt[lag:] - jt[:-lag]) - (d["tP"][lag:] - d["tP"][:-lag])).norm(dim=-1)
                    L["visual"] = L["visual"] + (huber(res, 0.003) * okl).sum() / okl.sum().clamp(min=1) / 0.003 * 0.3
            r = B @ (d["dR"] * knot_on)
            L["prior"] = L["prior"] + ((r[:, 3:] / 0.005) ** 2).sum(-1).mean() + ((r[:, :3] / np.deg2rad(3)) ** 2).sum(-1).mean() \
                + ((dp / 0.1) ** 2).mean()
            for n in objs:
                Ti = torch.linalg.inv(Tn[n])
                Vo = torch.einsum("fij,fpj->fpi", Ti[:, :3, :3], vw[:, ::4]) + Ti[:, None, :3, 3]
                L["collision"] = L["collision"] + (pen_parts(Vo, parts[n]).amax(1) * ov[n] * okh).sum() / 0.002 / F
        if stage != "physical":
            return L, Tn, HW
        # contact (selected hypotheses), dynamics, task
        held = {n: torch.zeros(F, dtype=torch.bool, device=DEV) for n in objs}
        supp = {n: torch.zeros(F, dtype=torch.bool, device=DEV) for n in objs}
        for c in active:
            a, b, mode = c["a"], c["b"], c["mode"]
            fa, fb = c["frames"]
            sl = slice(fa, fb + 1)
            if mode == "grasp" and a in HW and b in Tn:
                vw = HW[a][0][sl, c["verts"]]
                Ti = torch.linalg.inv(Tn[b][sl])
                Vo = torch.einsum("fij,fpj->fpi", Ti[:, :3, :3], vw) + Ti[:, None, :3, 3]
                anchor = torch.tensor(c["anchor_obj"], dtype=torch.float32, device=DEV)
                L["contact"] = L["contact"] + huber((Vo - anchor[None]).norm(dim=-1), 0.004).mean() / 0.004
                held[b][sl] = True
            elif mode == "contact_off" and a in HW and b in Tn:
                Ti = torch.linalg.inv(Tn[b][sl])
                Vo = torch.einsum("fij,fpj->fpi", Ti[:, :3, :3], HW[a][0][sl, ::8]) + Ti[:, None, :3, 3]
                dmin = torch.cdist(Vo, surf[b][None].expand(len(Vo), -1, -1)).amin(-1).amin(-1)
                L["contact"] = L["contact"] + torch.relu(th["contact_m"] - dmin).mean() / 0.004
            elif mode in ("support", "static") and a in Tn:
                supp[a][sl] = True
                if b == "table" or (mode == "support" and b is None):
                    P = torch.einsum("fij,pj->fpi", Tn[a][sl, :3, :3], surf[a]) + Tn[a][sl, None, :3, 3]
                    L["contact"] = L["contact"] + huber(P[..., 2].amin(-1), 0.003).mean() / 0.003
        for n in objs:
            p = Tn[n][:, :3, 3]
            vel = (p[2:] - p[:-2]) / (t[2:] - t[:-2])[:, None]
            acc = (p[2:] - 2 * p[1:-1] + p[:-2]) / ((t[2:] - t[:-2]) / 2)[:, None] ** 2
            free = ~held[n][1:-1] & ~supp[n][1:-1] & ov[n][1:-1]
            rest = supp[n][1:-1] & ~held[n][1:-1] & ov[n][1:-1]
            g = torch.tensor([0, 0, -9.81], device=DEV)
            L["dynamics"] = L["dynamics"] + (huber((acc - g).norm(dim=-1), 2.0) * free).sum() / free.sum().clamp(min=1) \
                + (huber(vel.norm(dim=-1), 0.01) * rest).sum() / rest.sum().clamp(min=1) / 0.01
        if task:
            for ph in task.get("phases", []):
                if ph["name"] in ("place", "release"):
                    f = int(np.argmin(np.abs(tr.t_s - ph["t_end_s"])))
                    for n in ph.get("entities", []):
                        if n in Tn:
                            P = Tn[n][f, :3, :3] @ surf[n].T + Tn[n][f, :3, 3:4]
                            gap = P[2].min()
                            L["task"] = L["task"] + torch.where(gap < 0.02, huber(gap, 0.003) / 0.003, torch.zeros_like(gap))
        return L, Tn, HW

    def solve(stage, n_it, lr=0.01):
        params = [dO[n] for n in objs] + [H[h]["dR"] for h in H] + [H[h]["dP"] for h in H]
        if n_it <= 0 or not params:
            return
        opt = torch.optim.Adam(params, lr=lr)
        wts = dict(W)
        if stage == "geometric":
            for k in ("contact", "dynamics", "task"):
                wts[k] = 0.0
            wts["collision"] *= 0.25
        for it in range(n_it):
            opt.zero_grad()
            L, _, _ = terms(stage)
            tot = sum(wts[k] * L[k] for k in C.LOSS_TERMS)
            if not torch.isfinite(tot):
                raise FloatingPointError(f"{stage} it {it}: non-finite loss " + str({k: float(v) for k, v in L.items()}))
            tot.backward()
            for p_ in params:
                if p_.grad is not None and not torch.isfinite(p_.grad).all():
                    raise FloatingPointError(f"{stage} it {it}: non-finite gradient")
            opt.step()

    def report(stage):
        with torch.no_grad():
            L, _, _ = terms(stage)
        return {k: float(v) for k, v in L.items()}

    rep = {"before": report("physical"), "spin_repair": spin_rep}
    solve("geometric", iters[0])
    rep["after_geometric"] = report("geometric")
    # stage 2: contact selection -- keep a hypothesised grasp only if the geometric solution supports it
    with torch.no_grad():
        Tn = {n: obj_T(n) for n in objs}
        HW = {h: hand_world(h)[:2] for h in H}
    sel_log = []
    for c in hyp:
        a, b, mode = c["a"], c["b"], c["mode"]
        fa, fb = c["frames"]
        if not win[fa:fb + 1].any():
            continue
        if mode in ("grasp", "contact_on") and a in HW and b in Tn:
            with torch.no_grad():
                vw = HW[a][0][fa:fb + 1]
                Ti = torch.linalg.inv(Tn[b][fa:fb + 1])
                Vo = torch.einsum("fij,fpj->fpi", Ti[:, :3, :3], vw) + Ti[:, None, :3, 3]
                d = torch.cdist(Vo, surf[b][None].expand(len(Vo), -1, -1)).amin(-1)  # (f, 778)
                med = d.median(0).values
                verts = torch.where(med < 0.012)[0]
            keep = len(verts) >= 8 and (not c.get("soft") or float(med.min()) < 0.02)
            if keep:
                anchor = Vo[:, verts].median(0).values.cpu().numpy()
                active.append({**c, "mode": "grasp", "verts": verts, "anchor_obj": anchor})
            sel_log.append({**{k: c[k] for k in ("a", "b", "mode", "frames", "source")}, "selected": bool(keep),
                            "n_contact_verts": int(len(verts)), "min_median_dist_m": float(med.min())})
        elif mode in ("support", "static", "contact_off"):
            active.append(c)
            sel_log.append({**{k: c[k] for k in ("a", "b", "mode", "frames", "source")}, "selected": True})
    rep["contact_selection"] = sel_log
    solve("physical", iters[2])
    rep["after_physical"] = report("physical")

    # ---------------- write the new version
    with torch.no_grad():
        Tn = {n: obj_T(n).cpu().numpy() for n in objs}
        HWn = {h: hand_world(h) for h in H}
    new = tr.copy()
    for n in objs:
        o = tr.objects[n]
        T = Tn[n].astype(float)
        T[~o["valid"]] = np.nan
        moved = np.linalg.norm(np.nan_to_num(T[:, :3, 3] - o["T_world_obj"][:, :3, 3]), axis=1) > 0.001
        src = o["source"].copy()
        src[moved & o["valid"]] = C.SOURCE_IDS["refined"]
        new.add_object(n, T, o["valid"], src, o["sigma_t_m"], o["sigma_r_rad"])
    for h in H:
        vw, jw, dp = HWn[h]
        hd = new.hands[h]
        vmask = hd["valid"]
        jn = jw.cpu().numpy().astype(float)
        hd["joints"][vmask] = jn[vmask]
        hd["hand_pose"][vmask] = (H[h]["hp"] + dp).cpu().numpy()[vmask]
        moved = np.linalg.norm(jn[:, 0] - tr.hands[h]["joints"][:, 0], axis=1) > 0.001
        hd["source"][moved & vmask] = C.SOURCE_IDS["refined"]
        # transl / global_orient in the camera frame: rigid world correction folded into the camera-frame params
        hd["transl"][vmask], hd["global_orient"][vmask] = fold_rigid(H[h], HWn[h], T_wc, mano, vmask)
        hd["repaired"] = H[h]["repaired"].copy()  # spin-repaired frames (excluded from the observed-frame 1 cm gate)
        hd["source"][H[h]["repaired"] & vmask] = C.SOURCE_IDS["refined"]
    name = name or f"phys_{len(reg)}"
    change = {"stage": "optimize", "proposals": proposals, "weights": W, "report": rep}
    versions.write_variant(run_name, name, new, version, change, reg[version]["hands_variant"])
    hio.write_json(hio.out_dir(run_name) / f"physics_report_{name}.json", rep)
    hio.write_artifact(run_name, "contacts", {"version": name, "contacts": [
        {"entity_a": c["a"], "entity_b": c["b"] or "table", "mode": "grasp" if c["mode"] == "grasp" else
         ("support" if c["mode"] in ("support", "static") else "none"),
         "t_start_s": float(tr.t_s[c["frames"][0]]), "t_end_s": float(tr.t_s[c["frames"][1]]),
         "frames": c["frames"], "region": {"n_contact_verts": int(len(c["verts"])) if "verts" in c else None},
         "evidence": c["source"], "confidence": 0.5 if c.get("soft") else 0.8} for c in active]},
        f"contacts_{name}.json")
    return name


_NORMALS = {}


def asset_normals(n):
    if n not in _NORMALS:
        from ..physics.wrench import surface_normals
        _NORMALS[n] = surface_normals(n).astype(np.float32)
    return _NORMALS[n]


def pen_parts(P, parts):
    """Differentiable penetration depth of points (..., P, 3) into a union of convex parts (max over parts)."""
    best = torch.zeros(P.shape[:-1], device=P.device)
    for n, d in parts:
        s = P @ n.T + d
        best = torch.maximum(best, torch.relu(-s.amax(-1)))
    return best


def finger_pen(jw, r=0.008):
    fingers = [[2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18, 19, 20]]
    pen = torch.zeros(jw.shape[0], device=jw.device)
    for i in range(5):
        for j in range(i + 1, 5):
            d = torch.cdist(jw[:, fingers[i]], jw[:, fingers[j]]).amin(dim=(-1, -2))
            pen = pen + torch.relu(r - d)
    return pen


def fold_rigid(Hd, HWn, T_wc, mano, vmask):
    """Camera-frame (transl, global_orient) whose MANO output matches the corrected world hand: the rigid world
    correction R_d about the wrist is conjugated into the camera frame (mirrored for left hands)."""
    from scipy.spatial.transform import Rotation as Rot
    vw, jw, dp = HWn
    Twc = T_wc.cpu().numpy()
    go = Hd["go"].cpu().numpy().copy()
    T = Hd["T"].cpu().numpy().copy()
    J0w = Hd["J0w"].cpu().numpy()
    jn = jw.cpu().numpy()
    M = np.diag([-1.0, 1, 1])
    for f in np.where(vmask)[0]:
        # rotation that best maps the uncorrected world joints (relative to wrist) onto the corrected ones
        a = J0w[f] - J0w[f, :1]
        b = jn[f] - jn[f, :1]
        U, _, Vt = np.linalg.svd(b.T @ a)
        Rd = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        Rc = Twc[f, :3, :3].T @ Rd @ Twc[f, :3, :3]
        Rp = Rc if Hd["is_right"] else M @ Rc @ M
        go[f] = (Rot.from_matrix(Rp) * Rot.from_rotvec(go[f])).as_rotvec()
    with torch.no_grad():
        _, j = hand_forward(mano, torch.tensor(go, dtype=torch.float32, device=DEV), Hd["hp"] + dp,
                            Hd["betas"], torch.zeros_like(Hd["T"]), Hd["is_right"])
    jc = np.einsum("fij,fpj->fpi", np.linalg.inv(Twc)[:, :3, :3], jn) + np.linalg.inv(Twc)[:, None, :3, 3]
    T = jc[:, 0] - j.cpu().numpy()[:, 0]
    return T[vmask], go[vmask]


def _hand_frame(J):
    """Wrist frame from joints (OpenPose order: 0 wrist, 5 index MCP, 9 middle MCP, 17 pinky MCP); (..., 3, 3)."""
    x = J[..., 5, :] - J[..., 17, :]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, J[..., 9, :] - J[..., 0, :])
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], -1)


def hand_track_points(tracks, h):
    """TrackEverything hand-joint tracks of one hand (tracks_te.npz has 'joint'; CoTracker files do not)."""
    if tracks is None or "joint" not in tracks or h not in [str(x) for x in tracks["names"]]:
        return None
    k = [str(x) for x in tracks["names"]].index(h)
    idx = np.where((tracks["owner"] == k) & (tracks["joint"] >= 0))[0]
    if len(idx) == 0:
        return None
    P = tracks["P_world"][:, idx].astype(float)
    V = tracks["vis"][:, idx] & np.isfinite(P).all(-1)
    return {"P": P, "V": V, "joint": tracks["joint"][idx].astype(int)}


def spin_repair(hd, ht, fps, fast=10.0, ratio=0.5, untracked_fast=20.0, merge_gap=8):
    """Frames where the fitted wrist turns fast but the hand's tracked surface points do not (fit flips).

    Pair (t, t+1) is suspect if the joint-frame rotation speed exceeds `fast` rad/s and the rotation implied by the
    TrackEverything hand points (Kabsch over >= 4 points visible in both frames) is below `ratio` of it; without
    track support only speeds above `untracked_fast` are suspect. Suspect pairs closer than `merge_gap` frames form
    one burst; bursts are dilated by one frame. Returns (mask, report). The masked frames are re-initialised by the
    caller (rotation slerp between the clean frames on either side: a starting point only) and then solved by the
    optimizer from tracks + smoothness instead of being pinned to the flipped fit."""
    from scipy.spatial.transform import Rotation as Rot
    J, valid = hd["joints"], hd["valid"] & np.isfinite(hd["joints"]).all((1, 2))
    F = len(J)
    R = np.full((F, 3, 3), np.nan)
    R[valid] = _hand_frame(J[valid])
    sus, rep = [], []
    for t in range(F - 1):
        if not (valid[t] and valid[t + 1]):
            continue
        s = np.linalg.norm(Rot.from_matrix(R[t].T @ R[t + 1]).as_rotvec()) * fps
        if s <= fast:
            continue
        a = np.nan
        if ht is not None:
            m = ht["V"][t] & ht["V"][t + 1]
            if m.sum() >= 4:
                A = ht["P"][t, m] - ht["P"][t, m].mean(0)
                B_ = ht["P"][t + 1, m] - ht["P"][t + 1, m].mean(0)
                U, _, Vt = np.linalg.svd(B_.T @ A)
                Rk = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                a = np.linalg.norm(Rot.from_matrix(Rk).as_rotvec()) * fps
        if (np.isfinite(a) and a < ratio * s) or (not np.isfinite(a) and s > untracked_fast):
            sus.append(t)
            rep.append({"frame": t, "fit_rad_s": round(float(s), 1), "track_rad_s": None if not np.isfinite(a) else round(float(a), 1)})
    mask = np.zeros(F, bool)
    if sus:
        segs = [[sus[0], sus[0] + 1]]
        for t in sus[1:]:
            if t - segs[-1][1] <= merge_gap:
                segs[-1][1] = t + 1
            else:
                segs.append([t, t + 1])
        for a, b in segs:
            mask[max(0, a - 1): min(F, b + 2)] = True
    return mask & valid, rep


def spin_reinit(hd, mask):
    """Starting point for repaired frames: slerp of global_orient / per-joint hand_pose and lerp of transl between
    the nearest clean valid frames (one-sided: the clean neighbour). Returns new (go, hp, T) arrays."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    go, hp, T = hd["global_orient"].copy(), hd["hand_pose"].copy(), hd["transl"].copy()
    clean = hd["valid"] & ~mask
    F = len(go)
    f = 0
    while f < F:
        if not mask[f]:
            f += 1
            continue
        e = f
        while e + 1 < F and mask[e + 1]:
            e += 1
        a = max([i for i in range(0, f) if clean[i]], default=None)
        b = min([i for i in range(e + 1, F) if clean[i]], default=None)
        for g in range(f, e + 1):
            if a is not None and b is not None:
                u = (g - a) / (b - a)
                go[g] = Slerp([0, 1], Rot.from_rotvec([go[a], go[b]]))(u).as_rotvec()
                hp[g] = np.concatenate([Slerp([0, 1], Rot.from_rotvec([hp[a, 3*k:3*k+3], hp[b, 3*k:3*k+3]]))(u).as_rotvec()
                                        for k in range(15)])
                T[g] = (1 - u) * T[a] + u * T[b]
            elif a is not None or b is not None:
                c = a if a is not None else b
                go[g], hp[g], T[g] = go[c], hp[c], T[c]
        f = e + 1
    return go, hp, T
