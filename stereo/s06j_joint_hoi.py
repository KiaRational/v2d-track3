"""Stage 6j: JOINT hand-object kinematic optimisation in the world frame (agile image).

Until now objects are fused first (fuse_objects) and hands refined against those FIXED objects
(s06h): one-way. Here one torch problem per episode solves both sides together:

  variables  per hand/frame: bounded corrections (wrist rot, 45-D fingers, T) from --ref_hands
             (mano_fit), started at --hands (mano_fit_phys), exactly like s06h's final pass;
             per object/frame: bounded world centroid shift + rotation delta on --init_objects
             (fused_all_sam3); per grasp segment (object, hand): offset G = object pose in the
             WRIST frame (mean, bounded) plus a small per-frame slip deviation.
  wrist      H_t = S_t [M R(go_t) M | joint0_t] (M = diag(-1,1,1) for left hands, which MANO
             mirrors): depends on the wrist parameters only (MCPs hang off the root joint).
  keypoints  object terms are metric: centroid + 6 points at +-5 cm along the object axes (as in
             eval_objects KP_RADIUS), so they sit next to s06h's metre^2 hand terms.
Objective (rho(s) = 2(sqrt(1+s)-1), soft-L1; per-frame means; default weights in --help):
  obs       w_obs  sum_m [ w_t s_t^2 rho(|c_t - c_m|^2/s_t^2) + w_r s_r^2 rho(min_g mean_k |R_t d_k - R_m g d_k|^2/s_r^2) ]
            FP (vis, vis^2) / stereo-ICP (w_icp fitness^2) exactly as fuse_objects, canonicalised
            (canon_measurements) and min over the discrete symmetry group; a continuous-symmetric
            object only uses its axis points (spin unobservable)
  couple    w_grasp sum_seg sum_t w_t s_g^2 rho(mean_k |O_t kp_k - H_t G_t kp_k|^2 / s_g^2), w_t = grasp
            confidence x 3-frame ramp at segment ends / number of hands coupled at t
  slip      G_t = G exp(xi_t): w_slip |xi|^2 + w_slip_vel |xi_{t+1} - xi_t|^2 (translation m, rotation x 5 cm)
  static    w_static sum_t s_t mean_k |K_{t+1} - K_t|^2, s_t = fuse_objects.damping_weights
  nohand    w_nohand rho-robust object velocity where no coupled segment is within 2 frames and no hand
            vertex is within --nohand_mm (recomputed from the current estimate at stage boundaries)
  obj_*     prior to the init track (keypoints; x (0.1 + 0.9 min(1, w_t)): an occluded init is a guess), world
            acceleration (motion_quality.acceleration_loss,
            fps-aware, not split: continuity across handovers), cubic B-spline (split at debounced
            coupling-phase changes, as fuse_objects), table (deepest object vertex in the tabletop slab - tol)
  hand_*    s06h's terms: pen (hand vertices in the object SDF at the CURRENT object pose, so
            both sides move), anchor (s06h contact anchors incl. per-grasp normalisation, in the current object
            frame, only frames with trusted stereo hand AND observed object, x frame confidence),
            table (motion_quality.table_slab_depth: the round tabletop slab of table_plane.py, not an infinite
            plane -- hands beside/under the table are free), data (ref joints; per frame x wdata = 0.25 + 0.75
            tri confidence), prior (x wdata),
            acceleration, spline -- plus tri: robust conf-weighted distance to hands_tri X_smooth (world)
Grasp phases: fusion_report hand_contact, debounced (3 frames) per object; after stage 1 re-detected from
the current estimate with hysteresis (enter: >= 3 vertices within 8 mm, exit: < 3 within 15 mm; a report grasp
grows <= 3 frames per end; report frames are dropped only where the object is SEEN and the hand > 40 mm away);
--redetect refine (default) only extends/trims report grasps, union also adds new ones. Handover: segments
of two hands on one object that abut (gap <= 2) are overlapped by 2 frames (cross-faded by the ramps); the
object trajectory is one variable sequence under an unsplit acceleration prior, so it cannot jump.
Schedule: stage 1 objects + G only (hands fixed), re-detect, stage 2 everything (Adam, cosine lr decay).
No 2D reprojection: wilor2d.json exists for few runs, and mano_fit already fits the 2D keypoints.
Outputs (never an input variant): <run>/<variant>/<obj>/poses/NNNNNN.json (camera frame, R/T),
<run>/stereo/<out>/<side>/NNNNNN.json + report.json (s06h format, args.variant = <variant>, s06h's robust
quality statistic), <run>/<variant>/joint_report.json (terms before/after, segments, slip, changes).
Usage: python s06j_joint_hoi.py <run> [--variant fused_joint] [--out mano_fit_joint]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano, SDFGrid, rotvec_to_matrix  # noqa: E402
from s06f_contact_refine import TIPS, load_T, mano_forward, matrix_to_rotvec  # noqa: E402
from s06h_grasp_plausible import FINGERS, load_segment_params, robust_stats, segment_anchor_params, segments, table_quality  # noqa: E402
from motion_quality import acceleration_loss, debounce_phases, spline_loss, spline_projectors, sticking_windows, table_slab_depth  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
KP_RADIUS = 0.05
MIRROR = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))


def rho(s):
    return 2 * (torch.sqrt(1 + s) - 1)


def bounded(x, limit):
    return limit * x / torch.sqrt(1 + (x * x).sum(-1, keepdim=True))


def unbounded(y, limit):
    nrm = y.norm(dim=-1, keepdim=True)
    y = y * torch.clamp(0.999 * limit / nrm.clamp(min=1e-12), max=1.0)
    return y / torch.sqrt(limit ** 2 - (y * y).sum(-1, keepdim=True))


def chordal_mean_t(R, w):
    U, _, Vt = torch.linalg.svd((w[:, None, None] * R).sum(0))
    d = torch.sign(torch.det(U @ Vt))
    return U @ torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d])) @ Vt


def hysteresis(enter, stay):
    on, out = False, np.zeros(len(enter), bool)
    for t in range(len(enter)):
        on = bool(enter[t]) if not on else bool(stay[t])
        out[t] = on
    return out


def runs_of(mask):
    idx = np.flatnonzero(mask)
    return [list(r) for r in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)] if len(idx) else []


def default_cfg(**kw):
    """argparse defaults as a namespace (tests and main share them)."""
    a = build_parser().parse_args(["_"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class Joint:
    """The joint problem on tensors. hands {side: {ts, is_right, ref {go,hp,be,T}, init {go,hp,T}|None,
    Sw (L,4,4), tri (L,21,3) world (nan = missing), conf (L,21)}}; objs [{name, W0 (n,4,4) world (nan =
    invalid), meas [(t, W, wt, wr)], sym {group, cont_axis, center}, sdf (torch local pts -> sdf),
    verts (V,3), static_w (n-1), grasp {side: frames}, radius}]; plane {up, height} | None.
    fk(go, hp, betas, is_right) -> local verts (L,778,3), joints (L,21,3) (camera frame minus T)."""

    def __init__(self, hands, objs, plane, fps, n, cfg, fk, dev, anchor_params=None, log=print):
        self.cfg, self.fk, self.dev, self.fps, self.n, self.log = cfg, fk, dev, float(fps), n, log
        self.anchor_params = anchor_params or (lambda side, name, seg: (cfg.squeeze_mm, cfg.contact_mm))
        f32 = dict(dtype=torch.float32, device=dev)
        self.f32 = f32
        self.up = torch.tensor(plane["up"], **f32) if plane else None
        self.table_h = float(plane["height"]) if plane else None
        self.plane = plane
        self.M = MIRROR.to(dev)
        self.freeze_hands = False
        self.H = {}
        for sd, h in hands.items():
            L = len(h["ts"])
            q = {k: torch.tensor(np.asarray(h["ref"][k]), **f32) for k in ("go", "hp", "be", "T")}
            e = dict(ts=list(h["ts"]), tix={t: i for i, t in enumerate(h["ts"])}, is_right=bool(h["is_right"]),
                     go0=q["go"], Rgo0=rotvec_to_matrix(q["go"]), hp0=q["hp"], be=q["be"], T0=q["T"],
                     Sw=torch.tensor(np.asarray(h["Sw"]), **f32))
            tri = np.asarray(h["tri"], float)
            ok = np.isfinite(tri).all(-1)
            conf = np.where(ok, np.asarray(h["conf"], float), 0.0)
            e["tri"], e["tri_ok"] = torch.tensor(np.nan_to_num(tri), **f32), torch.tensor(ok, **f32)
            e["conf"] = torch.tensor(conf, **f32)
            e["wdata"] = 0.25 + 0.75 * e["conf"].mean(1)  # a frame without stereo evidence is correctable
            kR, khp, kT = torch.zeros(L, 3, **f32), torch.zeros(L, 45, **f32), torch.zeros(L, 3, **f32)
            if h.get("init") is not None:  # start at --hands, corrections measured from --ref_hands
                go1 = torch.tensor(np.asarray(h["init"]["go"]), **f32)
                hp1 = torch.tensor(np.asarray(h["init"]["hp"]), **f32)
                T1 = torch.tensor(np.asarray(h["init"]["T"]), **f32)
                dR1 = matrix_to_rotvec(rotvec_to_matrix(go1) @ e["Rgo0"].transpose(-1, -2))
                kR = unbounded(dR1, math.radians(cfg.max_rotation_deg))
                khp = unbounded((hp1 - e["hp0"]).reshape(-1, 15, 3), math.radians(cfg.max_finger_deg)).reshape(-1, 45)
                kT = unbounded(T1 - e["T0"], cfg.max_shift_mm / 1000)
            e["kR"], e["khp"], e["kT"] = (x.detach().clone().requires_grad_(True) for x in (kR, khp, kT))
            self.H[sd] = e
        self.O = []
        for o in objs:
            W0 = np.asarray(o["W0"], float)
            valid = np.isfinite(W0[:, 0, 0])
            W0 = np.where(valid[:, None, None], np.nan_to_num(W0), np.eye(4))
            c = np.asarray(o["sym"].get("center", np.zeros(3)), float)
            ax = o["sym"].get("cont_axis")
            E = np.r_[np.eye(3), -np.eye(3)] * KP_RADIUS
            Dobs = np.array([ax, -np.asarray(ax)]) * KP_RADIUS / np.linalg.norm(ax) if ax is not None else E
            group = [np.eye(3)] if ax is not None else [np.asarray(g, float) for g in o["sym"].get("group", [np.eye(3)])]
            e = dict(name=o["name"], valid=torch.tensor(valid, device=dev), vidx=np.flatnonzero(valid),
                     R0=torch.tensor(W0[:, :3, :3], **f32), c=torch.tensor(c, **f32),
                     D=torch.tensor(np.r_[np.zeros((1, 3)), E], **f32), Dobs=torch.tensor(Dobs, **f32),
                     sdf=o["sdf"], radius=float(o["radius"]), grasp={k: list(v) for k, v in o.get("grasp", {}).items()},
                     carry={k: list(v) for k, v in o.get("carry", o.get("grasp", {})).items()},
                     verts=torch.tensor(np.asarray(o["verts"], float), **f32),
                     static_w=torch.tensor(np.asarray(o["static_w"], float), **f32))
            e["cw0"] = torch.tensor(W0[:, :3, :3] @ c + W0[:, :3, 3], **f32)
            m = [(int(t), np.asarray(W, float), float(wt), float(wr)) for t, W, wt, wr in o["meas"]
                 if 0 <= int(t) < n and valid[int(t)] and max(wt, wr) > 0]
            e["m_t"] = torch.tensor([q[0] for q in m], dtype=torch.long, device=dev)
            e["m_c"] = torch.tensor(np.array([q[1][:3, :3] @ c + q[1][:3, 3] for q in m]).reshape(-1, 3), **f32)
            mD = np.array([[q[1][:3, :3] @ g @ Dobs.T for g in group] for q in m]).reshape(len(m), len(group), 3, len(Dobs))
            e["m_D"] = torch.tensor(mD.transpose(0, 1, 3, 2), **f32)  # (M, group, K, 3): R_m g d_k
            e["m_wt"] = torch.tensor([q[2] for q in m], **f32)
            e["m_wr"] = torch.tensor([q[3] for q in m], **f32)
            rw = np.zeros(n)  # per-frame rotation trust of the measurements (G initialisation weight)
            for q in m:
                rw[q[0]] = max(rw[q[0]], q[3])
            e["rot_trust"] = torch.tensor(rw, **f32)
            pw = np.zeros(n)
            for q in m:
                pw[q[0]] = max(pw[q[0]], q[2])
            e["pos_trust"] = torch.tensor(pw, **f32)
            e["om"] = torch.zeros(n, 3, **f32).requires_grad_(True)
            e["dl"] = torch.zeros(n, 3, **f32).requires_grad_(True)
            self.O.append(e)
        with torch.no_grad():
            for e in self.O:
                e["K0"] = self.obj_forward(e)["K"].clone()
        self.segs = []
        self._init_anchors()

    # ---------------------------------------------------------------- forward models
    def hand_forward(self, sd):
        h, a = self.H[sd], self.cfg
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_hands):
            dR = bounded(h["kR"], math.radians(a.max_rotation_deg))
            dhp = bounded(h["khp"].reshape(-1, 15, 3), math.radians(a.max_finger_deg)).reshape(-1, 45)
            dT = bounded(h["kT"], a.max_shift_mm / 1000)
            R = rotvec_to_matrix(dR) @ h["Rgo0"]
            v, j = self.fk(matrix_to_rotvec(R), h["hp0"] + dhp, h["be"], h["is_right"])
            v, j = v + (h["T0"] + dT)[:, None], j + (h["T0"] + dT)[:, None]
            SR, Sp = h["Sw"][:, :3, :3], h["Sw"][:, None, :3, 3]
            vw, jw = torch.einsum("bij,bnj->bni", SR, v) + Sp, torch.einsum("bij,bnj->bni", SR, j) + Sp
            Rc = R if h["is_right"] else self.M @ R @ self.M
            return dict(dR=dR, dhp=dhp, dT=dT, R=R, j=j, vw=vw, jw=jw, HR=SR @ Rc, Hp=jw[:, 0])

    def obj_forward(self, e):
        a = self.cfg
        om = bounded(e["om"], math.radians(a.max_obj_rot_deg))
        cw = e["cw0"] + bounded(e["dl"], a.max_obj_shift_mm / 1000)
        R = rotvec_to_matrix(om) @ e["R0"]
        p = cw - R @ e["c"]
        return dict(R=R, p=p, cw=cw, K=cw[:, None] + torch.einsum("nij,kj->nki", R, e["D"]))

    def grasp_pose(self, g):
        a = self.cfg
        RG = rotvec_to_matrix(bounded(g["sR"], math.radians(a.max_slip_deg))) @ \
            (rotvec_to_matrix(bounded(g["gR"], math.radians(a.max_grasp_rot_deg))) @ g["RG0"])[None]
        tG = g["tG0"] + bounded(g["gT"], a.max_grasp_shift_mm / 1000) + bounded(g["sT"], a.max_slip_mm / 1000)
        return RG, tG

    @staticmethod
    def to_local(vw, R, p):
        return torch.einsum("bji,bnj->bni", R, vw - p[:, None])

    # ---------------------------------------------------------------- anchors (s06h, against the ref hands)
    def _init_anchors(self):
        a = self.cfg
        self.anchors, self.arows = {}, {}
        with torch.no_grad():
            saved = {sd: [h[k].detach().clone() for k in ("kR", "khp", "kT")] for sd, h in self.H.items()}
            for h in self.H.values():
                for k in ("kR", "khp", "kT"):
                    h[k].data.zero_()
            ref = {sd: self.hand_forward(sd) for sd in self.H}
            for sd, h in self.H.items():
                for k, x in zip(("kR", "khp", "kT"), saved[sd]):
                    h[k].data.copy_(x)
            objf = [self.obj_forward(e) for e in self.O]
        self.ref_j = {sd: ref[sd]["j"].detach() for sd in ref}
        for sd, h in self.H.items():
            ts, tix = h["ts"], h["tix"]
            ti = torch.tensor(ts, device=self.dev)
            anchors, arows = [], []
            h["deep"], h["pen_w"] = [], []
            for oi, (e, f) in enumerate(zip(self.O, objf)):
                loc0 = self.to_local(ref[sd]["vw"], f["R"][ti], f["p"][ti])
                d0 = e["sdf"](loc0.reshape(-1, 3)).view(len(ts), -1)
                val = e["valid"][ti]
                deep = (torch.relu(-d0).amax(1) > a.pen_cap_mm / 1000) & val
                h["deep"].append(deep)
                h["pen_w"].append(torch.where(deep, 0.1, 1.0) * val.float())
                vnp = val.cpu().numpy()
                for seg in segments([t for t in e["grasp"].get(sd, []) if t in tix and vnp[tix[t]]]):
                    idx = [tix[t] for t in seg]
                    squeeze_mm, contact_mm = self.anchor_params(sd, e["name"], seg)
                    cand_v = list(dict.fromkeys(TIPS + list(range(0, loc0.shape[1], a.vert_step))))
                    for k, v in enumerate(cand_v):
                        dd = d0[idx, v].abs().cpu().numpy()
                        windows = sticking_windows(seg, loc0[idx, v].cpu().numpy(), dd,
                                                   contact_m=contact_mm / 1000, slip_m=a.max_slip_mm / 1000)
                        if sum(map(len, windows)) < a.min_touch * len(idx):
                            continue
                        for win in windows:
                            # joint: anchors only where the stereo hand is trusted AND the object is observed -- elsewhere
                            # the ref hand-object placement is what the coupling should correct, not preserve
                            near = [idx[i] for i in win if h["wdata"][idx[i]] >= a.anchor_min_wdata
                                    and e["pos_trust"][ts[idx[i]]] >= a.anchor_min_obj]
                            if len(near) < 3:
                                continue
                            p = loc0[near, v].median(0).values.detach().clone().requires_grad_(True)
                            with torch.enable_grad():
                                sdv = e["sdf"](p[None])
                                (g,) = torch.autograd.grad(sdv.sum(), p)
                            nrm = g / g.norm().clamp(min=1e-9)
                            target = (p - nrm * (sdv + squeeze_mm / 1000)).detach()
                            anchors.append([near, v, oi, target, 1.0, seg[0]])
                            arows.append({"object": e["name"], "vertex": int(v),
                                          "finger": FINGERS[k] if k < len(TIPS) else "surface",
                                          "frames": [ts[near[0]], ts[near[-1]]],
                                          "point_object_frame": target.cpu().numpy().round(5).tolist(),
                                          "touching_frames": len(near), "grasp_start": int(seg[0]),
                                          "squeeze_mm": squeeze_mm, "contact_mm": contact_mm})
            per_seg = {}  # about five fingertips' worth of weight per grasp segment (s06h)
            for an in anchors:
                per_seg.setdefault((an[2], an[5]), []).append(an)
            for group in per_seg.values():
                n_regions = len({an[1] for an in group})
                for an in group:
                    an[4] = min(1.0, 5.0 / n_regions)
            self.anchors[sd], self.arows[sd] = anchors, arows

    # ---------------------------------------------------------------- grasp segments
    def contact_distance(self, hf, objf):
        """min SDF of each hand's vertices per (object, side, frame) (n,), inf where the hand is missing,
        and the count of vertices within enter / exit distances."""
        a, out = self.cfg, {}
        with torch.no_grad():
            for oi, (e, f) in enumerate(zip(self.O, objf)):
                for sd, h in self.H.items():
                    ti = torch.tensor(h["ts"], device=self.dev)
                    d = e["sdf"](self.to_local(hf[sd]["vw"], f["R"][ti], f["p"][ti]).reshape(-1, 3)).view(len(ti), -1)
                    dmin, ce, cx = np.full(self.n, np.inf), np.zeros(self.n, int), np.zeros(self.n, int)
                    ok = e["valid"][ti].cpu().numpy()
                    tt = np.asarray(h["ts"])[ok]
                    dmin[tt] = d.min(1).values.cpu().numpy()[ok]
                    ce[tt] = (d < a.enter_mm / 1000).sum(1).cpu().numpy()[ok]
                    cx[tt] = (d < a.exit_mm / 1000).sum(1).cpu().numpy()[ok]
                    out[(oi, sd)] = (dmin, ce, cx)
        return out

    def contact_sets(self, dist=None):
        """(oi, side) -> (bool contact (n,), confidence (n,), source label per frame)."""
        a, res = self.cfg, {}
        for oi, e in enumerate(self.O):
            for sd in self.H:
                rep = np.zeros(self.n, bool)
                rep[[t for t in e["carry"].get(sd, []) if 0 <= t < self.n]] = True  # coupling: carried frames only
                conf = rep.astype(float)
                src = np.where(rep, "report", "")
                if dist is not None and a.redetect != "off":
                    dmin, ce, cx = dist[(oi, sd)]
                    det = hysteresis(ce >= a.min_verts, cx >= a.min_verts)
                    new = rep.copy()
                    for r in runs_of(det):
                        if (a.redetect == "union" and len(r) >= a.min_seg) or rep[r].any():
                            rt = [t for t in r if rep[t]]  # refine: a report grasp grows <= max_extend frames per end
                            ext = 0 if getattr(a, "carry", 0) else a.max_extend  # carry boundaries come from the object moving: never grow them back
                            lo, hi = (rt[0] - ext, rt[-1] + ext) if rt and a.redetect == "refine" else (-1, self.n)
                            add = [t for t in r if not rep[t] and lo <= t <= hi]
                            new[add], conf[add], src[add] = True, a.redetect_conf, "redetect"
                    # the current estimate says clearly apart AND the object is seen there (not an occluded guess)
                    drop = rep & (dmin > a.release_mm / 1000) & (self.O[oi]["pos_trust"].cpu().numpy() >= 0.5)
                    new[drop], conf[drop] = False, 0.0
                    rep = new
                res[(oi, sd)] = (rep, conf, src)
        return res

    def build_segments(self, contact, hf, objf):
        a, segs = self.cfg, []
        for oi, e in enumerate(self.O):
            vnp = e["valid"].cpu().numpy()
            phases = debounce_phases([tuple(sd for sd in self.H if contact[(oi, sd)][0][t]) for t in range(self.n)],
                                     a.debounce)
            per = []
            for sd, h in self.H.items():
                fr = [t for t in range(self.n) if sd in phases[t] and t in h["tix"] and vnp[t]]
                for s in segments(fr):
                    if len(s) >= a.min_seg:
                        per.append({"oi": oi, "side": sd, "frames": list(s), "source": "report"})
            ext = []  # handover: abutting segments of the two hands overlap, both couplings active at the switch
            for A in per:
                for B in per:
                    if A["side"] != B["side"] and 0 < B["frames"][0] - A["frames"][-1] <= a.handover_gap + 1:
                        e0, s0 = A["frames"][-1], B["frames"][0]
                        ext.append((A, [t for t in range(e0 + 1, s0 + a.overlap) if t in self.H[A["side"]]["tix"] and vnp[t]]))
                        ext.append((B, [t for t in range(e0 - a.overlap + 1, s0) if t in self.H[B["side"]]["tix"] and vnp[t]]))
            for S, add in ext:
                S["frames"] = sorted(set(S["frames"]) | set(add))
                S["handover"] = True
            for A in per:  # overlapping segments of two hands on one object: a handover or a bimanual hold
                for B in per:
                    if A["side"] != B["side"] and set(A["frames"]) & set(B["frames"]):
                        A["handover"] = True
            segs += per
        ncpl = np.zeros((len(self.O), self.n))
        for g in segs:
            ncpl[g["oi"], g["frames"]] += 1
        with torch.no_grad():
            for g in segs:
                e, f, h, hfs = self.O[g["oi"]], objf[g["oi"]], self.H[g["side"]], hf[g["side"]]
                fr = g["frames"]
                L = len(fr)
                g["ft"] = torch.tensor(fr, device=self.dev)
                g["hi"] = torch.tensor([h["tix"][t] for t in fr], device=self.dev)
                conf = contact[(g["oi"], g["side"])][1]
                src = contact[(g["oi"], g["side"])][2]
                g["redetected"] = int(sum(src[t] == "redetect" for t in fr))
                ramp = np.minimum(np.arange(1, L + 1), np.arange(L, 0, -1)).clip(max=a.ramp) / a.ramp
                w = np.array([max(conf[t], a.redetect_conf) for t in fr]) * ramp / ncpl[g["oi"], fr]
                g["w"] = torch.tensor(w, **self.f32)
                HR, Hp = hfs["HR"][g["hi"]], hfs["Hp"][g["hi"]]
                RGs = HR.transpose(-1, -2) @ f["R"][g["ft"]]
                tGs = torch.einsum("fji,fj->fi", HR, f["p"][g["ft"]] - Hp)
                wi = e["rot_trust"][g["ft"]] + 0.05  # G from the frames where the object is clearly seen
                # A chordal mean of opposite FP branches can seed a 90-degree
                # grasp error that bounded optimization cannot subsequently undo.
                from evidence_fusion import robust_pose
                relative = np.tile(np.eye(4), (L, 1, 1))
                relative[:, :3, :3] = RGs.cpu().numpy()
                relative[:, :3, 3] = tGs.cpu().numpy()
                seed = robust_pose(relative, wi.cpu().numpy())
                g["RG0"] = torch.tensor(seed[:3, :3], **self.f32)
                g["tG0"] = torch.tensor(seed[:3, 3], **self.f32)
                for k, shape in (("gR", (3,)), ("gT", (3,)), ("sR", (L, 3)), ("sT", (L, 3))):
                    g[k] = torch.zeros(shape, **self.f32).requires_grad_(True)
        # spline phases: per object/frame the coupled sides, per hand/frame the coupled objects
        for oi, e in enumerate(self.O):
            ph = [tuple(sorted(g["side"] for g in segs if g["oi"] == oi and t in g["frames"])) for t in range(self.n)]
            vi = e["vidx"]
            e["proj"] = spline_projectors(vi, self.fps, [ph[t] for t in vi], self.cfg.spline_knot_seconds) \
                if a.smoothing == "spline" else []
        for sd, h in self.H.items():
            ph = [tuple(sorted(self.O[g["oi"]]["name"] for g in segs if g["side"] == sd and t in g["frames"])) for t in h["ts"]]
            h["proj"] = spline_projectors(h["ts"], self.fps, ph, a.spline_knot_seconds) if a.smoothing == "spline" else []
        self.segs = segs
        self.nohand_mask(hf, objf)

    def nohand_mask(self, hf, objf):
        a = self.cfg
        dist = self.contact_distance(hf, objf)
        for oi, e in enumerate(self.O):
            cov = np.zeros(self.n, bool)
            for g in self.segs:
                if g["oi"] == oi:
                    cov[max(0, g["frames"][0] - 2):g["frames"][-1] + 3] = True
            far = np.ones(self.n, bool)
            for sd in self.H:
                far &= dist[(oi, sd)][0] > a.nohand_mm / 1000
            vnp = e["valid"].cpu().numpy()
            m = far[:-1] & far[1:] & ~cov[:-1] & ~cov[1:] & vnp[:-1] & vnp[1:]
            e["nohand"] = torch.tensor(m, **self.f32)

    # ---------------------------------------------------------------- objective
    def losses(self):
        a, f32 = self.cfg, self.f32
        z = torch.zeros((), **f32)
        T = {k: z for k in ("obs", "couple", "slip", "slip_vel", "static", "nohand", "obj_prior", "obj_smooth",
                            "obj_spline", "obj_table", "pen", "anchor", "table", "data", "tri", "prior", "smooth", "spline")}
        hf = {sd: self.hand_forward(sd) for sd in self.H}
        objf = [self.obj_forward(e) for e in self.O]
        st2, sr2 = (a.sigma_obs_mm / 1000) ** 2, (KP_RADIUS * math.radians(a.sigma_obs_deg)) ** 2
        for e, f in zip(self.O, objf):
            nv = max(len(e["vidx"]), 1)
            if len(e["m_t"]):
                ec = ((f["cw"][e["m_t"]] - e["m_c"]) ** 2).sum(-1)
                Rd = torch.einsum("mij,kj->mki", f["R"][e["m_t"]], e["Dobs"])
                ek = ((Rd[:, None] - e["m_D"]) ** 2).sum(-1).mean(-1).min(1).values  # symmetry: nearest group element
                T["obs"] = T["obs"] + a.w_obs * (e["m_wt"] * st2 * rho(ec / st2) + e["m_wr"] * sr2 * rho(ek / sr2)).sum() / nv
            K, vm = f["K"], e["valid"].float()
            # prior to the init track only where that track was observed (an occluded init is a guess)
            pw = 0.1 + 0.9 * e["pos_trust"].clamp(max=1.0)
            T["obj_prior"] = T["obj_prior"] + a.w_obj_prior * ((((K - e["K0"]) ** 2).sum(-1).mean(-1)) * vm * pw).sum() / nv
            vi = e["vidx"]
            if len(vi) >= 3:
                T["obj_smooth"] = T["obj_smooth"] + a.w_obj_smooth * acceleration_loss(K[vi], vi, self.fps)
                T["obj_spline"] = T["obj_spline"] + a.w_obj_spline * spline_loss(K[vi], e["proj"])
            vel = ((K[1:] - K[:-1]) ** 2).sum(-1).mean(-1) * vm[1:] * vm[:-1]
            T["static"] = T["static"] + a.w_static * (e["static_w"] * vel).sum() / nv
            sn2 = (a.sigma_nohand_mm / 1000) ** 2
            T["nohand"] = T["nohand"] + a.w_nohand * (e["nohand"] * sn2 * rho(vel / sn2)).sum() / nv
            if self.table_h is not None and a.w_obj_table > 0:
                vw = torch.einsum("nij,vj->nvi", f["R"], e["verts"]) + f["p"][:, None]
                below = torch.relu(table_slab_depth(vw, self.plane, side="auto") - a.obj_table_tol_mm / 1000).amax(1)  # deepest vertex
                T["obj_table"] = T["obj_table"] + a.w_obj_table * (below ** 2 * vm).sum() / nv
        sg2 = (a.sigma_grasp_mm / 1000) ** 2
        for g in self.segs:
            e, f, hfs = self.O[g["oi"]], objf[g["oi"]], hf[g["side"]]
            nv = max(len(e["vidx"]), 1)
            RG, tG = self.grasp_pose(g)
            HR, Hp = hfs["HR"][g["hi"]], hfs["Hp"][g["hi"]]
            kp = e["c"] + e["D"]
            pred = torch.einsum("fij,kj->fki", HR @ RG, kp) + (torch.einsum("fij,fj->fi", HR, tG) + Hp)[:, None]
            err = ((f["K"][g["ft"]] - pred) ** 2).sum(-1).mean(-1)
            T["couple"] = T["couple"] + a.w_grasp * (g["w"] * sg2 * rho(err / sg2)).sum() / nv
            xs = torch.cat([bounded(g["sT"], a.max_slip_mm / 1000),
                            KP_RADIUS * bounded(g["sR"], math.radians(a.max_slip_deg))], 1)
            T["slip"] = T["slip"] + a.w_slip * (xs ** 2).sum() / nv
            if len(xs) > 1:
                T["slip_vel"] = T["slip_vel"] + a.w_slip_vel * ((xs[1:] - xs[:-1]) ** 2).sum() / nv
        st = (a.sigma_tri_mm / 1000) ** 2
        for sd, h in self.H.items():
            q, L = hf[sd], len(h["ts"])
            ti = torch.tensor(h["ts"], device=self.dev)
            vw = q["vw"]
            for oi, (e, f) in enumerate(zip(self.O, objf)):
                d = e["sdf"](self.to_local(vw, f["R"][ti], f["p"][ti]).reshape(-1, 3)).view(L, -1)
                T["pen"] = T["pen"] + a.w_pen * (torch.relu(-d) ** 2 * h["pen_w"][oi][:, None]).sum() / L
            for idx, v, oi, target, wa, _ in self.anchors[sd]:
                f = objf[oi]
                tt = ti[idx]
                loc = self.to_local(vw[idx][:, [v]], f["R"][tt], f["p"][tt])[:, 0]
                ramp = torch.arange(1, len(idx) + 1, device=self.dev, dtype=loc.dtype)
                ramp = torch.minimum(ramp, ramp.flip(0)).clamp(max=3) / 3
                T["anchor"] = T["anchor"] + a.w_anchor * wa * (((loc - target) ** 2).sum(-1) * ramp * h["wdata"][idx]).sum() / L
            if self.table_h is not None:
                T["table"] = T["table"] + a.w_table * (table_slab_depth(vw, self.plane, side="auto") ** 2).sum() / L
            T["data"] = T["data"] + a.w_data * (h["wdata"][:, None] * ((q["j"] - self.ref_j[sd]) ** 2).sum(-1)).mean()
            et = ((q["jw"] - h["tri"]) ** 2).sum(-1)
            T["tri"] = T["tri"] + a.w_tri * (h["conf"] * h["tri_ok"] * st * rho(et / st)).mean()
            # s06h's prior, per frame x the same stereo-confidence factor as the data term
            pf = (q["dhp"] ** 2).mean(-1) + a.w_rot * (q["dR"] ** 2).mean(-1) + a.w_trans * (q["dT"] ** 2).sum(-1)
            T["prior"] = T["prior"] + a.w_prior * (h["wdata"] ** a.prior_conf_pow * pf).mean()
            T["smooth"] = T["smooth"] + a.w_smooth * acceleration_loss(q["jw"], h["ts"], self.fps)
            T["spline"] = T["spline"] + a.w_spline * spline_loss(q["jw"], h["proj"])
        return T, hf, objf

    def params(self, hands=True):
        ps = [x for e in self.O for x in (e["om"], e["dl"])]
        ps += [g[k] for g in self.segs for k in ("gR", "gT", "sR", "sT")]
        if hands:
            ps += [h[k] for h in self.H.values() for k in ("kR", "khp", "kT")]
        return ps

    def run_stage(self, iters, hands, tag):
        a = self.cfg
        self.freeze_hands = not hands
        ps = self.params(hands)
        opt = torch.optim.Adam(ps, lr=a.lr)
        for it in range(iters):
            for gp in opt.param_groups:
                gp["lr"] = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / max(iters, 1))))
            T, _, _ = self.losses()
            loss = sum(T.values())
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite joint loss at {tag} it {it}: " +
                                 str({k: float(v) for k, v in T.items() if not torch.isfinite(v)}))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 100 == 0 or it == iters - 1:
                self.log(f"{tag} it {it:4d} loss {loss.item():.4e} " +
                         " ".join(f"{k} {v.item():.2e}" for k, v in T.items() if v.item() != 0))
        self.freeze_hands = False

    def state(self):
        with torch.no_grad():
            T, hf, objf = self.losses()
        return {k: float(v) for k, v in T.items()}, hf, objf


def solve(hands, objs, plane, fps, n, cfg, fk, dev="cpu", anchor_params=None, log=print):
    """Joint optimisation -> (problem, result dict). See Joint for the input layout."""
    P = Joint(hands, objs, plane, fps, n, cfg, fk, dev, anchor_params, log)
    with torch.no_grad():
        hf, objf = {sd: P.hand_forward(sd) for sd in P.H}, [P.obj_forward(e) for e in P.O]
    P.build_segments(P.contact_sets(), hf, objf)
    terms0, _, _ = P.state()
    segs0 = [(P.O[g["oi"]]["name"], g["side"], g["frames"][0], g["frames"][-1]) for g in P.segs]
    P.run_stage(cfg.iters_stage1, hands=False, tag="stage1")
    terms1, hf, objf = P.state()
    if cfg.redetect != "off":  # re-detect grasp phases from the current estimate (hysteresis), then re-initialise G
        P.build_segments(P.contact_sets(P.contact_distance(hf, objf)), hf, objf)
    P.run_stage(cfg.iters_stage2, hands=True, tag="stage2")
    terms2, hf, objf = P.state()
    return P, {"terms_before": terms0, "terms_after_stage1": terms1, "terms_after": terms2,
               "segments_initial": segs0, "hf": hf, "objf": objf}


def slip_stats(P):
    a, out = P.cfg, []
    with torch.no_grad():
        for g in P.segs:
            RG, tG = P.grasp_pose(g)
            dt = (bounded(g["sT"], a.max_slip_mm / 1000).norm(dim=-1) * 1000).cpu().numpy()
            dr = np.rad2deg(bounded(g["sR"], math.radians(a.max_slip_deg)).norm(dim=-1).cpu().numpy())
            out.append({"object": P.O[g["oi"]]["name"], "side": g["side"], "from": int(g["frames"][0]),
                        "to": int(g["frames"][-1]), "n": len(g["frames"]), "handover_overlap": bool(g.get("handover")),
                        "redetected_frames": g["redetected"], "w_mean": round(float(g["w"].mean()), 3),
                        "G_shift_from_init_mm": round(float(bounded(g["gT"], a.max_grasp_shift_mm / 1000).norm()) * 1000, 2),
                        "G_rot_from_init_deg": round(float(np.rad2deg(bounded(g["gR"], math.radians(a.max_grasp_rot_deg)).norm().item())), 2),
                        "slip_mean_mm": round(float(dt.mean()), 2), "slip_max_mm": round(float(dt.max()), 2),
                        "slip_max_deg": round(float(dr.max()), 2)})
    return out


def quality(P, hf, objf, a):
    """s06h's robust penetration statistic per side, against the joint objects."""
    out = {}
    with torch.no_grad():
        for sd, h in P.H.items():
            vw, L = hf[sd]["vw"], len(h["ts"])
            ti = torch.tensor(h["ts"], device=P.dev)
            obj_pen = torch.zeros(L, device=P.dev)
            obj_checked = torch.zeros(L, dtype=torch.bool, device=P.dev)
            deep_any = torch.zeros(L, dtype=torch.bool, device=P.dev)
            near = torch.zeros(vw.shape[:2], dtype=torch.bool, device=P.dev)
            contact = torch.zeros(L, dtype=torch.bool, device=P.dev)
            for oi, (e, f) in enumerate(zip(P.O, objf)):
                val = e["valid"][ti]
                use = val & ~h["deep"][oi]
                deep_any |= h["deep"][oi]
                d = e["sdf"](P.to_local(vw, f["R"][ti], f["p"][ti]).reshape(-1, 3)).view(L, -1)
                obj_pen = torch.where(use, torch.maximum(obj_pen, torch.relu(-d).amax(1)), obj_pen)
                obj_checked |= use
                rel = vw - f["cw"][ti][:, None]
                if P.up is not None:
                    horiz = (rel - (rel @ P.up)[..., None] * P.up).norm(dim=-1)
                    near |= (horiz <= e["radius"] + a.table_margin_m) & val[:, None]
                held = set(e["grasp"].get(sd, []))
                contact |= torch.tensor([t in held for t in h["ts"]], dtype=torch.bool, device=P.dev)
            obj_stats = robust_stats((obj_pen[obj_checked] * 1000).cpu().numpy(), a)
            obj_stats["frames_excluded_deep"] = int(deep_any.sum())
            if P.table_h is not None:  # s06h statistic: disk slab if table_plane.json has the round top
                tab_stats = table_quality(vw, P.plane, near, contact, a)
            else:
                tab_stats = None
            q = hf[sd]
            go = matrix_to_rotvec(rotvec_to_matrix(q["dR"]) @ h["Rgo0"]).cpu().numpy()
            finite = bool(np.isfinite(go).all() and torch.isfinite(q["dhp"]).all() and torch.isfinite(q["j"]).all())
            out[sd] = {"finite": finite, "max_object_penetration_mm": obj_stats["max_mm"],
                       "max_table_penetration_mm": tab_stats["max_mm"] if tab_stats else None,
                       "object": obj_stats, "table": tab_stats,
                       "criterion": f"per-frame max penetration: p95 <= {a.pass_p95_mm} mm and <= "
                                    f"{100 * a.pass_frac_over_10mm:g}% of frames > 10 mm (deep frames excluded)",
                       "passed": bool(finite and obj_stats["passed"] and tab_stats is not None and tab_stats["passed"])}
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--hands", default="mano_fit_phys", help="start the hands here (falls back to --ref_hands)")
    ap.add_argument("--ref_hands", default="mano_fit", help="data term, prior, trust limits and anchors are measured from these")
    ap.add_argument("--segment_params", default="mano_fit_selected", help="s06i dir: per-grasp squeeze/contact (if present)")
    ap.add_argument("--init_objects", default="fused_all_sam3", help="object track the corrections start from")
    ap.add_argument("--evidence_variant", default=None,
                    help="Use validated masked-depth observations and carry segments from evidence_fusion")
    ap.add_argument("--variant", default="fused_joint", help="OUTPUT object variant (report args.variant)")
    ap.add_argument("--out", default="mano_fit_joint", help="OUTPUT hands dir under stereo/")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--fp_source", default="poses_smoothed")
    ap.add_argument("--w_icp", type=float, default=4.0)
    ap.add_argument("--vis_weight", type=int, default=1)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--max_frames", type=int, default=3000, help="whole-episode optimisation; refuse longer episodes")
    ap.add_argument("--iters_stage1", type=int, default=300, help="objects + grasp offsets, hands fixed")
    ap.add_argument("--iters_stage2", type=int, default=900, help="everything jointly")
    ap.add_argument("--lr", type=float, default=3e-3)
    # trust limits (configurable, not measured limits)
    ap.add_argument("--max_shift_mm", type=float, default=25.0)
    ap.add_argument("--max_rotation_deg", type=float, default=20.0)
    ap.add_argument("--max_finger_deg", type=float, default=20.0)
    ap.add_argument("--max_obj_shift_mm", type=float, default=40.0)
    ap.add_argument("--max_obj_rot_deg", type=float, default=25.0)
    ap.add_argument("--max_grasp_shift_mm", type=float, default=30.0, help="segment offset G vs its initial value")
    ap.add_argument("--max_grasp_rot_deg", type=float, default=25.0)
    ap.add_argument("--max_slip_mm", type=float, default=10.0, help="per-frame slip (also s06h's anchor slip)")
    ap.add_argument("--max_slip_deg", type=float, default=8.0)
    # grasp phases
    ap.add_argument("--redetect", choices=("off", "refine", "union"), default="refine")
    ap.add_argument("--comotion", type=int, default=1, help="1: add co-motion / airborne grasps (comotion_grasps) to the report contact")
    ap.add_argument("--comotion_near_mm", type=float, default=100.0)
    ap.add_argument("--comotion_hands", default="hands_c", help="stereo/<dir> hands for the co-motion test (joints)")
    ap.add_argument("--carry", type=int, default=0, help="1: couple only on carried frames (fusion_report hand_carry) and use "
                    "colour-time (retimed) stereo-ICP; 0: legacy (whole contact, raw ICP time)")
    ap.add_argument("--enter_mm", type=float, default=8.0)
    ap.add_argument("--exit_mm", type=float, default=15.0)
    ap.add_argument("--max_extend", type=int, default=3, help="refine: frames a report grasp may grow per end")
    ap.add_argument("--min_verts", type=int, default=3)
    ap.add_argument("--release_mm", type=float, default=40.0, help="drop report contact frames where the hand is this far")
    ap.add_argument("--redetect_conf", type=float, default=0.7)
    ap.add_argument("--debounce", type=int, default=3)
    ap.add_argument("--min_seg", type=int, default=4)
    ap.add_argument("--ramp", type=int, default=3)
    ap.add_argument("--handover_gap", type=int, default=2)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--nohand_mm", type=float, default=20.0)
    # s06h
    ap.add_argument("--smoothing", choices=("spline", "acceleration"), default="spline")
    ap.add_argument("--spline_knot_seconds", type=float, default=0.15)
    ap.add_argument("--contact_mm", type=float, default=15.0)
    ap.add_argument("--min_touch", type=float, default=0.3)
    ap.add_argument("--anchor_min_wdata", type=float, default=0.6, help="anchor frames need 0.25 + 0.75 stereo conf >= this")
    ap.add_argument("--anchor_min_obj", type=float, default=0.3, help="... and object position trust (max w_t) >= this")
    ap.add_argument("--squeeze_mm", type=float, default=1.5)
    ap.add_argument("--vert_step", type=int, default=6)
    ap.add_argument("--pen_cap_mm", type=float, default=25.0)
    ap.add_argument("--table_margin_m", type=float, default=0.05)
    ap.add_argument("--pass_p95_mm", type=float, default=5.0)
    ap.add_argument("--pass_frac_over_10mm", type=float, default=0.05)
    # sigmas
    ap.add_argument("--sigma_obs_mm", type=float, default=10.0)
    ap.add_argument("--sigma_obs_deg", type=float, default=5.0)
    ap.add_argument("--sigma_grasp_mm", type=float, default=15.0, help="soft-L1 scale only (weight: w_grasp)")
    ap.add_argument("--sigma_tri_mm", type=float, default=15.0)
    ap.add_argument("--sigma_nohand_mm", type=float, default=2.0)
    ap.add_argument("--obj_table_tol_mm", type=float, default=3.0)
    # weights (metre^2 units, s06h's hand weights)
    ap.add_argument("--w_obs", type=float, default=1e4, help="1/sigma^2 per unit measurement weight: fuse_objects sigma_t 10 mm")
    ap.add_argument("--w_grasp", type=float, default=3e4, help="fuse_objects grasp sigma 6 mm")
    ap.add_argument("--w_slip", type=float, default=1e5, help="slip ~3 mm")
    ap.add_argument("--w_slip_vel", type=float, default=4e6, help="slip rate ~0.5 mm/frame")
    ap.add_argument("--w_static", type=float, default=1e6, help="fuse_objects sigma_vel 1 mm/frame")
    ap.add_argument("--w_nohand", type=float, default=1e6, help="robust (sigma_nohand_mm); only where no hand is within --nohand_mm")
    ap.add_argument("--w_obj_prior", type=float, default=400.0, help="~5 cm; holds unobservable cup spin")
    ap.add_argument("--w_obj_smooth", type=float, default=6e4, help="fuse_objects sigma_acc 4 mm/frame^2 at 20 Hz")
    ap.add_argument("--w_obj_spline", type=float, default=6e3, help="fuse_objects spline_weight 0.2 (per coordinate)")
    ap.add_argument("--w_obj_table", type=float, default=2e4)
    ap.add_argument("--w_anchor", type=float, default=3e3)
    ap.add_argument("--w_pen", type=float, default=2e4)
    ap.add_argument("--w_table", type=float, default=2e4)
    ap.add_argument("--w_data", type=float, default=1.5e3)
    ap.add_argument("--w_tri", type=float, default=1e3)
    ap.add_argument("--w_prior", type=float, default=1.0)
    ap.add_argument("--prior_conf_pow", type=float, default=1.0, help="0: s06h's unscaled prior")
    ap.add_argument("--w_rot", type=float, default=50.0)
    ap.add_argument("--w_trans", type=float, default=2e4)
    ap.add_argument("--w_smooth", type=float, default=5e3)
    ap.add_argument("--w_spline", type=float, default=3e3)
    ap.add_argument("--force", action="store_true", help="write into output dirs not created by s06j")
    return ap


# ---------------------------------------------------------------- I/O
def comotion_grasps(W0, verts, hands_st, slam, n, plane, a):
    """Grasps the contact detector misses (e03 pot: hand 6-7 cm from our pot surface for 160 airborne frames -- a
    hand/object offset, not a release). A hand holds the object at t when it is within --comotion_near_mm of the
    object surface AND either the object is AIRBORNE and moving (lowest point > 2 cm above the tabletop and its
    centre travels > 2 cm within +-10 frames: a lid resting on a still pot is not carried) or it CO-MOVES with the
    hand (object speed > 4 mm/frame and the palm velocity within 50 % of it, +-3-frame means). Both hands within
    3 cm -> both; else the nearer. Gaps <= 5 frames filled, runs < 4 frames dropped. Returns {side: [frames]}."""
    from fuse_objects import PALM, stereo_hands
    from scipy.spatial import cKDTree
    hands = stereo_hands(run_of(a), n, a.comotion_hands) or hands_st
    V = verts[:: max(1, len(verts) // 1500)]
    ok = np.isfinite(W0[:, 0, 0])
    X = {t: V @ W0[t, :3, :3].T + W0[t, :3, 3] for t in range(n) if ok[t]}
    c = np.array([X[t].mean(0) if ok[t] else np.full(3, np.nan) for t in range(n)])
    low = np.full(n, np.nan)
    if plane is not None:
        up, h = np.asarray(plane["up"], float), float(plane["height"])
        for t in X:
            low[t] = (X[t] @ up).min() - h
    travel = np.array([np.nanmax(np.linalg.norm(c[max(0, t - 10):t + 11] - c[t], axis=1)) if ok[t] else 0.0 for t in range(n)])
    airborne = (np.nan_to_num(low) > 0.02) & (travel > 0.02)

    def vel(P):
        v = np.full_like(P, np.nan)
        v[1:-1] = (P[2:] - P[:-2]) / 2
        out = np.full_like(P, np.nan)
        for t in range(n):
            s = v[max(0, t - 3):t + 4]
            s = s[np.isfinite(s[:, 0])]
            if len(s):
                out[t] = s.mean(0)
        return out
    vo = vel(c)
    dist, comove = {}, {}
    for sd, hd in hands.items():
        d = np.full(n, np.inf)
        palm = np.full((n, 3), np.nan)
        for t in X:
            if np.isfinite(hd["J"][t]).all():
                Jw = hd["J"][t] @ slam[t][:3, :3].T + slam[t][:3, 3]
                d[t] = cKDTree(X[t]).query(Jw)[0].min()
                palm[t] = Jw[PALM].mean(0)
        vh = vel(palm)
        so = np.linalg.norm(vo, axis=1)
        comove[sd] = np.nan_to_num((so > 0.004) & (np.linalg.norm(vo - vh, axis=1) < 0.5 * so)).astype(bool)
        dist[sd] = d
    out = {}
    for sd in dist:
        other = [o for o in dist if o != sd]
        g = (dist[sd] < a.comotion_near_mm / 1000) & (airborne | comove[sd])
        for o in other:   # the nearer hand holds it, unless both are on it
            g &= (dist[sd] <= dist[o]) | (dist[sd] < 0.03)
        idx = np.nonzero(g)[0]
        for t0, t1 in zip(idx, idx[1:]):
            if 1 < t1 - t0 <= 6:
                g[t0:t1] = True
        keep = np.zeros(n, bool)
        for r in runs_of(g):
            if len(r) >= 4:
                keep[r] = True
        out[sd] = np.nonzero(keep)[0].tolist()
    return out


def run_of(a):
    return ROOT / "runs" / a.run


def load_objects(run, a, slam, n, fps, plane, dev, rep):
    """Measurements exactly as fuse_objects main (FP poses_smoothed vis-weighted + gated stereo-ICP),
    canonicalised; damping weights from its cached contact/visibility; the init track in the world."""
    from fuse_objects import canon_measurements, damping_weights, grasp_segments, retime, stereo_hands
    from symmetry import mesh_symmetry
    hands_st = stereo_hands(run, n)
    evidence = None
    if a.evidence_variant:
        import hashlib
        evidence = json.loads((run / a.evidence_variant / "evidence_report.json").read_text())
        if (not evidence.get("complete") or evidence.get("mesh_root") != "meshes_recon"
                or a.obj_dir != "objects_recon"
                or os.environ.get("V2D_MESH_DIR") != "meshes_recon"
                or abs(evidence["fps"] - fps) > 1e-6):
            raise ValueError("Evidence must be complete and match reconstructed meshes, objects and timing")
    objs = []
    for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
        name, od = o["name"], run / a.obj_dir / o["name"]
        cache = od / f".contact_vis_{a.fp_source}.npz"
        if not cache.exists() or cache.stat().st_mtime < (run / a.obj_dir / "objects.json").stat().st_mtime:
            raise FileNotFoundError(f"{cache} missing or stale: run fuse_objects (V2D_OBJ_DIR={a.obj_dir}) first")
        c_ = np.load(cache)
        near, vis = c_["near"], c_["vis"]
        pd = run / a.init_objects / name / "poses"
        W0 = np.full((n, 4, 4), np.nan)
        for t in slam:
            if t < n and (pd / f"{t:06d}.json").exists():
                W0[t] = slam[t] @ load_T(pd / f"{t:06d}.json")
        mp = ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj"
        hc = rep.get(name, {}).get("hand_contact", {})
        cm = comotion_grasps(W0, np.asarray(trimesh.load(mp, force="mesh").vertices), hands_st, slam, n, plane, a) if a.comotion else {}
        hc = {sd: sorted(set(hc.get(sd, [])) | set(cm.get(sd, []))) for sd in ("left", "right")}
        carry = rep.get(name, {}).get("hand_carry") if a.carry else None  # contact is not carrying (fuse_objects)
        if carry is not None:
            carry = {sd: sorted(set(carry.get(sd, [])) | set(cm.get(sd, []))) for sd in ("left", "right")}
        hk = carry if carry is not None else hc
        side_at = [tuple(sd for sd in ("left", "right") if t in set(hc.get(sd, []))) for t in range(n)]
        touch_h = np.array([bool(x) for x in side_at])
        near = near | (np.convolve(touch_h.astype(float), np.ones(5), "same") > 0)
        carry_at = [tuple(sd for sd in ("left", "right") if t in set(hk.get(sd, []))) for t in range(n)]
        gsegs = grasp_segments(hands_st, carry_at, slam, n)
        held = np.zeros(n, bool)
        for g in gsegs:
            held[g["frames"]] = True
        fp, icp = [], []
        for p in sorted((od / a.fp_source).glob("*.json")):
            t = int(p.stem)
            if t in slam and t < n and vis[t] > 0.05:
                v = float(vis[t])
                fp.append((t, slam[t] @ load_T(p), (v, v ** 2) if a.vis_weight else 1.0))
        for p in sorted((od / "stereo_icp").glob("*.json")):
            t, d = int(p.stem), json.loads(p.read_text())
            if t not in slam or "T" not in d:
                continue
            if (d["fitness"] >= 0.5 and d["rmse_mm"] is not None and d["rmse_mm"] <= 4.0
                    and d["n_scan"] >= 300 and d["moved_from_init_mm"] <= 50):
                icp.append((t, slam[t] @ np.array(d["T"]), a.w_icp * d["fitness"] ** 2))
        icp = retime(icp, float(rep.get(name, {}).get("icp_time_shift", 0.0)) if a.carry else 0.0, slam)  # colour time, as fuse_objects
        sym = mesh_symmetry(mp)
        meas, n_ch = canon_measurements({"fp": fp, "icp": icp}, sym)
        mesh = trimesh.load(mp, force="mesh")
        verts = np.asarray(mesh.vertices)
        static_w, still, rest = damping_weights(near, vis, held, gsegs, {t: T_ for t, T_, _ in meas["fp"]}, fps,
                                                verts, plane, sym["cont_axis"])
        flat = [(t, T_, *((w, w) if np.isscalar(w) else w)) for src in ("fp", "icp") for t, T_, w in meas[src]]
        if evidence is not None:
            er = evidence["objects"][name]
            if not er["geometry_pass"] or hashlib.sha256(mp.read_bytes()).hexdigest() != er["mesh_sha256"]:
                raise ValueError(f"Failed geometry gate or changed mesh for {name}")
            if not er.get("observations"):
                raise ValueError("Evidence has no recorded observations; regenerate with current evidence_fusion")
            flat = [(r["frame"], np.asarray(r["pose_world"]), r["weight"], r["weight"])
                    for r in er["observations"]]
            # Shape hypotheses were independently scored; do not discard handles or
            # bristles by declaring their orientation unobservable here.
            sym = {"group": [np.eye(3)], "cont_axis": None, "center": np.asarray(mesh.centroid), "kind": "evidence"}
            hk = {sd: sorted({t for g in er["segments"] if g["side"] == sd
                             for t in range(g["from"], g["to"] + 1)}) for sd in ("left", "right")}
            hc = {sd: sorted(set(hc.get(sd, [])) | set(hk[sd])) for sd in hk}
            # Cached stillness was computed from the old pose hypotheses. Reuse
            # neither that rotation lock nor an occlusion-based stop prior.
            static_w = np.zeros(n - 1)
        c = np.asarray(verts).mean(0)
        sel = np.random.default_rng(0).choice(len(verts), min(400, len(verts)), replace=False)
        objs.append({"name": name, "W0": W0, "meas": flat, "sym": sym, "sdf": SDFGrid(mesh, res=160, device=dev),
                     "verts": verts[sel], "static_w": static_w, "grasp": hc, "carry": hk,
                     "radius": float(np.linalg.norm(verts - c, axis=1).max()),
                     "info": {"n_fp": len(meas["fp"]), "n_icp": len(meas["icp"]), "symmetry": sym["kind"],
                              "canonicalised": n_ch, "comotion_frames": {sd: len(v) for sd, v in cm.items()}, "static_frac": float((static_w > 0.5).mean())}})
    return objs, hands_st


def load_hands(run, a, slam, hands_st):
    def read(d, side):
        return [(int(f.stem), json.loads(f.read_text())) for f in sorted((run / "stereo" / d / side).glob("*.json"))
                if int(f.stem) in slam]
    hands, recs_all = {}, {}
    for side in ("left", "right"):
        recs = read(a.ref_hands, side)
        init = read(a.hands, side) if (run / "stereo" / a.hands / side).is_dir() else recs
        if [t for t, _ in init] != [t for t, _ in recs]:
            raise ValueError(f"{side}: {a.hands} and {a.ref_hands} cover different frames")
        if len(recs) < 3:
            continue
        ts = [t for t, _ in recs]
        st = hands_st.get(side)
        tri = np.full((len(ts), 21, 3), np.nan)
        conf = np.zeros((len(ts), 21))
        if st is not None:
            J, C = st["J"], st["C"]
            for i, t in enumerate(ts):
                if t < len(J) and np.isfinite(J[t]).all():
                    tri[i] = J[t] @ slam[t][:3, :3].T + slam[t][:3, 3]
                    conf[i] = C[t]
        mk = lambda rr: {"go": [r["mano"]["global_orient"] for _, r in rr], "hp": [r["mano"]["hand_pose"] for _, r in rr],
                         "be": [r["mano"]["betas"] for _, r in rr], "T": [r["T"] for _, r in rr]}  # noqa: E731
        hands[side] = {"ts": ts, "is_right": recs[0][1]["is_right"], "ref": mk(recs),
                       "init": mk(init) if init is not recs else None,
                       "Sw": np.stack([slam[t] for t in ts]), "tri": tri, "conf": conf}
        recs_all[side] = recs
    return hands, recs_all


def main(argv=None):
    a = build_parser().parse_args(argv)
    t_start = time.time()
    run = ROOT / "runs" / a.run
    if a.variant in (a.init_objects, "fused_all_sam3", "fp") or a.out in (a.hands, a.ref_hands, "mano_fit", "mano_fit_phys"):
        raise ValueError("Joint outputs must be new variants, never an input")
    for d in (run / a.variant, run / "stereo" / a.out):
        if d.exists() and not (d / ".s06j").exists() and not a.force:
            raise ValueError(f"{d} exists and was not written by s06j; refusing to overwrite (--force)")
    fps = float(json.loads((run / "episode.json").read_text())["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source fps")
    dev = a.device
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    n = max(slam) + 1
    if n > a.max_frames:
        raise ValueError(f"{n} frames > --max_frames {a.max_frames}: window the episode")
    tp = run / "table_plane.json"
    plane = json.loads(tp.read_text()) if tp.exists() else None
    if plane is None:
        print("WARNING: no table_plane.json -- table terms off, quality cannot pass", flush=True)
    rep = json.loads((run / f"fusion_report_{a.obj_dir.replace('objects_', '')}.json").read_text())   # the report of THIS obj_dir's fusion
    objs, hands_st = load_objects(run, a, slam, n, fps, plane, dev, rep)
    hands, recs = load_hands(run, a, slam, hands_st)
    sp = run / "stereo" / a.segment_params / "report.json" if a.segment_params else None
    seg_params = load_segment_params(run, a.segment_params, a) if sp is not None and sp.exists() else None
    mano = Mano(str(ROOT / "weights/hand"), dev)
    fk = lambda go, hp, be, is_right: mano_forward(mano, go, hp, be, is_right)  # noqa: E731
    P, res = solve(hands, objs, plane, fps, n, a, fk, dev,
                   lambda sd, nm, seg: segment_anchor_params(seg_params, sd, nm, seg, a),
                   lambda s: print(s, flush=True))
    hf, objf = res["hf"], res["objf"]
    qual = quality(P, hf, objf, a)
    # ---- objects: camera frame, fused_all_sam3 layout
    vroot = run / a.variant
    vroot.mkdir(parents=True, exist_ok=True)
    (vroot / ".s06j").write_text("written by s06j_joint_hoi.py\n")
    obj_rep = {}
    for e, f, o in zip(P.O, objf, objs):
        R, p = f["R"].detach().cpu().numpy().astype(float), f["p"].detach().cpu().numpy().astype(float)
        if not (np.isfinite(R).all() and np.isfinite(p).all()):
            raise ValueError(f"Non-finite object poses for {e['name']}; refusing output")
        pdir = vroot / e["name"] / "poses"
        pdir.mkdir(parents=True, exist_ok=True)
        for old in pdir.glob("*.json"):
            old.unlink()
        vnp = e["valid"].cpu().numpy()
        for t in sorted(slam):
            if t < n and vnp[t]:
                W = np.eye(4)
                W[:3, :3], W[:3, 3] = R[t], p[t]
                C = np.linalg.inv(slam[t]) @ W
                (pdir / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
        W0 = np.asarray(o["W0"])[vnp]
        dc = np.linalg.norm((f["cw"] - e["cw0"]).detach().cpu().numpy()[vnp], axis=1) * 1000
        dr = np.rad2deg(np.arccos(np.clip((np.einsum("nij,nij->n", R[vnp], W0[:, :3, :3]) - 1) / 2, -1, 1)))
        obj_rep[e["name"]] = {**o["info"], "centroid_change_mm": {"mean": round(float(dc.mean()), 2), "max": round(float(dc.max()), 2)},
                              "rot_change_deg": {"mean": round(float(dr.mean()), 2), "max": round(float(dr.max()), 2)},
                              "nohand_frac": round(float(e["nohand"].mean()), 3)}
    # ---- hands: s06h format
    outdir = run / "stereo" / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ".s06j").write_text("written by s06j_joint_hoi.py\n")
    if (outdir / "report.json").exists():
        (outdir / "report.json").unlink()
    report = {"args": vars(a), "table_height": P.table_h, "sides": {}}
    hand_rep = {}
    with torch.no_grad():
        for side in ("left", "right"):
            for old in (outdir / side).glob("*.json"):
                old.unlink()
            if side not in P.H:
                continue
            h, q = P.H[side], hf[side]
            go = matrix_to_rotvec(rotvec_to_matrix(q["dR"]) @ h["Rgo0"]).cpu().numpy()
            hp, T, jc = (h["hp0"] + q["dhp"]).cpu().numpy(), (h["T0"] + q["dT"]).cpu().numpy(), q["j"].cpu().numpy()
            if not qual[side]["finite"]:
                raise ValueError("Non-finite hand geometry; refusing output")
            ti = torch.tensor(h["ts"], device=P.dev)
            res_a = []
            for idx, v, oi, target, *_ in P.anchors[side]:
                f = objf[oi]
                loc = P.to_local(q["vw"][idx][:, [v]], f["R"][ti[idx]], f["p"][ti[idx]])[:, 0]
                res_a.append(float((loc - target).norm(dim=-1).mean()) * 1000)
            moved = float(((q["j"] - P.ref_j[side]) ** 2).sum(-1).sqrt().mean()) * 1000
            (outdir / side).mkdir(parents=True, exist_ok=True)
            for k, (t, r) in enumerate(recs[side]):
                r2 = dict(r)
                r2["mano"] = {"betas": r["mano"]["betas"], "global_orient": go[k].tolist(), "hand_pose": hp[k].tolist()}
                r2["T"], r2["joints"] = T[k].tolist(), jc[k].tolist()
                (outdir / side / f"{t:06d}.json").write_text(json.dumps(r2))
            arows = [dict(row, residual_mm=round(e_, 2)) for row, e_ in zip(P.arows[side], res_a)]
            report["sides"][side] = {"frames": len(h["ts"]), "anchors": arows, "mean_joint_shift_mm": moved,
                                     "anchor_residual_mm": float(np.mean(res_a)) if res_a else None, "quality": qual[side]}
            jw = q["jw"]
            tri, ok = h["tri"], h["tri_ok"] > 0
            wrist = (q["Hp"] - torch.einsum("bij,bj->bi", h["Sw"][:, :3, :3], P.ref_j[side][:, 0]) - h["Sw"][:, :3, 3]).norm(dim=-1)
            hand_rep[side] = {"mean_joint_shift_vs_ref_mm": round(moved, 2),
                              "wrist_shift_vs_ref_mm": {"mean": round(float(wrist.mean()) * 1000, 2), "max": round(float(wrist.max()) * 1000, 2)},
                              "tri_err_mm_median": round(float(((jw - tri).norm(dim=-1)[ok]).median()) * 1000, 2) if ok.any() else None,
                              "quality_passed": qual[side]["passed"]}
    report["quality_passed"] = bool(report["sides"]) and all(s["quality"]["passed"] for s in report["sides"].values())
    report["quality_summary"] = {sd: {k: s["quality"][k] for k in ("passed", "object", "table")}
                                 for sd, s in report["sides"].items()}
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    runtime = time.time() - t_start
    jr = {"episode": a.run, "args": vars(a), "runtime_s": round(runtime, 1), "frames": n, "fps": fps,
          "terms_before": res["terms_before"], "terms_after_stage1": res["terms_after_stage1"], "terms_after": res["terms_after"],
          "grasp_segments_initial": res["segments_initial"], "grasp_segments": slip_stats(P),
          "objects": obj_rep, "hands": hand_rep, "quality_passed": report["quality_passed"]}
    (vroot / "joint_report.json").write_text(json.dumps(jr, indent=1))
    print(f"s06j {a.run}: {runtime:.0f} s, quality_passed {report['quality_passed']}, objects "
          f"{json.dumps({k: [v['centroid_change_mm']['mean'], v['rot_change_deg']['mean']] for k, v in obj_rep.items()})}", flush=True)


if __name__ == "__main__":
    main()
