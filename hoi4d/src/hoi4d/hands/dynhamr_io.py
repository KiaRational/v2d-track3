"""Dyn-HaMR (github.com/ZhengdiYu/Dyn-HaMR @ fa9cd74) as the hand pose DETECTION stage, on our geometry.

prepare(): Dyn-HaMR data layout for one run under <root>:
  images/<run>/NNNNNN.jpg                        camera a colour frames (our ego_cam_a)
  dynhamr/cameras/<run>/shot-0/cameras.npz       OUR cameras -- SLAM poses in the hoi4d world (metric, table z = 0)
                                                 as w2c (N,4,4), calibrated intrinsics (N,4: fx fy cx cy), size
  -> Dyn-HaMR skips its own SLAM (cameras exist) and runs its hand tracker (WiLoR detector + HaMeR + ViTPose) and
     its world-frame optimisation (+ HMP motion prior) in our metric world.
Frames without a SLAM pose (rare) take the nearest valid pose (input only).
convert(): Dyn-HaMR export (dynhamr_export.py) -> stereo/<variant> hand records (see convert).
anchor(): Dyn-HaMR hands with the wrist DEPTH of our stereo-fitted hands (see anchor_depth).
Usage: python -m hoi4d.hands.dynhamr_io prepare <run> <root>  |  convert <run> <export.npz> [variant] [depth_ref]
       python -m hoi4d.hands.dynhamr_io anchor <run> [variant] [ref] [out_variant]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/scratch/kghasemz/v2d")


def prepare(run_name: str, root: str) -> dict:
    from .. import io as hio
    run = ROOT / "runs" / run_name
    man = json.loads((run / "hoi4d/scene_manifest.json").read_text())
    tr = hio.Trajectories.load(run / "hoi4d/trajectories_initial.npz")
    T_wc = tr.extra["T_world_cam"].copy()
    ok = np.array([np.isfinite(T).all() for T in T_wc])
    idx = np.where(ok)[0]
    for f in np.where(~ok)[0]:
        T_wc[f] = T_wc[idx[np.argmin(abs(idx - f))]]
    W, H = man["cameras"]["ego_cam_a"]["size"]
    K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
    F = man["frame_count"]
    img = Path(root) / "images" / run_name
    img.mkdir(parents=True, exist_ok=True)
    for f in range(F):
        o = img / f"{f:06d}.jpg"
        if not o.exists():
            Image.open(run / "recon_hawor/frames" / f"{f:06d}.png").convert("RGB").save(o, quality=95)
    cd = Path(root) / "dynhamr/cameras" / run_name / "shot-0"
    cd.mkdir(parents=True, exist_ok=True)
    w2c = np.linalg.inv(T_wc).astype(np.float32)
    intr = np.repeat(np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]], np.float32), F, 0)
    np.savez(cd / "cameras.npz", w2c=w2c, intrins=intr, height=H, width=W, focal=float(K[0, 0]))
    return {"frames": F, "size": [W, H], "slam_gaps_filled": int((~ok).sum()), "images": str(img), "cameras": str(cd / "cameras.npz")}



def operator_prior(run_name: str, out_json: str, src: str = "mano_fit") -> dict:
    """Image boxes of the OPERATOR's hands per frame, for Dyn-HaMR's detector (DYNHAMR_OPERATOR_PRIOR, hamer/run.py
    select_operator): our stereo hand fit <src> (mesh-free; the camera wearer's hands) projected with camera-a K, the
    21-joint box grown 1.3x then made square x1.2 like the detector's own enlarged boxes."""
    run = ROOT / "runs" / run_name
    man = json.loads((run / "hoi4d/scene_manifest.json").read_text())
    K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
    W, H = man["cameras"]["ego_cam_a"]["size"]
    frames, n = {}, {"left": 0, "right": 0}
    for side in ("left", "right"):
        for p in sorted((run / "stereo" / src / side).glob("[0-9]*.json")):
            J = np.asarray(json.loads(p.read_text())["joints"], float)
            if not np.isfinite(J).all() or (J[:, 2] < 0.05).any():
                continue
            uv = (J @ K.T)[:, :2] / J[:, 2:3]
            (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
            c, s = np.array([(x0 + x1) / 2, (y0 + y1) / 2]), max(x1 - x0, y1 - y0) * 1.3 * 1.2
            box = [max(0.0, c[0] - s / 2), max(0.0, c[1] - s / 2), min(W - 1.0, c[0] + s / 2), min(H - 1.0, c[1] + s / 2)]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue   # hand entirely outside the image
            frames.setdefault(int(p.stem), {})[side] = [float(v) for v in box]
            n[side] += 1
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps({"source": src, "image_size": [W, H], "frames": frames}))
    return {"run": run_name, "source": src, "frames_with_prior": n, "out": out_json}


def convert(run_name: str, export_npz: str, variant: str = "dynhamr", iters: int = 600, depth_ref: str | None = None,
            sigma: float = 2.0, despin: bool = False) -> dict:
    """Dyn-HaMR export (dynhamr_export.py) -> runs/<run>/stereo/<variant>/<side>/NNNNNN.json in the mano_fit schema
    (agile container: manotorch MANO, camera-a frame, left = mirrored right-hand params).
    Our MANO parameters are FITTED to Dyn-HaMR's posed vertices (same MANO topology, 778 vertex correspondences:
    shared betas, per-frame global_orient / hand_pose / T; Adam then LBFGS), so no convention (mean pose, joint
    order, left mirroring) is assumed. Frames written: the track's span from first to last detection; the flags
    has_wilor_2d/has_triangulation mark Dyn-HaMR's detected frames (vis_mask), the rest are its motion-prior infill.
    depth_ref (e.g. mano_fit_hop_phys): monocular Dyn-HaMR places a hand too far AND too large by the same factor
    (depth/size ambiguity). Per frame the whole hand is scaled about the camera centre by k = z_ref / z_dynhamr of the
    wrist (_depth_ratio): every joint keeps its exact image position, the wrist gets our stereo depth and the hand its
    metric size (absorbed by the fitted betas).
    despin: Dyn-HaMR flips the palm for 1-4 detected frames (mostly as the hand enters/leaves the view). Frames whose
    target orientation (Kabsch of its vertices) is > SPIN_DEG from the robust windowed mean orientation are outliers:
    they keep only their centroid in the data term, and a temporal smoothness term (joint + rotation + pose
    acceleration over consecutive frames) lets the optimizer fill their orientation and articulation."""
    import sys as _s
    import torch
    _s.path.insert(0, "/code/agile")
    from geometry import Mano
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
    from .. import io as hio
    run = ROOT / "runs" / run_name
    tr = hio.Trajectories.load(run / "hoi4d/trajectories_initial.npz")
    T_wc = tr.extra["T_world_cam"]
    camok = np.isfinite(T_wc).all(axis=(1, 2))
    d = np.load(export_npz)
    V, isr, vis = d["verts"].astype(np.float64), d["is_right"], d["vis_mask"]
    B, F = V.shape[:2]
    camok = camok[:F]   # Dyn-HaMR may stop early (frames 0..F-1 of the clip; e.g. ep0: 342 of 373)
    w2c_dh = np.tile(np.eye(4), (F, 1, 1)); w2c_dh[:, :3, :3] = d["cam_R"].reshape(-1, F, 3, 3)[0]; w2c_dh[:, :3, 3] = d["cam_t"].reshape(-1, F, 3)[0]
    rel = [w2c_dh[f] @ T_wc[f] for f in range(F) if camok[f]]   # identity iff same world; constant iff re-anchored
    cam_dev = [float(np.abs(r - np.eye(4))[:3].max()) for r in rel]
    rel_spread = float(max(np.abs(r - rel[0])[:3].max() for r in rel))
    mano = Mano(str(ROOT / "weights/hand"), "cuda")
    out = run / "stereo" / variant
    rep = {"export": str(export_npz), "source": str(d["source"]), "world_scale": d["world_scale"].ravel()[:2].tolist(),
           "camera_max_abs_dev": max(cam_dev), "camera_rel_spread": rel_spread, "sides": {}}
    for side in ("right", "left"):
        want = side == "right"
        cand = [k for k in range(B) if ((isr[k] > 0.5).mean() > 0.5) == want]
        if not cand:
            continue
        b = max(cand, key=lambda k: vis[k].sum())
        det = np.where(vis[b] & camok)[0]
        if len(det) == 0:
            continue
        fr = np.arange(det[0], det[-1] + 1)
        fr = fr[camok[fr]]
        # world -> camera with Dyn-HaMR's OWN cameras (its world may be re-anchored; the camera frame is shared)
        Vc = np.einsum("fij,fvj->fvi", w2c_dh[fr, :3, :3], V[b, fr]) + w2c_dh[fr, None, :3, 3]
        kinfo, idbad, raydev = None, np.zeros(len(fr), bool), None
        if depth_ref:
            wd = np.einsum("fij,fj->fi", w2c_dh[fr, :3, :3], d["joints"][b, fr, 0].astype(np.float64)) + w2c_dh[fr, :3, 3]
            zd = wd[:, 2]
            if despin:   # identity check (depth-free): Dyn-HaMR's wrist ray vs our stereo wrist; a wrong hand/track disagrees
                wr = _ref_wrist(run / "stereo", depth_ref, side, fr)
                cosang = np.einsum("fi,fi->f", wd, wr) / (np.linalg.norm(wd, axis=1) * np.linalg.norm(wr, axis=1))
                raydev = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                idbad = np.nan_to_num(raydev, nan=0.0) > RAY_DEG
            k, zr, okr = _depth_ratio(run / "stereo", depth_ref, side, fr, zd, sigma, exclude=idbad)
            if k is not None:
                Vc = Vc * k[:, None, None]
                kinfo = {"ref": depth_ref, "anchored": int(okr.sum()), "k": [float(k.min()), float(np.median(k)), float(k.max())],
                         "wrist_depth_offset_before_mm_median": float(np.median((zd - zr)[okr]) * 1e3),
                         "wrist_depth_resid_after_mm": [float(np.median(abs(zd * k - zr)[okr]) * 1e3),
                                                        float(np.percentile(abs(zd * k - zr)[okr], 95) * 1e3)]}
        tgt = torch.tensor(Vc, dtype=torch.float32, device="cuda")
        flip = torch.tensor([1.0 if want else -1.0, 1.0, 1.0], device="cuda")
        n = len(fr)
        # init: rest hand, per-frame Kabsch rotation onto the (un-mirrored) target
        with torch.no_grad():
            v0 = mano.layer(torch.zeros(1, 48, device="cuda"), torch.zeros(1, 10, device="cuda")).verts[0]
        tu = tgt * flip
        a = v0 - v0.mean(0); bb = tu - tu.mean(1, keepdim=True)
        U, _, Vt = torch.linalg.svd(bb.transpose(1, 2) @ a[None])
        Dg = torch.diag_embed(torch.stack([torch.ones(n, device="cuda"), torch.ones(n, device="cuda"), torch.sign(torch.linalg.det(U @ Vt))], 1))
        Rk = U @ Dg @ Vt
        out_m = np.zeros(n, bool)
        if despin:
            out_m = _spin_outliers(Rk.cpu().numpy()) | idbad
            if out_m.any() and (~out_m).sum() >= 2:   # init outliers from the nearest inlier (the fit then moves them)
                inl = np.where(~out_m)[0]
                near = inl[np.abs(np.arange(n)[:, None] - inl[None]).argmin(1)]
                Rk = Rk[torch.as_tensor(near, device="cuda")]
        go = matrix_to_axis_angle(Rk).clone().requires_grad_(True)
        wv = torch.tensor(~out_m, dtype=torch.float32, device="cuda")       # per-frame vertex-term weight
        wc = torch.tensor(out_m & ~idbad, dtype=torch.float32, device="cuda")   # spin outliers keep the centroid; wrong-hand frames nothing
        cons = torch.tensor(np.diff(fr) == 1, device="cuda")                 # consecutive-frame pairs
        acc_ok = (cons[:-1] & cons[1:]).float()
        hp = torch.zeros(n, 45, device="cuda", requires_grad=True)
        betas = torch.zeros(1, 10, device="cuda", requires_grad=True)
        Tt = torch.zeros(n, 3, device="cuda", requires_grad=True)

        def fwd():
            o = mano.layer(torch.cat([go, hp], 1), betas.expand(n, 10))
            vv, jj = o.verts * flip, o.joints * flip
            return vv + Tt[:, None], jj + Tt[:, None]
        def objective():
            vv, jj = fwd()
            d = ((vv - tgt) ** 2).sum(-1).mean(1)
            l = (wv * d).sum() / wv.sum().clamp(min=1) + 1e-6 * (betas ** 2).sum()
            if despin:
                l = l + (wc * ((vv.mean(1) - tgt.mean(1)) ** 2).sum(-1)).sum() / wv.sum().clamp(min=1)   # outliers: centroid only
                if n >= 3:
                    acc = lambda x: ((x[:-2] - 2 * x[1:-1] + x[2:]) ** 2).flatten(1).sum(1)
                    Rm = axis_angle_to_matrix(go)
                    l = l + SMOOTH_W * ((acc_ok * (acc(jj) / 21 + 1e-4 * acc(Rm) + 1e-5 * acc(hp))).sum() / acc_ok.sum().clamp(min=1))
            return l
        with torch.no_grad():
            Tt += tgt.mean(1) - fwd()[0].mean(1)
        opt = torch.optim.Adam([go, hp, betas, Tt], lr=0.02)
        for it in range(iters):
            opt.zero_grad()
            loss = objective()
            loss.backward(); opt.step()
        opt = torch.optim.LBFGS([go, hp, betas, Tt], lr=1, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = objective()
            l.backward(); return l
        opt.step(closure)
        with torch.no_grad():
            vv, jj = fwd()
            err = (vv - tgt).norm(dim=-1)
        sd = out / side
        if sd.exists():
            import shutil
            shutil.rmtree(sd)
        sd.mkdir(parents=True)
        G, H, Bt, TT, J = (x.detach().cpu().numpy() for x in (go, hp, betas[0], Tt, jj))
        for i, f in enumerate(fr):
            rec = {"mano": {"betas": Bt.tolist(), "global_orient": G[i].tolist(), "hand_pose": H[i].tolist()},
                   "is_right": want, "T": TT[i].tolist(), "joints": J[i].tolist(),
                   "has_triangulation": bool(vis[b, f]), "has_wilor_2d": bool(vis[b, f]), "source": "dynhamr"}
            (sd / f"{f:06d}.json").write_text(json.dumps(rec))
        rep["sides"][side] = {"track": int(b), "frames": [int(fr[0]), int(fr[-1])], "written": int(n),
                              "detected": int(vis[b, fr].sum()), "fit_vertex_err_mm": {"mean": float(err.mean() * 1000),
                              "p99": float(torch.quantile(err.flatten(), 0.99) * 1000), "max": float(err.max() * 1000)}}
        if despin:
            ein = err[torch.as_tensor(~out_m, device="cuda")]
            rd = raydev[np.isfinite(raydev)] if raydev is not None else np.zeros(0)
            rep["sides"][side]["despin"] = {"outlier_frames": fr[out_m].tolist(), "wrong_hand_frames": fr[idbad].tolist(),
                                            "ray_dev_deg_med_p95": [float(np.median(rd)), float(np.percentile(rd, 95))] if len(rd) else None,
                                            "unreliable": bool(idbad.mean() > 0.3 or out_m.mean() > 0.4),
                                            "inlier_vertex_err_mm": [float(ein.mean() * 1000), float(torch.quantile(ein.flatten(), 0.95) * 1000)]}
        if kinfo:
            rep["sides"][side]["depth_scale"] = kinfo
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


SPIN_DEG, SPIN_WIN = 45.0, 7     # outlier: > SPIN_DEG from the robust mean orientation of the +-SPIN_WIN frame window
SMOOTH_W = float(os.environ.get("DH_SMOOTH_W", 0.3))                   # acceleration weight (m^2 units like the vertex term); only matters where data is weak
RAY_DEG = 4.0                    # Dyn-HaMR wrist ray vs our stereo wrist (~4 cm at 60 cm): beyond it, a different hand


def _ref_wrist(stereo: Path, ref: str, side: str, fr: np.ndarray) -> np.ndarray:
    """(n,3) camera-frame wrist (joints[0]) of the reference hands; NaN where missing."""
    w = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        p = stereo / ref / side / f"{f:06d}.json"
        if p.exists():
            w[i] = json.loads(p.read_text())["joints"][0]
    return w


def _spin_outliers(R: np.ndarray) -> np.ndarray:
    """(n,3,3) per-frame target orientations -> bool outliers. Two passes: the window mean is recomputed without the
    first pass's outliers, so a 4-frame flip cannot drag its own reference."""
    from scipy.spatial.transform import Rotation as Rot
    n = len(R); rot = Rot.from_matrix(R); bad = np.zeros(n, bool)
    for _ in range(2):
        dev = np.zeros(n)
        for i in range(n):
            idx = [j for j in range(max(0, i - SPIN_WIN), min(n, i + SPIN_WIN + 1)) if not bad[j] or j == i]
            ref = rot[[j for j in idx if j != i] or idx].mean()
            dev[i] = np.degrees((rot[i] * ref.inv()).magnitude())
        bad = dev > SPIN_DEG
    return bad


def _depth_ratio(stereo: Path, ref: str, side: str, fr: np.ndarray, zd: np.ndarray, sigma: float, exclude=None):
    """Per-frame k = z_ref / z of the wrist, from <ref>'s wrist (joints[0], camera frame); log k Gaussian-smoothed over
    time (sigma frames) so the correction adds no jitter; frames without a <ref> wrist take k interpolated between the
    nearest anchored frames. Returns (k, z_ref, anchored mask) or (None, ...) if fewer than 2 anchors."""
    from scipy.ndimage import gaussian_filter1d
    zr = np.full(len(fr), np.nan)
    for i, f in enumerate(fr):
        p = stereo / ref / side / f"{f:06d}.json"
        if p.exists():
            zr[i] = json.loads(p.read_text())["joints"][0][2]
    ok = np.isfinite(zr) & (zr > 0.05) & np.isfinite(zd) & (zd > 0.05)   # a degenerate ref wrist (z <= 0) would NaN every k
    if exclude is not None:
        ok &= ~exclude
    if ok.sum() < 2:
        return None, zr, ok
    lk = np.interp(fr, fr[ok], np.log(zr[ok] / zd[ok]))
    return np.exp(gaussian_filter1d(lk, sigma, mode="nearest")), zr, ok


def anchor_depth(run_name: str, variant: str = "dynhamr", ref: str = "mano_fit_hop_phys", out_variant: str = "dynhamr_depth",
                 sigma: float = 2.0) -> dict:
    """Keep Dyn-HaMR's hand (articulation, shape, orientation, image position) and take the metric DEPTH of its wrist
    from our stereo-fitted hands (<ref>): per frame the hand slides along the camera ray through its wrist,
    wrist -> k * wrist with k = z_ref / z_dynhamr (the wrist's projection is unchanged). log k is Gaussian-smoothed
    over time (sigma frames) so the correction adds no jitter; frames without a <ref> wrist take k interpolated
    between the nearest anchored frames (a depth-scale correction only, the hand itself stays Dyn-HaMR's)."""
    stereo = ROOT / "runs" / run_name / "stereo"
    rep = {"variant": variant, "ref": ref, "sigma_frames": sigma, "sides": {}}
    for side in ("right", "left"):
        src = sorted((stereo / variant / side).glob("[0-9]*.json"))
        if not src:
            continue
        recs = {int(p.stem): json.loads(p.read_text()) for p in src}
        fr = np.array(sorted(recs))
        zd = np.array([recs[f]["joints"][0][2] for f in fr])
        k, zr, ok = _depth_ratio(stereo, ref, side, fr, zd, sigma)
        if k is None:
            continue
        od = stereo / out_variant / side
        if od.exists():
            import shutil
            shutil.rmtree(od)
        od.mkdir(parents=True)
        for i, f in enumerate(fr):
            r = recs[f]
            w = np.asarray(r["joints"][0])
            d = (k[i] - 1.0) * w
            r["T"] = (np.asarray(r["T"]) + d).tolist()
            r["joints"] = (np.asarray(r["joints"]) + d).tolist()
            r["source"] = f"dynhamr pose, wrist depth from {ref}"
            (od / f"{f:06d}.json").write_text(json.dumps(r))
        resid = (zd * k - zr)[ok]
        rep["sides"][side] = {"frames": int(len(fr)), "anchored": int(ok.sum()),
                              "depth_offset_before_mm": {"median": float(np.median((zd - zr)[ok]) * 1e3),
                                                         "p95_abs": float(np.percentile(abs(zd - zr)[ok], 95) * 1e3)},
                              "k": {"min": float(k.min()), "median": float(np.median(k)), "max": float(k.max())},
                              "wrist_depth_resid_after_mm": {"median_abs": float(np.median(abs(resid)) * 1e3),
                                                             "p95_abs": float(np.percentile(abs(resid), 95) * 1e3)}}
    (stereo / out_variant).mkdir(parents=True, exist_ok=True)
    (stereo / out_variant / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        print(json.dumps(prepare(sys.argv[2], sys.argv[3]), indent=1))
    elif sys.argv[1] == "convert":
        print(json.dumps(convert(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or ["dynhamr"]), depth_ref=(sys.argv[5:6] or [None])[0],
                                 despin=sys.argv[6:7] == ["despin"]), indent=1))
    elif sys.argv[1] == "prior":
        print(json.dumps(operator_prior(sys.argv[2], sys.argv[3], *sys.argv[4:5])))
    elif sys.argv[1] == "anchor":
        print(json.dumps(anchor_depth(sys.argv[2], *sys.argv[3:6]), indent=1))
