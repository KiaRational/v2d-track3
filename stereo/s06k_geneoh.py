"""GeneOH-Diffusion clean-up of the hand-object interaction (Liu & Yi, ICLR 2024), with unobserved frames left empty.

A frame where the hand was not detected has NO data: it is never filled by holding or interpolating a pose. Every
frame is a free variable of one optimisation; only observed frames carry a data term. Per side (left hands mirrored
in x into the right-hand model GeneOH was trained on):
  A  fill     MANO params for all frames: joint data term on observed frames only, joint/pose acceleration on all
              frames, velocity damping on leading/trailing frames (no observation on one side: the hand coasts to a
              stop instead of freezing or extrapolating away). Gap frames start from a slerp/lerp *initial guess*
              only; the optimiser, not the guess, decides them.
  B  GeneOH   world-frame hand + object sequence (object = the one the hand is closest to, per 60-frame window,
              only where the hand comes within --contact_m of it) -> the TACO path of GeneOH (spatial diffusion,
              SDEdit from t=100, 60-frame windows, stride 30), denoised joints back in the world frame, windows
              blended with triangular weights. The object trajectory is a fixed condition (never modified).
  C  fit      MANO params to the denoised joints (observed frames weight 1, filled frames 1/4: GeneOH saw only the
              optimiser's guess there), acceleration on all frames, pose prior to stage A.
Writes <run>/stereo/<out>/{left,right}/<t>.json for EVERY frame (same schema as mano_fit_*, plus "observed" and
"filled_by": null | "optimizer" | "geneoh") and report.json (counts, change on observed frames, GeneOH windows).
Runs in the geneoh image with cwd = the GeneOH-Diffusion repo root.
Usage: python s06k_geneoh.py <run> [--hands mano_fit_hop_phys] [--out mano_fit_hop_geneoh] [--seeds 0]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
GENEOH = Path(os.environ.get("GENEOH_ROOT", ROOT / "third_party/GeneOH-Diffusion"))
sys.path.insert(0, str(GENEOH))
from manopth.manolayer import ManoLayer  # noqa: E402  (GeneOH's patched manopth: required by its models)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot"))
from export_result_bundle import spurious_hand_frames  # noqa: E402

DEV = "cuda"
F = np.diag([-1.0, 1.0, 1.0])
WS, STRIDE = 60, 30  # GeneOH TACO windows (sample/predict_taco.py)


def log(*a):
    print(*a, flush=True)


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:  # as export_result_bundle.load_T
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


class Hand:
    """GeneOH's right MANO layer (axis-angle, no PCA). Joints in this layer's order, metres; the global rotation is
    applied about the rest root joint J0 (as in our fits), translation added in metres."""

    def __init__(self, betas, hp_offset):
        self.layer = ManoLayer(mano_root=str(GENEOH / "manopth/mano/models"), side="right", use_pca=False,
                               ncomps=45, flat_hand_mean=False).to(DEV)
        self.faces = self.layer.th_faces.cpu().numpy()
        self.b = torch.tensor(betas, dtype=torch.float32, device=DEV)[None]
        self.hp_offset = torch.tensor(hp_offset, dtype=torch.float32, device=DEV)  # ours -> this layer's hand_pose
        z = torch.zeros(1, 48, device=DEV)
        v, j = self.layer(z, self.b, torch.zeros(1, 3, device=DEV))
        self.scale = 1000.0 if float(v.abs().max()) > 10 else 1.0  # this manopth works in mm
        self.J0 = (j[0, 0] / self.scale).detach()

    def __call__(self, R, hp, T, verts=False):
        """R (n,3,3) world rotation, hp (n,45) OUR hand_pose convention, T (n,3) metres."""
        n = hp.shape[0]
        pose = torch.cat([torch.zeros(n, 3, device=DEV), hp + self.hp_offset], 1)
        v, j = self.layer(pose, self.b.expand(n, -1), torch.zeros(n, 3, device=DEV))
        v, j = v / self.scale - self.J0, j / self.scale - self.J0
        j = torch.einsum("nij,nkj->nki", R, j) + self.J0 + T[:, None]
        if not verts:
            return j
        return j, torch.einsum("nij,nkj->nki", R, v) + self.J0 + T[:, None]


def rotvec_to_R(r):
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / th
    K = torch.zeros(r.shape[0], 3, 3, device=r.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=r.device)[None] + s * K + (1 - c) * K @ K


def scene(run, a):
    """n, fps, c2w (gravity-aligned world, as export_result_bundle), objects {name: (o2c (n,4,4), points (P,3), faces)}."""
    meta = json.loads((run / "episode.json").read_text())
    fps = float(meta["fps"])
    n = len(list((run / "recon_hawor/frames").glob("*.png")))
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for p in (run / "recon_hawor/slam_poses").glob("*.json"):
        if 0 <= int(p.stem) < n:
            c2w[int(p.stem)] = load_T(p)
    tp = json.loads((run / "table_plane.json").read_text())
    A = np.eye(4)
    A[:3, :3] = Rotation.align_vectors([[0, 0, 1.0]], [np.array(tp["up"], float)])[0].as_matrix()
    A[2, 3] = -float(tp["height"])
    c2w = np.einsum("ij,njk->nik", A, c2w)
    objs = {}
    for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
        nm = o["name"]
        files = list((run / a.objects / nm / "poses").glob("*.json"))
        mesh_p = ROOT / "meshes_obj" / nm / f"{nm}.obj"
        if len(files) < n or not mesh_p.exists():
            log(f"object {nm}: {len(files)}/{n} poses, mesh {mesh_p.exists()} -> not used as GeneOH condition")
            continue
        o2c = np.tile(np.eye(4), (n, 1, 1))
        for p in files:
            if 0 <= int(p.stem) < n:
                o2c[int(p.stem)] = load_T(p)
        mesh = trimesh.load(mesh_p, force="mesh")
        np.random.seed(0)
        pts, _ = trimesh.sample.sample_surface(mesh, a.obj_points)
        objs[nm] = (o2c, np.asarray(pts, float), np.asarray(mesh.faces))
    return n, fps, c2w, objs


def read_side(run, hands, side, n, reject=()):
    """Observed frames only (spurious duplicate / isolated detections rejected as export does)."""
    recs = {}
    for f in sorted((run / "stereo" / hands / side).glob("*.json")):
        t = int(f.stem)
        if 0 <= t < n and t not in reject:
            r = json.loads(f.read_text())
            if all(np.isfinite(np.asarray(v, float)).all() for v in (r["mano"]["global_orient"], r["mano"]["hand_pose"], r["T"])):
                recs[t] = r
    return recs


def convention(recs, side):
    """hand_pose offset (ours -> GeneOH layer) and the permutation from this layer's joint order to ours, checked
    against the joints stored in the fit (left: x-mirrored right model). Refuses > 3 mm disagreement."""
    ts = sorted(recs)[:: max(1, len(recs) // 20)]
    r0 = recs[ts[0]]
    best = None
    lay = ManoLayer(mano_root=str(GENEOH / "manopth/mano/models"), side="right", use_pca=False, ncomps=45,
                    flat_hand_mean=False)
    mean = lay.th_hands_mean.reshape(-1).numpy()
    for name, off in (("same", np.zeros(45)), ("plus_mean", mean), ("minus_mean", -mean)):
        h = Hand(r0["mano"]["betas"], off)
        errs, perms = [], []
        for t in ts:
            r = recs[t]
            R = torch.tensor(Rotation.from_rotvec(r["mano"]["global_orient"]).as_matrix(), dtype=torch.float32, device=DEV)[None]
            hp = torch.tensor(r["mano"]["hand_pose"], dtype=torch.float32, device=DEV)[None]
            j = h(R, hp, torch.zeros(1, 3, device=DEV))[0].detach().cpu().numpy()
            if side == "left":
                j = j @ F
            j = j + np.asarray(r["T"])
            ours = np.asarray(r["joints"])
            d = np.linalg.norm(ours[:, None] - j[None], axis=-1)  # ours x layer
            perms.append(d.argmin(1))
            errs.append(d.min(1).mean())
        e = float(np.mean(errs)) * 1000
        perm = np.array([np.bincount(np.array(perms)[:, k]).argmax() for k in range(21)])
        log(f"[{side}] convention {name}: mean joint disagreement {e:.2f} mm, perm {perm.tolist()}")
        if best is None or e < best[0]:
            best = (e, name, off, perm)
    if best[0] > 3.0 or len(set(best[3].tolist())) != 21:
        raise RuntimeError(f"[{side}] cannot reproduce the stored joints with GeneOH's MANO ({best[0]:.1f} mm)")
    return best[2], best[3], best[1], best[0]


def init_fill(obs, R, hp, T):
    """Initial guess for the optimiser on unobserved frames (slerp/lerp between observed neighbours, nearest at the
    ends). Only an initialisation: nothing here is written out."""
    n = len(obs)
    idx = np.flatnonzero(obs)
    R, hp, T = R.copy(), hp.copy(), T.copy()
    miss = np.flatnonzero(~obs)
    if len(miss) == 0:  # every frame observed: nothing to initialise
        return R, hp, T
    if len(idx) == 1:
        R[miss], hp[miss], T[miss] = R[idx[0]], hp[idx[0]], T[idx[0]]
        return R, hp, T
    sl = Slerp(idx, Rotation.from_matrix(R[idx]))
    tt = np.clip(miss, idx[0], idx[-1])
    R[miss] = sl(tt).as_matrix()
    for k in range(45):
        hp[miss, k] = np.interp(tt, idx, hp[idx, k])
    for k in range(3):
        T[miss, k] = np.interp(tt, idx, T[idx, k])
    return R, hp, T


def optimise(hand, R0, hp0, T0, targets, w_t, obs, lead_trail, prior=None, iters=1500, sig=None):
    """Free variables on every frame. targets (n,21,3) world joints with per-frame weights w_t (0 = no data)."""
    sig = sig or {}
    n = len(R0)
    R0_t = torch.tensor(R0, dtype=torch.float32, device=DEV)
    dr = torch.zeros(n, 3, device=DEV, requires_grad=True)
    hp = torch.tensor(hp0, dtype=torch.float32, device=DEV).requires_grad_(True)
    T = torch.tensor(T0, dtype=torch.float32, device=DEV).requires_grad_(True)
    tgt = torch.tensor(targets, dtype=torch.float32, device=DEV)
    w = torch.tensor(w_t, dtype=torch.float32, device=DEV)
    lt = torch.tensor(lead_trail, dtype=torch.float32, device=DEV)
    hp_ref = torch.tensor(prior if prior is not None else hp0, dtype=torch.float32, device=DEV)
    s_d, s_a, s_v = sig.get("data", 0.008), sig.get("acc", 0.004), sig.get("vel", 0.006)
    s_pa, s_pr = sig.get("pose_acc", 0.05), sig.get("pose_prior", 0.5)
    opt = torch.optim.Adam([dr, hp, T], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-4)

    def losses():
        R = rotvec_to_R(dr) @ R0_t
        J = hand(R, hp, T)
        l_data = ((((J - tgt) ** 2).sum(-1).mean(-1)) * w).sum() / w.sum().clamp(min=1) / s_d ** 2
        acc = J[2:] - 2 * J[1:-1] + J[:-2]
        l_acc = (acc ** 2).sum(-1).mean() / s_a ** 2
        l_pacc = ((hp[2:] - 2 * hp[1:-1] + hp[:-2]) ** 2).mean() / s_pa ** 2
        vel = ((J[1:] - J[:-1]) ** 2).sum(-1).mean(-1)
        l_vel = (vel * torch.maximum(lt[1:], lt[:-1])).sum() / max(1.0, float(lt.sum())) / s_v ** 2
        l_prior = ((hp - hp_ref) ** 2).mean() / s_pr ** 2
        return l_data + l_acc + l_pacc + l_vel + 0.1 * l_prior, dict(data=l_data.item(), acc=l_acc.item(),
                                                                    pacc=l_pacc.item(), vel=l_vel.item())

    for it in range(iters):
        opt.zero_grad()
        loss, terms = losses()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == iters - 1:
            log(f"   it {it:5d} loss {loss.item():.3f} {terms}")
    with torch.no_grad():
        R = rotvec_to_R(dr) @ R0_t
        J, V = hand(R, hp, T, verts=True)
    return R.cpu().numpy(), hp.detach().cpu().numpy(), T.detach().cpu().numpy(), J.cpu().numpy(), V.cpu().numpy()


def run_geneoh(work, tag, R, hp, T, V, hand, o2w, pts, faces, seeds):
    """One full-length sequence through GeneOH's TACO predictor; returns {window start: (60,21,3) world joints}."""
    n = len(R)
    go = Rotation.from_matrix(R).as_rotvec()
    # GeneOH re-evaluates MANO from (rot_r, pose_r, trans_r, shape_r) and shifts it onto hand_verts[start]: feed our
    # world-frame params in its convention (hand_pose with the flat-hand offset).
    trans = T  # manopth rotates about the rest root joint and adds trans in metres before its mm scaling, like ours
    obj_v = np.einsum("nij,pj->npi", o2w[:, :3, :3], pts) + o2w[:, None, :3, 3]
    hp_g = hp + hand.hp_offset.cpu().numpy()
    d = {"rot_r": go.astype(np.float32), "pose_r": hp_g.astype(np.float32), "trans_r": trans.astype(np.float32),
         "shape_r": hand.b.cpu().numpy().astype(np.float32),
         "hand_pose": np.concatenate([go, hp_g], 1).astype(np.float32), "hand_trans": trans.astype(np.float32),
         "hand_shape": hand.b.cpu().numpy()[0].astype(np.float32), "hand_verts": V.astype(np.float32),
         "hand_faces": hand.faces, "obj_verts": obj_v.astype(np.float32), "obj_faces": faces, "obj_pose": o2w.astype(np.float32)}
    pkl = work / f"{tag}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(d, f)
    save = work / "pred"
    save.mkdir(exist_ok=True)
    env = dict(os.environ, GENEOH_PKL=str(pkl), GENEOH_TAG=tag, GENEOH_SAVE=str(save), GENEOH_USE_LEFT="",
               GENEOH_SEEDS=",".join(map(str, seeds)), PYTHONPATH=".")
    t0 = time.time()
    p = subprocess.run(["bash", "scripts/v2d_predict_taco.sh"], cwd=GENEOH, env=env, capture_output=True, text=True)
    (work / f"{tag}.log").write_text(p.stdout[-200000:] + "\n---stderr---\n" + p.stderr[-200000:])
    if p.returncode != 0:
        raise RuntimeError(f"GeneOH failed for {tag} (see {work / (tag + '.log')}): {p.stderr[-2000:]}")
    log(f"   GeneOH {tag}: {time.time() - t0:.0f} s")
    st = list(range(0, max(1, n - WS + 1), STRIDE))
    if st[-1] + WS < n:
        st.append(n - WS)
    out = {}
    for s in st:
        acc = []
        for seed in seeds:
            r = np.load(save / f"predicted_infos_seed_{seed}_tag_{tag}_st_{s}.npy", allow_pickle=True).item()
            rot, tr = r["tot_obj_rot"][0], r["tot_obj_transl"][0]  # (ws,3,3) = R^T, (ws,3) centroid
            o = r["outputs"]
            o = o[-len(rot):] if len(o) >= len(rot) else o
            wj = np.einsum("tkj,tji->tki", o, rot) + tr[:, None]
            tj = np.einsum("tkj,tji->tki", r["tot_rhand_joints"][0], rot) + tr[:, None] \
                if r["tot_rhand_joints"].ndim == 4 else None
            acc.append(wj)
            if tj is not None and seed == seeds[0]:
                out.setdefault("_input_check", []).append(tj)
        out[s] = np.mean(acc, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", default="mano_fit_hop_phys")
    ap.add_argument("--objects", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--out", default="mano_fit_hop_geneoh")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--contact_m", type=float, default=0.10, help="GeneOH only where the hand comes this close")
    ap.add_argument("--obj_points", type=int, default=3000)
    ap.add_argument("--no_geneoh", action="store_true", help="stage A only: optimiser fills undetected frames, detected frames kept as input")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    run = ROOT / "runs" / a.run
    n, fps, c2w, objs = scene(run, a)
    outdir = run / "stereo" / a.out
    work = ROOT / "tmp_diag/geneoh_work" / a.run
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    report = {"args": vars(a), "n_frames": n, "fps": fps, "objects": list(objs), "sides": {}}
    if outdir.exists():
        shutil.rmtree(outdir)
    spurious = spurious_hand_frames(run, a.hands, n)
    report["rejected_spurious"] = {s: sorted(int(t) for t in v) for s, v in spurious.items()}
    for side in ("left", "right"):
        recs = read_side(run, a.hands, side, n, spurious[side])
        if not recs:
            log(f"[{side}] no observations: nothing written (export keeps the side empty)")
            report["sides"][side] = {"observed": 0}
            continue
        off, perm, conv_name, conv_mm = convention(recs, side)
        hand = Hand(next(iter(recs.values()))["mano"]["betas"], off)
        M = F if side == "left" else np.eye(3)  # mirror left hands (and the whole scene) into the right-hand model
        c2w_m = c2w.copy()
        c2w_m[:, :3, :3] = np.einsum("ij,njk,kl->nil", M, c2w[:, :3, :3], M)
        c2w_m[:, :3, 3] = c2w[:, :3, 3] @ M.T
        Rc, tc = c2w_m[:, :3, :3], c2w_m[:, :3, 3]
        obs = np.zeros(n, bool)
        R = np.tile(np.eye(3), (n, 1, 1))
        hp, T = np.zeros((n, 45)), np.zeros((n, 3))
        J0 = hand.J0.cpu().numpy()
        for t, r in recs.items():
            obs[t] = True
            Rh = Rotation.from_rotvec(r["mano"]["global_orient"]).as_matrix()  # right-model param space (both sides)
            Tm = M @ np.asarray(r["T"], float)
            R[t] = Rc[t] @ Rh
            T[t] = Rc[t] @ (J0 + Tm) + tc[t] - J0
            hp[t] = r["mano"]["hand_pose"]
        idx = np.flatnonzero(obs)
        lead_trail = np.zeros(n)
        lead_trail[: idx[0]] = 1
        lead_trail[idx[-1] + 1:] = 1
        with torch.no_grad():
            J_obs = hand(torch.tensor(R, dtype=torch.float32, device=DEV), torch.tensor(hp, dtype=torch.float32, device=DEV),
                         torch.tensor(T, dtype=torch.float32, device=DEV)).cpu().numpy()
        Ri, hpi, Ti = init_fill(obs, R, hp, T)
        log(f"[{side}] {obs.sum()}/{n} observed ({conv_name}, {conv_mm:.2f} mm); stage A (fill)")
        RA, hpA, TA, JA, VA = optimise(hand, Ri, hpi, Ti, J_obs, obs.astype(float), obs, lead_trail)
        # stage B: GeneOH where the hand interacts with an object
        o2w = {nm: np.einsum("nij,njk->nik", c2w_m, np.einsum("ij,njk,kl->nil", np.diag([*np.diag(M), 1.0]), o2c,
                                                             np.diag([*np.diag(M), 1.0])))
               for nm, (o2c, _, _) in objs.items()}
        pts_m = {nm: p @ M.T for nm, (_, p, _) in objs.items()}  # mirrored mesh points (object frame)
        dist = {}
        for nm in objs:
            ow = np.einsum("nij,pj->npi", o2w[nm][:, :3, :3], pts_m[nm][::10]) + o2w[nm][:, None, :3, 3]
            dist[nm] = np.array([np.linalg.norm(JA[t][:, None] - ow[t][None], axis=-1).min() for t in range(n)])
        st = list(range(0, max(1, n - WS + 1), STRIDE))
        if st[-1] + WS < n:
            st.append(n - WS)
        win_obj = {}
        for s in st:
            sl = slice(s, s + WS)
            if not dist:
                continue
            nm = min(dist, key=lambda k: dist[k][sl].mean())
            if dist[nm][sl].min() <= a.contact_m:
                win_obj[s] = nm
        tgt_sum, w_sum = np.zeros((n, 21, 3)), np.zeros(n)
        windows_rep = {}
        if win_obj and not a.no_geneoh:
            for nm in sorted(set(win_obj.values())):
                faces = objs[nm][2]
                if side == "left":
                    faces = faces[:, [0, 2, 1]]
                res = run_geneoh(work, f"{side}_{nm}", RA, hpA, TA, VA, hand, o2w[nm], pts_m[nm], faces, seeds)
                chk = res.pop("_input_check", None)
                for s, nm_s in win_obj.items():
                    if nm_s != nm:
                        continue
                    wj = res[s]
                    L = min(WS, n - s)
                    tri = 1.0 - np.abs(np.arange(L) - (L - 1) / 2) / (L / 2) + 1e-3
                    tgt_sum[s:s + L] += wj[:L] * tri[:, None, None]
                    w_sum[s:s + L] += tri
                    windows_rep[s] = {"object": nm, "min_dist_m": float(dist[nm][s:s + L].min()),
                                      "change_mm": float(np.linalg.norm(wj[:L] - JA[s:s + L], axis=-1).mean() * 1000)}
                if chk is not None:
                    e = [np.linalg.norm(c[: min(WS, n - s)] - JA[s:s + min(WS, n - s)], axis=-1).mean() * 1000
                         for c, s in zip(chk, st)]
                    log(f"   GeneOH input vs stage A joints: {np.round(e, 2).tolist()} mm (should be ~0)")
        gen = w_sum > 0
        log(f"[{side}] GeneOH frames {gen.sum()}/{n}, windows {windows_rep}")
        if gen.any():
            tgt = JA.copy()
            tgt[gen] = tgt_sum[gen] / w_sum[gen][:, None, None]
            wt = np.where(obs, 1.0, 0.25)
            log(f"[{side}] stage C (fit to GeneOH joints)")
            RC, hpC, TC, JC, _ = optimise(hand, RA, hpA, TA, tgt, wt, obs, lead_trail, prior=hpA, iters=1500,
                                          sig={"data": 0.006})
        else:
            RC, hpC, TC, JC = RA.copy(), hpA.copy(), TA.copy(), JA.copy()
        if a.no_geneoh:  # detected frames keep their input pose exactly (joint-only refits twist the fingers)
            RC[obs], hpC[obs], TC[obs], JC[obs] = R[obs], hp[obs], T[obs], J_obs[obs]
        # back to the camera frame, our convention (left: un-mirror)
        sd = outdir / side
        sd.mkdir(parents=True)
        for t in range(n):
            Rh = Rc[t].T @ RC[t]
            Tm = Rc[t].T @ (J0 + TC[t] - tc[t]) - J0
            jc = (JC[t] - tc[t]) @ Rc[t]  # camera frame (mirrored)
            jc = jc @ M.T
            src = recs.get(t)
            rec = {"mano": {"betas": hand.b[0].tolist(), "global_orient": Rotation.from_matrix(Rh).as_rotvec().tolist(),
                            "hand_pose": hpC[t].tolist()},
                   "is_right": side == "right", "T": (M @ Tm).tolist(), "joints": jc[perm].tolist(),
                   "has_triangulation": bool(src and src.get("has_triangulation")),
                   "has_wilor_2d": bool(src and src.get("has_wilor_2d")), "observed": bool(obs[t]),
                   "filled_by": None if obs[t] else ("geneoh" if gen[t] else "optimizer")}
            (sd / f"{t:06d}.json").write_text(json.dumps(rec))
        ch_obs = np.linalg.norm(JC[obs] - J_obs[obs], axis=-1).mean() * 1000
        acc = lambda J: float(np.linalg.norm(J[2:] - 2 * J[1:-1] + J[:-2], axis=-1).mean() * 1000)  # noqa: E731
        runs_missing = []
        t = 0
        while t < n:
            if obs[t]:
                t += 1
                continue
            s = t
            while t < n and not obs[t]:
                t += 1
            runs_missing.append([s, t - 1, "lead" if s == 0 else "trail" if t == n else "interior"])
        report["sides"][side] = {
            "observed": int(obs.sum()), "filled_optimizer": int((~obs & ~gen).sum()), "filled_geneoh": int((~obs & gen).sum()),
            "geneoh_frames": int(gen.sum()), "missing_runs": runs_missing, "convention": conv_name,
            "convention_mm": conv_mm, "change_on_observed_mm": float(ch_obs),
            "jitter_obs_input_mm_per_frame2": acc(J_obs[idx[0]:idx[-1] + 1]) if obs.all() else None,
            "jitter_stageA_mm_per_frame2": acc(JA), "jitter_final_mm_per_frame2": acc(JC), "windows": windows_rep}
        log(f"[{side}] done: {json.dumps({k: v for k, v in report['sides'][side].items() if k != 'windows'})}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {outdir}")


if __name__ == "__main__":
    main()
