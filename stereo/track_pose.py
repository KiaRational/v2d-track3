"""Object trajectories from FoundationPose candidates + SAM 3 masks + TrackEverything 3D point tracks.

Inputs per object: every FoundationPose candidate per frame (fp_hyps.py, ~252, with ScoreNet scores), the SAM 3 object
and hand masks, object-surface 3D point tracks (te_obj_tracks.py, SLAM world), our mesh.

1 Viterbi over candidates (one per frame), cost = sum unary + sum pairwise:
   unary(t, k)  = [1 - score_norm x IoU]                    silhouette: projected candidate vs object mask, HAND pixels
                                                            excluded from both (a covering hand is not evidence)
                + W_PTS x mean(min(d, D_PTS)) / D_PTS        tracked points visible at t vs the candidate's surface
   pair(i -> j) = W_TRK x mean_k min(|M_ij a_k - b_k|, D_TRK) / D_TRK
                                                            M_ij = X_j X_i^-1 must move the points tracked between the
                                                            two frames exactly as they moved (spin / flip = huge cost)
                + W_SM x (|dp| / 2 cm + |drot| / 0.3 rad) / gap   weak smoothness (drot min over the symmetry group)
   With < MIN_COMMON common tracked points the smoothness term is weighted up instead (W_SM_NOTRK).
2 Long-range refinement (pose from points): every track gets object-frame coordinates q_n = median over its first
   visible frames of X_t^-1 p_n(t) (Viterbi poses); then per frame X_t = argmin sum_n huber(|X q_n - p_n(t)|, 1 cm)
   + |Log(X_vit^-1 X)|^2 / prior -- points seen at frame 50 and again at 200 must land on the same mesh spots, so a
   slow spin or flip cannot survive. Then light temporal smoothing (1 frame).
Frames without candidates are interpolated. Writes <run>/<out>_vit/<obj>/poses and <run>/<out>/<obj>/poses
(camera<-object) + track_pose_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python track_pose.py <run>... [--out tp_recon]"""
from __future__ import annotations

import argparse
import json
import time
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from fp_viterbi import rot_dist
from handlock import ang, interp_poses, smooth_poses
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
DS = 4
W_PTS, D_PTS = 1.0, 0.02
W_TRK, D_TRK, MIN_COMMON = 2.0, 0.03, 6
W_SM, W_SM_NOTRK = 0.1, 0.5
MAX_PAIR_PTS, MAX_UNARY_PTS, MAX_REF_PTS = 64, 128, 200   # random subsets (fixed seed): ample to tell a flip / spin
HUBER_M, PRIOR_RAD, PRIOR_M = 0.01, 0.15, 0.02


def silhouette_iou(P, S, mo, care, k):
    X = np.einsum("nij,pj->npi", P[:, :3, :3], S) + P[:, None, :3, 3]
    Hs, Ws = mo.shape
    u = (X[..., 0] / X[..., 2] * k["fx"] + k["cx"]) / DS
    v = (X[..., 1] / X[..., 2] * k["fy"] + k["cy"]) / DS
    # all candidates at once: splat (N, Hs, Ws), 3x3 dilation by shifted ORs, IoU over the cared-for pixels
    ok = (X[..., 2] > 0.05) & (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
    ci, pi = np.nonzero(ok)
    r = np.zeros((len(P), Hs + 2, Ws + 2), bool)
    r[ci, v[ci, pi].astype(int) + 1, u[ci, pi].astype(int) + 1] = True
    d = np.zeros((len(P), Hs, Ws), bool)
    for dy in range(3):
        for dx in range(3):
            d |= r[:, dy:dy + Hs, dx:dx + Ws]
    inter = (d & mo & care).sum((1, 2))
    uni = ((d | mo) & care).sum((1, 2))
    return np.where(uni > 0, inter / np.maximum(uni, 1), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="tp_recon")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        k = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        tz = np.load(run / obj_dir / "te_tracks.npz")
        TP, TV, TO, TQ = tz["P_world"], tz["vis"], tz["owner"], tz["queries"]
        tnames = list(tz["names"])
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm, oid = o["name"], o["id"]
            hd = run / obj_dir / nm / "fp_hyps"
            if not hd.is_dir() or nm not in tnames:
                continue
            mesh = trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh")
            S = trimesh.sample.sample_surface(mesh, 1500, seed=0)[0]
            Stree = cKDTree(trimesh.sample.sample_surface(mesh, 6000, seed=1)[0])
            group = [np.asarray(G) for G in mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")["group"]]
            if len(group) > 8:
                group = group[:: len(group) // 8]
            sel = np.nonzero(TO == tnames.index(nm))[0]
            rng = np.random.default_rng(0)
            t_start = time.time()
            P_tr, V_tr, t0_tr = TP[:n, sel].astype(float), TV[:n, sel], TQ[sel, 0].astype(int)
            # ---- candidates + unary
            ts, Xw, U = [], [], []
            for p in sorted(hd.glob("*.npz")):
                t = int(p.stem)
                if t not in slam or t >= n:
                    continue
                z = np.load(p)
                P, sc = z["poses"].astype(float), z["scores"].astype(float)
                m = cv2.imread(str(run / obj_dir / "masks" / str(oid) / f"{t:06d}.png"), 0)
                if m is None:
                    continue
                hm = np.zeros_like(m, bool)
                for hid in (2, 3):
                    h = cv2.imread(str(run / obj_dir / "masks" / str(hid) / f"{t:06d}.png"), 0)
                    if h is not None:
                        hm |= h > 127
                Hs, Ws = m.shape[0] // DS, m.shape[1] // DS
                mo = cv2.resize((m > 127).astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0
                care = ~(cv2.resize(hm.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0)
                iou = silhouette_iou(P, S, mo, care, k)
                s = (sc - sc.min()) / max(sc.max() - sc.min(), 1e-9)
                W_ = np.einsum("ij,njk->nik", slam[t], P)
                u = 1.0 - s * iou
                pts = P_tr[t][V_tr[t] & np.isfinite(P_tr[t][:, 0])]
                if len(pts) > MAX_UNARY_PTS:
                    pts = pts[rng.choice(len(pts), MAX_UNARY_PTS, replace=False)]
                if len(pts) >= MIN_COMMON:   # tracked surface points must lie on the candidate's surface
                    q = np.einsum("nji,nkj->nki", W_[:, :3, :3], pts[None] - W_[:, None, :3, 3])
                    d = Stree.query(q.reshape(-1, 3))[0].reshape(len(W_), -1)
                    u = u + W_PTS * np.minimum(d, D_PTS).mean(1) / D_PTS
                ts.append(t)
                Xw.append(W_)
                if len(ts) % 50 == 0:
                    print(f"{rn} {nm} unary {len(ts)} frames {time.time() - t_start:.0f} s", flush=True)
                U.append(u)
            if len(ts) < 2:
                continue
            # ---- Viterbi
            acc, back, n_trk = U[0].copy(), [], 0
            for i in range(1, len(ts)):
                ta, tb = ts[i - 1], ts[i]
                A, B = Xw[i - 1], Xw[i]
                gap = tb - ta
                dp = np.linalg.norm(A[:, None, :3, 3] - B[None, :, :3, 3], axis=-1)
                dr = rot_dist(A[:, :3, :3], B[:, :3, :3], group)
                sm = (dp / 0.02 + dr / 0.3) / gap
                com = V_tr[ta] & V_tr[tb] & np.isfinite(P_tr[ta][:, 0]) & np.isfinite(P_tr[tb][:, 0])
                if com.sum() >= MIN_COMMON:
                    a_, b_ = P_tr[ta][com], P_tr[tb][com]
                    if len(a_) > MAX_PAIR_PTS:
                        kk = rng.choice(len(a_), MAX_PAIR_PTS, replace=False)
                        a_, b_ = a_[kk], b_[kk]
                    # object-frame coords under every candidate at ta, re-posed by every candidate at tb
                    qa = np.einsum("nji,nkj->nki", A[:, :3, :3], a_[None] - A[:, None, :3, 3])     # (Na,K,3)
                    cost = np.zeros((len(A), len(B)))
                    for c0 in range(0, len(A), 32):
                        pr = np.einsum("bij,akj->abki", B[:, :3, :3], qa[c0:c0 + 32]) + B[None, :, None, :3, 3]
                        d = np.linalg.norm(pr - b_[None, None], axis=-1)
                        cost[c0:c0 + 32] = np.minimum(d, D_TRK).mean(-1) / D_TRK
                    tot = acc[:, None] + W_TRK * cost + W_SM * sm
                    n_trk += 1
                else:
                    tot = acc[:, None] + W_SM_NOTRK * sm
                back.append(tot.argmin(0))
                acc = tot.min(0) + U[i]
            print(f"{rn} {nm} viterbi done {time.time() - t_start:.0f} s", flush=True)
            pick = [int(acc.argmin())]
            for i in range(len(ts) - 1, 0, -1):
                pick.append(int(back[i - 1][pick[-1]]))
            pick = pick[::-1]
            vit = interp_poses({t: Xw[i][j] for i, (t, j) in enumerate(zip(ts, pick))}, n)
            # ---- long-range refinement: tracks anchored in the object frame
            Q = np.full((len(sel), 3), np.nan)
            for j in range(len(sel)):
                fr = [t for t in range(t0_tr[j], min(n, t0_tr[j] + 5)) if V_tr[t, j] and np.isfinite(P_tr[t, j, 0])]
                if fr:
                    Q[j] = np.median([np.linalg.inv(vit[t]) @ np.r_[P_tr[t, j], 1] for t in fr], 0)[:3]
            ref, n_used = {}, []
            for t in range(n):
                ok = V_tr[t] & np.isfinite(P_tr[t][:, 0]) & np.isfinite(Q[:, 0])
                X0 = vit[t]
                if ok.sum() < MIN_COMMON:
                    ref[t] = X0
                    n_used.append(0)
                    continue
                q, p = Q[ok], P_tr[t][ok]
                if len(q) > MAX_REF_PTS:
                    kk = rng.choice(len(q), MAX_REF_PTS, replace=False)
                    q, p = q[kk], p[kk]

                def res(x):
                    R = Rotation.from_rotvec(x[:3]).as_matrix() @ X0[:3, :3]
                    tt = X0[:3, 3] + x[3:]
                    return np.r_[((q @ R.T + tt) - p).ravel() / HUBER_M,
                                 x[:3] / PRIOR_RAD * np.sqrt(len(q)) * 0.3, x[3:] / PRIOR_M * np.sqrt(len(q)) * 0.3]
                sol = least_squares(res, np.zeros(6), loss="huber", f_scale=1.0, max_nfev=50)
                M = np.eye(4)
                M[:3, :3] = Rotation.from_rotvec(sol.x[:3]).as_matrix() @ X0[:3, :3]
                M[:3, 3] = X0[:3, 3] + sol.x[3:]
                ref[t] = M
                n_used.append(int(ok.sum()))
            fin = smooth_poses(ref, 1.0)
            for variant, poses in ((a.out + "_vit", vit), (a.out, fin)):
                od = run / variant / nm / "poses"
                od.mkdir(parents=True, exist_ok=True)
                for t in range(n):
                    C = np.linalg.inv(slam[t]) @ poses[t]
                    (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st_v = [ang(vit[t + 1][:3, :3], vit[t][:3, :3]) for t in range(n - 1)]
            st_f = [ang(fin[t + 1][:3, :3], fin[t][:3, :3]) for t in range(n - 1)]
            report[nm] = {"frames_with_hyps": len(ts), "pairs_with_tracks": n_trk, "tracks": int(len(sel)),
                          "argmax_kept_frac": round(float(np.mean([j == 0 for j in pick])), 3),
                          "points_per_frame_med": int(np.median(n_used)),
                          "vit_jumps10": int(sum(x > 10 for x in st_v)), "vit_max_step": round(max(st_v), 1),
                          "final_jumps10": int(sum(x > 10 for x in st_f)), "final_max_step": round(max(st_f), 1),
                          "refine_change_deg_med": round(float(np.median([ang(fin[t][:3, :3], vit[t][:3, :3]) for t in range(n)])), 1), "seconds": round(time.time() - t_start)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "track_pose_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
