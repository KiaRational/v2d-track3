"""Object trajectories from a hand-conditioned pose graph with slip (VICON-style, arXiv 2610.05180, vision-only).

One robust nonlinear least-squares problem per object over its world poses X_t = Exp(w_t) X0_t, p_t = p0_t + d_t
(X0 = the source trajectory, e.g. fused_bfix_recon). Factors (residuals divided by their sigma, Cauchy loss):
  visual     FoundationPose (raw) and gated stereo-ICP poses, mapped to the source's symmetry branch; sigma grows as
             the object is covered: sigma / max(vis, 0.05) -- a hand-covered frame barely counts.
  still      not grasped: X_t = X_{t-1} (0.5 mm / 0.2 deg).
  hand       grasped by hand h at t-1 and t: the object's world motion equals the hand's,
             Log((H_t H_{t-1}^-1)^-1 (X_t X_{t-1}^-1)) ~ 0 (sigma 3 mm / 1 deg, x (1 + 2 vis): loose where the object
             is clearly seen so vision corrects it, tight where it is covered). Rotation about the PINCH axis
             (thumb tip -> index tip) gets SLIP x the sigma: a thin object may pivot between two fingers.
  link       same grasp, frames t-L and t: object-in-hand offset H^-1 X constant within 8 deg / 15 mm (no drift).
  smooth     second differences of position / rotation (3 mm, 2 deg per frame^2).
H = palm pose of the hands variant (world), smoothed (1.5 frames), gaps interpolated. Grasps: --grasp vlm
(grasp_vlm.json intervals, cleaned as grasp_lock) or report (fusion_report grasp_segments).
Writes <run>/<out>/<obj>/poses (camera<-object) + hand_graph_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python hand_graph.py <run>... [--grasp vlm|report]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from fuse_objects import stereo_hands
from grasp_lock import clean_intervals
from handlock import ang, interp_poses, smooth_poses
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
DEG = np.pi / 180
S_VIS_T, S_VIS_R = 0.01, 5 * DEG
S_STILL_T, S_STILL_R = 0.0005, 0.2 * DEG
S_HAND_T, S_HAND_R, SLIP = 0.003, 1.0 * DEG, 5.0
S_LINK_T, S_LINK_R, LINK_L = 0.015, 8 * DEG, 10
S_ACC_T, S_ACC_R = 0.003, 2 * DEG


def log_R(R):
    return Rotation.from_matrix(R).as_rotvec()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_bfix_recon")
    ap.add_argument("--out", default="fused_hg_recon")
    ap.add_argument("--hands", default="hands_c")
    ap.add_argument("--grasp", choices=["vlm", "report"], default="vlm")
    ap.add_argument("--max_nfev", type=int, default=200)
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        hands = stereo_hands(run, n, a.hands)
        H, Jw = {}, {}
        for sd, hd in hands.items():
            P = {t: slam[t] @ hd["H"][t] for t in range(n) if np.isfinite(hd["H"][t]).all()}
            if len(P) >= 2:
                H[sd] = interp_poses(smooth_poses(P, 1.5), n)
                Jw[sd] = np.array([hd["J"][t] @ slam[t][:3, :3].T + slam[t][:3, 3] for t in range(n)])
        if a.grasp == "vlm":
            gv = json.loads((run / "grasp_vlm.json").read_text())
            grasps = {nm: clean_intervals(v["intervals"]) for nm, v in gv.items()}
        else:
            rep = json.loads((run / f"fusion_report{suf}.json").read_text())
            grasps = {nm: [(g["side"], g["from"], g["to"]) for g in v.get("grasp_segments", [])]
                      for nm, v in rep.items() if isinstance(v, dict)}
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if not src.is_dir() or nm not in grasps:
                continue
            X0 = np.array([slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)])
            od = run / obj_dir / nm
            cv = next(iter(sorted(od.glob(".contact_vis_*.npz"))), None)
            vis = np.load(cv)["vis"][:n] if cv is not None else np.ones(n)
            group = [np.asarray(S) for S in mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")["group"]]
            holder = [None] * n
            for sd, s, e in grasps[nm]:
                if sd in H:
                    for t in range(max(0, s), min(n, e + 1)):
                        holder[t] = holder[t] or sd
            # visual measurements
            meas = []
            for d, wk in ((od / "poses", 1.0), (run / ("icp" + suf) / nm / "poses", 1.5)):
                for t in range(n):
                    p = d / f"{t:06d}.json"
                    if not p.exists():
                        continue
                    M = slam[t] @ load_T(p)
                    if len(group) > 1:
                        S = min(group, key=lambda S: ang(M[:3, :3] @ S, X0[t][:3, :3]))
                        M[:3, :3] = M[:3, :3] @ S
                    meas.append((t, M, wk * max(float(vis[t]), 0.05)))
            # factor lists
            still = [t for t in range(1, n) if holder[t] is None and holder[t - 1] is None]
            handf = [t for t in range(1, n) if holder[t] is not None and holder[t] == holder[t - 1]]
            link = [t for t in range(LINK_L, n) if holder[t] is not None and
                    all(holder[u] == holder[t] for u in range(t - LINK_L, t + 1))]
            pinch = {}
            for t in handf:
                J = Jw[holder[t]][t]
                ax = J[8] - J[4] if np.isfinite(J[[4, 8]]).all() else np.zeros(3)
                nrm = np.linalg.norm(ax)
                pinch[t] = ax / nrm if nrm > 1e-3 else None
            Dh = {t: H[holder[t]][t] @ np.linalg.inv(H[holder[t]][t - 1]) for t in handf}
            Hinv = {sd: {t: np.linalg.inv(H[sd][t]) for t in range(n)} for sd in H}

            def poses(x):
                x = x.reshape(n, 6)
                R = Rotation.from_rotvec(x[:, :3]).as_matrix() @ X0[:, :3, :3]
                P = X0[:, :3, 3] + x[:, 3:]
                return R, P

            def T_(R, P, t):
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = R[t], P[t]
                return M

            def residuals(x):
                R, P = poses(x)
                r = []
                for t, M, w in meas:
                    r += list(log_R(R[t] @ M[:3, :3].T) * w / S_VIS_R) + list((P[t] - M[:3, 3]) * w / S_VIS_T)
                for t in still:
                    r += list(log_R(R[t] @ R[t - 1].T) / S_STILL_R) + list((P[t] - P[t - 1]) / S_STILL_T)
                for t in handf:
                    Do = T_(R, P, t) @ np.linalg.inv(T_(R, P, t - 1))
                    E = np.linalg.inv(Dh[t]) @ Do
                    infl = 1 + 2 * float(vis[t])
                    rr = log_R(E[:3, :3])
                    ax = pinch[t]
                    if ax is not None:
                        par = rr @ ax
                        rr = rr - par * ax + (par / SLIP) * ax
                    r += list(rr / (S_HAND_R * infl)) + list(E[:3, 3] / (S_HAND_T * infl))
                for t in link:
                    sd = holder[t]
                    G1 = Hinv[sd][t] @ T_(R, P, t)
                    G0 = Hinv[sd][t - LINK_L] @ T_(R, P, t - LINK_L)
                    E = np.linalg.inv(G0) @ G1
                    r += list(log_R(E[:3, :3]) / S_LINK_R) + list(E[:3, 3] / S_LINK_T)
                for t in range(1, n - 1):
                    r += list((log_R(R[t + 1] @ R[t].T) - log_R(R[t] @ R[t - 1].T)) / S_ACC_R)
                    r += list((P[t + 1] - 2 * P[t] + P[t - 1]) / S_ACC_T)
                return np.array(r)

            # Jacobian sparsity: every residual block touches 1-3 frames
            rows = []
            for t, _, _ in meas:
                rows.append([t])
            rows += [[t, t - 1] for t in still] + [[t, t - 1] for t in handf] + [[t, t - LINK_L] for t in link]
            rows += [[t - 1, t, t + 1] for t in range(1, n - 1)]
            Sp = lil_matrix((6 * len(rows), 6 * n), dtype=int)     # 6 residuals per block, same order as residuals()
            for k, fr in enumerate(rows):
                for f in fr:
                    Sp[6 * k:6 * k + 6, 6 * f:6 * f + 6] = 1
            x0 = np.zeros(6 * n)
            r0 = residuals(x0)
            assert len(r0) == Sp.shape[0], (len(r0), Sp.shape)
            sol = least_squares(residuals, x0, jac_sparsity=Sp.tocsr(), loss="cauchy", f_scale=1.0,
                                x_scale="jac", max_nfev=a.max_nfev, verbose=0)
            R, P = poses(sol.x)
            outd = run / a.out / nm / "poses"
            outd.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ T_(R, P, t)
                (outd / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st = [ang(R[t + 1], R[t]) for t in range(n - 1)]
            rel = [ang((Hinv[holder[t]][t] @ T_(R, P, t))[:3, :3], (Hinv[holder[t]][t - 1] @ T_(R, P, t - 1))[:3, :3])
                   for t in handf]
            report[nm] = {"cost0": round(float(0.5 * (r0 ** 2).sum()), 1), "cost": round(float(sol.cost), 1),
                          "nfev": sol.nfev, "grasped": sum(h is not None for h in holder), "n_meas": len(meas),
                          "max_step_deg": round(max(st), 1), "jumps10": int(sum(s > 10 for s in st)),
                          "src_jumps10": int(sum(ang(X0[t + 1][:3, :3], X0[t][:3, :3]) > 10 for t in range(n - 1))),
                          "in_hand_rot_p95_deg_per_frame": round(float(np.percentile(rel, 95)), 2) if rel else None,
                          "change_rot_deg_med": round(float(np.median([ang(R[t], X0[t][:3, :3]) for t in range(n)])), 1),
                          "change_cm_med": round(float(np.median(np.linalg.norm(P - X0[:, :3, 3], axis=1))) * 100, 2)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "hand_graph_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
