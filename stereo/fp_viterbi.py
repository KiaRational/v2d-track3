"""Whole-sequence choice among ALL FoundationPose hypotheses (Hand2Robot-style Viterbi, arXiv 2610.15769).
Per frame t, candidates k (fp_hyps.py dump, world frame via SLAM): unary weight
    w_tk = score_tk (min-max normalised per frame) x IoU_tk,
IoU = silhouette IoU of the candidate's projected surface points (splatted, 1/4 res) vs the object mask, with HAND
pixels (masks 2, 3) excluded from both -- a hand covering the object does not count against the right pose.
Transition cost between consecutive observed frames: |dp| / TR_M + |drot| / ROT_RAD, rotation distance minimised over
the mesh's symmetry group (a true symmetry costs nothing), divided by the frame gap. Viterbi minimises
    sum (1 - w_tk) + LAM * sum transition.
Frames without hypotheses are interpolated (slerp / lerp). Writes <run>/<out>/<obj>/poses (camera<-object) and
fp_viterbi_report.json. Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python fp_viterbi.py <run>..."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh

from handlock import ang, interp_poses
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
TR_M, ROT_RAD, LAM, DS = 0.02, 0.3, 0.5, 4


def rot_dist(A, B, group):
    """(Na,Nb) geodesic distance (rad) between rotation sets, min over the symmetry group (applied to B)."""
    best = None
    for S in group:
        tr = np.einsum("aij,bij->ab", A, B @ S)
        d = np.arccos(np.clip((tr - 1) / 2, -1, 1))
        best = d if best is None else np.minimum(best, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="fp_vit_recon")
    ap.add_argument("--lam", type=float, default=LAM)
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        k = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm, oid = o["name"], o["id"]
            hd = run / obj_dir / nm / "fp_hyps"
            if not hd.is_dir():
                continue
            S = trimesh.sample.sample_surface(trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh"), 1500, seed=0)[0]
            group = [np.asarray(G) for G in mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")["group"]]
            if len(group) > 8:   # continuous symmetry: subsample the group for the distance
                group = group[:: len(group) // 8]
            ts, Pw, U = [], [], []
            for p in sorted(hd.glob("*.npz")):
                t = int(p.stem)
                if t not in slam:
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
                X = np.einsum("nij,pj->npi", P[:, :3, :3], S) + P[:, None, :3, 3]
                u = (X[..., 0] / X[..., 2] * k["fx"] + k["cx"]) / DS
                v = (X[..., 1] / X[..., 2] * k["fy"] + k["cy"]) / DS
                iou = np.zeros(len(P))
                ker = np.ones((3, 3), np.uint8)
                for i in range(len(P)):
                    r = np.zeros((Hs, Ws), np.uint8)
                    ok = (X[i, :, 2] > 0.05) & (u[i] >= 0) & (u[i] < Ws) & (v[i] >= 0) & (v[i] < Hs)
                    r[v[i, ok].astype(int), u[i, ok].astype(int)] = 1
                    r = cv2.dilate(r, ker) > 0
                    inter, uni = (r & mo & care).sum(), ((r | mo) & care).sum()
                    iou[i] = inter / uni if uni else 0.0
                s = (sc - sc.min()) / max(sc.max() - sc.min(), 1e-9)
                ts.append(t)
                Pw.append(np.einsum("ij,njk->nik", slam[t], P))
                U.append(1.0 - s * iou)
            if len(ts) < 2:
                continue
            # Viterbi
            acc, back = U[0].copy(), []
            for i in range(1, len(ts)):
                gap = ts[i] - ts[i - 1]
                A, B = Pw[i - 1], Pw[i]
                dp = np.linalg.norm(A[:, None, :3, 3] - B[None, :, :3, 3], axis=-1)
                dr = rot_dist(A[:, :3, :3], B[:, :3, :3], group)
                tot = acc[:, None] + a.lam * (dp / TR_M + dr / ROT_RAD) / gap
                back.append(tot.argmin(0))
                acc = tot.min(0) + U[i]
            pick = [int(acc.argmin())]
            for i in range(len(ts) - 1, 0, -1):
                pick.append(int(back[i - 1][pick[-1]]))
            pick = pick[::-1]
            chosen = {t: Pw[i][j] for i, (t, j) in enumerate(zip(ts, pick))}
            full = interp_poses(chosen, n)
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ full[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st = [ang(full[t + 1][:3, :3], full[t][:3, :3]) for t in range(n - 1)]
            report[nm] = {"frames_with_hyps": len(ts), "argmax_kept_frac": round(float(np.mean([j == 0 for j in pick])), 3),
                          "mean_unary": round(float(np.mean([U[i][j] for i, j in enumerate(pick)])), 3),
                          "max_step_deg": round(max(st), 1), "jumps10": int(sum(x > 10 for x in st))}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "fp_viterbi_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
