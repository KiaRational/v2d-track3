"""Flip-branch repair of object trajectories, keeping the fused motion.

Between two impossible rotation steps (> SPIKE deg in one frame, i.e. > 500 deg/s at 20 fps: no hand turns these
objects that fast) the fused trajectory is accurate up to WHICH symmetry-like branch it sits on -- FoundationPose
flips a spoon's bowl or a sweeper end for end, and the fusion follows a run of such frames (e33 spoon: 140 deg in one
frame). The motion inside each piece is kept (vs public mocap the fused pieces are good: 0.87 of moving frames within
15 deg / 3 cm); only the branch of every piece is chosen: piece poses F_t @ Flip_k (identity or a 180-deg turn about
an object axis), one k per piece, by a dynamic program minimising
    sum(data) + LAMBDA * sum(rotation jump at piece boundaries),
data = median over the piece's frames of min(angle(pose, FoundationPose / stereo-ICP measurement), 90 deg). Steps
still > SPIKE afterwards (single bad frames) are bridged by slerp / lerp from the neighbouring good frames.
Writes <run>/<out>/<obj>/poses (camera<-object) and branchfix_report.json.
Usage: V2D_OBJ_DIR=objects_recon python branchfix.py <run>... [--src fused_rfree_recon] [--out fused_bfix_recon]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])
FLIPS = [np.eye(3), np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])]
SPIKE, LAMBDA = 25.0, 1.5


def ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_rfree_recon")
    ap.add_argument("--out", default="fused_bfix_recon")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if not src.is_dir():
                continue
            F = [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]
            meas = [[] for _ in range(n)]
            for d in (run / obj_dir / nm / "poses", run / ("icp" + suf) / nm / "poses"):
                for t in range(n):
                    p = d / f"{t:06d}.json"
                    if p.exists():
                        meas[t].append((slam[t] @ load_T(p))[:3, :3])
            steps = [ang(F[t + 1][:3, :3], F[t][:3, :3]) for t in range(n - 1)]
            cuts = [t + 1 for t, s in enumerate(steps) if s > SPIKE]
            bounds = [0] + cuts + [n]
            pieces = [list(range(b0, b1)) for b0, b1 in zip(bounds, bounds[1:]) if b1 > b0]

            def data(fr, Fl):
                v = [min(min(ang(F[t][:3, :3] @ Fl, M) for M in meas[t]), 90.0) for t in fr if meas[t]]
                return float(np.median(v)) if v else 0.0
            cost = [np.array([data(fr, Fl) for Fl in FLIPS]) for fr in pieces]
            acc, back = [cost[0]], []
            for i in range(1, len(pieces)):
                e, s = pieces[i - 1][-1], pieces[i][0]
                J = np.array([[ang(F[e][:3, :3] @ A, F[s][:3, :3] @ B) for B in FLIPS] for A in FLIPS])
                tot = acc[-1][:, None] + LAMBDA * J
                back.append(tot.argmin(0))
                acc.append(tot.min(0) + cost[i])
            pick = [int(acc[-1].argmin())]
            for i in range(len(pieces) - 1, 0, -1):
                pick.append(int(back[i - 1][pick[-1]]))
            pick = pick[::-1]
            out = [None] * n
            for fr, k in zip(pieces, pick):
                for t in fr:
                    M = F[t].copy()
                    M[:3, :3] = F[t][:3, :3] @ FLIPS[k]
                    out[t] = M
            # remaining single-frame spikes: bridge from the nearest frames on both sides with plausible steps
            bad = np.zeros(n, bool)
            for t in range(n - 1):
                if ang(out[t + 1][:3, :3], out[t][:3, :3]) > SPIKE:
                    bad[t + 1] = True
            bridged = int(bad.sum())
            t = 0
            while t < n:
                if not bad[t]:
                    t += 1
                    continue
                s = t
                while t < n and bad[t]:
                    t += 1
                lo, hi = s - 1, t          # frames lo and hi are good; s..t-1 are bridged
                if lo < 0 or hi >= n:      # run touches a sequence end: nothing to bridge to, keep it
                    continue
                sl = Slerp([lo, hi], Rotation.from_matrix([out[lo][:3, :3], out[hi][:3, :3]]))
                for u in range(s, t):
                    al = (u - lo) / (hi - lo)
                    M = np.eye(4)
                    M[:3, :3] = sl(u).as_matrix()
                    M[:3, 3] = (1 - al) * out[lo][:3, 3] + al * out[hi][:3, 3]
                    out[u] = M
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ out[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st = [ang(out[t + 1][:3, :3], out[t][:3, :3]) for t in range(n - 1)]
            report[nm] = {"pieces": len(pieces), "flipped_pieces": int(sum(k > 0 for k in pick)),
                          "flipped_frames": int(sum(len(fr) for fr, k in zip(pieces, pick) if k > 0)),
                          "bridged_frames": bridged, "max_step_before": round(max(steps), 1),
                          "max_step_after": round(max(st), 1)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "branchfix_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
