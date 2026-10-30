"""Physically consistent object trajectories from the fused ones: a rigid-segment model with one global choice of
pose branch (FoundationPose flips) per segment.

Why: the fusion treats "untouched objects stay still" and "held objects move rigidly with the hand" as soft costs, and
runs of flipped FoundationPose / ICP rotations still win locally -- a sweeper flips by 165 deg in the middle of a free
interval (e30), cups spin while resting once their (false) symmetry is removed (p21, p13). The robot cannot imitate an
object that turns by itself, and the episode terminates. Here the physics is a hard model:

  free   frames with no hand contact (fusion_report hand_contact, any side): the object is STATIC -- one pose per
         maximal free run (rotation = branch below, position = median of the fused positions);
  held   grasp segments (fusion_report grasp_segments): object = palm(t) @ G with ONE constant G per segment
         (palm = the hands variant's rigid palm frame, as in the fusion's grasp term);
  other  contact without a grasp (pushing, regrasping): the fused poses, unchanged.

Rotation branches per free / held segment: the medoid of the fused rotations (or of H^-1 F for held) and that medoid
composed with 180-deg flips about the object axes. A dynamic program over the segments in time order picks one branch
per segment minimising   sum(data) + LAMBDA * sum(boundary rotation jumps),  data = median over the segment's frames of
min(angle(pose, measurement), 90 deg) over the FoundationPose and stereo-ICP measurements. 'other' segments have one
branch (the fused poses). Boundaries are then blended over BLEND frames (slerp / lerp of the two neighbouring segment
poses), so the object never teleports. Writes <run>/<out>/<obj>/poses (camera<-object, like the fusion) and
<run>/<out>/rigidify_report.json.
Usage: V2D_OBJ_DIR=objects_recon python rigidify.py <run> [--src fused_all_recon] [--out fused_rigid_recon]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fuse_objects import stereo_hands
from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])
FLIPS = [np.eye(3), np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])]
LAMBDA, BLEND, MIN_SEG = 1.0, 3, 3
FREE_MOVE_M = 0.03   # a "free" run whose fused centre moves more than this keeps its fused poses (missed contact)


def ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def medoid(Rs):
    tr = np.einsum("aij,bij->ab", Rs, Rs)
    return Rs[np.argmin(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).sum(1))]


def segments(n, held, contact):
    """[(kind, side, frames)] covering 0..n-1 in order; kind free / held / other."""
    lab = []
    for t in range(n):
        if t in held:
            lab.append(("held", held[t]))
        elif t in contact:
            lab.append(("other", None))
        else:
            lab.append(("free", None))
    out, s = [], 0
    for t in range(1, n + 1):
        if t == n or lab[t] != lab[s]:
            out.append((lab[s][0], lab[s][1], list(range(s, t))))
            s = t
    # very short free / held runs carry no reliable branch: treat as 'other' (fused poses)
    return [(k if (k == "other" or len(f) >= MIN_SEG) else "other", sd, f) for k, sd, f in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_all_recon")
    ap.add_argument("--out", default="fused_rigid_recon")
    ap.add_argument("--hands", default=os.environ.get("V2D_FUSE_HANDS", "hands_c"))
    ap.add_argument("--free_only", action="store_true",
                    help="only untouched segments are made static; held segments keep the fused poses (our hand poses are "
                         "not accurate enough to carry an object rigidly: vs public mocap the held model lost 0.86 -> 0.72)")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        rep = json.loads((run / f"fusion_report{suf}.json").read_text())
        hands = stereo_hands(run, n, a.hands)
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if nm not in rep or not src.is_dir():
                continue
            F = {t: slam[t] @ load_T(src / f"{t:06d}.json") for t in sorted(slam) if (src / f"{t:06d}.json").exists()}
            if len(F) < n:
                print(f"{rn} {nm}: fused poses incomplete ({len(F)}/{n}), skipped", flush=True)
                continue
            meas = {t: [] for t in range(n)}
            for d in (run / obj_dir / nm / "poses", run / ("icp" + suf) / nm / "poses"):
                for t in range(n):
                    p = d / f"{t:06d}.json"
                    if p.exists() and t in slam:
                        meas[t].append((slam[t] @ load_T(p))[:3, :3])
            contact = {t for v in rep[nm].get("hand_contact", {}).values() for t in v}
            held = {}
            for g in rep[nm].get("grasp_segments", []):
                Hs = hands.get(g["side"], {}).get("H")
                for t in range(g["from"], g["to"] + 1):
                    if Hs is not None and np.isfinite(Hs[t]).all():
                        held[t] = g["side"]
            segs = segments(n, {} if a.free_only else held, contact | set(held))
            # branches per segment: list of (pose function t -> 4x4)
            cand = []
            for kind, side, fr in segs:
                if kind == "other":
                    cand.append([("fused", {t: F[t] for t in fr})])
                    continue
                if kind == "free":
                    p = np.median([F[t][:3, 3] for t in fr], 0)
                    if max(np.linalg.norm(F[t][:3, 3] - p) for t in fr) > FREE_MOVE_M:
                        # it moves although no hand contact was detected (missed contact): never freeze it
                        cand.append([("fused", {t: F[t] for t in fr})])
                        continue
                    R = medoid(np.array([F[t][:3, :3] for t in fr]))
                    br = []
                    for k, Fl in enumerate(FLIPS):
                        M = np.eye(4); M[:3, :3], M[:3, 3] = R @ Fl, p
                        br.append((f"free{k}", {t: M for t in fr}))
                    cand.append(br)
                    continue
                Hw = {t: slam[t] @ hands[side]["H"][t] for t in fr}
                Gs = np.array([np.linalg.inv(Hw[t]) @ F[t] for t in fr])
                Rg, tg = medoid(Gs[:, :3, :3]), np.median(Gs[:, :3, 3], 0)
                br = []
                for k, Fl in enumerate(FLIPS):
                    G = np.eye(4); G[:3, :3], G[:3, 3] = Rg @ Fl, tg
                    br.append((f"held{k}", {t: Hw[t] @ G for t in fr}))
                cand.append(br)

            def data(poses):
                v = [min(min(ang(poses[t][:3, :3], M) for M in meas[t]), 90.0) for t in poses if meas[t]]
                return float(np.median(v)) if v else 0.0
            cost = [[data(p) for _, p in br] for br in cand]
            # dynamic program over segments
            m = len(cand)
            acc = [np.array(cost[0])]
            back = []
            for i in range(1, m):
                prev_end = [p[max(p)] for _, p in cand[i - 1]]
                cur_start = [p[min(p)] for _, p in cand[i]]
                J = np.array([[ang(pe[:3, :3], cs[:3, :3]) for cs in cur_start] for pe in prev_end])
                tot = acc[-1][:, None] + LAMBDA * J
                back.append(tot.argmin(0))
                acc.append(tot.min(0) + np.array(cost[i]))
            pick = [int(acc[-1].argmin())]
            for i in range(m - 1, 0, -1):
                pick.append(int(back[i - 1][pick[-1]]))
            pick = pick[::-1]
            out = {}
            for (kind, side, fr), br, k in zip(segs, cand, pick):
                out.update(br[k][1])
            # blend segment boundaries (no teleport): BLEND frames each side, between the two segments' poses
            for i in range(1, m):
                b = segs[i][2][0]
                A, B = out[b - 1].copy(), out[b].copy()
                lo, hi = max(0, b - BLEND), min(n - 1, b + BLEND - 1)
                if ang(A[:3, :3], B[:3, :3]) < 1.0 and np.linalg.norm(A[:3, 3] - B[:3, 3]) < 0.002:
                    continue
                sl = Slerp([lo - 1, hi + 1], Rotation.from_matrix([out[max(lo - 1, 0)][:3, :3], out[min(hi + 1, n - 1)][:3, :3]]))
                for t in range(lo, hi + 1):
                    al = (t - lo + 1) / (hi - lo + 2)
                    M = np.eye(4)
                    M[:3, :3] = sl(t).as_matrix()
                    M[:3, 3] = (1 - al) * out[max(lo - 1, 0)][:3, 3] + al * out[min(hi + 1, n - 1)][:3, 3]
                    out[t] = M
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ out[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            steps = [ang(out[t + 1][:3, :3], out[t][:3, :3]) for t in range(n - 1)]
            report[nm] = {"segments": [{"kind": k, "side": sd, "from": f[0], "to": f[-1], "branch": cand[i][pick[i]][0],
                                        "data_deg": round(cost[i][pick[i]], 1)} for i, (k, sd, f) in enumerate(segs)],
                          "max_step_deg": round(max(steps), 1), "jumps10": int(sum(s > 10 for s in steps))}
            print(rn, nm, json.dumps({k: v for k, v in report[nm].items() if k != "segments"}),
                  f"{len(segs)} segments", flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "rigidify_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
