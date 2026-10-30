"""Flip repair: a hand-held or resting object cannot turn > SPIKE_DEG in one frame (900 deg/s at 20 fps); such a step is
a pose flip the fusion let through, and the robot terminates trying to imitate it. A spike at a followed by a spike
back at b (R_b close to R_{a-1}) marks the segment [a, b) as flipped: it is rotated back by the jump (R' = C R, C =
R_{a-1} R_a^T, about the object's own origin -- translation kept), preserving the motion inside it. An unmatched spike
corrects the shorter side (start or end of the clip). A repair is kept only if the new segment boundaries are continuous
(< OK_DEG). Writes the variant's poses in place (originals in <obj>/poses_predespike) + <obj>/despike.json.
Usage: python despike_objects.py <run> [--variant fused_all_recon]"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
SPIKE_DEG, OK_DEG, MAX_SEG = 35.0, 25.0, 90


def load(p):
    d = json.loads(p.read_text()); M = np.eye(4)
    if "R" in d:
        M[:3, :3], M[:3, 3] = np.asarray(d["R"]), d["T"]; return M
    from scipy.spatial.transform import Rotation
    q = d["rotation"]; M[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M[:3, 3] = d["translation"]
    return M


def ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def repair(R):
    """R: (n,3,3) world rotations of consecutive frames. Returns corrected copy and a log."""
    R = R.copy(); n = len(R); log = []
    for _ in range(20):   # repairs can expose / merge others; iterate until clean
        steps = np.array([ang(R[t], R[t - 1]) for t in range(1, n)])
        spikes = [t for t in range(1, n) if steps[t - 1] > SPIKE_DEG]
        if not spikes:
            break
        a = spikes[0]
        seg = None
        for b in spikes[1:]:
            if b - a > MAX_SEG:
                break
            if ang(R[b], R[a - 1]) < OK_DEG:
                seg = (a, b); break
        if seg is None:   # unmatched: the shorter side is the flipped one
            seg = (a, n) if n - a <= a else (0, a)
        s, e = seg
        C = R[s - 1] @ R[s].T if s > 0 else R[e] @ R[e - 1].T
        Rn = R.copy(); Rn[s:e] = np.einsum("ij,njk->nik", C, R[s:e])
        ok = (s == 0 or ang(Rn[s], Rn[s - 1]) < OK_DEG) and (e >= n or ang(Rn[e], Rn[e - 1]) < OK_DEG)
        log.append({"frames": [int(s), int(e - 1)], "jump_deg": round(ang(R[s], R[s - 1]) if s > 0 else ang(R[e], R[e - 1]), 1),
                    "applied": bool(ok)})
        if not ok:
            break    # leave the rest untouched rather than guess
        R = Rn
    return R, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variant", default="fused_all_recon")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    slam = {int(p.stem): load(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    for objd in sorted((run / a.variant).iterdir()):
        src = objd / ("poses_predespike" if (objd / "poses_predespike").exists() else "poses")
        if not src.is_dir():
            continue
        ts = [t for t in sorted(int(p.stem) for p in src.glob("[0-9]*.json")) if t in slam]
        if len(ts) < 3:
            continue
        W = np.array([slam[t] @ load(src / f"{t:06d}.json") for t in ts])
        Rn, log = repair(W[:, :3, :3])
        rep = {"repairs": log, "max_step_before": round(max(ang(W[i, :3, :3], W[i - 1, :3, :3]) for i in range(1, len(ts))), 1),
               "max_step_after": round(max(ang(Rn[i], Rn[i - 1]) for i in range(1, len(ts))), 1)}
        (objd / "despike.json").write_text(json.dumps(rep, indent=1))
        if not any(r["applied"] for r in log):
            continue
        if not (objd / "poses_predespike").exists():
            shutil.copytree(objd / "poses", objd / "poses_predespike")
        for i, t in enumerate(ts):
            Wt = W[i].copy(); Wt[:3, :3] = Rn[i]
            M = np.linalg.inv(slam[t]) @ Wt
            (objd / "poses" / f"{t:06d}.json").write_text(json.dumps({"R": M[:3, :3].tolist(), "T": M[:3, 3].tolist()}))
        print(a.run, objd.name, f"{sum(r['applied'] for r in log)} flips repaired; max step {rep['max_step_before']} -> "
              f"{rep['max_step_after']} deg", flush=True)


if __name__ == "__main__":
    main()
