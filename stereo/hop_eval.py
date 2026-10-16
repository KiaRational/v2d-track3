"""Gate for HOPformer hands: are its finger poses at least as good as WiLoR's where stereo can check them?

Per side, on frames with a stereo triangulation (hands_tri.json X_raw on `valid` joints), each model's
hand (local MANO joints, Mano.local convention) is aligned to the triangulated joints by a similarity
transform on the palm joints (wrist + 4 MCPs), then:
  tip_mm   fingertip error after the palm alignment (articulation only, no placement)
  pa_mm    PA-MPJPE over all valid joints
Models: wilor (the colour WiLoR record the MANO fit uses as its pose prior) and HOPformer variants
(its smplx params taken as-is / + hand_mean, left hands mirrored into the right-hand convention or not;
the right convention is the one that matches the triangulation).
Split: 'full' = all 21 joints triangulated, 'partial' = some joints occluded (target is sparser there).
Usage: python hop_eval.py <run> [--hop hopformer_raw] [--frames a:b]   (agile image)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code/agile")
from geometry import Mano  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
PALM, TIPS = [0, 5, 9, 13, 17], [4, 8, 12, 16, 20]
MIRROR = np.array([1.0, -1.0, -1.0])


def sim_align(A, B):
    """s, R, t minimising |s R A + t - B|."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum()
    return s, R, mb - s * R @ ma


def errors(J, X, w):
    ok = w > 0
    palm = [i for i in PALM if ok[i]]
    out = {}
    if len(palm) >= 4:
        s, R, t = sim_align(J[palm], X[palm])
        Y = (s * (R @ J.T)).T + t
        tips = [i for i in TIPS if ok[i]]
        if tips:
            out["tip_mm"] = float(np.linalg.norm(Y[tips] - X[tips], axis=1).mean() * 1000)
    if ok.sum() >= 6:
        s, R, t = sim_align(J[ok], X[ok])
        Y = (s * (R @ J[ok].T)).T + t
        out["pa_mm"] = float(np.linalg.norm(Y - X[ok], axis=1).mean() * 1000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hop", default="hopformer_raw")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--all_conventions", action="store_true")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    tri = json.loads((run / "stereo/hands_tri.json").read_text())
    mano = Mano(str(ROOT / "weights/hand"))
    hop_dir = run / "recon_hawor" / a.hop
    variants = ([(p, m) for p in ("hand_pose", "hand_pose_full") for m in (True, False)] if a.all_conventions
                else [("hand_pose_full", True)])  # the convention that matches (ep24 check)
    rows = []
    for side in ("left", "right"):
        S = tri.get(side, {})
        for k, Xr in S.get("X_raw", {}).items():
            t = int(k)
            if a.frames:
                s0, s1 = (int(x) if x else None for x in a.frames.split(":"))
                if not (s0 or 0) <= t < (s1 or 10 ** 9):
                    continue
            X = np.array(Xr, float)
            w = np.array(S["valid"][k], float) * np.isfinite(X).all(1)
            X = np.nan_to_num(X)
            if (w > 0).sum() < 8:
                continue
            row = {"side": side, "t": t, "split": "full" if (w > 0).all() else "partial"}
            fw = run / f"recon_hawor/wilor_raw/{t:06d}.json"
            if fw.exists():
                own = [d for d in json.loads(fw.read_text())
                       if d.get("score", 0) >= 0.3 and d["is_right"] == (side == "right")]
                if own:
                    d = max(own, key=lambda z: z.get("score", 0))
                    row["wilor"] = errors(mano.local(d)[1].cpu().numpy(), X, w)
            fh = hop_dir / f"{t:06d}.json"
            if fh.exists() and side in (h := json.loads(fh.read_text())):
                r = h[side]
                for p, mirror in variants:
                    if r.get(p) is None:
                        continue
                    go, hp = np.array(r["global_orient"]), np.array(r[p]).reshape(15, 3)
                    if side == "left" and mirror:
                        go, hp = go * MIRROR, hp * MIRROR
                    rec = {"is_right": side == "right",
                           "mano": {"global_orient": go.tolist(), "hand_pose": hp.reshape(-1).tolist(),
                                    "betas": r["betas"]}}
                    row[f"hop:{p}:{'mir' if mirror else 'nomir'}"] = errors(mano.local(rec)[1].cpu().numpy(), X, w)
            rows.append(row)
    models = sorted({k for r in rows for k in r if k == "wilor" or k.startswith("hop:")})
    summary = {}
    print(f"{a.run}: {len(rows)} triangulated hand-frames")
    for side in ("left", "right"):
        for split in ("full", "partial"):
            R = [r for r in rows if r["side"] == side and r["split"] == split]
            if not R:
                continue
            line = f"{side:5s} {split:7s} n={len(R):3d}"
            for m in models:
                both = [r for r in R if m in r and "wilor" in r]  # paired frames only
                for met in ("tip_mm", "pa_mm"):
                    v = [r[m][met] for r in both if met in r[m]]
                    if v:
                        summary.setdefault(f"{side}/{split}/{m}", {})[met] = float(np.median(v))
                        summary[f"{side}/{split}/{m}"]["n"] = len(v)
                if f"{side}/{split}/{m}" in summary:
                    e = summary[f"{side}/{split}/{m}"]
                    line += f" | {m}: tip {e.get('tip_mm', float('nan')):.1f} pa {e.get('pa_mm', float('nan')):.1f}"
            print(line)
    out = run / "stereo" / f"hop_eval_{a.hop}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}))
    print("wrote", out)


if __name__ == "__main__":
    main()
