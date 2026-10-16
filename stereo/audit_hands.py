"""Audit hand-identity failures in Stage 6b output (host env, CPU, seconds per episode).

Per episode, from <run>/stereo/<file> (default hands_tri.json), X_smooth palm centres:
  far     frames whose palm is beyond arm's reach (> --reach m from the head camera):
          a background person's hand
  cross   frames where the left palm is > 5 cm right of the right palm (left/right swap)
  jump    frames where a palm moves > 15 cm between consecutive frames (identity switch)
  swap    frames where exchanging the two labels explains the motion much better than
          keeping them (both hands present at t-1 and t)
Usage: python audit_hands.py <run> [...] [--file hands_tri.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ["V2D_ROOT"])
PALM = [0, 5, 9, 13, 17]


def palms(d, side, n):
    P = np.full((n, 3), np.nan)
    for t, X in d.get(side, {}).get("X_smooth", {}).items():
        if int(t) < n:
            P[int(t)] = np.asarray(X)[PALM].mean(0)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--file", default="hands_tri.json")
    ap.add_argument("--reach", type=float, default=0.9)
    a = ap.parse_args()
    tot = {}
    print(f"{'run':22s} {'frames L/R':>11s} {'far':>5s} {'cross':>6s} {'jump':>5s} {'swap':>5s}")
    for r in a.runs:
        f = ROOT / "runs" / r / "stereo" / a.file
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        n = len(list((ROOT / "runs" / r / "recon_hawor/frames").glob("*.png")))
        L, R = palms(d, "left", n), palms(d, "right", n)
        okL, okR = np.isfinite(L[:, 0]), np.isfinite(R[:, 0])
        far = int((np.linalg.norm(L, axis=1) > a.reach).sum() + (np.linalg.norm(R, axis=1) > a.reach).sum())
        both = okL & okR
        cross = int((both & (L[:, 0] > R[:, 0] + 0.05)).sum())
        jL = np.linalg.norm(np.diff(L, axis=0), axis=1)
        jR = np.linalg.norm(np.diff(R, axis=0), axis=1)
        jump = int((np.nan_to_num(jL) > 0.15).sum() + (np.nan_to_num(jR) > 0.15).sum())
        keep = jL + jR
        swapd = np.linalg.norm(L[1:] - R[:-1], axis=1) + np.linalg.norm(R[1:] - L[:-1], axis=1)
        swap = int(np.nansum((both[1:] & both[:-1]) & (swapd < 0.5 * keep) & (keep > 0.10)))
        row = dict(nL=int(okL.sum()), nR=int(okR.sum()), far=far, cross=cross, jump=jump, swap=swap)
        for k, v in row.items():
            tot[k] = tot.get(k, 0) + v
        print(f"{r:22s} {row['nL']:5d}/{row['nR']:<5d} {far:5d} {cross:6d} {jump:5d} {swap:5d}", flush=True)
    print("TOTAL", json.dumps(tot))


if __name__ == "__main__":
    main()
