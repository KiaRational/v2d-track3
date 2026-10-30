"""Resting objects sit ON the table: an untouched object near the tabletop is shifted along the table normal so its
lowest point is exactly on the table (eval ep25 left board started 13.5 mm inside the table -> the simulator pops it out;
eval ep20 pitcher started 36 mm above it -> it drops at t=0).

Touched = as rest_hold (grasp segments / VLM intervals / a finger joint within TOUCH_M of the surface, dilated by PAD).
For every untouched stretch of >= MIN_LEN frames: gap = median lowest-point height above the table top. Shift by -gap
when |gap| <= MAX_GAP_M and nothing else is under the object (no other object's mesh within STACK_M below its lowest
point at the stretch's middle frame: a cup in a cup / a lid on a pot keeps its height) and its centroid is over the
table disk (table_plane.json disk). Edges blended over BLEND frames. Rotation untouched.
Writes <run>/<out>/<obj>/poses (camera<-object) + rest_table_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python rest_table.py <run>... [--src tpfsu_recon] [--out tpfsut_recon]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from fuse_objects import stereo_hands
from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])
TOUCH_M, PAD, MIN_LEN, MAX_GAP_M, STACK_M, BLEND = 0.02, 3, 4, 0.05, 0.03, 4
FINGER_J = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="tpfsu_recon")
    ap.add_argument("--out", default="tpfsut_recon")
    ap.add_argument("--hands", default="hands_c")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        tp = json.loads((run / "table_plane.json").read_text())
        up, h = np.asarray(tp["up"], float), float(tp["height"])
        ctr, rad = (np.asarray(tp["center"], float), float(tp["radius"])) if "center" in tp and "radius" in tp else (None, None)
        hands = stereo_hands(run, n, a.hands)
        Jw = {sd: np.array([hd["J"][t][FINGER_J] @ slam[t][:3, :3].T + slam[t][:3, 3] for t in range(n)]) for sd, hd in hands.items()}
        rp = run / f"fusion_report{suf}.json"
        rep = json.loads(rp.read_text()) if rp.exists() else {}
        gv = json.loads((run / "grasp_vlm.json").read_text()) if (run / "grasp_vlm.json").exists() else {}
        objs = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if src.is_dir():
                V = np.asarray(trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh").vertices)
                objs[nm] = {"V": V[:: max(1, len(V) // 3000)], "X": [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]}
        report = {}
        for nm, ob in objs.items():
            V, X = ob["V"], ob["X"]
            W = lambda t: V @ X[t][:3, :3].T + X[t][:3, 3]   # noqa: E731
            touched = np.zeros(n, bool)
            for g in (rep.get(nm) if isinstance(rep.get(nm), dict) else {}).get("grasp_segments", []):
                touched[max(0, g["from"]):min(n, g["to"] + 1)] = True
            for ivs in gv.get(nm, {}).get("intervals", {}).values():
                for s, e in ivs:
                    touched[max(0, s):min(n, e + 1)] = True
            for t in range(n):
                if not touched[t]:
                    tree = cKDTree(W(t))
                    touched[t] = any(np.isfinite(J[t]).all() and tree.query(J[t])[0].min() < TOUCH_M for J in Jw.values())
            touched = np.convolve(touched.astype(float), np.ones(2 * PAD + 1), "same") > 0
            shift = np.zeros(n)
            log = []
            t = 0
            while t < n:
                if touched[t]:
                    t += 1
                    continue
                s = t
                while t < n and not touched[t]:
                    t += 1
                e = t
                if e - s < MIN_LEN:
                    continue
                gap = float(np.median([(W(u) @ up).min() - h for u in range(s, e, max(1, (e - s) // 10))]))
                mid = (s + e) // 2
                P = W(mid)
                c = P.mean(0)
                if abs(gap) > MAX_GAP_M or abs(gap) < 0.001:
                    continue
                if ctr is not None and np.linalg.norm((c - ctr) - ((c - ctr) @ up) * up) > rad:
                    log.append([s, e - 1, round(gap * 1000, 1), "off-disk"])
                    continue
                low = P[np.argmin(P @ up)]
                stacked = False
                for nm2, ob2 in objs.items():
                    if nm2 == nm:
                        continue
                    Q = ob2["V"] @ ob2["X"][mid][:3, :3].T + ob2["X"][mid][:3, 3]
                    below = Q[(Q @ up) < (low @ up) + 0.005]
                    if len(below):
                        lat = np.linalg.norm((below - low) - ((below - low) @ up)[:, None] * up, axis=1)
                        if (((low - below) @ up < STACK_M) & (lat < 0.08)).any():
                            stacked = True
                            break
                if stacked:
                    log.append([s, e - 1, round(gap * 1000, 1), "stacked"])
                    continue
                shift[s:e] = -gap
                log.append([s, e - 1, round(gap * 1000, 1), "snapped"])
            # blend the shift's edges
            sm = shift.copy()
            for t in range(n):
                lo, hi = max(0, t - BLEND), min(n, t + BLEND + 1)
                if shift[t] == 0 and (shift[lo:hi] != 0).any():
                    k = np.nonzero(shift[lo:hi])[0]
                    j = lo + k[np.argmin(abs(lo + k - t))]
                    sm[t] = shift[j] * (1 - abs(j - t) / (BLEND + 1))
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                M = X[t].copy()
                M[:3, 3] += sm[t] * up
                C = np.linalg.inv(slam[t]) @ M
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            report[nm] = {"stretches": log, "start_gap_mm": round(((W(0) @ up).min() - h) * 1000, 1),
                          "start_shift_mm": round(sm[0] * 1000, 1), "frames_shifted": int((sm != 0).sum())}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "rest_table_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
