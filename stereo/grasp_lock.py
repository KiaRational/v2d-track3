"""Objects stuck to the grasping hand, from the agentic grasp intervals (grasp_vlm.json), with the hand's contact
points matched to the object surface.

1 intervals  grasp_vlm.json per hand; same-hand intervals with gaps <= GAP frames merged; short (< SHORT frames)
             intervals of one hand while the other hand holds the object (VLM noise: a hand next to it) dropped;
             overlaps go to the hand already holding (handover when it lets go).
2 onset      the VLM starts a grasp a few frames before the fingers arrive. Onset = the first frame, within
             [vlm_on - 3, vlm_on + 12], where the hand's finger joints are closest to the (still) object (within
             5 mm of the window minimum). Before it the object does not move.
3 release    last frame, within [vlm_off - 10, vlm_off + 3], where the hand is within 1 cm of its closest approach
             to the object as SEEN (source poses).
4 contact    at onset the hand's contact joints (finger joints within 2 cm of the closest one, >= 3) are slid onto the
             object surface: hand shift delta (translation-only ICP onto the surface). The object stays where it is
             seen; the HAND moves (our hand position carries cm-level error, the scored object must start where seen).
5 lock       object(t) = wrist(t) @ G, G = wrist(onset)^-1 @ object(onset): translation and rotation follow the wrist.
             The hand shift rides with the wrist (delta rotated by the wrist's rotation since onset), so the
             fingers stay on the object. Shift eased in over EASE frames before onset and out after release.
6 landing    if the object is then seen resting (>= 5 well-visible, still frames), the position offset between the
             locked landing and that rest pose is spread linearly over the carry (object AND hand shift, so contact
             holds); rotation stays the wrist's.
Free stretches after a release keep the landed pose; a free stretch in which the object is seen moving (a lid
riding on a carried pot) follows the source, blended.
Writes <run>/<out>/<obj>/poses (camera<-object), <run>/stereo/<hands_out>/<side>/NNNNNN.json (hands variant with
the contact shift applied to joints and T) and <run>/<out>/grasp_lock_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python grasp_lock.py <run>... [--src fused_bfix_recon]"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from fuse_objects import stereo_hands
from handlock import ang, interp_poses, robust_pose, smooth_poses
from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])
SIG_WRIST, MOVE_M, VIS_OK, BLEND, EASE = 1.5, 0.03, 0.6, 3, 5
GAP, SHORT, MAX_SHIFT_M = 6, 8, 0.12
CONTACT_J = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]   # thumb + finger joints (OpenPose order)


def clean_intervals(ivs):
    """[(side, s, e)] sorted, non-overlapping (see module doc, step 1)."""
    per = {}
    for sd, lst in ivs.items():
        m = []
        for s, e in sorted(lst):
            if m and s - m[-1][1] <= GAP:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        per[sd] = m
    allv = [(sd, s, e) for sd, lst in per.items() for s, e in lst]
    keep = []
    for sd, s, e in allv:
        other = [(s2, e2) for sd2, s2, e2 in allv if sd2 != sd]
        if e - s + 1 < SHORT and any(s2 - GAP <= s and e <= e2 + GAP for s2, e2 in other):
            continue
        keep.append((sd, s, e))
    keep.sort(key=lambda x: x[1])
    out = []
    for sd, s, e in keep:
        if out and s <= out[-1][2]:
            if e <= out[-1][2]:
                continue
            s = out[-1][2] + 1
        out.append((sd, s, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_bfix_recon")
    ap.add_argument("--out", default="fused_glock_recon")
    ap.add_argument("--hands", default="hands_c")
    ap.add_argument("--hands_out", default="hands_glock")
    ap.add_argument("--no_drift", action="store_true", help="do not spread the landing position offset over the carry")
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        gv = json.loads((run / "grasp_vlm.json").read_text())
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        hands = stereo_hands(run, n, a.hands)
        wrist, Jw = {}, {}
        for sd, hd in hands.items():
            P = {t: slam[t] @ hd["H"][t] for t in range(n) if np.isfinite(hd["H"][t]).all()}
            if len(P) >= 2:
                wrist[sd] = interp_poses(smooth_poses(P, SIG_WRIST), n)
            Jw[sd] = np.array([hd["J"][t] @ slam[t][:3, :3].T + slam[t][:3, 3] for t in range(n)])
        shift = {sd: np.zeros((n, 3)) for sd in hands}
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if not src.is_dir() or nm not in gv:
                continue
            F = [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]
            cv = next(iter(sorted((run / obj_dir / nm).glob(".contact_vis_*.npz"))), None)
            vis = np.load(cv)["vis"][:n] if cv is not None else np.ones(n)
            S = trimesh.sample.sample_surface(trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh"), 4000, seed=0)[0]

            def surf(P):
                return cKDTree(S @ P[:3, :3].T + P[:3, 3])

            def hand_dist(sd, t, P):
                J = Jw[sd][t, CONTACT_J]
                return float(surf(P).query(J)[0].min()) if np.isfinite(J).all() else np.inf

            def contact_shift(sd, t, P):
                J = Jw[sd][t, CONTACT_J]
                if not np.isfinite(J).all():
                    return np.zeros(3)
                tree = surf(P)
                d = tree.query(J)[0]
                sel = np.argsort(d)[:max(3, int((d <= d.min() + 0.02).sum()))]
                delta = np.zeros(3)
                for _ in range(5):
                    q = tree.data[tree.query(J[sel] + delta)[1]]
                    delta += (q - (J[sel] + delta)).mean(0)
                nd = np.linalg.norm(delta)
                return delta if nd <= MAX_SHIFT_M else delta * MAX_SHIFT_M / nd

            def rest_pose(s, e):
                fr = list(range(s, max(e, s + 1)))
                w = np.array([max(float(vis[t]), 0.05) ** 2 for t in fr])
                return robust_pose(np.array([F[t][:3, :3] for t in fr]), np.array([F[t][:3, 3] for t in fr]), w)

            def seen_moving(s, e):
                c = np.array([F[t][:3, 3] for t in range(s, e)])
                return len(c) > 1 and np.linalg.norm(c - np.median(c, 0), axis=1).max() > MOVE_M

            events = [ev for ev in clean_intervals(gv[nm]["intervals"]) if ev[0] in wrist]
            out, mode = [None] * n, ["free"] * n
            cur, t0, log = None, 0, []

            def fill_free(s, e, after_event):
                nonlocal cur
                if e <= s:
                    return
                if after_event and seen_moving(s, e):
                    for t in range(s, e):
                        out[t], mode[t] = F[t], "source"
                    cur = F[e - 1]
                    return
                for t in range(s, e):
                    out[t] = cur.copy()
            for k, (sd, s, e) in enumerate(events):
                if s < t0:
                    s = t0
                if e - s + 1 < 3:
                    continue
                if cur is None:
                    cur = rest_pose(t0, s) if s > t0 else F[s].copy()
                win = range(max(t0, s - 3), min(e, s + 12) + 1)
                dd = {t: hand_dist(sd, t, cur) for t in win}
                dm = min(dd.values())
                ts = next(t for t in win if dd[t] <= dm + 0.005) if np.isfinite(dm) else s
                win2 = list(range(max(ts + 3, e - 10), min(n - 1, e + 3) + 1))
                d2 = {t: hand_dist(sd, t, F[t]) for t in win2}
                dm2 = min(d2.values()) if d2 else np.inf
                te = max([t for t in win2 if d2[t] <= dm2 + 0.01], default=e) if np.isfinite(dm2) else e
                fill_free(t0, ts, k > 0)
                delta = contact_shift(sd, ts, cur)
                Wd = wrist[sd]
                G = np.linalg.inv(Wd[ts]) @ cur
                for t in range(ts, te + 1):
                    out[t], mode[t] = Wd[t] @ G, "held:" + sd
                    shift[sd][t] = Wd[t][:3, :3] @ Wd[ts][:3, :3].T @ delta
                # landing: where the object is then seen resting
                nxt_s = events[k + 1][1] if k + 1 < len(events) else n
                land = None
                if not a.no_drift and nxt_s - (te + 1) >= 5 and not seen_moving(te + 1, nxt_s):
                    if sum(vis[t] >= VIS_OK for t in range(te + 1, nxt_s)) >= 5:
                        D = rest_pose(te + 1, nxt_s)[:3, 3] - out[te][:3, 3]
                        for t in range(ts, te + 1):
                            f = (t - ts + 1) / (te - ts + 1)
                            out[t] = out[t].copy()
                            out[t][:3, 3] += f * D
                            shift[sd][t] += f * D
                        land = round(float(np.linalg.norm(D)) * 100, 1)
                # ease the hand shift in / out
                for i in range(1, EASE + 1):
                    if ts - i >= 0 and not np.any(shift[sd][ts - i]):
                        shift[sd][ts - i] = shift[sd][ts] * (1 - i / (EASE + 1))
                    if te + i < n and not np.any(shift[sd][te + i]):
                        shift[sd][te + i] = shift[sd][te] * (1 - i / (EASE + 1))
                log.append({"hand": sd, "vlm": [s, e], "onset": ts, "release": te, "onset_gap_cm": round(dm * 100, 1),
                            "hand_shift_cm": round(float(np.linalg.norm(delta)) * 100, 1), "landing_offset_cm": land})
                cur, t0 = out[te].copy(), te + 1
            if cur is None:
                cur = rest_pose(0, n)
            fill_free(t0, n, bool(log))
            final = [T.copy() for T in out]
            for t in range(1, n):   # blend in / out of source-followed stretches (no teleport)
                if (mode[t] == "source") != (mode[t - 1] == "source"):
                    lo, hi = max(0, t - BLEND), min(n - 1, t + BLEND - 1)
                    A, B = out[max(lo - 1, 0)], out[min(hi + 1, n - 1)]
                    sl = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))
                    for u in range(lo, hi + 1):
                        al = (u - lo + 1) / (hi - lo + 2)
                        M = np.eye(4)
                        M[:3, :3], M[:3, 3] = sl(al).as_matrix(), (1 - al) * A[:3, 3] + al * B[:3, 3]
                        final[u] = M
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ final[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st = [ang(final[t + 1][:3, :3], final[t][:3, :3]) for t in range(n - 1)]
            report[nm] = {"grasps": log, "held_frames": sum(m.startswith("held") for m in mode),
                          "source_frames": mode.count("source"), "max_step_deg": round(max(st), 1)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        # hands with the contact shift (camera frame: joints and MANO translation)
        for sd in hands:
            sdir, odir = run / "stereo" / a.hands / sd, run / "stereo" / a.hands_out / sd
            shutil.rmtree(odir, ignore_errors=True)
            odir.mkdir(parents=True)
            for p in sorted(sdir.glob("[0-9]*.json")):
                t, r = int(p.stem), json.loads(p.read_text())
                if t < n and np.any(shift[sd][t]):
                    dc = slam[t][:3, :3].T @ shift[sd][t]
                    if r.get("joints") is not None:
                        r["joints"] = (np.asarray(r["joints"]) + dc).tolist()
                    r["T"] = (np.asarray(r["T"]) + dc).tolist()
                    r["grasp_lock_shift_cm"] = round(float(np.linalg.norm(dc)) * 100, 2)
                (odir / p.name).write_text(json.dumps(r))
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "grasp_lock_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
