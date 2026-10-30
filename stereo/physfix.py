"""Physical consistency of object trajectories (after branchfix.py): rest contact and airborne grasps.

  rest     untouched segments (no hand contact; branchfix/rigidify made them static): the object RESTS ON THE TABLE --
           shifted along the table normal so its lowest mesh point is at the table top (spoons sat 3-8 mm inside the
           table, an e25 board 11 mm, in every variant incl. raw FoundationPose).
  airborne frames of a grasp segment where the object is off the table (lowest point > AIR_M above it): the object
           is rigid in the hand -- object = palm(t) @ G, G fixed at the first airborne frame of the run (lift-off
           continuity). The user saw a grabbed spoon spin and float in the air (e03). On the table the fused motion
           is kept (vs public mocap a fully hand-rigid model lost accuracy: 0.86 -> 0.72 moving-ok).
Changes are blended over BLEND frames at every modified run's edges. Writes <run>/<out>/<obj>/poses + physfix_report.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python physfix.py <run>... [--src fused_bfix_recon] [--out fused_phys_recon]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

from fuse_objects import stereo_hands
from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])
AIR_M, BLEND, MIN_RUN = 0.02, 3, 4


def runs_of(mask):
    out, s = [], None
    for t, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = t
        elif not m and s is not None:
            out.append((s, t))
            s = None
    return out


def blend(out, orig, s, e, n):
    """Ease the modified run [s, e) into the original poses over BLEND frames on both sides."""
    for t0, t1, toward in ((max(0, s - BLEND), s, "in"), (e, min(n, e + BLEND), "out")):
        for t in range(t0, t1):
            w = (t - t0 + 1) / (t1 - t0 + 1)
            A, B = (orig[t], out[s]) if toward == "in" else (out[e - 1], orig[t])
            M = np.eye(4)
            M[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(w).as_matrix()
            M[:3, 3] = (1 - w) * A[:3, 3] + w * B[:3, 3]
            out[t] = M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_bfix_recon")
    ap.add_argument("--out", default="fused_phys_recon")
    ap.add_argument("--hands", default=os.environ.get("V2D_FUSE_HANDS", "hands_c"))
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    suf = "_" + obj_dir.replace("objects_", "")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        tp = json.loads((run / "table_plane.json").read_text())
        up, h = np.array(tp["up"], float), float(tp["height"])
        rep = json.loads((run / f"fusion_report{suf}.json").read_text())
        hands = stereo_hands(run, n, a.hands)
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if not src.is_dir() or nm not in rep:
                continue
            V = np.asarray(trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_recon") / nm / f"{nm}.obj",
                                        force="mesh").vertices)
            V = V[:: max(1, len(V) // 2000)]
            F = [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]
            low = lambda T: float(((V @ T[:3, :3].T + T[:3, 3]) @ up).min() - h)
            contact = {t for v in rep[nm].get("hand_contact", {}).values() for t in v}
            held = {}
            for g in rep[nm].get("grasp_segments", []):
                Hs = hands.get(g["side"], {}).get("H")
                for t in range(g["from"], g["to"] + 1):
                    if Hs is not None and np.isfinite(Hs[t]).all():
                        held[t] = g["side"]
            out = [T.copy() for T in F]
            # rest contact on untouched runs
            n_rest = 0
            for s, e in runs_of([t not in contact and t not in held for t in range(n)]):
                if e - s < MIN_RUN:
                    continue
                dz = -np.median([low(F[t]) for t in range(s, e)])
                if abs(dz) < 0.001:
                    continue
                for t in range(s, e):
                    out[t][:3, 3] = F[t][:3, 3] + dz * up
                blend(out, F, s, e, n)
                n_rest += e - s
            # airborne grasps: rigid in the hand from lift-off
            n_air = 0
            for s, e in runs_of([t in held and low(F[t]) > AIR_M for t in range(n)]):
                if e - s < MIN_RUN or len({held[t] for t in range(s, e)}) > 1:
                    continue
                Hs = hands[held[s]]["H"]
                Hw = {t: slam[t] @ Hs[t] for t in range(s, e)}
                G = np.linalg.inv(Hw[s]) @ out[s]
                for t in range(s, e):
                    out[t] = Hw[t] @ G
                blend(out, F, s, e, n)
                n_air += e - s
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t in range(n):
                C = np.linalg.inv(slam[t]) @ out[t]
                (od / f"{t:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            report[nm] = {"rest_frames": n_rest, "airborne_rigid_frames": n_air,
                          "initial_low_mm_before": round(low(F[0]) * 1000, 1), "initial_low_mm_after": round(low(out[0]) * 1000, 1)}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "physfix_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
