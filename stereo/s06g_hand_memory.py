"""Stage 6g: hand memory through tracking gaps, carried by the held object (agile image).

When a hand carrying an object leaves the camera view (or is only extrapolated, with neither
a triangulation nor a 2D detection), its track has nothing while the object keeps moving on
its fused trajectory -- the object floats with no hand holding it, which the policy stage
cannot reproduce (CHORD needs a contact to explain every motion).

While a hand holds a rigid object, the hand's pose in the object's frame is fixed. So a gap
that sits inside a grasp is filled from the object:
  grasp context  the hand touched object o within --near frames before the gap (t_b) and/or
                 after it (t_a) (fusion_report.json hand_contact, stereo-joint contact test)
  hand frame     F = [R_global_orient | wrist position], world frame via SLAM
  memory         G_b = W_o(t_b)^-1 F(t_b), G_a likewise; F(t) = W_o(t) G(t) with G interpolated
                 (SLERP + lerp) between the two ends, or held at G_b if the hand never returns
  fingers        hand_pose interpolated between t_b and t_a (held if only one end exists)
  release        with no return to the same object, frames are only filled while the object is
                 still moving (> --move_mm from its pose at t_b): a hand that let go and left
                 is not invented
Output: stereo/<hands>_mem/<side>/NNNNNN.json (inputs copied, filled frames flagged
"memory": true, "memory_object": o) + report.json.
Usage: python s06g_hand_memory.py <run> [--hands mano_fit] [--variant fused_all_sam3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", default="mano_fit")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--near", type=int, default=6, help="contact within this many frames of the gap counts")
    ap.add_argument("--max_gap", type=int, default=90, help="longest gap bridged when the hand returns")
    ap.add_argument("--move_mm", type=float, default=10.0)
    ap.add_argument("--min_valid", type=int, default=5, help="fewer triangulated joints than this = not observed")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    n = max(slam) + 1
    rep = json.loads((run / "fusion_report.json").read_text())
    objs = {}
    for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
        pd = run / a.variant / o["name"] / "poses"
        objs[o["name"]] = {t: slam[t] @ load_T(pd / f"{t:06d}.json") for t in slam if (pd / f"{t:06d}.json").exists()}
    mano = Mano(str(ROOT / "weights/hand"), "cuda")
    src, dst = run / "stereo" / a.hands, run / "stereo" / f"{a.hands}_mem"
    report = {"args": vars(a), "sides": {}}

    def wrist_local(rec):
        _, j = mano.local(rec)
        return j[0].cpu().numpy()

    for side in ("left", "right"):
        (dst / side).mkdir(parents=True, exist_ok=True)
        recs = {}
        for f in sorted((src / side).glob("*.json")):
            r = json.loads(f.read_text())
            recs[int(f.stem)] = r
            shutil.copy(f, dst / side / f.name)
        # observed = has data; frames the fit only extrapolated, or whose triangulation had too
        # few valid joints (a hand at the fisheye edge, joints filled in), count as missing
        weak = set()
        tri = run / "stereo/hands_tri.json"
        if tri.exists():
            for st in json.loads(tri.read_text()).get(side, {}).get("stats", []):
                if st.get("source") == "interp" or (st.get("n_valid") is not None and st["n_valid"] < a.min_valid):
                    weak.add(int(st["frame"]))
        obs = {t for t, r in recs.items()
               if (r.get("has_triangulation", True) or r.get("has_wilor_2d", True)) and t not in weak}

        def F_world(t):
            r = recs[t]
            Rg = Rotation.from_rotvec(r["mano"]["global_orient"]).as_matrix()
            F = np.eye(4)
            F[:3, :3], F[:3, 3] = Rg, wrist_local(r) + np.array(r["T"])
            return slam[t] @ F

        contact = {nm: set(rep.get(nm, {}).get("hand_contact", {}).get(side, [])) for nm in objs}
        filled, t = [], 0
        while t < n:
            if t in obs or t not in slam:
                t += 1
                continue
            g0 = t
            while t + 1 < n and (t + 1) not in obs:
                t += 1
            g1 = t
            t += 1
            tb = max((u for u in obs if u < g0), default=None)
            ta = min((u for u in obs if u > g1), default=None)
            best = None
            for nm, W in objs.items():
                cb = tb is not None and tb in W and any(u in contact[nm] for u in range(tb - a.near, g0))
                ca = ta is not None and ta in W and any(u in contact[nm] for u in range(g1 + 1, ta + a.near + 1))
                if cb or ca:
                    score = int(cb) + int(ca)
                    if best is None or score > best[0]:
                        best = (score, nm, cb, ca)
            if best is None:
                continue
            _, nm, cb, ca = best
            W = objs[nm]
            if not cb and g1 - g0 + 1 > a.max_gap:
                continue  # only a far-away return: too long to bridge
            ends = []
            if cb:
                ends.append((tb, np.linalg.inv(W[tb]) @ F_world(tb), np.array(recs[tb]["mano"]["hand_pose"])))
            if ca and g1 - g0 + 1 <= a.max_gap:
                ends.append((ta, np.linalg.inv(W[ta]) @ F_world(ta), np.array(recs[ta]["mano"]["hand_pose"])))
            if not ends:
                continue
            ref = recs[ends[0][0]]
            for u in range(g0, g1 + 1):
                if u not in W or u not in slam:
                    continue
                if len(ends) == 1 and cb and not ca:
                    # no return to this object: only while the object is still being carried
                    if np.linalg.norm(W[u][:3, 3] - W[tb][:3, 3]) * 1000 < a.move_mm and u > g0 + a.near:
                        break
                if len(ends) == 2:
                    al = (u - ends[0][0]) / (ends[1][0] - ends[0][0])
                    rots = Rotation.from_matrix([ends[0][1][:3, :3], ends[1][1][:3, :3]])
                    G = np.eye(4)
                    G[:3, :3] = Slerp([0, 1], rots)(al).as_matrix()
                    G[:3, 3] = (1 - al) * ends[0][1][:3, 3] + al * ends[1][1][:3, 3]
                    hp = (1 - al) * ends[0][2] + al * ends[1][2]
                else:
                    G, hp = ends[0][1], ends[0][2]
                Fc = np.linalg.inv(slam[u]) @ W[u] @ G
                rec = {"mano": {"betas": ref["mano"]["betas"], "global_orient": Rotation.from_matrix(Fc[:3, :3]).as_rotvec().tolist(),
                                "hand_pose": hp.tolist()}, "is_right": ref["is_right"]}
                rec["T"] = (Fc[:3, 3] - wrist_local(rec)).tolist()
                rec.update(memory=True, memory_object=nm, has_triangulation=False, has_wilor_2d=False)
                _, j = mano.local(rec)
                rec["joints"] = (j.cpu().numpy() + np.array(rec["T"])).tolist()
                (dst / side / f"{u:06d}.json").write_text(json.dumps(rec))
                filled.append(u)
        report["sides"][side] = {"filled": len(filled), "filled_frames": filled}
        print(f"{side}: filled {len(filled)} frames from held objects", flush=True)
    (dst / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
