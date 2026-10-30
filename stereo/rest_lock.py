"""Rest lock: an object no hand touches cannot rotate or drift. For every maximal stretch of frames without hand
contact (fusion_report hand_contact, dilated by PAD frames) of >= MIN_LEN frames in which the fused object centre stays
within MOVE_M (so a missed contact on a really moving object is NOT frozen), the pose is set to one constant pose: the
rotation medoid and the median position of the stretch. Writes the variant's poses in place (originals in
<obj>/poses_prelock) + <obj>/rest_lock.json. Usage: V2D_OBJ_DIR=objects_recon python rest_lock.py <run> [--variant V]"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
PAD, MIN_LEN, MOVE_M = 3, 6, 0.02


def load(p):
    d = json.loads(p.read_text()); M = np.eye(4)
    if "R" in d:
        M[:3, :3], M[:3, 3] = np.asarray(d["R"]), d["T"]; return M
    from scipy.spatial.transform import Rotation
    q = d["rotation"]; M[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M[:3, 3] = d["translation"]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variant", default="fused_all_recon")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    rep = json.loads((run / "fusion_report_recon.json").read_text())
    slam = {int(p.stem): load(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    for objd in sorted((run / a.variant).iterdir()):
        nm = objd.name
        if nm not in rep or not isinstance(rep[nm], dict):
            continue
        src = objd / ("poses_prelock" if (objd / "poses_prelock").exists() else "poses")
        ts = sorted(int(p.stem) for p in src.glob("[0-9]*.json"))
        if not ts:
            continue
        n = ts[-1] + 1
        touch = np.zeros(n, bool)
        for v in rep[nm].get("hand_contact", {}).values():
            for t in v:
                touch[max(0, t - PAD): t + PAD + 1] = True
        W = {t: slam[t] @ load(src / f"{t:06d}.json") for t in ts if t in slam}
        segs, out = [], {}
        free = [t for t in range(n) if not touch[t] and t in W]
        runs_, cur = [], []
        for t in free:
            if cur and t != cur[-1] + 1:
                runs_.append(cur); cur = []
            cur.append(t)
        if cur:
            runs_.append(cur)
        for r in runs_:
            if len(r) < MIN_LEN:
                continue
            C = np.array([W[t][:3, 3] for t in r])
            if np.linalg.norm(C - np.median(C, 0), axis=1).max() > MOVE_M:
                segs.append({"frames": [r[0], r[-1]], "locked": False, "reason": "moves"}); continue
            Rs = np.array([W[t][:3, :3] for t in r])
            tr = np.einsum("aij,bij->ab", Rs, Rs)
            A = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
            k = int(np.argmin(A.sum(1)))
            Wl = np.eye(4); Wl[:3, :3], Wl[:3, 3] = Rs[k], np.median(C, 0)
            for t in r:
                out[t] = Wl
            segs.append({"frames": [r[0], r[-1]], "locked": True, "rot_range_deg": round(float(A[k].max()), 1)})
        if not out:
            (objd / "rest_lock.json").write_text(json.dumps({"segments": segs}, indent=1)); continue
        if not (objd / "poses_prelock").exists():
            shutil.copytree(objd / "poses", objd / "poses_prelock")
        for t in ts:
            if t in out:
                M = np.linalg.inv(slam[t]) @ out[t]
                (objd / "poses" / f"{t:06d}.json").write_text(json.dumps({"R": M[:3, :3].tolist(), "T": M[:3, 3].tolist()}))
        (objd / "rest_lock.json").write_text(json.dumps({"segments": segs, "locked_frames": len(out)}, indent=1))
        print(a.run, nm, f"locked {len(out)} frames in {sum(s['locked'] for s in segs)} rest stretches; "
              f"max rot range removed {max([s.get('rot_range_deg', 0) for s in segs] + [0])} deg", flush=True)


if __name__ == "__main__":
    main()
