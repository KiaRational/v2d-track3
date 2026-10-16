"""Resting objects vs the table: lowest mesh vertex height above the table plane (world) on frames before the
first hand contact (median, cm), per source. Positive = floating, negative = sunk. Usage: python rest_gap.py <run> [...]"""
import json, os, sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT = Path(os.environ["V2D_ROOT"])


def load_T(p):
    d = json.loads(Path(p).read_text()); T = np.eye(4)
    if "R" in d: T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
    elif "T" in d and np.shape(d["T"]) == (4, 4): T = np.array(d["T"], float)
    elif "rotation" in d:
        q = d["rotation"]; T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = d["translation"]
    else: return None
    return T


for r in sys.argv[1:]:
    run = ROOT / "runs" / r
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    tp = json.loads((run / "table_plane.json").read_text()); up, h0 = np.array(tp["up"]), float(tp["height"])
    rep = json.loads((run / "fusion_report.json").read_text()); gt = json.loads((run / "gt_world.json").read_text())["objects"]
    for o in json.loads((run / "objects_sam3/objects.json").read_text()):
        nm = o["name"]; V = trimesh.load(ROOT / "meshes_obj" / nm / f"{nm}.obj", force="mesh").vertices
        hc = [t for fr in rep.get(nm, {}).get("hand_contact", {}).values() for t in fr]
        first = min(hc) if hc else len(slam)
        fr = [t for t in range(0, max(first - 3, 0)) if t in slam]
        out = []
        for v, d in (("fp", run / "objects_sam3" / nm / "poses"), ("icp", run / "objects_sam3" / nm / "stereo_icp"),
                     ("fused_all", run / "fused_all_sam3" / nm / "poses"), ("joint", run / "fused_joint" / nm / "poses")):
            g = []
            for t in fr:
                f = d / f"{t:06d}.json"
                if f.exists() and (T := load_T(f)) is not None:
                    W = slam[t] @ T; g.append(((V @ W[:3, :3].T + W[:3, 3]) @ up).min() - h0)
            out.append(f"{v} {np.median(g) * 100:+.2f}" if g else f"{v} -")
        g = [((V @ np.array(W)[:3, :3].T + np.array(W)[:3, 3]) @ up).min() - h0 for t, W in gt.get(nm, {}).items() if int(t) in fr]
        out.append(f"mocap {np.median(g) * 100:+.2f}" if g else "mocap -")
        print(f"{r} {nm:18s} rest frames {len(fr):3d} (first contact {first}):  " + "  ".join(out) + "  cm")
