"""When does each source say the object leaves the table? Height above the table (world, table_plane up) per frame
for mocap / FP / ICP / fused / joint, the stereo palm heights, and the SAM 3 mask centroid v (video), around lift-offs.
Usage: python lift_onset.py <run> <object>"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
    elif "T" in d and np.shape(d["T"]) == (4, 4):
        T = np.array(d["T"], float)
    elif "rotation" in d:
        q = d["rotation"]; T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = d["translation"]
    else:
        return None
    return T


import cv2  # noqa: E402
ROOT = Path(os.environ["V2D_ROOT"]); run = ROOT / "runs" / sys.argv[1]; nm = sys.argv[2]
slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}; n = len(slam)
tp = json.loads((run / "table_plane.json").read_text()); up, h0 = np.array(tp["up"]), float(tp["height"])
H = {}
H["mocap"] = {int(t): np.array(W)[:3, 3] @ up - h0 for t, W in json.loads((run / "gt_world.json").read_text())["objects"][nm].items()}
for v, d in (("fp", run / "objects_sam3" / nm / "poses"), ("icp", run / "objects_sam3" / nm / "stereo_icp"),
             ("fused", run / "fused_all_sam3" / nm / "poses"), ("joint", run / "fused_joint" / nm / "poses")):
    H[v] = {int(f.stem): (slam[int(f.stem)] @ T)[:3, 3] @ up - h0 for f in d.glob("*.json") if int(f.stem) in slam and (T := load_T(f)) is not None}
tri = json.loads((run / "stereo/hands_tri.json").read_text())
for side in ("left", "right"):
    X = tri.get(side, {}).get("X_smooth", {})
    H["palm_" + side] = {int(t): (slam[int(t)][:3, :3] @ np.array(x)[[0, 5, 9, 13, 17]].mean(0) + slam[int(t)][:3, 3]) @ up - h0 for t, x in X.items() if int(t) in slam}
for side in ("left", "right"):
    d = run / "stereo" / "mano_fit_joint" / side
    H["jointhand_" + side] = {int(f.stem): (slam[int(f.stem)][:3, :3] @ np.array(json.loads(f.read_text())["joints"])[[0, 5, 9, 13, 17]].mean(0) + slam[int(f.stem)][:3, 3]) @ up - h0 for f in d.glob("*.json") if int(f.stem) in slam} if d.exists() else {}
oid = [o["id"] for o in json.loads((run / "objects_sam3/objects.json").read_text()) if o["name"] == nm][0]
V = {}
for f in (run / "objects_sam3/masks" / str(oid)).glob("*.png"):
    m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if m is not None and (m > 0).sum() > 200:
        V[int(f.stem)] = float(np.nonzero(m)[0].mean())
rest = np.median([H["mocap"][t] for t in sorted(H["mocap"])[:10]])
lift = [t for t in range(1, n) if t in H["mocap"] and t - 1 in H["mocap"] and H["mocap"][t] - rest > 0.01 and H["mocap"][t - 1] - rest <= 0.01]
print(f"{sys.argv[1]} {nm}: mocap rest height {rest*100:.1f} cm; mocap lift-offs (>1 cm) at frames {lift}")
for L in lift[:3]:
    print(f"\n frame  " + "  ".join(f"{k:>12s}" for k in H) + "   mask_v(px)")
    for t in range(max(0, L - 12), min(n, L + 8)):
        print(f" {t:5d}  " + "  ".join(f"{(H[k][t] - (rest if not k.startswith(('palm', 'jointhand')) else 0)) * 100:12.1f}" if t in H[k] else f"{'-':>12s}" for k in H) + f"   {V.get(t, float('nan')):8.1f}")
    for k in H:
        if k.startswith(("palm", "jointhand")):
            continue
        on = [t for t in range(max(1, L - 20), min(n, L + 20)) if t in H[k] and H[k][t] - rest > 0.01]
        print(f"   {k:8s} first > 1 cm at frame {on[0] if on else None}")
