"""Trace ICP from the FP pose (proxy GT) for one frame: per-iteration step and residuals,
for the full symmetric ICP and for scan->mesh only."""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/kghasemz/v2d-vulcan/stereo")
import global_register as g
from object_icp import load_T, kabsch
run_name, name, t = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = g.Run(g.ROOT / "runs" / run_name, "objects/masks")
o = [x for x in r.objs if x["name"] == name][0]
others = [x["id"] for x in r.objs if x["id"] != o["id"]]
g.build_model(name); M = g.get_model(name)
P, info = g.segment(r, o["id"], others, t)
V = g.make_view(np.eye(4), info["depth"], info["obj"], info["occ"], info.get("plane"))
Tfp = load_T(r.dir / "objects" / name / "poses_smoothed" / f"{t:06d}.json")
d0 = M["tree"].query((P - Tfp[:3, 3]) @ Tfp[:3, :3])[0]
print("scan->mesh dist at FP: pct 50/80/95 mm", np.percentile(d0, [50, 80, 95]) * 1000, "n", len(P))
for it in (1, 2, 3, 5, 10, 20):
    T = g.icp_refine(M, P, cKDTree(P), [V], Tfp, r.K, iters=it)
    print(" iters", it, "moved mm", round(np.linalg.norm(T[:3, 3] - Tfp[:3, 3]) * 1000, 1), "rot", round(g.rot_dist(M, Tfp[:3, :3], T[:3, :3]), 1),
          "score", round(g.full_score(M, P, [V], r.K, T)["score"], 3))
# scan->mesh only
T = Tfp.copy()
for it in range(20):
    R, tt = T[:3, :3], T[:3, 3]
    d1, i1 = M["tree"].query((P - tt) @ R)
    k1 = np.argsort(d1)[: int(0.8 * len(d1))]
    dR, dt = kabsch(M["pts"][i1[k1]] @ R.T + tt, P[k1])
    D = np.eye(4); D[:3, :3], D[:3, 3] = dR, dt
    T = D @ T
print(" s2m-only 20 it: moved mm", round(np.linalg.norm(T[:3, 3] - Tfp[:3, 3]) * 1000, 1), "rot", round(g.rot_dist(M, Tfp[:3, :3], T[:3, :3]), 1),
      "score", round(g.full_score(M, P, [V], r.K, T)["score"], 3))
