"""Debug: score the FP pose (proxy GT) vs the global hypotheses for one frame.
Usage: python diag/dbg_global.py <run> <object> <frame> [mask_dir]"""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/kghasemz/v2d-vulcan/stereo")
import global_register as g
from object_icp import load_T
run_name, name, t = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = g.Run(g.ROOT / "runs" / run_name, sys.argv[4] if len(sys.argv) > 4 else "objects/masks")
o = [x for x in r.objs if x["name"] == name][0]
others = [x["id"] for x in r.objs if x["id"] != o["id"]]
g.build_model(name); M = g.get_model(name)
P, info = g.segment(r, o["id"], others, t)
V = g.make_view(np.eye(4), info["depth"], info["obj"], info["occ"], info.get("plane"))
Tfp = load_T(r.dir / "objects" / name / "poses_smoothed" / f"{t:06d}.json")
Tr = g.icp_refine(M, P, cKDTree(P), [V], Tfp, r.K)
print(run_name, name, t, "n_scan", len(P), "plane", info.get("plane") is not None)
print(" FP     ", {k: round(v, 3) for k, v in g.full_score(M, P, [V], r.K, Tfp).items()})
print(" FP+icp ", {k: round(v, 3) for k, v in g.full_score(M, P, [V], r.K, Tr).items()}, "moved mm", round(np.linalg.norm(Tr[:3, 3] - Tfp[:3, 3]) * 1000, 1))
Ts, cost = g.coarse(M, P, [V], r.K, np.r_[g.rotations(), Tfp[None, :3, :3]])
ns = len(cost) // (len(g.rotations()) + 1)
fpc = cost[-ns:]
j = int(np.argmin(fpc))
print(" coarse: FP-rot best seed", j, "cost", round(float(fpc[j]), 3), "rank", int((cost < fpc[j]).sum()), "of", len(cost),
      "t err cm", round(np.linalg.norm(Ts[-ns + j][:3, 3] - Tr[:3, 3]) * 100, 2), "| best", round(float(cost.min()), 3))
order = np.argsort(cost)
keep = g.nms(M, Ts, order[:800], g.N_REFINE, dt=0.02, dr=20.0)
print(" NMS candidates within 20deg & 3cm of FP+icp:", [(k, round(g.rot_dist(M, Tr[:3, :3], Ts[i][:3, :3]), 1)) for k, i in enumerate(keep)
      if g.rot_dist(M, Tr[:3, :3], Ts[i][:3, :3]) < 20 and np.linalg.norm(g.centre_of(M, Ts[i]) - g.centre_of(M, Tr)) < 0.03])
for h in g.register(M, P, [V], r.K)[:4]:
    print(" hyp", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items() if k != "T"},
          "rot_to_fpicp", round(g.rot_dist(M, Tr[:3, :3], h["T"][:3, :3]), 1),
          "ctr_cm", round(np.linalg.norm(g.centre_of(M, h["T"]) - g.centre_of(M, Tr)) * 100, 2))
