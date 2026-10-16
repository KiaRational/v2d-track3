"""Where along the dust pan (handle ring -> pan) are the human fingertips, human contacts and Sharpa fingertips?
Uses the npz cache written by ref_contacts_plot.py (object frame, projected on the mesh's long axis)."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation
d = np.load(sys.argv[1])
V = np.array([l.split()[1:4] for l in open(str(d["mesh"])) if l.startswith("v ")], float)
c = V.mean(0); ax = np.linalg.svd(V - c, full_matrices=False)[2][0]
proj = (V - c) @ ax
if np.mean(proj < -0.06) > 0.5: ax = -ax
proj = (V - c) @ ax
op, oq = d["object_body_position"][:, 0], d["object_body_wxyz"][:, 0]
R = Rotation.from_quat(oq[:, [1, 2, 3, 0]])
names = {"right": [str(x) for x in d["rnames"]], "left": [str(x) for x in d["lnames"]]}
print(f"mesh along long axis: {proj.min():.3f} .. {proj.max():.3f} m")
h = np.histogram(proj, bins=19); print("mesh cross-section profile (verts per 1 cm bin):", h[0].tolist())
for s in ("right", "left"):
    cp = d[f"mano_{s}_object_contact_positions"]; m = np.abs(cp).sum(-1) > 0
    tips = [k for k, n in enumerate(names[s]) if n.endswith("_tip_site")]
    fr = np.where(m.sum(1) >= 3)[0]
    if not len(fr): continue
    loc = lambda P, i: (R[i].inv().apply(P - op[i]) - c) @ ax
    mt = np.concatenate([loc(d[f"mano_{s}_joints"][i][[4, 8, 12, 16, 20]], i) for i in fr])
    st = np.concatenate([loc(d[f"robot_{s}_frames"][i][tips, :3], i) for i in fr])
    mc = np.concatenate([loc(cp[i][m[i]], i) for i in fr])
    pal = np.concatenate([loc(d[f"mano_{s}_joints"][i][[0]], i) for i in fr])
    q = lambda x: np.percentile(x, [10, 50, 90]).round(3).tolist()
    print(f"{s} ({len(fr)} grasp frames) position along axis, p10/p50/p90 (m):\n  human wrist {q(pal)}\n  human contacts {q(mc)}\n"
          f"  human fingertips {q(mt)}\n  Sharpa fingertips {q(st)}")
