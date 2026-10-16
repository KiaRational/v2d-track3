"""Read-only smoke check on one public episode: old vs new verifier hand proximity (--no_hands) and the
agent objective. Writes nothing into the run (verify --quiet). Usage: python smoke_refine_eval.py <run> <snapshot>"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

run, snap = sys.argv[1], Path(sys.argv[2])
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "refine_agent"))
import agent  # noqa: E402
import verify as new  # noqa: E402

spec = importlib.util.spec_from_file_location("verify_old", snap / "refine_agent/verify.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
for nm_mod, mod in (("old", old), ("new", new)):
    rep = mod.main([run, "--objects", "fused_all_sam3", "--no_hands", "--quiet"])[run]
    for nm, ser in rep["object"].items():
        ma, un = ser["moved_alone"], ser["unsupported"]
        p, alone = agent.per_frame(rep, nm, agent.calibration(run))
        wins = agent.problem_windows(p, alone, 0.7)
        print(f"{nm_mod} {nm:16s} frames {len(ser['table_pen'])}  moved_alone {int(sum(ma.values()))}/{len(ma)}  "
              f"unsupported {int(sum(un.values()))}/{len(un)}  objective {agent.fmt(agent.objective(p, alone))} "
              f"(unscorable {agent.n_missing(p)})  windows {len(wins)} {wins[:5]}", flush=True)

# how often is a hand "near" (joints only / contacts only / either)? must not be ~always
import json  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import trimesh  # noqa: E402
R = new.ROOT / "runs" / run
tri = json.loads((R / "stereo/hands_tri.json").read_text())
HJ = {s: {int(t): X for t, X in (tri.get(s, {}).get("X_smooth") or {}).items() if X is not None} for s in ("left", "right")}
fus = json.loads((R / "fusion_report.json").read_text())
for o in json.loads((R / "objects_sam3/objects.json").read_text()):
    nm = o["name"]
    mesh = trimesh.load(new.ROOT / "meshes_obj" / nm / f"{nm}.obj", force="mesh")
    tree = cKDTree(trimesh.sample.sample_surface(mesh, 3000, seed=0)[0])
    c = new.contact_frames(fus.get(nm))
    pd = R / "fused_all_sam3" / nm / "poses"
    nj = nc = ne = n = 0
    for f in sorted(pd.glob("*.json")):
        t, Wc = int(f.stem), new.load_T(f)
        j = new.hand_near_joints(t, Wc, tree, HJ)
        n, nj, nc, ne = n + 1, nj + j, nc + (t in c), ne + (j or t in c)
    print(f"near-hand {nm:16s} frames {n}: joints {nj}  contacts {nc}  either {ne}", flush=True)
