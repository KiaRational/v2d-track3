#!/bin/bash
#SBATCH --job-name=dustpan-thin
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/kghasemz/v2d/logs/%x_%j.out
# dust pan mesh thinned to the stereo-depth thickness (diag/mesh_fit.py dustpan_fit: x0.30 about the bottom face,
# top error 17.3 -> 2.3 mm); backup mini_dust_pan.prethin.obj (never compounded)
source ~/v2d-vulcan/env.sh
python - <<'PY'
import json, shutil, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path.home() / "v2d-vulcan/stereo"))
from object_icp import load_T
R = Path("/scratch/kghasemz/v2d"); d = R / "meshes_recon/mini_dust_pan"; S = 0.30
src = d / "mini_dust_pan.prethin.obj"
if not src.exists():
    shutil.copy(d / "mini_dust_pan.obj", src)
m = trimesh.load(src, force="mesh", process=False); V = np.asarray(m.vertices)
_, evec = np.linalg.eigh(np.cov((V - V.mean(0)).T)); thin = evec[:, 0]
run = R / "runs/evaluation_ep000038"; up = np.asarray(json.loads((run / "table_plane.json").read_text())["up"])
T0 = load_T(run / "recon_hawor/slam_poses/000000.json") @ load_T(run / "tpfsut_recon/mini_dust_pan/poses/000000.json")
if (T0[:3, :3] @ thin) @ up < 0: thin = -thin
dd = V @ thin; out = m.copy(); out.vertices = V - np.outer((dd - dd.min()) * (1 - S), thin); out.export(d / "mini_dust_pan.obj")
p = d / "PROVENANCE.json"
if p.exists():
    j = json.loads(p.read_text()); j["thickness_fit"] = {"scale": S, "axis_obj": thin.tolist(), "evidence": "stereo depth e38 rest, top error 17.3 -> 2.3 mm"}; p.write_text(json.dumps(j, indent=1))
print("thickness cm", round(float(np.ptp(dd)) * 100, 2), "->", round(float(np.ptp(np.asarray(out.vertices) @ thin)) * 100, 2))
PY
