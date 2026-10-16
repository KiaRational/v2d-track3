"""Per c x b pair details for given stereo frames (host venv). args: run s0 s1 [suffix]"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/v2d-vulcan/stereo"))
from hand_triangulation import triangulate_joints
ROOT = Path(os.environ["V2D_ROOT"]); run = ROOT / "runs" / sys.argv[1]
W = json.loads((run / "stereo/wilor2d.json").read_text())
if len(sys.argv) > 4:
    E = json.loads((run / f"stereo/wilor2d{sys.argv[4]}.json").read_text())
    for v in "cb": W[v] = [W[v][t] + E[v][t] for t in range(len(W[v]))]
cr = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
si = json.loads((run / "stereo/intrinsics.json").read_text())
ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
Ka = np.array([[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.]])
z = np.zeros(14)
calib = {"cameras": {"a": {"K": Ka, "dist": z}, "c": {"K": np.array(si["ego_cam_c"]["K"]), "dist": z},
                     "b": {"K": np.array(si["ego_cam_b"]["K"]), "dist": z}},
         "T_left_to_right": np.array(cr["stereo"]["T_left_to_right"]), "T_a_to_c": np.array(cr["T_ego_cam_a_to_ego_cam_c"])}
for s in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
    for ci, c in enumerate(W["c"][s]):
        for bi, b in enumerate(W["b"][s]):
            obs = {"c": (np.array(c["j2d"]), np.full(21, c["score"])), "b": (np.array(b["j2d"]), np.full(21, b["score"]))}
            X, rms, _, ok = triangulate_joints(obs, calib, ref="a", max_px=10)
            X2, rms2, _, ok2 = triangulate_joints(obs, calib, ref="a", max_px=1e6, depth_range=(0.01, 10))
            wm = np.linalg.norm(X2[0] - X2[9])
            print(f"s{s} c{ci}{'R' if c['is_right'] else 'L'}{c['score']:.2f} x{np.mean(c['bbox'][::2]):.0f} "
                  f"b{bi}{'R' if b['is_right'] else 'L'}{b['score']:.2f} x{np.mean(b['bbox'][::2]):.0f}: ok={ok.sum()} "
                  f"rms_med={np.nanmedian(rms2):.1f} z={np.nanmedian(X2[:, 2]):.2f} wrist-mcp={wm:.3f}")
