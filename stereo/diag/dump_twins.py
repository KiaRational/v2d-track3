"""Per stereo frame: c x b label-consistent pairs incl. flipped twins (agile image). args: run s0 s1"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/code/agile"); sys.path.insert(0, "/code/stereo")
from geometry import Mano, wilor_keypoints_2d
from hand_triangulation import triangulate_joints
ROOT = Path(os.environ["V2D_ROOT"]); run = ROOT / "runs" / sys.argv[1]
mano = Mano(str(ROOT / "weights/hand"), device="cpu")
cr = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
si = json.loads((run / "stereo/intrinsics.json").read_text())
ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
Ka = np.array([[ia["fx"], 0, ia["cx"]], [0, ia["fy"], ia["cy"]], [0, 0, 1.]]); z = np.zeros(14)
calib = {"cameras": {"a": {"K": Ka, "dist": z}, "c": {"K": np.array(si["ego_cam_c"]["K"]), "dist": z},
                     "b": {"K": np.array(si["ego_cam_b"]["K"]), "dist": z}},
         "T_left_to_right": np.array(cr["stereo"]["T_left_to_right"]), "T_a_to_c": np.array(cr["T_ego_cam_a_to_ego_cam_c"])}
def dets(v, s):
    out = []
    for suf in ("", "_flip", "_clahe", "_clahe_flip"):
        f = run / f"stereo/wilor_{v}{suf}/{s:06d}.json"
        for r in (json.loads(f.read_text()) if f.exists() else []):
            if r.get("score", 1) < 0.3: continue
            _, j = mano.local(r)
            out.append((suf or "orig", r["is_right"], np.mean([r["bbox"]["x0"], r["bbox"]["x1"]]),
                        wilor_keypoints_2d(j.numpy(), r, (1280, 800))))
    return out
for s in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
    for c in dets("c", s):
        for b in dets("b", s):
            if c[1] != b[1] or abs(c[2] - b[2]) > 300: continue
            obs = {"c": (c[3], np.ones(21)), "b": (b[3], np.ones(21))}
            X, rms, _, ok = triangulate_joints(obs, calib, ref="a", max_px=10)
            X2, rms2, _, _ = triangulate_joints(obs, calib, ref="a", max_px=1e6, depth_range=(0.01, 10))
            print(f"s{s} c:{c[0]}{'R' if c[1] else 'L'} x{c[2]:.0f}  b:{b[0]} x{b[2]:.0f}  ok={ok.sum()} "
                  f"rms={np.nanmedian(rms2):.1f} z={np.nanmedian(X2[:,2]):.2f} wm={np.linalg.norm(X2[0]-X2[9]):.3f}")
