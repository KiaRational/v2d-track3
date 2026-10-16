import sys, json, numpy as np
sys.path.insert(0, "/code/agile")
from geometry import Mano, wilor_keypoints_2d, pnp_translation
R = "/scratch/kghasemz/v2d/runs/public_ep000011"
intr = json.load(open(f"{R}/recon_hawor/intrinsics_stable.json")); K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
mano = Mano("/scratch/kghasemz/v2d/weights/hand")
t = 24
proj = lambda P: (P[..., :2] / P[..., 2:3]) @ K[:2, :2].T + K[:2, 2]
import os
for side in ("left", "right"):
    if not os.path.exists(f"{R}/smoke_hc/hands/{side}/{t:06d}.json"):
        print(side, "absent at frame", t); continue
    h = json.load(open(f"{R}/smoke_hc/hands/{side}/{t:06d}.json"))
    v, j = mano.local({"mano": h["mano"], "is_right": h["is_right"]}); v, j = v.cpu().numpy(), j.cpu().numpy()
    rec = [d for d in json.load(open(f"{R}/recon_hawor/wilor_raw/{t:06d}.json")) if d["is_right"] == h["is_right"]][0]
    j2d = wilor_keypoints_2d(j, rec, (2028, 1520))
    T = np.array(h["T"]); P = h["s_h"] * j + T
    print(side, "T", T.round(3), "s_h", round(h["s_h"], 3), "L_joint", h["L_joint"])
    print("   J2D (WiLoR) root/tip:", j2d[0].round(0), j2d[8].round(0), " bbox", rec["bbox"])
    print("   our proj root/tip  :", proj(P)[0].round(0), proj(P)[8].round(0), " local joint extent (m):", np.ptp(j, 0).round(3))
for o in ("beige_cup", "blue_cup"):
    p = json.load(open(f"{R}/smoke_hc/{o}/poses/{t:06d}.json")); T = np.array(p["T"])
    print(o, "T", T.round(3), "proj center", proj(T[None])[0].round(0), "scale", np.round(p["scale"], 3))
