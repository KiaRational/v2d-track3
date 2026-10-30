"""Per-frame observed-hand mask of one converted sequence, for the V2D training reward/termination
(flash_chord objectives/v2d_hand_valid.py, env V2D_HAND_VALID). From the export bundle: a frame is observed when the
hand is valid and NOT gap-filled; fills of <= 3 frames count as observed (short detector dropouts, the export's own
lerp is accurate there). Usage: python hand_valid.py <clip result.npz> <out.json>"""
import json
import sys

import numpy as np

z = np.load(sys.argv[1])
out = {}
for side in ("left", "right"):
    ok = np.asarray(z[f"hand_{side}_is_valid"], bool)
    fill = np.asarray(z[f"hand_{side}_is_interpolated"], bool) if f"hand_{side}_is_interpolated" in z.files else np.zeros_like(ok)
    obs = ok & ~fill
    d = np.diff(np.r_[0, fill.astype(int), 0]); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(s, e):
        if b - a <= 3:
            obs[a:b] = ok[a:b]
    out[side] = obs.astype(int).tolist()
    print(f"{side}: observed {obs.mean():.2f} of {len(obs)} frames", flush=True)
json.dump(out, open(sys.argv[2], "w"))
