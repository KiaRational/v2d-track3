"""Agentic grasp detection: a local VLM (Qwen3-VL via the vLLM server, nothing leaves the cluster) decides per frame
WHICH hand is grasping each object. Only frames where a hand comes near the object (candidate windows: closest hand
joint within CAND_M of the object surface at the source pose, padded by PAD frames) are asked, every STRIDE-th
frame; answers are filled in between, majority-filtered and debounced into grasp intervals [onset, release] per hand.
Each query shows the colour frame cropped around the object, the object's mask outlined in green and the two wrists
labelled L / R (projected hands joints).
Writes <run>/grasp_vlm.json {obj: {"intervals": {side: [[on, off], ...]}, "answers": {frame: side-set}}} and a
contact sheet per object to $V2D_VIZ/grasp_vlm/<run>_<obj>.jpg.
Usage (agile image, vLLM on port 8011): python grasp_vlm.py <run>... [--src fused_bfix_recon] [--hands hands_c]"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from fuse_objects import stereo_hands
from hoi4d.llm import ChatClient, image_part, parse_json_reply
from object_icp import load_T

ROOT = Path(os.environ["V2D_ROOT"])
CAND_M, PAD, STRIDE, MIN_RUN, FILL = 0.15, 6, 2, 4, 4

PROMPT = ("This is a frame from a head-mounted camera looking down at a table. The object outlined in GREEN is a {obj}. "
          "The person's hands are marked L (left wrist) and R (right wrist) when visible.\n"
          "Is a hand GRASPING the {obj} right now -- fingers closed around it or holding it so that the hand could lift or "
          "move it? A hand merely near it, hovering, pointing, or resting next to it is NOT grasping. A hand holding "
          "a DIFFERENT object is not grasping the {obj}.\n"
          'Reply with JSON only: {{"left": true|false, "right": true|false}}')


def crop_view(img, mask, wrists, K_pad=120):
    ys, xs = np.nonzero(mask) if mask is not None and mask.any() else (np.array([]), np.array([]))
    pts = [(x, y) for x, y in wrists.values()]
    X = np.r_[xs, [p[0] for p in pts]] if len(xs) or pts else np.array([0, img.shape[1]])
    Y = np.r_[ys, [p[1] for p in pts]] if len(ys) or pts else np.array([0, img.shape[0]])
    x0, x1 = int(max(0, X.min() - K_pad)), int(min(img.shape[1], X.max() + K_pad))
    y0, y1 = int(max(0, Y.min() - K_pad)), int(min(img.shape[0], Y.max() + K_pad))
    v = img.copy()
    if mask is not None and mask.any():
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cs, -1, (0, 255, 0), 4)
    for sd, (x, y) in wrists.items():
        cv2.circle(v, (int(x), int(y)), 14, (255, 0, 255), -1)
        cv2.putText(v, "L" if sd == "left" else "R", (int(x) + 16, int(y) + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (255, 0, 255), 4)
    return v[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", default="fused_bfix_recon")
    ap.add_argument("--hands", default="hands_c")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--model", default="qwen3-vl-32b")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    cli = ChatClient(f"http://127.0.0.1:{a.port}/v1", a.model)
    cli.wait_ready()
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        k = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        hands = stereo_hands(run, n, a.hands)
        out = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm, oid = o["name"], o["id"]
            src = run / a.src / nm / "poses"
            if not src.is_dir():
                continue
            V = np.asarray(trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh").vertices)
            V = V[:: max(1, len(V) // 1500)]
            dmin = np.full(n, np.inf)
            for t in range(n):
                p = src / f"{t:06d}.json"
                if not p.exists():
                    continue
                T = load_T(p)
                tree = cKDTree(V @ T[:3, :3].T + T[:3, 3])   # camera frame, like the hand joints
                for hd in hands.values():
                    if np.isfinite(hd["J"][t]).all():
                        dmin[t] = min(dmin[t], tree.query(hd["J"][t])[0].min())
            cand = np.convolve((dmin < CAND_M).astype(float), np.ones(2 * PAD + 1), "same") > 0
            ask = [t for t in np.nonzero(cand)[0] if t % STRIDE == 0]

            def query(t):
                img = cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png"))
                mp = run / obj_dir / "masks" / str(oid) / f"{t:06d}.png"
                m = cv2.imread(str(mp), 0) > 127 if mp.exists() else None
                wr = {}
                for sd, hd in hands.items():
                    J = hd["J"][t, 0]
                    if np.isfinite(J).all() and J[2] > 0.05:
                        wr[sd] = (J[0] / J[2] * k["fx"] + k["cx"], J[1] / J[2] * k["fy"] + k["cy"])
                view = crop_view(img, m, wr)
                msg = [{"role": "user", "content": [image_part(view[:, :, ::-1], long_side=768),
                                                    {"type": "text", "text": PROMPT.format(obj=nm.replace("_", " "))}]}]
                try:
                    r = parse_json_reply(cli.chat(msg, temperature=0.0, max_tokens=60)["content"] or "{}")
                except Exception as e:   # noqa: BLE001 -- one failed call must not kill the episode
                    print(rn, nm, t, "VLM error", e, flush=True)
                    r = {}
                return t, {sd for sd in ("left", "right") if r.get(sd) is True}, view
            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(query, ask))
            ans = {t: s for t, s, _ in res}
            intervals = {}
            for sd in ("left", "right"):
                g = np.zeros(n)
                known = sorted(ans)
                for t in range(n):   # nearest asked frame's answer inside candidate windows, else not grasping
                    if cand[t] and known:
                        u = min(known, key=lambda x: abs(x - t))
                        g[t] = float(abs(u - t) <= STRIDE and sd in ans[u])
                g = median_filter(g, size=5, mode="nearest") > 0.5
                idx = np.nonzero(g)[0]
                for t0, t1 in zip(idx, idx[1:]):
                    if 1 < t1 - t0 <= FILL + 1:
                        g[t0:t1] = True
                iv, s = [], None
                for t in range(n + 1):
                    on = t < n and g[t]
                    if on and s is None:
                        s = t
                    elif not on and s is not None:
                        if t - s >= MIN_RUN:
                            iv.append([s, t - 1])
                        s = None
                intervals[sd] = iv
            out[nm] = {"intervals": intervals, "asked": len(ask),
                       "answers": {str(t): sorted(s) for t, s in sorted(ans.items())}}
            print(rn, nm, "asked", len(ask), "intervals", json.dumps(intervals), flush=True)
            # contact sheet: every 4th asked frame with the answer
            tiles = []
            for t, s, v in res[::2][:48]:
                v = cv2.resize(v, (320, int(320 * v.shape[0] / max(v.shape[1], 1))))[:240]
                v = cv2.copyMakeBorder(v, 0, 240 - v.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(v, f"f{t} {'+'.join(sorted(s)) or 'none'}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)
                tiles.append(v)
            if tiles:
                while len(tiles) % 6:
                    tiles.append(np.zeros_like(tiles[0]))
                sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
                od = Path(os.environ.get("V2D_VIZ", ROOT / "viz")) / "grasp_vlm"
                od.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(od / f"{rn}_{nm}.jpg"), sheet)
        (run / "grasp_vlm.json").write_text(json.dumps(out, indent=1))
        print(rn, "VLM usage", json.dumps(cli.usage), flush=True)


if __name__ == "__main__":
    main()
