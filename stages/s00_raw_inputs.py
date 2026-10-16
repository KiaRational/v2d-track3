"""Stage 0: per-episode input overview videos.

Composites the head-mounted colour camera, the stereo pair (stored upright, even
though the rig is mounted rotated 180°) and, for public episodes, the mocap object
trajectories (top-down XY trace + height over time) into one 1920x1080 video.
Output: $V2D_VIZ/00_raw_inputs/<split>_episode_XXXXXX.mp4
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

DATA = Path(os.environ["V2D_DATA"])
OUT = Path(os.environ["V2D_VIZ"]) / "00_raw_inputs"
STREAMS = {"a": "observation.images.ego_cam_a", "b": "observation.images.ego_cam_b",
           "c": "observation.images.ego_cam_c"}
COLORS = [(66, 135, 245), (245, 90, 66), (80, 200, 120), (230, 190, 50), (190, 80, 220)]
W, H, RW = 1920, 1080, 480  # canvas, right-column width


def load_objects(episode: int):
    table = pq.read_table(DATA / f"public/data/chunk-000/episode_{episode:06d}.parquet")
    objs = table.column("observation.objects").to_pylist()
    names = [o["name"] for o in objs[0]]
    pos = np.full((len(objs), len(names), 3), np.nan)
    for t, frame in enumerate(objs):
        for i, o in enumerate(frame):
            if o["visible"]:
                pos[t, i] = o["pose"][:3]
    return names, pos


def traj_panel(names, pos, t, size=RW):
    """Top-down XY (upper) and height-vs-time (lower) with the current frame marked."""
    panel = np.full((size, size, 3), 24, np.uint8)
    xy_h, z_top = int(size * 0.62), int(size * 0.66)
    valid = pos[~np.isnan(pos[..., 0])]
    lo, hi = valid[:, :2].min(0) - 0.05, valid[:, :2].max(0) + 0.05
    s = (min(size, xy_h) - 30) / max(hi - lo)
    to_px = lambda p: (int(15 + (p[0] - lo[0]) * s), int(xy_h - 15 - (p[1] - lo[1]) * s))
    zmin, zmax = np.nanmin(pos[..., 2]) - 0.02, np.nanmax(pos[..., 2]) + 0.02
    T = len(pos)
    to_z = lambda k, z: (int(10 + k / max(T - 1, 1) * (size - 20)),
                         int(size - 12 - (z - zmin) / (zmax - zmin) * (size - z_top - 24)))
    for i, name in enumerate(names):
        c = COLORS[i % len(COLORS)]
        pts = [to_px(p) for p in pos[:, i] if not np.isnan(p[0])]
        for a, b in zip(pts, pts[1:]):
            cv2.line(panel, a, b, tuple(int(x * 0.5) for x in c), 1, cv2.LINE_AA)
        zs = [(k, z) for k, z in enumerate(pos[:, i, 2]) if not np.isnan(z)]
        for (k0, z0), (k1, z1) in zip(zs, zs[1:]):
            cv2.line(panel, to_z(k0, z0), to_z(k1, z1), c, 1, cv2.LINE_AA)
        if not np.isnan(pos[t, i, 0]):
            cv2.circle(panel, to_px(pos[t, i]), 6, c, -1, cv2.LINE_AA)
            cv2.circle(panel, to_z(t, pos[t, i, 2]), 4, c, -1, cv2.LINE_AA)
        cv2.putText(panel, name, (10, 18 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    cv2.line(panel, (to_z(t, zmin)[0], z_top), (to_z(t, zmin)[0], size - 8), (200, 200, 200), 1)
    cv2.putText(panel, "top-down XY (mocap)", (size - 170, xy_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,) * 3, 1)
    cv2.putText(panel, "height z(t)", (10, z_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,) * 3, 1)
    return panel


def render(split: str, episode: int, task: str, fps: float):
    OUT.mkdir(parents=True, exist_ok=True)
    caps = {k: cv2.VideoCapture(str(DATA / f"{split}/videos/chunk-000/{v}/episode_{episode:06d}.mp4"))
            for k, v in STREAMS.items()}
    n = int(caps["a"].get(cv2.CAP_PROP_FRAME_COUNT))
    names, pos = load_objects(episode) if split == "public" else ([], None)
    out_path = OUT / f"{split}_episode_{episode:06d}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(n):
        frames = {}
        for k, cap in caps.items():
            ok, f = cap.read()
            if not ok:
                break
            frames[k] = f  # stored videos are already upright despite rotation_deg=180
        if len(frames) < 3:
            break
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:, : W - RW] = cv2.resize(frames["a"], (W - RW, H), interpolation=cv2.INTER_AREA)
        # ego_cam_c is the left eye, ego_cam_b the right (verified from disparity sign).
        canvas[0:300, W - RW:] = cv2.resize(frames["c"], (RW, 300), interpolation=cv2.INTER_AREA)
        canvas[300:600, W - RW:] = cv2.resize(frames["b"], (RW, 300), interpolation=cv2.INTER_AREA)
        if pos is not None:
            canvas[600:1080, W - RW:] = traj_panel(names, pos, t)
        else:
            cv2.putText(canvas, "evaluation split: no GT", (W - RW + 20, 840),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160,) * 3, 2)
        label = f"{split} ep {episode}  |  {task}  |  frame {t}/{n - 1}  t={t / fps:5.2f}s"
        cv2.rectangle(canvas, (0, 0), (W - RW, 34), (0, 0, 0), -1)
        cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,) * 3, 2, cv2.LINE_AA)
        for y, txt in ((20, "stereo L (ego_cam_c)"), (320, "stereo R (ego_cam_b)")):
            cv2.putText(canvas, txt, (W - RW + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        writer.write(canvas)
    writer.release()
    return f"{out_path.name}: {t + 1} frames, objects={names}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args()
    info = json.loads((DATA / "public/meta/info.json").read_text())
    tasks = {e["episode_index"]: e["task"] for e in
             map(json.loads, (DATA / "public/meta/episodes_metadata.jsonl").read_text().splitlines())}
    jobs = [("public", e, tasks.get(e, "?")) for e in info["episode_indices"]]
    eval_eps = sorted(int(p.stem.split("_")[1]) for p in
                      (DATA / "evaluation/videos/chunk-000" / STREAMS["a"]).glob("*.mp4"))
    jobs += [("evaluation", e, "eval (task hidden)") for e in eval_eps]
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(render, *zip(*jobs), [info["fps"]] * len(jobs)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
