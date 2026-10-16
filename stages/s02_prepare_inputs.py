"""Stage 2: per-episode inputs for the toolkit's ego reconstruction pipeline.

For each episode run dir ($V2D_ROOT/runs/<split>_ep<idx>/):
  input_undistorted.mp4    ego_cam_a undistorted with the verified calibration (Stage 1)
  intrinsics_stable.json   exact pinhole K -> the pipeline skips MoGe intrinsics estimation
  episode.json             task, tracked objects, chosen manipulated object, prompt, mesh
Object meshes (.glb) are converted once to OBJ under $V2D_ROOT/meshes_obj/.
Viz: $V2D_VIZ/02_undistorted/<split>_ep<idx>.mp4 (raw | undistorted).
"""
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import trimesh

ROOT = Path(os.environ["V2D_ROOT"])
DATA = Path(os.environ["V2D_DATA"])
VIZ = Path(os.environ["V2D_VIZ"]) / "02_undistorted"
CALIB = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
CAM_A = "observation.images.ego_cam_a"


def convert_mesh(name: str) -> Path:
    out = ROOT / "meshes_obj" / name / f"{name}.obj"
    if not out.exists():
        src_dir = DATA / "public/mesh" / name
        src = src_dir / f"{name}.glb"
        if not src.exists():
            src = src_dir / f"{name}_visual.glb"
        mesh = trimesh.load(src, force="mesh")
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out)
    return out


def objects_and_motion(episode: int):
    table = pq.read_table(DATA / f"public/data/chunk-000/episode_{episode:06d}.parquet")
    frames = table.column("observation.objects").to_pylist()
    names = [o["name"] for o in frames[0]]
    path_len = {}
    for i, n in enumerate(names):
        pos = np.array([f[i]["pose"][:3] for f in frames if f[i]["visible"]])
        path_len[n] = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
    return names, path_len


def ffmpeg_writer(path: Path, w: int, h: int, fps: float, crf: int):
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
         "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)


def prepare(split: str, episode: int, task: str, fps: float):
    run = ROOT / "runs" / f"{split}_ep{episode:06d}"
    run.mkdir(parents=True, exist_ok=True)
    cam = CALIB["cameras"]["ego_cam_a"]
    K, dist, (w, h) = np.array(cam["K"]), np.array(cam["dist"]), cam["size"]
    Knew = np.array(CALIB["undistorted_ego_cam_a_K"])
    maps = cv2.initUndistortRectifyMap(K, dist, None, Knew, (w, h), cv2.CV_32FC1)

    (run / "intrinsics_stable.json").write_text(json.dumps(
        {"fx": Knew[0, 0], "fy": Knew[1, 1], "cx": Knew[0, 2], "cy": Knew[1, 2], "width": w, "height": h},
        indent=2))

    meta = {"split": split, "episode": episode, "task": task, "fps": fps,
            "source_video": str(DATA / f"{split}/videos/chunk-000/{CAM_A}/episode_{episode:06d}.mp4")}
    if split == "public":
        names, motion = objects_and_motion(episode)
        primary = max(motion, key=motion.get)
        meta.update(objects=names, object_path_length_m=motion, primary_object=primary,
                    object_prompt="a " + primary.replace("_", " "),
                    object_mesh=str(convert_mesh(primary)))
    (run / "episode.json").write_text(json.dumps(meta, indent=2))

    video = run / "input_undistorted.mp4"
    if video.exists():
        return f"{run.name}: exists"
    cap = cv2.VideoCapture(meta["source_video"])
    full = ffmpeg_writer(video, w, h, fps, crf=14)
    vw, vh = 1280, 480
    viz = ffmpeg_writer(VIZ / f"{split}_ep{episode:06d}.mp4", vw, vh, fps, crf=23)
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        und = cv2.remap(f, *maps, cv2.INTER_CUBIC)
        full.stdin.write(und.tobytes())
        side = np.hstack([cv2.resize(f, (vw // 2, vh)), cv2.resize(und, (vw // 2, vh))])
        cv2.putText(side, "raw ego_cam_a", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(side, "undistorted (verified calib)", (vw // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        viz.stdin.write(side.tobytes())
        n += 1
    for p in (full, viz):
        p.stdin.close()
        p.wait()
    return f"{run.name}: {n} frames, primary={meta.get('primary_object')}"


def main():
    VIZ.mkdir(parents=True, exist_ok=True)
    info = json.loads((DATA / "public/meta/info.json").read_text())
    tasks = {e["episode_index"]: e["task"] for e in
             map(json.loads, (DATA / "public/meta/episodes_metadata.jsonl").read_text().splitlines())}
    jobs = [("public", e, tasks.get(e, "?")) for e in info["episode_indices"]]
    eval_eps = sorted(int(p.stem.split("_")[1]) for p in (DATA / "evaluation/videos/chunk-000" / CAM_A).glob("*.mp4"))
    jobs += [("evaluation", e, "hidden") for e in eval_eps]
    # Convert meshes up front so parallel workers don't race on the same file.
    for name in sorted({p.name for p in (DATA / "public/mesh").iterdir()}):
        convert_mesh(name)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) // 2
    with ProcessPoolExecutor(max(workers, 1)) as ex:
        for msg in ex.map(prepare, *zip(*jobs), [info["fps"]] * len(jobs)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
