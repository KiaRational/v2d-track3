"""Stage 5: track *every* object in an episode (the toolkit pipeline handles one).

Per object: GroundingDINO on sampled frames until it is found (objects are not always
visible at frame 0; look-alikes such as pot vs. lid may not claim the same box),
then ONE joint SAM2 propagation over all objects + both hands (so masks respect
mutual occlusion), then FoundationPose + EKF smoothing per object with its mesh on
metric stereo depth (falls back to the toolkit's MoGe depth if Stage 3 hasn't run).

Reuses the toolkit's hand SAM2 prompts (ids 2/3) from recon_hawor/. Object ids are
1, 4, 5, ... Outputs in <run>/objects/, viz in $V2D_VIZ/05_objects/<run>.mp4.
Usage (GPU job, host env): python s05_objects.py <run_name> [--objects a,b]
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks

ROOT = Path(os.environ["V2D_ROOT"])
W = ROOT / "weights"
VIZ = Path(os.environ["V2D_VIZ"]) / "05_objects"
OBJECT_IDS = [1, 4, 5, 6, 7]  # 2 and 3 are the hands
COLORS = [(66, 135, 245), (245, 90, 66), (80, 200, 120), (230, 190, 50), (190, 80, 220)]


def iou(a, b):
    ix = max(0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    iy = max(0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    inter = ix * iy
    area = lambda r: (r["x1"] - r["x0"]) * (r["y1"] - r["y0"])
    return inter / (area(a) + area(b) - inter + 1e-9)


def detect(frames_dir: Path, work: Path, name: str, prompt: str, n_frames: int, taken: list, stride=10):
    """First sampled frame where GroundingDINO finds the object with a box not
    already claimed by another object (IoU < 0.5)."""
    for idx in list(range(0, n_frames, stride)):
        out = work / f"dino_{name}_{idx:06d}.json"
        if not out.exists():
            run_image_to_object_bboxes(image_path=str(frames_dir / f"{idx:06d}.png"), output_path=str(out),
                                       prompt=prompt, model_dir=str(W / "grounding_dino"), dev=False)
        dets = sorted(json.loads(out.read_text()), key=lambda d: -d["confidence"])
        for d in dets:
            if all(t["frame"] != idx or iou(d["box"], t["box"]) < 0.5 for t in taken):
                return {"frame": idx, "box": d["box"], "confidence": d["confidence"]}
    return None


def quat_wxyz_to_R(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def render_viz(run: Path, objs: list, n_frames: int, K: np.ndarray, fps: float):
    VIZ.mkdir(parents=True, exist_ok=True)
    frames_dir = run / "recon_hawor/frames"
    out = VIZ / f"{run.name}.mp4"
    vw, vh = 1014, 760
    s = vw / 2028
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    boxes = {}
    for o in objs:
        mesh = trimesh.load(o["mesh"], force="mesh")
        lo, hi = mesh.bounds
        boxes[o["id"]] = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for t in range(n_frames):
        img = cv2.resize(cv2.imread(str(frames_dir / f"{t:06d}.png")), (vw, vh))
        for k, o in enumerate(objs):
            c = COLORS[k % len(COLORS)]
            m = cv2.imread(str(run / f"objects/masks/{o['id']}/{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                cs, _ = cv2.findContours(cv2.resize((m > 0).astype(np.uint8), (vw, vh)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, c, 2)
            pf = run / f"objects/{o['name']}/poses_smoothed/{t:06d}.json"
            if pf.exists():
                p = json.loads(pf.read_text())
                R, tr = quat_wxyz_to_R(p["rotation"]), np.array(p["translation"])
                pts = boxes[o["id"]] @ R.T + tr
                if (pts[:, 2] > 0.05).all():
                    uv = (pts[:, :2] / pts[:, 2:3] @ K[:2, :2].T + K[:2, 2]) * s
                    for a, b in edges:
                        cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), c, 2, cv2.LINE_AA)
            cv2.putText(img, f"{o['name']} (id {o['id']})", (10, 26 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(img, f"frame {t}", (vw - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out), "-c:v", "libx264", "-crf", "22",
                    "-pix_fmt", "yuv420p", str(out.with_suffix(".h264.mp4"))], check=True)
    out.with_suffix(".h264.mp4").replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--objects", default=None, help="comma-separated object names (default: episode.json)")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    meta = json.loads((run / "episode.json").read_text())
    names = args.objects.split(",") if args.objects else meta["objects"]
    recon = run / "recon_hawor"
    frames_dir = recon / "frames"
    n = len(list(frames_dir.glob("*.png")))
    work = run / "objects"
    work.mkdir(exist_ok=True)

    # 1) detect every object
    objs, taken = [], []
    for k, name in enumerate(names):
        prompt = "a " + name.replace("_", " ")
        det = detect(frames_dir, work, name, prompt, n, taken)
        if det is None:
            print(f"WARNING: {name} never detected; skipping", flush=True)
            continue
        taken.append(det)
        objs.append({"id": OBJECT_IDS[k], "name": name, "prompt": prompt, "ref_frame": det["frame"],
                     "box": det["box"], "dino_confidence": det["confidence"],
                     "mesh": str(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj")})
        print(f"{name}: id {OBJECT_IDS[k]}, first seen frame {det['frame']} (conf {det['confidence']:.2f})", flush=True)

    # 2) one joint SAM2 pass: all objects + the toolkit's hand mask prompts
    hand_prompts = [p for p in json.loads((recon / "sam2_prompts.json").read_text())["prompts"]
                    if p["object_id"] in (2, 3)]
    # Mask paths are resolved relative to the prompts file inside the container
    # (only its directory is mounted), so copy the hand prompt masks next to it.
    for p in hand_prompts:
        if p.get("mask_path"):
            dst = work / p["mask_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes((recon / p["mask_path"]).read_bytes())
    prompts = [{"frame_index": o["ref_frame"], "object_id": o["id"], "points": None, "point_labels": None,
                "box": o["box"], "mask_path": None} for o in objs] + hand_prompts
    (work / "sam2_prompts.json").write_text(json.dumps({"prompts": prompts}, indent=2))
    masks_dir = work / "masks"
    if not all((masks_dir / str(o["id"])).is_dir() for o in objs):
        run_video_to_masks(video_path=str(run / "input_undistorted.mp4"), prompts_path=str(work / "sam2_prompts.json"),
                           masks_dir=str(masks_dir), weights_dir=str(W / "sam2"), dev=False)

    # 3) FoundationPose + EKF per object on metric stereo depth
    depth_dir = run / "depth_stereo" if (run / "depth_stereo").is_dir() else recon / "depth"
    intr = recon / "intrinsics_stable.json"
    for o in objs:
        od = work / o["name"]
        if not (od / "poses").is_dir() or len(list((od / "poses").glob("*.json"))) < n - o["ref_frame"]:
            run_video_to_poses(video_path=str(run / "input_undistorted.mp4"), depth_folder=str(depth_dir),
                               masks_folder=str(masks_dir / str(o["id"])), camera_intrinsics_path=str(intr),
                               mesh_path=o["mesh"], poses_dir=str(od / "poses"), weights_dir=str(W / "foundation_pose"),
                               reference_frame=o["ref_frame"], mask_depth=True, reregister_iou_thresh=0.3,
                               target_width=1014, target_height=760, dev=False)  # full res OOMs on large objects
        if not (od / "poses_smoothed").is_dir():
            run_ekf_smoothing(poses_dir=str(od / "poses"), mesh_path=o["mesh"], intrinsics_path=str(intr),
                              weights_dir=str(W / "foundation_pose"), output_dir=str(od / "poses_smoothed"),
                              masks_folder=str(masks_dir / str(o["id"])), process_noise_xy=0.01, process_noise_z=0.01,
                              process_noise_r=0.02, measurement_noise_xy=0.01, measurement_noise_z=0.04,
                              measurement_noise_r=0.02, dev=False)
        o["poses"] = str(od / "poses_smoothed")
        o["depth_source"] = depth_dir.name

    (work / "objects.json").write_text(json.dumps(objs, indent=2))
    K = np.array(json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())["undistorted_ego_cam_a_K"])
    render_viz(run, objs, n, K, meta["fps"])
    print(f"done: {[o['name'] for o in objs]}", flush=True)


if __name__ == "__main__":
    main()
