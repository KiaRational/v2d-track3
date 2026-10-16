"""Stage 6a: WiLoR hand detections in both stereo cameras (host env, GPU job).

The mono stereo pair (left = ego_cam_c, right = ego_cam_b) is undistorted to
square-pixel pinhole images (same construction as the colour camera, Stage 1);
undistortion keeps each camera's frame, so the calibrated extrinsics still apply.
Then the toolkit's WiLoR runs on each. Outputs in <run>/stereo/:
  cam_c.mp4, cam_b.mp4, intrinsics.json (pinhole K per camera), wilor_c/, wilor_b/
Usage: python s06a_stereo_wilor.py <run_name> [...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stages"))
from s01_calibration import square_pixel_K  # noqa: E402

from v2d.wilor.docker.run_video_to_hands import run_video_to_hands  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
DATA = Path(os.environ["V2D_DATA"])


def undistort_video(src: str, dst: Path, cam: dict, fps: float) -> np.ndarray:
    K, dist, (w, h) = np.array(cam["K"]), np.array(cam["dist"]), cam["size"]
    Knew = square_pixel_K({"K": K, "dist": dist, "size": (w, h)})
    maps = cv2.initUndistortRectifyMap(K, dist, None, Knew, (w, h), cv2.CV_32FC1)
    if not dst.exists():
        cap = cv2.VideoCapture(src)
        ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "14",
                               "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            ff.stdin.write(cv2.remap(f, *maps, cv2.INTER_CUBIC).tobytes())
        ff.stdin.close()
        ff.wait()
    return Knew


def main():
    calib = json.loads((ROOT / "calib/ego_calib_resolved.json").read_text())
    for name in sys.argv[1:]:
        run = ROOT / "runs" / name
        meta = json.loads((run / "episode.json").read_text())
        out = run / "stereo"
        out.mkdir(exist_ok=True)
        intr = {}
        for cam in ("ego_cam_c", "ego_cam_b"):
            src = DATA / f"{meta['split']}/videos/chunk-000/observation.images.{cam}/episode_{meta['episode']:06d}.mp4"
            short = cam[-1]
            Knew = undistort_video(str(src), out / f"cam_{short}.mp4", calib["cameras"][cam], meta["fps"])
            intr[cam] = {"K": Knew.tolist(), "size": calib["cameras"][cam]["size"]}
            if not (out / f"wilor_{short}").is_dir() or not any((out / f"wilor_{short}").iterdir()):
                run_video_to_hands(video_path=str(out / f"cam_{short}.mp4"), output_dir=str(out / f"wilor_{short}"),
                                   weights_dir=str(ROOT / "weights/wilor"), dev=False)
        (out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
        print(f"{name}: stereo WiLoR done ({len(list((out / 'wilor_c').glob('*.json')))} frames)", flush=True)


if __name__ == "__main__":
    main()
