"""Stage 6a v2: extra WiLoR passes on the stereo views (host env, GPU job).

 1. CLAHE pass: Stage 6a's undistorted <run>/stereo/cam_{c,b}.mp4 (run s06a first) are
    contrast-normalised (CLAHE on the grey image, clip 2.5, 8x8 tiles) and WiLoR is run
    again -> cam_{c,b}_clahe.mp4, wilor_{c,b}_clahe/. Recovers hands the detector misses
    in the grey fisheye views.
 2. Flipped-handedness pass: WiLoR's 2D joints depend on its left/right decision (left hands
    are mirrored through the right-hand model). On the fisheye views it often mislabels
    a hand (typically at the image border), and a mislabelled detection's joints cannot be
    triangulated against the correctly-labelled one in the other view. So every detection
    (plain and CLAHE, score >= 0.3) is re-inferred on its own box with the opposite label
    -> wilor_{c,b}_flip/, wilor_{c,b}_clahe_flip/ (same file layout). Stage 6b pairs views
    with consistent labels only and decides handedness by tracking.
Usage: python s06a_v2_stereo_wilor_clahe.py <run_name> [...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from v2d.wilor.docker.run_video_to_hands import run_video_to_hands  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
WEIGHTS = str(ROOT / "weights/wilor")


def clahe_video(src: Path, dst: Path, clip=2.5, tiles=8):
    if dst.exists():
        return
    cap = cv2.VideoCapture(str(src))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "14",
                           "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cl.apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        ff.stdin.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).tobytes())
    ff.stdin.close()
    ff.wait()


def flipped_pass(video: Path, det_dir: Path, out_dir: Path, min_score=0.3):
    """Re-run WiLoR on every detection box of det_dir with the opposite handedness."""
    if out_dir.is_dir() and any(out_dir.iterdir()):
        return
    with tempfile.TemporaryDirectory(dir=str(ROOT / "tmp_frames")) as req:
        for f in sorted(det_dir.glob("*.json")):
            recs = [r for r in json.loads(f.read_text()) if r.get("score", 0) >= min_score]
            if recs:  # frames without a request file are skipped by video_to_hands
                (Path(req) / f.name).write_text(json.dumps(
                    [{"bbox": r["bbox"], "is_right": not r["is_right"]} for r in recs]))
        run_video_to_hands(video_path=str(video), output_dir=str(out_dir), weights_dir=WEIGHTS,
                           bboxes_dir=req, dev=False)


def main():
    for name in sys.argv[1:]:
        out = ROOT / "runs" / name / "stereo"
        for v in "cb":
            clahe_video(out / f"cam_{v}.mp4", out / f"cam_{v}_clahe.mp4")
            d = out / f"wilor_{v}_clahe"
            if not d.is_dir() or not any(d.iterdir()):
                run_video_to_hands(video_path=str(out / f"cam_{v}_clahe.mp4"), output_dir=str(d),
                                   weights_dir=WEIGHTS, dev=False)
            flipped_pass(out / f"cam_{v}.mp4", out / f"wilor_{v}", out / f"wilor_{v}_flip")
            flipped_pass(out / f"cam_{v}_clahe.mp4", out / f"wilor_{v}_clahe", out / f"wilor_{v}_clahe_flip")
        print(f"{name}: CLAHE + flipped-handedness stereo WiLoR done", flush=True)


if __name__ == "__main__":
    main()
