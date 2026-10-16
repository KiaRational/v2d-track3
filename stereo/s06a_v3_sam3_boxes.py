"""Stage 6a v3: WiLoR prompted with the Stage-6d SAM 3 hand boxes (host env, GPU job).

Stage 6a runs WiLoR's own detector on whole frames, so every detection carries WiLoR's
own left/right decision -- which flips at the image border and happily fires on people
in the background. Stage 6a-v2 works around that by re-inferring every box with the
opposite label and letting Stage 6b guess. Here the boxes and their identity come from
Stage 6d instead: SAM 3 instance masks, gated to the wearer by metric distance and
assigned to two persistent tracks. WiLoR is then run in bbox mode
(`run_video_to_hands(..., bboxes_dir=...)`, the same path Stage 6a-v2 uses), with
``is_right`` forced to the SAM 3 track, so every output record is already labelled:

    rec["is_right"] == True  <=> SAM 3 track "right"

Stage 6b-v3 reads these directories and treats the label as a hard constraint (no
handedness voting, no flipped twins). Nothing Stage 6a / 6a-v2 wrote is touched.

Boxes are the SAM 3 mask bounding box, expanded by --pad (fraction of the box size) and
clipped to the image: WiLoR expects a bit of context around the hand, and SAM 3 masks
are tight.

Outputs (same per-frame JSON schema as Stage 6a):
    <run>/stereo/wilor_a_sam3/NNNNNN.json     colour camera  (<run>/input_undistorted.mp4)
    <run>/stereo/wilor_c_sam3/NNNNNN.json     stereo left    (<run>/stereo/cam_c.mp4)
    <run>/stereo/wilor_b_sam3/NNNNNN.json     stereo right   (<run>/stereo/cam_b.mp4)
    <run>/stereo/wilor_<v>_sam3_boxes/        the requests that were sent (for debugging)

Usage: python s06a_v3_sam3_boxes.py <run_name> [...] [--views acb] [--pad 0.12]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from v2d.wilor.docker.run_video_to_hands import run_video_to_hands  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
WEIGHTS = str(ROOT / "weights/wilor")
TRACK_IS_RIGHT = {"left": False, "right": True}


def video_for(run: Path, view: str) -> Path:
    return run / "input_undistorted.mp4" if view == "a" else run / f"stereo/cam_{view}.mp4"


def write_requests(run: Path, view: str, req: Path, pad: float, min_px: int, sources) -> int:
    """SAM 3 detections of one view -> WiLoR bbox request files. Returns the box count."""
    det_dir = run / "stereo/hands_sam3" / view
    size = None
    if view == "a":
        ia = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        size = (ia["width"], ia["height"])
    else:
        sint = json.loads((run / "stereo/intrinsics.json").read_text())
        size = tuple(sint[f"ego_cam_{view}"]["size"])
    n = 0
    for f in sorted(det_dir.glob("*.json")):
        out = []
        for r in json.loads(f.read_text()):
            if r.get("source") not in sources:
                continue
            x0, y0, x1, y1 = r["bbox"]
            w, h = x1 - x0, y1 - y0
            if w < min_px or h < min_px:
                continue
            # WiLoR wants context around the hand; SAM 3 masks are tight
            x0, x1 = x0 - pad * w, x1 + pad * w
            y0, y1 = y0 - pad * h, y1 + pad * h
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(size[0] - 1), x1), min(float(size[1] - 1), y1)
            if x1 - x0 < min_px or y1 - y0 < min_px:
                continue
            out.append({"bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "is_right": TRACK_IS_RIGHT[r["track"]], "track": r["track"]})
        if out:
            (req / f.name).write_text(json.dumps(out))
            n += len(out)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--views", default="acb")
    ap.add_argument("--pad", type=float, default=0.12, help="box expansion, fraction of box size")
    ap.add_argument("--min_px", type=int, default=24)
    ap.add_argument("--sources", default="det,fill",
                    help="which Stage-6d records to prompt with (det = SAM3 detection, fill = tracker gap fill)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    sources = set(a.sources.split(","))
    for name in a.runs:
        run = ROOT / "runs" / name
        if not (run / "stereo/hands_sam3/summary.json").exists():
            print(f"{name}: no stereo/hands_sam3 (run s06d_sam3_hands.py first); skipping", flush=True)
            continue
        for v in a.views:
            out_dir = run / f"stereo/wilor_{v}_sam3"
            if out_dir.is_dir() and any(out_dir.iterdir()):
                if not a.overwrite:
                    print(f"{name}/{v}: wilor_{v}_sam3 exists; skipping", flush=True)
                    continue
                shutil.rmtree(out_dir)
            req = run / f"stereo/wilor_{v}_sam3_boxes"
            if req.is_dir():
                shutil.rmtree(req)
            req.mkdir(parents=True)
            nb = write_requests(run, v, req, a.pad, a.min_px, sources)
            if not nb:
                print(f"{name}/{v}: no SAM 3 boxes", flush=True)
                continue
            run_video_to_hands(video_path=str(video_for(run, v)), output_dir=str(out_dir),
                               weights_dir=WEIGHTS, bboxes_dir=str(req), dev=False)
            got = len(list(out_dir.glob("*.json")))
            print(f"{name}/{v}: {nb} SAM 3 boxes -> WiLoR on {got} frames", flush=True)


if __name__ == "__main__":
    main()
