"""Run GroundingDINO on sampled frames with several prompts per object (runs INSIDE
the v2d_grounding_dino container; one model load for the whole episode).

Usage: python gdino_detect.py --frames_dir D --frames 0,5,10 --prompts JSON --model_dir M --out OUT.json
  prompts JSON: {"<object name>": ["prompt a", "prompt b"], ...}
Output: {"<frame>": [{"object": name, "prompt": p, "confidence": c, "label": l,
                      "box": {"x0","y0","x1","y1"}}, ...]}
"""
import argparse
import json

import numpy as np
from PIL import Image

from v2d.grounding_dino.lib.image_list_to_object_bboxes import _detect, _get_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.2)
    a = ap.parse_args()
    prompts = json.load(open(a.prompts))
    model = _get_model(a.model_dir)
    res = {}
    frames = [int(x) for x in a.frames.split(",")]
    for k, f in enumerate(frames):
        img = np.asarray(Image.open(f"{a.frames_dir}/{f:06d}.png").convert("RGB"))
        dets = []
        for name, plist in prompts.items():
            for p in plist:
                cap = p.strip().rstrip(".") + " ."
                for d in _detect(model, img, cap, a.box_threshold, a.text_threshold)[:4]:
                    dets.append({"object": name, "prompt": p, **d})
        res[str(f)] = dets
        if k % 20 == 0:
            print(f"gdino {k}/{len(frames)}", flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
