"""SAM 3 segmenter plug-in for sam2_driver.py (`--segmenter sam3`), via HF transformers.
Runs INSIDE the `sam3` image (defs/sam3.def). UNTESTED until the gated facebook/sam3
weights exist locally (default /scratch/kghasemz/v2d/weights/sam3, i.e. the output of
`huggingface-cli download facebook/sam3 --local-dir /scratch/kghasemz/v2d/weights/sam3`).

Two roles, same interface as the SAM2+GroundingDINO backend:
  detect(rgb, prompts)  - text-prompted instance masks (Sam3Model, "promptable concept
                          segmentation") -> replaces GroundingDINO boxes + SAM2 image masks;
  propagate(...)        - mask-prompted video tracking (Sam3TrackerVideoModel, SAM2-style
                          API) so the driver's re-prompting keeps working unchanged.
All validation (stereo-depth geometry vs mesh, duplicates, identity, re-prompting,
cleanup) stays in the driver and applies to either segmenter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Sam3Segmenter:
    def __init__(self, weights_dir, frames_dir, n_frames, device="cuda", score_thresh=0.3):
        from transformers import Sam3Model, Sam3Processor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        wd = str(weights_dir)
        if not Path(wd, "config.json").exists():
            raise FileNotFoundError(f"SAM3 weights not found in {wd} (gated: download facebook/sam3 there first)")
        self.device, self.thr = device, score_thresh
        self.model = Sam3Model.from_pretrained(wd, local_files_only=True).to(device).eval()
        self.proc = Sam3Processor.from_pretrained(wd, local_files_only=True)
        self.tmodel = Sam3TrackerVideoModel.from_pretrained(wd, local_files_only=True).to(device, dtype=torch.bfloat16).eval()
        self.tproc = Sam3TrackerVideoProcessor.from_pretrained(wd, local_files_only=True)
        self.frames_dir, self.n = Path(frames_dir), n_frames
        self._video = None

    @torch.inference_mode()
    def detect(self, rgb, prompts):
        """prompts {name: [text, ...]} -> [(name, conf, box{x0,y0,x1,y1}, mask bool HxW)]"""
        img = Image.fromarray(rgb)
        out = []
        for name, texts in prompts.items():
            for text in texts:
                inp = self.proc(images=img, text=text, return_tensors="pt").to(self.device)
                res = self.proc.post_process_instance_segmentation(
                    self.model(**inp), threshold=self.thr, mask_threshold=0.5,
                    target_sizes=inp.get("original_sizes").tolist())[0]
                for m, b, s in zip(res["masks"], res["boxes"], res["scores"]):
                    x0, y0, x1, y1 = [float(v) for v in b]
                    out.append((name, float(s), {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                                m.cpu().numpy().astype(bool)))
        return out

    def video(self):
        if self._video is None:
            self._video = [np.asarray(Image.open(self.frames_dir / f"{t:06d}.png").convert("RGB")) for t in range(self.n)]
        return self._video

    @torch.inference_mode()
    def _track_one(self, pl, first, last):
        """One object in its own session: the HF tracker runs every object at every
        frame and fails for objects without a conditioning output yet, and it only
        consumes an object's inputs at the frame it is first run on - so every
        prompt frame is run explicitly before propagating."""
        sess = self.tproc.init_video_session(video=self.video(), inference_device=self.device,
                                             video_storage_device="cpu", dtype=torch.bfloat16)
        H, W = self.video()[0].shape[:2]
        for t, m in sorted(pl, key=lambda x: x[0]):
            self.tproc.add_inputs_to_inference_session(inference_session=sess, frame_idx=t, obj_ids=1,
                                                       input_masks=m.astype(np.uint8))
            self.tmodel(inference_session=sess, frame_idx=t)
        res = {}
        for reverse in (True, False):
            start = last if reverse else first
            for out in self.tmodel.propagate_in_video_iterator(inference_session=sess, start_frame_idx=start,
                                                               reverse=reverse):
                t = out.frame_idx
                if reverse or t >= first or t not in res:
                    masks = self.tproc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                    res[t] = masks[0, 0].cpu().numpy().astype(bool)
        return res

    def propagate(self, prompts, hand_prompts, first, last):
        """prompts {oid: [(t, mask bool)]}, hand_prompts [(t, oid, mask bool)] ->
        {oid: {t: mask bool}}; backward from the object's last prompt, forward from
        its first (same merge rule as the SAM2 backend). Objects are tracked
        independently (SAM2-style trackers keep per-object memories anyway)."""
        allp = {oid: list(pl) for oid, pl in prompts.items()}
        for t, oid, m in hand_prompts:
            allp.setdefault(oid, []).append((t, m))
        return {oid: self._track_one(pl, min(t for t, _ in pl), max(t for t, _ in pl)) for oid, pl in allp.items()}
