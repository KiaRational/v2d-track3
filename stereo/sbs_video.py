"""Side-by-side comparison video: two (or more) MP4s next to each other, each with a title bar (agile image).
Frames are matched by index; shorter videos hold their last frame. Each panel is scaled to --height.
Usage: python sbs_video.py out.mp4 a.mp4:"Title A" b.mp4:"Title B" [--height 540] [--fps 20]
"""
import argparse

import cv2
import imageio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+", help="path:title")
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=float, default=None)
    a = ap.parse_args()
    readers, titles = [], []
    for s in a.inputs:
        p, _, t = s.partition(":")
        readers.append(imageio.get_reader(p))
        titles.append(t or p.rsplit("/", 1)[-1])
    fps = a.fps or readers[0].get_meta_data().get("fps", 20)
    iters = [iter(r) for r in readers]
    last = [None] * len(readers)
    w = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    while True:
        alive = False
        for i, it in enumerate(iters):
            try:
                last[i] = next(it)
                alive = True
            except StopIteration:
                pass
        if not alive or any(x is None for x in last):
            break
        panels = []
        for fr, t in zip(last, titles):
            h, wd = fr.shape[:2]
            p = cv2.resize(fr[..., :3], (int(round(wd * a.height / h)) // 2 * 2, a.height))
            bar = np.full((36, p.shape[1], 3), 30, np.uint8)
            cv2.putText(bar, t, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(np.vstack([bar, p]))
        w.append_data(np.hstack(panels))
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
