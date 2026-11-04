"""Side-by-side GIFs: human egocentric clip | learned Sharpa policy rollout (first episode of the looping render)."""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/scratch/kghasemz/v2d/report/policyfigs")
import make_policy_strips as M

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
EPS = ["e08", "e05", "e28", "e38", "e33", "e30", "e10", "e20"]
N, FPS, H, W = 60, 10, 240, 384
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
except OSError:
    FONT = ImageFont.load_default()

def tag(a, s, col):
    im = Image.fromarray(a); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 10 * len(s), 26], fill=col); d.text((5, 3), s, fill="white", font=FONT)
    return np.asarray(im)

for e in EPS:
    hv = M.R / f"runs/evaluation_ep0000{e[1:]}/input_undistorted.mp4"
    dh = M.dur(hv); dr = M.episode_len(M.VID[e], dh)
    frames = []
    for i in range(N):
        f = i / (N - 1) * 0.97
        h = cv2.resize(M.grab(hv, f * dh), (W, H))
        r = cv2.resize(M.crop(M.grab(M.VID[e], f * dr)), (W, H))
        frames.append(Image.fromarray(np.hstack([tag(h, "human", (60, 130, 60)), np.full((H, 4, 3), 255, np.uint8),
                                                  tag(r, "robot", (90, 90, 90))])))
    pal = [fr.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for fr in frames]
    pal[0].save(OUT / f"{e}.gif", save_all=True, append_images=pal[1:], duration=1000 // FPS, loop=0, optimize=True)
    print(e, (OUT / f"{e}.gif").stat().st_size // 1024, "KB", flush=True)
