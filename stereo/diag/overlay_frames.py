"""Draw WiLoR boxes+2D joints (red=left, blue=right label) on a/c/b frames. args: run k frames..."""
import json, os, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(os.environ["V2D_ROOT"]); run = ROOT / "runs" / sys.argv[1]; k = int(sys.argv[2])
suffix = os.environ.get("SUFFIX", "")
W = json.loads((run / f"stereo/wilor2d{suffix}.json").read_text())
if "a" not in W: W["a"] = json.loads((run / "stereo/wilor2d.json").read_text())["a"]
out = ROOT / "tmp_frames"; out.mkdir(exist_ok=True)
caps = {v: cv2.VideoCapture(str(run / f"stereo/cam_{v}.mp4")) for v in "cb"}
def grab(v, i):
    caps[v].set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = caps[v].read(); return f
for t in map(int, sys.argv[3:]):
    tiles = []
    for v, img in (("a", cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png"))), ("c", grab("c", t + k)), ("b", grab("b", t + k))):
        tt = t if v == "a" else t + k
        for d in W[v][tt]:
            col = (255, 0, 0) if d["is_right"] else (0, 0, 255)
            x0, y0, x1, y1 = map(int, d["bbox"]); cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            for p in np.array(d["j2d"]).astype(int): cv2.circle(img, tuple(p), 3, col, -1)
            cv2.putText(img, f"{'R' if d['is_right'] else 'L'} {d['score']:.2f}", (x0, y0 - 5), 0, 0.8, col, 2)
        if v == "a": img = cv2.resize(img, (1280, 960))
        cv2.putText(img, f"{v} frame {tt}", (10, 40), 0, 1.2, (0, 255, 255), 2)
        tiles.append(img)
    h = max(x.shape[0] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
    cv2.imwrite(str(out / f"wilor{suffix}_t{t:03d}.jpg"), cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))
print("done")
