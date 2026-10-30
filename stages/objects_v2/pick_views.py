"""Pick clean views of an object in the EVALUATION videos for single-view mesh reconstruction (no public scans used).

Uses our own per-episode SAM 3 masks from the SIZE-FREE segmentation (<obj_dir>/masks/<id>, s05_objects_v2.py
--masks_only: no mesh of any kind is read, so the meshes built from these views do not depend on the provided scans);
ids from <obj_dir>/objects.json. Distinct text prompts separate blue vs beige cup and pot vs lid. A view qualifies if the object mask is large,
does not touch the image border, and does not touch either hand mask or any other object mask (dilated by 15 px): an unoccluded, free-standing object.
Views are ranked by mask area, one per episode first (diversity), then by area. Writes the same cands.npz /
cands.json format spoon_mesh.py build reads, plus a contact sheet.
Usage: python pick_views.py --name blue_cup --runs evaluation_ep000010 ... --out DIR [--step 3] [--keep 6]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

ROOT = Path("/scratch/kghasemz/v2d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--keep", type=int, default=6)
    ap.add_argument("--min_area", type=int, default=3000)
    ap.add_argument("--obj_dir", default="objects_free")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cands = []
    for rn in a.runs:
        run = ROOT / "runs" / rn
        mid = {o["name"]: o["id"] for o in json.loads((run / a.obj_dir / "objects.json").read_text())}
        if mid.get(a.name) is None:
            continue
        md = run / a.obj_dir / "masks" / str(mid[a.name])
        for p in sorted(md.glob("*.png"))[:: a.step]:
            m = np.asarray(Image.open(p)) > 0
            area = int(m.sum())
            if area < a.min_area:
                continue
            ys, xs = np.nonzero(m)
            H, W = m.shape
            if xs.min() < 5 or ys.min() < 5 or xs.max() > W - 6 or ys.max() > H - 6:
                continue
            hands = np.zeros_like(m)  # hands AND every other object: an occluded / touching view bakes them into the mesh
            others = [v for k, v in mid.items() if v is not None and k != a.name]
            for hid in set([2, 3] + list(others)):
                hp = run / a.obj_dir / "masks" / str(hid) / p.name
                if hp.exists():
                    hands |= np.asarray(Image.open(hp)) > 0
            if (binary_dilation(hands, iterations=15) & m).any():
                continue
            cands.append({"run": rn, "frame": int(p.stem), "area": area,
                          "box": {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()), "y1": int(ys.max())},
                          "conf": 1.0, "mask": m})
    # one best view per episode first, then the rest by area
    cands.sort(key=lambda c: -c["area"])
    first, rest, seen = [], [], set()
    for c in cands:
        (rest if c["run"] in seen else first).append(c); seen.add(c["run"])
    cands = (first + rest)[: a.keep]
    np.savez_compressed(out / "cands.npz", masks=np.stack([c["mask"] for c in cands]) if cands else np.zeros((0, 1, 1), bool))
    (out / "cands.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "mask"} for c in cands], indent=1))
    tiles = []
    for i, c in enumerate(cands):
        rgb = np.asarray(Image.open(ROOT / "runs" / c["run"] / "recon_hawor/frames" / f"{c['frame']:06d}.png").convert("RGB")).copy()
        e = c["mask"] ^ np.roll(c["mask"], 1, 0) | c["mask"] ^ np.roll(c["mask"], 1, 1)
        rgb[e] = (255, 0, 0)
        b = c["box"]
        t = Image.fromarray(rgb[max(0, b["y0"] - 60): b["y1"] + 60, max(0, b["x0"] - 60): b["x1"] + 60]); t.thumbnail((320, 320))
        tile = Image.new("RGB", (320, 340), "white"); tile.paste(t, (0, 20))
        ImageDraw.Draw(tile).text((4, 4), f"#{i} {c['run'][-2:]} f{c['frame']} a{c['area']}", fill="black")
        tiles.append(tile)
    if tiles:
        sheet = Image.new("RGB", (min(6, len(tiles)) * 320, ((len(tiles) + 5) // 6) * 340), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 6) * 320, (i // 6) * 340))
        sheet.save(out / "contact_sheet.jpg", quality=85)
    print(f"{a.name}: {len(cands)} views " + ", ".join(f"{c['run'][-2:]}:f{c['frame']}({c['area']})" for c in cands), flush=True)


if __name__ == "__main__":
    main()
