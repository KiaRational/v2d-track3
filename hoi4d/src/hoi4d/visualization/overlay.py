"""Frame overlays: labelled SAM 3 instance outlines (VLM entity mapping, README 2.6/3) and the current
reconstruction projected into the colour camera (object surface samples, hand skeletons) for the reviewer."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .. import io as hio
from ..geometry.assets import asset, inv, transform

COLOURS = [(255, 80, 80), (80, 200, 255), (255, 210, 60), (150, 255, 120), (255, 120, 255), (255, 160, 60),
           (120, 140, 255), (60, 255, 220)]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


class Overlay:
    def __init__(self, run_name, man, tr):
        self.run = hio.ROOT / "runs" / run_name
        self.man = man
        self.tr = tr
        self.K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
        self.ents = [e for e in man["entities"] if e.get("mask_id") is not None]
        self.col = {e["id"]: COLOURS[i % len(COLOURS)] for i, e in enumerate(man["entities"])}

    def frame(self, f: int) -> np.ndarray:
        return np.asarray(Image.open(self.run / "recon_hawor/frames" / f"{f:06d}.png").convert("RGB"))

    def labelled(self, rgb: np.ndarray, f: int) -> np.ndarray:
        """Instance outlines with entity-ID labels (masks from objects_sam3/masks/<id>)."""
        im = rgb.copy()
        labels = []
        for e in self.ents:
            p = self.run / "objects_sam3/masks" / str(e["mask_id"]) / f"{f:06d}.png"
            if not p.exists():
                continue
            m = np.asarray(Image.open(p)) > 0
            if m.sum() < 50:
                continue
            edge = m ^ np.roll(m, 2, 0) | m ^ np.roll(m, 2, 1)
            edge |= np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
            im[edge] = self.col[e["id"]]
            ys, xs = np.nonzero(m)
            labels.append((e["id"], int(xs.min()), max(0, int(ys.min()) - 34)))
        pim = Image.fromarray(im)
        d = ImageDraw.Draw(pim)
        for name, x, y in labels:
            d.rectangle([x, y, x + 14 * len(name) + 8, y + 30], fill=(0, 0, 0))
            d.text((x + 4, y + 6), name, fill=self.col[name], font_size=24) if _pil_sizes() else \
                d.text((x + 4, y + 6), name, fill=self.col[name])
        return np.asarray(pim)

    def project(self, P_world: np.ndarray, f: int):
        Twc = self.tr.extra["T_world_cam"][f]
        Pc = transform(inv(Twc), P_world)
        ok = Pc[:, 2] > 0.05
        uv = Pc[:, :2] / Pc[:, 2:3] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        return uv, ok

    def reconstruction(self, rgb: np.ndarray, f: int, tr=None) -> np.ndarray:
        tr = tr or self.tr
        pim = Image.fromarray(rgb.copy())
        d = ImageDraw.Draw(pim)
        for n, o in tr.objects.items():
            if not o["valid"][f]:
                continue
            uv, ok = self.project(transform(o["T_world_obj"][f], asset(n).surface[::3]), f)
            for u, v in uv[ok]:
                d.point((u, v), fill=self.col.get(n, (255, 255, 255)))
            if ok.any():
                d.text(tuple(uv[ok].min(0)), n, fill=self.col.get(n, (255, 255, 255)))
        for h, hd in tr.hands.items():
            if not hd["valid"][f]:
                continue
            uv, ok = self.project(hd["joints"][f], f)
            for a, b in BONES:
                if ok[a] and ok[b]:
                    d.line([tuple(uv[a]), tuple(uv[b])], fill=self.col.get(h, (255, 255, 255)), width=3)
            if ok[0]:
                d.text(tuple(uv[0] + 8), h, fill=self.col.get(h, (255, 255, 255)))
        return np.asarray(pim)


def _pil_sizes():
    import PIL
    return tuple(int(x) for x in PIL.__version__.split(".")[:2]) >= (10, 1)
