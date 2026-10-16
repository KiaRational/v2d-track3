"""Mesh for an object with no provided scan (wooden_spoon) from the episode videos: SAM 3 finds it, SAM 3D Objects
reconstructs it, our stereo depth fixes the metric scale.

  find   (sam3 image)       text-prompted SAM 3 on every --step-th colour frame of the episodes -> candidate masks
                            (confident, not cut by the image border, not tiny), ranked; writes masks + a contact
                            sheet of the best candidates (crop with mask outline) to pick a clean, unoccluded view.
  build  (sam3d_objects)    SAM 3D Objects on one (episode, frame, mask) with a few seeds -> mesh. Metric scale:
                            SAM 3D lays the object out in its own (MoGe) point map; the ratio of our stereo depth to
                            that point map over the mask rescales it. Writes <out>/<name>_seed<k>.obj (metric,
                            centred, object frame), a reprojection overlay per seed, and report.json (extent cm).
Usage:
  python spoon_mesh.py find  --runs evaluation_ep000003 ... [--text "wooden spoon"] --out DIR
  python spoon_mesh.py build --run evaluation_ep000003 --frame 120 --cand DIR/cands.npz --idx 0 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))


def log(*a):
    print(*a, flush=True)


def find(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from seg_sam3 import Sam3Segmenter
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seg = None
    cands = []
    for rn in a.runs:
        frames = sorted((ROOT / "runs" / rn / "recon_hawor/frames").glob("*.png"))
        if not frames:
            log(f"{rn}: no colour frames, skipped")
            continue
        if seg is None:
            seg = Sam3Segmenter(ROOT / "weights/sam3", frames[0].parent, len(frames), score_thresh=0.3)
        for f in frames[:: a.step]:
            rgb = np.asarray(Image.open(f).convert("RGB"))
            H, W = rgb.shape[:2]
            for _, conf, box, m in seg.detect(rgb, {"obj": a.text}):
                area = int(m.sum())
                border = box["x0"] < 4 or box["y0"] < 4 or box["x1"] > W - 5 or box["y1"] > H - 5
                if area < a.min_area or border:
                    continue
                cands.append({"run": rn, "frame": int(f.stem), "conf": conf, "area": area, "box": box, "mask": m})
            log(f"{rn} frame {f.stem}: {len(cands)} candidates so far")
    cands.sort(key=lambda c: -c["conf"] * np.sqrt(c["area"]))
    cands = cands[: a.keep]
    np.savez_compressed(out / "cands.npz", masks=np.stack([c["mask"] for c in cands]) if cands else np.zeros((0, 1, 1), bool))
    meta = [{k: v for k, v in c.items() if k != "mask"} for c in cands]
    (out / "cands.json").write_text(json.dumps(meta, indent=1))
    tiles = []
    for i, c in enumerate(cands):
        rgb = np.asarray(Image.open(ROOT / "runs" / c["run"] / "recon_hawor/frames" / f"{c['frame']:06d}.png").convert("RGB")).copy()
        m = c["mask"]
        edge = m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)
        rgb[edge] = (255, 0, 0)
        b = c["box"]
        pad = 60
        x0, y0 = max(0, int(b["x0"]) - pad), max(0, int(b["y0"]) - pad)
        x1, y1 = min(rgb.shape[1], int(b["x1"]) + pad), min(rgb.shape[0], int(b["y1"]) + pad)
        t = Image.fromarray(rgb[y0:y1, x0:x1])
        t.thumbnail((320, 320))
        tile = Image.new("RGB", (320, 340), "white")
        tile.paste(t, (0, 20))
        ImageDraw.Draw(tile).text((4, 4), f"#{i} {c['run'][-2:]} f{c['frame']} c{c['conf']:.2f} a{c['area']}", fill="black")
        tiles.append(tile)
    if tiles:
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 320, rows * 340), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 320, (i // cols) * 340))
        sheet.save(out / "contact_sheet.jpg", quality=85)
    log(f"wrote {len(cands)} candidates -> {out}")


def build(a):
    import torch
    import trimesh
    S = ROOT / "third_party/sam-3d-objects"
    sys.path[:0] = [str(S), str(S / "notebook")]  # repo not pip-installed in the image
    os.chdir(S)
    from inference import Inference
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = ROOT / "runs" / a.run
    meta = json.loads((Path(a.cand).with_suffix(".json")).read_text())[a.idx]
    mask = np.load(a.cand)["masks"][a.idx]
    assert meta["run"] == a.run and meta["frame"] == a.frame, (meta, a.run, a.frame)
    rgb = np.asarray(Image.open(run / "recon_hawor/frames" / f"{a.frame:06d}.png").convert("RGB"))
    H, W = rgb.shape[:2]
    ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
    d_png = np.asarray(Image.open(run / "depth_stereo" / f"{a.frame:06d}.png")).astype(np.float64)
    depth = 65535.0 / np.maximum(d_png, 1) - 1.0  # toolkit DepthImage encoding; 65535 -> 0 m = invalid
    depth[d_png >= 65535] = 0
    if depth.shape != (H, W):
        depth = np.asarray(Image.fromarray(depth.astype(np.float32)).resize((W, H), Image.NEAREST))
    inf = Inference(str(S / "checkpoints/hf/pipeline.yaml"), compile=False)
    report = {"run": a.run, "frame": a.frame, "cand": meta, "seeds": {}}
    for seed in a.seeds:
        o = inf(rgb, mask, seed=seed)
        mesh = o["glb"]
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        # SAM 3D's own point map (pytorch3d camera: x left, y up, z forward), at its working resolution
        pm = o["pointmap"].numpy()
        mh, mw = pm.shape[:2]
        m_s = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((mw, mh), Image.NEAREST)) > 0
        d_s = np.asarray(Image.fromarray(depth.astype(np.float32)).resize((mw, mh), Image.NEAREST))
        ok = m_s & (d_s > 0.1) & np.isfinite(pm[..., 2]) & (pm[..., 2] > 0)
        ratio = float(np.median(d_s[ok] / pm[..., 2][ok])) if ok.sum() > 50 else float("nan")
        scale = np.asarray(o["scale"].detach().cpu()).reshape(-1)[:3]
        v = np.asarray(mesh.vertices) * scale[None] * ratio  # object frame, metric (SAM3D scale x stereo/MoGe)
        v = v - (v.max(0) + v.min(0)) / 2
        mm = trimesh.Trimesh(v, mesh.faces, vertex_colors=getattr(mesh.visual, "vertex_colors", None), process=False)
        p = out / f"{a.name}_seed{seed}.obj"
        mm.export(p)
        ext = np.sort(v.max(0) - v.min(0))[::-1] * 100
        report["seeds"][seed] = {"stereo_over_sam3d_depth": ratio, "valid_px": int(ok.sum()),
                                 "sam3d_scale": scale.tolist(), "extent_cm": ext.round(1).tolist(),
                                 "verts": len(v), "faces": len(mm.faces), "file": str(p)}
        log(f"seed {seed}: ratio {ratio:.3f}, extent {ext.round(1).tolist()} cm -> {p}")
        # overlay: mask outline + stereo points in the mask, for the record
        vis = rgb.copy()
        edge = mask ^ np.roll(mask, 1, 0) | mask ^ np.roll(mask, 1, 1)
        vis[edge] = (255, 0, 0)
        Image.fromarray(vis).save(out / f"{a.name}_input_f{a.frame}.jpg", quality=85)
        torch.cuda.empty_cache()
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("--runs", nargs="+", required=True)
    f.add_argument("--text", nargs="+", default=["wooden spoon"])
    f.add_argument("--step", type=int, default=10)
    f.add_argument("--min_area", type=int, default=1500)
    f.add_argument("--keep", type=int, default=36)
    f.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", required=True)
    b.add_argument("--frame", type=int, required=True)
    b.add_argument("--cand", required=True)
    b.add_argument("--idx", type=int, required=True)
    b.add_argument("--name", default="wooden_spoon")
    b.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    find(a) if a.cmd == "find" else build(a)


if __name__ == "__main__":
    main()
