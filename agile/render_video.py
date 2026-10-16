"""4D visualisation of AGILE results (runs in the `agile` image).

Per frame: [camera view: hands + objects over the video | third-person view of the
same 4D scene from a fixed viewpoint beside the objects]. Writes
$V2D_VIZ/06_agile/<run>_<tag>.mp4 (H.264, native fps).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings,
                                SoftPhongShader, TexturesVertex)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.utils import cameras_from_opencv_projection

from geometry import Mano
from render import Renderer

ROOT = Path(os.environ["V2D_ROOT"])
HAND_RGB = (0.85, 0.72, 0.62)
OBJ_RGB = [(0.25, 0.55, 0.95), (0.95, 0.45, 0.25), (0.35, 0.8, 0.45)]


def colored(verts, faces, rgb, dev):
    v = torch.as_tensor(verts, dtype=torch.float32, device=dev)
    f = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.int64, device=dev)
    return Meshes([v], [f], textures=TexturesVertex([torch.tensor(rgb, device=dev).expand(len(v), 3)]))


def render(scene, K, R_view, T_view, w, h, dev):
    cams = cameras_from_opencv_projection(torch.tensor(R_view, dtype=torch.float32, device=dev)[None],
                                          torch.tensor(T_view, dtype=torch.float32, device=dev)[None],
                                          torch.tensor(K, dtype=torch.float32, device=dev)[None],
                                          torch.tensor([[h, w]], dtype=torch.float32, device=dev))
    lights = PointLights(device=dev, location=[[0.0, -1.0, -1.0]])
    r = MeshRenderer(MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(image_size=(h, w), max_faces_per_bin=250_000)),
                     SoftPhongShader(device=dev, cameras=cams, lights=lights,
                                     blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))))
    img = r(scene)[0]
    return (img[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), img[..., 3].cpu().numpy()


def look_at(eye, target, up=np.array([0.0, -1.0, 0.0])):
    """World(=camera 0)->view rotation/translation for an OpenCV camera at `eye`."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tag", default="agile")
    args = ap.parse_args()
    dev = "cuda"
    ep = ROOT / "runs" / args.run
    res_dir = ep / args.tag
    meta = json.loads((ep / "episode.json").read_text())
    intr = json.loads((ep / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    W, H = 1014, 760
    s = W / intr["width"]
    Ks = K.copy()
    Ks[:2] *= s
    n = len(list((ep / "recon_hawor/frames").glob("*.png")))
    mano = Mano(str(ROOT / "weights/hand"), dev)
    objs = []
    for k, od in enumerate(sorted(p for p in res_dir.iterdir() if (p / "poses").is_dir())):
        prep = json.loads((ep / "agile" / od.name / "prepare.json").read_text())
        objs.append({"name": od.name, "renderer": Renderer(prep["mesh"], dev), "dir": od, "rgb": OBJ_RGB[k % 3]})

    # Third-person viewpoint: 60 deg around the vertical through the objects' mean position.
    centers = []
    for o in objs:
        for t in range(0, n, 10):
            p = o["dir"] / f"poses/{t:06d}.json"
            if p.exists():
                centers.append(json.loads(p.read_text())["T"])
    c = np.mean(centers, 0) if centers else np.array([0, 0, 0.5])
    ang = np.radians(60)
    eye = c + np.array([np.sin(ang), -0.35, -np.cos(ang)]) * np.linalg.norm(c)
    R_v, T_v = look_at(eye, c)

    out = Path(os.environ["V2D_VIZ"]) / "06_agile"
    out.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out / f"{args.run}_{args.tag}.mp4"), fps=meta["fps"], codec="libx264",
                                quality=8, macro_block_size=2)
    for t in range(n):
        parts = []
        for side in ("left", "right"):
            f = res_dir / f"hands/{side}/{t:06d}.json"
            if f.exists():
                h = json.loads(f.read_text())
                v, _ = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                v = h["s_h"] * v.cpu().numpy() + np.array(h["T"])
                parts.append(colored(v, mano.faces(h["is_right"]), HAND_RGB, dev))
        for o in objs:
            p = o["dir"] / f"poses/{t:06d}.json"
            if p.exists():
                pr = json.loads(p.read_text())
                R, T, sc = np.array(pr["R"]), np.array(pr["T"]), np.array(pr["scale"])
                v = (o["renderer"].verts.cpu().numpy() * sc) @ R.T + T
                parts.append(colored(v, o["renderer"].mesh.faces_packed().cpu().numpy(), o["rgb"], dev))
        frame = cv2.cvtColor(cv2.resize(cv2.imread(str(ep / f"recon_hawor/frames/{t:06d}.png")), (W, H)),
                             cv2.COLOR_BGR2RGB)
        left, right = frame.copy(), np.full((H, W, 3), 255, np.uint8)
        if parts:
            scene = join_meshes_as_scene(parts)
            rgb, a = render(scene, Ks, np.eye(3), np.zeros(3), W, H, dev)
            left = (frame * (1 - 0.6 * a[..., None]) + rgb * 0.6 * a[..., None]).astype(np.uint8)
            right, _ = render(scene, Ks, R_v, T_v, W, H, dev)
        panel = np.hstack([left, right])
        cv2.putText(panel, f"{args.run}  frame {t}  AGILE ({args.tag})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        cv2.putText(panel, "third-person view", (W + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
        writer.append_data(panel)
    writer.close()
    print(f"wrote {out / f'{args.run}_{args.tag}.mp4'}")


if __name__ == "__main__":
    main()
