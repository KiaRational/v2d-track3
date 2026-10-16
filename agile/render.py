"""PyTorch3D rendering for AGILE: soft silhouettes (L_mask) and textured views +
depth (L_dino point features / visibility), in an object-centred square crop."""
from __future__ import annotations

import math

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (AmbientLights, BlendParams, MeshRasterizer, MeshRenderer,
                                RasterizationSettings, SoftPhongShader, SoftSilhouetteShader)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


class Crop:
    """Square crop [x0, x0+size) x [y0, y0+size) of the full image, resampled to res x res."""

    def __init__(self, x0: float, y0: float, size: float, res: int):
        self.x0, self.y0, self.size, self.res = float(x0), float(y0), float(size), int(res)
        self.s = res / size

    @staticmethod
    def around(box, image_wh, res, expand=1.4, min_size=96):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = max(x1 - x0, y1 - y0, min_size) * expand
        return Crop(cx - size / 2, cy - size / 2, size, res)

    def K(self, K_full: torch.Tensor) -> torch.Tensor:
        K = K_full.clone()
        K[0, 0], K[1, 1] = K_full[0, 0] * self.s, K_full[1, 1] * self.s
        K[0, 2], K[1, 2] = (K_full[0, 2] - self.x0) * self.s, (K_full[1, 2] - self.y0) * self.s
        return K

    def to_crop(self, uv: torch.Tensor) -> torch.Tensor:
        return (uv - torch.tensor([self.x0, self.y0], device=uv.device)) * self.s

    def image(self, img: np.ndarray, interp) -> np.ndarray:
        """Resample a full-resolution image/mask into the crop (outside -> 0)."""
        import cv2
        M = np.array([[self.s, 0, -self.x0 * self.s], [0, self.s, -self.y0 * self.s]], np.float32)
        return cv2.warpAffine(img, M, (self.res, self.res), flags=interp, borderValue=0)


def silhouette_proxy(obj_path: str, faces=6000):
    """Decimated, welded copy of the scan for differentiable silhouettes. With ~100k
    sub-pixel faces in the crop, soft-rasteriser gradients are dominated by interior
    tessellation: on ep 11 autograd dL/dT was ~10x the finite difference and pointed
    elsewhere, and Adam drove L_mask *up*. Cached next to the mesh."""
    import trimesh
    cache = obj_path.replace(".obj", f"_sil{faces}.ply")
    try:
        m = trimesh.load(cache, process=False)
    except Exception:
        m = trimesh.load(obj_path, force="mesh", process=False)
        m = trimesh.Trimesh(m.vertices, m.faces, process=True)  # weld (drop UV seams)
        m = m.simplify_quadric_decimation(face_count=faces)
        m.export(cache)
    return m


class Renderer:
    def __init__(self, obj_path: str, device="cuda", sil_faces_per_pixel=50, sigma=1e-4):
        self.device = device
        self.mesh = load_objs_as_meshes([obj_path], device=device)
        self.verts = self.mesh.verts_packed()  # canonical, metres (full textured scan)
        proxy = silhouette_proxy(obj_path)
        self.sil_mesh = Meshes([torch.tensor(proxy.vertices, dtype=torch.float32, device=device)],
                               [torch.tensor(proxy.faces, dtype=torch.int64, device=device)])
        self.sil_verts = self.sil_mesh.verts_packed()
        blur = math.log(1.0 / 1e-4 - 1.0) * sigma
        # ~100k-face scans overflow the coarse rasteriser's default per-bin face budget
        # (incomplete silhouettes); give every bin room for the whole mesh.
        self.max_faces_per_bin = 120_000
        self.sil_settings = dict(blur_radius=blur, faces_per_pixel=sil_faces_per_pixel, perspective_correct=False,
                                 max_faces_per_bin=self.max_faces_per_bin)
        self.sil_blend = BlendParams(sigma=sigma, gamma=1e-4)
        self.lights = AmbientLights(device=device)  # unlit texture: real lighting unknown

    def _cameras(self, K: torch.Tensor, res: int):
        eye = torch.eye(3, device=self.device)[None]
        zero = torch.zeros(1, 3, device=self.device)
        size = torch.tensor([[res, res]], device=self.device, dtype=torch.float32)
        return cameras_from_opencv_projection(eye, zero, K[None], size)

    def posed(self, R: torch.Tensor, T: torch.Tensor, scale: torch.Tensor):
        """Mesh transformed to the OpenCV camera frame: R (s * p) + T."""
        v = (self.verts * scale) @ R.T + T
        return self.mesh.update_padded(v[None])

    def silhouette(self, R, T, scale, K_crop, res):
        cams = self._cameras(K_crop, res)
        rast = MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(image_size=res, **self.sil_settings))
        v = (self.sil_verts * scale) @ R.T + T
        img = MeshRenderer(rast, SoftSilhouetteShader(blend_params=self.sil_blend))(self.sil_mesh.update_padded(v[None]))
        return img[0, ..., 3]

    @torch.no_grad()
    def textured(self, R, T, scale, K_crop, res):
        """Unlit textured render (res,res,3 in [0,1]) and depth buffer (inf = empty)."""
        cams = self._cameras(K_crop, res)
        rast = MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(image_size=res, faces_per_pixel=1,
                                                                                        max_faces_per_bin=self.max_faces_per_bin))
        mesh = self.posed(R, T, scale)
        frags = rast(mesh)
        shader = SoftPhongShader(device=self.device, cameras=cams, lights=self.lights,
                                 blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)))
        rgb = shader(frags, mesh)[0, ..., :3]
        z = frags.zbuf[0, ..., 0]
        return rgb, torch.where(z > 0, z, torch.full_like(z, float("inf")))


def sample_surface(obj_path: str, n=20000, seed=0) -> np.ndarray:
    import trimesh
    mesh = trimesh.load(obj_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return pts.astype(np.float32)
