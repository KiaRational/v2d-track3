"""README stage 5.6: collision geometry next to the visual meshes. Cavity-aware: CoACD convex parts (cached by
stereo/physics_check.py in cache/coacd/<name>/part_*.obj; a cup is ~24 parts around its open cavity, never one
hull). Each part is stored as half-spaces (A x <= b) for fast signed-distance-like penetration tests."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from .. import io as hio

CACHE = hio.ROOT / ("cache/coacd" + {"meshes_obj": ""}.get(os.environ.get("V2D_MESH_DIR", "meshes_obj"), "_" + os.environ.get("V2D_MESH_DIR", "")))  # per mesh set


class ObjectAsset:
    def __init__(self, name: str, n_surface: int = 1500, seed: int = 0):
        self.name = name
        mesh = trimesh.load(hio.ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj", force="mesh")
        self.extent = mesh.bounds[1] - mesh.bounds[0]
        rng = np.random.default_rng(seed)
        pts, _ = trimesh.sample.sample_surface(mesh, n_surface, seed=int(rng.integers(1 << 30)))
        self.surface = np.asarray(pts)
        self.centroid = mesh.centroid if mesh.is_watertight else self.surface.mean(0)
        files = sorted((CACHE / name).glob("part_*.obj"))
        self.parts = []
        for f in files:
            v = np.asarray(trimesh.load(f, force="mesh").vertices)
            h = ConvexHull(v)
            self.parts.append((h.equations[:, :3], h.equations[:, 3]))  # n.x + d <= 0 inside
        self.collision = "coacd_parts" if self.parts else "none (no cached decomposition)"
        if not self.parts and os.environ.get("HOI4D_ALLOW_NO_COLLISION") != "1":
            # silently-zero penetration made every collision term / anomaly of a mesh set without a cache vanish
            raise FileNotFoundError(f"no CoACD parts for {name} in {CACHE / name} (build them: beat1/coacd_cache.sbatch; "
                                    "HOI4D_ALLOW_NO_COLLISION=1 to run without collision)")

    def penetration(self, pts_obj: np.ndarray) -> np.ndarray:
        """Depth (m, >= 0) of points (object frame) inside the collision geometry: max over parts of the
        min distance to that part's faces (0 when outside every part)."""
        if not self.parts:
            return np.zeros(len(pts_obj))
        best = np.zeros(len(pts_obj))
        for n, d in self.parts:
            s = pts_obj @ n.T + d  # (P, faces) signed plane distances, inside: all <= 0
            inside = (s <= 0).all(1)
            depth = np.where(inside, -s.max(1), 0.0)
            best = np.maximum(best, depth)
        return best


@lru_cache(maxsize=32)
def asset(name: str) -> ObjectAsset:
    return ObjectAsset(name)


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def inv(T: np.ndarray) -> np.ndarray:
    o = np.eye(4)
    o[:3, :3] = T[:3, :3].T
    o[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return o
