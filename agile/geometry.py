"""Geometry for AGILE (Shi et al., SIGGRAPH 2026): MANO, projection, constrained
trimmed ICP (Chetverikov et al. 2002), translation-only PnP, object SDF grid."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from manotorch.manolayer import ManoLayer
from scipy.spatial import cKDTree


class Mano:
    """The same MANO layer the toolkit uses for WiLoR/HaMeR records (manotorch,
    axis-angle, no PCA, right-hand model; left hands are mirrored in x)."""

    def __init__(self, assets_root: str, device="cuda"):
        self.layer = ManoLayer(rot_mode="axisang", use_pca=False, side="right", center_idx=None,
                               mano_assets_root=assets_root).to(device)
        self.faces_right = self.layer.th_faces.cpu().numpy()
        self.device = device

    @torch.no_grad()
    def local(self, rec: dict):
        """Posed verts (778,3) and joints (21,3) in the hand's local frame, with the
        predicted global orientation R_h applied (WiLoR convention), metres."""
        m = rec["mano"]
        pose = torch.tensor(np.r_[m["global_orient"], m["hand_pose"]], dtype=torch.float32, device=self.device)[None]
        betas = torch.tensor(m["betas"], dtype=torch.float32, device=self.device)[None]
        out = self.layer(pose, betas)
        v, j = out.verts[0], out.joints[0]
        if not rec["is_right"]:
            flip = torch.tensor([-1.0, 1.0, 1.0], device=self.device)
            v, j = v * flip, j * flip
        return v, j

    def faces(self, is_right: bool):
        return self.faces_right if is_right else self.faces_right[:, [0, 2, 1]]


def project(pts: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole projection of (...,3) camera-frame points to (...,2) pixels."""
    z = pts[..., 2:3].clamp(min=1e-6)
    return pts[..., :2] / z * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])


def wilor_keypoints_2d(joints_local: np.ndarray, rec: dict, image_wh) -> np.ndarray:
    """WiLoR's 2D keypoints J2D: its predicted 3D joints projected with its own
    full-image camera (focal_length, cam_t, principal point at the image centre)."""
    cam_t = np.asarray(rec["camera"]["cam_t"])
    f = rec["camera"]["focal_length"]
    p = joints_local + cam_t
    return f * p[:, :2] / p[:, 2:3] + np.asarray(image_wh, float) / 2


def unproject_mask(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, max_pts=20000, rng=None):
    v, u = np.nonzero(mask & (depth > 1e-3) & np.isfinite(depth))
    if len(u) > max_pts:
        idx = (rng or np.random.default_rng(0)).choice(len(u), max_pts, replace=False)
        u, v = u[idx], v[idx]
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def visible_vertices(verts: np.ndarray, faces: np.ndarray, cam_pos=np.zeros(3)) -> np.ndarray:
    """Vertices whose adjacent faces face the camera (depth sensors only see these)."""
    mesh = trimesh.Trimesh(verts, faces, process=False)
    facing = ((cam_pos - mesh.triangles_center) * mesh.face_normals).sum(1) > 0
    vis = np.zeros(len(verts), bool)
    vis[np.unique(faces[facing])] = True
    return vis


def constrained_trimmed_icp_scale(model: np.ndarray, scan: np.ndarray, t0: np.ndarray, s0=1.0,
                                  iters=50, keep=0.8):
    """Rotation-fixed ICP over (scale s, translation t) of s*model + t (model
    already rotated). Each step: nearest neighbours in both directions, keep the best
    `keep` fraction of each (trimmed ICP), closed-form least squares for (s, t)."""
    tree = cKDTree(scan)
    s, t = float(s0), np.asarray(t0, float).copy()
    for _ in range(iters):
        cur = s * model + t
        # Symmetric trimmed correspondences: model->scan alone is biased toward
        # shrinking (a smaller model fits inside the partial scan; we measured
        # s_h ~ 0.78 on ep 11), scan->model alone toward growing.
        d, idx = tree.query(cur)
        sel = np.argsort(d)[: max(3, int(keep * len(d)))]
        d2, idx2 = cKDTree(cur).query(scan)
        sel2 = np.argsort(d2)[: max(3, int(keep * len(d2)))]
        m = np.r_[model[sel], model[idx2[sel2]]]
        q = np.r_[scan[idx[sel]], scan[sel2]]
        mc, qc = m - m.mean(0), q - q.mean(0)
        s_new = float((mc * qc).sum() / max((mc * mc).sum(), 1e-12))
        t_new = q.mean(0) - s_new * m.mean(0)
        done = abs(s_new - s) < 1e-6 and np.linalg.norm(t_new - t) < 1e-7
        s, t = s_new, t_new
        if done:
            break
    return s, t, float(np.median(np.sort(d)[: len(sel)]))


def pnp_translation(points3d: np.ndarray, uv: np.ndarray, K: np.ndarray, iters=10) -> np.ndarray:
    """Translation-only PnP (rotation and scale fixed): linear least squares
    initialisation, then Gauss-Newton on the reprojection error."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X, Y, Z = points3d.T
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    A = np.zeros((2 * len(X), 3))
    b = np.zeros(2 * len(X))
    A[0::2, 0], A[0::2, 2], b[0::2] = fx, -du, du * Z - fx * X
    A[1::2, 1], A[1::2, 2], b[1::2] = fy, -dv, dv * Z - fy * Y
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(iters):
        P = points3d + t
        z = P[:, 2]
        r = np.r_[fx * P[:, 0] / z + cx - uv[:, 0], fy * P[:, 1] / z + cy - uv[:, 1]]
        J = np.zeros((2 * len(z), 3))
        J[: len(z), 0], J[: len(z), 2] = fx / z, -fx * P[:, 0] / z ** 2
        J[len(z):, 1], J[len(z):, 2] = fy / z, -fy * P[:, 1] / z ** 2
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        t += step
        if np.linalg.norm(step) < 1e-9:
            break
    return t


class SDFGrid:
    """Signed distance of the (canonical, unscaled) object mesh on a regular grid,
    sampled trilinearly in torch.

    Distance: KD-tree over dense surface samples (exact closest-point queries on the
    ~100k-face scans exhaust RAM). Sign: majority vote of the outward normals at the
    k nearest samples, which also works for scans that are not watertight (ray-based
    containment would silently disable penetration detection there)."""

    def __init__(self, mesh: trimesh.Trimesh, res=96, pad=0.06, device="cuda", n_samples=200_000, k=8):
        lo, hi = mesh.bounds[0] - pad, mesh.bounds[1] + pad
        axes = [np.linspace(lo[i], hi[i], res) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        pts, fidx = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
        normals = mesh.face_normals[fidx]
        dist, idx = cKDTree(pts).query(g, k=k, workers=-1)
        side = np.einsum("nkc,nkc->nk", g[:, None, :] - pts[idx], normals[idx])
        inside = (side < 0).sum(1) > k // 2
        sdf = np.where(inside, -dist[:, 0], dist[:, 0]).reshape(res, res, res)
        # grid_sample expects (N,C,D,H,W) with coords ordered (x=W, y=H, z=D).
        self.vol = torch.tensor(sdf.transpose(2, 1, 0), dtype=torch.float32, device=device)[None, None]
        self.lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.hi = torch.tensor(hi, dtype=torch.float32, device=device)
        self.watertight = mesh.is_watertight

    def __call__(self, pts: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        """SDF at object-local points; an anisotropic scale is handled by querying
        the canonical mesh at pts/scale and rescaling by the mean scale."""
        q = pts / scale if scale is not None else pts
        n = 2 * (q - self.lo) / (self.hi - self.lo) - 1
        val = F.grid_sample(self.vol, n.view(1, -1, 1, 1, 3), align_corners=True, padding_mode="border").view(-1)
        # Outside the grid, fall back to distance to the grid box plus the border value.
        outside = (n.abs() > 1).any(-1)
        extra = ((q - q.clamp(self.lo, self.hi)).norm(dim=-1))
        val = torch.where(outside, val.clamp(min=0) + extra, val)
        return val * (scale.mean() if scale is not None else 1.0)


def rotvec_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) -> (3,3). PyTorch3D's conversion is gradient-safe at w = 0,
    where every per-frame optimisation starts (a delta on the initial rotation)."""
    from pytorch3d.transforms import axis_angle_to_matrix

    return axis_angle_to_matrix(w)
