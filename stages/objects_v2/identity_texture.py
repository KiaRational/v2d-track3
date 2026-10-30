"""Identity + flip disambiguation for look-alike objects by TEXTURE, not shape.

The two wooden boards have the same outline and thickness (stereo ICP separates them
by only ~0.05 mm, i.e. noise), but their scans carry different surface detail: piece 1
has four mocap markers at the corners, piece 2 has three plus a printed "2", and the
interlocking slot sits at a different place. That detail is baked into each mesh's
texture, so the right mesh is the one whose texture, seen from the object's pose,
actually predicts the pixels.

For every hypothesis (mesh, pose-symmetry) we sample the mesh surface with its texture
colours, project it with the tracked pose, keep the front-facing points that land
inside the object's mask, and correlate predicted vs observed luminance/saturation
(correlation, so exposure and white balance drop out). Summed over frames this both
picks the mesh and fixes the 180 deg flips that FoundationPose leaves on a near-
symmetric board.

Used by s05_objects_v2.py when two objects' meshes are the same size.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

# 180 deg rotations about the mesh axes: the near-symmetries of a flat rectangular board.
SYMS = {
    "I": np.eye(3),
    "Rx": np.diag([1.0, -1.0, -1.0]),
    "Ry": np.diag([-1.0, 1.0, -1.0]),
    "Rz": np.diag([-1.0, -1.0, 1.0]),
}


def load_T_any(path):
    """Camera-from-object 4x4 from either pose format: the toolkit's
    {"rotation": w-first quat, "translation"} or the fused {"R", "T"}."""
    from scipy.spatial.transform import Rotation
    d = json.loads(Path(path).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), np.array(d["T"])
    else:
        q = d["rotation"]
        T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T[:3, 3] = d["translation"]
    return T


def sample_textured(mesh_path, n=30000, seed=0):
    """Surface points + normals + texture RGB."""
    m = trimesh.load(mesh_path, force="mesh")
    pts, fidx = trimesh.sample.sample_surface(m, n, seed=seed)
    nrm = m.face_normals[fidx]
    tri = m.faces[fidx]
    bary = trimesh.triangles.points_to_barycentric(m.vertices[tri], pts)
    if m.visual.kind != "texture":   # vertex-coloured mesh (our reconstructed meshes_recon): interpolate the colours
        vc = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32)
        return np.asarray(pts), np.asarray(nrm), (vc[tri] * bary[:, :, None]).sum(1)
    tex = np.asarray(m.visual.material.image.convert("RGB")).astype(np.float32)
    uv = m.visual.uv
    puv = (uv[tri] * bary[:, :, None]).sum(1)
    H, W = tex.shape[:2]
    px = np.clip((puv[:, 0] * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(((1 - puv[:, 1]) * (H - 1)).astype(int), 0, H - 1)
    return np.asarray(pts), np.asarray(nrm), tex[py, px]


def _feat(rgb_u8):
    """Luminance and saturation, the channels the markers/slot show up in."""
    hsv = cv2.cvtColor(rgb_u8.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    return hsv[:, 2].astype(np.float32), hsv[:, 1].astype(np.float32)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def score_frame(pts, nrm, col, T, K, img, mask, min_pts=250):
    """Correlation between texture-predicted and observed appearance."""
    P = pts @ T[:3, :3].T + T[:3, 3]
    N = nrm @ T[:3, :3].T
    vis = ((N * P).sum(1) < 0) & (P[:, 2] > 0.05)      # front-facing, in front of camera
    if vis.sum() < min_pts:
        return None
    P, C = P[vis], col[vis]
    uv = (P[:, :2] / P[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    h, w = mask.shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() < min_pts:
        return None
    u, v, C = u[ok], v[ok], C[ok]
    inm = mask[v, u]                                    # ignore hand-occluded pixels
    if inm.sum() < min_pts:
        return None
    u, v, C = u[inm], v[inm], C[inm]
    obs_v, obs_s = _feat(img[v, u])
    pre_v, pre_s = _feat(C)
    return _corr(pre_v, obs_v) + 0.5 * _corr(pre_s, obs_s), int(len(u))


def evaluate(run: Path, work: Path, objs, names, frames, pose_dir="poses_smoothed", n_samp=30000):
    """score[slot_id][mesh_name][sym] summed over frames."""
    from geom import load_mask
    ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
    tex = {nm: sample_textured(str(Path(objs[0]["mesh"]).parent.parent / nm / f"{nm}.obj"), n_samp)
           for nm in names}
    out = {}
    for o in objs:
        if o["name"] not in names:
            continue
        res = {nm: {s: [] for s in SYMS} for nm in names}
        for t in frames:
            mf = work / f"masks/{o['id']}/{t:06d}.png"
            pf = work / o["name"] / pose_dir / f"{t:06d}.json"
            if not mf.exists() or not pf.exists():
                continue
            mask = load_mask(mf)
            if mask is None or mask.sum() < 3000:
                continue
            img = cv2.cvtColor(cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png")), cv2.COLOR_BGR2RGB)
            T0 = load_T_any(pf)
            for nm in names:
                pts, nrm, col = tex[nm]
                for sname, S in SYMS.items():
                    T = T0.copy()
                    T[:3, :3] = T0[:3, :3] @ S
                    r = score_frame(pts, nrm, col, T, K, img, mask)
                    if r is not None:
                        res[nm][sname].append(r[0])
        out[o["id"]] = {nm: {s: (float(np.mean(v)), len(v)) for s, v in d.items() if v}
                        for nm, d in res.items()}
    return out


def decide(scores, objs, names):
    """Best (mesh, sym) per slot, then the globally best one-to-one assignment."""
    best = {}
    for oid, per_mesh in scores.items():
        best[oid] = {}
        for nm, per_sym in per_mesh.items():
            if not per_sym:
                continue
            s, (v, n) = max(per_sym.items(), key=lambda kv: kv[1][0])
            best[oid][nm] = {"sym": s, "score": v, "n_frames": n}
    ids = [o["id"] for o in objs if o["id"] in best and len(best[o["id"]]) == len(names)]
    if len(ids) != 2 or len(names) != 2:
        return None, best
    a, b = ids
    n0, n1 = names
    keep = best[a].get(n0, {}).get("score", -9) + best[b].get(n1, {}).get("score", -9)
    swap = best[a].get(n1, {}).get("score", -9) + best[b].get(n0, {}).get("score", -9)
    return {"slots": [a, b], "keep_score": keep, "swap_score": swap,
            "margin": abs(keep - swap), "decision": "swap" if swap > keep else "keep"}, best


AMBIG_MARGIN = 0.10   # below this the two meshes are not distinguishable


def resolve(run: Path, work: Path, objs, n, max_frames=22, pose_dir="poses_smoothed"):
    """Decide the identity of every same-size pair by texture and apply it.

    Returns a report; when a pair is swapped the mask directories are exchanged and
    the (now stale) pose directories removed so the caller re-runs FoundationPose
    with the right mesh. `sym` records the mesh-symmetry flip that best explains the
    appearance - useful downstream for the near-symmetric boards.
    """
    import shutil
    from identity_icp import same_size_pairs
    report = []
    for oa, ob in same_size_pairs(objs):
        names = [oa["name"], ob["name"]]
        frames = [t for t in range(0, n, max(1, n // max_frames))]
        sc = evaluate(run, work, [oa, ob], names, frames, pose_dir=pose_dir)
        dec, best = decide(sc, [oa, ob], names)
        rec = {"pair": names, "method": "texture"}
        if dec is None:
            rec.update(decision="unknown", confident=False)
            report.append(rec)
            continue
        ambiguous = dec["margin"] < AMBIG_MARGIN
        rec.update(decision=dec["decision"], confident=not ambiguous, margin=dec["margin"],
                   keep_score=dec["keep_score"], swap_score=dec["swap_score"],
                   per_slot={str(k): v for k, v in best.items()})
        if dec["decision"] == "swap" and not ambiguous:
            for sub in ("masks", "masks_ekf"):
                a, b = work / sub / str(oa["id"]), work / sub / str(ob["id"])
                if a.is_dir() and b.is_dir():
                    tmp = work / sub / "_swap"
                    a.rename(tmp); b.rename(a); tmp.rename(b)
            for o in (oa, ob):
                shutil.rmtree(work / o["name"], ignore_errors=True)   # poses are stale
        report.append(rec)
    return report
