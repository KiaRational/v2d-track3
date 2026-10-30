"""Metric size of a reconstructed mesh by render-and-compare against OUR observations (no ground truth).

Why: stereo_scale.py (scaled ICP of the mesh to the masked stereo points) is degenerate for flat objects -- the
visible surface is a near-planar patch that a smaller mesh fits just as well. The silhouette fixes it: at a known
depth, the object's outline in the image has one metric size.

Per picked evaluation view (pick_views.py: unoccluded, free-standing, own SAM 3 mask), fit s, R, t of
    verts_cam = s * R @ v + t
to minimise   (1 - soft IoU(rendered silhouette, mask))  +  robust |rendered depth - stereo depth| (inside both),
with pytorch3d (soft silhouette + differentiable z-buffer) on a crop around the mask. Starts: the best scaled-ICP
solutions of stereo_scale.fit_starts (pose hypotheses), each also at the silhouette-area scale. A view counts if the
final hard IoU >= 0.85 and the median depth residual <= 8 mm (fewer than 2 such views: the views within 0.05 IoU of
the best, IoU >= 0.7, flagged low_confidence). Uniform scale = median over counted views (reported).
Then ONE per-axis scale (canonical mesh axes) is fitted jointly over the counted views, a pose per view: SAM 3D
often squashes the axis it could not see (along the view ray); other viewpoints observe it. A mild prior on the
log-scale variance keeps an axis no view constrains near uniform. The per-axis scale is KEPT only if the views span
>= 20 deg of viewing direction in the object frame and the mean IoU gains >= 0.02 (else the uniform scale). The crop
margin is 0.25 x the mask box; a render touching the crop edge marks the view unreliable (its overflow is unseen).
--apply applies the final scale (per-axis if kept, else uniform) to meshes_recon/<obj>/<obj>.obj from its .orig.obj (never compounding) and records it in PROVENANCE.json.
An overlay sheet (mask outline green, fitted mesh outline red) goes to recon_meshes/render_scale/<obj>.jpg.
Usage: python render_scale.py <obj> [--apply]   (agile image: torch, pytorch3d, trimesh, cv2)
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stereo_scale import ROOT, fit_starts, view_points  # noqa: E402
from pytorch3d.renderer import (BlendParams, MeshRasterizer, RasterizationSettings,  # noqa: E402
                                SoftSilhouetteShader)
from pytorch3d.structures import Meshes  # noqa: E402
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402
from pytorch3d.utils import cameras_from_opencv_projection  # noqa: E402

DEV = "cuda"
CROP = 256          # longest side of the working crop (pixels)
N_STARTS = 6        # ICP pose hypotheses refined per view
IOU_MIN, DEPTH_MAX_MM = 0.85, 8.0
MARGIN = float(os.environ.get("RS_MARGIN", 0.25))        # crop margin (x mask bbox); 0.6 broke every fit (all starts IoU ~0), a render touching the edge is flagged clipped instead
AXIS_MIN_SPREAD_DEG, AXIS_MIN_IOU_GAIN = 20.0, 0.02   # per-axis scale needs different viewpoints AND a clear gain


def load_view(c, mask):
    run = ROOT / "runs" / c["run"]
    man = json.loads((run / "hoi4d/scene_manifest.json").read_text())
    K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
    png = np.asarray(Image.open(run / "depth_stereo" / f"{c['frame']:06d}.png")).astype(float)
    D = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, np.nan)
    assert D.shape == mask.shape, (D.shape, mask.shape)
    v, u = np.nonzero(mask)
    x0, x1, y0, y1 = u.min(), u.max() + 1, v.min(), v.max() + 1
    mg = int(MARGIN * max(x1 - x0, y1 - y0))
    x0, y0 = max(0, x0 - mg), max(0, y0 - mg)
    x1, y1 = min(mask.shape[1], x1 + mg), min(mask.shape[0], y1 + mg)
    ds = max(x1 - x0, y1 - y0) / CROP
    w, h = int(round((x1 - x0) / ds)), int(round((y1 - y0) / ds))
    m = cv2.resize(mask[y0:y1, x0:x1].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    d = cv2.resize(np.nan_to_num(D[y0:y1, x0:x1], nan=0.0).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    Kc = K.copy(); Kc[0, 2] -= x0; Kc[1, 2] -= y0; Kc[:2] /= ds
    return {"K": Kc, "mask": m, "depth": d, "size": (h, w), "crop": (x0, y0, ds), "Kfull": K}


def setup(view, n):
    """Batched (n) soft-silhouette + hard z-buffer renderers on the view's crop, and its targets."""
    h, w = view["size"]
    K = torch.tensor(view["K"], dtype=torch.float32, device=DEV)[None].expand(n, 3, 3)
    cams = cameras_from_opencv_projection(torch.eye(3, device=DEV)[None].expand(n, 3, 3), torch.zeros(n, 3, device=DEV),
                                          K, torch.tensor([[h, w]], device=DEV).expand(n, 2))
    sig = 1e-4
    m = torch.tensor(view["mask"], device=DEV)[None]
    D = torch.tensor(view["depth"], device=DEV)[None]
    return {"n": n, "m": m, "D": D, "mb": (m > 0.5) & (D > 0.1),
            "soft": MeshRasterizer(cams, RasterizationSettings((h, w), blur_radius=float(np.log(1 / 1e-4 - 1) * sig),
                                                                faces_per_pixel=25, max_faces_per_bin=200000)),
            "hard": MeshRasterizer(cams, RasterizationSettings((h, w), blur_radius=0.0, faces_per_pixel=1,
                                                                max_faces_per_bin=200000)),
            "shader": SoftSilhouetteShader(BlendParams(sigma=sig, gamma=1e-4))}


def place(V, r, t, ls):
    """verts_cam = R(r) @ (exp(ls) * v) + t; ls (n,1) uniform or (n,3) per canonical mesh axis."""
    return torch.einsum("nij,nvj->nvi", axis_angle_to_matrix(r), torch.exp(ls)[:, None] * V[None]) + t[:, None]


def render(st, vv, F):
    me = Meshes(verts=list(vv), faces=[F] * len(vv))
    return st["shader"](st["soft"](me), me)[..., 3], st["hard"](me).zbuf[..., 0]


def losses(st, sil, z):
    m, D = st["m"], st["D"]
    inter = (sil * m).sum((1, 2)); union = (sil + m - sil * m).sum((1, 2)) + 1e-6
    both = st["mb"] & (z > 0)
    res = torch.where(both, z - D, torch.zeros_like(z))
    hub = torch.nn.functional.huber_loss(res / 0.01, torch.zeros_like(res), delta=0.5, reduction="none")
    dl = (hub * both).sum((1, 2)) / both.sum((1, 2)).clamp(min=1)
    return 1 - inter / union, dl, res, both


@torch.no_grad()
def evaluate(st, sil, z):
    hs, m = z > 0, st["m"] > 0.5
    iou = (hs & m).sum((1, 2)) / ((hs | m).sum((1, 2)) + 1e-6)
    _, _, res, both = losses(st, sil, z)
    dmed = torch.stack([res[k][both[k]].abs().median() if both[k].any() else torch.tensor(1.0, device=DEV)
                        for k in range(len(z))])
    clipped = hs[:, 0].any(1) | hs[:, -1].any(1) | hs[:, :, 0].any(1) | hs[:, :, -1].any(1)  # render left the crop
    return iou, dmed, hs, clipped


def refine(V, F, view, inits, iters=300):
    """Per view, uniform scale: all pose hypotheses in one batch."""
    n = len(inits)
    st = setup(view, n)
    r = torch.tensor(np.stack([i[0] for i in inits]), dtype=torch.float32, device=DEV).requires_grad_(True)
    t = torch.tensor(np.stack([i[1] for i in inits]), dtype=torch.float32, device=DEV).requires_grad_(True)
    ls = torch.tensor(np.log([[i[2]] for i in inits]), dtype=torch.float32, device=DEV).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [r], "lr": 0.01}, {"params": [t], "lr": 0.003}, {"params": [ls], "lr": 0.01}])
    for it in range(iters):
        opt.zero_grad()
        iou_l, dl, _, _ = losses(st, *render(st, place(V, r, t, ls), F))
        (iou_l + dl).sum().backward()
        opt.step()
    with torch.no_grad():
        iou, dmed, hs, clip = evaluate(st, *render(st, place(V, r, t, ls), F))
    return [{"iou": float(iou[k]), "clipped": bool(clip[k]), "depth_mm": float(dmed[k] * 1000), "scale": float(torch.exp(ls[k])),
             "r": r[k].detach().cpu().numpy(), "t": t[k].detach().cpu().numpy(), "sil": hs[k].cpu().numpy()}
            for k in range(n)]


def joint_aniso(V, F, views, bests, s0, iters=300, w_iso=0.5):
    """All accepted views together: ONE per-axis scale (canonical mesh axes) shared by the views, a pose per view
    (from its uniform fit). Different viewpoints constrain different axes; a mild prior w_iso * var(log s) keeps an
    axis no view constrains near the uniform scale."""
    sts = [setup(v, 1) for v in views]
    ls = torch.full((1, 3), float(np.log(s0)), device=DEV, requires_grad=True)
    rs = [torch.tensor(b["r"][None], device=DEV, requires_grad=True) for b in bests]
    ts = [torch.tensor(b["t"][None], device=DEV, requires_grad=True) for b in bests]
    # (the canonical mesh is centred at its surface centroid, so a scale change keeps each view's object centre)
    opt = torch.optim.Adam([{"params": rs, "lr": 0.005}, {"params": ts, "lr": 0.002}, {"params": [ls], "lr": 0.005}])
    for it in range(iters):
        opt.zero_grad()
        tot = w_iso * ((ls - ls.mean()) ** 2).sum()
        for st, r, t in zip(sts, rs, ts):
            iou_l, dl, _, _ = losses(st, *render(st, place(V, r, t, ls), F))
            tot = tot + (iou_l + dl).sum() / len(sts)
        tot.backward()
        opt.step()
    out = []
    with torch.no_grad():
        for st, r, t in zip(sts, rs, ts):
            iou, dmed, hs, clip = evaluate(st, *render(st, place(V, r, t, ls), F))
            out.append({"iou": float(iou[0]), "clipped": bool(clip[0]), "depth_mm": float(dmed[0] * 1000), "sil": hs[0].cpu().numpy()})
    return torch.exp(ls[0]).detach().cpu().numpy().astype(float), out




def tile_of(view, sil, label):
    tile = cv2.cvtColor((np.clip(view["depth"] / 1.2, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for msk, col in ((view["mask"] > 0.5, (0, 255, 0)), (sil, (0, 0, 255))):
        cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(tile, cs, -1, col, 1)
    cv2.putText(tile, label, (3, 14), 0, 0.4, (255, 255, 255), 1)
    return cv2.copyMakeBorder(tile, 0, CROP - tile.shape[0], 0, CROP - tile.shape[1], cv2.BORDER_CONSTANT)


def main():
    obj = sys.argv[1]
    mp = ROOT / "meshes_recon" / obj / f"{obj}.obj"
    orig = mp.with_suffix(".orig.obj")
    cand = os.environ.get("RS_MESH")   # candidate mode: fit this mesh, report to render_scale/<obj>_<RS_TAG>.json, never apply
    mesh = trimesh.load(cand if cand else (orig if orig.exists() else mp), force="mesh")   # always fit the unscaled SAM 3D-size mesh
    vd = ROOT / "recon_meshes/views" / obj
    cands = json.loads((vd / "cands.json").read_text())
    masks = np.load(vd / "cands.npz")["masks"]
    lo = mesh   # low-poly copy for rendering only (same frame and size)
    if len(mesh.faces) > 10000:
        try:
            lo = mesh.simplify_quadric_decimation(face_count=10000)
        except Exception as ex:  # noqa: BLE001  (decimation backend missing: render the full mesh)
            print(f"decimation unavailable ({ex}); rendering {len(mesh.faces)} faces", flush=True)
    V = torch.tensor(np.asarray(lo.vertices), dtype=torch.float32, device=DEV)
    F = torch.tensor(np.asarray(lo.faces), dtype=torch.int64, device=DEV)
    est, views, bests = [], [], []
    for c, mk in zip(cands, masks):
        mk = mk.astype(bool)
        P = view_points(c, mk)
        if len(P) < 300:
            continue
        view = load_view(c, mk)
        starts = sorted(fit_starts(P, mesh), key=lambda q: q[0])[:N_STARTS]
        # each pose hypothesis at its ICP scale and at the silhouette-area scale (projected area vs mask area)
        inits = []
        for _, s, R, t in starts:
            rv = matrix_to_axis_angle(torch.tensor(R, dtype=torch.float32)).numpy()
            inits.append((rv, t, s))
            z = float(np.median(P[:, 2]))
            area_m2 = view["mask"].sum() * (view["crop"][2] * z) ** 2 / (view["Kfull"][0, 0] * view["Kfull"][1, 1])
            Vr = mesh.vertices @ R.T
            hull = cv2.convexHull((Vr[:, :2] * 1e4).astype(np.float32))
            proj = cv2.contourArea(hull) / 1e8 * 0.85        # convex hull overestimates a concave outline a little
            s_area = float(np.sqrt(area_m2 / max(proj, 1e-8)))
            c3 = t + s * (R @ mesh.vertices.mean(0))
            inits.append((rv, c3 - s_area * (R @ mesh.vertices.mean(0)), s_area))
        res = refine(V, F, view, inits)
        best = min(res, key=lambda q: (1 - q["iou"]) + q["depth_mm"] / 10.0 + float(q["clipped"]))
        e = {"run": c["run"], "frame": c["frame"], "scale": best["scale"], "iou": best["iou"], "depth_mm": best["depth_mm"],
             "used": best["iou"] >= IOU_MIN and best["depth_mm"] <= DEPTH_MAX_MM and not best["clipped"],
             "clipped": best["clipped"],
             "scales_all_starts": [round(q["scale"], 3) for q in res],
             "starts": [[round(q["iou"], 3), round(q["depth_mm"], 1), q["clipped"]] for q in res]}
        est.append(e); views.append(view); bests.append(best)
        print(f"{obj} {c['run'][-2:]} f{c['frame']}: scale {e['scale']:.3f}  IoU {e['iou']:.3f}  depth {e['depth_mm']:.1f} mm"
              f"  {'USED' if e['used'] else 'rejected'}  starts(iou,mm,clip) {e['starts']}  crop {view['crop']} size {view['size']}", flush=True)
    used = [k for k, e in enumerate(est) if e["used"]]
    fallback = len(used) < 2 and bool(est)
    if fallback:  # shape mismatch caps the IoU (e.g. a squashed axis): the best-fitting views, flagged low-confidence
        top = max(e["iou"] for e in est)
        used = [k for k, e in enumerate(est) if e["depth_mm"] <= DEPTH_MAX_MM and not e["clipped"]
                and e["iou"] >= max(0.7, top - 0.05)]
        for k in used:
            est[k]["used"] = "fallback"
    s = float(np.median([est[k]["scale"] for k in used])) if used else float("nan")
    spread = float(np.median(np.abs(np.array([est[k]["scale"] for k in used]) - s)) / s) if used else float("nan")
    ext = mesh.extents   # canonical mesh: axis-aligned extents along its principal axes (x, y, z)
    print(f"{obj}: UNIFORM scale {s:.3f} over {len(used)}/{len(est)} views (MAD {spread*100:.1f}%){' LOW-CONFIDENCE (fallback views)' if fallback else ''} -> extent "
          f"{np.round(np.sort(ext * s)[::-1] * 100, 1).tolist()} cm (was {np.round(np.sort(ext)[::-1] * 100, 1).tolist()})",
          flush=True)
    s3, joint, axis_check = None, [], None
    if len(used) >= 2:
        s3, joint = joint_aniso(V, F, [views[k] for k in used], [bests[k] for k in used], s)
        # per-axis only when it is identifiable and it pays: viewing directions in the object frame (-R^T t) must
        # span >= AXIS_MIN_SPREAD_DEG, the mean IoU must gain >= AXIS_MIN_IOU_GAIN over the per-view uniform fits,
        # and no joint render may leave its crop. Otherwise the uniform scale stands.
        dirs = []
        for k in used:
            R = axis_angle_to_matrix(torch.tensor(bests[k]["r"])[None])[0].numpy()
            dv = -R.T @ bests[k]["t"]; dirs.append(dv / np.linalg.norm(dv))
        spread_deg = max(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))) for a in dirs for b in dirs)
        gain = float(np.mean([j["iou"] for j in joint]) - np.mean([est[k]["iou"] for k in used]))
        axis_check = {"view_spread_deg": spread_deg, "iou_gain": gain, "clipped": any(j["clipped"] for j in joint)}
        axis_check["accepted"] = (spread_deg >= AXIS_MIN_SPREAD_DEG and gain >= AXIS_MIN_IOU_GAIN
                                  and not axis_check["clipped"])
        print(f"{obj}: PER-AXIS scale {np.round(s3, 3).tolist()} (x,y,z) -> extent "
              f"{np.round(np.sort(ext * s3)[::-1] * 100, 1).tolist()} cm; joint IoU "
              f"{[round(j['iou'], 3) for j in joint]} depth mm {[round(j['depth_mm'], 1) for j in joint]}; check {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in axis_check.items()})}", flush=True)
    od = ROOT / "recon_meshes/render_scale"; od.mkdir(parents=True, exist_ok=True)
    tiles = [tile_of(v, b["sil"], f"{e['run'][-4:]} s={e['scale']:.2f} iou={e['iou']:.2f}")
             for v, b, e in zip(views, bests, est)]
    rows = [np.concatenate(tiles, 1)] if tiles else []
    if joint:
        jt = [tile_of(views[k], j["sil"], f"joint iou={j['iou']:.2f}") for k, j in zip(used, joint)]
        rows.append(np.concatenate(jt + [np.zeros_like(jt[0])] * (len(tiles) - len(jt)), 1))
    if rows:
        cv2.imwrite(str(od / (f"{obj}_{os.environ.get('RS_TAG', 'cand')}.jpg" if cand else f"{obj}.jpg")), np.concatenate(rows, 0))
    final = s3 if (axis_check or {}).get("accepted") else (np.full(3, s) if used else None)
    rec = {"uniform": s, "mad_rel": spread, "low_confidence": fallback, "per_axis": None if s3 is None else s3.tolist(),
           "per_axis_check": axis_check, "final": None if final is None else final.tolist(),
           "joint": [{k: v for k, v in j.items() if k != "sil"} for j in joint], "views": est}
    tag = ("_" + os.environ["RS_TAG"]) if cand and os.environ.get("RS_TAG") else ""
    (od / f"{obj}{tag}.json").write_text(json.dumps(rec, indent=1))
    if cand:
        return
    if "--apply" in sys.argv and final is not None:
        if not orig.exists():
            mesh.export(orig)
        mesh.apply_transform(np.diag([*final, 1.0]))
        mesh.export(mp)
        pp = mp.parent / "PROVENANCE.json"
        prov = json.loads(pp.read_text())
        prov["render_scale"] = dict(rec, method="render-and-compare: per view (soft silhouette IoU to own SAM 3 mask + "
                                    "robust stereo depth), views with IoU>=0.85 and depth<=8 mm; then one per-axis scale "
                                    "(canonical axes) fitted jointly over those views with a pose per view, kept only if the views span >= 20 deg "
                                    "and the mean IoU gains >= 0.02 (else the uniform median)")
        prov["extent_cm"] = (np.sort(mesh.extents)[::-1] * 100).round(1).tolist()
        pp.write_text(json.dumps(prov, indent=1))
        print(f"{obj}: APPLIED {'per-axis' if final is s3 else 'uniform'} {np.round(final, 3).tolist()} -> extent {prov['extent_cm']} cm", flush=True)


if __name__ == "__main__":
    main()
