"""EgoPHI (ECCV 2026) contact and force on our reconstructions (egophi image).

EgoPHI predicts dense per-vertex contact probability and force (magnitude + direction) on both
hand meshes and the object mesh from one egocentric RGB frame, the hand meshes and the object
geometry. Inputs are built from our pipeline exactly in its ARCTIC conventions:
  image    colour frame resized to 224x224, [0, 1] (arctic_preprocess/resize224.py)
  hands    stereo MANO fit (stereo/<hands>) in the camera frame, metres, MANO vertex order
  object   scanned mesh decimated to ~--obj_verts vertices, template = (v - mean) / max|v - mean|
           with s = that max (the similarity scale of its training data), rigid (no articulation)
  bbox     SAM 3 object mask box at 224 px;  K at full resolution with H_orig, W_orig
Outputs <run>/stereo/egophi/<object>.npz (per frame: hand/object contact and force, EgoPHI's own
object pose) and, against our geometric contact (hand vertices within --geo_mm of the object at
its fused pose), per-frame agreement; video $V2D_VIZ/11_egophi/<run>_<object>.mp4 (hand vertices
coloured by EgoPHI contact, green outline = our geometric contact).
Usage: python egophi_run.py <run> [...] [--hands mano_fit] [--variant fused_all_sam3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ["V2D_ROOT"])
EGO = Path(os.environ.get("EGOPHI_ROOT", ROOT / "egophi"))
sys.path.insert(0, str(EGO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
# utils.py imports a config from the separate HACO repo, used only for HACO contact data: stub it
_lib = types.ModuleType("lib"); _core = types.ModuleType("lib.core"); _cfgm = types.ModuleType("lib.core.config")
_cfgm.cfg = types.SimpleNamespace(MODEL=types.SimpleNamespace(), TRAIN=types.SimpleNamespace())
sys.modules.update({"lib": _lib, "lib.core": _core, "lib.core.config": _cfgm})
import timm  # noqa: E402

_create = timm.create_model
timm.create_model = lambda *a, **k: _create(*a, **{**k, "pretrained": False})  # weights come from the checkpoint
from model import InteractionGNN  # noqa: E402
from utils import extract_2d_bboxes, prepare_mano_edges  # noqa: E402
from torch_geometric.utils import to_undirected  # noqa: E402
from geometry import Mano  # noqa: E402

HAND_IDS = {"left": 2, "right": 3}
F_ = torch.nn.functional


def forward_known_pose(m, x, vl, vr, vlc, vrc, tmpl, scale, bbox, K, el, er, eo, H, W, R, T):
    """EgoPHI's forward pass with the object held at a KNOWN pose (R, T: camera <- normalised template,
    verts = tmpl R^T * scale + T) instead of its own iterative estimate. Same layers and weights; the
    pose-update heads are skipped. On objects it never saw in training its own pose is 10-17 cm off, and
    its contact/force then refer to an object in the wrong place."""
    from model import project_and_sample
    from torchvision.ops import roi_align
    B, N_obj = x.shape[0], tmpl.shape[1]
    feat_map = m.backbone(x)
    H_feat = feat_map.shape[2]
    rois = torch.cat((torch.arange(B, device=x.device).float().view(-1, 1), bbox), dim=1)
    feat_map_obj = roi_align(feat_map, rois, output_size=feat_map.shape[2:], spatial_scale=H_feat / 224.0,
                             sampling_ratio=-1)
    h_l = F_.gelu(m.input_mlp_hand(torch.cat([m.vis_embed_hand(project_and_sample(vl, K, feat_map, H, W)),
                                              m.geo_embed_hand(vl)], -1)))
    h_r = F_.gelu(m.input_mlp_hand(torch.cat([m.vis_embed_hand(project_and_sample(vr, K, feat_map, H, W)),
                                              m.geo_embed_hand(vr)], -1)))
    verts_o = torch.bmm(tmpl, R.transpose(1, 2)) * scale.view(-1, 1, 1) + T.unsqueeze(1)
    h_o = F_.gelu(m.input_mlp_obj_pose(torch.cat([
        m.vis_embed_obj_pose(project_and_sample(verts_o, K, feat_map_obj, H, W)), m.geo_embed_obj_local(tmpl),
        m.geo_embed_obj_hand_rel_l(verts_o - vlc.unsqueeze(1).expand(-1, N_obj, -1)),
        m.geo_embed_obj_hand_rel_r(verts_o - vrc.unsqueeze(1).expand(-1, N_obj, -1))], -1)))
    for blk in m.pose_interaction_blocks:
        h_l, h_r, h_o = blk(h_l, h_r, h_o, el.t(), er.t(), eo)
    h_of = F_.gelu(m.input_mlp_obj_force(torch.cat([m.vis_embed_obj_force(project_and_sample(verts_o, K, feat_map, H, W)),
                                                   m.geo_embed_obj_cam(verts_o)], -1)))
    for blk in m.force_interaction_blocks:
        h_l, h_r, h_of = blk(h_l, h_r, h_of, el.t(), er.t(), eo)
    cl, fl, _ = m.head_hand(h_l)
    cr, fr, _ = m.head_hand(h_r)
    co, fo, _ = m.head_obj(h_of)
    return cl, cr, co, fl, fr, fo, verts_o


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def object_graph(name, n_verts):
    import fast_simplification
    m = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj", force="mesh")
    m.merge_vertices()
    if len(m.vertices) > n_verts:
        v, f = fast_simplification.simplify(m.vertices, m.faces, target_reduction=1 - n_verts / len(m.vertices))
        m = trimesh.Trimesh(v, f, process=True)
    V = np.asarray(m.vertices, np.float32)
    c = V.mean(0)
    s = float(np.abs(V - c).max())
    E = torch.tensor(np.asarray(m.edges_unique), dtype=torch.long).t().contiguous()
    return V, c, s, to_undirected(E), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", default="mano_fit")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--obj_verts", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--geo_mm", type=float, default=5.0)
    ap.add_argument("--thr", type=float, default=0.3, help="EgoPHI's own hand-contact threshold (compute_metrics.py)")
    ap.add_argument("--own_pose", action="store_true", help="let EgoPHI estimate the object pose itself")
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    model = InteractionGNN(d_model=512, n_head=4, num_pose_blocks=4, num_force_blocks=4,
                           backbone_feature_dim=768, num_pose_iterations=3).to(dev)
    ck = torch.load(EGO / "checkpoints/best_EgoPHI.pth", map_location=dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    e_l, e_r, _, _ = prepare_mano_edges()
    e_l, e_r = e_l.to(dev), e_r.to(dev)
    mano = Mano(str(ROOT / "weights/hand"), dev)
    print("EgoPHI loaded", flush=True)
    for r in a.runs:
        run = ROOT / "runs" / r
        ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]], np.float32)
        Hh, Ww = ki["height"], ki["width"]
        n = len(list((run / "recon_hawor/frames").glob("*.png")))
        hands = {}
        for side in ("left", "right"):
            for f in (run / "stereo" / a.hands / side).glob("*.json"):
                h = json.loads(f.read_text())
                v, _ = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                hands.setdefault(side, {})[int(f.stem)] = v.cpu().numpy() + np.array(h["T"])
        outdir = run / "stereo/egophi"
        outdir.mkdir(parents=True, exist_ok=True)
        for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
            nm = o["name"]
            V, c, s, E_o, mdec = object_graph(nm, a.obj_verts)
            tmpl = torch.tensor((V - c) / s, device=dev)
            pd = run / a.variant / nm / "poses"
            W = {t: load_T(pd / f"{t:06d}.json") for t in range(n) if (pd / f"{t:06d}.json").exists()}
            S = trimesh.sample.sample_surface(trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm / f"{nm}.obj", force="mesh"),
                                              20000, seed=0)[0]
            tree = cKDTree(S)
            # frames with a hand near the object (where contact can happen), in batches
            frames = []
            for t in sorted(W):
                near = False
                for side in ("left", "right"):
                    hv = hands.get(side, {}).get(t)
                    if hv is not None:
                        Wi = np.linalg.inv(W[t])
                        if tree.query(hv @ Wi[:3, :3].T + Wi[:3, 3])[0].min() < 0.05:
                            near = True
                if near:
                    frames.append(t)
            rec = {k: [] for k in ("t", "cl", "cr", "co", "fl", "fr", "fo", "geo_l", "geo_r", "obj_err_cm")}
            for b0 in range(0, len(frames), a.batch):
                fb = frames[b0:b0 + a.batch]
                imgs, vl, vr, boxes_m, geo = [], [], [], [], {"left": [], "right": []}
                for t in fb:
                    img = cv2.cvtColor(cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png")), cv2.COLOR_BGR2RGB)
                    imgs.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
                    m = cv2.imread(str(run / a.obj_dir / f"masks/{o['id']}/{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
                    m = np.zeros((Hh, Ww), np.uint8) if m is None else m
                    boxes_m.append(cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST) > 0)
                    for side, lst in (("left", vl), ("right", vr)):
                        hv = hands.get(side, {}).get(t)
                        if hv is None:  # absent hand: parked far away so it cannot interact
                            hv = np.tile(np.array([0, 0, 5.0]), (778, 1))
                        lst.append(hv.astype(np.float32))
                        Wi = np.linalg.inv(W[t])
                        d, _ = tree.query(hv @ Wi[:3, :3].T + Wi[:3, 3])
                        geo[side].append(d < a.geo_mm / 1000)
                x = torch.tensor(np.stack(imgs), device=dev).permute(0, 3, 1, 2)
                vl_t, vr_t = torch.tensor(np.stack(vl), device=dev), torch.tensor(np.stack(vr), device=dev)
                B = len(fb)
                boxes = extract_2d_bboxes(torch.tensor(np.stack(boxes_m), device=dev)).float()
                Kt = torch.tensor(K, device=dev)[None].expand(B, -1, -1)
                sc_t = torch.full((B,), s, device=dev)
                with torch.no_grad():
                    if a.own_pose:
                        out = model(x, vl_t, vr_t, vl_t.mean(1), vr_t.mean(1), tmpl[None].expand(B, -1, -1), sc_t,
                                    boxes, Kt, e_l, e_r, E_o, Hh, Ww)
                        out = out[:6] + (out[13],)
                    else:  # our fused object pose: camera <- mesh is (R, t); normalised template -> (R, R c + t)
                        Rk = torch.tensor(np.stack([W[t][:3, :3] for t in fb]), dtype=torch.float32, device=dev)
                        Tk = torch.tensor(np.stack([W[t][:3, :3] @ c + W[t][:3, 3] for t in fb]), dtype=torch.float32,
                                          device=dev)
                        out = forward_known_pose(model, x, vl_t, vr_t, vl_t.mean(1), vr_t.mean(1),
                                                 tmpl[None].expand(B, -1, -1), sc_t, boxes, Kt, e_l, e_r, E_o, Hh, Ww,
                                                 Rk, Tk)
                cl, cr, co, fl, fr, fo = (torch.sigmoid(z).reshape(B, -1).cpu().numpy() for z in out[:6])
                vo = out[6].reshape(B, -1, 3).cpu().numpy()
                for k, t in enumerate(fb):
                    ours = V @ W[t][:3, :3].T + W[t][:3, 3]
                    rec["t"].append(t)
                    for key, val in (("cl", cl[k]), ("cr", cr[k]), ("co", co[k]), ("fl", fl[k]), ("fr", fr[k]),
                                     ("fo", fo[k]), ("geo_l", geo["left"][k]), ("geo_r", geo["right"][k])):
                        rec[key].append(val)
                    rec["obj_err_cm"].append(float(np.linalg.norm(vo[k].mean(0) - ours.mean(0)) * 100))
            if not rec["t"]:
                print(f"{r} {nm}: no frames with a hand near the object", flush=True)
                continue
            arr = {k: np.array(v) for k, v in rec.items()}
            np.savez_compressed(outdir / f"{nm}.npz", **arr, obj_verts=V)
            # agreement with our geometric contact
            rows = []
            for side, ck_, gk in (("left", "cl", "geo_l"), ("right", "cr", "geo_r")):
                pe, pg = arr[ck_] > a.thr, arr[gk]
                inter, union = (pe & pg).sum(1), (pe | pg).sum(1)
                has_g, has_e = pg.any(1), pe.any(1)
                rows.append(f"{side}: frames touching ours {has_g.mean():.2f} / EgoPHI {has_e.mean():.2f}, "
                            f"frame agreement {(has_g == has_e).mean():.2f}, vertex IoU when both "
                            f"{np.mean((inter / np.maximum(union, 1))[has_g & has_e]) if (has_g & has_e).any() else 0:.2f}, "
                            f"contact verts ours {pg.sum(1).mean():.0f} / EgoPHI {pe.sum(1).mean():.0f}")
            print(f"{r} {nm}: {len(arr['t'])} frames, EgoPHI object-centre vs our pose {np.median(arr['obj_err_cm']):.1f} cm "
                  f"(median)\n    " + "\n    ".join(rows), flush=True)
            if a.no_video:
                continue
            viz = Path(os.environ["V2D_VIZ"]) / "11_egophi"
            viz.mkdir(parents=True, exist_ok=True)
            import imageio
            meta = json.loads((run / "episode.json").read_text())
            wr = imageio.get_writer(str(viz / f"{r}_{nm}.mp4"), fps=meta["fps"], codec="libx264", quality=8,
                                    macro_block_size=2)
            sc = 1014 / Ww
            for k, t in enumerate(arr["t"]):
                img = cv2.resize(cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png")), (1014, int(Hh * sc)))
                for side, ck_, gk in (("left", "cl", "geo_l"), ("right", "cr", "geo_r")):
                    hv = hands.get(side, {}).get(int(t))
                    if hv is None:
                        continue
                    uv = (hv[:, :2] / hv[:, 2:3] @ K[:2, :2].T + K[:2, 2]) * sc
                    for q, pc, g in zip(uv.astype(int), arr[ck_][k], arr[gk][k]):
                        if g:
                            cv2.circle(img, tuple(q), 4, (60, 220, 60), 1)
                        col = (int(255 * (1 - pc)), int(255 * (1 - pc)), 255) if pc > 0.2 else (200, 200, 200)
                        cv2.circle(img, tuple(q), 2, col, -1)
                Wt = W[int(t)]
                ov = V @ Wt[:3, :3].T + Wt[:3, 3]
                uv = (ov[:, :2] / ov[:, 2:3] @ K[:2, :2].T + K[:2, 2]) * sc
                for q, pc in zip(uv.astype(int), arr["co"][k]):
                    if pc > 0.2:
                        cv2.circle(img, tuple(q), 2, (0, 0, int(80 + 175 * pc)), -1)
                cv2.putText(img, f"{r} f{t} {nm}  EgoPHI contact: red (hand, object)  green ring: ours <{a.geo_mm:.0f}mm",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                wr.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            wr.close()


if __name__ == "__main__":
    main()
