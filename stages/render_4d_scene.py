"""Full 4D scene reconstruction video: hands, objects and the scene in one world frame
(runs in the `agile` image: pytorch3d, manotorch, trimesh, cv2, imageio).

Everything is placed in the fixed SLAM world frame, so the wearer's head motion does not
swing the view:
  scene    the static background as a coloured point cloud, fused from the metric stereo
           depth over the episode (hand and object pixels cut out, flying pixels at depth
           edges dropped, 7 mm voxels);
  hands    the fitted MANO sequence (stereo/mano_fit), skin coloured;
  objects  the scanned meshes, textured, at the chosen trajectory (default fused_all_sam3:
           SAM 3 masks + FoundationPose + stereo-ICP + symmetry + occlusion damping +
           hand-grasp memory);
  truth    on public episodes, each object's mocap pose as a green outline
           (gt_world.json from eval_objects.py), with the per-frame position error printed;
  camera   the wearer's head camera drawn as a frustum with its recent trail.
Panels: [camera view with the reconstruction overlaid | fixed third-person view of the whole
scene | close-up on the hand-object interaction].
Output: $V2D_VIZ/09_scene4d/<run>[_<objects>].mp4
Usage: python render_4d_scene.py <run> [--objects fused_all_sam3] [--hands mano_fit]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import (HardPhongShader, MeshRasterizer, PointLights, PointsRasterizationSettings,
                                PointsRasterizer, RasterizationSettings, TexturesVertex)
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.utils import cameras_from_opencv_projection
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
HAND_RGB = (0.87, 0.70, 0.58)
FILL_RGB = {"geneoh": (0.45, 0.62, 0.95), "haptic": (0.35, 0.80, 0.75), "optimizer": (0.75, 0.55, 0.95)}  # unobserved frames (s06k "filled_by")
GT_RGB = (0.15, 0.95, 0.25)
HAND_IDS = (2, 3)
PW, PH = 760, 570  # panel size


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]  # toolkit Transform3d, w-first
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def decode_depth(path):
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    png = png.astype(np.float32)
    d = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, 0)
    return np.where(np.isfinite(d), d, 0).astype(np.float32)


def read_mask(p, shape):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return np.zeros(shape, bool) if m is None else m > 0


def scene_cloud(run, obj_dir, obj_ids, K, slam, n, frame_step=4, px_step=4, voxel=0.007):
    """Static background: stereo depth back-projected to the world, hands/objects removed."""
    keys, cols = [], []
    ker_h, ker_o = np.ones((31, 31), np.uint8), np.ones((41, 41), np.uint8)
    for t in range(0, n, frame_step):
        if t not in slam:
            continue
        d = decode_depth(run / f"depth_stereo/{t:06d}.png")
        img = cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png"))
        if d is None or img is None:
            continue
        H, W = d.shape
        excl = np.zeros((H, W), np.uint8)
        for hid in HAND_IDS:
            excl |= cv2.dilate(read_mask(run / obj_dir / f"masks/{hid}/{t:06d}.png", (H, W)).astype(np.uint8), ker_h)
        for oid in obj_ids:
            excl |= cv2.dilate(read_mask(run / obj_dir / f"masks/{oid}/{t:06d}.png", (H, W)).astype(np.uint8), ker_o)
        med = cv2.medianBlur(d, 5)
        ok = (d > 0.15) & (d < 3.0) & (excl == 0) & (np.abs(d - med) < 0.03 * np.maximum(d, 1e-3))
        v, u = np.mgrid[0:H:px_step, 0:W:px_step]
        v, u = v.ravel(), u.ravel()
        keep = ok[v, u]
        v, u = v[keep], u[keep]
        z = d[v, u]
        P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        Pw = P @ slam[t][:3, :3].T + slam[t][:3, 3]
        keys.append(np.floor(Pw / voxel).astype(np.int32))
        cols.append(img[v, u, ::-1].astype(np.float32) / 255)
    keys, cols = np.concatenate(keys), np.concatenate(cols)
    uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    col = np.zeros((len(uk), 3), np.float32)
    np.add.at(col, inv, cols)
    col /= cnt[:, None]
    keep = cnt >= 2  # seen at least twice: drops sporadic stereo noise
    return ((uk[keep] + 0.5) * voxel).astype(np.float32), col[keep]


def mesh_colours(m):
    try:
        c = m.visual.to_color().vertex_colors[:, :3].astype(np.float32) / 255
        if len(c) == len(m.vertices) and c.std() > 0.01:
            return c
    except Exception:  # noqa: BLE001 - untextured or unusual visuals
        pass
    return np.tile(np.array([0.6, 0.6, 0.65], np.float32), (len(m.vertices), 1))


class View:
    """One OpenCV camera: world -> view (R, t), intrinsics K, panel size."""

    def __init__(self, K, R, t, dev):
        self.K, self.R, self.t, self.dev = K, R, t, dev
        self.cams = cameras_from_opencv_projection(
            torch.tensor(R, dtype=torch.float32, device=dev)[None], torch.tensor(t, dtype=torch.float32, device=dev)[None],
            torch.tensor(K, dtype=torch.float32, device=dev)[None], torch.tensor([[PH, PW]], dtype=torch.float32, device=dev))

    def project(self, P):
        Q = P @ self.R.T + self.t
        ok = Q[:, 2] > 0.02
        uv = Q[:, :2] / np.maximum(Q[:, 2:3], 1e-6) @ self.K[:2, :2].T + self.K[:2, 2]
        return uv, ok


def draw_meshes(view, parts, dev):
    """(rgb HxWx3 float, zbuf HxW with inf where empty)."""
    rgb = np.ones((PH, PW, 3), np.float32)
    z = np.full((PH, PW), np.inf, np.float32)
    if not parts:
        return rgb, z
    scene = join_meshes_as_scene(parts)
    rast = MeshRasterizer(cameras=view.cams, raster_settings=RasterizationSettings(
        image_size=(PH, PW), faces_per_pixel=1, max_faces_per_bin=300_000))
    frag = rast(scene)
    lights = PointLights(device=dev, location=[[0.0, -1.0, -1.0]], ambient_color=((0.55, 0.55, 0.55),))
    img = HardPhongShader(device=dev, cameras=view.cams, lights=lights)(frag, scene)[0, ..., :3]
    zb = frag.zbuf[0, ..., 0].cpu().numpy()
    hit = zb > 0
    rgb[hit] = img.clamp(0, 1).cpu().numpy()[hit]
    z[hit] = zb[hit]
    return rgb, z


def draw_points(view, pts_t, col_t, radius):
    rgb = np.ones((PH, PW, 3), np.float32)
    z = np.full((PH, PW), np.inf, np.float32)
    rast = PointsRasterizer(cameras=view.cams, raster_settings=PointsRasterizationSettings(
        image_size=(PH, PW), radius=radius, points_per_pixel=1, max_points_per_bin=400_000))
    frag = rast(Pointclouds(points=[pts_t]))
    idx = frag.idx[0, ..., 0].cpu().numpy()
    hit = idx >= 0
    rgb[hit] = col_t[idx[hit]]
    z[hit] = frag.zbuf[0, ..., 0].cpu().numpy()[hit]
    return rgb, z


def look_at(eye, target, up):
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    return R, -R @ eye


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--objects", default="fused_all_sam3", help="trajectory variant dir (<run>/<variant>/<obj>/poses)")
    ap.add_argument("--obj_dir", default="objects_sam3", help="segmentation dir (masks, objects.json)")
    ap.add_argument("--hands", default="mano_fit", help="<run>/stereo/<hands>/<side>/NNNNNN.json")
    ap.add_argument("--no_gt", action="store_true")
    ap.add_argument("--agent_log", default=None, help="refine/agent_log.json: banner on frames the agent fixed")
    ap.add_argument("--label", default=None, help="title for this video (e.g. BEFORE / AFTER agent)")
    ap.add_argument("--out_name", default=None, help="output file name (default <run>[tags].mp4)")
    ap.add_argument("--tracks", default=None, help="TAPIP3D tracks npz (tracks3d/run_tapip3d.py): draw 3D point "
                    "tracks, coloured per object/hand, with trails; hollow = predicted occluded/out of view")
    ap.add_argument("--max_tracks", type=int, default=160, help="tracks drawn per object/hand")
    args = ap.parse_args()
    dev = "cuda"
    run = ROOT / "runs" / args.run
    meta = json.loads((run / "episode.json").read_text())
    ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
    s = PW / ki["width"]
    Kp = K.copy()
    Kp[:2] *= s
    n = len(list((run / "recon_hawor/frames").glob("*.png")))
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}

    objs = []
    for o in json.loads((run / args.obj_dir / "objects.json").read_text()):
        pd = run / args.objects / o["name"] / "poses"
        if not pd.is_dir():
            pd = run / args.obj_dir / o["name"] / "poses_smoothed"
            print(f"{o['name']}: no {args.objects}, falling back to {pd}", flush=True)
        m = trimesh.load(ROOT / "meshes_obj" / o["name"] / f"{o['name']}.obj", force="mesh")
        objs.append({"name": o["name"], "id": o["id"], "dir": pd, "v": np.asarray(m.vertices, np.float32),
                     "f": np.asarray(m.faces), "c": mesh_colours(m)})
    gt = {}
    if not args.no_gt and (run / "gt_world.json").exists():
        g = json.loads((run / "gt_world.json").read_text())
        gt = {nm: {int(t): np.array(M) for t, M in d.items()} for nm, d in g["objects"].items()}
        s3 = g["sim3_world_to_mocap"]["s"]
    print(f"objects: {[o['name'] for o in objs]}  ground truth: {sorted(gt)}", flush=True)

    # per-frame world poses
    Wo = {o["name"]: {} for o in objs}
    for o in objs:
        for t in slam:
            p = o["dir"] / f"{t:06d}.json"
            if p.exists():
                Wo[o["name"]][t] = slam[t] @ load_T(p)
    mano = Mano(str(ROOT / "weights/hand"), dev)

    pts, col = scene_cloud(run, args.obj_dir, [o["id"] for o in objs], K, slam, n)
    print(f"scene cloud: {len(pts)} points", flush=True)
    pts_t, col_np = torch.tensor(pts, device=dev), col

    # world "up" = opposite of the head camera's mean image-down axis; views around the objects
    up = -np.mean([slam[t][:3, 1] for t in slam], 0)
    up /= np.linalg.norm(up)
    cam_c = np.mean([slam[t][:3, 3] for t in slam], 0)
    oc = [W[:3, 3] for d in Wo.values() for W in d.values()]
    obj_c = np.median(oc, 0) if oc else cam_c + 0.5 * slam[min(slam)][:3, 2]
    back = cam_c - obj_c
    back -= up * (back @ up)
    back /= max(np.linalg.norm(back), 1e-6)

    def orbit(deg, dist, lift):
        Rz = Rotation.from_rotvec(np.deg2rad(deg) * up).as_matrix()
        eye = obj_c + Rz @ back * dist + up * lift
        return look_at(eye, obj_c, up)

    d_co = float(np.linalg.norm(cam_c - obj_c))
    R3, t3 = orbit(55, max(1.4 * d_co, 0.9), 0.45)
    Rc, tc = orbit(-35, 0.6, 0.3)
    K3 = np.array([[0.9 * PW, 0, PW / 2], [0, 0.9 * PW, PH / 2], [0, 0, 1]])
    Kc = np.array([[1.1 * PW, 0, PW / 2], [0, 1.1 * PW, PH / 2], [0, 0, 1]])
    v3, vc = View(K3, R3, t3, dev), View(Kc, Rc, tc, dev)

    out = Path(os.environ["V2D_VIZ"]) / "09_scene4d"
    out.mkdir(parents=True, exist_ok=True)
    tag = ("" if args.objects == "fused_all_sam3" else f"_{args.objects}") + ("" if args.hands == "mano_fit" else f"_{args.hands}")
    if args.tracks:
        tag += "_tracks"
    out_file = out / (args.out_name or f"{args.run}{tag}.mp4")
    wr = imageio.get_writer(str(out_file), fps=meta["fps"], codec="libx264", quality=8,
                            macro_block_size=2)
    fixes = {}  # frame -> [(object, action)]
    if args.agent_log and Path(args.agent_log).exists():
        for e in json.loads(Path(args.agent_log).read_text()):
            if e.get("accepted"):
                for u in range(e["window"][0], e["window"][1] + 1):
                    fixes.setdefault(u, []).append((e["object"], e["chosen"]))
    TR = None
    if args.tracks:
        z = np.load(run / args.tracks if not Path(args.tracks).is_absolute() else args.tracks)
        rng = np.random.default_rng(0)
        sel = np.concatenate([rng.permutation(np.where(z["label"] == li)[0])[:args.max_tracks]
                              for li in range(len(z["labels"]))])
        pal = [(255, 80, 60), (60, 140, 255), (255, 200, 0), (180, 80, 255), (0, 200, 200)]
        names = list(z["labels"])
        cols = {li: ((255, 120, 0) if nm == "hand_left" else (255, 0, 170) if nm == "hand_right" else pal[li % len(pal)])
                for li, nm in enumerate(names)}
        TR = {"C": z["coords"][:, sel], "V": z["visibs"][:, sel], "L": z["label"][sel], "Q": z["queries"][sel, 0],
              "names": names, "cols": cols}
    trail = []
    for t in range(n):
        if t not in slam:
            continue
        parts, gparts, errs = [], [], []
        for side in ("left", "right"):
            f = run / "stereo" / args.hands / side / f"{t:06d}.json"
            if f.exists():
                h = json.loads(f.read_text())
                v, _ = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                v = v.cpu().numpy() + np.array(h["T"])
                vw = v @ slam[t][:3, :3].T + slam[t][:3, 3]
                parts.append(Meshes([torch.tensor(vw, dtype=torch.float32, device=dev)],
                                    [torch.tensor(np.ascontiguousarray(mano.faces(h["is_right"])), device=dev)],
                                    textures=TexturesVertex([torch.tensor(FILL_RGB.get(h.get("filled_by"), HAND_RGB), device=dev).expand(len(vw), 3)])))
        for o in objs:
            if t in Wo[o["name"]]:
                W = Wo[o["name"]][t]
                vw = o["v"] @ W[:3, :3].T + W[:3, 3]
                parts.append(Meshes([torch.tensor(vw, dtype=torch.float32, device=dev)], [torch.tensor(o["f"], device=dev)],
                                    textures=TexturesVertex([torch.tensor(o["c"], device=dev)])))
            G = gt.get(o["name"], {}).get(t)
            if G is not None:
                vg = o["v"] @ G[:3, :3].T + G[:3, 3]
                gparts.append(Meshes([torch.tensor(vg, dtype=torch.float32, device=dev)], [torch.tensor(o["f"], device=dev)],
                                     textures=TexturesVertex([torch.tensor(GT_RGB, device=dev).expand(len(vg), 3)])))
                if t in Wo[o["name"]]:
                    errs.append((o["name"], s3 * float(np.linalg.norm(Wo[o["name"]][t][:3, 3] - G[:3, 3])) * 100))
        cam_R = slam[t][:3, :3].T
        vcam = View(Kp, cam_R, -cam_R @ slam[t][:3, 3], dev)
        trail.append(slam[t][:3, 3].copy())
        trail = trail[-60:]

        panels = []
        for k, view in enumerate((vcam, v3, vc)):
            m_rgb, m_z = draw_meshes(view, parts, dev)
            if k == 0:
                bg = cv2.cvtColor(cv2.resize(cv2.imread(str(run / f"recon_hawor/frames/{t:06d}.png")), (PW, PH)),
                                  cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                hit = np.isfinite(m_z)[..., None]
                img = np.where(hit, 0.35 * bg + 0.65 * m_rgb, bg)
            else:
                p_rgb, p_z = draw_points(view, pts_t, col_np, 0.006 if k == 1 else 0.012)
                bg = np.full((PH, PW, 3), 0.93, np.float32)
                img = np.where((m_z < p_z)[..., None], m_rgb, np.where(np.isfinite(p_z)[..., None], p_rgb, bg))
            img = to_u8(img)
            if gparts:  # mocap truth as a green outline, so the reconstruction stays visible
                _, g_z = draw_meshes(view, gparts, dev)
                gm = np.isfinite(g_z).astype(np.uint8)
                cnt, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                img = np.ascontiguousarray(img)
                cv2.drawContours(img, cnt, -1, (40, 230, 60), 2, cv2.LINE_AA)
            if k == 1:  # head camera: frustum + trail
                c0 = slam[t][:3, 3]
                fr = np.array([[-1, -0.75, 1.2], [1, -0.75, 1.2], [1, 0.75, 1.2], [-1, 0.75, 1.2]]) * 0.06
                corners = fr @ slam[t][:3, :3].T + c0
                uv, ok = view.project(np.vstack([c0, corners, np.array(trail)]))
                if ok[:5].all():
                    q = uv.astype(int)
                    for i in range(1, 5):
                        cv2.line(img, tuple(q[0]), tuple(q[i]), (230, 60, 60), 2, cv2.LINE_AA)
                        cv2.line(img, tuple(q[i]), tuple(q[1 + i % 4]), (230, 60, 60), 2, cv2.LINE_AA)
                    tr = q[5:][ok[5:]]
                    if len(tr) > 1:
                        cv2.polylines(img, [tr.reshape(-1, 1, 2)], False, (230, 120, 60), 1, cv2.LINE_AA)
            if TR is not None and t < len(TR["C"]):  # 3D point tracks with an 8-frame trail
                img = np.ascontiguousarray(img)
                a0 = max(0, t - 8)
                for j in range(TR["C"].shape[1]):
                    c = TR["cols"][int(TR["L"][j])]
                    uv, ok = view.project(TR["C"][a0:t + 1, j].astype(np.float64))
                    if not ok[-1] or not (0 <= uv[-1, 0] < PW and 0 <= uv[-1, 1] < PH):
                        continue
                    q = uv.astype(int)
                    vis = TR["V"][a0:t + 1, j]
                    for i in range(1, len(q)):
                        if ok[i] and ok[i - 1] and vis[i] and vis[i - 1]:
                            cv2.line(img, tuple(q[i - 1]), tuple(q[i]), c, 1, cv2.LINE_AA)
                    if vis[-1]:
                        cv2.circle(img, tuple(q[-1]), 3, c, -1, cv2.LINE_AA)
                    else:
                        cv2.circle(img, tuple(q[-1]), 3, c, 1, cv2.LINE_AA)
            panels.append(img)
        frame = np.hstack(panels)
        if TR is not None:
            for i, nm in enumerate(TR["names"]):
                nv = int(TR["V"][min(t, len(TR["V"]) - 1)][TR["L"] == i].sum())
                cv2.putText(frame, f"{nm}: {nv} visible tracks", (10, 84 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            TR["cols"][i], 2)
            cv2.putText(frame, "dots = TAPIP3D 3D tracks (hollow: predicted occluded / out of view)", (10, PH - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"{args.run}  frame {t}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        for k, lab in enumerate(("camera view", "whole scene (world frame)", "close-up")):
            cv2.putText(frame, lab, (k * PW + 10, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        cv2.putText(frame, f"objects: {args.objects}   hands: {args.hands}" + ("   green outline = mocap truth" if gt else ""),
                    (PW + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        if any((run / "stereo" / args.hands / s / f"{t:06d}.json").exists() and json.loads((run / "stereo" / args.hands / s / f"{t:06d}.json").read_text()).get("filled_by") for s in ("left", "right")):
            cv2.putText(frame, "hand not detected -> teal: HaPTIC  blue: GeneOH  violet: optimiser", (PW + 10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 60, 40), 1)
        if args.label:
            cv2.putText(frame, args.label, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.putText(frame, args.label, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 200), 2)
        if t in fixes:
            txt = "AGENT FIX: " + ", ".join(f"{o} ({act})" for o, act in fixes[t])
            cv2.rectangle(frame, (0, PH - 44), (frame.shape[1], PH - 14), (0, 0, 180), -1)
            cv2.putText(frame, txt, (PW + 10, PH - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        for i, (nm, e) in enumerate(errs):
            cv2.putText(frame, f"{nm}: {e:4.1f} cm", (2 * PW + 10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (20, 140, 20) if e < 3 else (200, 40, 40), 2)
        wr.append_data(frame)
    wr.close()
    print(f"wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
