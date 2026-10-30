"""Stage 6h: physically plausible grasps with contacts at the observed places (agile image).

Starts from the stereo MANO fit and makes bounded corrections against one fixed object
trajectory version. Per-frame corrections can remove input jitter; robust world acceleration
regularizes the resulting motion using source timing. Adam optimizes:

  anchor    contacts at accurate, fixed places. For every grasp (a contiguous run of frames in
            which one hand holds one object, fusion_report hand_contact) and every fingertip that
            is near the surface in a good part of it, the contact location is the median of that
            fingertip's observed position in the OBJECT's frame, projected onto the surface.
            A fingertip on a held object does not slide, so it is pulled to that one point for the
            observed sticking interval -- slightly inside the surface (--squeeze_mm along the inward normal), the
            small squeeze a soft-contact solver (MuJoCo) needs to produce grip at all.
  pen       no hand vertex inside any object (signed distance field at the object's pose); frames
            whose initial penetration exceeds --pen_cap_mm (object pose suspect) are down-weighted
  table     no hand vertex inside the tabletop: the disk slab of table_plane.py (center/radius/thickness;
            hands beside or under the round table are free), else the infinite plane (legacy)
  data      joints stay close to the stereo fit (the observation)
  prior     small corrections; wrist translation and rotation corrections cost more
  smooth    world-frame joint acceleration
Output stereo/<out>/<side>/NNNNNN.json (same schema) + report.json with the anchors
(object-frame contact points per grasp and finger), usable as the contact reference.
Usage: python s06h_grasp_plausible.py <run> [--hands mano_fit] [--out mano_fit_plaus]
Final pass after s06i (corrections, trust limits and anchors measured against the stereo fit, starting
from the selected candidates, each grasp's anchors with its selected squeeze/contact distance):
       python s06h_grasp_plausible.py <run> --ref_hands mano_fit --hands mano_fit_selected \
              --segment_params mano_fit_selected --out mano_fit_phys
Quality (report.json quality_passed): per-frame max penetration over frames whose object pose is not
suspect (deep); table: slab depth over frames with a vertex over the disk (legacy plane: only on
contact frames / near an object footprint); passes when
the 95th percentile <= 5 mm and <= 5% of frames exceed 10 mm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano, SDFGrid, rotvec_to_matrix  # noqa: E402
from s06f_contact_refine import TIPS, load_T, mano_forward, matrix_to_rotvec  # noqa: E402
from motion_quality import acceleration_loss, sticking_windows, spline_projectors, spline_loss, table_disk_inside, table_slab_depth

ROOT = Path(os.environ["V2D_ROOT"])
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def segments(frames, max_gap=1):
    """Contiguous runs (fusion_report hand_contact is contiguous per grasp, see fuse_objects)."""
    frames = sorted(frames)
    out, cur = [], []
    for t in frames:
        if cur and t - cur[-1] > max_gap:
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def load_segment_params(run, sel_dir, a):
    """Per-grasp anchor parameters chosen by s06i: (default, [(side, object, first, last, params)])."""
    sel = json.loads((run / "stereo" / sel_dir / "report.json").read_text())
    cache = {}

    def cand_args(c):
        if c not in cache:
            args = json.loads((run / "stereo" / c / "report.json").read_text())["args"]
            cache[c] = {k: float(args[k]) for k in ("squeeze_mm", "contact_mm")}
        return cache[c]

    rows = [(s["side"], s["object"], s["frames"][0], s["frames"][-1], s.get("chosen_params") or cand_args(s["chosen"]))
            for s in sel["segments"]]
    return sel.get("default_params") or cand_args(sel["cands"][0]), rows


def segment_anchor_params(seg_params, side, name, seg, a):
    """squeeze_mm/contact_mm for one s06h grasp segment: the s06i segment it overlaps most."""
    if seg_params is None:
        return a.squeeze_mm, a.contact_mm
    default, rows = seg_params
    best, best_n = default, 0
    for sd, nm, f0, f1, p in rows:
        k = sum(f0 <= t <= f1 for t in seg) if (sd, nm) == (side, name) else 0
        if k > best_n:
            best, best_n = p, k
    return best["squeeze_mm"], best["contact_mm"]


def robust_stats(per_frame_mm, a):
    """Frame-level penetration statistic: robust to a few frames with suspect object poses."""
    x = np.asarray(per_frame_mm, float)
    if len(x) == 0:
        return {"frames": 0, "p95_mm": 0.0, "max_mm": 0.0, "frames_over_5mm": 0, "frames_over_10mm": 0,
                "frac_over_10mm": 0.0, "passed": True}
    frac = float(np.mean(x > 10.0))
    p95 = float(np.percentile(x, 95))
    return {"frames": int(len(x)), "p95_mm": round(p95, 3), "max_mm": round(float(x.max()), 3),
            "frames_over_5mm": int((x > 5.0).sum()), "frames_over_10mm": int((x > 10.0).sum()),
            "frac_over_10mm": round(frac, 4),
            "passed": bool(p95 <= a.pass_p95_mm and frac <= a.pass_frac_over_10mm)}


def table_quality(vw, plane, near, contact, a):
    """Per-frame max tabletop penetration. Disk (table_plane.py center/radius): the slab depth, over the
    frames with a vertex over the disk; legacy infinite plane: below-plane depth, only on contact frames
    or near an object's footprint (a hand lowered beside the table is not a penetration)."""
    import torch
    plane_depth = torch.relu(float(plane["height"]) - vw @ torch.as_tensor(plane["up"], dtype=vw.dtype, device=vw.device))
    if "radius" in plane:
        tmask = table_disk_inside(vw, plane) | contact[:, None]
        tab = table_slab_depth(vw, plane, side="auto").amax(1)
        st = robust_stats((tab[tmask.any(1)] * 1000).cpu().numpy(), a)
        st["model"] = "disk_slab"
    else:
        tmask = near | contact[:, None]
        tab = (plane_depth * tmask).amax(1)
        st = robust_stats((tab[tmask.any(1)] * 1000).cpu().numpy(), a)
        st["model"] = "infinite_plane"
    st["max_mm_unmasked"] = round(float(plane_depth.max()) * 1000, 3)  # infinite plane, for reference
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", default="mano_fit")
    ap.add_argument("--ref_hands", default=None,
                    help="final pass: measure the data term, prior and trust limits against these hands (the "
                         "original stereo fit, e.g. mano_fit) and start from --hands (e.g. mano_fit_selected)")
    ap.add_argument("--segment_params", default=None,
                    help="s06i output dir (e.g. mano_fit_selected): reuse each grasp's chosen candidate "
                         "squeeze_mm/contact_mm for its anchors; other grasps use the gentlest candidate's")
    ap.add_argument("--out", default="mano_fit_plaus")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--knot_step", type=int, default=1,
                    help="Correction spacing; 1 can remove frame jitter, with world acceleration regularization")
    ap.add_argument("--max_shift_mm", type=float, default=25.0)
    ap.add_argument("--max_rotation_deg", type=float, default=20.0)
    ap.add_argument("--max_finger_deg", type=float, default=20.0)
    ap.add_argument("--max_slip_mm", type=float, default=10.0)
    ap.add_argument("--smoothing", choices=("spline", "acceleration"), default="spline")
    ap.add_argument("--spline_knot_seconds", type=float, default=0.15)
    ap.add_argument("--w_spline", type=float, default=3000.0)
    ap.add_argument("--contact_mm", type=float, default=15.0, help="a fingertip this close counts as touching")
    ap.add_argument("--min_touch", type=float, default=0.3, help="fraction of a grasp a fingertip must touch")
    ap.add_argument("--squeeze_mm", type=float, default=1.5)
    ap.add_argument("--vert_step", type=int, default=6, help="also anchor every Nth MANO vertex found in contact")
    ap.add_argument("--pen_cap_mm", type=float, default=25.0)
    ap.add_argument("--table_margin_m", type=float, default=0.05,
                    help="quality check: table penetration counts on contact frames or within this "
                         "horizontal distance of an object's extent")
    ap.add_argument("--pass_p95_mm", type=float, default=5.0)
    ap.add_argument("--pass_frac_over_10mm", type=float, default=0.05)
    ap.add_argument("--w_anchor", type=float, default=3e3)
    ap.add_argument("--w_pen", type=float, default=2e4)
    ap.add_argument("--w_table", type=float, default=2e4)
    ap.add_argument("--w_data", type=float, default=1.5e3)
    ap.add_argument("--w_prior", type=float, default=1.0)
    ap.add_argument("--w_rot", type=float, default=50.0)
    ap.add_argument("--w_trans", type=float, default=2e4)
    ap.add_argument("--w_smooth", type=float, default=5e3)
    a = ap.parse_args()
    if a.knot_step < 1 or min(a.max_shift_mm, a.max_rotation_deg, a.max_finger_deg, a.max_slip_mm) <= 0:
        raise ValueError("Correction limits and knot spacing must be positive")
    if not 0 <= a.squeeze_mm <= 5:
        raise ValueError("Squeeze must be between 0 and 5 mm; do not repair bad poses by deep penetration")
    dev = a.device
    run = ROOT / "runs" / a.run
    fps = float(json.loads((run / "episode.json").read_text())["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source fps")
    if a.out in (a.hands, a.ref_hands):
        raise ValueError("Refinement output must differ from its input")
    seg_params = load_segment_params(run, a.segment_params, a) if a.segment_params else None
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    up = -np.mean([slam[t][:3, 1] for t in slam], 0)
    up /= np.linalg.norm(up)
    mano = Mano(str(ROOT / "weights/hand"), dev)
    rep = json.loads((run / "fusion_report.json").read_text())

    objs, lows = [], []
    for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
        nm = o["name"]
        pd = run / a.variant / nm / "poses"
        W = {t: slam[t] @ load_T(pd / f"{t:06d}.json") for t in slam if (pd / f"{t:06d}.json").exists()}
        m = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm / f"{nm}.obj", force="mesh")
        touched = set().union(*[set(v) for v in rep.get(nm, {}).get("hand_contact", {}).values()]) \
            if rep.get(nm, {}).get("hand_contact") else set()
        rest = [t for t in W if t not in touched] or list(W)
        lows += [((m.vertices @ W[t][:3, :3].T + W[t][:3, 3]) @ up).min() for t in rest[::5]]
        c = np.asarray(m.vertices).mean(0)
        objs.append({"name": nm, "W": W, "sdf": SDFGrid(m, res=160, device=dev), "center": c,
                     "radius": float(np.linalg.norm(np.asarray(m.vertices) - c, axis=1).max()),
                     "grasp": rep.get(nm, {}).get("hand_contact", {})})
    table_h = float(np.median(lows)) if lows else None
    tp = run / "table_plane.json"
    if tp.exists():  # measured support plane: true gravity direction and table height
        d = json.loads(tp.read_text())
        up, table_h = np.array(d["up"]), float(d["height"])
    up_t = torch.tensor(up, dtype=torch.float32, device=dev)
    # round top (table_plane.py center/radius/thickness): only the tabletop slab inside the disk is solid;
    # without the disk keys the legacy infinite plane
    plane = d if tp.exists() else ({"up": up.tolist(), "height": table_h} if table_h is not None else None)
    disk = plane is not None and "radius" in plane

    outdir = run / "stereo" / a.out
    # An interrupted rebuild must not leave a previous passing report attached
    # to a mixture of old and new hand records.
    if (outdir / "report.json").exists():
        (outdir / "report.json").unlink()
    report = {"args": vars(a), "table_height": table_h, "sides": {}}
    for side in ("left", "right"):
        for old in (outdir / side).glob("*.json"):
            old.unlink()
        def read(hands):
            return [(int(f.stem), json.loads(f.read_text())) for f in sorted((run / "stereo" / hands / side).glob("*.json"))
                    if int(f.stem) in slam]
        # recs: the reference the corrections are measured against (data term, prior, trust limits,
        # anchors); init: where the optimization starts (the same unless --ref_hands is given)
        recs = read(a.ref_hands or a.hands)
        init = read(a.hands) if a.ref_hands else recs
        if [t for t, _ in init] != [t for t, _ in recs]:
            raise ValueError(f"{side}: {a.hands} and {a.ref_hands} cover different frames")
        if len(recs) < 3:
            continue
        ts = [t for t, _ in recs]
        phases = [tuple(o["name"] for o in objs if t in o["grasp"].get(side, [])) for t in ts]
        projectors = (spline_projectors(ts, fps, phases, a.spline_knot_seconds)
                      if a.smoothing == "spline" else [])
        tix = {t: i for i, t in enumerate(ts)}
        is_right = recs[0][1]["is_right"]
        go0 = torch.tensor([r["mano"]["global_orient"] for _, r in recs], dtype=torch.float32, device=dev)
        hp0 = torch.tensor([r["mano"]["hand_pose"] for _, r in recs], dtype=torch.float32, device=dev)
        be = torch.tensor([r["mano"]["betas"] for _, r in recs], dtype=torch.float32, device=dev)
        T0 = torch.tensor([r["T"] for _, r in recs], dtype=torch.float32, device=dev)
        Sw = torch.tensor(np.stack([slam[t] for t in ts]), dtype=torch.float32, device=dev)
        for o in objs:
            M = np.stack([np.linalg.inv(o["W"][t]) if t in o["W"] else np.full((4, 4), np.nan) for t in ts])
            o["valid"] = torch.tensor(np.isfinite(M[:, 0, 0]), device=dev)
            o["Wi"] = torch.tensor(np.nan_to_num(M), dtype=torch.float32, device=dev)
            cw = np.stack([o["W"][t][:3, :3] @ o["center"] + o["W"][t][:3, 3] if t in o["W"] else np.zeros(3) for t in ts])
            o["cw"] = torch.tensor(cw, dtype=torch.float32, device=dev)

        def world_verts(dR, dhp, dT):
            R = rotvec_to_matrix(dR) @ rotvec_to_matrix(go0)
            v, j = mano_forward(mano, matrix_to_rotvec(R), hp0 + dhp, be, is_right)
            v, j = v + (T0 + dT)[:, None], j + (T0 + dT)[:, None]
            vw = torch.einsum("bij,bnj->bni", Sw[:, :3, :3], v) + Sw[:, None, :3, 3]
            return vw, j

        def to_local(vw, o):
            return torch.einsum("bij,bnj->bni", o["Wi"][:, :3, :3], vw) + o["Wi"][:, None, :3, 3]

        with torch.no_grad():
            z = torch.zeros_like(go0)
            vw0, j0 = world_verts(z, torch.zeros_like(hp0), torch.zeros_like(T0))
        # ---- contact anchors: fixed, observed, surface-projected, squeezed
        anchors, arows = [], []  # (frame index list, tip vertex, object, target point in object frame)
        weight = {}
        for o in objs:
            loc0 = to_local(vw0, o)
            d0 = o["sdf"](loc0.reshape(-1, 3)).view(len(ts), -1)
            deep = torch.relu(-d0).amax(1) > a.pen_cap_mm / 1000
            o["deep"] = deep & o["valid"]  # object pose suspect: excluded from the quality statistic
            weight[o["name"]] = torch.where(deep, 0.1, 1.0) * o["valid"].float()
            for seg in segments([t for t in o["grasp"].get(side, []) if t in tix and t in o["W"]]):
                idx = [tix[t] for t in seg]
                squeeze_mm, contact_mm = segment_anchor_params(seg_params, side, o["name"], seg, a)
                # every hand region that is in contact, not just the fingertips: a pot held by its
                # handle is held by the palm and the finger segments
                cand_v = list(dict.fromkeys(TIPS + list(range(0, 778, a.vert_step))))
                for k, v in enumerate(cand_v):
                    dd = d0[idx, v].abs().cpu().numpy()
                    windows = sticking_windows(seg, loc0[idx, v].cpu().numpy(), dd,
                                               contact_m=contact_mm / 1000, slip_m=a.max_slip_mm / 1000)
                    if sum(map(len, windows)) < a.min_touch * len(idx):
                        continue
                    for win in windows:
                        near = [idx[i] for i in win]
                        p = loc0[near, v].median(0).values.detach().clone().requires_grad_(True)
                        sd = o["sdf"](p[None])
                        (g,) = torch.autograd.grad(sd.sum(), p)
                        n = g / g.norm().clamp(min=1e-9)
                        with torch.no_grad():
                            target = p - n * (sd + squeeze_mm / 1000)
                        anchors.append([near, v, o, target.detach(), 1.0, seg[0]])
                        arows.append({"object": o["name"], "vertex": int(v),
                                      "finger": FINGERS[k] if k < len(TIPS) else "surface", "frames": [ts[near[0]], ts[near[-1]]],
                                      "point_object_frame": target.cpu().numpy().round(5).tolist(),
                                      "touching_frames": len(near), "grasp_start": int(seg[0]),
                                      "squeeze_mm": squeeze_mm, "contact_mm": contact_mm})

        # each grasp carries about five fingertips' worth of anchor weight, however many regions touch
        # (grouped by the grasp segment, not by the sticking window: a grasp has many windows)
        per_seg = {}
        for an in anchors:
            per_seg.setdefault((id(an[2]), an[5]), []).append(an)
        for group in per_seg.values():
            n_regions = len({an[1] for an in group})
            for an in group:
                an[4] = min(1.0, 5.0 / n_regions)
        # ---- smooth corrections on control points
        idxf = torch.tensor(ts, dtype=torch.float32, device=dev)
        knots = torch.arange(float(ts[0]), float(ts[-1]) + a.knot_step, float(a.knot_step), device=dev)
        B = torch.clamp(1 - (idxf[:, None] - knots[None]).abs() / a.knot_step, min=0)
        B = B / B.sum(1, keepdim=True)
        def bounded(x, limit):
            # Smooth radial bound; the whole correction vector stays inside the limit.
            return limit * x / torch.sqrt(1 + (x * x).sum(-1, keepdim=True))

        def unbounded(y, limit):
            # Inverse of bounded(); corrections at/over the limit are pulled just inside it.
            nrm = y.norm(dim=-1, keepdim=True)
            y = y * torch.clamp(0.999 * limit / nrm.clamp(min=1e-12), max=1.0)
            return y / torch.sqrt(limit ** 2 - (y * y).sum(-1, keepdim=True))

        kR = torch.zeros(B.shape[1], 3, device=dev)
        khp = torch.zeros(B.shape[1], 45, device=dev)
        kT = torch.zeros(B.shape[1], 3, device=dev)
        if init is not recs:  # start at --hands, corrections measured from --ref_hands
            go1 = torch.tensor([r["mano"]["global_orient"] for _, r in init], dtype=torch.float32, device=dev)
            hp1 = torch.tensor([r["mano"]["hand_pose"] for _, r in init], dtype=torch.float32, device=dev)
            T1 = torch.tensor([r["T"] for _, r in init], dtype=torch.float32, device=dev)
            dR1 = matrix_to_rotvec(rotvec_to_matrix(go1) @ rotvec_to_matrix(go0).transpose(-1, -2))
            Bp = torch.linalg.pinv(B)
            kR = Bp @ unbounded(dR1, np.deg2rad(a.max_rotation_deg))
            khp = Bp @ unbounded((hp1 - hp0).reshape(-1, 15, 3), np.deg2rad(a.max_finger_deg)).reshape(-1, 45)
            kT = Bp @ unbounded(T1 - T0, a.max_shift_mm / 1000)
        kR, khp, kT = (x.detach().clone().requires_grad_(True) for x in (kR, khp, kT))

        def corrections():
            return (bounded(B @ kR, np.deg2rad(a.max_rotation_deg)),
                    bounded((B @ khp).reshape(-1, 15, 3), np.deg2rad(a.max_finger_deg)).reshape(-1, 45),
                    bounded(B @ kT, a.max_shift_mm / 1000))

        opt = torch.optim.Adam([kR, khp, kT], lr=a.lr)
        for it in range(a.iters):
            dR, dhp, dT = corrections()
            vw, j = world_verts(dR, dhp, dT)
            l_pen = torch.zeros((), device=dev)
            for o in objs:
                d = o["sdf"](to_local(vw, o).reshape(-1, 3)).view(len(ts), -1)
                l_pen = l_pen + (torch.relu(-d) ** 2 * weight[o["name"]][:, None]).sum() / len(ts)
            l_anc = torch.zeros((), device=dev)
            for idx, v, o, target, wa, _ in anchors:
                loc = to_local(vw[idx][:, [v]], {"Wi": o["Wi"][idx]})[:, 0]
                # Ease into/out of a contact; never pull a released finger back for
                # the remainder of an object-level grasp segment.
                ramp = torch.arange(1, len(idx) + 1, device=dev, dtype=loc.dtype)
                ramp = torch.minimum(ramp, ramp.flip(0)).clamp(max=3) / 3
                l_anc = l_anc + wa * (((loc - target) ** 2).sum(-1) * ramp).sum() / len(ts)
            l_tab = torch.zeros((), device=dev)
            if table_h is not None:
                l_tab = (table_slab_depth(vw, plane, side="auto") ** 2).sum() / len(ts)
            l_data = ((j - j0) ** 2).sum(-1).mean()
            l_prior = (dhp ** 2).mean() + a.w_rot * (dR ** 2).mean() + a.w_trans * (dT ** 2).sum(-1).mean()
            jw = torch.einsum("bij,bnj->bni", Sw[:, :3, :3], j) + Sw[:, None, :3, 3]
            l_smooth = acceleration_loss(jw, ts, fps)
            loss = (a.w_pen * l_pen + a.w_anchor * l_anc + a.w_table * l_tab + a.w_data * l_data
                    + a.w_prior * l_prior + a.w_smooth * l_smooth + a.w_spline * spline_loss(jw, projectors))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite refinement loss; refusing to write hand motion")
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 150 == 0 or it == a.iters - 1:
                print(f"{side} it {it:4d} pen {l_pen.item():.2e} anchor {l_anc.item():.2e} table {l_tab.item():.2e} "
                      f"data {1000 * l_data.sqrt().item():.1f} mm", flush=True)

        with torch.no_grad():
            dR, dhp, dT = corrections()
            vw, j = world_verts(dR, dhp, dT)
            go = matrix_to_rotvec(rotvec_to_matrix(dR) @ rotvec_to_matrix(go0)).cpu().numpy()
            hp, T, jc = (hp0 + dhp).cpu().numpy(), (T0 + dT).cpu().numpy(), j.cpu().numpy()
            res = []
            for idx, v, o, target, *_ in anchors:
                loc = to_local(vw[idx][:, [v]], {"Wi": o["Wi"][idx]})[:, 0]
                res.append(float((loc - target).norm(dim=-1).mean()) * 1000)
            moved = float(((j - j0) ** 2).sum(-1).sqrt().mean()) * 1000
            # Object penetration per frame, over objects whose pose is valid and not suspect (deep);
            # table penetration only on this hand's contact frames or near an object's footprint
            # (the plane is infinite: a hand lowered beside the table is not a penetration).
            obj_pen = torch.zeros(len(ts), device=dev)
            obj_checked = torch.zeros(len(ts), dtype=torch.bool, device=dev)
            deep_any = torch.zeros(len(ts), dtype=torch.bool, device=dev)
            near = torch.zeros(vw.shape[:2], dtype=torch.bool, device=dev)
            contact = torch.zeros(len(ts), dtype=torch.bool, device=dev)
            for o in objs:
                use = o["valid"] & ~o["deep"]
                deep_any |= o["deep"]
                d = o["sdf"](to_local(vw, o).reshape(-1, 3)).view(len(ts), -1)
                obj_pen = torch.where(use, torch.maximum(obj_pen, torch.relu(-d).amax(1)), obj_pen)
                obj_checked |= use
                rel = vw - o["cw"][:, None]
                horiz = (rel - (rel @ up_t)[..., None] * up_t).norm(dim=-1)
                near |= (horiz <= o["radius"] + a.table_margin_m) & o["valid"][:, None]
                held = set(o["grasp"].get(side, []))
                contact |= torch.tensor([t in held for t in ts], dtype=torch.bool, device=dev)
            obj_stats = robust_stats((obj_pen[obj_checked] * 1000).cpu().numpy(), a)
            obj_stats["frames_excluded_deep"] = int(deep_any.sum())
            if table_h is not None:
                tab_stats = table_quality(vw, plane, near, contact, a)
            else:
                tab_stats = None
            finite = bool(np.isfinite(go).all() and np.isfinite(hp).all() and np.isfinite(T).all() and np.isfinite(jc).all())
            quality = {"finite": finite, "max_object_penetration_mm": obj_stats["max_mm"],
                       "max_table_penetration_mm": tab_stats["max_mm"] if tab_stats else None,
                       "object": obj_stats, "table": tab_stats,
                       "criterion": f"per-frame max penetration: p95 <= {a.pass_p95_mm} mm and <= "
                                    f"{100 * a.pass_frac_over_10mm:g}% of frames > 10 mm (deep frames excluded)",
                       "passed": bool(finite and obj_stats["passed"] and tab_stats is not None and tab_stats["passed"])}
            print(f"{side} quality: object {obj_stats} table {tab_stats} -> passed {quality['passed']}", flush=True)
            if not finite:
                raise ValueError("Non-finite hand geometry; refusing output")
        (outdir / side).mkdir(parents=True, exist_ok=True)
        for k, (t, r) in enumerate(recs):
            r2 = dict(r)
            r2["mano"] = {"betas": r["mano"]["betas"], "global_orient": go[k].tolist(), "hand_pose": hp[k].tolist()}
            r2["T"], r2["joints"] = T[k].tolist(), jc[k].tolist()
            (outdir / side / f"{t:06d}.json").write_text(json.dumps(r2))
        for row, e in zip(arows, res):
            row["residual_mm"] = round(e, 2)
        report["sides"][side] = {"frames": len(ts), "anchors": arows, "mean_joint_shift_mm": moved,
                                 "anchor_residual_mm": float(np.mean(res)) if res else None, "quality": quality}
        print(f"{side}: {len(anchors)} contact anchors, residual {np.mean(res) if res else 0:.1f} mm, "
              f"mean joint shift {moved:.1f} mm", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    report["quality_passed"] = bool(report["sides"]) and all(s["quality"]["passed"] for s in report["sides"].values())
    report["quality_summary"] = {sd: {k: s["quality"][k] for k in ("passed", "object", "table")}
                                 for sd, s in report["sides"].items()}
    (outdir / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
