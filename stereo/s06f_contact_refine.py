"""Stage 6f: physically plausible hands -- remove hand-object penetration, keep contact
(agile image: torch, manotorch, pytorch3d).

Why. The policy stage (FlashCHORD / CHORD) learns from contact wrenches: where the hand
touches the object and how that contact can push it. A finger reconstructed 1-2 cm INSIDE
the object gives a wrong contact location and normal, i.e. a wrench the robot cannot
produce (its fingers stop at the surface). The audit (audit_contact.py) finds exactly that:
5-24 mm deepest penetration in most grasp frames, while contact gaps are rare.

What. Only the HANDS move; object trajectories are what gets scored and stay fixed. Per hand
and frame, corrections (dR 3, dpose 45, dT 3) to the stereo MANO fit (stereo/<hands>) are
optimised jointly over the whole episode with Adam:
  pen      sum over hand vertices of relu(-sdf)^2 in every object's signed distance field
           (agile/geometry.SDFGrid at the object's pose): pushes vertices out along the normal
  contact  grasp frames only: fingertips that were within --contact_mm of the surface stay on
           it (|sdf| -> 0), so removing penetration does not open gaps
  data     fitted joints stay close to the stereo fit (the observation)
  prior    small pose corrections (natural hand shape)
  smooth   joint acceleration over time
Output: stereo/<hands>_refined/<side>/NNNNNN.json (same schema as the input) + report.json.
Usage: python s06f_contact_refine.py <run> [--variant fused_all_sam3] [--hands mano_fit]
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
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano, SDFGrid, rotvec_to_matrix  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
TIPS = [744, 320, 443, 554, 671]


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


def mano_forward(mano, go, hp, betas, is_right):
    """Differentiable MANO (Mano.local is no-grad): verts, joints (B,778,3),(B,21,3), local frame."""
    out = mano.layer(torch.cat([go, hp], 1), betas)
    v, j = out.verts, out.joints
    if not is_right:
        flip = torch.tensor([-1.0, 1.0, 1.0], device=v.device)
        v, j = v * flip, j * flip
    return v, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--hands", default="mano_fit")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w_pen", type=float, default=2e4, help="per m^2 of penetration")
    ap.add_argument("--w_contact", type=float, default=5e3)
    ap.add_argument("--w_data", type=float, default=2e3, help="per m^2 of joint displacement")
    ap.add_argument("--w_prior", type=float, default=1.0)
    ap.add_argument("--w_smooth", type=float, default=5e3)
    ap.add_argument("--contact_mm", type=float, default=15.0)
    ap.add_argument("--knot_step", type=int, default=8, help="frames between correction control points (0 = per frame)")
    ap.add_argument("--w_rot", type=float, default=50.0, help="extra prior on wrist-rotation corrections")
    ap.add_argument("--w_trans", type=float, default=2e4, help="prior on wrist translation corrections, per m^2")
    ap.add_argument("--pen_cap_mm", type=float, default=25.0,
                    help="deeper initial penetration = the OBJECT pose is wrong there: that frame is left out")
    ap.add_argument("--out", default=None, help="output dir name in stereo/ (default <hands>_refined)")
    a = ap.parse_args()
    dev = "cuda"
    run = ROOT / "runs" / a.run
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    mano = Mano(str(ROOT / "weights/hand"), dev)
    rep = json.loads((run / "fusion_report.json").read_text())

    objs = []
    for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
        nm = o["name"]
        pd = run / a.variant / nm / "poses"
        W = {t: slam[t] @ load_T(pd / f"{t:06d}.json") for t in slam if (pd / f"{t:06d}.json").exists()}
        sdf = SDFGrid(trimesh.load(ROOT / "meshes_obj" / nm / f"{nm}.obj", force="mesh"), res=160, device=dev)
        objs.append({"name": nm, "W": W, "sdf": sdf, "grasp": rep.get(nm, {}).get("hand_contact", {})})

    outdir = run / "stereo" / (a.out or f"{a.hands}_refined")
    report = {"args": vars(a), "sides": {}}
    for side in ("left", "right"):
        files = sorted((run / "stereo" / a.hands / side).glob("*.json"))
        recs = [(int(f.stem), json.loads(f.read_text())) for f in files if int(f.stem) in slam]
        if len(recs) < 3:
            continue
        ts = [t for t, _ in recs]
        is_right = recs[0][1]["is_right"]
        go0 = torch.tensor([r["mano"]["global_orient"] for _, r in recs], dtype=torch.float32, device=dev)
        hp0 = torch.tensor([r["mano"]["hand_pose"] for _, r in recs], dtype=torch.float32, device=dev)
        be = torch.tensor([r["mano"]["betas"] for _, r in recs], dtype=torch.float32, device=dev)
        T0 = torch.tensor([r["T"] for _, r in recs], dtype=torch.float32, device=dev)
        Sw = torch.tensor(np.stack([slam[t] for t in ts]), dtype=torch.float32, device=dev)  # cam -> world
        # per object: world -> object-local for each frame (identity rows where the pose is missing)
        for o in objs:
            M = np.stack([np.linalg.inv(o["W"][t]) if t in o["W"] else np.full((4, 4), np.nan) for t in ts])
            o["valid"] = torch.tensor(np.isfinite(M[:, 0, 0]), device=dev)
            o["Wi"] = torch.tensor(np.nan_to_num(M), dtype=torch.float32, device=dev)
            o["gmask"] = torch.tensor([t in set(o["grasp"].get(side, [])) for t in ts], device=dev)
        with torch.no_grad():
            _, j0 = mano_forward(mano, go0, hp0, be, is_right)
            j0 = j0 + T0[:, None]  # the stereo fit's joints, camera frame (same frame as j below)

        def world_verts(dR, dhp, dT):
            # global orientation correction composed on the fitted one, in the camera frame
            R = rotvec_to_matrix(dR) @ rotvec_to_matrix(go0)
            aa = matrix_to_rotvec(R)
            v, j = mano_forward(mano, aa, hp0 + dhp, be, is_right)
            v = v + (T0 + dT)[:, None]
            j = j + (T0 + dT)[:, None]
            vw = torch.einsum("bij,bnj->bni", Sw[:, :3, :3], v) + Sw[:, None, :3, 3]
            return vw, j

        def local_sdf(vw, o):
            vl = torch.einsum("bij,bnj->bni", o["Wi"][:, :3, :3], vw) + o["Wi"][:, None, :3, 3]
            return o["sdf"](vl.reshape(-1, 3)).view(vw.shape[0], -1)

        with torch.no_grad():
            vw0, _ = world_verts(torch.zeros_like(go0), torch.zeros_like(hp0), torch.zeros_like(T0))
            tips0 = {o["name"]: local_sdf(vw0, o)[:, TIPS] for o in objs}
            # A hand error is millimetres to ~2 cm; a hand several cm inside an object means the
            # object's pose is wrong in that frame, and moving the hand to fit it would ruin a good
            # hand (ep23: 87 mm into a 26 mm board). Such frames keep their observed hand.
            trust, bad = {}, {}
            for o in objs:
                deep = torch.relu(-local_sdf(vw0, o)).amax(1) > a.pen_cap_mm / 1000
                trust[o["name"]] = (o["valid"] & ~deep).float()
                bad[o["name"]] = [ts[i] for i in torch.nonzero(deep & o["valid"]).flatten().tolist()]
        # Corrections live on control points every --knot_step frames and are linearly
        # interpolated in between, so they cannot add frame-to-frame jitter (per-frame
        # corrections made the hands twitch). knot_step 0 = the old per-frame form.
        idx = torch.tensor(ts, dtype=torch.float32, device=dev)
        if a.knot_step > 0:
            knots = torch.arange(float(ts[0]), float(ts[-1]) + a.knot_step, float(a.knot_step), device=dev)
            B = torch.clamp(1 - (idx[:, None] - knots[None]).abs() / a.knot_step, min=0)
            B = B / B.sum(1, keepdim=True)
        else:
            B = torch.eye(len(ts), device=dev)
        kR = torch.zeros(B.shape[1], 3, device=dev, requires_grad=True)
        khp = torch.zeros(B.shape[1], 45, device=dev, requires_grad=True)
        kT = torch.zeros(B.shape[1], 3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([kR, khp, kT], lr=a.lr)
        for it in range(a.iters):
            dR, dhp, dT = B @ kR, B @ khp, B @ kT
            vw, j = world_verts(dR, dhp, dT)
            loss_pen = loss_con = torch.zeros((), device=dev)
            for o in objs:
                d = local_sdf(vw, o)
                vmask = trust[o["name"]][:, None]
                loss_pen = loss_pen + (torch.relu(-d) ** 2 * vmask).sum() / len(ts)
                keep = (o["gmask"][:, None] & (tips0[o["name"]].abs() < a.contact_mm / 1000)).float() * vmask
                loss_con = loss_con + (d[:, TIPS] ** 2 * keep).sum() / len(ts)
            loss_data = ((j - j0) ** 2).sum(-1).mean()
            loss_prior = (dhp ** 2).mean() + a.w_rot * (dR ** 2).mean() + a.w_trans * (dT ** 2).sum(-1).mean()
            jw = torch.einsum("bij,bnj->bni", Sw[:, :3, :3], j) + Sw[:, None, :3, 3]  # world: head motion is not hand motion
            acc = jw[2:] - 2 * jw[1:-1] + jw[:-2]
            loss_smooth = (acc ** 2).sum(-1).mean()
            loss = (a.w_pen * loss_pen + a.w_contact * loss_con + a.w_data * loss_data
                    + a.w_prior * loss_prior + a.w_smooth * loss_smooth)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 100 == 0 or it == a.iters - 1:
                print(f"{side} it {it:4d}  pen {loss_pen.item():.3e}  contact {loss_con.item():.3e}  "
                      f"data {1000 * loss_data.sqrt().item():.2f} mm  smooth {loss_smooth.item():.2e}", flush=True)

        with torch.no_grad():
            dR, dhp, dT = B @ kR, B @ khp, B @ kT
            vw, j = world_verts(dR, dhp, dT)
            R = rotvec_to_matrix(dR) @ rotvec_to_matrix(go0)
            go = matrix_to_rotvec(R).cpu().numpy()
            hp = (hp0 + dhp).cpu().numpy()
            T = (T0 + dT).cpu().numpy()
            jc = j.cpu().numpy()
            pen_b = pen_a = 0.0
            for o in objs:  # over trusted frames only (suspect frames are reported separately)
                m = trust[o["name"]][:, None]
                pen_b = max(pen_b, float((torch.relu(-local_sdf(vw0, o)) * m).amax()) * 1000)
                pen_a = max(pen_a, float((torch.relu(-local_sdf(vw, o)) * m).amax()) * 1000)
            moved = float(((j - j0) ** 2).sum(-1).sqrt().mean()) * 1000
        (outdir / side).mkdir(parents=True, exist_ok=True)
        for k, (t, r) in enumerate(recs):
            r2 = dict(r)
            r2["mano"] = {"betas": r["mano"]["betas"], "global_orient": go[k].tolist(), "hand_pose": hp[k].tolist()}
            r2["T"] = T[k].tolist()
            r2["joints"] = jc[k].tolist()
            (outdir / side / f"{t:06d}.json").write_text(json.dumps(r2))
        report["sides"][side] = {"frames": len(ts), "max_pen_before_mm": pen_b, "max_pen_after_mm": pen_a,
                                 "mean_joint_shift_mm": moved,
                                 "object_pose_suspect_frames": bad}
        nbad = sum(len(v) for v in bad.values())
        print(f"{side}: max penetration {pen_b:.1f} -> {pen_a:.1f} mm, mean joint shift {moved:.1f} mm, "
              f"{nbad} frames left out (object pose suspect)", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps(report, indent=2))


def matrix_to_rotvec(R):
    from pytorch3d.transforms import matrix_to_axis_angle
    return matrix_to_axis_angle(R)


if __name__ == "__main__":
    main()
