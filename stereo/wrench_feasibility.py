"""Contact-wrench feasibility of the reconstructed grasps (agile image: torch, SDFGrid, scipy).

The standard FlashCHORD trains with (reward term contact_wrench_support): the reference contacts
-- positions, surface normals, friction cones -- must be able to produce the force and torque that
move the object along its recorded trajectory, gravity included. Per lifted grasp frame:

  contacts   hand vertices within --contact_mm of the object surface (object signed distance
             field), merged into 1 cm patches; normal = inward surface normal (-grad sdf)
  required   w = [F; tau] the hands must apply: F = m (a - g), tau = I alpha (object frame
             rotated to the world), a/alpha from the object trajectory (central differences);
             m, COM, inertia from the dataset URDF (0.3 kg placeholder, surface-shell inertia)
  feasible?  forces inside linearised friction cones (8 edges) at the contacts, f >= 0:
             min ||G f - w|| by NNLS; residual r = ||G f - w|| / ||w|| (torques scaled by 10 cm)
             feasible when r < --tol
Both the reward's friction (mu = 0.1, strict: needs real opposition) and the simulator's contact
friction (mu = 1.0) are evaluated. Frames with the object resting on the table are skipped (the
table supports it). Writes stereo/wrench_<hands>.json (per frame, per grasp) and prints one row
per object.
Usage: python wrench_feasibility.py <run> [...] [--hands mano_fit]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano, SDFGrid  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
L_SCALE = 0.1  # m: torques are compared in units of force x 10 cm
G = 9.81


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


def urdf_inertial(name):
    u = ROOT / "data/track_3/public/urdf" / f"{name}_rigid.urdf"
    if not u.exists():  # objects with no provided scan (wooden_spoon): our own URDF, same 0.3 kg placeholder
        u = ROOT / "urdf_extra" / f"{name}_rigid.urdf"
    return parse_inertial(u.read_text())


def parse_inertial(x):
    """Mass, COM and inertia (about the COM, in the link frame) of the first <inertial> block.
    URDF gives the inertia in the inertial <origin> frame (xyz, rpy = fixed-axis roll-pitch-yaw);
    it is rotated into the link frame: I_link = R I R^T."""
    blk = re.search(r"<inertial>(.*?)</inertial>", x, re.S).group(1)

    def attr(tag, name, default):
        t = re.search(rf"<{tag}\b[^>]*>", blk)
        v = re.search(rf'\b{name}="([^"]+)"', t.group(0)) if t else None
        return v.group(1) if v else default

    m = float(attr("mass", "value", "nan"))
    com = np.array([float(v) for v in attr("origin", "xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in attr("origin", "rpy", "0 0 0").split()])
    g = {k: float(attr("inertia", k, "0")) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    I = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
    R = Rotation.from_euler("xyz", rpy).as_matrix()  # extrinsic x-y-z = URDF rpy
    return m, com, R @ I @ R.T


def cone_edges(n, mu, k=8):
    """k unit edge directions of the friction cone around inward normal n."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    E = [n + mu * (np.cos(th) * t1 + np.sin(th) * t2) for th in np.linspace(0, 2 * np.pi, k, endpoint=False)]
    return np.array([e / np.linalg.norm(e) for e in E])


def feasibility(P, N, c, w, mu):
    """Relative residual of the best non-negative cone combination producing wrench w."""
    if len(P) == 0:
        return 1.0
    cols = []
    for p, n in zip(P, N):
        for e in cone_edges(n, mu):
            cols.append(np.r_[e, np.cross(p - c, e) / L_SCALE])
    A = np.array(cols).T
    b = np.r_[w[:3], w[3:] / L_SCALE]
    lam, res = nnls(A, b, maxiter=2000)
    return float(res / max(np.linalg.norm(b), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", default="mano_fit")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--contact_mm", type=float, default=5.0)
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--mus", type=float, nargs="+", default=[0.1, 1.0])
    a = ap.parse_args()
    dev = "cuda"
    mano = Mano(str(ROOT / "weights/hand"), dev)
    print(f"{'run':18s} {'object':18s} {'hands':22s} {'frames':>6s} " +
          " ".join(f"{'feas mu=' + str(m):>13s}" for m in a.mus) + f" {'patches':>8s}", flush=True)
    for r in a.runs:
        run = ROOT / "runs" / r
        meta = json.loads((run / "episode.json").read_text())
        fps = meta["fps"]
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        tp = json.loads((run / "table_plane.json").read_text())
        up, table_h = np.array(tp["up"]), float(tp["height"])
        rep = json.loads((run / "fusion_report.json").read_text())
        hands = {}
        for side in ("left", "right"):
            for f in (run / "stereo" / a.hands / side).glob("*.json"):
                t = int(f.stem)
                if t in slam:
                    h = json.loads(f.read_text())
                    v, _ = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                    vc = v.cpu().numpy() + np.array(h["T"])
                    hands.setdefault(side, {})[t] = vc @ slam[t][:3, :3].T + slam[t][:3, 3]
        out = {}
        for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
            nm = o["name"]
            pd = run / a.variant / nm / "poses"
            W = {t: slam[t] @ load_T(pd / f"{t:06d}.json") for t in slam if (pd / f"{t:06d}.json").exists()}
            mesh = trimesh.load(ROOT / "meshes_obj" / nm / f"{nm}.obj", force="mesh")
            sdf = SDFGrid(mesh, res=160, device=dev)
            m, com, I = urdf_inertial(nm)
            held = sorted(set().union(*[set(v) for v in rep.get(nm, {}).get("hand_contact", {}).values()] or [set()]))
            rows = []
            for t in held:
                if not all(u in W for u in (t - 1, t, t + 1)):
                    continue
                Wt = W[t]
                Vt = mesh.vertices @ Wt[:3, :3].T + Wt[:3, 3]
                over = True  # round top (table_plane.py center/radius): only over the disk does the table support
                if "radius" in tp:
                    c0 = Vt.mean(0) - np.asarray(tp["center"])
                    over = np.linalg.norm(c0 - (c0 @ up) * up) <= tp["radius"]
                if (Vt @ up).min() - table_h < 0.01 and over:
                    continue  # resting on the table: the table supports it
                c = Wt[:3, :3] @ com + Wt[:3, 3]
                cm = [W[u][:3, :3] @ com + W[u][:3, 3] for u in (t - 1, t, t + 1)]
                acc = (cm[2] - 2 * cm[1] + cm[0]) * fps ** 2
                R0, R1, R2 = (W[u][:3, :3] for u in (t - 1, t, t + 1))
                w1 = Rotation.from_matrix(R1 @ R0.T).as_rotvec() * fps
                w2 = Rotation.from_matrix(R2 @ R1.T).as_rotvec() * fps
                alpha = (w2 - w1) * fps
                Iw = Wt[:3, :3] @ I @ Wt[:3, :3].T
                F = m * (acc + G * up)  # hands supply the net force minus gravity
                omega = 0.5 * (w1 + w2)
                tau = Iw @ alpha + np.cross(omega, Iw @ omega)
                P, N = [], []
                for side, seq in hands.items():
                    if t not in seq:
                        continue
                    Wi = np.linalg.inv(Wt)
                    vl = seq[t] @ Wi[:3, :3].T + Wi[:3, 3]
                    q = torch.tensor(vl, dtype=torch.float32, device=dev, requires_grad=True)
                    d = sdf(q)
                    (g,) = torch.autograd.grad(d.sum(), q)
                    d, g = d.detach().cpu().numpy(), g.cpu().numpy()
                    sel = np.abs(d) < a.contact_mm / 1000
                    if not sel.any():
                        continue
                    pts, nrm = vl[sel], -g[sel] / np.maximum(np.linalg.norm(g[sel], axis=1, keepdims=True), 1e-9)
                    keys = np.floor(pts / 0.01).astype(int)
                    _, inv = np.unique(keys, axis=0, return_inverse=True)
                    for k in range(inv.max() + 1):
                        pk, nk = pts[inv.ravel() == k].mean(0), nrm[inv.ravel() == k].mean(0)
                        P.append(Wt[:3, :3] @ pk + Wt[:3, 3])
                        N.append(Wt[:3, :3] @ (nk / max(np.linalg.norm(nk), 1e-9)))
                res = {str(mu): feasibility(np.array(P), np.array(N), c, np.r_[F, tau], mu) for mu in a.mus}
                rows.append({"t": t, "patches": len(P), "residual": {k: round(v, 3) for k, v in res.items()}})
            summ = {"lifted_grasp_frames": len(rows),
                    "feasible_frac": {str(mu): (float(np.mean([x["residual"][str(mu)] < a.tol for x in rows]))
                                                if rows else None) for mu in a.mus},
                    "mean_patches": float(np.mean([x["patches"] for x in rows])) if rows else None, "per_frame": rows}
            out[nm] = summ
            ff = summ["feasible_frac"]
            fr = " ".join(f"{('-' if ff[str(mu)] is None else format(ff[str(mu)], '.2f')):>13s}" for mu in a.mus)
            print(f"{r:18s} {nm:18s} {a.hands:22s} {len(rows):6d} {fr} "
                  f"{summ['mean_patches'] or 0:8.1f}", flush=True)
        (run / "stereo" / f"wrench_{a.hands}.json").write_text(json.dumps(
            {"hands": a.hands, "tol": a.tol, "objects": out}, indent=1))


if __name__ == "__main__":
    main()
