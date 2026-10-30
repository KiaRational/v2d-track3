"""Fused hand-object contact: our geometry + EgoPHI, gated by geometry (agile image).

EgoPHI alone flags half of every hand (200-400 of 778 vertices), including hands nowhere near the
object, and overlaps our geometric contact by 0-11%; averaging it in would spread false contact over
the hand. So geometry stays the backbone and EgoPHI may only ADD contact where geometry is unsure:
  g(v)   geometric evidence from the hand vertex's distance d to the object surface (at the fused
         object pose): exp(-(max(d - 1 mm, 0) / --sigma_mm)^2)
  e(v)   EgoPHI's contact probability, counted only within --gate_mm of the surface
  p(v)   1 - (1 - g) (1 - w e [d < gate])        (noisy-OR: either source can assert contact)
  contact where p > 0.5
The object side is fused the same way: geometric = object vertices within --geo_mm of a hand vertex,
EgoPHI's object contact gated to object vertices within --gate_mm of a hand.
Checks (no contact ground truth exists; these are physical consistency checks):
  moving-object frames with no contact     (motion nothing explains -- a policy cannot learn it)
  contact on hands farther than 2 cm        (false contact)
  contact-wrench feasibility (mu 0.1 / 1.0) of the contact points, as wrench_feasibility.py
Writes <run>/stereo/contacts_<tag>.npz per object (per frame hand/object contact probability + masks,
for geometric-only and fused) and prints the comparison.
Usage: python contact_fusion.py <run> [...] [--hands mano_fit_phys] [--w 0.6]
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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agile"))
from geometry import Mano  # noqa: E402
from wrench_feasibility import G as GRAV, feasibility, load_T, urdf_inertial  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", default="mano_fit_phys")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--sigma_mm", type=float, default=5.0)
    ap.add_argument("--gate_mm", type=float, default=15.0)
    ap.add_argument("--geo_mm", type=float, default=5.0)
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--tag", default="fused")
    a = ap.parse_args()
    mano = Mano(str(ROOT / "weights/hand"), "cuda")
    for r in a.runs:
        run = ROOT / "runs" / r
        meta = json.loads((run / "episode.json").read_text())
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        tp = json.loads((run / "table_plane.json").read_text())
        up, table_h = np.array(tp["up"]), float(tp["height"])
        hands = {}
        for side in ("left", "right"):
            for f in (run / "stereo" / a.hands / side).glob("*.json"):
                h = json.loads(f.read_text())
                v, _ = mano.local({"mano": h["mano"], "is_right": h["is_right"]})
                hands.setdefault(side, {})[int(f.stem)] = v.cpu().numpy() + np.array(h["T"])  # camera frame
        for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
            nm = o["name"]
            ef = run / "stereo/egophi" / f"{nm}.npz"
            if not ef.exists():
                continue
            E = np.load(ef)
            Vo = E["obj_verts"]  # EgoPHI's decimated object vertices, mesh frame
            pd = run / a.variant / nm / "poses"
            W = {t: load_T(pd / f"{t:06d}.json") for t in slam if (pd / f"{t:06d}.json").exists()}
            mesh = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm / f"{nm}.obj", force="mesh")
            P, fi = trimesh.sample.sample_surface(mesh, 30000, seed=0)
            Nrm, tree = mesh.face_normals[fi], cKDTree(P)
            m_, com, I = urdf_inertial(nm)
            out = {k: [] for k in ("t", "p_l", "p_r", "geo_l", "geo_r", "fus_l", "fus_r", "p_obj", "geo_obj", "fus_obj")}
            stats = {k: [] for k in ("moving", "any_geo", "any_fus", "far_l", "far_r", "fp_geo", "fp_fus",
                                     "feas_geo_01", "feas_fus_01", "feas_geo_1", "feas_fus_1")}
            ts = [int(t) for t in E["t"]]
            for k, t in enumerate(ts):
                if t not in W:
                    continue
                Wi = np.linalg.inv(W[t])
                rec = {}
                all_pts, all_n = {"geo": [], "fus": []}, {"geo": [], "fus": []}
                hand_local = {}
                for side, ek in (("left", "cl"), ("right", "cr")):
                    hv = hands.get(side, {}).get(t)
                    if hv is None:
                        rec[side] = (np.zeros(778), np.zeros(778, bool), np.zeros(778, bool))
                        continue
                    hl = hv @ Wi[:3, :3].T + Wi[:3, 3]
                    hand_local[side] = hl
                    d, idx = tree.query(hl)
                    g = np.exp(-(np.maximum(d - 0.001, 0) / (a.sigma_mm / 1000)) ** 2)
                    e = E[ek][k] * (d < a.gate_mm / 1000)
                    p = 1 - (1 - g) * (1 - a.w * e)
                    geo, fus = g > 0.5, p > 0.5
                    rec[side] = (p, geo, fus)
                    for key, msk in (("geo", geo), ("fus", fus)):
                        all_pts[key] += list(P[idx[msk]])
                        all_n[key] += list(Nrm[idx[msk]])
                    far = d.min() > 0.02
                    stats["far_l" if side == "left" else "far_r"].append(far)
                    if far:
                        stats["fp_geo"].append(geo.any())
                        stats["fp_fus"].append(fus.any())
                # object side
                if hand_local:
                    dh = cKDTree(np.vstack(list(hand_local.values()))).query(Vo)[0]
                else:
                    dh = np.full(len(Vo), np.inf)
                geo_o = dh < a.geo_mm / 1000
                p_o = 1 - (1 - geo_o.astype(float)) * (1 - a.w * E["co"][k] * (dh < a.gate_mm / 1000))
                # moving object? (world frame, central difference)
                if t - 1 in W and t + 1 in W and t - 1 in slam and t + 1 in slam:
                    v_w = (slam[t + 1] @ W[t + 1])[:3, 3] - (slam[t - 1] @ W[t - 1])[:3, 3]
                    moving = np.linalg.norm(v_w) / 2 > 0.005
                    stats["moving"].append(moving)
                    stats["any_geo"].append(rec["left"][1].any() or rec["right"][1].any())
                    stats["any_fus"].append(rec["left"][2].any() or rec["right"][2].any())
                # wrench feasibility for lifted frames (object off the table)
                Wt = W[t]
                if t in slam:
                    Ww = slam[t] @ Wt
                    if ((mesh.vertices @ Ww[:3, :3].T + Ww[:3, 3]) @ up).min() - table_h > 0.01:
                        c = com
                        g_obj = Ww[:3, :3].T @ up  # object frame "up"
                        wreq = np.r_[m_ * GRAV * g_obj, np.zeros(3)]
                        for key in ("geo", "fus"):
                            Pk, Nk = np.array(all_pts[key]), -np.array(all_n[key])
                            if len(Pk):
                                keys = np.floor(Pk / 0.01).astype(int)
                                _, inv = np.unique(keys, axis=0, return_inverse=True)
                                inv = inv.ravel()
                                Pk = np.array([Pk[inv == j].mean(0) for j in range(inv.max() + 1)])
                                Nk = np.array([Nk[inv == j].mean(0) for j in range(inv.max() + 1)])
                                Nk /= np.maximum(np.linalg.norm(Nk, axis=1, keepdims=True), 1e-9)
                            for mu, sk in ((0.1, "01"), (1.0, "1")):
                                stats[f"feas_{key}_{sk}"].append(feasibility(Pk, Nk, c, wreq, mu) < 0.1)
                out["t"].append(t)
                for side, pk, gk, fk in (("left", "p_l", "geo_l", "fus_l"), ("right", "p_r", "geo_r", "fus_r")):
                    out[pk].append(rec[side][0].astype(np.float16))
                    out[gk].append(rec[side][1])
                    out[fk].append(rec[side][2])
                out["p_obj"].append(p_o.astype(np.float16))
                out["geo_obj"].append(geo_o)
                out["fus_obj"].append(p_o > 0.5)
            np.savez_compressed(run / "stereo" / f"contacts_{a.tag}_{nm}.npz", obj_verts=Vo,
                                **{k: np.array(v) for k, v in out.items()})
            mv = np.array(stats["moving"])
            ag, af = np.array(stats["any_geo"]), np.array(stats["any_fus"])
            f = lambda x: "  -" if len(x) == 0 else f"{100 * np.mean(x):3.0f}%"  # noqa: E731
            nv = lambda key: np.mean([m.sum() for m in out[key]]) if out[key] else 0  # noqa: E731
            print(f"{r} {nm}: {len(out['t'])} frames | contact verts/hand geo L{nv('geo_l'):.0f} R{nv('geo_r'):.0f} "
                  f"fused L{nv('fus_l'):.0f} R{nv('fus_r'):.0f} | moving frames without contact: geo "
                  f"{f(~ag[mv])} fused {f(~af[mv])} | contact on far hands: geo {f(stats['fp_geo'])} fused "
                  f"{f(stats['fp_fus'])} | wrench-feasible lifted frames mu0.1 geo {f(stats['feas_geo_01'])} fused "
                  f"{f(stats['feas_fus_01'])}, mu1 geo {f(stats['feas_geo_1'])} fused {f(stats['feas_fus_1'])}",
                  flush=True)


if __name__ == "__main__":
    main()
