"""Generic object-pose filter: no physically impossible jumps, then light smoothing. Applies to any pose variant.

1 symmetry unwrap  for symmetric meshes, every frame takes the group element closest to the previous output frame
                   (a "180 deg jump" of a cup about its own axis is a relabelling, not motion).
2 flip / reject    Viterbi over states per frame {as is, 180-deg flip about each of the mesh's 3 principal axes THROUGH
                   ITS CENTROID}; consecutive KEPT frames t0 < t1 (gap g <= MAX_GAP) must satisfy
                   rotation <= g * MAX_DEG and centre travel <= g * MAX_M (hard), plus a weak smoothness cost.
                   Skipping a frame costs C_SKIP (x visibility, so hand-covered frames are cheap to drop), a flipped
                   state costs C_FLIP -- flips win only when they make the trajectory continuous over a stretch.
3 fill + smooth    skipped frames slerp/lerp between kept neighbours; Gaussian SE(3) smoothing (SIGMA frames).
Writes <run>/<out>/<obj>/poses (camera<-object) + pose_filter_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python pose_filter.py <run>... --src <variant> --out <variant>"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from handlock import ang, interp_poses, smooth_poses
from object_icp import load_T
from symmetry import mesh_symmetry

ROOT = Path(os.environ["V2D_ROOT"])
MAX_DEG, MAX_M, MAX_GAP = 20.0, 0.06, 40
C_SKIP, C_FLIP, C_SWITCH, W_SM, SIGMA = 1.0, 0.05, 5.0, 0.02, 1.5
GAP_EXTRA_DEG = 3.0   # rotation allowed across a skipped gap: max_deg * min(g, 2) + GAP_EXTRA_DEG * (g - 1)
OPEN_UP = {"blue_cup", "beige_cup", "water_pitcher", "white_pot"}   # rest / are carried open side up (not the lid)
P_UP = 1.0   # per frame, x how far the open side points down (0..1)


def ray_hits(V, Fc, o, d):
    """Moller-Trumbore: does the ray o + s d (s > 1 mm) hit any triangle of the mesh?"""
    A, B, C = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    e1, e2 = B - A, C - A
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - A
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q * d).sum(1) * inv
    tt = (q * e2).sum(1) * inv
    return bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 1e-3)).any())


def open_direction(mesh, axes):
    """Object-frame unit vector of the open end (cup mouth): a ray from the centroid along it leaves the mesh without
    hitting anything while the opposite ray hits the base. First such direction among the candidate axes, else None."""
    V, Fc = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    c = V.mean(0)
    for a_ in axes:
        a_ = np.asarray(a_, float) / np.linalg.norm(a_)
        hp, hn = ray_hits(V, Fc, c, a_), ray_hits(V, Fc, c, -a_)
        if hp != hn:
            return -a_ if hp else a_
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_deg", type=float, default=MAX_DEG)
    ap.add_argument("--max_m", type=float, default=MAX_M)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    a = ap.parse_args()
    obj_dir = os.environ.get("V2D_OBJ_DIR", "objects_recon")
    mesh_dir = os.environ.get("V2D_MESH_DIR", "meshes_recon")
    for rn in a.runs:
        run = ROOT / "runs" / rn
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        n = max(slam) + 1
        report = {}
        for o in json.loads((run / obj_dir / "objects.json").read_text()):
            nm = o["name"]
            src = run / a.src / nm / "poses"
            if not src.is_dir():
                continue
            X = [slam[t] @ load_T(src / f"{t:06d}.json") for t in range(n)]
            mesh = trimesh.load(ROOT / mesh_dir / nm / f"{nm}.obj", force="mesh")
            V = np.asarray(mesh.vertices)
            c = V.mean(0)
            _, _, ax = np.linalg.svd(V - c, full_matrices=False)          # principal axes (rows)
            Tc = np.eye(4); Tc[:3, 3] = c
            flips = [np.eye(4)]
            for k in range(3):   # 180 deg about principal axis k through the centroid (object frame)
                R = 2 * np.outer(ax[k], ax[k]) - np.eye(3)
                F = np.eye(4); F[:3, :3] = R
                flips.append(Tc @ F @ np.linalg.inv(Tc))
            sym = mesh_symmetry(ROOT / mesh_dir / nm / f"{nm}.obj")
            group = [np.asarray(S) for S in sym["group"]]
            tpp = run / "table_plane.json"
            up = np.asarray(json.loads(tpp.read_text())["up"], float) if tpp.exists() else None
            odir = None
            if nm in OPEN_UP and up is not None:
                odir = open_direction(mesh, ([sym["cont_axis"]] if sym["cont_axis"] is not None else []) + [ax[0], ax[1], ax[2]])
            cv = next(iter(sorted((run / obj_dir / nm).glob(".contact_vis_*.npz"))), None)
            vis = np.load(cv)["vis"][:n] if cv is not None else np.ones(n)
            # 1 symmetry unwrap (group elements are rotations about the centroid frame of mesh_symmetry: apply about c)
            n_unwrap = 0
            if len(group) > 1:
                G4 = []
                for S in group:
                    M = np.eye(4); M[:3, :3] = S
                    G4.append(Tc @ M @ np.linalg.inv(Tc))
                for t in range(1, n):
                    best = min(G4, key=lambda M: ang((X[t] @ M)[:3, :3], X[t - 1][:3, :3]))
                    if ang(best[:3, :3], np.eye(3)) > 1e-3:
                        n_unwrap += 1
                    X[t] = X[t] @ best
            # 2 Viterbi over (frame, state) with skips
            ctr = lambda M: M[:3, :3] @ c + M[:3, 3]   # noqa: E731  world centroid
            S_ = len(flips)
            Xs = [[X[t] @ flips[s] for s in range(S_)] for t in range(n)]
            upp = np.zeros((n, S_))   # open side pointing down: a cup / pitcher / pot is never upside down
            if odir is not None:
                for t in range(n):
                    for s in range(S_):
                        upp[t, s] = P_UP * max(0.0, -float((Xs[t][s][:3, :3] @ odir) @ up))
            INF = 1e18
            cost = np.full((n, S_), INF)
            back = np.full((n, S_, 2), -1, int)
            skip_cost = C_SKIP * np.maximum(vis, 0.05)
            cum_skip = np.r_[0, np.cumsum(skip_cost)]
            for s in range(S_):
                cost[0, s] = (C_FLIP if s else 0.0) + upp[0, s]
            for t in range(1, n):   # starting the path at t (frames before it skipped)
                for s in range(S_):
                    cost[t, s] = cum_skip[t] + (C_FLIP if s else 0.0) + upp[t, s]
            for t in range(1, n):
                for s in range(S_):
                    Rt, pt = Xs[t][s][:3, :3], ctr(Xs[t][s])
                    best, arg = cost[t, s], (-1, -1)
                    for g in range(1, min(MAX_GAP, t) + 1):
                        u = t - g
                        skip = cum_skip[t] - cum_skip[u + 1]
                        for r in range(S_):
                            if cost[u, r] >= INF:
                                continue
                            dr = ang(Rt, Xs[u][r][:3, :3])
                            dp = np.linalg.norm(pt - ctr(Xs[u][r]))
                            # dropped frames are no evidence of motion: a gap may not hide a big turn (a 139 deg
                            # upside-down flip of the ep10 cup became a smooth fake spin when bridged at g * max_deg)
                            if dr > a.max_deg * min(g, 2) + GAP_EXTRA_DEG * (g - 1) or dp > g * a.max_m:
                                continue
                            v = cost[u, r] + skip + W_SM * (dr / g) + (C_FLIP if s else 0.0) + (C_SWITCH if s != r else 0.0) + upp[t, s]
                            if v < best:
                                best, arg = v, (u, r)
                    cost[t, s], back[t, s] = best, arg
            # end: the last frames may be skipped too
            tail = np.r_[cum_skip[n] - cum_skip[1:n + 1]]
            tot = cost + tail[:, None]
            t, s = np.unravel_index(np.argmin(tot), tot.shape)
            kept = {}
            while t >= 0 and s >= 0:
                kept[int(t)] = Xs[t][s]
                t, s = back[t, s]
            n_flip = sum(1 for t_ in kept if not np.allclose(kept[t_], X[t_]))
            full = interp_poses(kept, n)
            fin = smooth_poses(full, a.sigma) if a.sigma > 0 else full
            od = run / a.out / nm / "poses"
            od.mkdir(parents=True, exist_ok=True)
            for t_ in range(n):
                C = np.linalg.inv(slam[t_]) @ fin[t_]
                (od / f"{t_:06d}.json").write_text(json.dumps({"R": C[:3, :3].tolist(), "T": C[:3, 3].tolist()}))
            st0 = [ang(X[t_ + 1][:3, :3], X[t_][:3, :3]) for t_ in range(n - 1)]
            st = [ang(fin[t_ + 1][:3, :3], fin[t_][:3, :3]) for t_ in range(n - 1)]
            report[nm] = {"symmetry": sym["kind"], "open_dir": None if odir is None else np.round(odir, 2).tolist(), "upside_down_frames_after": int(sum(1 for t_ in range(n) if odir is not None and float((fin[t_][:3, :3] @ odir) @ up) < 0)), "unwrapped_frames": n_unwrap, "kept": len(kept),
                          "dropped": n - len(kept), "flipped": n_flip,
                          "max_step_before": round(max(st0), 1), "max_step_after": round(max(st), 1),
                          "jumps10_before": int(sum(x > 10 for x in st0)), "jumps10_after": int(sum(x > 10 for x in st))}
            print(rn, nm, json.dumps(report[nm]), flush=True)
        (run / a.out).mkdir(exist_ok=True)
        (run / a.out / "pose_filter_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
