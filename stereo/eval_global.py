"""Position-only vs full-6D evaluation of object trajectories (mocap, public split).

Uses the same world->mocap alignment as eval_objects.py --align fp (robust SE(3) on the FP
trajectory poses of all objects; --sim3 frees the scale) for every variant. Per object and variant:
  centre_cm   error of the mesh bounding-box centre (position only, rotation-free)
  trans_cm    error of the mesh origin (as eval_objects), rot_deg, rot_sym_deg (symmetry
              axis only, for auto-detected symmetric meshes), ADDS_cm, jitter_mm
Extra rows for stereo_global diagnostics (from objects/<name>/stereo_global/*.json):
  sg_loc_raw     visible-points centroid (step 2, uncorrected)       -> centre_cm
  sg_loc_anchor  centroid pushed along the ray by the mean visible-depth offset
  sg_top1        per-frame best-scoring hypothesis (no Viterbi / no multi-frame)
  sg_oracle      per-frame best hypothesis by ADD-S among the saved top-k (upper bound)
Writes <run>/eval_global.json. Usage: python eval_global.py <run> [--variants ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stages"))
from eval_objects import T_from_json, robust_align  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
DATA = Path(os.environ["V2D_DATA"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variants", nargs="+", default=["fp", "icp", "fused", "stereo_global_raw", "stereo_global", "fused_global",
                                                          "stereo_global_v2_raw", "stereo_global_v2", "fused_global_v2"])
    ap.add_argument("--sim3", action="store_true", help="free the alignment scale (default SE(3), s = 1)")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    meta = json.loads((run / "episode.json").read_text())
    frames = pq.read_table(DATA / f"public/data/chunk-000/episode_{meta['episode']:06d}.parquet") \
        .column("observation.objects").to_pylist()
    names = [o["name"] for o in frames[0]]
    n = len(frames)
    gt = {}
    for i, nm in enumerate(names):
        Ms = np.full((n, 4, 4), np.nan)
        for t, f in enumerate(frames):
            if f[i]["visible"]:
                x, y, z, qw, qx, qy, qz = f[i]["pose"]
                Ms[t] = np.eye(4)
                Ms[t, :3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                Ms[t, :3, 3] = (x, y, z)
        gt[nm] = Ms
    cams = {t: T_from_json(p, "toolkit")[0] for t in range(n)
            if (p := run / f"recon_hawor/slam_poses/{t:06d}.json").exists()}

    def load_variant(v, nm):
        Ms = np.full((n, 4, 4), np.nan)
        d = run / "objects" / nm / "poses_smoothed" if v == "fp" else run / v / nm / "poses"
        for t in cams:
            p = d / f"{t:06d}.json"
            if p.exists():
                try:  # other jobs may be rewriting a variant concurrently
                    Ms[t] = cams[t] @ T_from_json(p, "toolkit" if v == "fp" else "agile")[0]
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        return Ms

    fp = {nm: load_variant("fp", nm) for nm in names}
    idx = [(nm, t) for nm in names for t in range(n) if np.isfinite(fp[nm][t, 0, 0]) and np.isfinite(gt[nm][t, 0, 0])]
    # robust pose alignment as eval_objects.py (SE(3) unless --sim3; symmetric objects contribute only
    # their well-defined points; outlier pose pairs dropped)
    fr = json.loads((run / "fusion_report.json").read_text()) if (run / "fusion_report.json").exists() else {}
    sy = {nm: (fr[nm].get("symmetry", "none"), fr[nm].get("sym_axis")) for nm in names if isinstance(fr.get(nm), dict)}
    s, R, tr, keep = robust_align([fp[nm][t] for nm, t in idx], [gt[nm][t] for nm, t in idx], with_scale=a.sim3,
                                  syms=[sy.get(nm, ("none", None)) for nm, _ in idx])
    to_m = lambda X: s * X @ R.T + tr

    rep = {"episode": a.run, "sim3_scale": float(s), "align_mode": "sim3_robust_pose" if a.sim3 else "se3_robust_pose",
           "align_inlier_frac": float(keep.mean()), "variants": {}}
    for nm in names:
        mesh = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm / f"{nm}.obj", force="mesh")
        ctr = (mesh.bounds[0] + mesh.bounds[1]) / 2
        V = mesh.vertices[np.random.default_rng(0).choice(len(mesh.vertices), 2000, False)]
        mdl = ROOT / "cache/global_models" / f"{nm}.npz"
        axis = None
        if mdl.exists():
            z = np.load(mdl)
            axis = z["axis"] if bool(z["has_axis"]) else None
        G = gt[nm]

        def metrics(Ms, centre_only=None):
            ce, te, re, rs, adds, pos, pf = [], [], [], [], [], [], {}
            for t in range(n):
                if not np.isfinite(G[t, 0, 0]):
                    continue
                cg = G[t, :3, :3] @ ctr + G[t, :3, 3]
                if centre_only is not None:
                    if t in centre_only:
                        ce.append(np.linalg.norm(to_m(centre_only[t]) - cg))
                    continue
                if not np.isfinite(Ms[t, 0, 0]):
                    continue
                Ra, pa = R @ Ms[t, :3, :3], to_m(Ms[t, :3, 3])
                ce.append(np.linalg.norm(Ra @ ctr + pa - cg))
                pf[t] = [round(float(ce[-1]) * 100, 2), round(float(np.degrees(np.arccos(np.clip((np.trace(G[t, :3, :3].T @ Ra) - 1) / 2, -1, 1)))), 1)]
                te.append(np.linalg.norm(pa - G[t, :3, 3]))
                re.append(np.degrees(np.arccos(np.clip((np.trace(G[t, :3, :3].T @ Ra) - 1) / 2, -1, 1))))
                if axis is not None:
                    rs.append(np.degrees(np.arccos(np.clip((Ra @ axis) @ (G[t, :3, :3] @ axis), -1, 1))))
                Pg = V @ G[t, :3, :3].T + G[t, :3, 3]
                adds.append(cKDTree(Pg).query(V @ Ra.T + pa)[0].mean())
                pos.append(pa)
            m = {"frames": len(ce), "centre_cm": float(np.mean(ce) * 100) if ce else None, "per_frame_centre_cm_rot_deg": pf,
                 "centre_cm_median": float(np.median(ce) * 100) if ce else None}
            if te:
                m.update(trans_cm=float(np.mean(te) * 100), rot_deg=float(np.mean(re)),
                         rot_deg_median=float(np.median(re)), rot_sym_deg=float(np.mean(rs)) if rs else None,
                         ADDS_cm=float(np.mean(adds) * 100),
                         jitter_mm=float(np.linalg.norm(np.diff(np.array(pos), 2, 0), axis=1).mean() * 1000)
                         if len(pos) > 3 else None)
            return m

        for v in a.variants:
            if v != "fp" and not (run / v / nm / "poses").exists():
                continue
            rep["variants"].setdefault(v, {})[nm] = metrics(fp[nm] if v == "fp" else load_variant(v, nm))
        for sgv in [v for v in a.variants if v.startswith("stereo_global") and not v.endswith("_raw")]:
          sg = run / "objects" / nm / sgv
          sfx = sgv[len("stereo_global"):]  # "" or e.g. "_v2"
          if sg.exists():
            raw, anc = {}, {}
            top1, orac = np.full((n, 4, 4), np.nan), np.full((n, 4, 4), np.nan)
            for t in cams:
                p = sg / f"{t:06d}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                if "loc_raw" in d:
                    raw[t] = (cams[t] @ np.r_[d["loc_raw"], 1])[:3]
                    anc[t] = (cams[t] @ np.r_[d["loc_anchor"], 1])[:3]
                hy = [h for h in d.get("hyps", []) if not h["src"].startswith("segment")]
                if hy:
                    top1[t] = cams[t] @ np.array(max(hy, key=lambda h: h["score"])["T"])
                if d.get("hyps") and np.isfinite(G[t, 0, 0]):
                    Pg = V @ G[t, :3, :3].T + G[t, :3, 3]
                    tg = cKDTree(Pg)
                    best = min(d["hyps"], key=lambda h: tg.query(
                        V @ (R @ (cams[t] @ np.array(h["T"]))[:3, :3]).T + to_m((cams[t] @ np.array(h["T"]))[:3, 3]))[0].mean())
                    orac[t] = cams[t] @ np.array(best["T"])
            rep["variants"].setdefault("sg_loc_raw" + sfx, {})[nm] = metrics(None, raw)
            rep["variants"].setdefault("sg_loc_anchor" + sfx, {})[nm] = metrics(None, anc)
            rep["variants"].setdefault("sg_top1" + sfx, {})[nm] = metrics(top1)
            rep["variants"].setdefault("sg_oracle" + sfx, {})[nm] = metrics(orac)
    (run / "eval_global.json").write_text(json.dumps(rep, indent=1))
    f = lambda x, k=2: "   -  " if x is None else f"{x:6.{k}f}"
    print(f"{a.run} {rep['align_mode']} scale {s:.3f} inliers {keep.mean():.2f}")
    for v, objs in rep["variants"].items():
        for nm, m in objs.items():
            print(f"{v:14s} {nm:18s} n={m['frames']:4d} centre {f(m['centre_cm'])} trans {f(m.get('trans_cm'))} "
                  f"rot {f(m.get('rot_deg'), 1)} rot_sym {f(m.get('rot_sym_deg'), 1)} ADD-S {f(m.get('ADDS_cm'))} "
                  f"jitter {f(m.get('jitter_mm'))}")


if __name__ == "__main__":
    main()
