"""Upside-down / flipped object poses (pose estimation of near-mirror-symmetric shapes, e.g. a domed pot lid whose
silhouette is the same from both sides): per frame the fused pose and its three 180-degree flips about the mesh's
canonical axes are scored against the object's own stereo points (SAM mask, eroded; robust mean point-to-surface
distance, truncated at TRUNC). A flip is adopted only where it is clearly better over a stretch of frames (rolling
median gain >= GAIN_MM and error ratio <= RATIO, segments >= MIN_SEG frames); otherwise the pose is untouched.
Geometrically symmetric flips (e.g. a flat board) score the same and are never adopted.
Writes the corrected poses in place (originals kept in <obj>/poses_preflip) and <obj>/flip_report.json.
Usage: V2D_OBJ_DIR=objects_recon V2D_MESH_DIR=meshes_recon python flip_check.py <run> --variant fused_all_recon"""
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import binary_closing, median_filter
from scipy.spatial import cKDTree

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))
TRUNC, GAIN_MM, RATIO, MIN_SEG, WIN = 0.03, 1.0, 0.7, 8, 15   # thin lid: a flip gains only ~2 mm but halves the error
JUMP_DEG = 90.0   # an original frame-to-frame rotation jump this large = the pose estimate switched sides
FLIPS = {"x": np.diag([1.0, -1, -1]), "y": np.diag([-1.0, 1, -1]), "z": np.diag([-1.0, -1, 1])}


def icp(surf, P, R, T, iters=15, keep=0.8):
    """Rigid trimmed point-to-point ICP of the camera points P onto the mesh surface samples (object frame), from
    (R, T) = object->camera. Returns the refined (R, T)."""
    S = surf.data
    for _ in range(iters):
        Po = (P - T) @ R
        d, k = surf.query(Po)
        sel = d <= np.quantile(d, keep)
        A, B = S[k[sel]], P[sel]                   # object-frame model points <-> camera points
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        Dg = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ Dg @ U.T
        T = cb - R @ ca
    return R, T


def depth_m(run, f):
    p = run / "depth_stereo" / f"{f:06d}.png"
    if not p.exists():
        return None
    png = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(float)
    return np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variant", default="fused_all_recon")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    od = run / os.environ.get("V2D_OBJ_DIR", "objects_recon")
    ids = {o["name"]: o["id"] for o in json.loads((od / "objects.json").read_text())}
    K = json.loads((run / "intrinsics_stable.json").read_text())
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    out = {}
    for objd in sorted((run / a.variant).iterdir()):
        nm = objd.name
        pdir = objd / ("poses_preflip" if (objd / "poses_preflip").exists() else "poses")   # never compound
        if nm not in ids or not pdir.is_dir():
            continue
        mesh = trimesh.load(ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / nm / f"{nm}.obj", force="mesh")
        surf = cKDTree(trimesh.sample.sample_surface(mesh, 30000, seed=0)[0])
        frames = sorted(int(p.stem) for p in pdir.glob("[0-9]*.json"))
        hyp = ["id", *FLIPS]
        E = np.full((len(frames), len(hyp)), np.nan)
        poses, FIT = {}, {}
        for i, f in enumerate(frames):
            r = json.loads((pdir / f"{f:06d}.json").read_text())
            R, T = np.asarray(r["R"]), np.asarray(r["T"])
            poses[f] = r
            mp = od / "masks" / str(ids[nm]) / f"{f:06d}.png"
            D = depth_m(run, f) if mp.exists() else None
            if D is None:
                continue
            m = cv2.erode((cv2.imread(str(mp), 0) > 127).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            v, u = np.nonzero(m & np.isfinite(D) & (D > 0.1) & (D < 2.0))
            if len(v) < 200:
                continue
            sel = np.random.default_rng(f).choice(len(v), min(3000, len(v)), replace=False)
            z = D[v[sel], u[sel]]
            P = np.stack([(u[sel] - cx) * z / fx, (v[sel] - cy) * z / fy, z], 1)
            for j, h in enumerate(hyp):
                Rh, Th = icp(surf, P, R if h == "id" else R @ FLIPS[h], T)   # each hypothesis at ITS best fit
                Po = (P - Th) @ Rh                         # camera points -> object frame of hypothesis h
                d, _ = surf.query(Po)
                E[i, j] = np.minimum(d, TRUNC).mean()
                FIT[(i, j)] = (Rh, Th)
        rep = {"frames_scored": int(np.isfinite(E[:, 0]).sum()), "flipped": {}}
        n = len(frames)
        flip_to = np.zeros(n, int)
        on_any = np.zeros(n, bool)
        for j in range(1, len(hyp)):
            g = (E[:, 0] - E[:, j]) * 1000
            g = np.where(np.isfinite(g), g, 0.0)
            gm = median_filter(g, size=WIN, mode="nearest")
            rat = median_filter(np.where(np.isfinite(E[:, j] / E[:, 0]), E[:, j] / E[:, 0], 1.0), size=WIN, mode="nearest")
            on_any |= (gm >= GAIN_MM) & (rat <= RATIO)
        on_any = binary_closing(np.r_[False, on_any, False], np.ones(WIN, bool))[1:-1] | on_any   # no short gap inside
        # a real flip error starts/ends where the ORIGINAL poses jump (pose estimate switched sides) or at the clip
        # ends; a segment boundary with no such jump nearby would CREATE a discontinuity -> rejected
        Rs = [np.asarray(poses[f]["R"]) for f in frames]
        jump = [k for k in range(1, n) if frames[k] - frames[k - 1] == 1 and
                np.degrees(np.arccos(np.clip((np.trace(Rs[k] @ Rs[k - 1].T) - 1) / 2, -1, 1))) > JUMP_DEG]
        rep["original_jumps_at"] = [frames[k] for k in jump]

        def snap(b, at_end):
            if b == (n if at_end else 0):
                return b
            near = [k for k in jump if abs(k - b) <= WIN]
            return min(near, key=lambda k: abs(k - b)) if near else None
        d = np.diff(np.r_[0, on_any.astype(int), 0]); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        for a0, b0 in zip(s, e):
            if b0 - a0 < MIN_SEG:
                continue
            a1, b1 = snap(a0, False), snap(b0, True)
            rats = {jj: float(np.nanmedian(E[a0:b0, jj]) / np.nanmedian(E[a0:b0, 0])) for jj in range(1, len(hyp))}

            def bjump(jj):   # rotation jump the flip would leave at the snapped boundaries (continuity)
                if a1 is None or b1 is None:
                    return 0.0
                Rf = lambda k: FIT.get((k, jj), (Rs[k] @ FLIPS[hyp[jj]], None))[0]
                ang = lambda A, B: np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))
                return (ang(Rf(a1), Rs[a1 - 1]) if a1 > 0 else 0.0) + (ang(Rf(b1 - 1), Rs[b1]) if b1 < n else 0.0)
            good = [jj for jj in rats if rats[jj] <= RATIO] or [min(rats, key=rats.get)]
            j = min(good, key=bjump)                      # ONE flip axis per segment: fits the depth AND stays continuous
            ratio = rats[j]
            ok = a1 is not None and b1 is not None and b1 - a1 >= MIN_SEG and ratio <= RATIO
            rep.setdefault("segments", []).append({"frames": [frames[a0], frames[b0 - 1]], "axis": hyp[j], "ratio": round(ratio, 2),
                                                    "snapped": None if not ok else [frames[a1], frames[b1 - 1]], "adopted": ok})
            if ok:
                flip_to[a1:b1] = j
                rep["flipped"].setdefault(hyp[j], []).append([frames[a1], frames[b1 - 1], round(ratio, 2)])
        rep["err_mm_median"] = {h: round(float(np.nanmedian(E[:, j]) * 1000), 2) for j, h in enumerate(hyp)}
        rep["n_flipped_frames"] = int((flip_to > 0).sum())
        out[nm] = rep
        print(a.run, nm, json.dumps(rep), flush=True)
        if a.dry:
            continue
        if not rep["n_flipped_frames"]:
            if (objd / "poses_preflip").exists():   # an earlier run flipped it: restore the original poses
                shutil.rmtree(objd / "poses")
                shutil.copytree(objd / "poses_preflip", objd / "poses")
                shutil.rmtree(objd / "poses_preflip")
                (objd / "flip_report.json").unlink(missing_ok=True)
            continue
        if not (objd / "poses_preflip").exists():
            shutil.copytree(objd / "poses", objd / "poses_preflip")
        # Per segment ONE object-frame correction C (robust mean of the per-frame ICP fits relative to the plain flip),
        # applied to every frame: the fused trajectory's smoothness is kept. (Writing the per-frame ICP fits put raw
        # single-frame errors back -- a 12 cm / 75 deg spike in e19's sweeper.)
        corr = np.zeros(n, int)
        d = np.diff(np.r_[0, (flip_to > 0).astype(int), 0]); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        seg_C = {}
        for k, (a0, b0) in enumerate(zip(s, e)):
            j = flip_to[a0]
            Cs, ts_ = [], []
            for i in range(a0, b0):
                if (i, j) in FIT:
                    RF = Rs[i] @ FLIPS[hyp[j]]
                    Rh, Th = FIT[(i, j)]
                    Cs.append(RF.T @ Rh); ts_.append(RF.T @ (Th - np.asarray(poses[frames[i]]["T"])))
            if Cs:
                U, _, Vt = np.linalg.svd(np.mean(Cs, 0))
                Cm = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                if np.degrees(np.arccos(np.clip((np.trace(Cm) - 1) / 2, -1, 1))) > 20:   # ICP disagreed: plain flip
                    Cm = np.eye(3); tm = np.zeros(3)
                else:
                    tm = np.median(ts_, 0)
                    tm = tm if np.linalg.norm(tm) < 0.02 else tm * 0.02 / np.linalg.norm(tm)
            else:
                Cm, tm = np.eye(3), np.zeros(3)
            seg_C[k] = (Cm, tm)
            corr[a0:b0] = k
        for i, f in enumerate(frames):
            r = dict(poses[f])
            if flip_to[i]:
                Cm, tm = seg_C[corr[i]]
                RF = np.asarray(r["R"]) @ FLIPS[hyp[flip_to[i]]]
                Rf, Tf = RF @ Cm, np.asarray(r["T"]) + RF @ tm
                r["R"], r["T"], r["flip"] = Rf.tolist(), Tf.tolist(), hyp[flip_to[i]]
            (objd / "poses" / f"{f:06d}.json").write_text(json.dumps(r))
        (objd / "flip_report.json").write_text(json.dumps(rep, indent=1))
    return out


if __name__ == "__main__":
    main()
