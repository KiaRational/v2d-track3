"""Table plane (gravity direction + support height) and its round top (host env).

The head camera looks down at the table, so its image axes say nothing reliable about
gravity (an estimate from the camera's mean "down" axis was tilted by the head pitch and
tipped every object over in simulation). The table is the dominant plane under the objects:
RANSAC on world-frame stereo points around the objects gives its normal (= up, oriented
towards the camera) and height. Where mocap exists (public split), the angle to the true
vertical is reported as a check.
The table is a DISK, and hands also go under it, so an infinite plane is wrong for penetration
tests: plane inliers from more frames (whole field of view) are projected onto the plane, the
outer contour of their largest connected occupancy component is extracted (holes = objects /
hands on the table are ignored), and a circle is fitted to it by RANSAC -- the table rim is an arc,
field-of-view cut-offs are not -- subject to the disk containing ~all inliers, then refined by
least squares on the arc inliers. Nominal top thickness --thickness.
Writes <run>/table_plane.json {"up": [3], "height": h (p . up = h on the table), "inlier_frac",
"gt_angle_deg", "center": [3] (world, on the plane), "radius": m, "thickness": m, "disk_fit":
{arc_coverage, contour_inlier_frac, ok}, "diag_hands": ...}; an existing file keeps its up/height
(only disk keys are added) unless --refit. PNG: $V2D_VIZ/14_joint/<run>_table_disk.png.
Usage: python table_plane.py <run> [...] [--obj_dir objects_sam3] [--variant fused_all_sam3] [--refit]
       compliant: --obj_dir objects_free --from_masks --refit (object centres from size-free masks; no mesh, no pose)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from object_icp import decode_depth, load_T

ROOT = Path(os.environ["V2D_ROOT"])


def fit_plane(P, guess, c, rng):
    best = (0, None, None)
    for _ in range(2000):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ guess < 0:
            n = -n
        if n @ guess < 0.3:  # the table faces roughly up, not a wall
            continue
        h = n @ s[0]
        inl = np.abs(P @ n - h) < 0.008
        # the support plane lies BELOW the objects' centres
        if inl.sum() > best[0] and (c @ n) > h:
            best = (int(inl.sum()), n, h)
    _, n, h = best
    inl = np.abs(P @ n - h) < 0.008
    Q = P[inl]  # refine by least squares on the inliers
    m = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - m)
    n = Vt[2] if Vt[2] @ n > 0 else -Vt[2]
    return n, float(n @ m), inl


def circle3(p):
    """Circle through three 2D points -> (centre, radius) or None."""
    (ax, ay), (bx, by), (cx, cy) = p
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.array([ux, uy]), float(np.hypot(ax - ux, ay - uy))


def fit_disk(Q2, obj2, rng, cell=0.01, tol=0.015, r_range=(0.2, 1.5), iters=4000):
    """Q2 (N,2) plane inliers in plane coordinates -> (centre2, radius, info, grid artefacts)."""
    lo = Q2.min(0) - 5 * cell
    ij = np.floor((Q2 - lo) / cell).astype(int)
    shape = ij.max(0) + 6
    occ = np.zeros(shape[::-1], np.uint8)            # rows = y, cols = x
    np.add.at(occ, (ij[:, 1], ij[:, 0]), 1)
    occ = ((occ >= 2) * 255).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    nlab, lab = cv2.connectedComponents(occ)
    oc = np.floor((obj2 - lo) / cell).astype(int)
    pick = [lab[y, x] for x, y in oc if 0 <= y < lab.shape[0] and 0 <= x < lab.shape[1] and lab[y, x] > 0]
    k = max(set(pick), key=pick.count) if pick else 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    comp = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    C = np.concatenate([c[:, 0] for c in cs]).astype(float) * cell + lo + cell / 2  # outer contour, plane coords
    F = np.argwhere(comp > 0)[:, ::-1].astype(float) * cell + lo + cell / 2           # filled cells
    best = (-1, None, None)
    for _ in range(iters):
        cr = circle3(C[rng.choice(len(C), 3, replace=False)])
        if cr is None or not r_range[0] <= cr[1] <= r_range[1]:
            continue
        ctr, r = cr
        if np.mean(np.linalg.norm(F - ctr, axis=1) <= r + tol) < 0.98:  # the top must contain the seen table
            continue
        score = int((np.abs(np.linalg.norm(C - ctr, axis=1) - r) < tol).sum())
        if score > best[0]:
            best = (score, ctr, r)
    if best[1] is None:
        return None, None, {"ok": False, "reason": "no circle contains the table inliers"}, (comp, lo, C)
    _, ctr, r = best
    for _ in range(3):  # algebraic (Kasa) refinement on the arc inliers
        m = np.abs(np.linalg.norm(C - ctr, axis=1) - r) < tol
        A = np.c_[2 * C[m], np.ones(m.sum())]
        sol = np.linalg.lstsq(A, (C[m] ** 2).sum(1), rcond=None)[0]
        ctr, r = sol[:2], float(np.sqrt(sol[2] + sol[:2] @ sol[:2]))
    m = np.abs(np.linalg.norm(C - ctr, axis=1) - r) < tol
    ang = np.arctan2(*(C[m] - ctr).T[::-1])
    cover = float(len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 72))) / 72)
    info = {"arc_coverage": round(cover, 3), "contour_inlier_frac": round(float(m.mean()), 3),
            "contained_frac": round(float(np.mean(np.linalg.norm(F - ctr, axis=1) <= r + tol)), 3),
            "n_contour": int(len(C)), "ok": bool(cover >= 0.12 and m.sum() >= 30)}
    return ctr, r, info, (comp, lo, C)


def fit_disk_fixed_r(C, F, r, rng, tol=0.015, min_contained=0.88, iters=4000):
    """Centre of a circle of known radius r on the outer contour C (plane coords), F = filled cells.
    Two contour points give the two centres at distance r from both; RANSAC on arc inliers, subject to
    the disk containing ~all seen table cells, then Gauss-Newton on the arc inliers (r fixed)."""
    best = (-1, None)
    for _ in range(iters):
        a, b = C[rng.choice(len(C), 2, replace=False)]
        d = np.linalg.norm(b - a)
        if not 1e-3 < d < 2 * r:
            continue
        mid, perp = (a + b) / 2, np.array([a[1] - b[1], b[0] - a[0]]) / d
        h = np.sqrt(r * r - d * d / 4)
        for ctr in (mid + h * perp, mid - h * perp):
            if np.mean(np.linalg.norm(F - ctr, axis=1) <= r + tol) < min_contained:
                continue
            score = int((np.abs(np.linalg.norm(C - ctr, axis=1) - r) < tol).sum())
            if score > best[0]:
                best = (score, ctr)
    if best[1] is None:
        return None, {"ok": False, "reason": f"no r={r:.3f} circle contains the table inliers"}
    ctr = best[1]
    for _ in range(10):
        m = np.abs(np.linalg.norm(C - ctr, axis=1) - r) < tol
        v = C[m] - ctr
        n = np.linalg.norm(v, axis=1)
        J = -v / n[:, None]                       # d(|C - ctr| - r)/d ctr
        ctr = ctr - np.linalg.lstsq(J, n - r, rcond=None)[0]
    m = np.abs(np.linalg.norm(C - ctr, axis=1) - r) < tol
    ang = np.arctan2(*(C[m] - ctr).T[::-1])
    cover = float(len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 72))) / 72)
    return ctr, {"arc_coverage": round(cover, 3), "contour_inlier_frac": round(float(m.mean()), 3),
                 "contained_frac": round(float(np.mean(np.linalg.norm(F - ctr, axis=1) <= r + tol)), 3),
                 "n_contour": int(len(C)), "ok": bool(cover >= 0.12 and m.sum() >= 30), "fixed_radius": r}


def draw(path, comp, lo, C, ctr, r, obj2, cell=0.01, scale=4):
    img = cv2.cvtColor((comp * 140).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    px = lambda q: tuple(int(v) for v in ((np.asarray(q) - lo) / cell * scale))  # noqa: E731
    for q in C[::3]:
        cv2.circle(img, px(q), 1, (0, 200, 255), -1)
    if ctr is not None:
        cv2.circle(img, px(ctr), int(r / cell * scale), (0, 0, 255), 2)
        cv2.drawMarker(img, px(ctr), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    for q in obj2:
        cv2.circle(img, px(q), 6, (255, 120, 0), 2)
    ys, xs = np.nonzero(comp)  # crop to the table (+ the circle) with a margin
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if ctr is not None:
        cx, cy = (ctr - lo) / cell
        x0, x1 = min(x0, cx - r / cell), max(x1, cx + r / cell)
        y0, y1 = min(y0, cy - r / cell), max(y1, cy + r / cell)
    m = 10
    img = img[max(0, int(y0 - m) * scale):int(y1 + m) * scale, max(0, int(x0 - m) * scale):int(x1 + m) * scale]
    img = np.ascontiguousarray(img[::-1])  # plane y up
    cv2.putText(img, f"table r = {r:.3f} m (red), plane inliers (grey), rim (yellow), objects (blue)" if r else "no fit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def hand_diag(run, slam, rec, hands="mano_fit_phys"):
    """Fraction of hand frames with the wrist below the table top, inside vs outside the disk."""
    up, c, r, th = np.array(rec["up"]), np.array(rec["center"]), rec["radius"], rec["thickness"]
    out = {"hands": hands}
    for side in ("left", "right"):
        W = []
        for f in sorted((run / "stereo" / hands / side).glob("*.json")):
            t = int(f.stem)
            if t in slam:
                j0 = np.array(json.loads(f.read_text())["joints"][0])
                W.append(slam[t][:3, :3] @ j0 + slam[t][:3, 3])
        if not W:
            continue
        d = np.array(W) - c
        h = d @ up
        rad = np.linalg.norm(d - h[:, None] * up, axis=1)
        ins = rad < r
        out[side] = {"frames": len(W), "inside_disk": int(ins.sum()),
                     "below_top_inside_frac": round(float((h[ins] < 0).mean()), 3) if ins.any() else None,
                     "in_slab_inside_frac": round(float(((h > -th) & (h < 0))[ins].mean()), 3) if ins.any() else None,
                     "under_table_inside_frac": round(float((h[ins] <= -th).mean()), 3) if ins.any() else None,
                     "below_top_outside_frac": round(float((h[~ins] < 0).mean()), 3) if (~ins).any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--obj_dir", default="objects_sam3")
    ap.add_argument("--variant", default="fused_all_sam3")
    ap.add_argument("--from_masks", action="store_true",
                    help="object centres from <obj_dir>/masks + stereo depth instead of <variant> poses (compliant: no mesh)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--disk_frames", type=int, default=24)
    ap.add_argument("--thickness", type=float, default=0.03, help="nominal table top thickness (m)")
    ap.add_argument("--refit", action="store_true", help="refit up/height too (default: keep an existing file's)")
    ap.add_argument("--fixed_radius", type=float, default=0.0,
                    help="known table radius (m): refit only the centre when the free fit's arc coverage is low")
    ap.add_argument("--fixed_below_cov", type=float, default=0.8,
                    help="use --fixed_radius when the free fit covers less than this fraction of the rim")
    a = ap.parse_args()
    for r in a.runs:
        rng = np.random.default_rng(0)
        run = ROOT / "runs" / r
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        ts = sorted(slam)
        ki = json.loads((run / "recon_hawor/intrinsics_stable.json").read_text())
        K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
        cen = []
        for o in json.loads((run / a.obj_dir / "objects.json").read_text()):
            if a.from_masks:   # object centres from the masks + stereo depth: no object pose / mesh involved
                for t in ts[::10]:
                    mp = run / a.obj_dir / "masks" / str(o["id"]) / f"{t:06d}.png"
                    d = decode_depth(run / f"depth_stereo/{t:06d}.png") if mp.exists() else None
                    if d is None:
                        continue
                    v, u = np.nonzero((cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0) & (d > 0.15))
                    if len(u) < 100:
                        continue
                    z = d[v, u]
                    X = np.median(np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1), 0)
                    cen.append(slam[t][:3, :3] @ X + slam[t][:3, 3])
                continue
            pd = run / a.variant / o["name"] / "poses"
            cen += [(slam[t] @ load_T(pd / f"{t:06d}.json"))[:3, 3] for t in ts[::10] if (pd / f"{t:06d}.json").exists()]
        c = np.median(cen, 0)
        guess = -np.mean([slam[t][:3, 1] for t in ts], 0)
        guess /= np.linalg.norm(guess)

        def points(nf, max_dist, step=6, zmax=2.0):
            P = []
            for t in np.linspace(ts[0], ts[-1], nf).round().astype(int):
                d = decode_depth(run / f"depth_stereo/{t:06d}.png")
                if d is None:
                    continue
                v, u = np.mgrid[0:d.shape[0]:step, 0:d.shape[1]:step]
                v, u = v.ravel(), u.ravel()
                z = d[v, u]
                ok = (z > 0.15) & (z < zmax)
                v, u, z = v[ok], u[ok], z[ok]
                X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
                Xw = X @ slam[t][:3, :3].T + slam[t][:3, 3]
                P.append(Xw[np.linalg.norm(Xw - c, axis=1) < max_dist])
            return np.concatenate(P)
        f = run / "table_plane.json"
        if f.exists() and not a.refit:
            rec = json.loads(f.read_text())
            n, h = np.array(rec["up"]), float(rec["height"])
        else:
            P = points(a.frames, 0.6)
            if len(P) > 60000:
                P = P[rng.choice(len(P), 60000, replace=False)]
            n, h, inl = fit_plane(P, guess, c, rng)
            rec = {"up": n.tolist(), "height": h, "inlier_frac": float(inl.mean()), "n_points": int(len(P)),
                   "camera_axis_angle_deg": float(np.degrees(np.arccos(np.clip(n @ guess, -1, 1))))}
            gw = run / "gt_world.json"
            if gw.exists():
                R = np.array(json.loads(gw.read_text())["sim3_world_to_mocap"]["R"])
                rec["gt_angle_deg"] = float(np.degrees(np.arccos(np.clip((R @ n)[2], -1, 1))))
        # ---- round top: plane inliers over the whole view, projected onto the plane
        Pd = points(a.disk_frames, 2.0, step=4, zmax=3.0)
        Q = Pd[np.abs(Pd @ n - h) < 0.01]
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        o = h * n
        B = np.stack([e1, e2], 1)
        obj2 = (np.array(cen) - o) @ B
        ctr, rad, info, (comp, lo, C) = fit_disk((Q - o) @ B, obj2, rng)
        if a.fixed_radius and info.get("arc_coverage", 0.0) < a.fixed_below_cov:
            # partial rim (field-of-view cut-offs fit as arc): same table in every episode -> known radius
            F = np.argwhere(comp > 0)[:, ::-1].astype(float) * 0.01 + lo + 0.005
            ctr_f, info_f = fit_disk_fixed_r(C, F, a.fixed_radius, rng)
            info_f["free_fit"] = {"radius": None if rad is None else round(float(rad), 4), **info}
            ctr, rad, info = ctr_f, (a.fixed_radius if ctr_f is not None else None), info_f
        info["n_plane_points"] = int(len(Q))
        rec["disk_fit"] = info
        rec["thickness"] = a.thickness
        if ctr is not None:
            rec["center"] = (o + B @ ctr).tolist()
            rec["radius"] = float(rad)
        draw(Path(os.environ.get("V2D_VIZ", ROOT / "viz")) / "14_joint" / f"{r}_table_disk.png", comp, lo, C, ctr, rad, obj2)
        if ctr is not None:
            rec["diag_hands"] = hand_diag(run, slam, rec)
        f.write_text(json.dumps(rec, indent=1))
        print(f"{r}: up={np.round(n, 3)}  inliers {rec.get('inlier_frac', float('nan')):.2f}  vs mocap vertical "
              f"{rec.get('gt_angle_deg', float('nan')):.1f} deg  disk r={rec.get('radius', float('nan')):.3f} m {info}  "
              f"hands {json.dumps(rec.get('diag_hands'))}", flush=True)


if __name__ == "__main__":
    main()
