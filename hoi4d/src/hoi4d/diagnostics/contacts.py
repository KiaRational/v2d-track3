"""README stage 7 contact/support checks + contacts.json (stage 9 input), from one trajectory version.

Per frame and entity pair (world frame, metres):
  hand-object   hand = MANO joints + bone midpoints as spheres (finger radius); distance to the object surface,
                penetration into its cavity-aware collision parts
  object-object surface distance + penetration (A's surface samples inside B's parts, and vice versa)
  object-table  lowest surface point over the disk (table top z = 0): gap / penetration
Derived: contact intervals with a mode (grasp / support / release), grasp slip (fingertips drift in the object
frame), unsupported objects (elevated or moving, no hand, no support), unstable rest (COM outside the support
polygon of the resting contact points). All are candidates with measurements, never verdicts.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from ..geometry.assets import asset, inv, transform
from .motion import runs

TIPS = [4, 8, 12, 16, 20]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_R = 0.008


def hand_points(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spheres along the skeleton: joints + bone midpoints; radius per point."""
    mids = np.stack([(J[a] + J[b]) / 2 for a, b in BONES])
    P = np.concatenate([J, mids])
    r = np.full(len(P), FINGER_R)
    r[0] = 0.02  # wrist / palm
    return P, r


def frame_pairs(tr, cfg, stride: int = 1) -> dict:
    """Per-frame measurements for every pair; NaN where an entity is missing."""
    F = tr.F
    th = cfg["diagnostics"]
    names = list(tr.objects)
    A = {n: asset(n) for n in names}
    trees = {n: cKDTree(A[n].surface) for n in names}
    out = {"frames": np.arange(0, F, stride)}
    for n in names:
        o = tr.objects[n]
        zmin = np.full(F, np.nan); pen_t = np.full(F, np.nan); low = [None] * F
        for f in out["frames"]:
            if not o["valid"][f]:
                continue
            P = transform(o["T_world_obj"][f], A[n].surface)
            inside = np.linalg.norm(P[:, :2], axis=1) < 0.2375 + 0.01
            z = P[inside, 2] if inside.any() else np.array([np.inf])
            zmin[f] = z.min()
            pen_t[f] = max(0.0, -z.min()) if np.isfinite(z.min()) else 0.0
            low[f] = P[inside & (P[:, 2] < z.min() + 0.003)] if inside.any() else None
        out[(n, "table")] = {"gap_m": zmin, "pen_m": pen_t, "low_pts": low}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = np.full(F, np.nan); pen = np.full(F, np.nan); above = np.full(F, np.nan)
            oa, ob = tr.objects[a], tr.objects[b]
            for f in out["frames"]:
                if not (oa["valid"][f] and ob["valid"][f]):
                    continue
                Ta, Tb = oa["T_world_obj"][f], ob["T_world_obj"][f]
                Pa_in_b = transform(inv(Tb) @ Ta, A[a].surface)
                Pb_in_a = transform(inv(Ta) @ Tb, A[b].surface)
                d[f] = trees[b].query(Pa_in_b)[0].min()
                pen[f] = max(A[b].penetration(Pa_in_b).max(), A[a].penetration(Pb_in_a).max())
                above[f] = transform(Ta, A[a].centroid[None])[0, 2] - transform(Tb, A[b].centroid[None])[0, 2]
            out[(a, b)] = {"dist_m": d, "pen_m": pen, "a_above_b_m": above}
    for h, hd in tr.hands.items():
        for n in names:
            o = tr.objects[n]
            d = np.full(F, np.nan); pen = np.full(F, np.nan); ntips = np.zeros(F, int); thumb = np.zeros(F, bool)
            tips_obj = np.full((F, 5, 3), np.nan)
            for f in out["frames"]:
                if not (hd["valid"][f] and o["valid"][f]):
                    continue
                P, r = hand_points(hd["joints"][f])
                Po = transform(inv(o["T_world_obj"][f]), P)
                dd = trees[n].query(Po)[0] - r
                d[f] = dd.min()
                pen[f] = A[n].penetration(Po).max()
                close = dd[TIPS] < th["contact_m"]
                ntips[f] = close.sum()
                thumb[f] = close[0]
                tips_obj[f] = Po[TIPS]
            out[(h, n)] = {"dist_m": d, "pen_m": pen, "n_tips": ntips, "thumb": thumb, "tips_obj": tips_obj}
    return out


def contacts_and_anomalies(tr, pairs: dict, cfg: dict, speed: dict) -> tuple[list, list]:
    th = cfg["diagnostics"]
    t = tr.t_s
    fr = tr.frame
    C, A = [], []

    def iv(a, b):
        return {"t_start_s": float(t[a]), "t_end_s": float(t[b]), "frames": [int(fr[a]), int(fr[b])]}

    grasp = {}
    for key, v in pairs.items():
        if not isinstance(key, tuple):
            continue
        a, b = key
        if a.startswith("hand_"):
            g = (v["n_tips"] >= 2) & v["thumb"]
            touch = np.nan_to_num(v["dist_m"], nan=1) < th["contact_m"]
            grasp[(a, b)] = g
            for s, e in runs(g, min_len=th["min_interval_frames"], merge_gap=2):
                C.append({"entity_a": a, "entity_b": b, "mode": "grasp", **iv(s, e),
                          "region": {"tips_in_contact_mean": float(v["n_tips"][s:e + 1].mean())},
                          "evidence": "thumb + >=1 fingertip within contact distance of the object surface",
                          "confidence": float(np.clip(v["n_tips"][s:e + 1].mean() / 4, 0.3, 1))})
                C.append({"entity_a": a, "entity_b": b, "mode": "release", **iv(e, min(e + 1, tr.F - 1)),
                          "region": {}, "evidence": "end of grasp interval", "confidence": 0.5})
                # slip: fingertip drift in the object frame over the grasp
                tips = v["tips_obj"][s:e + 1]
                ok = np.isfinite(tips).all(axis=(1, 2))
                if ok.sum() >= 3:
                    drift = np.linalg.norm(tips[ok] - tips[ok][0], axis=2)
                    med = np.median(drift[:, :3], axis=1)
                    if med.max() > th["slip_m"]:
                        A.append({"type": "contact_slip", "entities": [a, b], **iv(s, e),
                                  "measurements": {"max_tip_drift_m": float(med.max())},
                                  "thresholds": {"slip_m": th["slip_m"]},
                                  "severity": float(min(1, med.max() / (3 * th["slip_m"])))})
            for s, e in runs(touch & ~g, min_len=th["min_interval_frames"], merge_gap=2):
                C.append({"entity_a": a, "entity_b": b, "mode": "sticking", **iv(s, e), "region": {},
                          "evidence": "hand touching the object without a thumb-opposed grasp", "confidence": 0.4})
            pen = np.nan_to_num(v["pen_m"])
            for s, e in runs(pen > th["penetration_m"], merge_gap=2):
                A.append({"type": "hand_object_penetration", "entities": [a, b], **iv(s, e),
                          "measurements": {"max_pen_m": float(pen[s:e + 1].max())},
                          "thresholds": {"penetration_m": th["penetration_m"]},
                          "severity": float(min(1, pen[s:e + 1].max() / 0.03))})
        elif b == "table":
            gap = v["gap_m"]
            sup = np.nan_to_num(gap, nan=1) < th["support_gap_m"]
            for s, e in runs(sup, min_len=th["min_interval_frames"], merge_gap=2):
                C.append({"entity_a": a, "entity_b": "table", "mode": "support", **iv(s, e), "region": {},
                          "evidence": "lowest surface point within support gap of the table top", "confidence": 0.8})
            pen = np.nan_to_num(v["pen_m"])
            for s, e in runs(pen > th["penetration_m"], merge_gap=2):
                A.append({"type": "object_support_penetration", "entities": [a, "table"], **iv(s, e),
                          "measurements": {"max_pen_m": float(pen[s:e + 1].max())},
                          "thresholds": {"penetration_m": th["penetration_m"]},
                          "severity": float(min(1, pen[s:e + 1].max() / 0.03))})
        else:
            d = np.nan_to_num(v["dist_m"], nan=1)
            on = (d < th["support_gap_m"])
            for s, e in runs(on, min_len=th["min_interval_frames"], merge_gap=2):
                ab = np.nanmedian(v["a_above_b_m"][s:e + 1])
                top, bot = (a, b) if ab > 0 else (b, a)
                C.append({"entity_a": top, "entity_b": bot, "mode": "support", **iv(s, e),
                          "region": {"centroid_height_diff_m": float(abs(ab))},
                          "evidence": "object surfaces within support gap", "confidence": 0.6})
            pen = np.nan_to_num(v["pen_m"])
            for s, e in runs(pen > th["penetration_m"], merge_gap=2):
                A.append({"type": "object_object_penetration", "entities": [a, b], **iv(s, e),
                          "measurements": {"max_pen_m": float(pen[s:e + 1].max())},
                          "thresholds": {"penetration_m": th["penetration_m"]},
                          "severity": float(min(1, pen[s:e + 1].max() / 0.03))})
    # unsupported: elevated or moving, no grasp/touch by any hand, no support by table/another object
    for n, o in tr.objects.items():
        held = np.zeros(tr.F, bool)
        supported = np.zeros(tr.F, bool)
        for key, v in pairs.items():
            if not isinstance(key, tuple):
                continue
            if key[1] == n and key[0].startswith("hand_"):
                held |= np.nan_to_num(v["dist_m"], nan=1) < 2 * th["contact_m"]
            if key == (n, "table"):
                supported |= np.nan_to_num(v["gap_m"], nan=1) < th["support_gap_m"]
            if key[0] != key[1] and n in key and not key[0].startswith("hand_") and key[1] != "table":
                supported |= np.nan_to_num(v["dist_m"], nan=1) < th["support_gap_m"]
        sp = np.nan_to_num(speed[n])
        bad = o["valid"] & ~held & ~supported
        moving_alone = o["valid"] & ~held & (sp > 3 * th["rest_speed_mps"])
        for s, e in runs(bad | moving_alone, min_len=th["min_interval_frames"], merge_gap=1):
            p = o["T_world_obj"][s:e + 1, :3, 3]
            az = np.gradient(np.gradient(p[:, 2], t[s:e + 1]), t[s:e + 1]) if e - s >= 3 else np.array([np.nan])
            A.append({"type": "unsupported_object", "entities": [n], **iv(s, e),
                      "measurements": {"elevated_frames": int(bad[s:e + 1].sum()),
                                       "moving_without_hand_frames": int(moving_alone[s:e + 1].sum()),
                                       "max_speed_mps": float(sp[s:e + 1].max()),
                                       "median_vertical_acc_mps2": float(np.nanmedian(az)),
                                       "free_fall_like": bool(np.nanmedian(az) < -6)},
                      "thresholds": {"support_gap_m": th["support_gap_m"], "contact_m": 2 * th["contact_m"],
                                     "rest_speed_mps": th["rest_speed_mps"]},
                      "severity": float(min(1, (e - s + 1) / 10))})
        # unstable rest on the table: COM outside the support polygon of the resting contact points
        tab = pairs.get((n, "table"))
        if tab is None:
            continue
        rest = o["valid"] & (sp < th["rest_speed_mps"]) & ~held & (np.nan_to_num(tab["gap_m"], nan=1) < 0.004)
        for s, e in runs(rest, min_len=10):
            f = (s + e) // 2
            lp = tab["low_pts"][f]
            if lp is None or len(lp) < 3:
                continue
            com = transform(o["T_world_obj"][f], asset(n).centroid[None])[0, :2]
            try:
                hull = ConvexHull(lp[:, :2])
                inside = Delaunay(lp[hull.vertices, :2]).find_simplex(com) >= 0
            except Exception:  # noqa: BLE001 -- degenerate (line contact)
                inside = False
            if not inside:
                A.append({"type": "unstable_rest", "entities": [n, "table"], **iv(s, e),
                          "measurements": {"support_points": int(len(lp)), "com_xy": com.tolist()},
                          "thresholds": {"rest_speed_mps": th["rest_speed_mps"], "contact_z_m": 0.004},
                          "severity": 0.6})
    return C, A
