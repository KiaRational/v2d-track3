"""Rest consistency: an object nobody touches cannot move.

A typical failure (eval ep30 sweeper): the object rests untouched for the first frames with a WRONG pose (stable,
so no motion check fires), then the estimator snaps to the right pose the moment the hand arrives -- a 106 deg
turn in one frame at 0.9 cm hand distance. Because the hand is there, the snap looks like a grasp.

find_jumps():  per object, a pose step larger than jump_rot_deg / jump_trans_m within one frame, inside the
               contact onset (hand joints within contact_m of the surface around the jump) or release window,
               with the other side of the jump resting (static, untouched) for >= min_rest frames and the near
               side stable for >= min_stable frames.
repair():      for each jump, both candidates -- the current rest poses and the pose from the stable side of the
               jump held constant -- are scored on the REST frames against the camera evidence (silhouette distance
               of projected surface samples to the object's SAM 3 mask + stereo depth residual). The rest segment
               takes the stable-side pose only if it explains those frames clearly better (score <= accept_ratio
               of the current); otherwise the jump is reported as unresolved for the review agent.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as Rot

from .. import io as hio
from ..geometry.assets import asset, transform

DS = 4  # evidence downsampling


def _contact_dist(tr, n, S):
    o = tr.objects[n]
    d = np.full(tr.F, np.inf)
    for hd in tr.hands.values():
        for f in np.where(o["valid"] & hd["valid"])[0]:
            P = transform(o["T_world_obj"][f], S)
            d[f] = min(d[f], np.min(np.linalg.norm(hd["joints"][f][:, None] - P[None], axis=-1)))
    return d


def _steps(T, valid):
    F = len(T)
    dp, dr = np.full(F, np.nan), np.full(F, np.nan)  # step f -> f+1 stored at f
    for f in range(F - 1):
        if valid[f] and valid[f + 1]:
            dp[f] = np.linalg.norm(T[f + 1][:3, 3] - T[f][:3, 3])
            dr[f] = np.degrees(np.linalg.norm(Rot.from_matrix(T[f][:3, :3].T @ T[f + 1][:3, :3]).as_rotvec()))
    return dp, dr


def find_jumps(tr, cfg) -> list[dict]:
    c = {"jump_rot_deg": 30.0, "jump_trans_m": 0.03, "contact_m": 0.02, "window": 3, "rest_step_m": 0.005,
         "rest_step_deg": 3.0, "min_rest": 5, "min_stable": 3, **(cfg or {})}
    out = []
    for n, o in tr.objects.items():
        T, V = o["T_world_obj"], o["valid"]
        S = asset(n).surface[::10]
        dc = _contact_dist(tr, n, S)
        touched = dc < c["contact_m"]
        dp, dr = _steps(T, V)
        calm = (dp < c["rest_step_m"]) & (dr < c["rest_step_deg"])
        for f in range(tr.F - 1):
            if not (dr[f] > c["jump_rot_deg"] or dp[f] > c["jump_trans_m"]):
                continue
            near = touched[max(0, f - c["window"]): f + c["window"] + 2].any()
            if not near:
                continue  # untouched jumps are already 'jump' anomalies in diagnostics/motion
            # onset: resting + untouched BEFORE the jump, stable after
            a = f
            while a - 1 >= 0 and calm[a - 1] and (not touched[a - 1] or f - (a - 1) <= c["window"] + 2):
                a -= 1
            b = f + 1
            while b + 1 < tr.F and calm[b] and b - f <= c["min_stable"] + 2:
                b += 1
            if f - a + 1 >= c["min_rest"] and b - (f + 1) >= c["min_stable"] - 1:
                out.append({"object": n, "kind": "onset", "jump_frame": f, "rest": [a, f], "stable": [f + 1, b],
                            "jump_m": float(dp[f]), "jump_deg": float(dr[f]), "hand_dist_m": float(dc[f])})
                continue
            # release: stable BEFORE the jump (while held / at release), resting + untouched after it
            a = f + 1
            while a + 1 < tr.F and calm[a] and (not touched[a + 1] or (a + 1) - f <= c["window"] + 2):
                a += 1
            b = f
            while b - 1 >= 0 and calm[b - 1] and f - b <= c["min_stable"] + 2:
                b -= 1
            if a - f >= c["min_rest"] and f - b + 1 >= c["min_stable"]:
                out.append({"object": n, "kind": "release", "jump_frame": f, "rest": [f + 1, a], "stable": [b, f],
                            "jump_m": float(dp[f]), "jump_deg": float(dr[f]), "hand_dist_m": float(dc[f])})
    return out


class _Evidence:
    def __init__(self, run_name, man):
        self.run = hio.ROOT / "runs" / run_name
        self.K = np.asarray(man["cameras"]["ego_cam_a"]["K"], float)
        self.mid = {e["id"]: e.get("mask_id") for e in man["entities"]}
        self._c = {}

    def frame(self, n, f):
        k = (n, f)
        if k not in self._c:
            mp = self.run / hio.obj_dir() / "masks" / str(self.mid.get(n)) / f"{f:06d}.png"
            dp = self.run / "depth_stereo" / f"{f:06d}.png"
            if not mp.exists():
                self._c[k] = None
            else:
                m = np.asarray(Image.open(mp))[::DS, ::DS] > 0
                D = None
                if dp.exists():
                    png = np.asarray(Image.open(dp))[::DS, ::DS].astype(float)
                    D = np.where((png > 0) & (png < 65535), 65535.0 / np.maximum(png, 1) - 1.0, np.nan)
                self._c[k] = (m, distance_transform_edt(~m) if m.any() else None, D)
        return self._c[k]

    def score(self, n, f, T_world_obj, T_wc, S):
        """Silhouette (mean px outside the mask, full-res px) + depth (median |z - stereo| of in-mask samples, cm)."""
        ev = self.frame(n, f)
        if ev is None or ev[1] is None:
            return None
        m, dt, D = ev
        Pc = transform(np.linalg.inv(T_wc), transform(T_world_obj, S))
        z = Pc[:, 2]
        ok = z > 0.05
        u = (self.K[0, 0] * Pc[ok, 0] / z[ok] + self.K[0, 2]) / DS
        v = (self.K[1, 1] * Pc[ok, 1] / z[ok] + self.K[1, 2]) / DS
        h, w = m.shape
        ui, vi = np.clip(u.round().astype(int), 0, w - 1), np.clip(v.round().astype(int), 0, h - 1)
        sil = float(np.mean(dt[vi, ui])) * DS
        dep = 0.0
        if D is not None:
            inm = m[vi, ui] & np.isfinite(D[vi, ui])
            if inm.sum() >= 10:
                # nearest surface only: samples behind the front face are compared to the front depth
                dep = float(np.median(np.clip(z[ok][inm] - D[vi[inm], ui[inm]], 0, None))) * 100
        return sil / 10.0 + dep


def repair(run_name, man, tr, cfg=None):
    """Returns (new Trajectories, report). Only rest segments whose evidence clearly prefers the held pose change."""
    from ..conventions import SOURCE_IDS
    c = {"accept_ratio": 0.8, **(cfg or {})}
    jumps = find_jumps(tr, c)
    new = tr.copy()
    ev = _Evidence(run_name, man)
    T_wc = tr.extra["T_world_cam"]
    rep = []
    for j in jumps:
        n = j["object"]
        o = new.objects[n]
        S = asset(n).surface[::8]
        s0, s1 = j["stable"]
        ref = s0 + 1 if j["kind"] == "onset" else s1 - 1  # a frame inside the stable side, off the jump itself
        ref = int(np.clip(ref, s0, s1))
        Tc = o["T_world_obj"][ref].copy()
        a, b = j["rest"]
        cur, cand = [], []
        for f in range(a, b + 1):
            if not o["valid"][f] or not np.isfinite(T_wc[f]).all():
                continue
            x = ev.score(n, f, o["T_world_obj"][f], T_wc[f], S)
            y = ev.score(n, f, Tc, T_wc[f], S)
            if x is not None and y is not None:
                cur.append(x); cand.append(y)
        r = {**j, "frames_scored": len(cur), "score_current": float(np.mean(cur)) if cur else None,
             "score_held": float(np.mean(cand)) if cand else None, "applied": False}
        if cur and np.mean(cand) <= c["accept_ratio"] * np.mean(cur):
            for f in range(a, b + 1):
                if o["valid"][f]:
                    o["T_world_obj"][f] = Tc
                    o["source"][f] = SOURCE_IDS["refined"]
            r["applied"] = True
        rep.append(r)
    return new, rep
