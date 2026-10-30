"""README stage 7 driver: deterministic diagnostics of one trajectory version -> anomalies.json, contacts.json,
diag_<version>.npz (per-frame series for the agent tools / plots).

Own checks (motion, anatomy, contacts/support/penetration/slip/stability) + the existing observation checks,
read from their reports when present for the same variants (refine_agent/verify.py: depth/mask/free-space,
hand triangulation/2D; stereo/wrench_feasibility.py; stereo/physics_check.py --stability; CoTracker agreement).
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

from .. import io as hio
from ..geometry.assets import asset
from . import contacts as hc
from . import motion as hm

VERIFY_MAP = {"object_in_free_space": "depth_disagreement", "object_depth_mismatch": "depth_disagreement",
              "object_mask_mismatch": "silhouette", "object_jitter": "high_acceleration",
              "object_rot_jitter": "rotation_jump", "object_in_table": "object_support_penetration",
              "hand_off_image": "reprojection", "hand_off_stereo": "cross_view_disagreement",
              "hand_penetration": "hand_object_penetration", "object_moved_alone": "unsupported_object",
              "object_unsupported": "unsupported_object"}


def symmetry_group(run_name, name):
    rep = hio.ROOT / "runs" / run_name / "fusion_report.json"
    if rep.exists():
        d = json.loads(rep.read_text()).get(name, {})
        if d.get("symmetry") == "continuous" and d.get("sym_axis") is not None:
            from scipy.spatial.transform import Rotation
            ax = np.asarray(d["sym_axis"], float)
            return [Rotation.from_rotvec(ax / np.linalg.norm(ax) * a).as_matrix() for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)]
        if d.get("sym_group", 0) > 1:  # fusion_report stores only the group size: rebuild the rotations from the mesh
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "stereo"))
            from symmetry import mesh_symmetry
            return [np.asarray(g) for g in mesh_symmetry(hio.ROOT / os.environ.get("V2D_MESH_DIR", "meshes_obj") / name / f"{name}.obj")["group"]]
    return None


def run(run_name: str, version: str, tr: hio.Trajectories, cfg: dict, objects_variant: str, hands_variant: str,
        stride: int = 1) -> dict:
    th = cfg["diagnostics"]
    t, fr = tr.t_s, tr.frame
    an, series = [], {}
    speed = {}
    for n, o in tr.objects.items():
        sym = symmetry_group(run_name, n)
        m = hm.object_motion(o["T_world_obj"], o["valid"], t, sym)
        speed[n] = m["speed_mps"]
        series[n] = m
        an += hm.motion_anomalies(n, m, t, fr, th, o.get("sigma_t_m"))
        miss = ~o["valid"]
        for s, e in hm.runs(miss, min_len=1):
            an.append({"type": "tracking_loss", "entities": [n], "t_start_s": float(t[s]), "t_end_s": float(t[e]),
                       "frames": [int(fr[s]), int(fr[e])], "measurements": {"frames_missing": int(e - s + 1)},
                       "thresholds": {}, "severity": float(min(1, (e - s + 1) / 20))})
        interp = o["source"] == 3
        for s, e in hm.runs(interp, min_len=5):
            an.append({"type": "tracking_loss", "entities": [n], "t_start_s": float(t[s]), "t_end_s": float(t[e]),
                       "frames": [int(fr[s]), int(fr[e])],
                       "measurements": {"frames_interpolated_no_measurement": int(e - s + 1)},
                       "thresholds": {"min_frames": 5}, "severity": float(min(0.5, (e - s + 1) / 40))})
    for h, hd in tr.hands.items():
        Tj = np.tile(np.eye(4), (tr.F, 1, 1))
        Tj[:, :3, 3] = hd["joints"][:, 0]
        m = hm.object_motion(Tj, hd["valid"], t)
        series[h] = m
        an += [a for a in hm.motion_anomalies(h, m, t, fr, th) if a["type"] in ("translation_jump", "high_acceleration")]
        anat = hm.hand_anatomy(hd["joints"], hd["valid"], hd["hand_pose"])
        series[h].update(anat)
        bad = (np.nan_to_num(anat["joint_angle_max_rad"]) > 2.0) | (np.nan_to_num(anat["finger_min_dist_m"], nan=1) < 0.008)
        for s, e in hm.runs(bad, merge_gap=1):
            an.append({"type": "hand_anatomy", "entities": [h], "t_start_s": float(t[s]), "t_end_s": float(t[e]),
                       "frames": [int(fr[s]), int(fr[e])],
                       "measurements": {"max_joint_angle_rad": float(np.nanmax(anat["joint_angle_max_rad"][s:e + 1])),
                                        "min_finger_distance_m": float(np.nanmin(anat["finger_min_dist_m"][s:e + 1]))},
                       "thresholds": {"joint_angle_rad": 2.0, "finger_distance_m": 0.008}, "severity": 0.5})
    pairs = hc.frame_pairs(tr, cfg, stride)
    contacts, an_c = hc.contacts_and_anomalies(tr, pairs, cfg, speed)
    an += an_c
    an += imported(run_name, objects_variant, hands_variant, tr)
    for i, a in enumerate(sorted(an, key=lambda a: (a["t_start_s"], a["type"]))):
        a["id"] = f"{version}-A{i:03d}"
        a.setdefault("status", "candidate")
        a.setdefault("evidence", {"version": version, "frames": a["frames"]})
    an = sorted(an, key=lambda a: a["id"])
    out = {"version": version, "objects_variant": objects_variant, "hands_variant": hands_variant,
           "thresholds": th, "anomalies": an,
           "note": "candidates with measurements; fast motion, symmetry, depth noise and unseen contacts can look "
                   "like failures"}
    hio.write_artifact(run_name, "anomalies", out, f"anomalies_{version}.json")
    hio.write_artifact(run_name, "contacts", {"version": version, "contacts": contacts}, f"contacts_{version}.json")
    np.savez_compressed(hio.out_dir(run_name) / f"diag_{version}.npz",
                        **{f"{k}/{kk}": vv for k, v in series.items() for kk, vv in v.items()},
                        **{f"pair/{a}/{b}/{kk}": vv for (a, b), v in ((k, v) for k, v in pairs.items() if isinstance(k, tuple))
                           for kk, vv in v.items() if kk not in ("low_pts", "tips_obj")})
    return {"anomalies": out, "contacts": contacts, "pairs": pairs, "series": series}


def imported(run_name, objects_variant, hands_variant, tr) -> list[dict]:
    """Existing observation/physics checks for exactly these variants (skipped when absent)."""
    r = hio.ROOT / "runs" / run_name
    out = []
    t, fr = tr.t_s, tr.frame
    vp = r / "refine" / f"verify_hoi4d_{objects_variant}.json"
    if vp.exists():
        rep = json.loads(vp.read_text())
        for it in rep.get("issues", []):
            typ = VERIFY_MAP.get(it["type"])
            if typ is None:
                continue
            a, b = it["frames"]
            ent = it["target"] if it["target"] in tr.objects else f"hand_{it['target']}"
            out.append({"type": typ, "entities": [ent], "t_start_s": float(t[a]), "t_end_s": float(t[b]),
                        "frames": [int(fr[a]), int(fr[b])],
                        "measurements": {it["metric"]: it["value_mean"], "source": "refine_agent/verify.py"},
                        "thresholds": {it["metric"]: it["threshold"]},
                        "severity": float(min(1, it["severity"] / 2))})
    wp = r / "stereo" / f"wrench_{hands_variant}.json"
    if wp.exists():
        rep = json.loads(wp.read_text())
        for n, d in rep.get("objects", {}).items():
            pf = d.get("per_frame", [])
            bad = np.zeros(tr.F, bool)
            for row in pf:
                res = row.get("residual", {}).get("1.0")
                if res is not None and res > rep.get("tol", 0.1):
                    bad[int(row["t"])] = True
            for s, e in hm.runs(bad, min_len=2, merge_gap=2):
                out.append({"type": "wrench_infeasible", "entities": [n], "t_start_s": float(t[s]), "t_end_s": float(t[e]),
                            "frames": [int(fr[s]), int(fr[e])],
                            "measurements": {"frames": int(bad[s:e + 1].sum()), "mu": 1.0,
                                             "source": "stereo/wrench_feasibility.py"},
                            "thresholds": {"residual_tol": rep.get("tol", 0.1)}, "severity": 0.7})
    from ..tracking import cotracker
    tk = cotracker.load_tracks(run_name)
    if tk is not None:
        for n, d in cotracker.agreement(tr, tk, run_name=run_name).items():
            res = np.asarray(d["rigid_resid_m"], float)
            bad = np.nan_to_num(res) > d.get("threshold_m", 0.015)
            for s, e in hm.runs(bad, min_len=2, merge_gap=1):
                out.append({"type": "depth_disagreement", "entities": [n],
                            "t_start_s": float(t[s]), "t_end_s": float(t[e]), "frames": [int(fr[s]), int(fr[e])],
                            "measurements": {"cotracker_rigid_resid_m": float(np.nanmax(res[s:e + 1])),
                                             "source": "CoTracker3 tracks lifted by stereo depth"},
                            "thresholds": {"rigid_resid_m": d.get("threshold_m", 0.015)}, "severity": 0.6})
    return out


def summary_metrics(diag: dict, tr) -> dict:
    an = diag["anomalies"]["anomalies"]
    pairs = diag["pairs"]
    pen = {}
    for k, v in pairs.items():
        if isinstance(k, tuple):
            x = np.nan_to_num(v["pen_m"])
            pen["/".join(k)] = {"max_m": float(x.max()), "frames_over_6mm": int((x > 0.006).sum())}
    jit = {n: float(np.nanmedian(s["acc_mps2"])) for n, s in diag["series"].items() if "acc_mps2" in s}
    by_type = {}
    for a in an:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    return {"anomaly_count": len(an), "anomalies_by_type": by_type,
            "severity_sum": float(sum(a["severity"] for a in an)), "penetration": pen,
            "median_acc_mps2": jit, "contacts": len(diag["contacts"])}
