"""Review-agent tools (README stage 8) + stage-10 apply/validate/accept/rollback.

The model reads through these tools; its only write paths queue bounded re-estimation (rerun_stage) or optimizer
constraint proposals (propose_refinement). apply_pending() executes the queue into NEW trajectory versions with
validated estimators/optimizers, re-runs diagnostics, compares, and accepts or rolls back on numbers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .. import conventions as C
from .. import io as hio
from ..diagnostics import run as drun
from ..export import versions
from ..llm import image_part
from ..physics import wrench
from ..tracking import cotracker
from ..visualization.overlay import Overlay

CODE = Path(os.environ.get("V2D_CODE", "/code"))
OBJ_OVERRIDES = {"reicp", "no_fp", "fp_low", "hold"}


class Toolbox:
    def __init__(self, run: str, cfg: dict):
        self.run = run
        self.cfg = cfg
        self.dir = hio.out_dir(run)
        self.manifest = json.loads((self.dir / "scene_manifest.json").read_text())
        tp = self.dir / "task.json"
        self.task = json.loads(tp.read_text()) if tp.exists() else None
        self.current = versions.current(run)
        self._tr = {}
        self._diag = {}
        self.findings = []
        self.pending = []
        self.tracks = cotracker.load_tracks(run)
        self.ids = {e["id"] for e in self.manifest["entities"]}
        self.overlay = Overlay(run, self.manifest, self.traj(self.current))
        self.obj_overrides = json.loads((self.dir / "object_overrides.json").read_text()) \
            if (self.dir / "object_overrides.json").exists() else {}
        self._ensure_diag(self.current)

    # ----------------------------------------------------------- state
    def traj(self, ver):
        if ver not in self._tr:
            self._tr[ver] = hio.Trajectories.load(Path(versions.load(self.run)[ver]["npz"]))
        return self._tr[ver]

    def _ensure_diag(self, ver):
        if ver not in self._diag:
            v = versions.load(self.run)[ver]
            self._diag[ver] = drun.run(self.run, ver, self.traj(ver), self.cfg, v["objects_variant"], v["hands_variant"])
            self._diag[ver]["metrics"] = drun.summary_metrics(self._diag[ver], self.traj(ver))
            if self.tracks is not None:
                self._diag[ver]["cotracker"] = cotracker.agreement(self.traj(ver), self.tracks, run_name=self.run)
        return self._diag[ver]

    def anomalies(self):
        return self._ensure_diag(self.current)["anomalies"]["anomalies"]

    def recompute_anomalies(self):
        self._ensure_diag(self.current)

    def _window(self, interval):
        t = self.traj(self.current).t_s
        a, b = sorted(float(x) for x in interval)
        i0 = int(np.searchsorted(t, a - 1e-6))
        i1 = int(np.searchsorted(t, b + 1e-6)) - 1
        i0 = min(max(0, i0), len(t) - 1)
        i1 = min(len(t) - 1, max(i1, i0))
        return i0, i1

    def _entity(self, e):
        if e not in self.ids:
            raise ValueError(f"unknown entity {e!r}; valid: {sorted(self.ids)}")
        return e

    # ----------------------------------------------------------- dispatch
    def call(self, name, args):
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            raise ValueError(f"unknown tool {name}")
        return fn(**args)

    def t_get_task_hypotheses(self):
        if self.task is None:
            return {"available": False}
        return {k: self.task.get(k) for k in ("description", "phases", "contact_hypotheses", "stacking_vs_nesting",
                                              "alternatives", "unknowns", "mapping_issues")}

    def t_get_trajectory(self, entity_id, interval):
        e = self._entity(entity_id)
        tr = self.traj(self.current)
        i0, i1 = self._window(interval)
        idx = np.unique(np.linspace(i0, i1, min(40, i1 - i0 + 1)).astype(int))
        rows = []
        if e in tr.objects:
            o = tr.objects[e]
            R0 = None
            for i in idx:
                row = {"frame": int(tr.frame[i]), "t_s": round(float(tr.t_s[i]), 3), "valid": bool(o["valid"][i]),
                       "source": C.SOURCE_CODES[int(o["source"][i])]}
                if o["valid"][i]:
                    T = o["T_world_obj"][i]
                    R0 = T[:3, :3] if R0 is None else R0
                    row["pos_mm"] = np.round(T[:3, 3] * 1000, 1).tolist()
                    row["rot_rel_deg"] = np.round(np.rad2deg(Rotation.from_matrix(R0.T @ T[:3, :3]).as_rotvec()), 1).tolist()
                rows.append(row)
        elif e in tr.hands:
            h = tr.hands[e]
            for i in idx:
                row = {"frame": int(tr.frame[i]), "t_s": round(float(tr.t_s[i]), 3), "valid": bool(h["valid"][i]),
                       "source": C.SOURCE_CODES[int(h["source"][i])]}
                if h["valid"][i]:
                    row["wrist_mm"] = np.round(h["joints"][i, 0] * 1000, 1).tolist()
                    row["thumb_tip_mm"] = np.round(h["joints"][i, 4] * 1000, 1).tolist()
                    row["index_tip_mm"] = np.round(h["joints"][i, 8] * 1000, 1).tolist()
                rows.append(row)
        else:
            return {"entity": e, "note": "static support (table top is z = 0, disk radius "
                    f"{self.manifest['support']['radius_m']} m)"}
        return {"entity": e, "version": self.current, "frame": "world z-up, mm", "rows": rows}

    def t_inspect_frames(self, interval, views=None, max_frames=3):
        views = views or ["overlay"]
        i0, i1 = self._window(interval)
        fr = np.unique(np.linspace(i0, i1, max(1, min(int(max_frames), 4))).astype(int))
        tr = self.traj(self.current)
        imgs = []
        for f in fr:
            rgb = self.overlay.frame(int(f))
            for v in views:
                im = self.overlay.reconstruction(rgb, int(f), tr) if v == "overlay" else \
                    self.overlay.labelled(rgb, int(f)) if v == "colour" else None
                if im is None:
                    continue
                imgs.append({"type": "text", "text": f"frame {f} t={tr.t_s[f]:.2f}s view={v}"})
                imgs.append(image_part(im, 640))
        return {"frames": fr.tolist(), "views": views, "_images": imgs,
                "note": "overlay: projected object surface samples + hand skeletons of the current version"}

    def t_compute_motion_metrics(self, entity_id, interval):
        e = self._entity(entity_id)
        d = self._ensure_diag(self.current)
        i0, i1 = self._window(interval)
        s = d["series"].get(e)
        if s is None:
            return {"entity": e, "note": "no motion series"}
        out = {"entity": e, "frames": [i0, i1]}
        for k, v in s.items():
            v = np.asarray(v)[i0:i1 + 1]
            if v.ndim == 1 and np.isfinite(v).any():
                out[k] = {"max": float(np.nanmax(v)), "median": float(np.nanmedian(v)),
                          "argmax_frame": int(i0 + np.nanargmax(v))}
        ct = (d.get("cotracker") or {}).get(e)
        if ct is not None:
            r = np.asarray(ct["rigid_resid_m"])[i0:i1 + 1]
            out["cotracker_rigid_resid_m"] = {"max": float(np.nanmax(r)) if np.isfinite(r).any() else None,
                                              "median": float(np.nanmedian(r)) if np.isfinite(r).any() else None,
                                              "points_median": float(np.median(np.asarray(ct["n_points"])[i0:i1 + 1]))}
        out["thresholds"] = self.cfg["diagnostics"]
        return out

    def t_check_contacts(self, interval, entity_id=None):
        d = self._ensure_diag(self.current)
        i0, i1 = self._window(interval)
        t = self.traj(self.current).t_s
        res = {}
        for k, v in d["pairs"].items():
            if not isinstance(k, tuple) or (entity_id and entity_id not in k):
                continue
            row = {}
            for kk in ("dist_m", "gap_m", "pen_m"):
                if kk in v:
                    x = np.asarray(v[kk])[i0:i1 + 1]
                    if np.isfinite(x).any():
                        row[kk] = {"min": float(np.nanmin(x)), "max": float(np.nanmax(x))}
            res["/".join(k)] = row
        cons = [c for c in d["contacts"] if c["t_end_s"] >= t[i0] and c["t_start_s"] <= t[i1]
                and (not entity_id or entity_id in (c["entity_a"], c["entity_b"]))]
        slip = [a for a in d["anomalies"]["anomalies"] if a["type"] == "contact_slip"
                and a["t_end_s"] >= t[i0] and a["t_start_s"] <= t[i1]]
        return {"window_frames": [i0, i1], "pairs": res, "contact_intervals": cons, "slip": slip}

    def t_check_physics(self, interval, entity_id, parameter_ranges=None):
        e = self._entity(entity_id)
        tr = self.traj(self.current)
        if e not in tr.objects:
            return {"error": "physics feasibility is computed for rigid objects"}
        i0, i1 = self._window(interval)
        pr = parameter_ranges or {}
        mus = [float(x) for x in pr.get("mu", [0.3, 0.6, 1.0])][:4]
        ms = [float(x) for x in pr.get("mass_kg", [0.05, 0.3, 0.6])][:4]
        lo, hi = self.cfg["physics"]["mu_range"], self.cfg["physics"]["mass_kg_range"]
        mus = [min(max(m, lo[0] / 3), 1.5) for m in mus]
        ms = [min(max(m, 0.01), hi[1] * 3) for m in ms]
        return wrench.feasibility(tr, e, range(i0, i1 + 1), mus, ms)

    def t_rerun_stage(self, stage, entity_id, interval, overrides, reason):
        e = self._entity(entity_id)
        i0, i1 = self._window(interval)
        if len(self.pending) >= 6:
            return {"queued": False, "error": "pending queue full (6) for this round"}
        if stage == "object_fusion":
            kind = overrides.get("kind") or next((k for k in overrides if k in OBJ_OVERRIDES), None)
            if kind not in OBJ_OVERRIDES or e not in self.traj(self.current).objects:
                return {"queued": False, "error": f"object_fusion needs an object and overrides.kind in {sorted(OBJ_OVERRIDES)}"}
            item = {"stage": stage, "entity": e, "frames": [i0, i1], "kind": kind, "reason": reason}
        elif stage == "hand_fit":
            sq = float(overrides.get("squeeze_mm", 1.5))
            cm = float(overrides.get("contact_mm", 15))
            if sq not in (1.5, 3.0, 5.0) or cm not in (15.0, 30.0) or not e.startswith("hand_"):
                return {"queued": False, "error": "hand_fit needs a hand entity, squeeze_mm in {1.5,3,5}, contact_mm in {15,30}"}
            item = {"stage": stage, "entity": e, "frames": [i0, i1], "squeeze_mm": sq, "contact_mm": cm, "reason": reason}
        else:
            return {"queued": False, "error": "unknown stage"}
        self.pending.append(item)
        return {"queued": True, "item": item, "note": "applied after this round; accepted only if checks improve"}

    def t_propose_refinement(self, interval, constraints, evidence):
        i0, i1 = self._window(interval)
        ok = []
        for c in constraints:
            if c.get("kind") not in ("contact_on", "contact_off", "static", "support", "keep_symmetric"):
                continue
            if c.get("entity_a") not in self.ids or (c.get("entity_b") and c["entity_b"] not in self.ids):
                continue
            ok.append({k: c[k] for k in ("kind", "entity_a", "entity_b") if k in c})
        if not ok:
            return {"queued": False, "error": "no valid constraints (check kinds and entity ids)"}
        if len(self.pending) >= 6:
            return {"queued": False, "error": "pending queue full (6) for this round"}
        item = {"stage": "optimize", "frames": [i0, i1], "constraints": ok, "evidence": evidence}
        self.pending.append(item)
        return {"queued": True, "item": item}

    def t_compare_versions(self, before, after, interval=None):
        reg = versions.load(self.run)
        for v in (before, after):
            if v not in reg:
                return {"error": f"unknown version {v}; known: {[k for k in reg if not k.startswith('_')]}"}
        return compare(self, before, after, self._window(interval) if interval else None)

    def t_record_finding(self, **kw):
        kw["recorded"] = hio.now()
        self.findings.append(kw)
        return {"recorded": True, "n_findings": len(self.findings)}

    def t_finish(self, summary):
        return {"finished": True}

    # ----------------------------------------------------------- stage 10
    def apply_pending(self, cfg_review):
        decisions = []
        items, self.pending = self.pending, []
        for k, it in enumerate(items):
            name = f"r{len(versions.load(self.run))}_{it['stage']}_{k}"
            prev_ovr = json.loads(json.dumps(self.obj_overrides))
            try:
                ok = self._apply(it, name)
            except Exception as ex:  # noqa: BLE001 -- a failed repair is a rejected repair
                decisions.append({"item": it, "version": name, "accepted": False, "error": repr(ex)[:400]})
                continue
            if not ok:
                decisions.append({"item": it, "version": name, "accepted": False, "error": "estimator produced nothing"})
                continue
            cmp_ = compare(self, self.current, name, (max(0, it["frames"][0] - 3), it["frames"][1] + 3))
            acc = accept(cmp_, self.cfg["acceptance"])
            versions.set_accepted(self.run, name, acc["accepted"], {"compare": cmp_, **acc})
            decisions.append({"item": it, "version": name, **acc, "compare": cmp_["summary"]})
            if acc["accepted"]:
                self.current = name
                if it["stage"] == "object_fusion":
                    hio.write_json(self.dir / "object_overrides.json", self.obj_overrides)
            else:
                self.obj_overrides = prev_ovr
                versions.reject_cleanup(self.run, name)
        return decisions

    def _apply(self, it, name):
        cur = self.traj(self.current)
        reg = versions.load(self.run)
        base_h = reg[self.current]["hands_variant"]
        if it["stage"] == "object_fusion":
            nm, (lo, hi), kind = it["entity"], it["frames"], it["kind"]
            ov = json.loads(json.dumps(self.obj_overrides.get(nm, {})))
            if kind == "reicp":
                icp_dir = f"stereo_icp_hoi4d_{lo}_{hi}"
                _py(CODE / "stereo/object_icp.py", [self.run, "--init_variant", reg[self.current]["objects_variant"],
                                                    "--out_name", icp_dir, "--frames", lo, hi, "--only_objects", nm])
                ov["icp_extra"] = ov.get("icp_extra", []) + [icp_dir]
                ov["icp_mult"] = ov.get("icp_mult", []) + [[lo, hi, 3.0]]
            elif kind == "no_fp":
                ov["fp_mult"] = ov.get("fp_mult", []) + [[lo, hi, 0.0]]
            elif kind == "fp_low":
                ov["fp_mult"] = ov.get("fp_mult", []) + [[lo, hi, 0.25]]
            elif kind == "hold":
                ov["static"] = ov.get("static", []) + [[lo, hi, 1.0]]
            allov = {**self.obj_overrides, nm: ov}
            of = self.dir / f"ovr_{name}.json"
            hio.write_json(of, allov)
            tv = f"hoi4d_fuse_{name}"
            _py(CODE / "stereo/fuse_objects.py", [self.run, "--overrides", of, "--out_variant", tv, "--only_objects", nm])
            pd_ = hio.ROOT / "runs" / self.run / tv / nm / "poses"
            if not pd_.is_dir() or not any(pd_.glob("*.json")):
                return False
            new = cur.copy()
            T_wc = cur.extra["T_world_cam"]
            T = np.full((cur.F, 4, 4), np.nan)
            for f in range(cur.F):
                p = pd_ / f"{f:06d}.json"
                if p.exists() and np.isfinite(T_wc[f]).all():
                    d = json.loads(p.read_text())
                    M = np.eye(4); M[:3, :3] = d["R"]; M[:3, 3] = d["T"]
                    T[f] = T_wc[f] @ M
                    U, _, Vt = np.linalg.svd(T[f, :3, :3]); T[f, :3, :3] = U @ Vt
            valid = np.isfinite(T).all(axis=(1, 2))
            src = cur.objects[nm]["source"].copy()
            src[lo:hi + 1][valid[lo:hi + 1]] = C.SOURCE_IDS["refined"]
            src[~valid] = 0
            new.add_object(nm, T, valid, src)
            shutil.rmtree(hio.ROOT / "runs" / self.run / tv, ignore_errors=True)
            versions.write_variant(self.run, name, new, self.current, {"stage": "object_fusion", **it}, base_h)
            self.obj_overrides = allov  # kept only if accepted (restored on reject below)
            return True
        if it["stage"] == "hand_fit":
            side = it["entity"].split("_")[1]
            out = f"hoi4d_hf_{name}"
            _py(CODE / "stereo/s06h_grasp_plausible.py", [self.run, "--variant", reg[self.current]["objects_variant"],
                                                         "--hands", base_h, "--squeeze_mm", it["squeeze_mm"],
                                                         "--contact_mm", it["contact_mm"], "--out", out])
            new = cur.copy()
            lo, hi = it["frames"]
            ramp = 4  # blend in/out over 4 frames: no step at the window edges (user ruling: no jitter)
            hd = new.hands[it["entity"]]
            T_wc = cur.extra["T_world_cam"]
            for f in range(max(0, lo - ramp), min(cur.F, hi + ramp + 1)):
                p = hio.ROOT / "runs" / self.run / "stereo" / out / side / f"{f:06d}.json"
                if not p.exists() or not hd["valid"][f]:
                    continue
                d = json.loads(p.read_text())
                w = 1.0 if lo <= f <= hi else max(0.0, 1 - min(abs(f - lo), abs(f - hi)) / (ramp + 1))
                Jw = np.asarray(d["joints"]) @ T_wc[f, :3, :3].T + T_wc[f, :3, 3]
                hd["joints"][f] = (1 - w) * hd["joints"][f] + w * Jw
                hd["transl"][f] = (1 - w) * hd["transl"][f] + w * np.asarray(d["T"])
                hd["source"][f] = C.SOURCE_IDS["refined"]
            shutil.rmtree(hio.ROOT / "runs" / self.run / "stereo" / out, ignore_errors=True)
            versions.write_variant(self.run, name, new, self.current, {"stage": "hand_fit", **it}, base_h)
            return True
        if it["stage"] == "optimize":
            from ..physics import optimize
            optimize.run(self.run, self.current, self.cfg, [it], name, accept_registry=False)
            return True
        return False


def _py(script, args):
    r = subprocess.run([sys.executable, str(script)] + [str(x) for x in args], capture_output=True, text=True,
                       env={**os.environ, "V2D_OBJ_DIR": "objects_sam3"})
    if r.returncode:
        raise RuntimeError(f"{Path(script).name} failed: {r.stderr[-600:]}")
    return r.stdout


def compare(tb: Toolbox, before: str, after: str, window=None) -> dict:
    """Numbers only: anomaly severity (all / window), penetration, jitter, CoTracker agreement, visual critic."""
    db, da = tb._ensure_diag(before), tb._ensure_diag(after)
    t = tb.traj(before).t_s
    if window:  # repair windows are padded by +-3 frames: clamp to the sequence
        window = (min(max(0, int(window[0])), len(t) - 1), min(max(0, int(window[1])), len(t) - 1))

    def sev(d, w):
        an = d["anomalies"]["anomalies"]
        if w:
            an = [a for a in an if a["t_end_s"] >= t[w[0]] and a["t_start_s"] <= t[w[1]]]
        return float(sum(a["severity"] for a in an)), len(an)

    def ct(d, w):
        vals = []
        for n, v in (d.get("cotracker") or {}).items():
            r = np.asarray(v["rigid_resid_m"])
            r = r[w[0]:w[1] + 1] if w else r
            if np.isfinite(r).any():
                vals.append(float(np.nanmean(r)))
        return float(np.mean(vals)) if vals else None

    out = {"before": before, "after": after, "window": window,
           "severity_all": [sev(db, None)[0], sev(da, None)[0]],
           "severity_window": [sev(db, window)[0], sev(da, window)[0]] if window else None,
           "cotracker_resid_all_m": [ct(db, None), ct(da, None)],
           "cotracker_resid_window_m": [ct(db, window), ct(da, window)] if window else None,
           "penetration_before": db["metrics"]["penetration"], "penetration_after": da["metrics"]["penetration"],
           "median_acc_before": db["metrics"]["median_acc_mps2"], "median_acc_after": da["metrics"]["median_acc_mps2"]}
    vis = visual_scores(tb, before, after, window)
    out["visual"] = vis
    # per entity (README 10.5: accept only repairs whose declared checks improve -> decided per entity)
    ents = sorted(set(tb.traj(before).objects) | set(tb.traj(before).hands))
    per = {}
    for e in ents:
        def sev_e(d):
            return float(sum(a["severity"] for a in d["anomalies"]["anomalies"] if e in a["entities"]))
        row = {"severity": [sev_e(db), sev_e(da)]}
        for tag, d in (("before", db), ("after", da)):
            r = (d.get("cotracker") or {}).get(e)
            if r is not None and np.isfinite(np.asarray(r["rigid_resid_m"], float)).any():
                row.setdefault("cotracker_resid_m", []).append(float(np.nanmean(r["rigid_resid_m"])))
        if e in vis.get("per_object", {}):
            row["visual_error_prob"] = vis["per_object"][e]
        if e in tb.traj(before).hands:
            row.update(hand_checks(tb.traj(before).hands[e], tb.traj(after).hands[e], tb.traj(before).t_s))
        per[e] = row
    out["per_entity"] = per
    out["summary"] = {k: out[k] for k in ("severity_all", "severity_window", "cotracker_resid_all_m",
                                          "cotracker_resid_window_m")} | {"visual": vis}
    return out


def visual_scores(tb, before, after, window):
    """Depth/mask fit of the objects (existing calibrated critic, refine_agent/verify.py; no mocap)."""
    sys.path.insert(0, str(CODE / "refine_agent"))
    import verify  # noqa: E402
    from agent import calibration, objective, per_frame  # noqa: E402
    reg = versions.load(tb.run)
    out = {}
    cal = calibration(tb.run)
    for tag, v in (("before", before), ("after", after)):
        rep = verify.main([tb.run, "--objects", reg[v]["objects_variant"], "--no_hands", "--quiet"])[tb.run]
        tot, totw = [], []
        for nm in tb.traj(v).objects:
            p, alone = per_frame(rep, nm, cal)
            j = objective(p, alone)
            out.setdefault("per_object", {}).setdefault(nm, []).append(j)
            if j is not None:
                tot.append(j)
            if window:
                jw = objective(p, alone, window[0], window[1])
                if jw is not None:
                    totw.append(jw)
        out[tag] = {"error_prob_all": float(np.mean(tot)) if tot else None,
                    "error_prob_window": float(np.mean(totw)) if totw else None}
    return out


def hand_checks(h0, h1, t):
    """User rulings for hands: observed (detected) frames stay at the observation; no added finger jitter."""
    obs = (h0["source"] == 1) & h0["valid"] & h1["valid"]
    rep = h1.get("repaired")
    if rep is not None:  # spin-repaired frames: the fit there was a flip contradicted by the hand tracks
        obs &= ~rep.astype(bool)
    dev = np.linalg.norm(h1["joints"][obs] - h0["joints"][obs], axis=-1)
    dt2 = float(np.median(np.diff(t))) ** 2

    def fing(h):
        rel = h["joints"][:, [4, 8, 12, 16, 20]] - h["joints"][:, :1]
        a = np.linalg.norm(np.diff(rel, 2, axis=0), axis=-1) / dt2
        return float(np.nanpercentile(a, 95))
    return {"observed_joint_dev_p95_m": float(np.percentile(dev, 95)) if dev.size else 0.0,
            "finger_rel_acc_p95": [fing(h0), fing(h1)]}


def accept_entities(c: dict, acc: dict) -> dict:
    """Per-entity decision: its own anomaly severity must drop (or CoTracker residual drop >= min_improvement with
    severity not growing) and, for objects, its visual error probability may grow <= max_visual_degradation."""
    out = {}
    for e, r in c["per_entity"].items():
        s0, s1 = r["severity"]
        ct = r.get("cotracker_resid_m")
        ct_gain = (ct[0] - ct[1]) / max(ct[0], 1e-6) if ct and len(ct) == 2 else 0.0
        v = r.get("visual_error_prob")
        vdeg = None if not v or None in v else (v[1] - v[0]) / max(v[0], 1e-3)
        if "finger_rel_acc_p95" in r:  # hands: own severity must drop or tracking improve without severity growing
            better = (s1 < s0 - 1e-6) or (ct_gain >= acc["min_improvement"] and s1 <= s0 + 1e-6)
            ok = better and (vdeg is None or vdeg <= acc["max_visual_degradation"])
        else:
            # objects, calibrated on 20 public episodes x 2 trackers vs mocap (beat1/calib_accept.py, validation
            # only): anomaly severity was uninformative (dropping it changed nothing; its depth term rejected
            # mocap-better poses); tracking agreement + image evidence: 9 better / 2 worse accepted, 0 better
            # rejected (old gate: 8 / 6 / 1).
            ok = ct_gain > acc.get("object_track_gain_min", 0.10) and vdeg is not None \
                and vdeg < acc.get("object_visual_degradation_max", 0.05)
        if "finger_rel_acc_p95" in r:  # hands: stay on the observations, no added jitter
            fa = r["finger_rel_acc_p95"]
            ok = ok and r["observed_joint_dev_p95_m"] <= acc.get("hand_obs_dev_p95_m", 0.01) \
                and fa[1] <= fa[0] * (1 + acc.get("hand_jitter_tol", 0.05))
        out[e] = {"accepted": bool(ok), "severity": r["severity"], "cotracker_gain": ct_gain, "visual_degradation": vdeg}
    return out


def accept(c: dict, acc: dict) -> dict:
    """Stage 10 acceptance: targeted residual (window severity) must drop >= min_improvement, visual evidence must
    not degrade beyond tolerance, whole-sequence severity must not grow."""
    sw = c["severity_window"] or c["severity_all"]
    improve = (sw[0] - sw[1]) / max(sw[0], 1e-6)
    vb, va = c["visual"]["before"], c["visual"]["after"]
    vk = "error_prob_window" if c["window"] else "error_prob_all"
    vdeg = None if vb[vk] is None or va[vk] is None else (va[vk] - vb[vk]) / max(vb[vk], 1e-3)
    gall = c["severity_all"][1] - c["severity_all"][0]
    need = acc["min_improvement"] if c["window"] else 1e-6  # whole-sequence repair: checks must improve at all
    ok = improve >= need and (vdeg is None or vdeg <= acc["max_visual_degradation"]) \
        and gall <= acc["max_global_degradation"] + 1e-9
    hand_fail = [e for e, r in c.get("per_entity", {}).items() if "finger_rel_acc_p95" in r and not (
        r["observed_joint_dev_p95_m"] <= acc.get("hand_obs_dev_p95_m", 0.01)
        and r["finger_rel_acc_p95"][1] <= r["finger_rel_acc_p95"][0] * (1 + acc.get("hand_jitter_tol", 0.05)))]
    ok = ok and not hand_fail
    return {"accepted": bool(ok), "targeted_improvement": float(improve), "visual_degradation": vdeg,
            "hands_failing_user_rules": hand_fail,
            "global_severity_change": float(gall),
            "rule": f"window improvement >= {acc['min_improvement']}, visual degradation <= "
                    f"{acc['max_visual_degradation']}, global severity change <= {acc['max_global_degradation']}"}
