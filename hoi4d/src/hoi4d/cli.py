"""hoi4d command line (run inside the agile image: torch, pytorch3d, trimesh, scipy).

  python -m hoi4d.cli ingest   <run>            scene_manifest.json + trajectories_initial.npz (+ version registry)
  python -m hoi4d.cli diagnose <run> [--version V] [--stride 1]   anomalies_V.json, contacts_V.json, diag_V.npz
  python -m hoi4d.cli task     <run>            task.json via the local VLM server (must be running)
  python -m hoi4d.cli review   <run>            bounded GPT-Astra-role review loop (server must be running)
  python -m hoi4d.cli optimize <run> [--version V] [--proposals F]   staged physics/contact optimisation -> new version
  python -m hoi4d.cli export   <run>            final artifacts: trajectories_refined.npz, contacts.json, anomalies.json,
                                                metrics.json, run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from . import io as hio

HERE = Path(__file__).resolve().parents[2]


def load_cfg(path=None):
    return yaml.safe_load(open(path or HERE / "configs/default.yaml"))


def cmd_ingest(a, cfg):
    from .export import versions
    from .ingest.manifest import build_initial, build_manifest
    man, T_wc = build_manifest(a.run, cfg)
    hio.write_artifact(a.run, "scene_manifest", man)
    tr = build_initial(a.run, cfg, man, T_wc)
    versions.register_initial(a.run, tr, cfg["inputs"]["objects_variant"], tr.meta["inputs"]["hands"])
    print(f"{a.run}: {man['frame_count']} frames, entities {[e['id'] for e in man['entities']]}, "
          f"hands {tr.meta['inputs']['hands']}")
    for n, o in tr.objects.items():
        print(f"  {n}: valid {o['valid'].sum()}/{tr.F}, sources {np.bincount(o['source'], minlength=7).tolist()}")
    for n, h in tr.hands.items():
        print(f"  {n}: valid {h['valid'].sum()}/{tr.F}, sources {np.bincount(h['source'], minlength=7).tolist()}")
    z = [o["T_world_obj"][o["valid"], 2, 3].min() for o in tr.objects.values() if o["valid"].any()]
    print(f"  lowest object origin z in world: {min(z):.3f} m (table top = 0)")


def load_version(run, version):
    from .export import versions
    reg = versions.load(run)
    v = reg[version]
    return hio.Trajectories.load(Path(v["npz"])), v


def cmd_diagnose(a, cfg):
    from .diagnostics import run as drun
    from .export import versions
    ver = a.version or versions.current(a.run)
    tr, v = load_version(a.run, ver)
    d = drun.run(a.run, ver, tr, cfg, v["objects_variant"], v["hands_variant"], a.stride)
    m = drun.summary_metrics(d, tr)
    hio.write_json(hio.out_dir(a.run) / f"metrics_diag_{ver}.json", m)
    print(json.dumps(m, indent=1))


def cmd_task(a, cfg):
    from .llm import ChatClient
    from .semantics import task as st
    from .visualization.overlay import Overlay
    man = json.loads((hio.out_dir(a.run) / "scene_manifest.json").read_text())
    tr, _ = load_version(a.run, "initial")
    ov = Overlay(a.run, man, tr)
    client = ChatClient.local(cfg["vlm"])
    client.wait_ready()
    speed = None
    if tr.objects:
        sp = [np.nan_to_num(np.linalg.norm(np.gradient(o["T_world_obj"][:, :3, 3], tr.t_s, axis=0), axis=1))
              for o in tr.objects.values()]
        speed = np.max(sp, axis=0)
    task = st.run(a.run, client, cfg["vlm"], man, ov.frame, ov.labelled, speed)
    hio.write_artifact(a.run, "task", task)
    print(json.dumps({k: task[k] for k in ("description", "phases")}, indent=1)[:4000])


def cmd_review(a, cfg):
    from .agents import review
    from .agents.tools import Toolbox
    from .llm import ChatClient
    client = ChatClient.local(cfg["vlm"]) if cfg["reviewer"]["provider"] == "local" else \
        ChatClient("https://api.openai.com/v1", cfg["reviewer"]["model"], "OPENAI_API_KEY")
    if cfg["reviewer"]["provider"] == "local":
        client.wait_ready()
    tb = Toolbox(a.run, cfg)
    s = review.run(a.run, client, cfg["reviewer"], tb)
    hio.write_json(hio.out_dir(a.run) / "review_summary.json", s)
    print(json.dumps({k: s[k] for k in ("final_version", "usage", "wall_s")}, indent=1))


def cmd_optimize(a, cfg):
    from .export import versions
    from .physics import optimize as po
    ver = a.version or versions.current(a.run)
    props = json.loads(Path(a.proposals).read_text()) if a.proposals else []
    name = po.run(a.run, ver, cfg, props, a.name)
    print("new version", name)


def cmd_track(a, cfg):
    from .tracking import cotracker
    man = json.loads((hio.out_dir(a.run) / "scene_manifest.json").read_text())
    tr, _ = load_version(a.run, "initial")
    print(json.dumps(cotracker.track(a.run, man, tr.extra["T_world_cam"], cfg["cotracker"]), indent=1))
    ag = cotracker.agreement(tr, cotracker.load_tracks(a.run), run_name=a.run)
    for n, d in ag.items():
        r = d["rigid_resid_m"]
        print(f"  {n}: CoTracker rigid residual median {np.nanmedian(r) * 1000:.1f} mm, p95 {np.nanpercentile(r, 95) * 1000:.1f} mm")


def cmd_refine(a, cfg):
    """Stage 9 on the whole sequence from the current version, then stage-10 acceptance."""
    from .agents.tools import Toolbox, accept, accept_entities, compare
    from .export import versions
    from .physics import optimize as po
    tb = Toolbox(a.run, cfg)
    if a.version:  # refine from an explicit version (A/B runs), not the registry's current one
        tb.current = a.version
    # rest consistency first (deterministic, evidence-gated): wrong rest poses that snap at first contact
    tb.current, _rest = apply_rest_fix(a.run, tb.current, cfg)
    # hand spin repair (local, TrackEverything evidence), before the whole-sequence solve
    tb.current, _spin = apply_spin_fix(a.run, tb.current, cfg)
    cur = tb.current
    name = a.name or f"phys_global_{len(versions.load(a.run))}"
    po.run(a.run, cur, cfg, [], name)
    c = compare(tb, cur, name)
    acc = accept(c, cfg["acceptance"])
    acc["entities"] = accept_entities(c, cfg["acceptance"])
    keep = [e for e, d in acc["entities"].items() if d["accepted"]]
    # always per entity (calibration vs public mocap: a whole-sequence refine made most objects worse, so a
    # whole-version accept must not carry them in): accept the version only if EVERY entity passed its own gate
    acc["accepted"] = bool(keep) and len(keep) == len(acc["entities"])
    versions.set_accepted(a.run, name, acc["accepted"], {"compare": c, **acc})
    final = name if acc["accepted"] else cur
    if not acc["accepted"] and keep:
        # entity-wise repair: take only the entities whose own checks improved, re-validate the combination
        merged = tb.traj(cur).copy()
        new = tb.traj(name)
        for e in keep:
            if e in new.objects:
                merged.objects[e] = {k: v.copy() for k, v in new.objects[e].items()}
            elif e in new.hands:
                merged.hands[e] = {k: v.copy() for k, v in new.hands[e].items()}
        mname = name + "_sel"
        versions.write_variant(a.run, mname, merged, cur, {"stage": "optimize_entity_select", "from": name,
                                                           "entities": keep}, versions.load(a.run)[cur]["hands_variant"])
        c2 = compare(tb, cur, mname)
        acc2 = accept(c2, cfg["acceptance"])
        ent2 = accept_entities(c2, cfg["acceptance"])  # the combination must still pass every kept entity's gate
        acc2["entities"] = ent2
        acc2["accepted"] = all(ent2.get(e, {}).get("accepted", False) for e in keep) and not acc2["hands_failing_user_rules"]
        versions.set_accepted(a.run, mname, acc2["accepted"], {"compare": c2, **acc2})
        print(json.dumps({"version": mname, "entities": keep, **acc2, "compare": c2["summary"]}, indent=1))
        if acc2["accepted"]:
            final = mname
        else:
            versions.reject_cleanup(a.run, mname)
    if not acc["accepted"]:
        versions.reject_cleanup(a.run, name)
    reg_ = versions.load(a.run)  # the refine result is the current version even when nothing was accepted
    reg_["_current"] = final       # (else a stale _current from earlier runs would survive a refine from --version)
    versions.save(a.run, reg_)
    print("current version:", final)
    hio.append_jsonl(hio.out_dir(a.run) / "review_log.jsonl", {"event": "global_refine", "version": name,
                                                                 "parent": cur, **acc, "compare": c["summary"]})
    print(json.dumps({"version": name, **acc, "compare": c["summary"]}, indent=1))


def cmd_export(a, cfg):
    from .export import final
    final.run(a.run, cfg)


def cmd_te_prepare(a, cfg):
    """Stage 6.2 (TrackEverything backend), step 1 in agile: queries -> hoi4d/te_queries.npz."""
    from .tracking import trackeverything
    man = json.loads((hio.out_dir(a.run) / "scene_manifest.json").read_text())
    tr, _ = load_version(a.run, "initial")
    print(json.dumps(trackeverything.prepare(a.run, man, tr, cfg["trackeverything"]), indent=1))


def cmd_track_compare(a, cfg):
    """A/B of the two tracking backends on the same version: object rigid residual + hand-track displacement
    agreement with the detected joints (offset-free, short windows)."""
    from .tracking import cotracker
    tr, _ = load_version(a.run, a.version or "initial")
    res = {}
    for name, f in (("cotracker", "tracks.npz"), ("trackeverything", "tracks_te.npz")):
        p = hio.out_dir(a.run) / f
        if not p.exists():
            continue
        tk = dict(np.load(p, allow_pickle=False))
        ag = cotracker.agreement(tr, tk, run_name=a.run)
        res[name] = {n: {"median_mm": float(np.nanmedian(d["rigid_resid_m"]) * 1000),
                         "p95_mm": float(np.nanpercentile(d["rigid_resid_m"], 95) * 1000)} for n, d in ag.items()}
        if "joint" in tk:  # hand joints: |d joint(t..t+k) - d track(t..t+k)| over visible pairs, k = 1, 5, 10
            from .conventions import SOURCE_IDS
            hs = {}
            for h, hd in tr.hands.items():
                if h not in list(tk["names"]):
                    continue
                k = list(tk["names"]).index(h)
                sel = np.where((tk["owner"] == k) & (tk["joint"] >= 0))[0]
                obs = hd["source"] == SOURCE_IDS["observed"]
                for lag in (1, 5, 10):
                    e = []
                    for i in sel:
                        j = tk["joint"][i]
                        P, V = tk["P_world"][:, i], tk["vis"][:, i]
                        t = np.arange(len(P) - lag)
                        ok = V[t] & V[t + lag] & obs[t] & obs[t + lag]
                        if ok.any():
                            dj = hd["joints"][t[ok] + lag, j] - hd["joints"][t[ok], j]
                            dp = P[t[ok] + lag] - P[t[ok]]
                            e.append(np.linalg.norm(dj - dp, axis=1))
                    e = np.concatenate(e) if e else np.array([np.nan])
                    hs[f"{h}_lag{lag}"] = {"median_mm": float(np.nanmedian(e) * 1000), "p95_mm": float(np.nanpercentile(e, 95) * 1000), "n": int(np.isfinite(e).sum())}
            res[name]["hands"] = hs
    hio.write_json(hio.out_dir(a.run) / "track_compare.json", res)
    print(json.dumps(res, indent=1))


def apply_rest_fix(run, cur, cfg, name=None):
    """Rest consistency (an untouched object cannot move): fix wrong rest poses revealed by a jump at the hand's
    first contact / release. Evidence-gated per jump (physics/rest_consistency.py). Returns the version to continue
    from (a new accepted version if anything changed, else cur) and the report."""
    from .export import versions
    from .physics import rest_consistency as rc
    man = json.loads((hio.out_dir(run) / "scene_manifest.json").read_text())
    tr, _ = load_version(run, cur)
    new, rep = rc.repair(run, man, tr, cfg.get("rest_consistency"))
    hio.write_json(hio.out_dir(run) / f"rest_fix_{cur}.json", rep)
    if not any(r["applied"] for r in rep):
        return cur, rep
    name = name or f"rest_{len(versions.load(run))}"
    versions.write_variant(run, name, new, cur, {"stage": "rest_consistency", "jumps": rep},
                           versions.load(run)[cur]["hands_variant"])
    versions.set_accepted(run, name, True, {"rule": "rest frames explained >= 20% better by the held stable-side pose",
                                            "jumps": rep})
    return name, rep


def cmd_rest_fix(a, cfg):
    cur = a.version or json.loads((hio.out_dir(a.run) / "versions.json").read_text()).get("_current", "initial")
    v, rep = apply_rest_fix(a.run, cur, cfg, a.name)
    print(json.dumps({"from": cur, "version": v, "jumps": rep}, indent=1, default=float))


def apply_spin_fix(run, cur, cfg, name=None):
    """Hand spin repair as its own LOCAL step: fit flips (fast wrist turns the TrackEverything hand points do not
    show) are re-solved with optimizer knots only around each burst; only the repaired hands are taken over, every
    other frame stays as detected. Accepted per hand if the user's hand rules hold (detected frames outside the
    repair within 1 cm p95, no added fingertip jitter) and the repaired frames no longer spin (< 10 rad/s)."""
    from .agents.tools import hand_checks
    from .export import versions
    from .physics import optimize as po
    from .tracking import cotracker
    tr, _ = load_version(run, cur)
    tracks = cotracker.load_tracks(run)
    fps = 1.0 / float(np.median(np.diff(tr.t_s)))
    props, masks = [], {}
    for h, hd in tr.hands.items():
        m, _ = po.spin_repair(hd, po.hand_track_points(tracks, h), fps)
        masks[h] = m
        f = 0
        while f < tr.F:
            if m[f]:
                e = f
                while e + 1 < tr.F and m[e + 1]:
                    e += 1
                props.append({"frames": [f, e], "constraints": [], "evidence": f"{h} spin repair"})
                f = e + 1
            else:
                f += 1
    if not props:
        return cur, {"repaired": {}}
    base = name or f"spin_{len(versions.load(run))}"
    trial = po.run(run, cur, cfg, props, base + "_trial", accept_registry=False)
    new = hio.Trajectories.load(hio.out_dir(run) / f"trajectories_{trial}.npz")
    merged, rep = tr.copy(), {"repaired": {}, "windows": [p["frames"] for p in props]}
    for h, m in masks.items():
        if not m.any():
            continue
        chk = hand_checks(tr.hands[h], new.hands[h], tr.t_s)
        R = po._hand_frame(np.nan_to_num(new.hands[h]["joints"]))
        from scipy.spatial.transform import Rotation as Rot
        idx = [f for f in np.where(m)[0] if f + 1 < tr.F and new.hands[h]["valid"][f] and new.hands[h]["valid"][f + 1]]
        sp = [np.linalg.norm(Rot.from_matrix(R[f].T @ R[f + 1]).as_rotvec()) * fps for f in idx]
        ok = chk["observed_joint_dev_p95_m"] <= cfg["acceptance"].get("hand_obs_dev_p95_m", 0.01) \
            and chk["finger_rel_acc_p95"][1] <= chk["finger_rel_acc_p95"][0] * (1 + cfg["acceptance"].get("hand_jitter_tol", 0.05)) \
            and (max(sp) if sp else 0) < 10.0
        rep["repaired"][h] = {"frames": int(m.sum()), "accepted": bool(ok), **chk,
                              "max_wrist_speed_after_rad_s": float(max(sp)) if sp else None}
        if ok:
            merged.hands[h] = {k: v.copy() for k, v in new.hands[h].items()}
    versions.reject_cleanup(run, trial)
    if not any(d["accepted"] for d in rep["repaired"].values()):
        return cur, rep
    versions.write_variant(run, base, merged, cur, {"stage": "hand_spin_repair", **rep}, versions.load(run)[cur]["hands_variant"])
    versions.set_accepted(run, base, True, {"rule": "hand rules + no spin in repaired frames", **rep})
    return base, rep


def cmd_spin_fix(a, cfg):
    cur = a.version or json.loads((hio.out_dir(a.run) / "versions.json").read_text()).get("_current", "initial")
    v, rep = apply_spin_fix(a.run, cur, cfg, a.name)
    print(json.dumps({"from": cur, "version": v, **rep}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "track", "te_prepare", "track_compare", "rest_fix", "spin_fix", "diagnose", "task", "review", "optimize", "refine", "export"])
    ap.add_argument("run")
    ap.add_argument("--version")
    ap.add_argument("--name")
    ap.add_argument("--proposals")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    from .tracking import cotracker
    cotracker.TRACKS_FILE = {"cotracker": "tracks.npz", "trackeverything": "tracks_te.npz"}[
        os.environ.get("HOI4D_TRACKER", cfg.get("tracking", {}).get("backend", "cotracker"))]
    globals()[f"cmd_{a.cmd}"](a, cfg)


if __name__ == "__main__":
    main()
