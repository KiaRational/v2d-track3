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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "track", "te_prepare", "track_compare", "diagnose", "task", "review", "optimize", "refine", "export"])
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
