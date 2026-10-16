"""README stage 11: export. trajectories_refined.npz (= accepted current version, with its revision chain),
contacts.json, anomalies.json (final statuses from the review findings; unresolved ones stay visible),
metrics.json, run_manifest.json (models, checkpoints, config hash, hardware, seeds, timings, provenance),
report.md. Initial + every version stay in hoi4d/ (versions.json)."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .. import io as hio
from . import versions


def chain(reg, v):
    out = []
    while v:
        out.append(v)
        v = reg[v].get("parent")
    return out[::-1]


def run(run_name, cfg):
    d = hio.out_dir(run_name)
    reg = versions.load(run_name)
    cur = versions.current(run_name)
    tr = hio.Trajectories.load(Path(reg[cur]["npz"]))
    tr.meta["revision_chain"] = chain(reg, cur)
    tr.meta["refined_from"] = "initial"
    tr.save(d / "trajectories_refined.npz")
    for kind in ("contacts", "anomalies"):
        src = d / f"{kind}_{cur}.json"
        if src.exists():
            shutil.copy(src, d / f"{kind}.json")
    # final anomaly statuses from the review findings
    rs = d / "review_summary.json"
    findings = json.loads(rs.read_text())["findings"] if rs.exists() else []
    an = json.loads((d / "anomalies.json").read_text()) if (d / "anomalies.json").exists() else {"anomalies": []}
    init = json.loads((d / "anomalies_initial.json").read_text())["anomalies"]
    by_id = {f.get("anomaly_id"): f for f in findings}
    final_ids = {(a["type"], tuple(a["entities"]), a["frames"][0]) for a in an["anomalies"]}
    resolved = []
    for a in init:
        f = by_id.get(a["id"])
        key = (a["type"], tuple(a["entities"]), a["frames"][0])
        st = f["status"] if f else "candidate"
        if key not in final_ids and cur != "initial":
            st = "repaired"
        elif st == "candidate":
            st = "unresolved"
        resolved.append({**a, "status": st, "review": f})
    hio.write_artifact(run_name, "anomalies", {"version": cur, "initial_anomalies": resolved,
                                                "anomalies": an["anomalies"]}, "anomalies.json")
    m = {"version": cur, "reconstruction": {}, "tracking": {}, "contact": {}, "physics": {}, "runtime": {}}
    for v in ("initial", cur):
        p = d / f"metrics_diag_{v}.json"
        if p.exists():
            md = json.loads(p.read_text())
            m["contact"][v] = md.get("penetration")
            m["tracking"][v] = {"median_acc_mps2": md.get("median_acc_mps2"), "anomalies": md.get("anomalies_by_type")}
    if cur in reg and reg[cur].get("decision"):
        m["reconstruction"]["last_acceptance"] = {k: v for k, v in reg[cur]["decision"].items() if k != "compare"}
        m["reconstruction"]["visual"] = reg[cur]["decision"].get("compare", {}).get("visual")
    for f in d.glob("stability_*.json"):
        m["physics"][f.stem] = json.loads(f.read_text())
    if rs.exists():
        s = json.loads(rs.read_text())
        m["runtime"]["review"] = {"usage": s.get("usage"), "wall_s": s.get("wall_s"), "rounds": len(s.get("rounds", []))}
    tj = d / "task.json"
    if tj.exists():
        m["runtime"]["vlm"] = json.loads(tj.read_text()).get("model")
    hio.write_artifact(run_name, "metrics", m, "metrics.json")
    cfg_bytes = json.dumps(cfg, sort_keys=True).encode()
    rm = {"run": run_name, "created": hio.now(), "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
          "config": cfg, "current_version": cur, "versions": {k: {kk: vv for kk, vv in v.items() if kk != "decision"}
                                                              for k, v in reg.items() if not k.startswith("_")},
          "models": {"segmentation": "SAM 3 (weights/sam3)", "stereo": "FoundationStereo (sif/foundation_stereo_official)",
                     "object_pose": "FoundationPose + stereo ICP + SE(3) fusion (existing pipeline)",
                     "hands": "WiLoR/HOPformer prior + stereo MANO fit + s06h grasp physics (existing pipeline)",
                     "tracking": f"CoTracker3 {cfg['cotracker']['checkpoint']} (CC-BY-NC-4.0)",
                     "vlm_and_reviewer": f"{cfg['vlm']['model_dir']} served by vLLM (sif/vllm.sif)"},
          "hardware": {"node": platform.node(), "gpu": _gpu()}, "seeds": {"surface_sampling": 0, "cotracker_seeds": 0}}
    hio.write_json(d / "run_manifest.json", rm)
    report(run_name, d, cur, resolved, m)
    print(f"exported {run_name}: current version {cur}, chain {chain(reg, cur)}")


def _gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def report(run_name, d, cur, resolved, m):
    L = [f"# hoi4d report: {run_name}", "", f"Final version: `{cur}`", ""]
    tj = d / "task.json"
    if tj.exists():
        t = json.loads(tj.read_text())
        L += ["## Task (VLM, soft hypothesis)", "", t["description"], ""]
        L += [f"- {p['name']} {p['t_start_s']:.2f}-{p['t_end_s']:.2f}s {p['entities']} (conf {p['confidence']:.2f})"
              for p in t["phases"]]
        L.append("")
    L += ["## Anomalies (initial version) and outcome", "", "| id | type | entities | time (s) | status |", "|---|---|---|---|---|"]
    for a in resolved:
        L.append(f"| {a['id']} | {a['type']} | {', '.join(a['entities'])} | {a['t_start_s']:.2f}-{a['t_end_s']:.2f} | {a['status']} |")
    L += ["", "## Metrics", "", "```json", json.dumps(m, indent=1, default=str)[:6000], "```"]
    (d / "report.md").write_text("\n".join(L))
