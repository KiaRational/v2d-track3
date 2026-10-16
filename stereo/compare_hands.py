"""Compare hand reconstructions (e.g. WiLoR-prior stereo fit vs HOPformer-prior stereo fit, before/after the
physics grasp stage) on every run, with no ground truth needed (host env, json only).

Per run and hand variant (<run>/stereo/<hands>):
  fit        mano_fit-style report.json: MPJPE to the stereo triangulation (mm), jitter (mm/frame^2),
             reprojection to the colour WiLoR 2D keypoints (px) -- right/left averaged
  contact    contact_audit_<hands>.json, per object over grasp frames: pen_mm (deepest finger inside),
             pen_gt10 (fraction of grasp frames > 10 mm inside), gap_mm, tips_near
  wrench     wrench_<hands>.json: fraction of lifted frames whose contacts can hold the object (mu 1.0, 0.1)
  stable     stability_<hands>.json: MuJoCo hold test, fraction of lifted frames the object stays < 2 cm
Writes <out>/compare.json and <out>/compare.md (per-episode rows + split means).
Usage: python compare_hands.py --lists public_list.txt eval_list.txt --hands mano_fit mano_fit_phys mano_fit_hop
       mano_fit_hop_phys --out DIR
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fit_metrics(run, hands):
    r = load(run / "stereo" / hands / "report.json")
    if not r:
        return {}
    out = {}
    for key, name in (("mpjpe_tri_mm_median", "mpjpe_mm"), ("jitter_all_frames_mm_per_frame2", "jitter"),
                      ("reproj_wilor2d_px_median", "reproj_px")):
        v = []
        for side in r.get("sides", {}).values():
            m = side.get("mano_fit") or {}
            if m.get(key) is not None:
                v.append(m[key])
        if v:
            out[name] = float(np.mean(v))
    return out


def episode_metrics(run, hands):
    m = {"fit": fit_metrics(run, hands), "objects": {}}
    ca = load(run / "stereo" / f"contact_audit_{hands}.json") or {}
    wr = load(run / "stereo" / f"wrench_{hands}.json") or {}
    st = load(run / "stereo" / f"stability_{hands}.json") or {}
    names = set(ca.get("objects", {})) | set(wr.get("objects", {})) | set(st.get("objects", {}))
    for o in sorted(names):
        e = {}
        c = ca.get("objects", {}).get(o)
        if c:
            e.update(pen_mm=c.get("pen_mm"), pen_gt10=c.get("pen_gt10"), gap_mm=c.get("gap_mm"),
                     tips_near=c.get("tips_near"), grasp_frames=c.get("grasp_frames"))
        w = wr.get("objects", {}).get(o)
        if w and w.get("feasible_frac"):
            e.update(wrench_mu1=w["feasible_frac"].get("1.0"), wrench_mu01=w["feasible_frac"].get("0.1"))
        s = st.get("objects", {}).get(o)
        if s:
            e.update(stable=s.get("stable_frac_2cm"))
        m["objects"][o] = e
    return m


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def f(x, d=1, pct=False):
    if x is None:
        return "–"
    return f"{x * 100:.0f}%" if pct else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--hands", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for lst in a.lists:
        for name in Path(lst).read_text().split():
            run = ROOT / "runs" / name
            res[name] = {h: episode_metrics(run, h) for h in a.hands}
    (out / "compare.json").write_text(json.dumps(res, indent=1))

    cols = [("pen_mm", "pen mm", 1, False), ("pen_gt10", ">10mm", 0, True), ("gap_mm", "gap mm", 1, False),
            ("tips_near", "tips", 1, False), ("wrench_mu1", "hold μ1", 0, True), ("wrench_mu01", "hold μ.1", 0, True),
            ("stable", "sim hold", 0, True)]
    lines = ["# Hand reconstruction comparison", "",
             "Per object over grasp frames. pen = deepest finger inside the object (lower is better); >10mm = share of "
             "grasp frames deeper than 10 mm; gap = closest hand point to the surface when not touching; tips = "
             "fingertips within 1 cm; hold μ = share of lifted frames whose contacts can carry the object at friction "
             "μ; sim hold = MuJoCo hold test. Fit columns: MPJPE to the stereo triangulation, jitter (mm/frame²).", ""]
    for split in ("public", "evaluation"):
        runs = [r for r in res if r.startswith(split)]
        if not runs:
            continue
        lines += [f"## {split} — mean over objects", "",
                  "| hands | " + " | ".join(c[1] for c in cols) + " | MPJPE | jitter |",
                  "|---" * (len(cols) + 3) + "|"]
        for h in a.hands:
            objs = [e for r in runs for e in res[r][h]["objects"].values()]
            fits = [res[r][h]["fit"] for r in runs]
            row = [f(mean([e.get(k) for e in objs]), d, p) for k, _, d, p in cols]
            row += [f(mean([x.get("mpjpe_mm") for x in fits])), f(mean([x.get("jitter") for x in fits]))]
            lines.append(f"| {h} | " + " | ".join(row) + " |")
        lines += ["", f"## {split} — per episode and object", "",
                  "| episode | object | hands | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 3) + "|"]
        for r in runs:
            objs = sorted({o for h in a.hands for o in res[r][h]["objects"]})
            for o in objs:
                for h in a.hands:
                    e = res[r][h]["objects"].get(o, {})
                    lines.append(f"| {r.split('_ep')[-1]} | {o} | {h} | " +
                                 " | ".join(f(e.get(k), d, p) for k, _, d, p in cols) + " |")
        lines.append("")
    (out / "compare.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
