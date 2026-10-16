"""Stage 6i: pick, per grasp, the hand candidate that physics says holds the object (host env).

The simulate -> diagnose -> correct loop. s06h produces candidate hand sequences that differ in
how firmly the contacts press (--squeeze_mm); physics_check.py --stability simulates each grasp
frame (hands frozen, object released at its reconstructed pose, gravity) and records how far the
object falls. Here, for every grasp segment (one hand holding one object, fusion_report
hand_contact), the candidate with the smallest mean fall is chosen -- preferring the gentlest
squeeze when several hold (fall < --ok_cm). Outside grasps the gentlest candidate is used.
Choices are per segment, not per frame (per-frame choices would jitter), with a short crossfade
at segment boundaries.
Output stereo/<out>/<side>/NNNNNN.json + report.json (per segment: candidates' falls, choice).
Each segment records the chosen candidate's squeeze_mm/contact_mm (chosen_params); the final pass
(s06h --ref_hands mano_fit --hands <out> --segment_params <out>) reuses them.
Usage: python s06i_select_grasps.py <run> --cands mano_fit_s15 mano_fit_s30 mano_fit_s50 [--out mano_fit_selected]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(os.environ["V2D_ROOT"])


def segments(frames, max_gap=1):
    """Contiguous runs (fusion_report hand_contact is contiguous per grasp, see fuse_objects)."""
    out, cur = [], []
    for t in sorted(frames):
        if cur and t - cur[-1] > max_gap:
            out.append(cur)
            cur = []
        cur.append(t)
    return out + ([cur] if cur else [])


def blend(r0, r1, al):
    """Crossfade two hand records (same frame): SLERP the wrist rotation, lerp the rest."""
    g = Slerp([0, 1], Rotation.from_rotvec([r0["mano"]["global_orient"], r1["mano"]["global_orient"]]))(al)
    out = dict(r1)
    out["mano"] = {"betas": r1["mano"]["betas"], "global_orient": g.as_rotvec().tolist(),
                   "hand_pose": ((1 - al) * np.array(r0["mano"]["hand_pose"]) + al * np.array(r1["mano"]["hand_pose"])).tolist()}
    out["T"] = ((1 - al) * np.array(r0["T"]) + al * np.array(r1["T"])).tolist()
    out["joints"] = ((1 - al) * np.array(r0["joints"]) + al * np.array(r1["joints"])).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cands", nargs="+", required=True, help="candidate hand dirs, gentlest squeeze first")
    ap.add_argument("--out", default="mano_fit_selected",
                    help="Intermediate candidates; run s06h afterward to produce mano_fit_phys")
    ap.add_argument("--ok_cm", type=float, default=2.0)
    ap.add_argument("--fade", type=int, default=0,
                    help="Legacy blend; use final MANO refinement to smooth candidate boundaries")
    ap.add_argument("--tol", type=float, default=0.1, help="wrench residual below which a frame is feasible")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    rep = json.loads((run / "fusion_report.json").read_text())
    stab, wr = {}, {}
    for c in a.cands:
        f = run / "stereo" / f"stability_{c}.json"
        stab[c] = json.loads(f.read_text())["objects"] if f.exists() else {}
        f = run / "stereo" / f"wrench_{c}.json"
        wr[c] = json.loads(f.read_text())["objects"] if f.exists() else {}
    # anchor parameters of each candidate, so the final s06h pass can rebuild the chosen anchors
    params = {}
    for c in a.cands:
        f = run / "stereo" / c / "report.json"
        args = json.loads(f.read_text()).get("args", {}) if f.exists() else {}
        keys = ("squeeze_mm", "contact_mm")
        params[c] = {k: float(args[k]) for k in keys} if all(k in args for k in keys) else None
    report = {"cands": a.cands, "default_params": params[a.cands[0]], "segments": []}
    for side in ("left", "right"):
        recs = {c: {int(p.stem): json.loads(p.read_text()) for p in (run / "stereo" / c / side).glob("*.json")}
                for c in a.cands}
        base = recs[a.cands[0]]
        if not base:
            continue
        choice = {t: a.cands[0] for t in base}
        for nm, d in rep.items():
            if not isinstance(d, dict):
                continue
            for seg in segments(d.get("hand_contact", {}).get(side, [])):
                ss = set(seg)
                falls, feas = {}, {}
                for c in a.cands:
                    v = [x["disp_cm"] for x in stab[c].get(nm, {}).get("per_frame", []) if x["t"] in ss]
                    if v:
                        falls[c] = float(np.mean(v))
                    rows = [x for x in wr[c].get(nm, {}).get("per_frame", []) if x["t"] in ss]
                    if rows:  # fraction of frames feasible at the reward's and the simulator's friction
                        feas[c] = tuple(float(np.mean([x["residual"][mu] < a.tol for x in rows])) for mu in ("0.1", "1.0"))
                if not falls and not feas:
                    continue  # object on the table throughout: nothing to hold up
                # wrench feasibility first (the training reward's standard), then the MuJoCo hold test,
                # then the gentlest candidate (candidate order)
                key = lambda c: (-feas.get(c, (0, 0))[0], -feas.get(c, (0, 0))[1],  # noqa: E731
                                 round(falls.get(c, 99), 0), a.cands.index(c))
                pick = min([c for c in a.cands if c in feas or c in falls], key=key)
                ok = [c for c in a.cands if feas.get(c, (0, 0))[1] >= 0.9
                      and c in falls and falls[c] < a.ok_cm]
                for t in seg:
                    if t in choice:
                        choice[t] = pick
                report["segments"].append({"side": side, "object": nm, "frames": [seg[0], seg[-1]],
                                           "fall_cm": {k: round(v, 2) for k, v in falls.items()},
                                           "feasible_mu01_mu1": {k: [round(x, 2) for x in v] for k, v in feas.items()},
                                           "chosen": pick, "chosen_params": params[pick], "held": pick in ok})
        outd = run / "stereo" / a.out / side
        outd.mkdir(parents=True, exist_ok=True)
        ts = sorted(base)
        for i, t in enumerate(ts):
            c = choice[t]
            r = recs[c].get(t, base[t])
            # crossfade over the frames after a change of candidate
            for k in range(1, a.fade + 1):
                if i - k >= 0 and choice[ts[i - k]] != c:
                    prev = recs[choice[ts[i - k]]].get(t, base[t])
                    r = blend(prev, r, k / (a.fade + 1))
                    break
            (outd / f"{t:06d}.json").write_text(json.dumps(r))
    held = [s["held"] for s in report["segments"]]
    report["held_frac"] = float(np.mean(held)) if held else None
    (run / "stereo" / a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{a.run}: {len(held)} lifted grasps, wrench-feasible (mu=1, >=90% of frames) with the chosen candidate: "
          f"{sum(held)}", flush=True)
    for s in report["segments"]:
        print(f"  {s['side']:5s} {s['object']:16s} {s['frames']}  feasible {s['feasible_mu01_mu1']}  "
              f"falls {s['fall_cm']}  -> {s['chosen']}", flush=True)


if __name__ == "__main__":
    main()
