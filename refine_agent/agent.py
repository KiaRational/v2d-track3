"""Refinement agent: verify -> diagnose -> correct -> re-verify -> accept or revert (agile image).

For every object of an episode:
  1. verify the current trajectory (verify.py, observation + physics checks, no mocap) and turn the
     calibrated per-frame error probability (critic_calibration.json) into problem windows, worst
     first; frames where the object moves with no hand on it are problem windows too
  2. for the worst unresolved window, try the correctors (each = the object fusion re-run with
     cumulative per-window overrides, written to a trial variant):
       reicp   stereo-ICP re-registered from the CURRENT trajectory in the window, trusted x3
       no_fp   FoundationPose dropped in the window (stereo, smoothness and hands carry it)
       fp_low  FoundationPose trusted x0.25 in the window
       hold    the object held still in the window
  3. re-verify each trial over the whole trajectory; keep the best only if the window's error
     improves by >= --min_gain and the whole trajectory does not get worse; otherwise revert
  4. log every decision with its numbers (refine/agent_log.json); after an accepted trial the problem
     windows are recomputed from the re-verified trajectory (ranges already tried are skipped);
     continue until no untried window is left or --max_windows are done
The objective counts frames the critic cannot score (or a trial lost) as error probability 1. On a
public episode the critic is the leave-one-episode-out calibration (critic_calibration_loeo.json).
Output: trajectory variant fused_agent (camera frame, the usual layout), refine/agent_overrides.json.
Usage: python agent.py <run> [...] [--base fused_all_sam3] [--max_windows 6]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEREO = HERE.parent / "stereo"
sys.path.insert(0, str(HERE))
import verify  # noqa: E402

ROOT = Path(os.environ["V2D_ROOT"])
# compliant tree (V2D_CRITIC=uncalibrated): the mocap-fitted calibration files are never read
_UNCAL = os.environ.get("V2D_CRITIC") == "uncalibrated"
CAL = None if _UNCAL else json.loads((HERE / "critic_calibration.json").read_text())
# leave-one-episode-out fits (validate_critic.py --fit): on a public episode the agent must not use a
# critic that was fitted on that episode's mocap
_LOEO = HERE / "critic_calibration_loeo.json"
CAL_LOEO = json.loads(_LOEO.read_text()) if _LOEO.exists() and not _UNCAL else {}
ENV = {**os.environ, "V2D_OBJ_DIR": os.environ.get("V2D_OBJ_DIR", "objects_sam3")}
MISSING_P = 1.0  # error probability of a frame the critic cannot score (no depth / mask evidence, or no pose)


# V2D_CRITIC=uncalibrated (compliant tree): the critic must not carry weights fitted on public mocap. Same three
# observation features, equal fixed weights; acceptance only compares before/after, so the scale is immaterial.
UNCALIBRATED = {"features": ["free_space", "depth_agree", "mask_iou"], "transform": ["x", "1-x", "1-x"],
                "coef": [2.0, 2.0, 2.0], "intercept": -2.0, "source": "fixed equal weights (no mocap fit)"}
if _UNCAL:
    CAL = UNCALIBRATED


def calibration(run_name):
    """The critic calibration for one episode: its LOEO fit when there is one, the full fit otherwise."""
    if os.environ.get("V2D_CRITIC") == "uncalibrated":
        return UNCALIBRATED
    e = CAL_LOEO.get("episodes", {}).get(run_name)
    if e is None:
        return CAL
    return {**CAL, "features": CAL_LOEO.get("features", CAL["features"]),
            "transform": CAL_LOEO.get("transform", CAL["transform"]), "coef": e["coef"], "intercept": e["intercept"]}


def err_prob(ser, t, cal=None):
    cal = cal or CAL
    f = []
    for name, tr in zip(cal["features"], cal["transform"]):
        v = ser.get(name, {}).get(str(t))
        if v is None:
            return None
        f.append({"x": v, "1-x": 1 - v, "log1p": np.log1p(v)}[tr])
    z = float(np.dot(cal["coef"], f) + cal["intercept"])
    return 1 / (1 + np.exp(-z))


def per_frame(rep, obj, cal=None):
    ser = rep["object"][obj]
    frames = sorted({int(t) for m in ser.values() for t in m})
    p = {t: err_prob(ser, t, cal) for t in frames}
    alone = {t: ser.get("moved_alone", {}).get(str(t), 0) or 0 for t in frames}
    return p, alone


def _frames(p, lo, hi, frames):
    ts = set(p) if frames is None else set(frames) | set(p)
    return sorted(t for t in ts if lo is None or lo <= t <= hi)


def n_missing(p, lo=None, hi=None, frames=None):
    """Frames in [lo, hi] (of p, plus the reference frame set) the critic could not score."""
    return sum(p.get(t) is None for t in _frames(p, lo, hi, frames))


def objective(p, alone, lo=None, hi=None, frames=None):
    """Mean observation-based error probability (+ a penalty for motion no hand explains) in [lo, hi].
    A frame the critic cannot score -- missing features (object pushed off-image / behind the camera)
    or, with frames= (the base trajectory's frames), no pose at all -- counts as p = MISSING_P, so a
    trial cannot gain by moving the object out of the observations."""
    ts = _frames(p, lo, hi, frames)
    if not ts:
        return None
    vals = [MISSING_P if p.get(t) is None else p[t] for t in ts]
    return float(np.mean(vals) + 0.5 * np.mean([alone.get(t, 0) for t in ts]))


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def overlaps(w, tried):
    return any(not (w[1] < a or w[0] > b) for a, b in tried)


def problem_windows(p, alone, thr, gap=3, min_len=2):
    bad = sorted(t for t in p if (p[t] is not None and p[t] > thr) or alone[t] > 0.5)
    out, cur = [], []
    for t in bad:
        if cur and t - cur[-1] > gap:
            out.append(cur)
            cur = []
        cur.append(t)
    out += [cur] if cur else []
    wins = [(w[0], w[-1]) for w in out if len(w) >= min_len]
    return sorted(wins, key=lambda w: -sum((p[t] or 0) + alone[t] for t in range(w[0], w[1] + 1) if t in p))


def carried(run, obj, lo, hi, speed=0.004):
    """Is a hand holding AND moving the object anywhere in [lo, hi]? (fusion_report hand contact +
    world-frame palm speed from the stereo hands)"""
    rep = json.loads((run / "fusion_report.json").read_text()).get(obj, {})
    tri = json.loads((run / "stereo/hands_tri.json").read_text())
    from verify import load_T
    slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
    for side, frames in rep.get("hand_contact", {}).items():
        X = tri.get(side, {}).get("X_smooth", {})
        for t in frames:
            if lo <= t <= hi and str(t) in X and str(t + 1) in X and t in slam and t + 1 in slam:
                a_ = slam[t][:3, :3] @ np.mean(np.array(X[str(t)])[[0, 5, 9, 13, 17]], 0) + slam[t][:3, 3]
                b_ = slam[t + 1][:3, :3] @ np.mean(np.array(X[str(t + 1)])[[0, 5, 9, 13, 17]], 0) + slam[t + 1][:3, 3]
                if np.linalg.norm(b_ - a_) > speed:
                    return True
    return False


def run_py(script, args):
    r = subprocess.run([sys.executable, str(script)] + [str(x) for x in args], env=ENV, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{script.name} failed: {r.stderr[-800:]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base", default="fused_all_sam3")
    ap.add_argument("--out", default="fused_agent")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--min_gain", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--max_windows", type=int, default=6)
    ap.add_argument("--pad", type=int, default=3)
    a = ap.parse_args()
    for r in a.runs:
        run = ROOT / "runs" / r
        (run / "refine").mkdir(exist_ok=True)
        log, ovr = [], {}
        objs = [o["name"] for o in json.loads((run / "objects_sam3/objects.json").read_text())]
        # the agent's working trajectory starts as the base trajectory
        for nm in objs:
            dst = run / a.out / nm
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(run / a.base / nm, dst)
        cal = calibration(r)
        print(f"{r}: critic calibration {'leave-one-episode-out' if cal is not CAL else 'full fit'}", flush=True)
        for nm in objs:
            t0 = time.time()
            rep = verify.main([r, "--objects", a.out, "--only_objects", nm, "--no_hands", "--quiet"])[r]
            p, alone = per_frame(rep, nm, cal)
            base_frames = sorted(p)  # the reference frame set: a trial that loses a frame is penalised there
            J0 = objective(p, alone, frames=base_frames)
            wins = problem_windows(p, alone, a.thr)
            print(f"{r} {nm}: objective {fmt(J0)} ({n_missing(p)} unscorable frames), "
                  f"{len(wins)} problem windows {wins[:8]}", flush=True)
            done, tried = 0, []
            while done < a.max_windows:
                # windows from the CURRENT trajectory (re-verified after every accepted trial), skipping
                # every range already tried
                wins = [w for w in problem_windows(p, alone, a.thr) if not overlaps(w, tried)]
                if not wins:
                    break
                w0, w1 = wins[0]
                done += 1
                lo, hi = max(0, w0 - a.pad), w1 + a.pad
                tried.append((lo, hi))
                Jw = objective(p, alone, lo, hi, frames=base_frames)
                Jall = objective(p, alone, frames=base_frames)
                icp_dir = f"stereo_icp_agent_{lo}_{hi}"
                # a dir kept from an earlier agent run could hold frames this registration drops
                shutil.rmtree(run / ENV["V2D_OBJ_DIR"] / nm / icp_dir, ignore_errors=True)
                run_py(STEREO / "object_icp.py", [r, "--init_variant", a.out, "--out_name", icp_dir,
                                                  "--frames", lo, hi, "--only_objects", nm])
                base_ov = json.loads(json.dumps(ovr.get(nm, {})))
                extra = base_ov.get("icp_extra", []) + [icp_dir]

                def with_(**kw):
                    o = json.loads(json.dumps(base_ov))
                    for k, v in kw.items():
                        o[k] = o.get(k, []) + v if isinstance(v, list) and k != "icp_extra" else v
                    return o
                cands = {"reicp": with_(icp_extra=extra, icp_mult=[[lo, hi, 3.0]]),
                         "no_fp": with_(icp_extra=extra, fp_mult=[[lo, hi, 0.0]]),
                         "fp_low": with_(icp_extra=extra, fp_mult=[[lo, hi, 0.25]])}
                # holding still is only physically allowed when no hand is moving the object in the window
                if not carried(run, nm, lo, hi):
                    cands["hold"] = with_(static=[[lo, hi, 1.0]])
                trials = {}
                for cname, cov in cands.items():
                    tv = f"agent_trial_{cname}"
                    ofile = run / "refine" / f"ovr_{cname}.json"
                    ofile.write_text(json.dumps({**ovr, nm: cov}))
                    # trial variant names are reused across windows and objects: never score stale poses
                    shutil.rmtree(run / tv, ignore_errors=True)
                    try:
                        run_py(STEREO / "fuse_objects.py", [r, "--overrides", ofile, "--out_variant", tv, "--only_objects", nm])
                    except RuntimeError as e:
                        print(f"   {cname}: failed ({str(e)[:120]})", flush=True)
                        continue
                    pd_ = run / tv / nm / "poses"
                    if not pd_.is_dir() or not any(pd_.glob("*.json")):  # fuse_objects skipped the object
                        print(f"   {cname}: failed (no poses written)", flush=True)
                        continue
                    rp = verify.main([r, "--objects", tv, "--only_objects", nm, "--no_hands", "--quiet"])[r]
                    pp, aa = per_frame(rp, nm, cal)
                    trials[cname] = (objective(pp, aa, lo, hi, frames=base_frames), objective(pp, aa, frames=base_frames),
                                     cov, tv, pp, aa, n_missing(pp, lo, hi, base_frames), n_missing(pp, frames=base_frames))
                if not trials:
                    log.append({"object": nm, "window": [w0, w1], "padded": [lo, hi], "trials": {}, "chosen": None,
                                "accepted": False, "note": "every trial failed"})
                    continue
                best = min(trials, key=lambda k: trials[k][1] if trials[k][1] is not None else 9)
                jw, ja, cov, tv, pp, aa, _, _ = trials[best]
                ok = (jw is not None and Jw is not None and ja is not None and Jall is not None
                      and jw <= (1 - a.min_gain) * Jw and ja <= Jall + a.tol)
                entry = {"object": nm, "window": [w0, w1], "padded": [lo, hi],
                         "before": {"window": Jw, "all": Jall, "missing_window": n_missing(p, lo, hi, base_frames),
                                    "missing_all": n_missing(p, frames=base_frames)},
                         "trials": {k: {"window": v[0], "all": v[1], "missing_window": v[6], "missing_all": v[7]}
                                    for k, v in trials.items()},
                         "chosen": best if ok else None, "accepted": bool(ok)}
                log.append(entry)
                print(f"   window {w0}-{w1}: before {fmt(Jw)} / all {fmt(Jall)}; " +
                      ", ".join(f"{k} {fmt(v[0])}/{fmt(v[1])} (miss {v[7]})" for k, v in trials.items()) +
                      f"  -> {'ACCEPT ' + best if ok else 'reject'}", flush=True)
                if ok:  # the trial becomes the agent's working trajectory
                    ovr[nm] = cov
                    shutil.rmtree(run / a.out / nm)
                    shutil.copytree(run / tv / nm, run / a.out / nm)
                    p, alone = pp, aa
            J1 = objective(p, alone, frames=base_frames)
            print(f"{r} {nm}: objective {fmt(J0)} -> {fmt(J1)}  ({time.time() - t0:.0f} s)", flush=True)
            log.append({"object": nm, "summary": {"objective_before": J0, "objective_after": J1,
                                                  "missing_after": n_missing(p, frames=base_frames)}})
        (run / "refine" / "agent_log.json").write_text(json.dumps(log, indent=1))
        (run / "refine" / "agent_overrides.json").write_text(json.dumps(ovr, indent=1))
        for d in run.glob("agent_trial_*"):
            shutil.rmtree(d)
        # per-window stereo-ICP runs: keep only the ones the final overrides reference (agent_overrides.json
        # replays through fuse_objects need them); the rest are dead weight
        keep = {(nm, d) for nm, o in ovr.items() for d in o.get("icp_extra", [])}
        for nm in objs:
            for d in (run / ENV["V2D_OBJ_DIR"] / nm).glob("stereo_icp_agent_*"):
                if (nm, d.name) not in keep:
                    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
