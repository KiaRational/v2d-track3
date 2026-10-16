"""Does the verifier see real errors? Checks its per-frame object signals against mocap (host env).

The verifier never uses mocap; on the public split we can ask whether its signals predict where
the reconstruction is actually wrong. Per public episode and object, the per-frame object position
error vs mocap (in our world frame via gt_world.json, the evaluation's robust SE(3); s = 1) is paired
with each verifier metric; over all frames of all episodes this reports the Spearman correlation and
the ROC AUC of the metric for detecting frames with error > --err_cm, and how well the issue
windows cover the high-error frames (recall) and how many flagged frames are really bad (precision).
--fit writes critic_calibration.json (all public episodes) and critic_calibration_loeo.json (per public
episode, fitted without it; the agent uses that entry on public episodes). AUCs use average ranks (ties).
Usage: python validate_critic.py [--tag base] [--err_cm 3]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr

ROOT = Path(os.environ["V2D_ROOT"])


def load_T(p):
    d = json.loads(Path(p).read_text())
    T = np.eye(4)
    if "R" in d:
        T[:3, :3], T[:3, 3] = np.array(d["R"]), d["T"]
        return T
    q = d["rotation"]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = d["translation"]
    return T


def auc(score, label):
    """Probability that a bad frame scores higher than a good one (ties count half)."""
    s, y = np.asarray(score), np.asarray(label, bool)
    if y.all() or not y.any():
        return None
    ranks = rankdata(s)  # average ranks: tied scores (binary flags, saturated fractions) count half
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--objects", default="fused_all_sam3")
    ap.add_argument("--err_cm", type=float, default=3.0)
    ap.add_argument("--fit", action="store_true", help="fit the calibrated error probability")
    a = ap.parse_args()
    runs = (ROOT / "public_runs.txt").read_text().split()
    # metric -> direction (+1: higher = worse)
    metrics = {"free_space": 1, "depth_agree": -1, "mask_iou": -1, "acc_mm": 1, "acc_deg": 1, "table_pen": 1,
               "unsupported": 1, "moved_alone": 1}
    rows = {m: ([], []) for m in metrics}
    flagged, bad_all = [], []
    # observation-only: a trajectory cannot improve these by being frozen or over-smoothed
    # (with acc_mm in the objective the agent learned to 'hold still' carried objects)
    FEAT = ["free_space", "depth_agree", "mask_iou"]
    samples = []  # (episode, features, error) for the calibrated error probability
    for r in runs:
        run = ROOT / "runs" / r
        vf, gf = run / "refine" / f"verify_{a.tag}.json", run / "gt_world.json"
        if not vf.exists() or not gf.exists():
            continue
        V = json.loads(vf.read_text())
        G = json.loads(gf.read_text())
        s3 = G["sim3_world_to_mocap"]["s"]
        slam = {int(p.stem): load_T(p) for p in (run / "recon_hawor/slam_poses").glob("*.json")}
        for nm, ser in V["object"].items():
            gt = G["objects"].get(nm, {})
            pd = run / a.objects / nm / "poses"
            err = {}
            for t_, M in gt.items():
                t = int(t_)
                f = pd / f"{t:06d}.json"
                if f.exists() and t in slam:
                    err[t] = s3 * np.linalg.norm((slam[t] @ load_T(f))[:3, 3] - np.array(M)[:3, 3]) * 100
            win = set()
            for i in V["issues"]:
                if i["target"] == nm:
                    win |= set(range(i["frames"][0], i["frames"][1] + 1))
            for t, e in err.items():
                flagged.append(t in win)
                bad_all.append(e > a.err_cm)
                f_ = [ser.get(m, {}).get(str(t)) for m in FEAT]
                if all(v is not None for v in f_):
                    samples.append((r, f_, e))
            for m, sgn in metrics.items():
                for t_, v in ser.get(m, {}).items():
                    t = int(t_)
                    if t in err and v is not None:
                        rows[m][0].append(sgn * v)
                        rows[m][1].append(err[t])
    print(f"{'metric':14s} {'frames':>7s} {'spearman':>9s} {'AUC(err>' + str(a.err_cm) + 'cm)':>14s}")
    for m, (s, e) in rows.items():
        if len(s) < 20:
            continue
        rho = spearmanr(s, e).correlation
        au = auc(s, np.array(e) > a.err_cm)
        print(f"{m:14s} {len(s):7d} {rho:9.2f} {('-' if au is None else f'{au:.2f}'):>14s}")
    if samples and a.fit:
        class LogisticRegression:  # class-balanced, L2-regularised, Newton iterations (no sklearn here)
            def __init__(self, **_):
                self.coef_, self.intercept_ = None, None

            def fit(self, X, y, lam=1e-2):
                Xb = np.c_[X, np.ones(len(X))]
                w = np.zeros(Xb.shape[1])
                cw = np.where(y, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
                for _ in range(50):
                    p = 1 / (1 + np.exp(-Xb @ w))
                    g = Xb.T @ (cw * (p - y)) / len(y) + lam * w
                    Hm = (Xb * (cw * p * (1 - p))[:, None]).T @ Xb / len(y) + lam * np.eye(len(w))
                    w -= np.linalg.solve(Hm, g)
                self.coef_, self.intercept_ = w[None, :-1], w[-1:]
                return self

            def predict_proba(self, X):
                p = 1 / (1 + np.exp(-(X @ self.coef_[0] + self.intercept_[0])))
                return np.c_[1 - p, p]
        eps = sorted({x[0] for x in samples})
        X = np.array([x[1] for x in samples], float)
        X[:, 1], X[:, 2] = 1 - X[:, 1], 1 - X[:, 2]          # all features: higher = worse
        y = np.array([x[2] > a.err_cm for x in samples])
        ep = np.array([x[0] for x in samples])
        prob = np.zeros(len(y))
        loeo = {}
        for e_ in eps:  # leave-one-episode-out: every score is from a model that never saw that episode
            tr = ep != e_
            if y[tr].any():
                clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
                prob[~tr] = clf.predict_proba(X[~tr])[:, 1]
                loeo[e_] = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
        print(f"\ncalibrated error probability (leave-one-episode-out): AUC {auc(prob, y):.2f}")
        for thr in (0.5, 0.7, 0.8, 0.9):
            f_ = prob > thr
            print(f"  p > {thr}: flags {100 * f_.mean():4.1f}% of frames, recall {100 * (f_ & y).sum() / max(y.sum(), 1):3.0f}%, "
                  f"precision {100 * (f_ & y).sum() / max(f_.sum(), 1):3.0f}%")
        full = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
        (Path(__file__).resolve().parent / "critic_calibration.json").write_text(json.dumps(
            {"features": FEAT, "transform": ["x", "1-x", "1-x"], "coef": full.coef_[0].tolist(),
             "intercept": float(full.intercept_[0]), "err_cm": a.err_cm,
             "note": "fitted on the 20 public episodes (mocap); thresholds chosen with leave-one-episode-out"}, indent=1))
        print("  wrote critic_calibration.json")
        # the agent uses these on the public episodes, so its critic never saw that episode's mocap
        (Path(__file__).resolve().parent / "critic_calibration_loeo.json").write_text(json.dumps(
            {"features": FEAT, "transform": ["x", "1-x", "1-x"], "err_cm": a.err_cm, "episodes": loeo,
             "note": "per public episode: fitted on the other public episodes (leave-one-episode-out)"}, indent=1))
        print(f"  wrote critic_calibration_loeo.json ({len(loeo)} episodes)")
    fl, bd = np.array(flagged), np.array(bad_all)
    if len(fl):
        print(f"\nissue windows vs mocap: {bd.mean() * 100:.0f}% of frames are bad (>{a.err_cm} cm); "
              f"flagged {fl.mean() * 100:.0f}% of frames; recall {100 * (fl & bd).sum() / max(bd.sum(), 1):.0f}%, "
              f"precision {100 * (fl & bd).sum() / max(fl.sum(), 1):.0f}% (chance = {bd.mean() * 100:.0f}%)")


if __name__ == "__main__":
    main()
