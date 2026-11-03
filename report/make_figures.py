"""Report figures 4-11 (static PNG, light theme, reference palette; one y-axis per panel).
fig4  Kaggle score per submission (small multiples per metric)
fig5  standing per metric vs every team (current leaderboards)
fig6  per-episode success heatmap (MP-SR and SP-SR) x training run, selected run outlined
fig7  reward terms over training (SAC t10 friction 1.5, median over 20 scored episodes)
fig8  objects at rest: rotation in 3 s before vs after simulator settling
fig9  object-pose variants vs public mocap (DEV ONLY): moving / resting frames within 15 deg & 3 cm
fig10 dust pan: stereo-depth top surface vs mesh before/after thinning
fig11 hard episodes across runs (AUC and MP-SR)"""
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("/scratch/kghasemz/v2d")
OUT = R / "report"
FC = R / "robot/fc_out"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#5d5d58", "#a9a8a1", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "axes.axisbelow": True})
EPS = ["01", "03", "05", "06", "08", "10", "14", "15", "16", "19", "20", "25", "28", "30", "32", "33", "35", "36", "37", "38"]
RUNS = [("t3", "t3_e{e}_task2"), ("t6", "t6_e{e}_task2"), ("t6 seed 7", "t6_e{e}_task2s7"),
        ("t10", "t10_e{e}_task2"), ("t10 friction 1.5", "t10_e{e}_task2_fr15")]


def metrics(run):
    f = sorted(glob.glob(str(FC / run / "eval/metrics_*.json")))
    if not f:
        return None
    return json.load(open(f[-1]))["metrics"]


def run_name(tmpl, e):
    r = tmpl.format(e=e)
    if e == "20" and r.startswith("t6_"):
        r = r.replace("t6_e20", "t7_e20")
    return r


def save(fig, name):
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", name, flush=True)


def fig4():
    subs = ["Oct 15*", "t3", "best1", "best2", "best3"]
    data = {"AUC (higher better)": [0.529, 0.504, 0.532, 0.553, 0.584], "SP-SR (higher better)": [0.40, 0.50, 0.55, 0.60, 0.65],
            "MP-SR (higher better)": [0.15, 0.15, 0.20, 0.15, 0.15], "RPE cm (lower better)": [8.50, 9.23, 8.10, 7.54, 7.35],
            "MPPE cm (lower better)": [6.84, 8.60, 7.82, 7.18, 6.06], "CD-O cm (lower better)": [0.088, 0.407, 0.407, 0.401, 0.401]}
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2), constrained_layout=True)
    for ax, (k, v) in zip(axs.ravel(), data.items()):
        cols = [MUTED] + [CAT[0]] * 4
        ax.bar(range(5), v, color=cols, width=0.6)
        for i, x in enumerate(v):
            ax.annotate(f"{x:g}", (i, x), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(range(5), subs)
        ax.set_title(k, loc="left", fontsize=10.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Kaggle scores per submission  (* Oct 15 used the organiser scans as meshes - not a compliant submission)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "fig4_kaggle_history.png")


def fig5():
    lb = {"auc": ("AUC", False), "sp-sr": ("SP-SR", False), "mp-sr": ("MP-SR", False), "rpe": ("RPE (cm)", True),
          "mppe": ("MPPE (cm)", True), "cd-o": ("CD-O (cm)", True)}
    fig, axs = plt.subplots(1, 6, figsize=(14, 5.2), constrained_layout=True)
    for ax, (k, (lab, lower)) in zip(axs, lb.items()):
        f = sorted(glob.glob(str(R / f"submission/lb4/{k}/*.csv")))[-1]
        rows = [l.strip() for l in open(f, encoding="utf-8-sig").readlines()[1:] if l.strip()]
        sc, ours, rank = [], None, None
        for l in rows:
            parts = re.findall(r'"[^"]*"|[^,]+', l)
            s = float(parts[4]); sc.append(s)
            if "Kiarash" in l:
                ours, rank = s, int(parts[0])
        sc = np.array(sc)
        off = 0
        if k == "cd-o":
            off = int((sc >= 5).sum()); sc = sc[sc < 5]
        if k in ("rpe", "mppe"):   # a few teams report thousands of cm: keep the field readable
            off = int((sc > 30).sum()); sc = sc[sc <= 30]
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(sc))
        ax.scatter(jitter, sc, s=22, color=MUTED, zorder=2)
        ax.scatter([0], [ours], s=80, color=CAT[0], zorder=3, edgecolor=SURF, linewidth=2)
        ax.annotate(f"us: #{rank} of {len(rows)}", (0, ours), xytext=(10, 0), textcoords="offset points", fontsize=9,
                    color=INK, va="center")
        ax.set_xlim(-0.5, 1.2)
        ax.set_xticks([])
        ax.set_title(lab + ("  (lower better)" if lower else "") + (f"\n{off} teams off-scale" if off else ""), loc="left", fontsize=10)
        if lower:
            ax.invert_yaxis()
    fig.suptitle("Our standing per metric vs every team (current public leaderboards; best of each team)", x=0.01, ha="left",
                 fontsize=12)
    save(fig, "fig5_leaderboard.png")


def fig6():
    sel = {l.split()[0]: l.split()[1] for l in open(R / "submission/best3/selection.txt") if l.strip()}
    for key, lab, name in (("maniptrans_sr", "MP-SR (strict: every object < 3 cm & < 30 deg)", "fig6_episode_mpsr.png"),
                           ("spider_sr_uncentered", "SP-SR (loose: < 10 cm & < 0.5 rad)", "fig6_episode_spsr.png")):
        M = np.full((len(EPS), len(RUNS)), np.nan)
        for i, e in enumerate(EPS):
            for j, (_, tmpl) in enumerate(RUNS):
                m = metrics(run_name(tmpl, e))
                if m:
                    M[i, j] = m[key]
        fig, ax = plt.subplots(figsize=(7.4, 9), constrained_layout=True)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f2f6fc"] + BLUE_RAMP)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, e in enumerate(EPS):
            for j, (_, tmpl) in enumerate(RUNS):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if M[i, j] > 0.55 else INK)
                if sel.get(e) == run_name(tmpl, e):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=CAT[1], linewidth=2.2))
        ax.set_xticks(range(len(RUNS)), [r[0] for r in RUNS], rotation=20)
        ax.set_yticks(range(len(EPS)), [f"e{e}" for e in EPS])
        ax.grid(False)
        ax.set_title(f"{lab}\nper scored episode and run, own-reference evaluation; orange = run submitted in best3",
                     loc="left", fontsize=10)
        save(fig, name)


def fig7():
    keys = [("reward_components/object_keypoints_weighted_mean_per_env_step", "object keypoints"),
            ("reward_components/contact_location_weighted_mean_per_env_step", "contact location"),
            ("reward_components/reference_progress_weighted_mean_per_env_step", "reference progress"),
            ("reward_components/task_completion_weighted_mean_per_env_step", "task completion"),
            ("reward_components/termination_weighted_mean_per_env_step", "termination penalty"),
            ("reward/total_mean_per_env_step", "total reward")]
    runs = [json.load(open(f)) for f in sorted(glob.glob(str(OUT / "curves/t10_e*_task2_fr15.json")))]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    grid = np.linspace(0, 250, 300)
    for ax, (k, lab) in zip(axs.ravel(), keys):
        Y = []
        for j in runs:
            x = np.array(j["x_frac"]) * j["total_env_steps"] / 1e6
            y = np.array([np.nan if v is None else v for v in j["series"][k]], float)
            ok = np.isfinite(y)
            if ok.sum() > 5:
                Y.append(np.interp(grid, x[ok], y[ok]))
        Y = np.array(Y)
        ax.fill_between(grid, np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0), color=CAT[2], alpha=0.15, linewidth=0)
        ax.plot(grid, np.nanmedian(Y, 0), color=CAT[2])
        ax.set_title(lab + " (per env step)", loc="left", fontsize=10.5)
        ax.set_xlabel("Environment steps (millions)")
    fig.suptitle("Reward terms during training - SAC t10 friction 1.5, median over 20 scored episodes (band 25-75th pct)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "fig7_reward_terms.png")


def fig8():
    before = {}
    for l in open(R / "beat1/diag/settle/lift_summary.txt"):
        p = l.split()
        if p[0] == "lift0_alone0":
            before[(p[1].replace("t7_", "t6_"), p[2].rstrip(":"))] = float(p[4])
    after = {}
    for l in open(R / "logs/t10-check_2038018.out"):
        m = re.match(r"(t\d_e\d+)\s+ROT_T body (\d+): ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", l.strip())
        if m:
            after[(m.group(1).replace("t7_", "t6_"), m.group(2))] = float(m.group(5))
    names = {}
    for k in before:
        b = int(k[1]) - 52 if k[1].isdigit() and int(k[1]) >= 52 else int(k[1])
        names[k] = (k[0], str(b))
    pairs = []
    for k, v in before.items():
        kk = (k[0], names[k][1])
        if kk in after:
            pairs.append((f"{k[0][3:]} obj{int(kk[1]) + 1}", v, after[kk]))
    pairs.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    for i, (lab, b, a) in enumerate(pairs):
        ax.plot([min(a, b), max(a, b)], [i, i], color=GRID, linewidth=2.5, zorder=1)
    ax.scatter([p[1] for p in pairs], range(len(pairs)), s=46, color=CAT[1], label="before (reconstructed rest pose)", zorder=2)
    ax.scatter([p[2] for p in pairs], range(len(pairs)), s=46, color=CAT[0], label="after simulator settling", zorder=3)
    ax.set_yticks(range(len(pairs)), [p[0] for p in pairs], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 150)
    ax.set_xlabel("object rotation after 3 s at rest, no robot (deg, symlog)")
    ax.legend(frameon=False, loc="lower right")
    mb, ma = np.mean([p[1] for p in pairs]), np.mean([p[2] for p in pairs])
    ax.set_title(f"Objects at rest in the simulator: mean rotation {mb:.1f} -> {ma:.1f} deg", loc="left", fontsize=11)
    save(fig, "fig8_sim_settle.png")


def fig9():
    files = ["bench_c3", "bench_hg", "bench_tp", "bench_rest", "bench_spin", "bench_hlock", "bench_hla"]
    lab = {"fused_all_recon_c2": "c2 fusion", "fused_bfix_recon": "c3 fusion + flip repair", "fused_phys_recon": "rigid lift-off lock",
           "fused_hg_recon": "soft hand pose graph", "tp_recon_vit": "c14 candidate Viterbi", "tp_recon": "c14 + track anchoring",
           "tpf_recon": "c14 + pose filter (c17)", "tp_hg_recon": "c14 + hand graph", "tpfr_recon": "c17 + rest hold",
           "tpfs_recon": "c17 + spin guard (c19)", "fused_hlock_recon": "hand lock (raw meas.)", "fused_hla_recon": "air lock"}
    acc = {}
    for f in files:
        p = R / f"beat1/diag/{f}.jsonl"
        if not p.exists():
            continue
        for l in open(p):
            try:
                j = json.loads(l)
            except Exception:
                continue
            v = j.get("variant")
            if v not in lab or not j.get("moving"):
                continue
            acc.setdefault(v, {"m": {}, "r": {}})
            acc[v]["m"][(j["run"], j["object"])] = j["moving"]["ok15deg_3cm"]
            if j.get("rest"):
                acc[v]["r"][(j["run"], j["object"])] = j["rest"]["ok15deg_3cm"]
    rows = [(lab[v], np.mean(list(d["m"].values())), np.mean(list(d["r"].values())) if d["r"] else np.nan, len(d["m"]))
            for v, d in acc.items()]
    rows.sort(key=lambda x: x[1])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True, sharey=True)
    for ax, k, t in ((axs[0], 1, "Moving frames (object carried)"), (axs[1], 2, "Resting frames")):
        vals = [r[k] for r in rows]
        ax.barh(range(len(rows)), vals, color=[CAT[0] if "c14" in r[0] or "c17" in r[0] or "c19" in r[0] else MUTED for r in rows],
                height=0.6)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1.05)
        ax.set_title(t + " within 15 deg & 3 cm of mocap", loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    fig.suptitle("Object-pose variants vs public mocap (20 public episodes, 33 objects; development only - not used by the pipeline)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "fig9_pose_variants_mocap.png")


def fig10():
    txt = open(R / "logs/mesh-fit_2032042.out").read()
    rows = re.findall(r"^\s+([\d.]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*$", txt, re.M)
    x = np.array([float(r[0]) for r in rows]); dep = np.array([float(r[1]) for r in rows])
    top = np.array([float(r[2]) for r in rows]); und = np.array([float(r[3]) for r in rows])
    thin = und + 0.30 * (top - und)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    ax.fill_between(x, und, top, color=CAT[1], alpha=0.12, linewidth=0)
    ax.plot(x, top, color=CAT[1], label="mesh top surface - before (3.05 cm thick)")
    ax.plot(x, thin, color=CAT[0], label="mesh top surface - after thinning x0.30 (0.92 cm)")
    ax.plot(x, und, color=MUTED, linestyle="--", label="mesh underside (unchanged)")
    ax.scatter(x, dep, s=40, color=INK, zorder=3, label="stereo depth, top surface (p90)")
    ax.set_xlabel("position along the dust pan from the ring handle (cm)")
    ax.set_ylabel("height above the table (mm)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Dust pan resting in eval ep38: reconstructed mesh vs measured depth (mean top error 17.3 -> 2.3 mm)",
                 loc="left", fontsize=10.5)
    save(fig, "fig10_dustpan_depth.png")


def fig11():
    hard = ["19", "25", "30", "32", "38", "10", "03"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, key, t in ((axs[0], "add_auc", "AUC of ADD"), (axs[1], "maniptrans_sr", "MP-SR (strict success)")):
        w = 0.15
        for j, (rl, tmpl) in enumerate(RUNS):
            vals = []
            for e in hard:
                m = metrics(run_name(tmpl, e))
                vals.append(m[key] if m else np.nan)
            ax.bar(np.arange(len(hard)) + (j - 2) * w, vals, width=w * 0.9, color=CAT[j], label=rl)
        ax.set_xticks(range(len(hard)), [f"e{e}" for e in hard])
        ax.set_ylim(0, 1.05)
        ax.set_title(t, loc="left", fontsize=10.5)
        ax.grid(axis="x", visible=False)
    axs[0].legend(frameon=False, fontsize=8.5, ncol=2)
    fig.suptitle("Hard episodes across training runs (own-reference evaluation): settled starts + friction 1.5 solve the dust pan (e38)",
                 x=0.01, ha="left", fontsize=11.5)
    save(fig, "fig11_hard_episodes.png")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    for f in (fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11):
        if only and f.__name__ not in only:
            continue
        try:
            f()
        except Exception as ex:   # one broken figure must not stop the others
            print("FAILED", f.__name__, repr(ex), flush=True)
