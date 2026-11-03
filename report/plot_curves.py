"""Training-curve figures for the report from report/curves/<run>.json (extract_curves.py).
fig1_sac_vs_ppo.png  - c2 SAC vs c2 PPO (same 40 episodes, same contact reward): median across episodes + IQR band
fig2_sac_task.png    - SAC task reward: t3 vs t10 friction 1.5 on the 20 scored episodes
fig3_per_episode.png - t10 friction 1.5, episode return per scored episode (small multiples)
One y-axis per panel; x = environment steps (millions)."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("/scratch/kghasemz/v2d/report")
SURF, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5d5d58", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2})
PANELS = [("episode/return_mean_per_completed_episode", "Episode return", 1.0),
          ("episode/reference_progress_mean_per_completed_episode", "Reference progress (fraction of motion)", 1.0),
          ("tracking_error/object_position_m_mean_env_max_per_step", "Object position error (cm)", 100.0),
          ("episode/length_mean_per_completed_episode", "Episode length (control steps)", 1.0)]


def load(pattern):
    out = []
    for f in sorted(glob.glob(str(D / "curves" / pattern))):
        j = json.load(open(f))
        if j.get("total_env_steps"):
            out.append(j)
    return out


def band(runs, key, scale, grid):
    Y = []
    for j in runs:
        x = np.array(j["x_frac"]) * j["total_env_steps"] / 1e6
        y = np.array([np.nan if v is None else v for v in j["series"][key]], float) * scale
        ok = np.isfinite(y)
        if ok.sum() < 5:
            continue
        Y.append(np.interp(grid, x[ok], y[ok], left=np.nan, right=np.nan))
    Y = np.array(Y)
    return np.nanmedian(Y, 0), np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0), len(Y)


def compare(groups, fname, title, labels=True):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for ax, (key, lab, sc) in zip(axs.ravel(), PANELS):
        for name, runs, col in groups:
            xmax = max(j["total_env_steps"] for j in runs) / 1e6
            grid = np.linspace(0, xmax, 300)
            med, lo, hi, n = band(runs, key, sc, grid)
            ax.fill_between(grid, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(grid, med, color=col, label=f"{name} (n={n})")
            i = np.where(np.isfinite(med))[0]
            if labels and len(i):
                ax.annotate(name, (grid[i[-1]], med[i[-1]]), xytext=(4, 0), textcoords="offset points",
                            color=INK2, fontsize=9, va="center")
        ax.set_title(lab, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Environment steps (millions)")
    axs[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.savefig(D / fname, dpi=160)
    print("wrote", D / fname)


def main():
    sac, ppo = load("c2_*_sac.json"), load("c2_*_ppo.json")
    compare([("SAC", sac, BLUE), ("PPO", ppo, ORANGE)], "fig1_sac_vs_ppo.png",
            "SAC vs PPO: same 40 episodes, same contact reward (median across episodes, band = 25-75th pct)")
    t3, t10 = load("t3_e*_task2.json"), load("t10_e*_task2_fr15.json")
    compare([("SAC t3", t3, BLUE), ("SAC t10 friction 1.5", t10, AQUA)], "fig2_sac_task.png",
            "SAC with the task + contact reward on the 20 scored episodes: t3 vs t10 (settled starts, friction 1.5)", labels=False)
    fig, axs = plt.subplots(4, 5, figsize=(14, 9.5), constrained_layout=True, sharex=True)
    for ax, j in zip(axs.ravel(), t10):
        x = np.array(j["x_frac"]) * j["total_env_steps"] / 1e6
        y = np.array([np.nan if v is None else v for v in j["series"]["episode/return_mean_per_completed_episode"]], float)
        ax.plot(x, y, color=AQUA)
        ax.set_title(j["run"].split("_")[1], loc="left", fontsize=10, color=INK)
    for ax in axs[-1]:
        ax.set_xlabel("Env steps (M)")
    fig.suptitle("Episode return per scored episode - SAC t10 friction 1.5", x=0.01, ha="left", fontsize=12.5)
    fig.savefig(D / "fig3_per_episode.png", dpi=140)
    print("wrote", D / "fig3_per_episode.png")
    # numbers for the text
    for name, runs in (("SAC", sac), ("PPO", ppo)):
        fin = [np.nanmean([v for v in j["series"]["episode/reference_progress_mean_per_completed_episode"][-20:] if v is not None]) for j in runs]
        print(name, "final reference progress median", round(float(np.nanmedian(fin)), 3), "env steps", runs[0]["total_env_steps"])


if __name__ == "__main__":
    main()
