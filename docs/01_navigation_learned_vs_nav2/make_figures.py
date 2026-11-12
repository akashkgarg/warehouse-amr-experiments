"""Figures for the warehouse AMR report. Run with any Python that has matplotlib."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HOME = os.path.expanduser("~/warehouse_amr")
OUT = os.path.join(HOME, "report", "media")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load_log(path):
    rows = []
    for line in open(path):
        if line.startswith("{"):
            r = json.loads(line)
            if "stats" in r:
                rows.append(r)
    return rows


def smooth(y, k=5):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def training_curves():
    rows = load_log(os.path.join(HOME, "runs/sensor4/train.log"))
    x = np.array([r["samples"] for r in rows]) / 1e6
    dagger_end = max(r["samples"] for r in rows if r["phase"] == "dagger") / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, t in zip(axes, ("carter", "forklift")):
        for k, c in (("success", "tab:green"), ("collision", "tab:red"), ("human_contact", "tab:orange"),
                     ("stuck", "tab:gray")):
            y = [r["stats"].get(t, {}).get(k, np.nan) for r in rows]
            ax.plot(x, smooth(y, 9), color=c, label=k.replace("_", " "))
        ax.axvline(dagger_end, color="k", ls="--", lw=1)
        ax.text(dagger_end, 0.93, " DAgger | PPO →", fontsize=9)
        ax.set_title(f"{t} (rolling training episodes)")
        ax.set_xlabel("env samples (millions)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="center right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"), dpi=130)


def encoding_fix():
    old = load_log(os.path.join(HOME, "runs/sensor3_linscan/train.log"))
    new = [r for r in load_log(os.path.join(HOME, "runs/sensor4/train.log")) if r["phase"] == "dagger"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for rows, label, c in ((old, "scan / 10 m (old)", "tab:red"), (new, "1 / (1 + 2·clearance) (new)", "tab:blue")):
        it = [r["iter"] for r in rows if r["phase"] == "dagger"]
        y = [r["stats"]["carter"]["collision"] for r in rows if r["phase"] == "dagger"]
        ax.plot(it, smooth(y, 5), color=c, label=label)
    ax.set_xlabel("DAgger iteration (expert share β goes 1 → 0 by iter 20)")
    ax.set_ylabel("carter collision rate")
    ax.legend(frameon=False)
    ax.set_title("Student collisions vs. lidar encoding")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "encoding_fix.png"), dpi=130)


def carter_spinout():
    # Measured with /tmp/wh_turnprobe2.py: carter_0 yaw rate (rad/s) per 0.1 s step under a constant
    # [speed 0.8, turn 0.9] command (= 1.2 m/s, 1.35 rad/s requested).
    stock = [1.71, 1.99, 2.24, 2.59, 3.01, 3.46, 3.71, 3.96, 4.29, 4.66, 4.95, 4.67]
    cap = [1.09, 1.24, 1.40, 1.52, 1.59, 1.59, 1.60, 1.61, 1.60, 1.61, 1.61, 1.60]
    fb = [1.03, 1.28, 1.41, 1.48, 1.49, 1.49, 1.48, 1.49, 1.48, 1.49, 1.48, 1.49]
    t = np.arange(len(stock)) * 0.1
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(t, stock, "o-", color="tab:red", label="stock Nova Carter")
    ax.plot(t, cap, "o-", color="tab:orange", label="+ 0.7 m/s² lateral accel cap")
    ax.plot(t, fb, "o-", color="tab:blue", label="+ gyro yaw-rate feedback")
    ax.axhline(1.35, color="k", ls="--", lw=1)
    ax.text(0.02, 1.15, "commanded 1.35 rad/s", fontsize=9)
    ax.set_xlabel("time since command (s)")
    ax.set_ylabel("measured yaw rate (rad/s)")
    ax.set_title("Carter spin-out under a forward + turn command")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "carter_spinout.png"), dpi=130)


def expert_time_breakdown():
    # From /tmp/wh_expertdiag.py (1500 steps, 3 carters, random goals), before and after the drive + corridor fixes.
    cats = ["full speed", "slowing for turn/goal", "near obstacle", "turning in place", "yielding to worker",
            "backing up"]
    before = [14.0, 21.6, 7.2, 55.6, 0.6, 1.1]
    after = [59.6, 12.4, 14.5, 4.6, 5.0, 4.0]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(cats))
    ax.barh(y + 0.2, before, 0.4, color="tab:red", label="before (mean 0.08 m/s)")
    ax.barh(y - 0.2, after, 0.4, color="tab:blue", label="after (mean 0.78 m/s)")
    ax.set_yticks(y, cats)
    ax.invert_yaxis()
    ax.set_xlabel("% of carter driving time")
    ax.set_title("What the scripted carter driver spends its time doing")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "expert_time.png"), dpi=130)


def final_results(bench_dir):
    runs = [("expert.json", "Scripted expert\n(ground truth)"), ("nav2.json", "Nav2 + AMCL"),
            ("dagger_amcl.json", "Learned (DAgger)\n+ AMCL"), ("ppo_amcl.json", "Learned (PPO)\n+ AMCL"),
            ("ppo_noisemodel.json", "Learned (PPO)\nnoise model")]
    runs = [(f, l) for f, l in runs if os.path.exists(os.path.join(bench_dir, f))]
    data = {f: json.load(open(os.path.join(bench_dir, f)))["summary"] for f, _ in runs}
    keys = [("success_rate", "success", "tab:green"), ("collision_rate", "collision", "tab:red"),
            ("human_contact_rate", "worker contact", "tab:orange"), ("stuck_rate", "stuck", "tab:gray")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, t in zip(axes, ("carter", "forklift")):
        x = np.arange(len(runs))
        for j, (k, label, c) in enumerate(keys):
            ax.bar(x + (j - 1.5) * 0.2, [data[f][t][k] for f, _ in runs], 0.2, color=c, label=label)
        ns = [data[f][t]["episodes"] for f, _ in runs]
        ax.set_title(f"{t} ({min(ns)}–{max(ns)} episodes per controller)")
        ax.set_xticks(x, [l for _, l in runs], fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, loc="upper center", ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_results.png"), dpi=130)


if __name__ == "__main__":
    what = sys.argv[1:] or ["training", "encoding", "spinout", "expert"]
    if "training" in what:
        training_curves()
    if "encoding" in what:
        encoding_fix()
    if "spinout" in what:
        carter_spinout()
    if "expert" in what:
        expert_time_breakdown()
    if "final" in what:
        final_results(os.path.join(HOME, "bench/final"))
    print("ok")
