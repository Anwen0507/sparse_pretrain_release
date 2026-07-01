#!/usr/bin/env python3
"""Plot per-iteration loss across all seeds: mean/min/percentile band of loss_at_all_active
(the loss of each seed's full trained mask) plus the mean discovered-circuit loss."""
from sparse_pretrain.paths import OUTPUTS
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
TARGET = 0.15

iters, mean_all, min_all, p10, p90, mean_circ, nsucc = [], [], [], [], [], [], []
for it in range(0, 50):
    summ = EXP / f"iter{it:02d}" / "iteration_summary.json"
    if not summ.exists():
        continue
    # iteration_summary.json["seed_results"] is the complete per-seed record for EVERY
    # iteration (the in-process iter 0 never wrote per-seed *_result.json files).
    rs = json.load(open(summ))["seed_results"]
    L = np.array([r["loss_at_all_active"] for r in rs if r.get("loss_at_all_active") is not None], float)
    cl = [r["circuit_loss"] for r in rs if r.get("target_achieved") and r.get("circuit_loss") is not None]
    iters.append(it)
    mean_all.append(L.mean()); min_all.append(L.min())
    p10.append(np.percentile(L, 10)); p90.append(np.percentile(L, 90))
    mean_circ.append(np.mean(cl) if cl else np.nan)
    nsucc.append(sum(r.get("target_achieved", False) for r in rs))

iters = np.array(iters)
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.fill_between(iters, p10, p90, color="C0", alpha=0.18, label="10–90th pct (all 100 seeds)")
ax.plot(iters, mean_all, "-o", color="C0", lw=2, label="mean loss of full trained mask (all 100 seeds)")
ax.plot(iters, min_all, "--", color="C9", lw=1.5, label="best seed (min loss)")
ax.plot(iters, mean_circ, "-s", color="C2", lw=1.8, label="mean discovered-circuit loss (successful seeds)")
ax.axhline(TARGET, color="C3", ls="--", lw=2, label=f"target loss = {TARGET}")

# annotate success counts
for x, y, n in zip(iters, mean_all, nsucc):
    ax.annotate(f"{n}/100", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color="C0")
# mark exhaustion
ax.axvspan(iters[-1] - 0.5, iters[-1] + 0.5, color="C3", alpha=0.08)
ax.annotate("EXHAUSTED\n(0/100)", (iters[-1], mean_all[-1]), textcoords="offset points",
            xytext=(-6, -28), ha="right", fontsize=9, color="C3", fontweight="bold")

ax.set_xlabel("iteration (cumulative universal nodes excluded)")
ax.set_ylabel("task loss")
ax.set_xticks(iters)
ax.set_ylim(0, max(p90) * 1.08)
ax.set_title("Mean loss across circuits vs iteration — ss_d128_f1 / dummy_pronoun\n"
             "loss rises as universal nodes are removed; best seed crosses target only at exhaustion")
ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(EXP / "mean_loss_trajectory.png", dpi=140)
print("saved", EXP / "mean_loss_trajectory.png")
print("iter  nsucc  mean_all  min_all  mean_circuit")
for i, it in enumerate(iters):
    mc = f"{mean_circ[i]:.3f}" if mean_circ[i] == mean_circ[i] else "  -  "
    print(f"{it:>4} {nsucc[i]:>5}  {mean_all[i]:>8.3f} {min_all[i]:>8.3f}  {mc}")
