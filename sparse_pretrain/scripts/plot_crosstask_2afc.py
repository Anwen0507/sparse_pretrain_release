#!/usr/bin/env python3
"""Regenerate the cross-task ablation plot with 2AFC accuracy (from accuracy_2afc.json)."""
from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
d = json.load(open(EXP / "accuracy_2afc.json"))["crosstask_2afc"]
order = ["dummy_pronoun", "ioi_relaxed", "dummy_tense", "dummy_article"]
base = [100 * d[t]["base"] for t in order]
core = [100 * d[t]["core"] for t in order]
rmu = [100 * d[t]["rand_mean"] for t in order]
rsd = [100 * d[t]["rand_std"] for t in order]

fig, ax = plt.subplots(figsize=(10.5, 6))
x = np.arange(len(order)); w = 0.27
ax.bar(x - w, base, w, label="baseline (full model)", color="C0")
ax.bar(x, core, w, label="core-ablated (−84 core)", color="C3")
ax.bar(x + w, rmu, w, yerr=rsd, capsize=4, label="random-84 control", color="C7")
ax.axhline(50, ls="--", color="k", lw=1.5, label="chance (50%)")
for i, c in enumerate(core):
    ax.text(i, c + 1.5, f"{c:.0f}%", ha="center", va="bottom", fontsize=9, color="C3", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=12)
ax.set_ylabel("2AFC accuracy: P(correct logit > incorrect logit)")
ax.set_ylim(0, 108)
ax.set_title("Ablating the 84-node pronoun core across tasks — 2AFC accuracy (ss_d128_f1)\n"
             "core-specific = red collapses toward chance vs blue AND vs grey")
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(EXP / "crosstask_ablation.png", dpi=140)
print("regenerated", EXP / "crosstask_ablation.png")
print("  task            base  core  rand")
for t, b, c, r in zip(order, base, core, rmu):
    print(f"  {t:>14}  {b:4.0f}% {c:4.0f}% {r:4.0f}%")
