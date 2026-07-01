#!/usr/bin/env python3
"""
Figures for the family-mixture test: cluster-sorted Jaccard heatmaps for key
iterations, silhouette vs product-Bernoulli null, and per-iteration family
composition (stacked bars, recurring families colored consistently).
"""

from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = OUTPUTS
OUT_DIR = BASE / "ss_d128_f1_pronoun_repeat1"
report = json.load(open(OUT_DIR / "family_clustering.json"))
npz = np.load(OUT_DIR / "family_clustering.npz")

HEAT = {"original": [0, 3, 6, 8], "repeat1": [2, 3, 6, 13]}

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.3)

for row, run in enumerate(["original", "repeat1"]):
    for col, t in enumerate(HEAT[run]):
        ax = fig.add_subplot(gs[row, col])
        J = npz[f"{run}_iter{t:02d}_J"]
        lab = npz[f"{run}_iter{t:02d}_labels"]
        sizes = {c: (lab == c).sum() for c in np.unique(lab)}
        order = np.argsort([-sizes[c] * 1000 + c for c in lab], kind="stable")
        ax.imshow(J[np.ix_(order, order)], vmin=0, vmax=1, cmap="viridis")
        b = 0
        for c in sorted(sizes, key=lambda c: -sizes[c]):
            b += sizes[c]
            if b < len(lab):
                ax.axhline(b - 0.5, color="w", lw=1)
                ax.axvline(b - 0.5, color="w", lw=1)
        info = report[run]["iterations"][str(t)]
        ax.set_title(f"{run} iter {t}: N={info['N']}, k={info['k']},\n"
                     f"sil={info['silhouette']:.2f} (null95={info['null95']:.2f})",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

# silhouette vs null
for col, run in enumerate(["original", "repeat1"]):
    ax = fig.add_subplot(gs[2, col])
    its = sorted(int(t) for t in report[run]["iterations"])
    sil = [report[run]["iterations"][str(t)]["silhouette"] for t in its]
    nul = [report[run]["iterations"][str(t)]["null95"] for t in its]
    ax.plot(its, sil, "o-", color="tab:purple", label="real silhouette")
    ax.plot(its, nul, "s--", color="gray", label="null 95th pct")
    ax.set_title(f"{run}: clustering vs product-Bernoulli null", fontsize=9)
    ax.set_xlabel("iteration"); ax.set_ylabel("best silhouette")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

# family composition stacked bars
for col, run in enumerate(["original", "repeat1"]):
    ax = fig.add_subplot(gs[2, col + 2])
    its = sorted(int(t) for t in report[run]["iterations"])
    appear = {}
    for t in its:
        for fid in report[run]["iterations"][str(t)]["families"].values():
            appear[fid] = appear.get(fid, 0) + 1
    recurring = sorted([f for f, n in appear.items() if n >= 2],
                       key=lambda f: -appear[f])
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % 10) for i, f in enumerate(recurring)}
    for t in its:
        info = report[run]["iterations"][str(t)]
        b = 0
        for c, fid in sorted(info["families"].items(),
                             key=lambda kv: -info["cluster_sizes"][kv[0]]):
            n = info["cluster_sizes"][c]
            ax.bar(t, n, bottom=b, color=colors.get(fid, "lightgray"),
                   edgecolor="white", lw=0.4)
            if n >= 8 and fid in colors:
                ax.text(t, b + n / 2, fid, ha="center", va="center", fontsize=6)
            b += n
    ax.set_title(f"{run}: family composition (gray = one-off)", fontsize=9)
    ax.set_xlabel("iteration"); ax.set_ylabel("successful circuits")
    ax.grid(alpha=0.3, axis="y")

fig.suptitle("Family-mixture test: within-iteration circuit clustering "
             "(agglomerative on node Jaccard, silhouette-selected k)", fontsize=12)
out = OUT_DIR / "family_clustering.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
