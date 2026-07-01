#!/usr/bin/env python3
"""
Consecutive-iteration Jaccard for ALL universality-pruning runs.

Reuses iteration_sets()/jaccard() from iteration_jaccard.py: per iteration,
union of nodes over successful circuits filtered to nodes in >=2 circuits;
Jaccard between consecutive iterations (iterations with <2 successful
circuits are skipped, matching the original script).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sparse_pretrain.scripts.iteration_jaccard import BASE, iteration_sets, jaccard


def main():
    exp_dirs = sorted(d for d in BASE.iterdir() if (d / "state.json").exists())

    fig, ax = plt.subplots(figsize=(10, 6))
    ncols = 4
    nrows = (len(exp_dirs) + ncols - 1) // ncols
    fig_hm, axes_hm = plt.subplots(nrows, ncols,
                                   figsize=(4.2 * ncols, 4.0 * nrows))
    axes_hm = np.atleast_1d(axes_hm).ravel()
    cmap = plt.get_cmap("tab10")
    results = {}
    for ci, exp_dir in enumerate(exp_dirs):
        sets = iteration_sets(exp_dir)
        its = sorted(i for i, (s, n) in sets.items() if n >= 2)
        print(f"{exp_dir.name}:")
        for i in its:
            s, n = sets[i]
            print(f"  iter {i:2d}: n_success={n:3d}, |nodes in >=2 circuits|={len(s)}")

        consec = [(its[k + 1], jaccard(sets[its[k]][0], sets[its[k + 1]][0]))
                  for k in range(len(its) - 1)]
        ax.plot([x for x, _ in consec], [y for _, y in consec], "o-",
                color=cmap(ci % 10), label=exp_dir.name)

        M = np.array([[jaccard(sets[a][0], sets[b][0]) for b in its] for a in its])
        ax_m = axes_hm[ci]
        im = ax_m.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax_m.set_xticks(range(len(its)), its, fontsize=6)
        ax_m.set_yticks(range(len(its)), its, fontsize=6)
        ax_m.set_title(exp_dir.name, fontsize=9)
        ax_m.set_xlabel("iteration", fontsize=8)
        fig_hm.colorbar(im, ax=ax_m, fraction=0.046)

        results[exp_dir.name] = {
            "iterations": its,
            "set_sizes": {i: len(sets[i][0]) for i in its},
            "n_success": {i: sets[i][1] for i in its},
            "consecutive_jaccard": {f"{its[k]}->{its[k+1]}": consec[k][1]
                                    for k in range(len(consec))},
            "pairwise_jaccard": M.tolist(),
        }

    ax.set_xlabel("iteration")
    ax.set_ylabel("Jaccard vs previous iteration")
    ax.set_title("Consecutive-iteration Jaccard, all runs\n"
                 "(nodes in ≥2 successful circuits, singletons removed)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_png = BASE / "iteration_jaccard_all.png"
    fig.savefig(out_png, dpi=150)

    for ax_m in axes_hm[len(exp_dirs):]:
        ax_m.set_visible(False)
    fig_hm.suptitle("Pairwise between-iteration Jaccard per run "
                    "(nodes in ≥2 successful circuits)")
    fig_hm.tight_layout(rect=[0, 0, 1, 0.96])
    out_hm = BASE / "iteration_jaccard_all_heatmaps.png"
    fig_hm.savefig(out_hm, dpi=150)

    out_json = BASE / "iteration_jaccard_all.json"
    json.dump(results, open(out_json, "w"), indent=2, default=str)
    print(f"wrote {out_png}\nwrote {out_hm}\nwrote {out_json}")


if __name__ == "__main__":
    main()
