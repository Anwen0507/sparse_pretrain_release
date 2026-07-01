#!/usr/bin/env python3
"""
Jaccard similarity of node sets BETWEEN iterations of a universality-pruning run.

Per iteration: union of nodes over all successful circuits, minus nodes that
appear in only one circuit (count >= 2 filter). Then Jaccard between these
per-iteration sets: consecutive-iteration series + full pairwise matrix, for
both the original and repeat ss_d128_f1_pronoun runs.
"""

from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = OUTPUTS
RUNS = [
    ("original (2026-05-26)", BASE / "ss_d128_f1_pronoun", "tab:blue"),
    ("seeds 100-199 (2026-06-12, fresh seeds)", BASE / "ss_d128_f1_pronoun_seeds100_199", "tab:orange"),
]


def circuit_nodes(circuit_mask):
    nodes = set()
    for key, m in circuit_mask.items():
        for idx in torch.where(m.bool())[0].cpu().tolist():
            nodes.add((key, idx))
    return nodes


def iteration_sets(exp_dir):
    """{iteration: (set of nodes in >=2 successful circuits, n_success)}"""
    state = json.load(open(exp_dir / "state.json"))
    sets = {}
    for i in range(state["next_iter"] + 1):
        it_dir = exp_dir / f"iter{i:02d}"
        sp = it_dir / "iteration_summary.json"
        if not sp.exists():
            continue
        summary = json.load(open(sp))
        ok_seeds = [r["seed"] for r in summary["seed_results"] if r["target_achieved"]]
        counts = {}
        for seed in ok_seeds:
            mask = torch.load(it_dir / f"seed{seed}_circuit.pt",
                              map_location="cpu", weights_only=True)
            for node in circuit_nodes(mask):
                counts[node] = counts.get(node, 0) + 1
        sets[i] = ({n for n, c in counts.items() if c >= 2}, len(ok_seeds))
    return sets


def jaccard(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main():
    fig = plt.figure(figsize=(14, 5))
    ax_line = fig.add_subplot(1, 3, 1)
    ax_mats = [fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]

    results = {}
    for (label, exp_dir, color), ax_m in zip(RUNS, ax_mats):
        sets = iteration_sets(exp_dir)
        its = sorted(i for i, (s, n) in sets.items() if n >= 2)  # need >=2 circuits
        print(f"{label}:")
        for i in its:
            s, n = sets[i]
            print(f"  iter {i:2d}: n_success={n:3d}, |nodes in >=2 circuits|={len(s)}")

        consec = [(its[k + 1], jaccard(sets[its[k]][0], sets[its[k + 1]][0]))
                  for k in range(len(its) - 1)]
        ax_line.plot([x for x, _ in consec], [y for _, y in consec], "o-",
                     color=color, label=label)

        M = np.array([[jaccard(sets[a][0], sets[b][0]) for b in its] for a in its])
        im = ax_m.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax_m.set_xticks(range(len(its)), its, fontsize=7)
        ax_m.set_yticks(range(len(its)), its, fontsize=7)
        ax_m.set_title(f"pairwise Jaccard: {label}", fontsize=10)
        ax_m.set_xlabel("iteration")
        fig.colorbar(im, ax=ax_m, fraction=0.046)

        results[label] = {
            "iterations": its,
            "set_sizes": {i: len(sets[i][0]) for i in its},
            "n_success": {i: sets[i][1] for i in its},
            "consecutive_jaccard": {f"{its[k]}->{its[k+1]}": consec[k][1]
                                    for k in range(len(consec))},
            "pairwise_jaccard": M.tolist(),
        }

    ax_line.set_xlabel("iteration")
    ax_line.set_ylabel("Jaccard vs previous iteration")
    ax_line.set_title("Consecutive-iteration Jaccard\n(nodes in ≥2 successful circuits)")
    ax_line.set_ylim(0, 1)
    ax_line.grid(alpha=0.3)
    ax_line.legend(fontsize=8)

    fig.suptitle("Between-iteration node-set similarity (union of successful circuits, "
                 "singleton nodes removed)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = RUNS[1][1] / "iteration_jaccard.png"
    fig.savefig(out_png, dpi=150)
    out_json = RUNS[1][1] / "iteration_jaccard.json"
    json.dump(results, open(out_json, "w"), indent=2, default=str)
    print(f"wrote {out_png}\nwrote {out_json}")


if __name__ == "__main__":
    main()
