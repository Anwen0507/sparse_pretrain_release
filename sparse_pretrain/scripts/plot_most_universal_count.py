#!/usr/bin/env python3
"""For every universality-pruning experiment under outputs/universality_pruning, plot the
ABSOLUTE NUMBER of circuits the most-universal node appears in, per iteration.

Universality is a fraction over *successful* circuits (node_universality divides by
n_success), so the absolute count = round(max_universality * n_success) = the number of
successful circuits whose pruned circuit contains the top (rank-1) node. We draw n_success
(total successful circuits = the ceiling) for context, so the gap below it shows circuits
where even the most-universal node is absent, and n_rank1_nodes (how many nodes share that
top count).

The rank-1 sets are disjoint across iterations (each iteration force-excludes its universal
core), so "most universal per iteration" is a per-iteration quantity, not a cross-iteration
tally.

Writes a per-experiment PNG into each exp dir plus a combined small-multiples overview at base.

Usage: plot_most_universal_count.py [BASE_DIR]
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json, glob, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else str(OUTPUTS)

NAN = float("nan")

def load(exp_dir):
    H = json.load(open(os.path.join(exp_dir, "state.json")))["history"]
    its   = [h["iteration"] for h in H]
    nsucc = [h.get("n_success") for h in H]
    nseed = [h.get("num_seeds") for h in H]
    nr    = [h.get("n_rank1_nodes", len(h.get("rank1_nodes", []))) for h in H]
    # absolute number of successful circuits containing the most-universal node
    cnt = []
    for h in H:
        mu, ns = h.get("max_universality"), h.get("n_success")
        cnt.append(NAN if (mu is None or ns is None) else round(mu * ns))
    return its, cnt, nsucc, nseed, nr

data = []
for d in sorted(glob.glob(os.path.join(BASE, "*"))):
    if os.path.isfile(os.path.join(d, "state.json")):
        data.append((os.path.basename(d),) + load(d))

ymax = max((max([s for s in nseed if s] + [0]) for _, _, _, _, nseed, _ in data), default=100)

# ---------- per-experiment PNG ----------
for name, its, cnt, nsucc, nseed, nr in data:
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    # top: absolute number of circuits containing the most-universal node
    axT.plot(its, nsucc, "--", color="0.6", lw=1.3, label="# successful circuits (ceiling)")
    axT.plot(its, cnt, "-o", color="C0", lw=2, ms=5,
             label="# circuits with the most-universal node")
    axT.set_ylabel("number of circuits")
    axT.set_ylim(0, ymax * 1.08)
    axT.grid(alpha=.3)
    axT.legend(loc="upper right", fontsize=8, framealpha=.9)
    axT.set_title(f"{name} — circuits containing the most-universal node per iteration")
    for x, c in zip(its, cnt):
        if c == c:  # not NaN
            axT.annotate(f"{int(c)}", xy=(x, c), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=6.5, color="C0")
    # bottom: how many nodes share that top count (rank-1 set size)
    axB.bar(its, nr, color="C1", alpha=.85)
    axB.set_ylabel("# nodes at\nthat count")
    axB.set_xlabel("iteration")
    axB.grid(alpha=.3)
    for x, n in zip(its, nr):
        if n:
            axB.annotate(str(n), xy=(x, n), xytext=(0, 2), textcoords="offset points",
                         ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, name, "most_universal_node_count_per_iter.png"), dpi=130)
    plt.close(fig)

# ---------- combined small-multiples overview ----------
n = len(data); ncol = 3; nrow = math.ceil(n / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.2 * nrow), squeeze=False)
for i, (name, its, cnt, nsucc, nseed, nr) in enumerate(data):
    ax = axes[i // ncol][i % ncol]
    ax.plot(its, nsucc, "--", color="0.6", lw=1.1, label="# successful circuits")
    ax.plot(its, cnt, "-o", color="C0", lw=1.8, ms=4, label="# with top node")
    ax.set_ylim(0, ymax * 1.08)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("iteration"); ax.set_ylabel("# circuits", fontsize=8)
    ax.grid(alpha=.3)
    ax2 = ax.twinx()  # rank-1 set size as faint bars
    ax2.bar(its, nr, color="C1", alpha=.22, zorder=0)
    ax2.set_ylabel("# rank-1 nodes", fontsize=8, color="C1")
    ax2.tick_params(axis="y", labelcolor="C1", labelsize=7)
    if i == 0:
        ax.legend(loc="upper right", fontsize=7, framealpha=.9)
for j in range(n, nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
fig.suptitle("Number of circuits containing the most-universal node per iteration "
             "(line, left) with rank-1 set size (bars, right) — all experiments", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(BASE, "most_universal_node_count_all.png")
fig.savefig(out, dpi=130); plt.close(fig)

print(f"Saved combined -> {out}  | {n} experiments")
for name, its, cnt, nsucc, nseed, nr in data:
    vals = [c for c in cnt if c == c]
    print(f"  {name:32s} iters={len(its):>3}  circuits-with-top-node range "
          f"[{int(min(vals)) if vals else 0}, {int(max(vals)) if vals else 0}]  peak #rank1={max(nr) if nr else 0}")
