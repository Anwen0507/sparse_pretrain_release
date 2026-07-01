#!/usr/bin/env python3
"""Trajectory plot for a 'peel' universality-pruning run (universality_pruning_experiment_peel.py).

Matches the 2x2 style of universality_pruning_report.py but uses peel-aware fields
(n_excluded_this_iter, peel_sequence, n_peel_ablation_steps) and overlays the original
rank-1 run for comparison.

Usage: plot_peel_trajectory.py PEEL_EXP_DIR [BASELINE_EXP_DIR]
"""
import sys, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(sys.argv[1])
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else None

st = json.load(open(EXP / "state.json")); H = st["history"]
its = [h["iteration"] for h in H]
succ = [h["n_success"] for h in H]
excl_before = [h.get("excluded_before", 0) for h in H]
excl_this = [h.get("n_excluded_this_iter", 0) for h in H]
nsteps = [h.get("n_peel_ablation_steps", 0) for h in H]
peelseq = [h.get("peel_sequence", []) for h in H]

def col(key, sub="mean"):
    return [((h.get(key) or {}).get(sub) if isinstance(h.get(key), dict) else None) for h in H]
size = col("circuit_size_nodes")
njac, eunw, ewt = col("node_jaccard"), col("edge_jaccard_unweighted"), col("edge_jaccard_weighted")
def xy(y):
    pts = [(x, v) for x, v in zip(its, y) if v is not None]
    return ([p[0] for p in pts], [p[1] for p in pts])

bh = bn = None
if BASE and (BASE / "state.json").exists():
    bst = json.load(open(BASE / "state.json")); bh = bst["history"]; bn = len(bst["excluded"])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A: success rate + mean circuit size
a = ax[0, 0]
a.plot(its, succ, "-o", color="C3", label="success")
a.axhline(0, color="grey", lw=.8, ls=":")
a.set_xlabel("iteration"); a.set_ylabel("circuits at target loss / 100", color="C3")
a.set_ylim(-4, 104); a.set_title("Success rate & mean circuit size"); a.grid(alpha=.3)
a2 = a.twinx(); xs, ys = xy(size); a2.plot(xs, ys, "-s", color="C0")
a2.set_ylabel("mean circuit size (nodes)", color="C0")

# B: circuit agreement
a = ax[0, 1]
for y, lab, m in [(njac, "node", "-o"), (eunw, "edge (unweighted)", "-^"), (ewt, "edge (|W|-weighted)", "-s")]:
    xs, ys = xy(y); a.plot(xs, ys, m, label=lab)
a.set_xlabel("iteration"); a.set_ylabel("mean pairwise Jaccard")
a.set_title("Circuit agreement (similarity)"); a.legend(); a.grid(alpha=.3); a.set_ylim(0, .7)

# C: cumulative excluded -- peel vs original
a = ax[1, 0]
a.plot(its, excl_before, "-o", color="C2", label="peel (thr=1)")
if bh is not None:
    a.plot([h["iteration"] for h in bh], [h.get("excluded_before", 0) for h in bh],
           "--x", color="grey", label="original rank-1")
a.legend()
a.set_xlabel("iteration"); a.set_ylabel("cumulative nodes excluded (at iter start)")
a.set_title("Cumulative excluded universal nodes"); a.grid(alpha=.3)

# D: per-iter exclusions, colored by whether peeling fired (>1 ablation step), labelled with peel_sequence
a = ax[1, 1]
colors = ["C1" if n >= 2 else "C4" for n in nsteps]
a.bar(its, excl_this, color=colors)
for x, v, seq, n in zip(its, excl_this, peelseq, nsteps):
    if n >= 2 or v >= 30:
        a.annotate(str(seq), (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color="C1", label="peeling fired (>1 step) → found secondary cluster"),
                  Patch(color="C4", label="single step (= rank-1 tier)")], fontsize=8)
a.set_xlabel("iteration"); a.set_ylabel("nodes excluded this iter")
a.set_title("Excluded per iteration (label = peel_sequence)"); a.grid(alpha=.3)

cmp = f"   |   original rank-1: {bn} nodes / {bh[-1]['iteration']} iters" if bh is not None else ""
fig.suptitle("Universality pruning — PEEL variant (thr=1, rank-1 fallback) — ss_d128_f1 / dummy_pronoun (target 0.15)\n"
             f"exhausted at iter {its[-1]}; {len(st['excluded'])} nodes excluded{cmp}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = EXP / "peel_trajectory.png"; fig.savefig(out, dpi=140)
print("Saved ->", out)
