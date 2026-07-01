#!/usr/bin/env python3
"""
Check the 'tie law' for rank-1 block dumps: with N successful circuits and
per-node inclusion probabilities q_v, the expected number of nodes contained in
ALL N circuits (= tied at universality 1.0) is sum_v q_v^N.

Honest test: predict iteration t's dump size from iteration t-1's q profile
(minus t-1's rank-1 exclusions) at the observed N_t. Also sweep counterfactual
N on a fixed healthy profile to show the explosion at small N.
"""

from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path

import torch

BASE = OUTPUTS
RUNS = [
    ("original", BASE / "ss_d128_f1_pronoun"),
    ("repeat1", BASE / "ss_d128_f1_pronoun_repeat1"),
]


def circuit_nodes(circuit_mask):
    nodes = set()
    for key, m in circuit_mask.items():
        for idx in torch.where(m.bool())[0].cpu().tolist():
            nodes.add(f"{key}#{idx}")
    return nodes


def load_run(exp_dir):
    state = json.load(open(exp_dir / "state.json"))
    out = {}
    for i in range(state["next_iter"] + 1):
        it_dir = exp_dir / f"iter{i:02d}"
        sp = it_dir / "iteration_summary.json"
        if not sp.exists():
            continue
        s = json.load(open(sp))
        ok = [r["seed"] for r in s["seed_results"] if r["target_achieved"]]
        counts = {}
        for seed in ok:
            mask = torch.load(it_dir / f"seed{seed}_circuit.pt",
                              map_location="cpu", weights_only=True)
            for node in circuit_nodes(mask):
                counts[node] = counts.get(node, 0) + 1
        out[i] = {
            "N": len(ok),
            "counts": counts,
            "rank1": set(s.get("rank1_nodes", [])),
            "max_u": s.get("max_universality"),
            "obs_dump": s.get("n_rank1_nodes"),
        }
    return out


def pred_dump(counts, N_prev, rank1_prev, n):
    """sum of (q_v)^n over nodes surviving the previous exclusion."""
    return sum((c / N_prev) ** n for v, c in counts.items() if v not in rank1_prev)


def main():
    for label, exp_dir in RUNS:
        run = load_run(exp_dir)
        its = sorted(i for i in run if run[i]["N"] >= 2)
        print(f"\n=== {label} ===")
        print(f"{'iter':>4} {'N_t':>4} {'max_u':>6} {'obs dump':>9} {'pred (prev q)':>14}")
        for t in its:
            if t - 1 not in run or run[t - 1]["N"] < 2:
                pred = None
            else:
                prev = run[t - 1]
                pred = pred_dump(prev["counts"], prev["N"], prev["rank1"], run[t]["N"])
            r = run[t]
            ps = f"{pred:10.1f}" if pred is not None else "       n/a"
            print(f"{t:>4} {r['N']:>4} {r['max_u']:>6.3f} {r['obs_dump']:>9} {ps:>14}")

        # counterfactual-N sweep on a healthy profile (last iter with N>=80)
        healthy = max((i for i in its if run[i]["N"] >= 80), default=its[0])
        prof = run[healthy]
        print(f"  counterfactual dump size on iter-{healthy} profile (N={prof['N']}):")
        for n in [2, 3, 5, 10, 15, 20, 30, 50, 100]:
            v = sum((c / prof["N"]) ** n for c in prof["counts"].values())
            print(f"    if N_t={n:>3}: expected nodes-in-all = {v:8.1f}")


if __name__ == "__main__":
    main()
