#!/usr/bin/env python3
"""
Corrected dump law: the exclusion rule is 'all nodes tied at the MAXIMUM
empirical frequency', not 'frequency 1.0'. With counts K_v ~ Bin(N, q_v)
(independent across seeds), under across-node independence:

  E|R| = sum_v P(K_v = M) = sum_v sum_m pmf_v(m) * prod_{u != v} cdf_u(m)

Compare old lower bound (sum q^N), corrected formula, and observed dumps,
predicting iteration t from iteration t-1's profile. Also P(M = N) per
iteration, and a counterfactual-N sweep on a healthy profile.
"""

from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln

BASE = OUTPUTS
RUNS = [
    ("original", BASE / "ss_d128_f1_pronoun"),
    ("repeat1", BASE / "ss_d128_f1_pronoun_repeat1"),
]
EPS = 1e-300


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
            for node in circuit_nodes(torch.load(it_dir / f"seed{seed}_circuit.pt",
                                                 map_location="cpu", weights_only=True)):
                counts[node] = counts.get(node, 0) + 1
        out[i] = {"N": len(ok), "counts": counts,
                  "rank1": set(s.get("rank1_nodes", [])),
                  "max_u": s.get("max_universality"),
                  "obs_dump": s.get("n_rank1_nodes")}
    return out


def q_profile(prev):
    """q estimates from a previous iteration, minus its rank-1 exclusions."""
    return np.array([c / prev["N"] for v, c in prev["counts"].items()
                     if v not in prev["rank1"]])


def binom_pmf_cdf(q, n):
    """pmf[v, m], cdf[v, m] for m = 0..n; q clipped away from {0,1}."""
    q = np.clip(q, 1e-12, 1 - 1e-12)[:, None]
    m = np.arange(n + 1)[None, :]
    logpmf = (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
              + m * np.log(q) + (n - m) * np.log(1 - q))
    pmf = np.exp(logpmf)
    cdf = np.minimum(np.cumsum(pmf, axis=1), 1.0)
    return pmf, cdf


def expected_ties_at_max(q, n):
    """E|R|, P(M = n) under across-node independence."""
    pmf, cdf = binom_pmf_cdf(q, n)
    logcdf = np.log(np.maximum(cdf, EPS))
    L = logcdf.sum(axis=0)[None, :]            # log prod_u cdf_u(m)
    ties = (pmf * np.exp(L - logcdf)).sum()    # sum_v sum_m pmf*prod_{u!=v}cdf
    p_m_eq_n = 1.0 - np.exp(np.log(np.maximum(1 - pmf[:, n], EPS)).sum())
    return ties, p_m_eq_n


def main():
    for label, exp_dir in RUNS:
        run = load_run(exp_dir)
        its = sorted(i for i in run if run[i]["N"] >= 2)
        print(f"\n=== {label} ===")
        print(f"{'iter':>4} {'N_t':>4} {'max_u':>6} {'obs':>5} {'old SUMq^N':>11} "
              f"{'new ties@max':>13} {'P(M=N)':>8}")
        for t in its:
            if t - 1 not in run or run[t - 1]["N"] < 2:
                continue
            q = q_profile(run[t - 1])
            n = run[t]["N"]
            old = float((q ** n).sum())
            new, pmn = expected_ties_at_max(q, n)
            r = run[t]
            print(f"{t:>4} {n:>4} {r['max_u']:>6.3f} {r['obs_dump']:>5} "
                  f"{old:>11.1f} {new:>13.1f} {pmn:>8.2f}")

        healthy = max((i for i in its if run[i]["N"] >= 80), default=its[0])
        q = q_profile(run[healthy])
        print(f"  counterfactual sweep on iter-{healthy} profile "
              f"(N={run[healthy]['N']}, post-rank1):")
        print(f"  {'N_t':>5} {'old SUMq^N':>11} {'new ties@max':>13} {'P(M=N)':>8}")
        for n in [2, 3, 5, 10, 15, 20, 30, 50, 100]:
            old = float((q ** n).sum())
            new, pmn = expected_ties_at_max(q, n)
            print(f"  {n:>5} {old:>11.1f} {new:>13.1f} {pmn:>8.2f}")


if __name__ == "__main__":
    main()
