#!/usr/bin/env python3
"""
Test the avalanche model (outputs/.../avalanche_model_note/avalanche_model.pdf)
against EVERY universality-pruning run, not just the two in the note.

Runs (auto-detected std vs peel by presence of `peel_steps` in iteration_summary):

  STANDARD (exclude all nodes tied at max universality each iter):
    og10names                 (run O, seeds 0-99,   10-name kernel)
    og10names_repeat          (run O', bit-identical re-run)
    og10names_seeds100_199    (run O'' -- same kernel, seeds 100-199)  <- Prediction 6.1
    cast15_nosplit            (run B, 15-name kernel)
    cast15_fold0              (15-name, fold-0 names-split)
    cast15_og10plus2          (12-name og10+2 rare)

  PEEL / multiexclude (inner loop: repeatedly exclude max-universality tier and set
  aside circuits containing it, until top frequency <= peel_freq_threshold):
    cast15_nosplit_multiexclude
    cast15_fold0_multiexclude
    cast15_og10plus2_multiexclude

What we test
-----------
1. DUMP LAW (Prop 3.1), out-of-sample: predict the per-iteration exclusion count from
   the PREVIOUS iteration's inclusion profile q_v and the CURRENT success count N_t,
   via  E|R| = sum_v P(K_v = max_u K_u),  K_v ~ Binomial(N_t, q_v) (independence approx).
   Compare to observed n_rank1_nodes. The lower bound is sum_v q_v^{N_t}.
2. P(M=N) (parameter-free): predicted P(max universality == 1.0) = 1 - prod_v(1-q_v^N)
   vs the observed indicator (max_u == 1.0). Tests the "tie below 100%" regime.
3. PREDICTION 6.1: og10names vs og10names_seeds100_199 -- same kernel, different seed
   set. Path-dependence of the terminal core.
4. PEEL: step-0 tier out-of-sample (same as std), and per-peel-step IN-SAMPLE ties-at-max
   on the reconstructed shrinking pool (tests the law in the small-pool avalanche regime).

Outputs JSON + figure into avalanche_model_note/.
"""

from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln

BASE = OUTPUTS
NOTE = BASE / "avalanche_model_note"
CACHE = NOTE / "_cache_counts.json"
EPS = 1e-300

STD_RUNS = [
    ("og10names", "O  (seeds 0-99, 10-name)"),
    ("og10names_repeat", "O' (bit-identical)"),
    ("og10names_seeds100_199", "O'' (seeds 100-199)"),
    ("cast15_nosplit", "B  (15-name)"),
    ("cast15_fold0", "fold0 (15-name split)"),
    ("cast15_og10plus2", "og10plus2 (12-name)"),
]
PEEL_RUNS = [
    ("cast15_nosplit_multiexclude", "B-peel"),
    ("cast15_fold0_multiexclude", "fold0-peel"),
    ("cast15_og10plus2_multiexclude", "og10plus2-peel"),
]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def circuit_nodes(mask):
    return [f"{k}#{i}" for k, m in mask.items()
            for i in torch.where(m.bool())[0].cpu().tolist()]


def load_run(exp_dir, want_circuits=False):
    """Per-iteration dict: N, counts{node:cnt over successful circuits}, rank1, max_u,
    obs_dump, peel_steps (if peel). If want_circuits, also store per-seed node-sets."""
    state = json.load(open(exp_dir / "state.json"))
    out = {}
    for i in range(state["next_iter"] + 1):
        sp = exp_dir / f"iter{i:02d}" / "iteration_summary.json"
        if not sp.exists():
            continue
        s = json.load(open(sp))
        ok = [r["seed"] for r in s["seed_results"] if r.get("target_achieved")]
        counts, circuits = {}, {}
        for seed in ok:
            cp = exp_dir / f"iter{i:02d}" / f"seed{seed}_circuit.pt"
            if not cp.exists():
                continue
            nodes = circuit_nodes(torch.load(cp, map_location="cpu", weights_only=True))
            for v in nodes:
                counts[v] = counts.get(v, 0) + 1
            if want_circuits:
                circuits[seed] = nodes
        rec = {"N": len(ok), "counts": counts,
               "rank1": list(s.get("rank1_nodes", [])),
               "obs_dump": s.get("n_excluded_this_iter", s.get("n_rank1_nodes")),
               "max_u": s.get("max_universality"),
               "peel_steps": s.get("peel_steps")}
        if want_circuits:
            rec["circuits"] = circuits
        out[i] = rec
    return out


def load_all():
    if CACHE.exists():
        raw = json.load(open(CACHE))
        # keys come back as strings; restore int iter keys
        return {run: {int(k): v for k, v in d.items()} for run, d in raw.items()}
    data = {}
    for run, _ in STD_RUNS:
        data[run] = load_run(BASE / run, want_circuits=False)
    for run, _ in PEEL_RUNS:
        data[run] = load_run(BASE / run, want_circuits=True)
    json.dump(data, open(CACHE, "w"))
    return data


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def _binom_pmf_cdf(q, n):
    q = np.clip(np.asarray(q, float), 1e-12, 1 - 1e-12)[:, None]
    m = np.arange(n + 1)[None, :]
    logpmf = (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
              + m * np.log(q) + (n - m) * np.log(1 - q))
    pmf = np.exp(logpmf)
    cdf = np.minimum(np.cumsum(pmf, axis=1), 1.0)
    return pmf, cdf


def expected_ties_at_max(q, n):
    """E|R| = sum_v P(K_v = M), M = max_u K_u, K independent Binomial(n, q_v)."""
    if len(q) == 0 or n < 1:
        return 0.0
    pmf, cdf = _binom_pmf_cdf(q, n)
    logcdf = np.log(np.maximum(cdf, EPS))
    L = logcdf.sum(axis=0)[None, :]
    return float((pmf * np.exp(L - logcdf)).sum())


def p_max_eq_n(q, n):
    """P(M = n) = 1 - prod_v (1 - q_v^n)  (the max node is in ALL n circuits)."""
    if len(q) == 0 or n < 1:
        return 0.0
    q = np.clip(np.asarray(q, float), 1e-12, 1 - 1e-12)
    return float(1.0 - np.exp(np.log(np.maximum(1 - q ** n, EPS)).sum()))


def lower_bound(q, n):
    if len(q) == 0 or n < 1:
        return 0.0
    return float((np.asarray(q, float) ** n).sum())


def q_profile(rec):
    """Inclusion profile of an iteration's pool, excluding the nodes it then dumped
    (those are gone in the next state). Frequencies relative to that pool's N."""
    r1 = set(rec["rank1"])
    return np.array([c / rec["N"] for v, c in rec["counts"].items()
                     if v not in r1 and c > 0])


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_standard(data):
    rows = []
    for run, label in STD_RUNS:
        d = data[run]
        for t in sorted(d):
            if t - 1 not in d:
                continue
            prev, cur = d[t - 1], d[t]
            if prev["N"] < 2 or cur["N"] < 2:
                continue
            q = q_profile(prev)
            pred = expected_ties_at_max(q, cur["N"])
            lb = lower_bound(q, cur["N"])
            pmn = p_max_eq_n(q, cur["N"])
            rows.append({
                "run": run, "label": label, "t": t, "N_prev": prev["N"], "N": cur["N"],
                "pred_dump": pred, "lower_bound": lb, "obs_dump": cur["obs_dump"],
                "pred_pMN": pmn, "obs_maxu1": int(abs((cur["max_u"] or 0) - 1.0) < 1e-9),
                "max_u": cur["max_u"],
            })
    return rows


def reconstruct_peel(rec):
    """Replay the inner peel loop from stored circuits, applying the ties-at-max law
    at each step on the current pool. Returns list of per-step dicts."""
    if not rec.get("peel_steps") or not rec.get("circuits"):
        return []
    pool = {s: set(ns) for s, ns in rec["circuits"].items()}
    steps = []
    for st in rec["peel_steps"]:
        if not pool:
            break
        n = len(pool)
        cnt = {}
        for ns in pool.values():
            for v in ns:
                cnt[v] = cnt.get(v, 0) + 1
        q = np.array([c / n for c in cnt.values()])
        pred = expected_ties_at_max(q, n)
        obs_M = st["top_freq"]
        obs_size = st["n_top_nodes"]
        # only the >threshold steps actually exclude (last step is the stop value)
        if obs_size and obs_size > 0 and obs_M and obs_M > 0:
            steps.append({"step": st["step"], "pool_n": n,
                          "pred_size": pred, "obs_size": obs_size,
                          "obs_M": obs_M, "max_freq": max(cnt.values())})
            # set aside circuits containing any of the top nodes
            top = set(st.get("top_nodes", []))
            pool = {s: ns for s, ns in pool.items() if not (ns & top)}
    return steps


def test_peel(data):
    tier0, perstep = [], []
    for run, label in PEEL_RUNS:
        d = data[run]
        for t in sorted(d):
            rec = d[t]
            # tier-0 out-of-sample (prev full profile -> first tier size)
            if t - 1 in d and d[t - 1]["N"] >= 2 and rec["N"] >= 2 and rec.get("peel_steps"):
                s0 = rec["peel_steps"][0]
                if s0["n_top_nodes"] > 0:
                    q = q_profile(d[t - 1])
                    tier0.append({"run": run, "label": label, "t": t, "N": rec["N"],
                                  "pred": expected_ties_at_max(q, rec["N"]),
                                  "obs": s0["n_top_nodes"]})
            # per-step in-sample on shrinking pool
            for s in reconstruct_peel(rec):
                s.update({"run": run, "label": label, "t": t})
                perstep.append(s)
    return tier0, perstep


def seed_ensemble(data):
    """Prediction 6.1: og10names (seeds 0-99) vs og10names_seeds100_199 (100-199)."""
    A = json.load(open(BASE / "og10names" / "state.json"))
    Bd = json.load(open(BASE / "og10names_seeds100_199" / "state.json"))
    exA = set(tuple(x) for x in A["excluded"])
    exB = set(tuple(x) for x in Bd["excluded"])
    inter = exA & exB
    jac = len(inter) / len(exA | exB) if (exA | exB) else 0.0
    da, db = data["og10names"], data["og10names_seeds100_199"]
    return {
        "A_iters": A["next_iter"], "A_core": len(exA),
        "B_iters": Bd["next_iter"], "B_core": len(exB),
        "core_overlap": len(inter), "core_jaccard": jac,
        "A_in_B": len(inter) / len(exA), "B_in_A": len(inter) / len(exB),
        "A_N0": da[0]["N"], "B_N0": db[0]["N"],
        "A_meandump": len(exA) / A["next_iter"], "B_meandump": len(exB) / Bd["next_iter"],
        "A_maxdump": max(da[t]["obs_dump"] for t in da if da[t]["obs_dump"]),
        "B_maxdump": max(db[t]["obs_dump"] for t in db if db[t]["obs_dump"]),
        "A_traj": [da[t]["N"] for t in sorted(da)],
        "B_traj": [db[t]["N"] for t in sorted(db)],
    }


# --------------------------------------------------------------------------- #
def fmt_metrics(rows, pk="pred_dump", ok="obs_dump"):
    p = np.array([r[pk] for r in rows]); o = np.array([r[ok] for r in rows], float)
    lo, ho = np.log10(np.maximum(p, .3)), np.log10(np.maximum(o, .3))
    r = np.corrcoef(lo, ho)[0, 1] if len(rows) > 1 else float("nan")
    medfac = float(np.median(np.maximum(p, .5) / np.maximum(o, .5)))
    return {"n": len(rows), "logcorr": float(r), "median_pred_over_obs": medfac,
            "mae": float(np.abs(p - o).mean()), "rmse": float(np.sqrt(((p - o) ** 2).mean()))}


def main():
    data = load_all()
    std = test_standard(data)
    tier0, perstep = test_peel(data)
    ens = seed_ensemble(data)

    print("\n" + "=" * 92)
    print("1. DUMP LAW (out-of-sample, prev-iter profile -> current dump), STANDARD runs")
    print("=" * 92)
    for run, label in STD_RUNS:
        rr = [r for r in std if r["run"] == run]
        if not rr:
            continue
        print(f"\n  {label}   [{run}]")
        print(f"     {'t':>3} {'N_t':>4} {'pred':>8} {'lbnd':>7} {'obs':>5}  {'P(M=N)':>7} {'maxu=1?':>7}  flag")
        for r in rr:
            flag = "<<< avalanche" if r["N"] < 10 else ""
            print(f"     {r['t']:>3} {r['N']:>4} {r['pred_dump']:>8.1f} {r['lower_bound']:>7.1f}"
                  f" {r['obs_dump']:>5} {r['pred_pMN']:>7.2f} {r['obs_maxu1']:>7}  {flag}")
        print(f"     metrics: {fmt_metrics(rr)}")
    print(f"\n  >> ALL standard transitions: {fmt_metrics(std)}")
    sub = [r for r in std if r["N"] < 10]
    if sub:
        print(f"  >> small-pool (N<10) avalanche transitions: {fmt_metrics(sub)}")

    print("\n" + "=" * 92)
    print("2. P(M=N) parameter-free check (does predicted P(max_u=1) match observed?)")
    print("=" * 92)
    # calibration buckets
    for lo, hi in [(0.0, 0.5), (0.5, 0.9), (0.9, 0.999), (0.999, 1.01)]:
        b = [r for r in std if lo <= r["pred_pMN"] < hi]
        if b:
            frac = np.mean([r["obs_maxu1"] for r in b])
            print(f"   pred P(M=N) in [{lo:.3f},{hi:.3f}): n={len(b):>3}  observed max_u==1 fraction={frac:.2f}")
    # the interesting low-prob cases
    odd = [r for r in std if r["pred_pMN"] < 0.9]
    print(f"\n   transitions where model predicts max_u<1 likely (P(M=N)<0.9): {len(odd)}")
    for r in sorted(odd, key=lambda x: x["pred_pMN"])[:12]:
        print(f"     {r['run']:>24} t={r['t']:>2} N={r['N']:>3}  P(M=N)={r['pred_pMN']:.2f}"
              f"  observed max_u={r['max_u']:.3f}  -> maxu==1? {r['obs_maxu1']}")

    print("\n" + "=" * 92)
    print("3. PREDICTION 6.1: same 10-name kernel, seeds 0-99 vs 100-199")
    print("=" * 92)
    for k, v in ens.items():
        if k.endswith("traj"):
            print(f"   {k}: {v}")
        else:
            print(f"   {k:>16}: {v:.4f}" if isinstance(v, float) else f"   {k:>16}: {v}")

    print("\n" + "=" * 92)
    print("4. PEEL runs")
    print("=" * 92)
    print("   (a) first-tier size, out-of-sample (prev full profile -> step-0 tier):")
    print(f"       {fmt_metrics(tier0, 'pred', 'obs')}")
    print("   (b) per-peel-step in-sample ties-at-max on the shrinking pool:")
    print(f"       {fmt_metrics(perstep, 'pred_size', 'obs_size')}")
    small_steps = [s for s in perstep if s["pool_n"] < 10]
    if small_steps:
        print(f"       small-pool peel steps (pool<10): {fmt_metrics(small_steps,'pred_size','obs_size')}")
        for s in sorted(small_steps, key=lambda x: x["pool_n"])[:10]:
            print(f"         {s['run']:>26} t={s['t']:>2} step={s['step']} pool={s['pool_n']:>3}"
                  f"  pred={s['pred_size']:>6.1f}  obs={s['obs_size']:>4}")

    out = {"standard": std, "peel_tier0": tier0, "peel_perstep": perstep,
           "seed_ensemble": ens,
           "metrics": {"std_all": fmt_metrics(std),
                       "std_small": fmt_metrics(sub) if sub else None,
                       "peel_tier0": fmt_metrics(tier0, "pred", "obs"),
                       "peel_perstep": fmt_metrics(perstep, "pred_size", "obs_size")}}
    json.dump(out, open(NOTE / "model_test_all_runs.json", "w"), indent=2)
    print(f"\nwrote {NOTE / 'model_test_all_runs.json'}")
    return out


if __name__ == "__main__":
    main()
