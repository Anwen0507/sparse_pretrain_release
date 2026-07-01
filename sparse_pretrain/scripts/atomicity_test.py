#!/usr/bin/env python3
"""
Atomicity test for the atom-excision design.

QUESTION
--------
If NMF atoms are truly atomic (independent building blocks), then knocking out one
atom's nodes and re-pruning should leave the OTHER atoms intact -- so re-factoring the
survivors should recover them. If atoms interact, the survivors drift. This adjudicates
whether the iterative *re-factoring* MCTS tree is well-defined (atomic) or whether we
must fall back to a fixed-dictionary subset search (interacting).

METHOD
------
Take the existing atom-excision output. Re-factor (NMF) three circuit ensembles over a
COMMON node space and Hungarian-match the control's SURVIVING atoms (all but the knocked-
out target) into each, reporting the assigned cosine per survivor:
  - atom@4   : survivors -> re-factored atom-knockout circuits      (the test)
  - random@4 : survivors -> re-factored matched-random-knockout     (matched-perturbation baseline)
  - bootstrap: survivors -> re-factored resamples of control        (pure-resampling noise CEILING)

READOUT
-------
  rec_atom ~= ceiling                      -> atoms ARE atomic (knockout disrupts survivors
                                              no more than resampling noise); re-factoring tree OK.
  ceiling > rec_atom ~= rec_random         -> generic perturbation drift, not atom-specific.
  rec_atom < rec_random                    -> the atom knockout SPECIFICALLY disrupts the
                                              survivors -> atoms INTERACT -> fixed-dictionary.
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

from sparse_pretrain.scripts.universality_pruning_experiment import circuit_nodes


def load_circuits(it_dir):
    return [circuit_nodes(torch.load(p, map_location="cpu", weights_only=True))
            for p in sorted(Path(it_dir).glob("seed*_circuit.pt"))]


def build_X(circuit_sets, index):
    X = np.zeros((len(circuit_sets), len(index)), dtype=np.float32)
    for i, s in enumerate(circuit_sets):
        for nd in s:
            j = index.get(nd)
            if j is not None:
                X[i, j] = 1.0
    return X


def factor(X, k, seed=0):
    """NMF over active columns, padded back to the full common node space."""
    amask = X.sum(0) > 0
    k = max(1, min(k, X.shape[0] - 1, int(amask.sum())))
    Xa = X[:, amask]
    nmf = NMF(n_components=k, init="nndsvda", solver="cd", max_iter=800, tol=1e-4, random_state=seed)
    nmf.fit_transform(Xa)
    H = np.zeros((k, X.shape[1]), dtype=np.float64)
    H[:, np.where(amask)[0]] = nmf.components_
    return H


def cos_matrix(Hs, Ht):
    A = Hs / np.maximum(np.linalg.norm(Hs, axis=1, keepdims=True), 1e-12)
    B = Ht / np.maximum(np.linalg.norm(Ht, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def matched_cosines(Hs, Ht):
    """Hungarian max-cosine assignment; returns the assigned cosine for each row of Hs."""
    C = cos_matrix(Hs, Ht)
    ri, ci = linear_sum_assignment(-C)
    out = np.full(Hs.shape[0], np.nan)
    out[ri] = C[ri, ci]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True, help="atom-excision output dir (has atom_excision.json + iter dirs)")
    ap.add_argument("--dose", type=int, default=4, help="which dose's knockout to test (default 4 = all atom nodes)")
    ap.add_argument("--boot", type=int, default=8, help="bootstrap resamples for the noise ceiling")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp = Path(args.exp_dir)
    report = json.load(open(exp / "atom_excision.json"))
    K = int(report["k"])
    conds = report["conditions"]
    atom_iter = next(c["iterN"] for c in conds if c["kind"] == "atom" and c["dose"] == args.dose)
    rand_iters = [c["iterN"] for c in conds if c["kind"] == "random" and c["dose"] == args.dose]
    ctrl_iter = next(c["iterN"] for c in conds if c["kind"] == "control")
    forbid = set()
    for s in report["target"]["candidate_nodes"][:args.dose]:
        key, idx = s.rsplit("#", 1)
        forbid.add((key, int(idx)))

    ctrl = load_circuits(exp / f"iter{ctrl_iter:02d}")
    atom = load_circuits(exp / f"iter{atom_iter:02d}")
    rand = []
    for it in rand_iters:
        rand += load_circuits(exp / f"iter{it:02d}")
    print(f"K={K} | control n={len(ctrl)}  atom@{args.dose} n={len(atom)}  random@{args.dose} n={len(rand)}", flush=True)
    print(f"knocked-out (forbidden) nodes: {sorted(f'{k}#{i}' for k, i in forbid)}", flush=True)

    # common node space across all three ensembles
    nodes = sorted(set().union(*ctrl, *atom, *rand)) if (ctrl and atom and rand) else []
    index = {nd: j for j, nd in enumerate(nodes)}
    Xc, Xa, Xr = build_X(ctrl, index), build_X(atom, index), build_X(rand, index)

    # control dictionary; identify the knocked-out atom = the one loading most on forbidden nodes
    Hc = factor(Xc, K, args.seed)
    fcols = np.array([index[n] for n in forbid if n in index], dtype=int)
    tgt = int(np.argmax(Hc[:, fcols].sum(1))) if fcols.size else 0
    survivors = [a for a in range(Hc.shape[0]) if a != tgt]
    Hs = Hc[survivors]
    print(f"control atom {tgt} carries the forbidden nodes (load {Hc[tgt, fcols].sum():.2f}); "
          f"testing recurrence of the other {len(survivors)} atoms\n", flush=True)

    Ha = factor(Xa, K, args.seed)
    Hr = factor(Xr, K, args.seed)
    rec_atom = matched_cosines(Hs, Ha)
    rec_rand = matched_cosines(Hs, Hr)

    # noise ceiling: re-factor bootstrap resamples of control, match the same survivors
    rng = np.random.default_rng(args.seed)
    boots = []
    for b in range(args.boot):
        idx = rng.integers(0, len(ctrl), len(ctrl))
        Hb = factor(Xc[idx], K, args.seed + 1 + b)
        boots.append(matched_cosines(Hs, Hb))
    rec_boot = np.nanmean(np.stack(boots), axis=0)

    def summ(v):
        v = v[~np.isnan(v)]
        return f"mean={v.mean():.3f} median={np.median(v):.3f} min={v.min():.3f}  (>=0.8: {int((v>=0.8).sum())}/{len(v)})"

    print(f"recurrence of surviving atoms (Hungarian-matched cosine):")
    print(f"  CEILING  (bootstrap control) : {summ(rec_boot)}")
    print(f"  atom@{args.dose}   knockout         : {summ(rec_atom)}")
    print(f"  random@{args.dose} knockout         : {summ(rec_rand)}")

    ma, mr, mb = np.nanmean(rec_atom), np.nanmean(rec_rand), np.nanmean(rec_boot)
    print(f"\n  gaps:  ceiling-atom = {mb-ma:+.3f}   random-atom = {mr-ma:+.3f}", flush=True)
    if mb - ma <= 0.05:
        verdict = "ATOMIC: knockout disrupts survivors no more than resampling noise -> re-factoring tree is well-defined."
    elif ma >= mr - 0.03:
        verdict = "GENERIC DRIFT: survivors drift under perturbation, but the atom knockout is no worse than a matched-random knockout -> not atom-specific; re-factoring usable with atom-matching."
    else:
        verdict = "INTERACTING: the atom knockout disrupts survivors MORE than a matched-random knockout -> atoms are NOT independent -> use the fixed-dictionary subset search, not a re-factoring tree."
    print(f"  VERDICT: {verdict}", flush=True)

    # per-survivor detail for the atom knockout (which recur, which don't)
    print(f"\n  per-survivor atom@{args.dose} recurrence (control atom -> best cosine), sorted:")
    order = np.argsort(rec_atom)
    for a in order:
        tn = "?"  # brief label: top node of the control survivor
        top = int(np.argmax(Hc[survivors[a]]))
        tn = nodes[top][0] + "#" + str(nodes[top][1])
        print(f"    ctrl atom {survivors[a]:>2} (top {tn:>22}): atom@{args.dose} cos={rec_atom[a]:.3f}  "
              f"random cos={rec_rand[a]:.3f}  ceiling cos={rec_boot[a]:.3f}", flush=True)

    out = {"k": K, "dose": args.dose, "target_atom": tgt, "n_survivors": len(survivors),
           "forbidden": sorted(f"{k}#{i}" for k, i in forbid),
           "rec_atom_mean": float(ma), "rec_random_mean": float(mr), "rec_ceiling_mean": float(mb),
           "rec_atom": [float(x) for x in rec_atom], "rec_random": [float(x) for x in rec_rand],
           "rec_ceiling": [float(x) for x in rec_boot], "verdict": verdict}
    json.dump(out, open(exp / "atomicity_test.json", "w"), indent=2)
    print(f"\nwrote {exp / 'atomicity_test.json'}", flush=True)


if __name__ == "__main__":
    main()
