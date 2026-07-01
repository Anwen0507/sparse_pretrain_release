#!/usr/bin/env python3
"""
Test the family-mixture hypothesis: cluster successful circuits within each
iteration on pairwise node-Jaccard, compare clustering quality (silhouette)
against a product-Bernoulli null with the same q profile (no correlations),
extract cluster cores, and track families across iterations.

Outputs: console table, family_clustering.json, family_clustering.npz
(per-iteration Jaccard matrices + labels) in the repeat1 exp dir.
"""

from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

BASE = OUTPUTS
RUNS = [
    ("original", BASE / "ss_d128_f1_pronoun"),
    ("repeat1", BASE / "ss_d128_f1_pronoun_repeat1"),
]
OUT_DIR = BASE / "ss_d128_f1_pronoun_repeat1"
RNG = np.random.default_rng(0)
MIN_N_CLUSTER = 5
KMAX = 8
NULL_REPS = 30
CORE_FRAC = 0.8
MATCH_J = 0.3


def circuit_nodes(circuit_mask):
    nodes = set()
    for key, m in circuit_mask.items():
        for idx in torch.where(m.bool())[0].cpu().tolist():
            nodes.add(f"{key}#{idx}")
    return frozenset(nodes)


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
        if not ok:
            continue
        sets = [circuit_nodes(torch.load(it_dir / f"seed{seed}_circuit.pt",
                                         map_location="cpu", weights_only=True))
                for seed in ok]
        out[i] = {"N": len(ok), "sets": sets, "seeds": ok,
                  "rank1": set(s.get("rank1_nodes", []))}
    return out


def bool_matrix(sets):
    nodes = sorted(set().union(*sets))
    idx = {v: j for j, v in enumerate(nodes)}
    X = np.zeros((len(sets), len(nodes)), dtype=bool)
    for i, s in enumerate(sets):
        X[i, [idx[v] for v in s]] = True
    return X, nodes


def jaccard_from_bool(X):
    inter = (X.astype(np.float64) @ X.T.astype(np.float64))
    sizes = X.sum(1).astype(np.float64)
    union = sizes[:, None] + sizes[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def silhouette(D, labels):
    n = len(labels)
    vals = np.zeros(n)
    cl = np.unique(labels)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any() or len(cl) < 2:
            vals[i] = 0.0
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in cl if c != labels[i])
        m = max(a, b)
        vals[i] = (b - a) / m if m > 0 else 0.0
    return float(vals.mean())


def best_clustering(J, kmax):
    D = 1.0 - J
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    best = (1, np.ones(len(J), dtype=int), -1.0)
    for k in range(2, min(kmax, len(J) - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        sil = silhouette(D, lab)
        if sil > best[2]:
            best = (k, lab, sil)
    return best


def null_best_sils(q, N, kmax, reps):
    out = []
    for _ in range(reps):
        X = RNG.random((N, len(q))) < q[None, :]
        X = X[X.sum(1) > 0]
        if len(X) < 3:
            out.append(0.0)
            continue
        out.append(best_clustering(jaccard_from_bool(X), kmax)[2])
    return np.array(out)


def cluster_cores(X, nodes, labels, frac=CORE_FRAC):
    cores = {}
    for c in np.unique(labels):
        sub = X[labels == c]
        cores[int(c)] = {nodes[j] for j in np.where(sub.mean(0) >= frac)[0]}
    return cores


def jacc(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main():
    npz, report = {}, {}
    for label, exp_dir in RUNS:
        run = load_run(exp_dir)
        its = sorted(run)
        report[label] = {"iterations": {}, "lineage": []}
        families = []  # {id, core, last_iter}
        print(f"\n=== {label} ===")
        print(f"{'iter':>4} {'N':>4} {'k*':>3} {'sil':>6} {'null95':>7} {'verdict':>9} "
              f"{'withinJ':>8} {'betwJ':>6}  sizes -> family ids")
        for t in its:
            r = run[t]
            X, nodes = bool_matrix(r["sets"])
            J = jaccard_from_bool(X)
            npz[f"{label}_iter{t:02d}_J"] = J

            if r["N"] >= MIN_N_CLUSTER:
                k, lab, sil = best_clustering(J, KMAX)
                q = X.mean(0)
                nulls = null_best_sils(q, r["N"], KMAX, NULL_REPS)
                null95 = float(np.quantile(nulls, 0.95))
                clustered = sil > null95
            else:
                k, lab, sil = 1, np.ones(r["N"], dtype=int), float("nan")
                nulls, null95, clustered = np.array([]), float("nan"), False
            npz[f"{label}_iter{t:02d}_labels"] = lab

            cores = cluster_cores(X, nodes, lab)
            off = ~np.eye(len(J), dtype=bool)
            samec = lab[:, None] == lab[None, :]
            withinJ = float(J[samec & off].mean()) if (samec & off).any() else float("nan")
            betwJ = float(J[~samec].mean()) if (~samec).any() else float("nan")

            # family matching: restrict an old core to nodes still allowed at t
            assignments = {}
            cands = []
            for c, core in cores.items():
                for f in families:
                    dropped = set()
                    for s in range(f["last_iter"], t):
                        if s in run:
                            dropped |= run[s]["rank1"]
                    jv = jacc(core, f["core"] - dropped)
                    cands.append((jv, c, f["id"]))
            used_c, used_f = set(), set()
            for jv, c, fid in sorted(cands, reverse=True):
                if jv < MATCH_J or c in used_c or fid in used_f:
                    continue
                assignments[c] = fid
                used_c.add(c)
                used_f.add(fid)
                f = next(f for f in families if f["id"] == fid)
                f["core"], f["last_iter"] = cores[c], t
            for c in cores:
                if c not in assignments:
                    fid = f"F{len(families)}"
                    families.append({"id": fid, "core": cores[c], "last_iter": t})
                    assignments[c] = fid

            sizes = {int(c): int((lab == c).sum()) for c in np.unique(lab)}
            fam_str = ", ".join(f"{sizes[c]}->{assignments[c]}" for c in sorted(sizes))
            verdict = "CLUSTERED" if clustered else ("n/a" if r["N"] < MIN_N_CLUSTER else "single")
            print(f"{t:>4} {r['N']:>4} {k:>3} {sil:>6.3f} {null95:>7.3f} {verdict:>9} "
                  f"{withinJ:>8.3f} {betwJ:>6.3f}  {fam_str}")

            report[label]["iterations"][t] = {
                "N": r["N"], "k": k, "silhouette": sil, "null95": null95,
                "clustered": bool(clustered), "within_J": withinJ, "between_J": betwJ,
                "cluster_sizes": sizes,
                "families": {str(c): assignments[c] for c in assignments},
                "cores": {str(c): sorted(v) for c, v in cores.items()},
            }
        report[label]["lineage"] = [
            {"id": f["id"], "last_iter": f["last_iter"], "core_size": len(f["core"])}
            for f in families
        ]

    # --- transition analysis: repeat iterations 3 -> 4/5 survivors -> 6 ---
    label, exp_dir = RUNS[1]
    run = load_run(exp_dir)
    rep = report[label]["iterations"]
    trans = {}
    if all(t in run for t in (3, 4, 5, 6)):
        cores3 = {c: set(v) for c, v in rep[3]["cores"].items()}
        cores6 = {c: set(v) for c, v in rep[6]["cores"].items()}
        drop34 = run[3]["rank1"]
        drop36 = run[3]["rank1"] | run[4]["rank1"] | run[5]["rank1"]
        print("\n=== repeat transition: iter-3 clusters vs 4/5 survivors vs iter-6 clusters ===")
        for t in (4, 5):
            for i, s in enumerate(run[t]["sets"]):
                js = {c: round(jacc(s, core - drop34), 3) for c, core in cores3.items()}
                j6 = {c: round(jacc(s, core), 3) for c, core in cores6.items()}
                print(f"iter-{t} survivor {i} (|C|={len(s)}): J to iter3 cores {js} | to iter6 cores {j6}")
                trans[f"iter{t}_survivor{i}"] = {"to_iter3": js, "to_iter6": j6}
        m = {f"3.{c3}->6.{c6}": round(jacc(core3 - drop36, core6), 3)
             for c3, core3 in cores3.items() for c6, core6 in cores6.items()}
        print("iter-3 cores (minus nodes excluded by iter 6) vs iter-6 cores:", m)
        trans["cores3_vs_cores6"] = m
    report["transition"] = trans

    json.dump(report, open(OUT_DIR / "family_clustering.json", "w"), indent=2, default=str)
    np.savez_compressed(OUT_DIR / "family_clustering.npz", **npz)
    print(f"\nwrote {OUT_DIR / 'family_clustering.json'}")
    print(f"wrote {OUT_DIR / 'family_clustering.npz'}")


if __name__ == "__main__":
    main()
