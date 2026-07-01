#!/usr/bin/env python3
"""
Metric x clustering-algorithm robustness check for ONE iteration of a
motif-excision run. Re-clusters the iteration's successful circuits under three
circuit-similarity representations and compares to the node-Jaccard baseline:

  (1) node-set Jaccard          -- current baseline (a circuit = bag of nodes)
  (2) weighted-edge Jaccard     -- Ruzicka over |W|-weighted edges (typed by family)
  (3) typed Weisfeiler-Lehman   -- subtree kernel; node label = location type,
                                   edge label = weight-family, directed, h rounds

clustered with:
  - UPGMA (average linkage) on all three  -> isolates the METRIC effect (same algo)
  - spectral clustering on the WL kernel   -> native algorithm for a PSD kernel

For each (metric, algo): best k by silhouette, silhouette vs a product-Bernoulli
null (independent nodes, matched marginals, MEASURED IN THAT METRIC), agreement
(Adjusted Rand Index) vs the node-Jaccard/UPGMA baseline partition, and -- if the
iteration has a prior excised motif -- the survival verdict (does that motif's
core reappear in a current cluster).

Usage: metric_clustering_robustness.py RUN_DIR ITER [--null-reps 30] [--wl-h 2]
Everything runs on CPU.
"""
import sys, os, json, argparse, glob
from collections import Counter
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.scripts.universality_pruning_experiment import (
    NodeSpace, build_weight_maps, circuit_nodes,
)

CORE_FRAC = 0.7
MATCH_J = 0.3


# ---------------------------------------------------------------------------
# representations
# ---------------------------------------------------------------------------
def node_bool(node_sets):
    nodes = sorted(set().union(*node_sets)) if node_sets else []
    idx = {v: j for j, v in enumerate(nodes)}
    X = np.zeros((len(node_sets), len(nodes)), bool)
    for i, s in enumerate(node_sets):
        for v in s:
            X[i, idx[v]] = True
    return X, nodes


def jaccard_D(X):
    inter = X.astype(float) @ X.T.astype(float)
    sz = X.sum(1).astype(float)
    union = sz[:, None] + sz[None, :] - inter
    return 1.0 - inter / np.maximum(union, 1e-12)


def typed_edges(mask, weight_maps, eps=0.0):
    """{(src_key,src_idx,dst_key,dst_idx): |w|} for active, nonzero-weight edges."""
    active = {k: torch.where(m.bool())[0].cpu().tolist() for k, m in mask.items()}
    out = {}
    for sk, dk, W in weight_maps:
        sl, dl = active.get(sk, []), active.get(dk, [])
        if not sl or not dl:
            continue
        sub = W[dl][:, sl].abs()
        nz = (sub > eps).nonzero()
        for a, b in nz.tolist():
            out[(sk, sl[b], dk, dl[a])] = float(sub[a, b])
    return out


def ruzicka_D(edge_dicts):
    keys = sorted(set().union(*[set(e) for e in edge_dicts])) if edge_dicts else []
    idx = {k: j for j, k in enumerate(keys)}
    E = np.zeros((len(edge_dicts), len(keys)))
    for i, e in enumerate(edge_dicts):
        for k, w in e.items():
            E[i, idx[k]] = w
    N = len(edge_dicts); D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            mn = np.minimum(E[i], E[j]).sum(); mx = np.maximum(E[i], E[j]).sum()
            D[i, j] = D[j, i] = 1.0 - (mn / mx if mx > 0 else 0.0)
    return D


def wl_kernel(graphs, h, relabel):
    """graphs: list of (labels{node:int}, adj{node:[(etype,dir,nb)]}). Returns normalized K."""
    n = len(graphs)
    labels = [dict(g[0]) for g in graphs]
    feats = [Counter(l.values()) for l in labels]
    for _ in range(h):
        newl = []
        for i in range(n):
            lab, adj = labels[i], graphs[i][1]
            nl = {}
            for node, l in lab.items():
                sig = (l, tuple(sorted((et, d, lab[nb]) for et, d, nb in adj[node])))
                nl[node] = relabel.setdefault(sig, len(relabel))
            newl.append(nl)
        for i in range(n):
            labels[i] = newl[i]
            feats[i].update(newl[i].values())
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ci, cj = feats[i], feats[j]
            s = sum(ci[k] * cj[k] for k in (set(ci) & set(cj)))
            K[i, j] = K[j, i] = s
    d = np.sqrt(np.diag(K)); d[d == 0] = 1.0
    return K / np.outer(d, d)


def build_graph(mask, weight_maps, relabel, eps=0.0):
    active = {k: torch.where(m.bool())[0].cpu().tolist() for k, m in mask.items()}
    lab = {}
    for k, idxs in active.items():
        for i in idxs:
            lab[(k, i)] = relabel.setdefault(("type", k), len(relabel))   # node label = location type
    adj = {nd: [] for nd in lab}
    for sk, dk, W in weight_maps:
        sl, dl = active.get(sk, []), active.get(dk, [])
        if not sl or not dl:
            continue
        et = relabel.setdefault(("edge", sk, dk), len(relabel))
        nz = (W[dl][:, sl].abs() > eps).nonzero()
        for a, b in nz.tolist():
            s, d = (sk, sl[b]), (dk, dl[a])
            adj[s].append((et, 1, d)); adj[d].append((et, 0, s))
    return lab, adj


# ---------------------------------------------------------------------------
# clustering + silhouette + null
# ---------------------------------------------------------------------------
def silhouette(D, labels):
    n = len(labels); cl = np.unique(labels)
    if len(cl) < 2:
        return -1.0
    v = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]; same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in cl if c != labels[i])
        m = max(a, b); v[i] = (b - a) / m if m > 0 else 0.0
    return float(v.mean())


def upgma(D, kmax):
    if len(D) < 3:
        return 1, np.ones(len(D), int), -1.0
    Z = linkage(squareform(D, checks=False), method="average")
    best = (1, np.ones(len(D), int), -1.0)
    for k in range(2, min(kmax, len(D) - 1) + 1):
        lab = fcluster(Z, k, "maxclust"); s = silhouette(D, lab)
        if s > best[2]:
            best = (k, lab, s)
    return best


def spectral(K, kmax):
    from sklearn.cluster import SpectralClustering
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * K))
    best = (1, np.ones(len(K), int), -1.0)
    for k in range(2, min(kmax, len(K) - 1) + 1):
        lab = SpectralClustering(n_clusters=k, affinity="precomputed",
                                 random_state=0).fit_predict(K)
        s = silhouette(D, lab)
        if s > best[2]:
            best = (k, lab, s)
    return best


def cores(node_sets, labels, frac=CORE_FRAC):
    X, nodes = node_bool(node_sets)
    out = {}
    for c in np.unique(labels):
        sub = X[labels == c]
        out[int(c)] = {nodes[j] for j in np.where(sub.mean(0) >= frac)[0]}
    return out


def jacc(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("iter", type=int)
    ap.add_argument("--null-reps", type=int, default=30)
    ap.add_argument("--wl-h", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args()
    run = a.run_dir.rstrip("/"); it = a.iter
    model_name = json.load(open(run + "/run_args.json"))["model"]
    print(f"loading {model_name} (cpu) ...", flush=True)
    model, _ = load_model(model_name, "cpu")
    ml = PruningConfig().mask_locations
    nspace = NodeSpace(model, ml)
    wmaps = build_weight_maps(model, ml)
    del model

    it_dir = f"{run}/iter{it:02d}"
    cps = sorted(glob.glob(it_dir + "/seed*_circuit.pt"))
    masks = [torch.load(p, map_location="cpu", weights_only=True) for p in cps]
    node_sets = [circuit_nodes(m) for m in masks]
    edge_dicts = [typed_edges(m, wmaps) for m in masks]
    N = len(masks)
    print(f"iter {it}: {N} circuits | mean nodes {np.mean([len(s) for s in node_sets]):.0f} "
          f"| mean edges {np.mean([len(e) for e in edge_dicts]):.0f}", flush=True)

    relabel = {}
    graphs = [build_graph(m, wmaps, relabel) for m in masks]
    Kwl = wl_kernel(graphs, a.wl_h, relabel)

    Dn = jaccard_D(node_bool(node_sets)[0])
    De = ruzicka_D(edge_dicts)
    Dwl = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * Kwl))

    # observed clusterings
    runs = {}
    runs["node-Jaccard / UPGMA"] = (upgma(Dn, a.kmax), Dn)
    runs["wedge-Jaccard / UPGMA"] = (upgma(De, a.kmax), De)
    runs["WL-kernel / UPGMA"] = (upgma(Dwl, a.kmax), Dwl)
    try:
        runs["WL-kernel / spectral"] = (spectral(Kwl, a.kmax), Dwl)
    except Exception as e:
        print(f"  (spectral skipped: {e})", flush=True)

    base_lab = runs["node-Jaccard / UPGMA"][0][1]

    # null: independent-node Bernoulli, matched marginals, measured in each metric
    Xobs, _ = node_bool(node_sets)
    q = Xobs.mean(0); allnodes = sorted(set().union(*node_sets))
    rng = np.random.default_rng(0)
    null = {k: [] for k in runs}
    for _ in range(a.null_reps):
        keep = rng.random((N, len(q))) < q[None, :]
        nsets = [{allnodes[j] for j in np.where(keep[i])[0]} for i in range(N)]
        nsets = [s if s else {allnodes[0]} for s in nsets]
        nmasks = []
        for s in nsets:
            md = {}
            for (k, i) in s:
                md.setdefault(k, torch.zeros(nspace.dim[k], dtype=torch.bool))[i] = True
            nmasks.append(md)
        nDn = jaccard_D(node_bool(nsets)[0])
        nDe = ruzicka_D([typed_edges(m, wmaps) for m in nmasks])
        rl = {}; ngraphs = [build_graph(m, wmaps, rl) for m in nmasks]
        nK = wl_kernel(ngraphs, a.wl_h, rl); nDwl = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * nK))
        null["node-Jaccard / UPGMA"].append(upgma(nDn, a.kmax)[2])
        null["wedge-Jaccard / UPGMA"].append(upgma(nDe, a.kmax)[2])
        null["WL-kernel / UPGMA"].append(upgma(nDwl, a.kmax)[2])
        if "WL-kernel / spectral" in runs:
            try:
                null["WL-kernel / spectral"].append(spectral(nK, a.kmax)[2])
            except Exception:
                null["WL-kernel / spectral"].append(0.0)

    # prior excised motif for survival
    st = json.load(open(run + "/state.json"))
    prior = next((m for m in st.get("motifs", []) if m["excised_iter"] == it - 1), None)
    excl = set()
    if os.path.exists(it_dir + "/excluded_input.json"):
        excl = {tuple(x) for x in json.load(open(it_dir + "/excluded_input.json"))}
    allowed = (set(tuple(x) for x in prior["core"]) - excl) if prior else None

    from sklearn.metrics import adjusted_rand_score
    print(f"\n{'metric / algorithm':24} {'k':>2} {'sil':>6} {'null95':>6} {'real?':>5} "
          f"{'ARI':>5}  survival")
    rows = {}
    for name, ((k, lab, sil), D) in runs.items():
        n95 = float(np.quantile(null[name], 0.95)) if null[name] else float("nan")
        real = sil > n95
        ari = adjusted_rand_score(base_lab, lab)
        surv = "n/a (iter0)" if prior is None else None
        if prior is not None:
            cc = cores(node_sets, lab); rj = max((jacc(c, allowed) for c in cc.values()), default=0.0)
            surv = f"{'REAPPEAR' if rj >= MATCH_J else 'gone':8} J={rj:.2f}"
        szs = dict(Counter(lab.tolist()))
        print(f"{name:24} {k:>2} {sil:6.3f} {n95:6.3f} {str(real):>5} {ari:5.2f}  {surv}  sizes={szs}")
        rows[name] = {"k": int(k), "silhouette": float(sil), "null95": n95, "clustered": bool(real),
                      "ari_vs_node_jaccard": float(ari), "survival": surv,
                      "cluster_sizes": {str(x): int(c) for x, c in szs.items()}}
    out = {"run": run, "iter": it, "model": model_name, "n_circuits": N,
           "prior_motif": prior["id"] if prior else None, "results": rows}
    json.dump(out, open(f"{it_dir}/metric_robustness.json", "w"), indent=2)
    print(f"\nwrote {it_dir}/metric_robustness.json")


if __name__ == "__main__":
    main()
