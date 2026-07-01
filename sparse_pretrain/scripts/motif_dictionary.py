#!/usr/bin/env python3
"""
Dictionary-learning decomposition of pruned-circuit ensembles -- an overlapping
successor to the HARD clustering in universality_motif_excision.py.

WHY
---
The motif-excision loop assigns each circuit to ONE family (average-linkage on
node-Jaccard, k by silhouette, significance vs a product-Bernoulli null). On
these ensembles that collapses to one giant blob + tiny outliers
(e.g. iter00 cast15: cluster_sizes {1:97, 2:3}), because a circuit is really
backbone + motif-A + motif-B SIMULTANEOUSLY -- the "patchwork" finding. Hard
clustering cannot represent that overlap.

WHAT THIS DOES INSTEAD
----------------------
Treat the per-iteration circuit x node incidence matrix X (n_circuits x N_nodes,
binary -- exactly what bool_matrix() builds) as a dataset of sparse codes and
factor it with NMF:

        X  ~=  W (n_circuits x K)  .  H (K x N_nodes)

  * H rows = MOTIF ATOMS (soft node-distributions; the soft version of `cores`).
  * W rows = per-circuit LOADINGS -- one circuit can load several atoms (overlap).

Two structures are separated explicitly:

  1. FREQUENCY BACKBONE -- nodes with high marginal frequency (present in most
     circuits). This is the shared substrate; the marginal/independent model
     already captures it, which is why rank-1 NMF ~= the null.
  2. CO-OCCURRENCE MOTIFS -- structure BEYOND marginals: groups of nodes that
     appear together more than their independent rates predict.

K is chosen by HELD-OUT reconstruction over circuits, scored against a matched
COLUMN-SHUFFLED null (each node permuted across circuits independently -- the
held-out, NMF-pipeline version of the product-Bernoulli null: exact marginals,
co-occurrence destroyed). real << null at K  =>  genuine motif structure at that
rank. Where the real curve stops separating from the null is the effective
universality rank; if it never saturates, the ensemble is diffuse/high-rank
(many weak motifs), which is itself the finding.

Reads circuits already on disk; does NOT load the model or re-prune.

Usage:
    python -m sparse_pretrain.scripts.motif_dictionary --exp-dir outputs/universality_pruning/cast15_fold0_motif --kmax 16
    python -m sparse_pretrain.scripts.motif_dictionary --exp-dir outputs/universality_pruning/sparse_d1024_motif --kmax 16
"""

import os, sys, json, glob, argparse, warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Data: circuit x node matrix over a FIXED global node space
# ---------------------------------------------------------------------------
def circuit_node_set(cm):
    """Set of (key, idx) active nodes -- model-free copy of
    universality_pruning_experiment.circuit_nodes()."""
    nodes = set()
    for key, m in cm.items():
        for idx in torch.where(m.bool())[0].cpu().tolist():
            nodes.add((key, idx))
    return nodes


def global_node_space(exp_dir):
    """Ordered [(key, idx), ...] over the FULL (un-excluded) node space, from the
    dict structure of any circuit .pt. One fixed column order makes atoms
    comparable across iterations (excluded nodes are always-zero columns)."""
    any_pt = next(iter(sorted(glob.glob(str(exp_dir / "iter*" / "seed*_circuit.pt")))), None)
    if any_pt is None:
        raise SystemExit(f"no seed*_circuit.pt under {exp_dir}")
    cm = torch.load(any_pt, map_location="cpu", weights_only=True)
    nodes = [(key, idx) for key in cm.keys() for idx in range(cm[key].shape[0])]
    return nodes, {nd: j for j, nd in enumerate(nodes)}


def load_iteration(it_dir, index, N):
    """X (n_success, N) float32 binary, plus the contributing seeds."""
    rows, seeds = [], []
    for rp in sorted(glob.glob(str(it_dir / "seed*_result.json"))):
        if not json.load(open(rp)).get("target_achieved"):
            continue
        cp = rp.replace("_result.json", "_circuit.pt")
        if not os.path.exists(cp):
            continue
        cm = torch.load(cp, map_location="cpu", weights_only=True)
        v = np.zeros(N, dtype=np.float32)
        for nd in circuit_node_set(cm):
            v[index[nd]] = 1.0
        rows.append(v)
        seeds.append(int(Path(rp).stem.replace("seed", "").replace("_result", "")))
    X = np.stack(rows) if rows else np.zeros((0, N), dtype=np.float32)
    return X, seeds


# ---------------------------------------------------------------------------
# Frequency backbone (the shared substrate; captured by marginals)
# ---------------------------------------------------------------------------
def frequency_backbone(freq, anodes, topn=20):
    out = {f"n_freq_ge_{int(t*100)}": int((freq >= t).sum()) for t in (0.9, 0.7, 0.5, 0.3)}
    order = np.argsort(freq)[::-1]
    out["top_nodes"] = [{"node": f"{anodes[j][0]}#{anodes[j][1]}", "freq": round(float(freq[j]), 3)}
                        for j in order[:topn] if freq[j] >= 0.5]
    return out


# ---------------------------------------------------------------------------
# Held-out reconstruction: REAL vs matched COLUMN-SHUFFLED null
# ---------------------------------------------------------------------------
def relerr(Xte, Xhat):
    return float(np.linalg.norm(Xte - Xhat) / max(np.linalg.norm(Xte), 1e-12))


def _nmf_heldout(Xtr, Xte, k, seed):
    nmf = NMF(n_components=k, init="nndsvda", solver="cd", max_iter=800,
              tol=1e-4, random_state=seed)
    nmf.fit(Xtr)
    return relerr(Xte, nmf.transform(Xte) @ nmf.components_)


def heldout_curves(Xa, ks, reps, heldout_frac, base_seed):
    """For each K: mean+/-SE held-out relerr on REAL circuits and on a matched
    column-shuffled null (independent per-node permutation = product-Bernoulli
    with exact marginals). Same train/test split applied to both."""
    n, M = Xa.shape
    n_te = max(1, int(round(n * heldout_frac)))
    real = {k: [] for k in ks}
    null = {k: [] for k in ks}
    for r in range(reps):
        rng = np.random.default_rng(base_seed + r)
        perm = rng.permutation(n)
        te, tr = perm[:n_te], perm[n_te:]
        Xsh = Xa.copy()
        for j in range(M):                       # destroy co-occurrence, keep marginals
            Xsh[:, j] = Xsh[rng.permutation(n), j]
        for k in ks:
            if k > len(tr) - 1:
                real[k].append(np.nan); null[k].append(np.nan); continue
            real[k].append(_nmf_heldout(Xa[tr], Xa[te], k, base_seed + r))
            null[k].append(_nmf_heldout(Xsh[tr], Xsh[te], k, base_seed + r))

    def stat(a):
        a = np.array([x for x in a if not np.isnan(x)], dtype=float)
        if len(a) == 0:
            return {"mean": float("nan"), "se": float("nan"), "n": 0}
        return {"mean": float(a.mean()),
                "se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                "n": int(len(a))}

    return {k: stat(real[k]) for k in ks}, {k: stat(null[k]) for k in ks}


def select_k(real, null, ks):
    """k_best        = argmin held-out real relerr.
    k_parsimonious  = smallest k within 1 SE of k_best.
    k_cooccur       = smallest k whose real curve sits >2 SE below the null
                      (first rank with significant co-occurrence structure).
    saturated       = did the best k come in below kmax (real curve flattened)?"""
    valid = [k for k in ks if not np.isnan(real[k]["mean"])]
    if not valid:
        return {"k_best": None, "k_parsimonious": None, "k_cooccur": None, "saturated": None}
    k_best = min(valid, key=lambda k: real[k]["mean"])
    thr = real[k_best]["mean"] + real[k_best]["se"]
    k_par = min(k for k in valid if real[k]["mean"] <= thr)
    k_co = None
    for k in valid:
        gap = null[k]["mean"] - real[k]["mean"]
        se = (real[k]["se"] ** 2 + null[k]["se"] ** 2) ** 0.5
        if gap > 2 * se:
            k_co = k; break
    return {"k_best": k_best, "k_parsimonious": k_par, "k_cooccur": k_co,
            "saturated": bool(k_best < max(valid))}


# ---------------------------------------------------------------------------
# Atom reporting (full-data fit at the chosen K)
# ---------------------------------------------------------------------------
def describe_atoms(Xa, k, anodes, freq, seed, topn):
    """Fit NMF on all circuits at rank k; describe each atom by top nodes and its
    loading profile. Loadings are row-normalised to compositions (each circuit =
    distribution over atoms). An atom is 'backbone-aligned' if its top nodes are
    mostly high-frequency substrate, else a 'specific' co-occurrence motif."""
    nmf = NMF(n_components=k, init="nndsvda", solver="cd", max_iter=800, tol=1e-4,
              random_state=seed)
    W = nmf.fit_transform(Xa)
    H = nmf.components_
    comp = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    atoms = []
    for a in range(k):
        hr = H[a]; order = np.argsort(hr)[::-1]
        top_j = [j for j in order[:topn] if hr[j] > 0]
        top = [{"node": f"{anodes[j][0]}#{anodes[j][1]}",
                "weight": round(float(hr[j] / max(hr.max(), 1e-12)), 3),
                "freq": round(float(freq[j]), 3)} for j in top_j]
        share = comp[:, a]
        top_freq = float(np.mean([freq[j] for j in top_j])) if top_j else 0.0
        atoms.append({
            "atom": a,
            "mean_share": round(float(share.mean()), 3),
            "support_frac": round(float((share >= 0.15).mean()), 3),
            "share_cv": round(float(share.std() / share.mean()) if share.mean() > 0 else float("nan"), 3),
            "top_node_mean_freq": round(top_freq, 3),
            "role": "backbone-aligned" if top_freq >= 0.6 else "specific-motif",
            "top_nodes": top,
        })
    atoms.sort(key=lambda d: d["mean_share"], reverse=True)
    return atoms, relerr(Xa, W @ H), H


# ---------------------------------------------------------------------------
# Compare to the existing HARD clustering (motif_summary.json)
# ---------------------------------------------------------------------------
def vs_hard(it_dir, atoms, H, anodes, freq, backbone_thr=0.6):
    """Compare the soft dictionary to the hard clustering's DOMINANT (largest) cluster
    -- the most universal circuit type, the only size-robust object (minority/singleton
    clusters let any atom fit them trivially, so 'most-discriminative cluster' is NOT a
    valid target). Metric of record: cosine on DISCRIMINATIVE nodes (freq<backbone_thr)
    between the best atom's weight vector and the dominant cluster's consensus-core
    indicator, both backbone-masked. dominant_core_disc_frac says whether that universal
    circuit is a real motif (>0) or just backbone (~0); when ~0 there is nothing
    discriminative to recover and cos_disc is None. Full-space cosine and top-15 Jaccard
    are kept ONLY as backbone-confounded sanity numbers; do NOT use them as metrics."""
    p = it_dir / "motif_summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    out = {"hard_k": s.get("k"), "hard_clustered": s.get("clustered"),
           "hard_cluster_sizes": s.get("cluster_sizes"), "hard_silhouette": s.get("silhouette")}
    cores = s.get("cores") or {}
    sizes = {c: int(n) for c, n in (s.get("cluster_sizes") or {}).items()}
    if not (cores and atoms):
        return out

    col = {f"{k}#{i}": c for c, (k, i) in enumerate(anodes)}
    disc = freq < backbone_thr                              # non-backbone (discriminative) mask
    dom = max(cores, key=lambda c: sizes.get(c, len(cores[c])))    # largest = most universal
    core_cols = np.array([col[s_] for s_ in cores[dom] if s_ in col], dtype=int)
    core = np.zeros(len(anodes)); core[core_cols] = 1.0
    core_d = core * disc
    out.update({"dominant_hard_cluster": dom, "dominant_cluster_size": sizes.get(dom),
                "dominant_core_size": len(cores[dom]), "dominant_core_active": int(core_cols.size),
                "dominant_core_disc": int(core_d.sum()),
                "dominant_core_disc_frac": round(float(core_d.sum()) / max(int(core_cols.size), 1), 3)})

    def cos(u, v):
        nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else None

    def best(target, disc_only):
        if target.sum() == 0:
            return None, None
        ba, bv = None, -1.0
        for at in atoms:
            v = cos(H[at["atom"]] * (disc if disc_only else 1.0), target)
            if v is not None and v > bv:
                ba, bv = at["atom"], v
        return ba, (None if bv < 0 else round(bv, 3))

    da, dcd = best(core_d, True)     # metric of record: discriminative recovery of the universal motif
    fa, fcf = best(core, False)      # backbone-confounded sanity
    dnodes = set(cores[dom])
    jbest = max((len(dnodes & (an := set(d["node"] for d in at["top_nodes"][:15])))
                 / max(len(dnodes | an), 1) for at in atoms), default=0.0)
    out.update({"best_matching_atom": da, "best_match_cosine_disc": dcd,
                "full_cosine_atom": fa, "full_cosine_sanity": fcf,
                "jaccard_top15_sanity": round(jbest, 3)})
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(report, out_png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skipping plot: {e})"); return
    its = [r for r in report["iterations"] if r.get("k_parsimonious")]
    if not its:
        print("  (no decomposed iterations; skipping plot)"); return
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.6))
    xs = [r["iteration"] for r in its]

    ax0.plot(xs, [r["k_parsimonious"] for r in its], "o-", label="dict rank K (parsimonious)")
    ax0.plot(xs, [r["k_cooccur"] or 0 for r in its], "v-", alpha=0.6, label="K_cooccur (first sig.)")
    hard = [(r["iteration"], r["vs_hard"]["hard_k"]) for r in its
            if r.get("vs_hard") and r["vs_hard"].get("hard_k")]
    if hard:
        ax0.plot([h[0] for h in hard], [h[1] for h in hard], "^:", alpha=0.6, label="hard cluster k")
    ax0.set_xlabel("iteration"); ax0.set_ylabel("# motifs"); ax0.set_title("Universality rank")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

    ax0b = ax0.twinx()
    ax0b.plot(xs, [r["frequency_backbone"]["n_freq_ge_70"] for r in its], "s--",
              color="gray", alpha=0.5)
    ax0b.set_ylabel("backbone size (freq>=0.7)", color="gray")

    cmap = plt.get_cmap("viridis")
    for r in its:
        c = cmap(r["iteration"] / max(1, max(xs)))
        ks = sorted(int(k) for k in r["relerr_real"])
        gain = [r["relerr_null"][str(k)]["mean"] - r["relerr_real"][str(k)]["mean"] for k in ks]
        ax1.plot(ks, gain, "-", color=c, alpha=0.7)
    ax1.set_xlabel("K (atoms)"); ax1.set_ylabel("held-out gain over null (null - real)")
    ax1.set_title("Co-occurrence signal vs K\n(rising/non-saturating => diffuse high-rank)")
    ax1.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(xs)))
    fig.colorbar(sm, ax=ax1, label="iteration")

    r0 = its[0]
    ks = sorted(int(k) for k in r0["relerr_real"])
    ax2.plot(ks, [r0["relerr_real"][str(k)]["mean"] for k in ks], "o-", label="real")
    ax2.plot(ks, [r0["relerr_null"][str(k)]["mean"] for k in ks], "x--", label="shuffled null")
    ax2.set_xlabel("K (atoms)"); ax2.set_ylabel("held-out relerr")
    ax2.set_title(f"iter{r0['iteration']:02d}: real vs null"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--reps", type=int, default=6, help="held-out splits per K")
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--min-n", type=int, default=10, help="skip iters with fewer successful circuits")
    ap.add_argument("--topn", type=int, default=12, help="top nodes reported per atom")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", default="all", help="'all' or comma list e.g. 0,1,2")
    ap.add_argument("--out", default="motif_dictionary.json")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    nodes, index = global_node_space(exp_dir)
    N = len(nodes)
    ks_all = list(range(1, args.kmax + 1))
    print(f"exp-dir: {exp_dir}\nglobal node space: N={N}; K in 1..{args.kmax}; "
          f"reps={args.reps}", flush=True)

    it_dirs = sorted(glob.glob(str(exp_dir / "iter*")))
    if args.iters != "all":
        want = {int(x) for x in args.iters.split(",")}
        it_dirs = [d for d in it_dirs if int(Path(d).name.replace("iter", "")) in want]

    report = {"exp_dir": str(exp_dir), "N_nodes": N, "kmax": args.kmax, "reps": args.reps,
              "heldout_frac": args.heldout_frac, "iterations": []}

    for d in it_dirs:
        it = int(Path(d).name.replace("iter", ""))
        X, seeds = load_iteration(Path(d), index, N)
        n = len(X)
        amask = X.sum(0) > 0
        anodes = [nodes[j] for j in np.where(amask)[0]]
        Xa = X[:, amask]                          # active columns only (fast; same relerr)
        freq = Xa.mean(0) if n else np.zeros(0)
        active = int(amask.sum())
        print(f"\niter{it:02d}: {n} circuits, {active} active nodes "
              f"(circuit size mean {Xa.sum(1).mean():.0f})" if n else
              f"\niter{it:02d}: {n} circuits", flush=True)
        if n < args.min_n:
            print(f"  < min-n ({args.min_n}); skipping decomposition", flush=True)
            report["iterations"].append({"iteration": it, "n_success": n,
                                         "n_active_nodes": active, "skipped": True})
            continue

        bb = frequency_backbone(freq, anodes)
        n_tr = n - max(1, int(round(n * args.heldout_frac)))
        ks = [k for k in ks_all if k <= min(n_tr - 1, active)]
        real, null = heldout_curves(Xa, ks, args.reps, args.heldout_frac, args.seed)
        sel = select_k(real, null, ks)
        atoms, recon, H = describe_atoms(Xa, sel["k_parsimonious"], anodes, freq, args.seed, args.topn)
        hard = vs_hard(Path(d), atoms, H, anodes, freq)

        rec = {"iteration": it, "n_success": n, "n_active_nodes": active,
               **sel, "frequency_backbone": bb,
               "relerr_real": {str(k): real[k] for k in ks},
               "relerr_null": {str(k): null[k] for k in ks},
               "fullfit_relerr_at_k_parsimonious": round(recon, 4),
               "atoms": atoms, "vs_hard": hard}
        report["iterations"].append(rec)

        kb = sel["k_best"]
        print(f"  backbone: {bb['n_freq_ge_90']} nodes @freq>=.9, {bb['n_freq_ge_70']} @>=.7, "
              f"{bb['n_freq_ge_50']} @>=.5  (of {active} active)", flush=True)
        print(f"  K: best={kb} parsimonious={sel['k_parsimonious']} cooccur={sel['k_cooccur']} "
              f"saturated={sel['saturated']} (hard k={hard.get('hard_k') if hard else '?'})", flush=True)
        print(f"  held-out relerr @K={kb}: real={real[kb]['mean']:.3f} null={null[kb]['mean']:.3f} "
              f"gain={null[kb]['mean']-real[kb]['mean']:.3f}", flush=True)
        nspec = sum(1 for a in atoms if a["role"] == "specific-motif")
        print(f"  atoms@K={sel['k_parsimonious']}: "
              f"{len(atoms)-nspec} backbone-aligned + {nspec} specific motif(s)", flush=True)
        for a in atoms[:sel["k_parsimonious"]]:
            tn = ", ".join(d["node"] for d in a["top_nodes"][:4])
            print(f"    atom{a['atom']} [{a['role']:16}] share={a['mean_share']:.2f} "
                  f"support={a['support_frac']:.2f} topfreq={a['top_node_mean_freq']:.2f}  top: {tn}",
                  flush=True)
        if hard and hard.get("dominant_hard_cluster") is not None:
            print(f"  hard dom cluster {hard['dominant_hard_cluster']} (n={hard['dominant_cluster_size']}): "
                  f"core {hard['dominant_core_size']} -> {hard['dominant_core_disc']} discrim "
                  f"(frac={hard['dominant_core_disc_frac']:.2f})  best atom{hard['best_matching_atom']} "
                  f"cos_disc={hard['best_match_cosine_disc']} (J_sanity={hard['jaccard_top15_sanity']})",
                  flush=True)

    out_path = exp_dir / args.out
    json.dump(report, open(out_path, "w"), indent=2, default=str)
    print(f"\nwrote {out_path}", flush=True)
    make_plot(report, exp_dir / args.out.replace(".json", ".png"))


if __name__ == "__main__":
    main()
