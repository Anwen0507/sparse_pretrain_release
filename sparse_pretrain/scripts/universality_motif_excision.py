#!/usr/bin/env python3
"""
Motif-excision universality pruning -- a principled successor to
universality_pruning_experiment.py.

WHY
---
The original loop excludes the max-universality node(s) each iteration
(`rank1_nodes`). The max-universality node is the one shared across the MOST
circuits, i.e. the least motif-specific node -- the shared BACKBONE (embedding
reader, unembedding writer, the duplicate/induction head every solution routes
through). So that loop peels the backbone, not motifs, and "one motif per
iteration" is unsupported. The family-clustering result already shows several
motifs COEXIST within every iteration.

WHAT THIS DOES INSTEAD (per iteration)
--------------------------------------
  1. Re-prune K seeds with the cumulative exclusion held off. (Reuses the
     ORIGINAL worker verbatim via subprocess --worker, so circuit discovery,
     split handling, and hparams are byte-for-byte identical -- only the
     SELECTION and STOP rules change.)
  2. Cluster the successful circuits on node-Jaccard (average-linkage, k by
     silhouette) and test the silhouette against a product-Bernoulli null with
     matched per-node marginals. sil > null95 => real co-occurrence structure
     beyond independent inclusion => distinct motifs exist.
  3. TARGET motif = largest clustered family (the dominant / most reproducible
     way the model solves the task).
  4. Exclude the motif's DISCRIMINATIVE CORE: nodes frequent INSIDE the family
     (q_in >= core_frac) AND specific to it (q_in - q_out >= margin). This is
     the opposite of max-universality -- it removes what makes the motif
     distinct, sparing the substrate shared with the other motifs.
  5. Lineage across iterations = the SURVIVAL CHECK: the excised motif should
     not reappear while sibling motifs persist. Recovery rates are the
     "with confidence I removed ONE motif" statement.
  6. STOP when no distinct motif remains: 0 circuits, or N < min_n_cluster, or
     not clustered (single diffuse family), or no discriminative node exists to
     excise without hitting the shared substrate.

Each successful iteration removes >= 1 node, so the loop terminates. Iterations
now count MOTIFS, not backbone depth.
"""

from sparse_pretrain.paths import OUTPUTS, NAME_POOLS
import sys, os, json, argparse, time, subprocess
from pathlib import Path


import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.scripts.universality_pruning_experiment import (
    NodeSpace, build_weight_maps, circuit_nodes, _seed_chunks, get_hparams,
)

# original script supplies the pruning worker; we shell out to its --worker mode
ORIG_SCRIPT = str(Path(__file__).resolve().parent / "universality_pruning_experiment.py")


# ---------------------------------------------------------------------------
# Clustering + null (node = (key, idx) tuple; vendored from family_clustering.py
# and adapted to tuple nodes + an injectable RNG, with discriminative scoring)
# ---------------------------------------------------------------------------
def bool_matrix(node_sets):
    nodes = sorted(set().union(*node_sets)) if node_sets else []
    idx = {v: j for j, v in enumerate(nodes)}
    X = np.zeros((len(node_sets), len(nodes)), dtype=bool)
    for i, s in enumerate(node_sets):
        for v in s:
            X[i, idx[v]] = True
    return X, nodes


def jaccard_from_bool(X):
    inter = X.astype(np.float64) @ X.T.astype(np.float64)
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
    """Average-linkage; choose k in [2, kmax] maximizing silhouette."""
    D = 1.0 - J
    np.fill_diagonal(D, 0.0)
    if len(J) < 3:
        return 1, np.ones(len(J), dtype=int), -1.0
    Z = linkage(squareform(D, checks=False), method="average")
    best = (1, np.ones(len(J), dtype=int), -1.0)
    for k in range(2, min(kmax, len(J) - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        sil = silhouette(D, lab)
        if sil > best[2]:
            best = (k, lab, sil)
    return best


def null_best_sils(rng, q, N, kmax, reps):
    """Best silhouette under product-Bernoulli with the SAME per-node marginals
    (destroys node-node correlations, keeps frequencies + the k-search)."""
    out = []
    for _ in range(reps):
        X = rng.random((N, len(q))) < q[None, :]
        X = X[X.sum(1) > 0]
        if len(X) < 3:
            out.append(0.0)
            continue
        out.append(best_clustering(jaccard_from_bool(X), kmax)[2])
    return np.array(out)


def cluster_cores(X, nodes, labels, frac):
    """Identity core per cluster: nodes in >= frac of the cluster's circuits."""
    cores = {}
    for c in np.unique(labels):
        sub = X[labels == c]
        cores[int(c)] = {nodes[j] for j in np.where(sub.mean(0) >= frac)[0]}
    return cores


def jacc(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Discriminative core (the exclusion target)
# ---------------------------------------------------------------------------
RELAX_LADDER = [(1.0, 1.0), (1.0, 0.5), (1.0, 0.0)]  # (margin_mult, core_frac_mult)


def discriminative_core(X, nodes, labels, target, core_frac, margin):
    """Nodes frequent INSIDE the target family (q_in >= core_frac) and specific
    to it (q_in - q_out >= margin). Relaxes margin then core_frac if empty.
    Returns (selected_nodes, info_dict)."""
    inc = labels == target
    q_in = X[inc].mean(0)
    q_out = X[~inc].mean(0) if (~inc).any() else np.zeros(X.shape[1])
    disc = q_in - q_out
    chosen, level = [], None
    for mm, cf in RELAX_LADDER:
        cur_margin, cur_cf = margin * mm, core_frac * cf
        sel = [j for j in range(len(nodes)) if q_in[j] >= cur_cf and disc[j] >= cur_margin]
        if sel:
            chosen, level = sel, (cur_cf, cur_margin)
            break
    info = {
        "level": level,  # (core_frac, margin) actually used, or None if empty
        "selected": [{"node": f"{k}#{i}", "q_in": float(q_in[j]),
                      "q_out": float(q_out[j]), "disc": float(disc[j])}
                     for j in chosen for (k, i) in [nodes[j]]],
    }
    return [nodes[j] for j in chosen], info


# ---------------------------------------------------------------------------
# One iteration: launch ORIGINAL worker over seeds, collect, cluster, select
# ---------------------------------------------------------------------------
def run_iteration(it, excluded, args, node_space, exp_dir, motifs, rng):
    it_dir = exp_dir / f"iter{it:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nITERATION {it}  | excluded so far: {len(excluded)} nodes "
          f"| seeds: {args.num_seeds} | workers: {args.num_workers}\n{'='*72}", flush=True)

    with open(it_dir / "excluded_input.json", "w") as f:
        json.dump([list(x) for x in sorted(excluded)], f)
    for p in list(it_dir.glob("seed*_result.json")) + list(it_dir.glob("seed*_circuit.pt")):
        p.unlink()

    # --- launch the ORIGINAL script's worker over contiguous seed chunks ---
    t0 = time.time()
    procs = []
    for j, (s, e) in enumerate(_seed_chunks(args.num_seeds, args.num_workers)):
        cmd = [sys.executable, ORIG_SCRIPT, "--worker",
               "--iter", str(it),
               "--seed-start", str(s + args.seed_offset),
               "--seed-end", str(e + args.seed_offset),
               "--model", args.model, "--tokenizer", args.tokenizer, "--task", args.task,
               "--target-loss", str(args.target_loss), "--num-steps", str(args.num_steps),
               "--batch-size", str(args.batch_size), "--eval-batches", str(args.eval_batches),
               "--bisect-iters", str(args.bisect_iters), "--exp-dir", str(exp_dir),
               "--device", args.device,
               "--split-over", args.split_over, "--split-seed", str(args.split_seed),
               "--test-frac", str(args.test_frac),
               "--heldout-fold", str(args.heldout_fold), "--name-pool", str(args.name_pool)]
        wlog = open(it_dir / f"worker{j}.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=wlog, stderr=subprocess.STDOUT), wlog, (s, e)))
    fails = []
    for p, wlog, rng_ in procs:
        rc = p.wait(); wlog.close()
        if rc != 0:
            fails.append((rng_, rc))
    if fails:
        print(f"  WARNING: {len(fails)} worker(s) exited nonzero: {fails}", flush=True)
    elapsed = time.time() - t0

    # --- collect ---
    node_sets, succ_seeds = [], []
    for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
        rp = it_dir / f"seed{seed}_result.json"
        if not rp.exists():
            continue
        r = json.load(open(rp))
        if not r.get("target_achieved"):
            continue
        import torch
        cm = torch.load(it_dir / f"seed{seed}_circuit.pt", map_location="cpu", weights_only=True)
        node_sets.append(circuit_nodes(cm))
        succ_seeds.append(seed)
    n_succ = len(node_sets)
    print(f"\n  -> {n_succ}/{args.num_seeds} circuits reached target loss  "
          f"({elapsed/60:.1f} min, {args.num_workers} workers)", flush=True)

    summary = {"iteration": it, "num_seeds": args.num_seeds, "n_success": n_succ,
               "success_rate": n_succ / args.num_seeds, "elapsed_sec": elapsed,
               "excluded_before": len(excluded), "success_seeds": succ_seeds}

    if n_succ == 0:
        summary["decision"] = {"stop": True, "reason": "no_circuits"}
        with open(it_dir / "motif_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary, set(), None

    # --- cluster + null ---
    X, nodes = bool_matrix(node_sets)
    J = jaccard_from_bool(X)
    if n_succ >= args.min_n_cluster:
        k, lab, sil = best_clustering(J, args.kmax)
        q = X.mean(0)
        nulls = null_best_sils(rng, q, n_succ, args.kmax, args.null_reps)
        null95 = float(np.quantile(nulls, 0.95)) if len(nulls) else float("nan")
        clustered = bool(sil > null95)
    else:
        k, lab, sil = 1, np.ones(n_succ, dtype=int), float("nan")
        null95, clustered = float("nan"), False

    cores = cluster_cores(X, nodes, lab, args.core_frac)
    sizes = {int(c): int((lab == c).sum()) for c in np.unique(lab)}
    off = ~np.eye(len(J), dtype=bool)
    samec = lab[:, None] == lab[None, :]
    withinJ = float(J[samec & off].mean()) if (samec & off).any() else float("nan")
    betwJ = float(J[~samec].mean()) if (~samec).any() else float("nan")

    # --- lineage: match this iter's clusters to prior motifs (SURVIVAL CHECK) ---
    lineage = []
    for c in sorted(cores):
        best_fid, best_j = None, 0.0
        for m in motifs:
            allowed = m["core"] - excluded  # restrict prior core to still-allowed nodes
            jv = jacc(cores[c], allowed)
            if jv > best_j:
                best_fid, best_j = m["id"], jv
        lineage.append({"cluster": c, "size": sizes[c],
                        "matched_motif": best_fid if best_j >= args.match_j else None,
                        "match_jaccard": round(best_j, 3)})
    # did the motif excised LAST iteration reappear?
    survival = None
    last_excised = next((m for m in motifs if m.get("excised_iter") == it - 1), None)
    if last_excised is not None:
        allowed = last_excised["core"] - excluded
        reappear_j = max((jacc(cores[c], allowed) for c in cores), default=0.0)
        survival = {"excised_motif": last_excised["id"],
                    "reappeared": bool(reappear_j >= args.match_j),
                    "max_core_jaccard_to_current": round(reappear_j, 3),
                    "n_sibling_motifs_present": sum(1 for L in lineage if L["matched_motif"])}

    summary.update({
        "k": int(k), "silhouette": float(sil), "null95": null95, "clustered": clustered,
        "within_jaccard": withinJ, "between_jaccard": betwJ, "cluster_sizes": sizes,
        "cores": {str(c): [f"{kk}#{ii}" for (kk, ii) in sorted(v)] for c, v in cores.items()},
        "lineage": lineage, "survival": survival,
    })
    print(f"  cluster: k={k} sil={sil:.3f} null95={null95:.3f} "
          f"-> {'CLUSTERED' if clustered else 'single/diffuse'}  sizes={sizes}", flush=True)
    print(f"  node Jaccard within={withinJ:.3f} between={betwJ:.3f}", flush=True)
    if survival is not None:
        s = survival
        print(f"  SURVIVAL: motif {s['excised_motif']} excised@{it-1} "
              f"reappeared={s['reappeared']} (J={s['max_core_jaccard_to_current']}); "
              f"{s['n_sibling_motifs_present']} sibling motif(s) still present", flush=True)

    # --- STOP if no distinct motif remains ---
    if not clustered or k < 2:
        reason = "too_few_circuits" if n_succ < args.min_n_cluster else "not_clustered"
        summary["decision"] = {"stop": True, "reason": reason}
        with open(it_dir / "motif_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary, set(), None

    # --- TARGET = largest clustered family; exclude its DISCRIMINATIVE core ---
    target = max(sizes, key=lambda c: sizes[c])
    excise, dinfo = discriminative_core(X, nodes, lab, target, args.core_frac, args.disc_margin)
    if not excise:
        summary["decision"] = {"stop": True, "reason": "no_discriminative_node", "target": target}
        with open(it_dir / "motif_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("  no discriminative node to excise without hitting the shared substrate -> STOP",
              flush=True)
        return summary, set(), None

    new_id = f"M{len(motifs)}"
    motif = {"id": new_id, "core": cores[target], "excised_iter": it,
             "excised_nodes": set(excise), "target_cluster": target, "size": sizes[target]}
    summary["decision"] = {
        "stop": False, "target_cluster": target, "target_size": sizes[target],
        "motif_id": new_id, "excise_level": dinfo["level"],
        "n_excise": len(excise),
        "excised_nodes": [f"{k}#{i}" for (k, i) in sorted(excise)],
        "excised_detail": dinfo["selected"],
    }
    with open(it_dir / "motif_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  TARGET motif {new_id} = cluster {target} (size {sizes[target]}); "
          f"excise {len(excise)} discriminative node(s) "
          f"@ (core_frac,margin)={dinfo['level']}:", flush=True)
    for d in dinfo["selected"][:10]:
        print(f"      {d['node']:>28}  q_in={d['q_in']:.2f} q_out={d['q_out']:.2f} disc={d['disc']:+.2f}",
              flush=True)
    return summary, set(excise), motif


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_state(exp_dir):
    p = exp_dir / "state.json"
    if p.exists():
        return json.load(open(p))
    return {"next_iter": 0, "excluded": [], "history": [], "motifs": [], "exhausted": False}


def save_state(exp_dir, state):
    with open(exp_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2, default=str)


def motifs_from_state(state):
    out = []
    for m in state.get("motifs", []):
        out.append({"id": m["id"], "core": set(tuple(x) for x in m["core"]),
                    "excised_iter": m["excised_iter"],
                    "excised_nodes": set(tuple(x) for x in m["excised_nodes"]),
                    "target_cluster": m["target_cluster"], "size": m["size"]})
    return out


def motifs_to_state(motifs):
    return [{"id": m["id"], "core": [list(x) for x in sorted(m["core"])],
             "excised_iter": m["excised_iter"],
             "excised_nodes": [list(x) for x in sorted(m["excised_nodes"])],
             "target_cluster": m["target_cluster"], "size": m["size"]} for m in motifs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # --- identical to universality_pruning_experiment.py (passed through to worker) ---
    ap.add_argument("--model", default="jacobcd52/ss_d128_f1")
    ap.add_argument("--task", default="dummy_pronoun")
    ap.add_argument("--tokenizer", default="SimpleStories/SimpleStories-1.25M")
    ap.add_argument("--target-loss", type=float, default=0.15, dest="target_loss")
    ap.add_argument("--num-seeds", type=int, default=100, dest="num_seeds")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset")
    ap.add_argument("--num-steps", type=int, default=2000, dest="num_steps")
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    ap.add_argument("--eval-batches", type=int, default=5, dest="eval_batches")
    ap.add_argument("--bisect-iters", type=int, default=15, dest="bisect_iters")
    ap.add_argument("--skip-carbs", action="store_true", dest="skip_carbs")
    ap.add_argument("--carbs-runs", type=int, default=32, dest="carbs_runs")
    ap.add_argument("--num-workers", type=int, default=10, dest="num_workers")
    ap.add_argument("--max-iters", type=int, default=1000, dest="max_iters")
    ap.add_argument("--exp-dir",
                    default=str(OUTPUTS / "ss_d128_f1_pronoun_motif"), dest="exp_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-over", default="none",
                    choices=["none", "examples", "names", "templates", "names_templates"],
                    dest="split_over")
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    ap.add_argument("--test-frac", type=float, default=0.2, dest="test_frac")
    ap.add_argument("--heldout-fold", type=int, default=0, dest="heldout_fold")
    ap.add_argument("--name-pool", dest="name_pool",
                    default=str(NAME_POOLS / "name_pool_cast15.json"))
    # --- motif-excision knobs ---
    ap.add_argument("--core-frac", type=float, default=0.7, dest="core_frac",
                    help="A node is in a family's core if present in >= this fraction of the "
                         "family's circuits.")
    ap.add_argument("--disc-margin", type=float, default=0.3, dest="disc_margin",
                    help="A node is motif-DISCRIMINATIVE if q_in - q_out >= this (relaxed if the "
                         "target family has no such node).")
    ap.add_argument("--kmax", type=int, default=8, dest="kmax")
    ap.add_argument("--null-reps", type=int, default=200, dest="null_reps",
                    help="Product-Bernoulli null draws for the silhouette significance test.")
    ap.add_argument("--null-seed", type=int, default=0, dest="null_seed")
    ap.add_argument("--min-n-cluster", type=int, default=5, dest="min_n_cluster")
    ap.add_argument("--match-j", type=float, default=0.3, dest="match_j",
                    help="Core-Jaccard threshold for matching a cluster to a prior motif (lineage).")
    args = ap.parse_args()
    args.name_pool = str(Path(args.name_pool).resolve())

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # identity guard (same spirit as the original): refuse to resume across a different
    # task/split identity, which would corrupt the cumulative exclusion.
    prev_path = exp_dir / "run_args.json"
    if prev_path.exists():
        prev = json.load(open(prev_path))
        for k in ("model", "task", "split_over", "split_seed", "test_frac",
                  "heldout_fold", "name_pool", "seed_offset"):
            if k in prev and prev[k] != getattr(args, k):
                raise SystemExit(f"exp-dir {exp_dir} was created with {k}={prev[k]!r} but this run "
                                 f"uses {getattr(args, k)!r}; use a fresh --exp-dir.")
    with open(prev_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # names_templates: snapshot the exact split for the worker (identical to original main)
    if args.split_over == "names_templates":
        from transformers import AutoTokenizer
        from sparse_pretrain.scripts.pronoun_split import make_pronoun_fold_split
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _, _, _, split_info = make_pronoun_fold_split(
            tok, heldout_fold=args.heldout_fold, pool_path=args.name_pool)
        si_path = exp_dir / "split_info.json"
        if si_path.exists():
            prev_si = json.load(open(si_path))
            if (prev_si["train_names"], prev_si["test_names"]) != (
                    split_info["train_names"], split_info["test_names"]):
                raise SystemExit(f"{si_path} disagrees with the current pool {args.name_pool}; "
                                 f"use a fresh --exp-dir.")
            split_info = prev_si
        else:
            json.dump(split_info, open(si_path, "w"), indent=2)
        print(f"names_templates split: {split_info['n_train_names']} train names; "
              f"held-out fold {args.heldout_fold}: {split_info['n_test_names']} names "
              f"({split_info['n_test_pairs']} test pairs)", flush=True)

    print(f"Loading model {args.model} for node space ...", flush=True)
    import torch
    model, _ = load_model(args.model, args.device)
    mask_locations = PruningConfig().mask_locations
    node_space = NodeSpace(model, mask_locations)
    del model; torch.cuda.empty_cache()
    print(f"Total maskable nodes: {node_space.total}", flush=True)

    get_hparams(args, exp_dir)  # ensure exp_dir/hparams.json for the worker
    state = load_state(exp_dir)
    excluded = set(tuple(x) for x in state["excluded"])
    motifs = motifs_from_state(state)
    rng = np.random.default_rng(args.null_seed)

    if state.get("exhausted"):
        print("Experiment already marked exhausted. Nothing to do.", flush=True)
        return

    done = 0
    while done < args.max_iters:
        it = state["next_iter"]
        summary, new_excl, motif = run_iteration(it, excluded, args, node_space, exp_dir, motifs, rng)
        state["history"].append({k: v for k, v in summary.items()
                                 if k not in ("success_seeds",)})

        if summary.get("decision", {}).get("stop"):
            reason = summary["decision"]["reason"]
            print(f"\n*** STOP at iteration {it}: {reason}. "
                  f"{len(motifs)} motif(s) excised over {it} iteration(s). ***", flush=True)
            state["exhausted"] = True
            save_state(exp_dir, state)
            break

        excluded |= new_excl
        if motif is not None:
            motifs.append(motif)
        state["excluded"] = [list(x) for x in sorted(excluded)]
        state["motifs"] = motifs_to_state(motifs)
        state["next_iter"] = it + 1
        save_state(exp_dir, state)
        done += 1

        if len(excluded) >= node_space.total:
            print("\n*** All nodes excluded. Stopping. ***", flush=True)
            state["exhausted"] = True
            save_state(exp_dir, state)
            break

    print(f"\nDone. Ran {done} iteration(s) this invocation. next_iter={state['next_iter']}, "
          f"motifs excised={len(motifs)}, cumulative excluded={len(excluded)}, "
          f"exhausted={state.get('exhausted', False)}", flush=True)
    print(f"State + per-iteration motif summaries in: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
