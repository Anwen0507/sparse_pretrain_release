#!/usr/bin/env python3
"""
Functional epistasis / atomicity test -- RE-FACTORING-FREE.

WHY (vs scripts/atomicity_test.py)
----------------------------------
The first atomicity test asked "are the NMF atoms atomic?" by knocking out one
atom, re-pruning, RE-FACTORING the survivors, and Hungarian-matching them back.
That route conflates two things:
  (i)  factorization-coupling artifact -- when the knocked-out atom A is replaced
       by a re-routed A', a JOINT re-factor of the new ensemble can re-distribute
       node loadings across the survivors even if they are functionally untouched
       (NMF is rotation-ambiguous; the basis moved), and
  (ii) genuine functional reorganization of the survivors.
The recurrence drop (0.53 vs 0.84 ceiling) therefore over-states interaction,
because the bootstrap ceiling removes NOTHING so it omits the coupling that ANY
knockout induces. (The cleaner read there was atom@4 0.53 vs matched-random@4
0.63 -- a much smaller, partially-controlled gap.)

This test never re-factors anything. It measures atomicity with PURE INTERVENTION
ARITHMETIC on the fixed model, so it cannot suffer artifact (i).

DEFINITION
----------
Atoms A,B are functionally independent (= atomic / modular) iff the damage of
knocking them out together is the sum of knocking out each:

    cost(S)      = mean over the SOURCE circuits of the held-out task LOSS increase
                   when node-set S is tau-pinned OFF (the immediate oracle from
                   atom_excision.py -- deterministic, no re-prune, no factoring).
    epsilon(A,B) = cost(A u B) - cost(A) - cost(B)

    epsilon ~ 0   -> additive  -> A,B independent       -> ATOMIC
    epsilon < 0   -> sub-additive -> redundant / overlapping roles
    epsilon > 0   -> super-additive -> synergistic; the real unit is the PAIR

Loss (not 2AFC) is the additivity readout -- 2AFC saturates and would fake
sub-additivity. Even loss is mildly curved, so epsilon is calibrated against a
MATCHED-RANDOM NULL: epsilon over random disjoint non-backbone node-sets of the
same size distribution. z = (epsilon - mu0) / sd0. |z| >= z_thr ==> the pair
interacts BEYOND generic curvature.

ATOM UNITS (NMF-only; no hard clusters, no Jaccard)
---------------------------------------------------
Each atom's unit = the non-backbone (freq<thr) nodes it OWNS by argmax loading
with ownership-share >= min_share. Disjoint by construction (a node has one
argmax owner), so a shared node can never fake an interaction. Backbone atoms
own ~no non-backbone nodes and drop out automatically -> the test is about the
discriminative motifs, derived purely from H and freq.

OUTPUT / "FINDING TRULY ATOMIC ATOMS"
-------------------------------------
Build the interaction graph on the testable atoms (edge where |z| >= z_thr). Its
CONNECTED COMPONENTS are the truly atomic units: a multi-atom component is a set
of NMF atoms that are not independent and should be merged into one unit; a
singleton is a genuinely atomic motif. If almost no edges survive, the NMF atoms
are already functionally atomic at this granularity (and the earlier re-factoring
non-atomicity was the coupling artifact, not real interaction).
"""

import sys, json, argparse, time, itertools, shutil
from pathlib import Path

import numpy as np

from sparse_pretrain.scripts.atom_excision import build_dictionary, cols_to_nodes, cos
from sparse_pretrain.scripts.universality_pruning_experiment import circuit_nodes

PASS_THROUGH = ["model", "task", "tokenizer", "target_loss", "num_steps", "batch_size",
                "eval_batches", "bisect_iters", "split_over", "split_seed", "test_frac",
                "heldout_fold", "name_pool"]


# ---------------------------------------------------------------------------
# Atom units (disjoint, NMF-only): each non-backbone node -> its argmax-loading
# atom, kept if that atom owns >= min_share of the node's total loading.
# ---------------------------------------------------------------------------
def atom_owned_cols(D, backbone_thr, min_share, min_size):
    H, freq = D["H"], D["freq"]
    K, n = H.shape
    disc = freq < backbone_thr
    colsum = H.sum(0)
    owned = {a: [] for a in range(K)}
    detail = {}
    for j in range(n):
        if not disc[j] or colsum[j] <= 0:
            continue
        a = int(np.argmax(H[:, j]))
        share = float(H[a, j] / colsum[j])
        if share >= min_share:
            owned[a].append(j)
            detail[j] = (a, share)
    for a in owned:
        owned[a].sort(key=lambda j: H[a, j], reverse=True)
    testable = {a: cs for a, cs in owned.items() if len(cs) >= min_size}
    return testable, detail


def node_label(D, j):
    key, idx = D["anodes"][j]
    return f"{key}#{idx}"


# ---------------------------------------------------------------------------
# Immediate cost oracle (verbatim conventions from atom_excision.immediate_phase):
# tau-pin a circuit on, ablate a node-set off, read held-out loss + 2AFC.
# Returns cost(node_list) -> {"loss","2afc","hit"} as mean increase/drop over the
# source circuits, with results cached by frozenset so singletons are reused.
# ---------------------------------------------------------------------------
def make_oracle(args, exp_dir, src_dir):
    import torch
    from contextlib import nullcontext
    from transformers import AutoTokenizer
    from sparse_pretrain.src.pruning.config import PruningConfig
    from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
    from sparse_pretrain.src.pruning.run_pruning import load_model
    from sparse_pretrain.scripts.pronoun_split import tasks_from_split_info

    device = args.device
    print("  [oracle] loading model + held-out task ...", flush=True)
    model, _ = load_model(args.model, device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hp = json.load(open(exp_dir / "hparams.json"))
    pc = PruningConfig(
        k_coef=hp["k_coef"], weight_decay=hp["weight_decay"], lr=hp["lr"], beta2=hp["beta2"],
        heaviside_temp=hp["heaviside_temp"], init_noise_scale=0.01, init_noise_bias=0.1,
        lr_warmup_frac=0.0, num_steps=args.num_steps, batch_size=args.batch_size, seq_length=0,
        device=device, log_every=10**9, target_loss=args.target_loss, ablation_type="zero",
        mask_token_embeds=False)
    si = json.load(open(exp_dir / "split_info.json"))
    _, _, test_task = tasks_from_split_info(tok, si, task_seed=0)
    pos, neg, corr, inc, ep = test_task.full_batch()
    pos, neg = pos.to(device), neg.to(device)
    corr, inc, ep = corr.to(device), inc.to(device), ep.to(device)

    mm = MaskedSparseGPT(model, pc); mm.to(device); mm.eval()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    def force(nodeset):
        with torch.no_grad():
            for msk in mm.masks.masks.values():
                msk.tau.fill_(-1.0)
            bk = {}
            for (key, idx) in nodeset:
                bk.setdefault(key, []).append(idx)
            for key, idxs in bk.items():
                mm.masks.masks[key].tau[idxs] = 1.0

    def off(nodes):
        with torch.no_grad():
            for (key, idx) in nodes:
                mm.masks.masks[key].tau[idx] = -1.0

    def ev():
        with autocast, torch.no_grad():
            _, tm = mm.compute_task_loss(pos, neg, corr, inc, ep)
            _, tb = mm.compute_task_loss(pos, neg, corr, inc, ep, use_binary_loss=True)
        return float(tm["task_loss"]), float(tb["binary_accuracy"])

    circuits = [circuit_nodes(torch.load(p, map_location="cpu", weights_only=True))
                for p in sorted((src_dir / f"iter{args.iter:02d}").glob("seed*_circuit.pt"))]
    base = []
    for C in circuits:
        force(C); l, a = ev(); base.append((l, a))
    base_2afc = float(np.mean([a for _, a in base]))
    base_loss = float(np.mean([l for l, _ in base]))
    print(f"  [oracle] {len(circuits)} source circuits; base held-out 2AFC={base_2afc:.3f} "
          f"loss={base_loss:.4f}", flush=True)

    cache, n_eval = {}, [0]

    def cost(node_list):
        key = frozenset((k, int(i)) for k, i in node_list)
        if key in cache:
            return cache[key]
        if not key:
            r = {"loss": 0.0, "2afc": 0.0, "hit": 0.0}; cache[key] = r; return r
        dloss, d2afc, hit = [], [], []
        for C, (bl, ba) in zip(circuits, base):
            force(C); off(key)
            al, aa = ev()
            dloss.append(al - bl); d2afc.append(ba - aa); hit.append(len(key & C))
        r = {"loss": float(np.mean(dloss)), "2afc": float(np.mean(d2afc)),
             "hit": float(np.mean(hit))}
        cache[key] = r; n_eval[0] += 1
        return r

    meta = {"base_2afc": round(base_2afc, 4), "base_loss": round(base_loss, 4),
            "n_circuits": len(circuits), "n_eval": n_eval}

    def cleanup():
        nonlocal mm, model
        del mm, model
        torch.cuda.empty_cache()

    return cost, meta, cleanup


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def connected_components(nodes, edges):
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for x in nodes:
        comps.setdefault(find(x), []).append(x)
    return [sorted(c) for c in comps.values()]


# ---------------------------------------------------------------------------
# Plot: z-score interaction heatmap over testable atoms
# ---------------------------------------------------------------------------
def make_plot(report, D, testable, out_png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skipping plot: {e})"); return
    atoms = report["testable_atoms"]
    Z = np.array(report["z_matrix"], dtype=float)
    labels = [f"a{a}\n{report['atom_units'][str(a)]['top_node']}" for a in atoms]
    zmax = max(2.0, np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 2.0)
    fig, ax = plt.subplots(figsize=(1.2 * len(atoms) + 3, 1.2 * len(atoms) + 2))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-zmax, vmax=zmax)
    ax.set_xticks(range(len(atoms))); ax.set_yticks(range(len(atoms)))
    ax.set_xticklabels(labels, fontsize=7, rotation=90); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if i == j or not np.isfinite(Z[i, j]):
                continue
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(Z[i, j]) > zmax * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="z(epsilon) vs matched-random null")
    mods = [m for m in report["modules"] if len(m) > 1]
    sub = (f"size-matched null mu0={report['null']['global_mu0']:.3f} "
           f"sd0={report['null']['global_sd0']:.3f} (n={report['null']['n_eps']}) | "
           f"z_thr={report['z_thr']} | ~{report['expected_false_pos']} sig pairs by chance\n"
           f"{report['verdict']}")
    ax.set_title("Functional epistasis (no re-factoring): z of "
                 "epsilon=cost(AuB)-cost(A)-cost(B)\n" + sub, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True,
                    help="dictionary run (iter*/seed*_circuit.pt, hparams.json, split_info.json, "
                         "run_args.json). Pruning/eval config is read from its run_args.json.")
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--backbone-thr", type=float, default=0.6, dest="backbone_thr")
    ap.add_argument("--min-share", type=float, default=0.25, dest="min_share",
                    help="a non-backbone node is OWNED by its argmax atom only if that atom holds "
                         ">= this fraction of the node's total loading (disjoint, confident units).")
    ap.add_argument("--min-size", type=int, default=1, dest="min_size",
                    help="min owned non-backbone nodes for an atom to be testable.")
    ap.add_argument("--null-pool", type=int, default=8, dest="null_pool",
                    help="# random non-backbone sets drawn per atom-size, reused across pairs.")
    ap.add_argument("--null-reps", type=int, default=8, dest="null_reps",
                    help="max matched-random eps draws per (size_a,size_b) -> SIZE-MATCHED null, "
                         "since eps scales with set size (atom 1 has many more nodes than atom 3).")
    ap.add_argument("--z-thr", type=float, default=2.0, dest="z_thr",
                    help="|z| >= this => the pair interacts beyond generic curvature (graph edge).")
    ap.add_argument("--rng-seed", type=int, default=0, dest="rng_seed")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny check: null_sets=4.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    src_args = json.load(open(src_dir / "run_args.json"))
    for k in PASS_THROUGH:
        setattr(args, k, src_args[k])
    if args.split_over != "names_templates":
        raise SystemExit("expects the names_templates split (held-out 2AFC/loss readout).")
    if args.smoke:
        args.null_pool, args.null_reps = 3, 3
    for fn in ("hparams.json", "split_info.json"):
        if not (exp_dir / fn).exists():
            shutil.copy(src_dir / fn, exp_dir / fn)

    # --- dictionary + disjoint NMF-only atom units ---
    print(f"Building fixed dictionary from {src_dir.name}/iter{args.iter:02d} ...", flush=True)
    D = build_dictionary(src_dir, args.iter)
    testable, detail = atom_owned_cols(D, args.backbone_thr, args.min_share, args.min_size)
    atoms = sorted(testable)
    print(f"  K={D['k']} atoms over {len(D['anodes'])} active nodes, {D['n_circuits']} circuits", flush=True)
    units = {}
    for a in atoms:
        cs = testable[a]
        units[str(a)] = {"size": len(cs),
                         "nodes": [node_label(D, j) for j in cs],
                         "loadings": [round(float(D["H"][a, j]), 4) for j in cs],
                         "shares": [round(detail[j][1], 3) for j in cs],
                         "freq": [round(float(D["freq"][j]), 3) for j in cs],
                         "top_node": node_label(D, cs[0])}
        print(f"    atom {a:>2}: {len(cs)} nodes  {[node_label(D, j) for j in cs]}", flush=True)
    if len(atoms) < 2:
        raise SystemExit(f"need >=2 testable atoms, got {len(atoms)} "
                         f"(loosen --min-share / --min-size).")

    cost, meta, cleanup = make_oracle(args, exp_dir, src_dir)

    # --- singleton + pair costs -> epsilon ---
    print("\n=== epsilon matrix (immediate, no re-factoring) ===", flush=True)
    csingle = {a: cost(cols_to_nodes(D, testable[a])) for a in atoms}
    pairs = []
    for a, b in itertools.combinations(atoms, 2):
        cu = cost(cols_to_nodes(D, testable[a] + testable[b]))
        eL = cu["loss"] - csingle[a]["loss"] - csingle[b]["loss"]
        e2 = cu["2afc"] - csingle[a]["2afc"] - csingle[b]["2afc"]
        pairs.append({"a": a, "b": b, "cost_a": round(csingle[a]["loss"], 4),
                      "cost_b": round(csingle[b]["loss"], 4), "cost_union": round(cu["loss"], 4),
                      "eps_loss": round(eL, 4), "eps_2afc": round(e2, 4),
                      "cohit_union": round(cu["hit"], 2)})

    # --- SIZE-MATCHED matched-random null for epsilon (loss) ---
    # eps grows with set size, so each atom pair (sizes sa,sb) is compared to random
    # disjoint non-backbone set-pairs of THE SAME sizes; random singleton costs are
    # cached per size and reused across size-pairs.
    disc_pool = [j for j in range(len(D["anodes"])) if D["freq"][j] < args.backbone_thr]
    rng = np.random.default_rng(args.rng_seed)
    sizes_needed = sorted({len(testable[a]) for a in atoms})
    rsets = {}
    for s in sizes_needed:
        ss = min(s, len(disc_pool))
        sel = [sorted(int(x) for x in rng.choice(disc_pool, size=ss, replace=False))
               for _ in range(args.null_pool)]
        rsets[s] = [(c, cost(cols_to_nodes(D, c))["loss"]) for c in sel]

    null_cache, all_null_eps = {}, []

    def null_for(sa, sb):
        key = tuple(sorted((sa, sb)))
        if key in null_cache:
            return null_cache[key]
        eps = []
        for (ra, la) in rsets[key[0]]:
            for (rb, lb) in rsets[key[1]]:
                if set(ra) & set(rb):
                    continue
                cu = cost(cols_to_nodes(D, ra + rb))["loss"]
                eps.append(cu - la - lb)
                if len(eps) >= args.null_reps:
                    break
            if len(eps) >= args.null_reps:
                break
        null_cache[key] = eps; all_null_eps.extend(eps)
        return eps

    for p in pairs:
        nl = null_for(len(testable[p["a"]]), len(testable[p["b"]]))
        mu, sd = (float(np.mean(nl)), float(np.std(nl))) if len(nl) > 1 else (0.0, 0.0)
        p["null_mu"], p["null_sd"], p["null_n"] = round(mu, 4), round(sd, 4), len(nl)
        p["z"] = round((p["eps_loss"] - mu) / sd, 2) if (len(nl) >= 3 and sd > 0) else None
    mu0 = float(np.mean(all_null_eps)) if all_null_eps else 0.0
    sd0 = float(np.std(all_null_eps)) if len(all_null_eps) > 1 else 0.0
    print(f"  null: {len(all_null_eps)} matched-random eps over {len(null_cache)} size-pairs "
          f"(global mu0={mu0:.4f} sd0={sd0:.4f})", flush=True)

    # --- graph + modules ---
    edges = [(p["a"], p["b"]) for p in pairs if p["z"] is not None and abs(p["z"]) >= args.z_thr]
    modules = connected_components(atoms, edges)
    nontrivial = [m for m in modules if len(m) > 1]

    idx = {a: i for i, a in enumerate(atoms)}
    Z = np.full((len(atoms), len(atoms)), np.nan)
    for p in pairs:
        if p["z"] is not None:
            Z[idx[p["a"]], idx[p["b"]]] = Z[idx[p["b"]], idx[p["a"]]] = p["z"]

    from math import erfc, sqrt
    n_tested = sum(1 for p in pairs if p["z"] is not None)
    expected_fp = round(n_tested * erfc(args.z_thr / sqrt(2)), 1)  # two-sided normal tail
    nsig = len(edges)
    sub = sum(1 for p in pairs if p["z"] is not None and p["z"] <= -args.z_thr)
    sup = sum(1 for p in pairs if p["z"] is not None and p["z"] >= args.z_thr)
    density = nsig / n_tested if n_tested else 0.0       # fraction of pairs non-additive
    sign = ("mostly sub-additive (redundant/overlapping roles)" if sub >= sup
            else "mostly super-additive (jointly-necessary units)")
    if nsig <= expected_fp:
        verdict = (f"FUNCTIONALLY ATOMIC: {nsig}/{n_tested} pairs exceed |z|>={args.z_thr}, "
                   f"~{expected_fp} expected by chance -> no interaction above noise. The NMF "
                   f"atoms are functionally independent at this granularity; earlier re-factoring "
                   f"non-atomicity was the coupling artifact, not real interaction.")
    elif density >= 0.30:
        verdict = (f"HOLISTIC: {nsig}/{n_tested} pairs non-additive ({density:.0%}; "
                   f"~{expected_fp} by chance; {sub} sub, {sup} super) -> interactions are dense, "
                   f"no modular decomposition; a single entangled basin.")
    else:
        verdict = (f"SPARSELY INTERACTING: {nsig}/{n_tested} pairs non-additive ({density:.0%}, "
                   f"~{expected_fp} by chance; {sign}); most pairs ARE additive. Truly atomic "
                   f"units = singletons + merged modules "
                   f"{['+'.join('a'+str(a) for a in m) for m in nontrivial]}.")
    print(f"\n  VERDICT: {verdict}", flush=True)
    topsig = sorted([p for p in pairs if p["z"] is not None and abs(p["z"]) >= args.z_thr],
                    key=lambda p: -abs(p["z"]))
    for p in topsig[:12]:
        print(f"    a{p['a']}~a{p['b']}: z={p['z']:+.1f} eps={p['eps_loss']:+.3f} "
              f"(cost_a={p['cost_a']} cost_b={p['cost_b']} union={p['cost_union']} "
              f"cohit={p['cohit_union']})", flush=True)

    report = {"src_dir": str(src_dir), "exp_dir": str(exp_dir), "iter": args.iter,
              "k": D["k"], "backbone_thr": args.backbone_thr, "min_share": args.min_share,
              "min_size": args.min_size, "z_thr": args.z_thr,
              "base": {"loss": meta["base_loss"], "2afc": meta["base_2afc"],
                       "n_circuits": meta["n_circuits"]},
              "testable_atoms": atoms, "atom_units": units,
              "singleton_cost_loss": {str(a): round(csingle[a]["loss"], 4) for a in atoms},
              "singleton_2afc_drop": {str(a): round(csingle[a]["2afc"], 4) for a in atoms},
              "pairs": sorted(pairs, key=lambda p: (p["z"] if p["z"] is not None else 0)),
              "null": {"global_mu0": round(mu0, 4), "global_sd0": round(sd0, 4),
                       "n_eps": len(all_null_eps), "n_size_pairs": len(null_cache),
                       "size_matched": True},
              "z_matrix": [[None if not np.isfinite(v) else round(float(v), 2) for v in row]
                           for row in Z],
              "modules": modules, "n_significant_pairs": nsig, "n_pairs_tested": n_tested,
              "expected_false_pos": expected_fp, "verdict": verdict,
              "n_oracle_evals": meta["n_eval"][0]}
    json.dump(report, open(exp_dir / "epistasis_test.json", "w"), indent=2, default=str)
    print(f"\nwrote {exp_dir / 'epistasis_test.json'} ({meta['n_eval'][0]} oracle evals)", flush=True)
    make_plot(report, D, testable, exp_dir / "epistasis_test.png")
    cleanup()


if __name__ == "__main__":
    main()
