#!/usr/bin/env python3
"""
v2 -- RE-FACTORIZING iterative atom-peel (cluster-free, functional).

WHAT THIS IS
------------
The single-knockout atom_excision showed the universal motif is REPLACEABLE
(re-prune recovers at ~zero cost by recruiting a different atom). This experiment
asks how DEEP that redundancy goes: strip the preferred substrate one atom at a
time, re-pruning AND re-factoring each time, until the task can no longer be
recovered. The collapse depth = how many stacked independent solutions exist; the
sequence of dominant motifs = the fallback hierarchy.

This is "v2": unlike atom_excision (FIXED iter-0 dictionary), every level is
re-factored fresh, so each fallback solution is described in ITS OWN basis (the
fixed dictionary is structurally blind to emergent fallback structure -- re-pruned
circuits load only ~0.4 onto the original atoms).

CLUSTER-FREE AT EVERY DECISION (no motif_summary.json, no Jaccard):
  - which atoms exist      : NMF on the level's circuits (rank RE-SELECTED per level
                             by held-out reconstruction, same criterion as the source)
  - each atom's node-set   : disjoint argmax-ownership of non-backbone nodes
                             (epistasis_test.atom_owned_cols; share>=min_share)
  - backbone vs motif      : node marginal frequency (freq<thr)
  - which atom to peel      : FUNCTIONAL -- the immediate-damage oracle (highest
                             held-out loss increase when its nodes are ablated)
  - all cross-level claims : FUNCTIONAL (feasibility / recovery cost / 2AFC /
                             universality). NEVER atom identity across levels --
                             that re-factoring-identity comparison is exactly the
                             coupling artifact epistasis_test exposed.

THE LOOP (level d = 0,1,2,...)
  0. level 0 circuits = the source run's iter circuits (full model, no exclusions).
  a. factor the level's circuits (NMF) -> atoms; define units (argmax-ownership).
  b. score each unit by oracle damage on the level's circuits; record universality
     (fraction of the level's circuits that use it).
  c. PEEL the max-damage discriminative atom; add its nodes to the cumulative
     forbidden set F.
  d. re-prune N seeds with F pinned off (the recoverable verify, reusing the
     ORIGINAL worker) -> next level's circuits + feasibility/size/loss/2AFC.
  repeat until feasibility collapses (<feas_eps) or atoms/nodes exhausted.

Note on the a0/a5 over-split (from epistasis_test): we do NOT hardcode a merge.
Because we re-factor, if level 0 peels half a motif the re-pruned circuits still
lean on the other half, which then dominates and is peeled next -- the loop
self-corrects, and the per-level universality/damage of the peeled atom makes an
over-split remnant visible (low support following a strong motif).

BASELINE -- matched-random peel: a parallel peel that, at each depth, forbids the
SAME NUMBER of frequency-matched random non-backbone nodes (cumulative, excluding
every atom-owned node), re-pruned identically (functional-only, no re-factor),
R reps. The atom-vs-random feasibility/cost gap separates "stripped the
load-bearing hierarchy" from "removed N nodes".

Resumable: a level/condition whose seed results already exist is skipped; peel
selection is deterministic (fixed NMF seed) so F rebuilds identically on resume.
MPS daemon required for the multi-worker re-prune (nvidia-cuda-mps-control -d).
"""

import sys, os, json, argparse, time, shutil
from pathlib import Path

import numpy as np 

from sparse_pretrain.scripts.motif_dictionary import (
    global_node_space, load_iteration, describe_atoms, heldout_curves, select_k,
)
from sparse_pretrain.scripts.atom_excision import (
    reprune_condition, collect, cols_to_nodes, cos, read_parsimonious_k,
)
from sparse_pretrain.scripts.epistasis_test import atom_owned_cols, node_label
from sparse_pretrain.scripts.universality_pruning_experiment import circuit_nodes

PASS_THROUGH = ["model", "task", "tokenizer", "target_loss", "num_steps", "batch_size",
                "eval_batches", "bisect_iters", "split_over", "split_seed", "test_frac",
                "heldout_fold", "name_pool"]


# ---------------------------------------------------------------------------
# Per-level NMF re-factor (cluster-free units)
# ---------------------------------------------------------------------------
def select_level_k(Xa, ksel):
    """Per-level NMF rank. mode 'fixed' -> use ksel['fixed'] for all levels; else
    RE-SELECT k on THIS level's circuits by the same held-out-reconstruction
    criterion motif_dictionary.py used for the source (k_parsimonious = smallest k
    within 1 SE of the held-out-relerr minimum). Falls back to the source iter-0 K
    when a level has too few circuits to split."""
    n, active = Xa.shape
    fb = max(1, min(ksel["fallback"], active, n - 1))
    if ksel["mode"] == "fixed":
        return max(1, min(ksel["fixed"], active, n - 1)), "user_fixed"
    if n < ksel["min_n"]:
        return fb, "min_n_fallback"
    n_tr = n - max(1, int(round(n * ksel["heldout_frac"])))
    ks = [k for k in range(1, ksel["kmax"] + 1) if k <= min(n_tr - 1, active)]
    if not ks:
        return fb, "no_ks_fallback"
    real, null = heldout_curves(Xa, ks, ksel["reps"], ksel["heldout_frac"], ksel["seed"])
    kp = select_k(real, null, ks).get("k_parsimonious")
    return (int(kp), "heldout_parsimonious") if kp else (fb, "select_none_fallback")


def factor_level(it_dir, nodes, index, ksel, backbone_thr, min_share, min_size, seed=0, topn=12):
    X, _ = load_iteration(Path(it_dir), index, len(nodes))
    if len(X) == 0:
        return None, None, None, None
    amask = X.sum(0) > 0
    anodes = [nodes[j] for j in np.where(amask)[0]]
    Xa = X[:, amask]
    freq = Xa.mean(0)
    kk, kmethod = select_level_k(Xa, ksel)
    kk = max(1, min(kk, len(anodes), len(Xa) - 1))
    atoms, _, H = describe_atoms(Xa, kk, anodes, freq, seed, topn)
    D = dict(anodes=anodes, freq=freq, H=H, atoms=atoms, k=kk, n_circuits=len(Xa),
             col={f"{key}#{i}": c for c, (key, i) in enumerate(anodes)})
    units, detail = atom_owned_cols(D, backbone_thr, min_share, min_size)
    return D, units, detail, {"k": int(kk), "k_method": kmethod, "n_active": int(amask.sum())}


def load_circuit_sets(it_dir):
    import torch
    return [circuit_nodes(torch.load(p, map_location="cpu", weights_only=True))
            for p in sorted(Path(it_dir).glob("seed*_circuit.pt"))]


# ---------------------------------------------------------------------------
# Immediate-damage oracle (verbatim conventions from atom_excision.immediate_phase
# / epistasis_test.make_oracle) -- load model ONCE, re-point circuits per level.
# ---------------------------------------------------------------------------
class Oracle:
    def __init__(self, args, exp_dir):
        import torch
        from contextlib import nullcontext
        from transformers import AutoTokenizer
        from sparse_pretrain.src.pruning.config import PruningConfig
        from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
        from sparse_pretrain.src.pruning.run_pruning import load_model
        from sparse_pretrain.scripts.pronoun_split import tasks_from_split_info
        self.torch = torch
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
        self.batch = (pos.to(device), neg.to(device), corr.to(device), inc.to(device), ep.to(device))
        self.mm = MaskedSparseGPT(model, pc); self.mm.to(device); self.mm.eval()
        self.model = model
        self.autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                         if device == "cuda" else nullcontext())
        self.circuits, self.base = [], []

    def _force(self, nodeset):
        with self.torch.no_grad():
            for msk in self.mm.masks.masks.values():
                msk.tau.fill_(-1.0)
            bk = {}
            for (key, idx) in nodeset:
                bk.setdefault(key, []).append(idx)
            for key, idxs in bk.items():
                self.mm.masks.masks[key].tau[idxs] = 1.0

    def _off(self, nodes):
        with self.torch.no_grad():
            for (key, idx) in nodes:
                self.mm.masks.masks[key].tau[idx] = -1.0

    def _ev(self):
        pos, neg, corr, inc, ep = self.batch
        with self.autocast, self.torch.no_grad():
            _, tm = self.mm.compute_task_loss(pos, neg, corr, inc, ep)
            _, tb = self.mm.compute_task_loss(pos, neg, corr, inc, ep, use_binary_loss=True)
        return float(tm["task_loss"]), float(tb["binary_accuracy"])

    def set_circuits(self, circuit_sets):
        self.circuits = circuit_sets
        self.base = []
        for C in circuit_sets:
            self._force(C); l, a = self._ev(); self.base.append((l, a))

    def base_stats(self):
        if not self.base:
            return 0.0, 0.0
        return (float(np.mean([l for l, _ in self.base])),
                float(np.mean([a for _, a in self.base])))

    def cost(self, node_list):
        """mean held-out loss increase / 2AFC drop over the current circuits when
        node_list is ablated; + mean #nodes hit per circuit."""
        excl = set((k, int(i)) for k, i in node_list)
        if not excl or not self.circuits:
            return {"loss": 0.0, "2afc": 0.0, "hit": 0.0}
        dloss, d2afc, hit = [], [], []
        for C, (bl, ba) in zip(self.circuits, self.base):
            self._force(C); self._off(excl)
            al, aa = self._ev()
            dloss.append(al - bl); d2afc.append(ba - aa); hit.append(len(excl & C))
        return {"loss": float(np.mean(dloss)), "2afc": float(np.mean(d2afc)),
                "hit": float(np.mean(hit))}

    def cleanup(self):
        del self.mm, self.model
        self.torch.cuda.empty_cache()


def score_units(oracle, D, units):
    """Score every unit on the current circuits: functional damage + universality."""
    out = {}
    for a, cols in units.items():
        nl = cols_to_nodes(D, cols)
        c = oracle.cost(nl)
        ns = set((k, int(i)) for k, i in nl)
        support = float(np.mean([1.0 if (ns & C) else 0.0 for C in oracle.circuits])) \
            if oracle.circuits else 0.0
        out[a] = {"atom": a, "size": len(cols), "damage_loss": round(c["loss"], 4),
                  "damage_2afc": round(c["2afc"], 4), "support": round(support, 3),
                  "top_node": node_label(D, cols[0]),
                  "nodes": [node_label(D, j) for j in cols]}
    return out


# ---------------------------------------------------------------------------
# Matched-random cumulative peel (frequency-matched, excludes all atom nodes)
# ---------------------------------------------------------------------------
def matched_random_extend(D0, target_nodes, exclude_nodes, rng, backbone_thr, knn=8):
    """Draw len(target_nodes) non-backbone nodes whose level-0 freq matches the
    targets', excluding `exclude_nodes` (all atom nodes + already chosen)."""
    freq, anodes, col = D0["freq"], D0["anodes"], D0["col"]
    excl_cols = {col[f"{k}#{i}"] for (k, i) in exclude_nodes if f"{k}#{i}" in col}
    pool = [j for j in range(len(anodes)) if freq[j] < backbone_thr and j not in excl_cols]
    chosen = []
    for (k, i) in target_nodes:
        key = f"{k}#{i}"
        f = freq[col[key]] if key in col else 0.0
        cands = sorted([p for p in pool if p not in chosen], key=lambda p: abs(freq[p] - f))
        pick = cands[:knn] or cands
        if not pick:
            break
        chosen.append(int(rng.choice(pick)))
    return [[anodes[c][0], int(anodes[c][1])] for c in chosen]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(report, out_png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skipping plot: {e})"); return
    atom = report["atom_trajectory"]
    ad = [r["depth"] for r in atom]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    def rand_series(key):
        reps = report.get("random_trajectory") or []
        by = {}
        for rep in reps:
            for r in rep:
                by.setdefault(r["depth"], []).append(r.get(key))
        xs = sorted(by)
        vals = [[v for v in by[d] if v is not None] for d in xs]
        return xs, [np.mean(v) if v else np.nan for v in vals], [np.std(v) if len(v) > 1 else 0 for v in vals]

    # feasibility
    ax = axes[0][0]
    ax.plot(ad, [r.get("feasibility") for r in atom], "o-", label="atom-peel")
    if report.get("random_trajectory"):
        xs, m, s = rand_series("feasibility")
        ax.errorbar(xs, m, yerr=s, fmt="s--", color="gray", label="matched-random", capsize=3)
    ax.axhline(report["feas_eps"], color="red", ls=":", lw=1, label="collapse thr")
    ax.set_xlabel("peel depth"); ax.set_ylabel("re-prune feasibility"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("Feasibility vs depth (how deep redundancy goes)"); ax.legend(); ax.grid(alpha=0.3)

    # recovery cost
    ax = axes[0][1]
    ax.plot(ad, [r.get("mean_circuit_size") for r in atom], "o-", label="atom-peel")
    if report.get("random_trajectory"):
        xs, m, s = rand_series("mean_circuit_size")
        ax.errorbar(xs, m, yerr=s, fmt="s--", color="gray", label="matched-random", capsize=3)
    ax.set_xlabel("peel depth"); ax.set_ylabel("recovered circuit size (nodes)")
    ax.set_title("Recovery cost vs depth"); ax.legend(); ax.grid(alpha=0.3)

    # peeled-atom composition (attention vs MLP) -- the fallback-basin switch.
    # Read alongside recovery cost: cheap early basins are attention-centred; when
    # they exhaust the re-prune falls back on an MLP-dominated basin and cost explodes.
    ax = axes[0][2]
    peeled = [(r["depth"], r["peeled"]) for r in atom if r.get("peeled")]
    def _frac(nodes):
        a = sum("attn" in n for n in nodes); m = sum("mlp" in n for n in nodes)
        t = max(len(nodes), 1)
        return a / t, m / t, (len(nodes) - a - m) / t
    bd  = [d for d, _ in peeled]
    fa  = [_frac(p["nodes"])[0] for _, p in peeled]
    fm  = [_frac(p["nodes"])[1] for _, p in peeled]
    fo  = [_frac(p["nodes"])[2] for _, p in peeled]
    dmg = [p["damage_loss"] for _, p in peeled]
    strong = [d > 0.3 for d in dmg]          # basin-defining atoms vs near-zero filler
    edge = ["black" if s else "none" for s in strong]
    lw   = [1.8 if s else 0.0 for s in strong]
    ax.bar(bd, fa, color="steelblue", edgecolor=edge, linewidth=lw, label="attention")
    ax.bar(bd, fm, bottom=fa, color="indianred", edgecolor=edge, linewidth=lw, label="MLP")
    if any(o > 1e-9 for o in fo):
        ax.bar(bd, fo, bottom=[a + m for a, m in zip(fa, fm)], color="lightgray", label="other")
    for d, s, dl in zip(bd, strong, dmg):    # annotate damage on basin-defining atoms
        if s:
            ax.text(d, 1.03, f"{dl:.2f}", ha="center", va="bottom", rotation=90,
                    fontsize=7, fontweight="bold")
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.5)
    # Mark a basin switch ONLY on runs that actually collapse, and locate it at the
    # ONSET OF TERMINAL DECLINE (feasibility never recovers >=0.9 afterwards) -- not the
    # first feasibility dip, which fires on the early re-prune noise of non-collapsing
    # runs (e.g. d1024 oscillates 0.3-0.9 early but never collapses -> no switch).
    trans = None
    if report["summary"].get("atom_collapse_depth") is not None:
        fbd = [(r["depth"], r.get("feasibility") if r.get("feasibility") is not None else 0.0)
               for r in atom]
        for i, (d, _) in enumerate(fbd):
            if d > 0 and all(f < 0.9 for _, f in fbd[i:]):
                trans = d; break
    if trans is not None:                    # mark the cost/feasibility phase transition
        ax.axvline(trans - 0.5, color="purple", ls="--", lw=1.5)
        bb = dict(boxstyle="round", fc="white", ec="purple", alpha=0.85)
        ax.text((trans - 1) / 2, 0.6, "attention\nbasins", ha="center", va="center",
                fontsize=8, color="purple", bbox=bb)
        ax.text(trans + (bd[-1] - trans) / 2, 0.3, "MLP\nfallback", ha="center",
                va="center", fontsize=8, color="purple", bbox=bb)
    ax.set_ylim(0, 1.3); ax.set_xlabel("peel depth")
    ax.set_ylabel("peeled-atom node fraction")
    ax.set_title("Basin composition (attn→MLP switch; bold = damage>0.3)")
    ax.legend(loc="center left", fontsize=8); ax.grid(alpha=0.3, axis="y")

    # peeled-atom quality gradient (atom arm only)
    ax = axes[1][0]
    pd = [r["depth"] for r in atom if r.get("peeled")]
    ax.plot(pd, [r["peeled"]["support"] for r in atom if r.get("peeled")], "o-", label="universality (support)")
    ax2 = ax.twinx()
    ax2.plot(pd, [r["peeled"]["damage_loss"] for r in atom if r.get("peeled")], "^--",
             color="darkorange", label="functional load (damage)")
    ax.set_xlabel("peel depth"); ax.set_ylabel("dominant-atom universality")
    ax2.set_ylabel("dominant-atom damage (loss)")
    ax.set_title("Quality gradient of the peeled motif"); ax.grid(alpha=0.3)
    l1, la = ax.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la + lb, fontsize=8)

    # held-out 2afc
    ax = axes[1][1]
    ax.plot(ad, [r.get("mean_test_2afc") for r in atom], "o-", label="atom-peel")
    if report.get("random_trajectory"):
        xs, m, s = rand_series("mean_test_2afc")
        ax.errorbar(xs, m, yerr=s, fmt="s--", color="gray", label="matched-random", capsize=3)
    ax.set_xlabel("peel depth"); ax.set_ylabel("held-out 2AFC (survivors)")
    ax.set_title("Behaviour vs depth"); ax.legend(); ax.grid(alpha=0.3)

    axes[1][2].axis("off")

    fig.suptitle(f"v2 re-factorizing atom-peel  |  {report['exp_dir'].split('/')[-1]}  |  "
                 f"atom collapse depth={report['summary'].get('atom_collapse_depth')}", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--iter", type=int, default=0, help="source iteration for level 0.")
    ap.add_argument("--max-depth", type=int, default=8, dest="max_depth")
    ap.add_argument("--feas-eps", type=float, default=0.1, dest="feas_eps",
                    help="stop when re-prune feasibility falls below this.")
    ap.add_argument("--backbone-thr", type=float, default=0.6, dest="backbone_thr")
    ap.add_argument("--min-share", type=float, default=0.25, dest="min_share")
    ap.add_argument("--min-size", type=int, default=1, dest="min_size")
    ap.add_argument("--k", type=int, default=-1,
                    help="fixed NMF rank for ALL levels (>0). Default -1 = RE-SELECT k per level by "
                         "held-out reconstruction (same criterion as motif_dictionary.py).")
    ap.add_argument("--kmax", type=int, default=16, dest="kmax")
    ap.add_argument("--ksel-reps", type=int, default=6, dest="ksel_reps")
    ap.add_argument("--ksel-heldout-frac", type=float, default=0.25, dest="ksel_heldout_frac")
    ap.add_argument("--ksel-min-n", type=int, default=10, dest="ksel_min_n")
    ap.add_argument("--ksel-seed", type=int, default=0, dest="ksel_seed")
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--rand-reps", type=int, default=2, dest="rand_reps")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--num-seeds", type=int, default=100, dest="num_seeds")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset")
    ap.add_argument("--num-workers", type=int, default=16, dest="num_workers")
    ap.add_argument("--rng-seed", type=int, default=0, dest="rng_seed")
    ap.add_argument("--skip-random", action="store_true", dest="skip_random")
    ap.add_argument("--fresh-level0", action="store_true", dest="fresh_level0",
                    help="Generate level-0 by re-pruning the new seed block with an EMPTY "
                         "exclusion instead of copying src_dir/iter{iter}. iter00 then uses the "
                         "SAME seeds [seed_offset, seed_offset+num_seeds) as every deeper level -> "
                         "a self-contained independent replicate that relies on no existing result "
                         "directory (source is read only for pruning config/hparams/split).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    src_args = json.load(open(src_dir / "run_args.json"))
    for k in PASS_THROUGH:
        setattr(args, k, src_args[k])
    args.name_pool = str(Path(args.name_pool).resolve())
    if args.split_over != "names_templates":
        raise SystemExit("expects the names_templates split (held-out 2AFC readout).")
    if args.smoke:
        (args.num_seeds, args.num_workers, args.max_depth, args.rand_reps,
         args.num_steps, args.ksel_min_n, args.kmax) = 4, 4, 2, 1, 200, 2, 4

    for fn in ("hparams.json", "split_info.json"):
        if not (exp_dir / fn).exists():
            shutil.copy(src_dir / fn, exp_dir / fn)
    json.dump(vars(args), open(exp_dir / "run_args.json", "w"), indent=2, default=str)

    k0 = read_parsimonious_k(src_dir, args.iter)   # source iter-0 K (fallback)
    ksel = {"mode": "fixed" if args.k > 0 else "perlevel", "fixed": args.k,
            "kmax": args.kmax, "reps": args.ksel_reps, "heldout_frac": args.ksel_heldout_frac,
            "seed": args.ksel_seed, "min_n": args.ksel_min_n, "fallback": k0}

    # level-0 circuits. Default: copy the source iter circuits (carries the source's
    # seeds). --fresh-level0: re-prune the new seed block with an EMPTY exclusion so
    # iter00 uses the SAME seeds as every deeper level -> internally-consistent
    # independent replicate, no reliance on an existing result directory.
    lvl0 = exp_dir / "iter00"; lvl0.mkdir(exist_ok=True)
    if args.fresh_level0:
        print(f"\n=== FRESH LEVEL 0: re-prune seeds "
              f"[{args.seed_offset},{args.seed_offset + args.num_seeds}) (empty exclusion) ===", flush=True)
        agg0, secs0 = reprune_condition({"iterN": 0, "excluded": []}, args, exp_dir)
        print(f"  level-0: feas={agg0.get('feasibility')} size={agg0.get('mean_circuit_size')} "
              f"2afc={agg0.get('mean_test_2afc')} ({secs0/60:.1f}m)", flush=True)
    elif not list(lvl0.glob("seed*_result.json")):
        for p in (src_dir / f"iter{args.iter:02d}").glob("seed*_result.json"):
            shutil.copy(p, lvl0 / p.name)
        for p in (src_dir / f"iter{args.iter:02d}").glob("seed*_circuit.pt"):
            shutil.copy(p, lvl0 / p.name)

    # node space: from the freshly-pruned exp_dir when --fresh-level0 (key/shape
    # structure is model-determined, so it is identical to the source's), else the source.
    nodes, index = global_node_space(exp_dir if args.fresh_level0 else src_dir)
    print(f"global node space: {len(nodes)} nodes | K mode={ksel['mode']}"
          + (f" (re-select per level: kmax={args.kmax} reps={args.ksel_reps} "
             f"heldout_frac={args.ksel_heldout_frac}; source iter0 K={k0})"
             if ksel["mode"] == "perlevel" else f" (fixed K={args.k})"), flush=True)

    oracle = Oracle(args, exp_dir)

    report = {"src_dir": str(src_dir), "exp_dir": str(exp_dir), "iter": args.iter, "k": k0,
              "backbone_thr": args.backbone_thr, "min_share": args.min_share,
              "feas_eps": args.feas_eps, "num_seeds": args.num_seeds,
              "atom_trajectory": [], "random_trajectory": [], "summary": {}}

    # ---- ATOM-PEEL ARM ----
    print("\n=== ATOM-PEEL ARM ===", flush=True)
    F = set()                      # cumulative forbidden (key,int(idx))
    peeled_deltas = []             # nodes peeled at each depth (for the random arm)
    D0 = None
    slot = 0
    cur_dir = lvl0
    for depth in range(0, args.max_depth + 1):
        oracle.set_circuits(load_circuit_sets(cur_dir))
        agg = collect(cur_dir, args)
        bl, ba = oracle.base_stats()
        D, units, _, kinfo = factor_level(cur_dir, nodes, index, ksel, args.backbone_thr,
                                          args.min_share, args.min_size, topn=args.topn)
        if depth == 0:
            D0 = D
        rec = {"depth": depth, "iterN": slot, "n_excluded": len(F),
               "k": (kinfo["k"] if kinfo else None),
               "k_method": (kinfo["k_method"] if kinfo else None),
               "n_units": (len(units) if units else 0), "base_2afc": round(ba, 4), **agg}
        terminal = (units is None or not units or agg.get("feasibility", 0) < args.feas_eps)
        if not terminal:
            scores = score_units(oracle, D, units)
            peel = max(scores, key=lambda a: scores[a]["damage_loss"])
            rec["peeled"] = scores[peel]
            top = sorted(scores.values(), key=lambda s: -s["damage_loss"])[:4]
            print(f"  depth {depth}: feas={agg.get('feasibility')} size={agg.get('mean_circuit_size')} "
                  f"2afc={agg.get('mean_test_2afc')} | K={kinfo['k']}({kinfo['k_method']}) "
                  f"{len(units)} units; PEEL a{peel} dmg={scores[peel]['damage_loss']} "
                  f"supp={scores[peel]['support']} {scores[peel]['nodes']}", flush=True)
            rec["top_units"] = top
        else:
            rec["peeled"] = None
            print(f"  depth {depth}: feas={agg.get('feasibility')} -> TERMINAL "
                  f"({'no units' if not units else 'feasibility collapsed'})", flush=True)
        report["atom_trajectory"].append(rec)
        json.dump(report, open(exp_dir / "atom_peel.json", "w"), indent=2, default=str)
        if terminal:
            break

        # peel -> extend F -> re-prune the next level
        delta = cols_to_nodes(D, units[peel])
        peeled_deltas.append(delta)
        F |= set((k, int(i)) for k, i in delta)
        slot += 1
        cond = {"iterN": slot, "excluded": [list(x) for x in sorted(F)]}
        _, secs = reprune_condition(cond, args, exp_dir)
        print(f"    re-pruned depth {depth+1} ({len(F)} forbidden) in {secs/60:.1f}m", flush=True)
        cur_dir = exp_dir / f"iter{slot:02d}"

    coll = next((r["depth"] for r in report["atom_trajectory"]
                 if r.get("feasibility", 1) < args.feas_eps), None)
    report["summary"]["atom_collapse_depth"] = coll
    report["summary"]["atom_max_depth_reached"] = report["atom_trajectory"][-1]["depth"]
    json.dump(report, open(exp_dir / "atom_peel.json", "w"), indent=2, default=str)

    # ---- MATCHED-RANDOM ARM ----
    if not args.skip_random and peeled_deltas:
        print("\n=== MATCHED-RANDOM ARM ===", flush=True)
        atom_all = set((k, int(i)) for d in peeled_deltas for k, i in d)
        for r in range(args.rand_reps):
            rng = np.random.default_rng(args.rng_seed + 1 + r)
            Rset, cum_excl, rep_rec = set(), set(atom_all), []
            for depth, delta in enumerate(peeled_deltas, start=1):
                new = matched_random_extend(D0, delta, cum_excl | Rset, rng,
                                            args.backbone_thr, args.knn)
                Rset |= set((k, int(i)) for k, i in new)
                slot += 1
                cond = {"iterN": slot, "excluded": [list(x) for x in sorted(Rset)]}
                agg, secs = reprune_condition(cond, args, exp_dir)
                rr = {"depth": depth, "rep": r, "iterN": slot, "n_excluded": len(Rset), **agg}
                rep_rec.append(rr)
                print(f"  rand r{r} depth {depth}: feas={agg.get('feasibility')} "
                      f"size={agg.get('mean_circuit_size')} 2afc={agg.get('mean_test_2afc')} "
                      f"({len(Rset)} forbidden, {secs/60:.1f}m)", flush=True)
            report["random_trajectory"].append(rep_rec)
            json.dump(report, open(exp_dir / "atom_peel.json", "w"), indent=2, default=str)
        # mean random collapse depth
        depths = {}
        for rep in report["random_trajectory"]:
            for r_ in rep:
                depths.setdefault(r_["depth"], []).append(r_.get("feasibility", 1))
        rcoll = next((d for d in sorted(depths) if np.mean(depths[d]) < args.feas_eps), None)
        report["summary"]["random_collapse_depth"] = rcoll

    oracle.cleanup()
    json.dump(report, open(exp_dir / "atom_peel.json", "w"), indent=2, default=str)
    print(f"\nwrote {exp_dir / 'atom_peel.json'}", flush=True)
    print(f"SUMMARY: {report['summary']}", flush=True)
    make_plot(report, exp_dir / "atom_peel.png")


if __name__ == "__main__":
    main()
