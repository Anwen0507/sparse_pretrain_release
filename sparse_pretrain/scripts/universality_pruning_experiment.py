#!/usr/bin/env python3
"""
Iterative universality pruning experiment on a DENSE weight-sparse-transformer.

Protocol (per user spec):
  1. Run circuit pruning (boolean node mask, Gao et al. 2025) with N random seeds.
     Each seed trains a continuous mask, then discretizes (bisection) to the smallest
     node set achieving `target_loss`. Seeds whose trained mask reaches target loss
     yield a "circuit".
  2. Among the successful circuits, compute:
       - Jaccard similarity of NODES (pairwise mean/std/min/max),
       - Jaccard similarity of WEIGHTED EDGES (pairwise; unweighted and |W|-weighted),
       - node UNIVERSALITY = fraction of successful circuits containing each node.
  3. Force-exclude every node tied at the maximum universality ("all rank-1 nodes"),
     adding them to a cumulative exclusion set.
  4. Re-run pruning with N seeds, with the excluded nodes held off.
  5. Repeat until an iteration finds 0 circuits that achieve target loss.

DENSE-model edge definition
---------------------------
The model `jacobcd52/ss_d128_f1` is dense (all weights nonzero), so a circuit is
defined by its active NODES. An EDGE is a learned linear-map weight connecting two
active nodes; it is present iff BOTH endpoints are active. The six edge families per
layer L are:
    attn_in  -> attn_q      (W_Q  = c_attn.weight[0:HD])
    attn_in  -> attn_k      (W_K  = c_attn.weight[HD:2HD])
    attn_in  -> attn_v      (W_V  = c_attn.weight[2HD:3HD])
    attn_v   -> attn_out    (W_O  = attn.c_proj.weight)
    mlp_in   -> mlp_neuron  (W_fc = mlp.c_fc.weight)
    mlp_neuron -> mlp_out   (W_proj = mlp.c_proj.weight)
Edge weight = |W[dst_idx, src_idx]| (shared across circuits since the base model is
fixed). Residual/identity connections are not "weighted" edges and are excluded.
"""

from sparse_pretrain.paths import OUTPUTS, NAME_POOLS
import sys, os, json, argparse, time, gc, subprocess, math
from pathlib import Path
from itertools import combinations
from collections import Counter
from contextlib import nullcontext


import numpy as np
import torch
from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.trainer import PruningTrainer
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.discretize import evaluate_at_k

CENTER_HPARAMS = {"k_coef": 1e-3, "weight_decay": 1e-3, "lr": 1e-2, "beta2": 0.95, "heaviside_temp": 1.0}


# ---------------------------------------------------------------------------
# Node space + weighted edges
# ---------------------------------------------------------------------------
def _dim_for_loc(loc, mc):
    if loc in ("attn_in", "attn_out", "mlp_in", "mlp_out"):
        return mc.d_model
    if loc in ("attn_q", "attn_k", "attn_v"):
        return mc.n_heads * mc.d_head
    if loc == "mlp_neuron":
        return mc.d_mlp
    raise ValueError(loc)


class NodeSpace:
    """Stable global integer id for every (location_key, index) node."""
    def __init__(self, model, mask_locations):
        mc = model.config
        self.keys, self.offset, self.dim = [], {}, {}
        off = 0
        for L in range(mc.n_layer):
            for loc in mask_locations:
                key = f"layer{L}_{loc}"
                d = _dim_for_loc(loc, mc)
                self.keys.append(key); self.offset[key] = off; self.dim[key] = d
                off += d
        self.total = off

    def gid(self, key, idx):
        return self.offset[key] + idx


def build_weight_maps(model, mask_locations):
    """List of (src_key, dst_key, W_cpu) with convention W[dst_idx, src_idx]."""
    mc = model.config
    HD = mc.n_heads * mc.d_head
    maps = []
    for L in range(mc.n_layer):
        blk = model.blocks[L]
        Wqkv = blk.attn.c_attn.weight.detach().float().cpu()   # (3HD, d_model)
        Wo = blk.attn.c_proj.weight.detach().float().cpu()     # (d_model, HD)
        Wfc = blk.mlp.c_fc.weight.detach().float().cpu()       # (d_mlp, d_model)
        Wproj = blk.mlp.c_proj.weight.detach().float().cpu()   # (d_model, d_mlp)
        maps += [
            (f"layer{L}_attn_in", f"layer{L}_attn_q", Wqkv[0:HD, :]),
            (f"layer{L}_attn_in", f"layer{L}_attn_k", Wqkv[HD:2 * HD, :]),
            (f"layer{L}_attn_in", f"layer{L}_attn_v", Wqkv[2 * HD:3 * HD, :]),
            (f"layer{L}_attn_v", f"layer{L}_attn_out", Wo),
            (f"layer{L}_mlp_in", f"layer{L}_mlp_neuron", Wfc),
            (f"layer{L}_mlp_neuron", f"layer{L}_mlp_out", Wproj),
        ]
    # keep only families whose endpoints are both masked locations
    valid = set(f"layer{L}_{loc}" for L in range(mc.n_layer) for loc in mask_locations)
    return [(s, d, W) for (s, d, W) in maps if s in valid and d in valid]


def circuit_nodes(circuit_mask):
    """Set of (key, idx) active nodes."""
    nodes = set()
    for key, m in circuit_mask.items():
        for idx in torch.where(m.bool())[0].cpu().tolist():
            nodes.add((key, idx))
    return nodes


def circuit_edges(circuit_mask, weight_maps, node_space):
    """Dict {edge_gid:int -> |w|:float} for edges with both endpoints active."""
    active = {key: torch.where(m.bool())[0].cpu().tolist() for key, m in circuit_mask.items()}
    edges = {}
    N = node_space.total
    for src_key, dst_key, W in weight_maps:
        s_idx = active.get(src_key, [])
        d_idx = active.get(dst_key, [])
        if not s_idx or not d_idx:
            continue
        sub = W[d_idx][:, s_idx].abs()  # (n_dst, n_src)
        s_off = node_space.offset[src_key]
        d_off = node_space.offset[dst_key]
        sub_list = sub.tolist()
        for a, di in enumerate(d_idx):
            dgid = d_off + di
            row = sub_list[a]
            for b, si in enumerate(s_idx):
                eid = (s_off + si) * N + dgid
                edges[eid] = row[b]
    return edges


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------
def _stats(vals):
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def node_jaccard_stats(node_sets):
    vals = []
    for i, j in combinations(range(len(node_sets)), 2):
        a, b = node_sets[i], node_sets[j]
        u = len(a | b)
        if u:
            vals.append(len(a & b) / u)
    return _stats(vals), vals


def edge_jaccard_stats(edge_dicts):
    """Return (unweighted_stats, weighted_stats, unweighted_vals, weighted_vals)."""
    unw, wt = [], []
    for i, j in combinations(range(len(edge_dicts)), 2):
        ea, eb = edge_dicts[i], edge_dicts[j]
        ka, kb = set(ea), set(eb)
        union = ka | kb
        if not union:
            continue
        inter = ka & kb
        unw.append(len(inter) / len(union))
        num = sum(min(ea[k], eb[k]) for k in inter)               # min == max for shared edges
        den = sum(max(ea.get(k, 0.0), eb.get(k, 0.0)) for k in union)
        wt.append(num / den if den > 0 else 0.0)
    return _stats(unw), _stats(wt), unw, wt


def node_universality(node_sets):
    """{(key,idx) -> fraction of circuits containing it}."""
    n = len(node_sets)
    c = Counter()
    for s in node_sets:
        c.update(s)
    return {node: cnt / n for node, cnt in c.items()}


def rank1_nodes(freq):
    """All nodes tied at the maximum universality."""
    if not freq:
        return [], 0.0
    mx = max(freq.values())
    return [node for node, f in freq.items() if f == mx], mx


# ---------------------------------------------------------------------------
# Node exclusion
# ---------------------------------------------------------------------------
def apply_node_exclusion(masked_model, excluded):
    """Pin tau=-1 for excluded (key,idx) nodes and re-pin after every clamp step."""
    by_key = {}
    for key, idx in excluded:
        by_key.setdefault(key, []).append(idx)
    with torch.no_grad():
        for key, idxs in by_key.items():
            masked_model.masks.masks[key].tau[idxs] = -1.0
    if not by_key:
        return
    orig_clamp = masked_model.clamp_mask_parameters

    def clamp_and_pin():
        orig_clamp()
        with torch.no_grad():
            for key, idxs in by_key.items():
                masked_model.masks.masks[key].tau[idxs] = -1.0

    masked_model.clamp_mask_parameters = clamp_and_pin


# ---------------------------------------------------------------------------
# Single seed
# ---------------------------------------------------------------------------
def run_single_seed(seed, hparams, model, tokenizer, excluded, args, weight_maps, node_space):
    device = args.device
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pc = PruningConfig(
        k_coef=hparams["k_coef"], weight_decay=hparams["weight_decay"], lr=hparams["lr"],
        beta2=hparams["beta2"], heaviside_temp=hparams["heaviside_temp"],
        init_noise_scale=0.01, init_noise_bias=0.1, lr_warmup_frac=0.0,
        num_steps=args.num_steps, batch_size=args.batch_size, seq_length=0, device=device,
        log_every=10**9, target_loss=args.target_loss, ablation_type="zero",
        mask_token_embeds=False,
    )
    # Tasks. split_over="none" -> legacy fixed train/val TEMPLATE split (get_task).
    # split_over="names_templates" -> name holdout CROSSED with the fixed template split:
    #   mask training on TRAIN_TEMPLATES x 80% names, success gate + bisection on
    #   VAL_TEMPLATES x the SAME 80% names, and SUPERVAL_TEMPLATES x the held-out 20%
    #   names touched ONLY to report generalization (loss + 2AFC; never gates success or
    #   picks k -> no leakage). Names/folds come from the clean balanced pool; the fold is
    #   FIXED across model seeds (identical held-out set for every circuit).
    # Other split_over values: random prune/test split over pooled templates; prune +
    # select (mask training, success gate, AND size bisection) all run on the 80% PRUNE
    # set. task_seed=seed so per-run sampling still varies. See scripts/pronoun_split.py.
    test_task = None
    if args.split_over == "none":
        train_task = get_task(args.task, tokenizer, seed=seed, split="train")
        val_task = get_task(args.task, tokenizer, seed=seed, split="val")
    elif args.split_over == "names_templates":
        # tasks come from the exp_dir's launch-time split_info.json SNAPSHOT, not the live
        # pool file -- the pool may be regenerated while a run is in flight
        from sparse_pretrain.scripts.pronoun_split import tasks_from_split_info
        with open(Path(args.exp_dir) / "split_info.json") as f:
            split_info = json.load(f)
        train_task, val_task, test_task = tasks_from_split_info(
            tokenizer, split_info, task_seed=seed)
    else:
        from sparse_pretrain.scripts.pronoun_split import make_pronoun_split
        train_task, test_task, _ = make_pronoun_split(
            tokenizer, test_frac=args.test_frac, split_over=args.split_over,
            split_seed=args.split_seed, task_seed=seed)
        val_task = train_task  # success gate + bisection both run on the 80% prune set

    mm = MaskedSparseGPT(model, pc); mm.to(device)
    apply_node_exclusion(mm, excluded)
    trainer = PruningTrainer(masked_model=mm, task=train_task, val_task=val_task,
                             config=pc, use_wandb=False)

    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with autocast:
        trainer.train(num_steps=args.num_steps, show_progress=False,
                      histogram_every=0, pareto_probe_every=0)

    num_active = mm.masks.get_total_active_nodes()
    total_nodes = mm.masks.get_total_nodes()
    orig = mm.get_mask_state()

    eb = args.eval_batches
    with autocast:
        loss_all = evaluate_at_k(mm, val_task, num_active, pc, num_batches=eb)
    mm.load_mask_state(orig)

    target_achieved = bool(loss_all <= args.target_loss)
    best_k, best_loss, circuit_mask = num_active, float(loss_all), None
    test_loss, test_2afc = None, None
    if target_achieved:
        low, high = 1, num_active
        for _ in range(args.bisect_iters):
            if low > high:
                break
            mid = (low + high) // 2
            with autocast:
                l = evaluate_at_k(mm, val_task, mid, pc, num_batches=eb)
            mm.load_mask_state(orig)
            if l <= args.target_loss:
                best_k, best_loss = mid, float(l)
                high = mid - 1
            else:
                low = mid + 1
        mm.masks.keep_top_k(best_k)
        circuit_mask = {k: v.detach().cpu().clone() for k, v in mm.get_circuit_mask().items()}
        # Held-out generalization: evaluate the CHOSEN size-best_k circuit on the 20%
        # test split, over EVERY held-out pair (deterministic, noise-free). Reporting
        # ONLY -- never affects target_achieved or best_k (no leakage into selection).
        if test_task is not None:
            pos, neg, corr, inc, ep = test_task.full_batch()
            pos, neg = pos.to(device), neg.to(device)
            corr, inc, ep = corr.to(device), inc.to(device), ep.to(device)
            mm.eval()
            with autocast, torch.no_grad():
                _, tm = mm.compute_task_loss(pos, neg, corr, inc, ep)
                # same forward with binary loss for 2AFC accuracy (correct > incorrect);
                # bf16 is fine for logit comparisons (only calibrated probs need fp32)
                _, tb = mm.compute_task_loss(pos, neg, corr, inc, ep, use_binary_loss=True)
            test_loss = float(tm["task_loss"])
            test_2afc = float(tb["binary_accuracy"])
        mm.load_mask_state(orig)

    res = {
        "seed": seed, "target_achieved": target_achieved,
        "loss_at_all_active": float(loss_all), "num_active_after_training": int(num_active),
        "total_nodes": int(total_nodes), "circuit_size": int(best_k) if target_achieved else None,
        "circuit_loss": float(best_loss) if target_achieved else None,
        "test_loss": test_loss, "test_2afc": test_2afc,
        "generalizes": (bool(test_loss <= args.target_loss) if test_loss is not None else None),
        "circuit_mask": circuit_mask,
    }
    del mm, trainer
    return res


# ---------------------------------------------------------------------------
# Parallel workers (seeds are independent -> data-parallel across processes)
# ---------------------------------------------------------------------------
def _seed_chunks(num_seeds, num_workers):
    per = math.ceil(num_seeds / max(1, num_workers))
    chunks, s = [], 0
    while s < num_seeds:
        chunks.append((s, min(s + per, num_seeds)))
        s += per
    return chunks


def worker_main(args):
    """Run a contiguous range of seeds for one iteration; write per-seed result + circuit."""
    device = args.device
    model, _ = load_model(args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    exp_dir = Path(args.exp_dir)
    with open(exp_dir / "hparams.json") as f:
        hparams = json.load(f)
    it_dir = exp_dir / f"iter{args.iter:02d}"
    with open(it_dir / "excluded_input.json") as f:
        excluded = set(tuple(x) for x in json.load(f))

    for seed in range(args.seed_start, args.seed_end):
        try:
            r = run_single_seed(seed, hparams, model, tokenizer, excluded, args, None, None)
        except Exception as e:
            torch.cuda.empty_cache(); gc.collect()
            r = {"seed": seed, "target_achieved": False, "error": f"{type(e).__name__}: {e}"}
        if r.get("target_achieved"):
            torch.save(r["circuit_mask"], it_dir / f"seed{seed}_circuit.pt")
            print(f"  [w] seed {seed:>3}: PASS size={r['circuit_size']:>3} loss={r['circuit_loss']:.4f}", flush=True)
        else:
            print(f"  [w] seed {seed:>3}: {r.get('error', 'fail loss=%.4f' % r.get('loss_at_all_active', float('nan')))}", flush=True)
        with open(it_dir / f"seed{seed}_result.json", "w") as f:
            json.dump({k: v for k, v in r.items() if k != "circuit_mask"}, f)
        if (seed - args.seed_start) % 5 == 0:
            torch.cuda.empty_cache(); gc.collect()


# ---------------------------------------------------------------------------
# One iteration (orchestrator: launch workers, collect, analyze)
# ---------------------------------------------------------------------------
def run_iteration(it, excluded, args, weight_maps, node_space, exp_dir):
    it_dir = exp_dir / f"iter{it:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\nITERATION {it}  | excluded so far: {len(excluded)} nodes | seeds: {args.num_seeds} | workers: {args.num_workers}\n{'='*70}", flush=True)

    # publish the exclusion set for workers; clear stale per-seed outputs
    with open(it_dir / "excluded_input.json", "w") as f:
        json.dump([list(x) for x in sorted(excluded)], f)
    for p in list(it_dir.glob("seed*_result.json")) + list(it_dir.glob("seed*_circuit.pt")):
        p.unlink()

    # launch worker subprocesses over contiguous seed chunks
    t0 = time.time()
    procs = []
    for j, (s, e) in enumerate(_seed_chunks(args.num_seeds, args.num_workers)):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
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
    for p, wlog, rng in procs:
        rc = p.wait(); wlog.close()
        if rc != 0:
            fails.append((rng, rc))
    if fails:
        print(f"  WARNING: {len(fails)} worker(s) exited nonzero: {fails}", flush=True)
    elapsed = time.time() - t0

    # collect per-seed results written by workers
    results = []
    for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
        rp = it_dir / f"seed{seed}_result.json"
        r = json.load(open(rp)) if rp.exists() else {"seed": seed, "target_achieved": False, "error": "no result file"}
        if r.get("target_achieved"):
            r["circuit_mask"] = torch.load(it_dir / f"seed{seed}_circuit.pt", map_location="cpu", weights_only=True)
        results.append(r)
    successful = [r for r in results if r.get("target_achieved")]
    seed_summaries = [{k: v for k, v in r.items() if k != "circuit_mask"} for r in results]
    n_succ = len(successful)
    print(f"\n  -> {n_succ}/{args.num_seeds} circuits reached target loss  ({elapsed/60:.1f} min, {args.num_workers} workers)", flush=True)

    summary = {
        "iteration": it, "num_seeds": args.num_seeds, "n_success": n_succ,
        "success_rate": n_succ / args.num_seeds, "elapsed_sec": elapsed,
        "excluded_before": len(excluded), "seed_results": seed_summaries,
    }

    if n_succ >= 1:
        node_sets = [circuit_nodes(r["circuit_mask"]) for r in successful]
        edge_dicts = [circuit_edges(r["circuit_mask"], weight_maps, node_space) for r in successful]
        sizes = [len(s) for s in node_sets]
        esizes = [len(e) for e in edge_dicts]
        njac, _ = node_jaccard_stats(node_sets)
        eunw, ewt, _, _ = edge_jaccard_stats(edge_dicts)
        freq = node_universality(node_sets)
        r1, mx = rank1_nodes(freq)
        top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:15]

        summary.update({
            "circuit_size_nodes": _stats(sizes),
            "circuit_size_edges": _stats(esizes),
            "node_jaccard": njac,
            "edge_jaccard_unweighted": eunw,
            "edge_jaccard_weighted": ewt,
            "max_universality": mx,
            "n_rank1_nodes": len(r1),
            "rank1_nodes": [f"{k}#{i}" for (k, i) in sorted(r1)],
            "top_universal_nodes": [{"node": f"{k}#{i}", "freq": f} for (k, i), f in top],
        })
        if n_succ >= 2:
            print(f"  node Jaccard:           mean={njac['mean']:.3f}  (min={njac['min']:.3f} max={njac['max']:.3f})", flush=True)
            print(f"  edge Jaccard (unweight):mean={eunw['mean']:.3f}", flush=True)
            print(f"  edge Jaccard (|W| wt):  mean={ewt['mean']:.3f}", flush=True)
        else:  # pairwise stats are undefined for a single circuit (_stats([]) -> None)
            print("  Jaccard stats:          n/a (single circuit); ALL its nodes are rank-1", flush=True)
        print(f"  max universality={mx:.3f} -> {len(r1)} rank-1 node(s) to exclude", flush=True)
        for (k, i), f in top[:8]:
            print(f"      {f:5.2f}  {k}#{i}", flush=True)
        # Held-out generalization aggregates (absent on legacy split_over="none" runs).
        sel_losses = [r["circuit_loss"] for r in successful if r.get("circuit_loss") is not None]
        test_losses = [r["test_loss"] for r in successful if r.get("test_loss") is not None]
        test_afcs = [r["test_2afc"] for r in successful if r.get("test_2afc") is not None]
        if test_losses:
            n_gen = sum(bool(r.get("generalizes")) for r in successful)
            summary.update({"select_loss": _stats(sel_losses),
                            "heldout_test_loss": _stats(test_losses),
                            "n_generalize": n_gen, "generalize_rate": n_gen / n_succ})
            ht = summary["heldout_test_loss"]
            print(f"  held-out test loss:     mean={ht['mean']:.4f}  (min={ht['min']:.4f} max={ht['max']:.4f})", flush=True)
            print(f"  generalize @ target {args.target_loss}: {n_gen}/{n_succ} ({100*n_gen/n_succ:.0f}%)", flush=True)
        if test_afcs:
            summary["heldout_test_2afc"] = _stats(test_afcs)
            ha = summary["heldout_test_2afc"]
            print(f"  held-out test 2AFC:     mean={ha['mean']:.1%}  (min={ha['min']:.1%} max={ha['max']:.1%})", flush=True)
        new_excl = set((k, i) for (k, i) in r1)
    else:
        summary.update({"rank1_nodes": [], "n_rank1_nodes": 0})
        new_excl = set()

    with open(it_dir / "iteration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary, new_excl


# ---------------------------------------------------------------------------
# Hparam bootstrap
# ---------------------------------------------------------------------------
def get_hparams(args, exp_dir):
    hp_path = exp_dir / "hparams.json"
    if hp_path.exists():
        with open(hp_path) as f:
            hp = json.load(f)
        print(f"Loaded cached hparams: {hp}", flush=True)
        return hp
    if args.skip_carbs:
        hp = dict(CENTER_HPARAMS)
        print(f"Using center hparams (skip-carbs): {hp}", flush=True)
    else:
        print("Running CARBS sweep to select hyperparameters...", flush=True)
        from sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
        cfg = CleanSweepConfig(
            model_path=args.model, task_name=args.task, num_runs=args.carbs_runs,
            parallel_suggestions=1, num_steps=args.num_steps, target_loss=args.target_loss,
            init_noise_scale=0.01, init_noise_bias=0.1, lr_warmup_frac=0.0,
            use_wandb=False, ablation_type="zero", mask_token_embeds=False,
            k_coef_center=1e-3, output_base_dir=str(exp_dir / "carbs"), device=args.device,
        )
        run_carbs_sweep(cfg)
        ckpt = exp_dir / "carbs" / f"{args.model.split('/')[-1]}_zero_noembed" / "best_checkpoint" / "hparams.json"
        if ckpt.exists():
            with open(ckpt) as f:
                hp = {k: v for k, v in json.load(f).items() if k != "suggestion_uuid"}
            print(f"CARBS selected hparams: {hp}", flush=True)
        else:
            hp = dict(CENTER_HPARAMS)
            print(f"CARBS found no target-achieving run; falling back to center hparams: {hp}", flush=True)
    with open(hp_path, "w") as f:
        json.dump(hp, f, indent=2)
    return hp


# ---------------------------------------------------------------------------
# State / checkpointing
# ---------------------------------------------------------------------------
def load_state(exp_dir):
    p = exp_dir / "state.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"next_iter": 0, "excluded": [], "history": [], "exhausted": False}


def save_state(exp_dir, state):
    with open(exp_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jacobcd52/ss_d128_f1")
    ap.add_argument("--task", default="dummy_pronoun")
    ap.add_argument("--tokenizer", default="SimpleStories/SimpleStories-1.25M")
    ap.add_argument("--target-loss", type=float, default=0.15, dest="target_loss")
    ap.add_argument("--num-seeds", type=int, default=100, dest="num_seeds")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset",
                    help="Run seeds [offset, offset+num_seeds) instead of [0, num_seeds). "
                         "Use a disjoint range (e.g. 100) for an independent replicate.")
    ap.add_argument("--num-steps", type=int, default=2000, dest="num_steps")
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    ap.add_argument("--eval-batches", type=int, default=5, dest="eval_batches")
    ap.add_argument("--bisect-iters", type=int, default=15, dest="bisect_iters")
    ap.add_argument("--carbs-runs", type=int, default=32, dest="carbs_runs")
    ap.add_argument("--skip-carbs", action="store_true", dest="skip_carbs")
    ap.add_argument("--num-workers", type=int, default=10, dest="num_workers",
                    help="Parallel seed worker processes per iteration.")
    ap.add_argument("--max-iters", type=int, default=1000, dest="max_iters",
                    help="Stop after this many NEW iterations this invocation (pilot uses a small value).")
    ap.add_argument("--exp-dir", default=str(OUTPUTS / "ss_d128_f1_pronoun"), dest="exp_dir")
    ap.add_argument("--device", default="cuda")
    # held-out prune/test split (pronoun task only). "none" keeps the legacy template split.
    ap.add_argument("--split-over", default="none",
                    choices=["none", "examples", "names", "templates", "names_templates"],
                    dest="split_over",
                    help="Random prune/test split axis. names_templates=clean-pool name holdout "
                         "crossed with the fixed template split (train on TRAIN_TEMPLATES x 80%% "
                         "names, gate/bisect on VAL_TEMPLATES x same names, report on "
                         "SUPERVAL_TEMPLATES x held-out names); names=hold out whole names over "
                         "pooled templates; templates=hold out whole contexts; examples=hold out "
                         "(template,name) pairs; none=legacy fixed template split.")
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed",
                    help="Seed for the prune/test PARTITION. FIXED across all model seeds so the "
                         "held-out test set is identical for every circuit.")
    ap.add_argument("--test-frac", type=float, default=0.2, dest="test_frac",
                    help="Fraction of examples held out as the test set (0.2 -> 80/20).")
    # names_templates mode only:
    ap.add_argument("--heldout-fold", type=int, default=0, dest="heldout_fold",
                    help="[names_templates] Which clean-pool fold (0..4) is the held-out 20%% "
                         "name set. FIXED across all model seeds.")
    ap.add_argument("--name-pool", dest="name_pool",
                    default=str(NAME_POOLS / "name_pool_cast15.json"),
                    help="[names_templates] Name pool JSON with folds (build_cast_name_pool.py).")
    # worker-mode (internal): run a seed range for one iteration
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--iter", type=int, default=0, dest="iter")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--seed-end", type=int, default=0, dest="seed_end")
    args = ap.parse_args()
    args.name_pool = str(Path(args.name_pool).resolve())

    if args.worker:
        worker_main(args)
        return

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    # refuse to resume an exp_dir whose task/split identity differs (stale state.json
    # exclusions + a different prune/test split would silently corrupt the experiment)
    prev_path = exp_dir / "run_args.json"
    if prev_path.exists():
        with open(prev_path) as f:
            prev = json.load(f)
        for k in ("model", "task", "split_over", "split_seed", "test_frac",
                  "heldout_fold", "name_pool", "seed_offset"):
            if k in prev and prev[k] != getattr(args, k):
                raise SystemExit(f"exp-dir {exp_dir} was created with {k}={prev[k]!r} but this "
                                 f"run uses {getattr(args, k)!r}; use a fresh --exp-dir.")
    with open(prev_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # names_templates: snapshot the exact name/template split at LAUNCH. Workers and the
    # downstream eval (recompute_2afc.py) read ONLY this snapshot, so the name-pool file
    # can keep evolving without touching an in-flight experiment. On resume, refuse to
    # continue if the pool file has changed away from the snapshot.
    if args.split_over == "names_templates":
        from sparse_pretrain.scripts.pronoun_split import make_pronoun_fold_split
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _, _, _, split_info = make_pronoun_fold_split(
            tok, heldout_fold=args.heldout_fold, pool_path=args.name_pool)
        si_path = exp_dir / "split_info.json"
        if si_path.exists():
            with open(si_path) as f:
                prev_si = json.load(f)
            if (prev_si["train_names"], prev_si["test_names"]) != (
                    split_info["train_names"], split_info["test_names"]):
                raise SystemExit(f"{si_path} disagrees with the current pool {args.name_pool} "
                                 f"(pool regenerated since launch?); use a fresh --exp-dir.")
            split_info = prev_si
        else:
            with open(si_path, "w") as f:
                json.dump(split_info, f, indent=2)
        print(f"names_templates split: {split_info['n_train_names']} train names x "
              f"{len(split_info['train_templates'])}/{len(split_info['val_templates'])} "
              f"train/val templates; held out fold {args.heldout_fold}: "
              f"{split_info['n_test_names']} names x {len(split_info['superval_templates'])} "
              f"superval templates ({split_info['n_test_pairs']} test pairs)", flush=True)

    print(f"Loading model {args.model} (orchestrator, for edge/node analysis) ...", flush=True)
    model, _ = load_model(args.model, args.device)
    mask_locations = PruningConfig().mask_locations
    node_space = NodeSpace(model, mask_locations)
    weight_maps = build_weight_maps(model, mask_locations)  # moved to CPU inside
    del model; torch.cuda.empty_cache()
    print(f"Total maskable nodes: {node_space.total} | edge families: {len(weight_maps)}", flush=True)

    get_hparams(args, exp_dir)  # ensure exp_dir/hparams.json exists for workers
    state = load_state(exp_dir)
    excluded = set(tuple(x) for x in state["excluded"])

    if state.get("exhausted"):
        print("Experiment already marked exhausted. Nothing to do.", flush=True)
        return

    done_this_run = 0
    while done_this_run < args.max_iters:
        it = state["next_iter"]
        summary, new_excl = run_iteration(it, excluded, args, weight_maps, node_space, exp_dir)
        state["history"].append({k: v for k, v in summary.items() if k != "seed_results"})

        if summary["n_success"] == 0:
            print(f"\n*** EXHAUSTED at iteration {it}: 0 circuits achieved target loss. ***", flush=True)
            state["exhausted"] = True
            save_state(exp_dir, state)
            break

        excluded |= new_excl
        state["excluded"] = [list(x) for x in sorted(excluded)]
        state["next_iter"] = it + 1
        save_state(exp_dir, state)
        done_this_run += 1

        if len(excluded) >= node_space.total:
            print("\n*** All nodes excluded. Stopping. ***", flush=True)
            state["exhausted"] = True
            save_state(exp_dir, state)
            break

    print(f"\nDone. Ran {done_this_run} iteration(s) this invocation. "
          f"next_iter={state['next_iter']}, cumulative excluded={len(excluded)}, "
          f"exhausted={state.get('exhausted', False)}", flush=True)
    print(f"State + per-iteration summaries in: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
