#!/usr/bin/env python3
"""
Atom-excision universality experiment (built for the weight-sparse d1024 run).

WHY
---
The motif-dictionary work established (Jaccard-free) that for d1024 the MOST
UNIVERSAL circuit (the hard clustering's dominant cluster) is itself a
discriminative motif, recovered cleanly by one NMF atom (cos_disc ~0.7-0.94);
for dense d128 the universal circuit is just backbone. So d1024 is the clean
target for a CAUSAL test of the dictionary: knock out the universal motif's
nodes and ask whether the model can re-route, and at what cost.

WHAT THIS DOES
--------------
  1. Build a FIXED dictionary from one source iteration's circuits (NMF, the
     same decomposition motif_dictionary.py reports). No re-factoring -> atom
     identity is stable, so the knockout is interpretable.
  2. TARGET atom = the atom that best recovers the dominant hard cluster's
     DISCRIMINATIVE core (max cos on non-backbone nodes) -- i.e. the universal
     motif. (Fallback: highest-share specific-motif atom.)
  3. EXCISION set (dose-response): the target atom's OWN discriminative nodes --
     non-backbone (freq<thr) nodes the atom owns (argmax loading), ranked by
     loading. Doses = top-m for increasing m. We excise the atom-SPECIFIC nodes,
     never the shared backbone (that would trivially break everything).
  4. MATCHED-FREQUENCY RANDOM baseline: for each dose, R replicate sets of m
     random non-backbone nodes matched to the excised nodes' marginal frequency
     (excludes the atom's own nodes). Separates "this motif matters" from "you
     removed m nodes of similar frequency".
  5. Readouts are FUNCTIONAL, never node-overlap (see memory
     avoid-jaccard-for-circuit-comparison):
       - IMMEDIATE (no re-prune): ablate the excised nodes from each source
         circuit (tau pinning) and measure held-out task loss increase + 2AFC
         drop -> how much the existing universal circuit DEPENDS on the motif.
       - RECOVERABLE (re-prune): re-run pruning with the nodes forbidden
         (reuses the ORIGINAL worker verbatim) -> feasibility (success rate),
         recovery cost (circuit size), circuit/held-out loss, held-out 2AFC ->
         how well the model ROUTES AROUND the motif.
       The immediate-vs-recoverable gap = need vs route-around.
     The one structural readout is loadings-cosine: project re-pruned circuits
     onto the FIXED dictionary and compare loading profiles to control (does the
     re-routed circuit recruit a DIFFERENT atom?).

Each condition is one worker "iteration" (its own iterNN/ dir with
excluded_input.json), so the proven worker machinery is reused unchanged.
Resumable: a condition whose results already exist is skipped.
"""

import sys, os, json, argparse, time, subprocess, shutil, math
from pathlib import Path

import numpy as np

from sparse_pretrain.scripts.motif_dictionary import (
    global_node_space, load_iteration, describe_atoms,
)
from sparse_pretrain.scripts.universality_pruning_experiment import (
    circuit_nodes, _seed_chunks,
)

ORIG = str(Path(__file__).resolve().parent / "universality_pruning_experiment.py")
PASS_THROUGH = ["model", "task", "tokenizer", "target_loss", "num_steps", "batch_size",
                "eval_batches", "bisect_iters", "split_over", "split_seed", "test_frac",
                "heldout_fold", "name_pool"]


# ---------------------------------------------------------------------------
# Dictionary + target atom + excision sets
# ---------------------------------------------------------------------------
def read_parsimonious_k(src_dir, it, fallback=8):
    p = src_dir / "motif_dictionary.json"
    if p.exists():
        for r in json.load(open(p)).get("iterations", []):
            if r.get("iteration") == it and r.get("k_parsimonious"):
                return int(r["k_parsimonious"])
    return fallback


def build_dictionary(src_dir, it, seed=0, topn=12):
    nodes, index = global_node_space(src_dir)
    X, seeds = load_iteration(src_dir / f"iter{it:02d}", index, len(nodes))
    if len(X) == 0:
        raise SystemExit(f"no successful circuits in {src_dir}/iter{it:02d}")
    amask = X.sum(0) > 0
    anodes = [nodes[j] for j in np.where(amask)[0]]
    Xa = X[:, amask]
    freq = Xa.mean(0)
    k = read_parsimonious_k(src_dir, it)
    k = max(1, min(k, len(anodes), len(Xa) - 1))
    atoms, recon, H = describe_atoms(Xa, k, anodes, freq, seed, topn)
    return dict(anodes=anodes, freq=freq, H=H, atoms=atoms, k=k, n_circuits=len(Xa),
                col={f"{key}#{i}": c for c, (key, i) in enumerate(anodes)})


def pick_target_atom(D, src_dir, it, backbone_thr):
    """Atom best recovering the dominant hard cluster's discriminative core."""
    H, freq, anodes, col = D["H"], D["freq"], D["anodes"], D["col"]
    disc = freq < backbone_thr
    p = src_dir / f"iter{it:02d}" / "motif_summary.json"
    if p.exists():
        s = json.load(open(p))
        cores = s.get("cores") or {}
        sizes = {c: int(n) for c, n in (s.get("cluster_sizes") or {}).items()}
        if cores:
            dom = max(cores, key=lambda c: sizes.get(c, len(cores[c])))
            cc = np.array([col[x] for x in cores[dom] if x in col], dtype=int)
            core = np.zeros(len(anodes)); core[cc] = 1.0
            core_d = core * disc
            if core_d.sum() > 0 and np.linalg.norm(core_d) > 0:
                best_a, best_v = None, -1.0
                for a in range(H.shape[0]):
                    h = H[a] * disc
                    nu = np.linalg.norm(h)
                    v = float(h @ core_d / (nu * np.linalg.norm(core_d))) if nu > 0 else -1.0
                    if v > best_v:
                        best_a, best_v = a, v
                return best_a, {"method": "cos_disc_to_dominant_core", "cos_disc": round(best_v, 3),
                                "dominant_cluster": dom, "dominant_core_disc": int(core_d.sum())}
    spec = [a for a in D["atoms"] if a["role"] == "specific-motif"] or D["atoms"]
    return spec[0]["atom"], {"method": "fallback_top_share_specific"}


def atom_candidate_cols(D, atom, backbone_thr, load_frac=0.1):
    """Non-backbone (freq<thr) nodes the target atom loads on at >= load_frac of its
    PEAK loading, ranked by loading (descending). Argmax-ownership is unreliable here:
    at K~14 the NMF loadings are near-tied across atoms, so we rank by this atom's
    loading directly. Non-backbone filtering already spares the shared substrate, so
    high-loading non-backbone nodes are the motif's discriminative footprint."""
    H, freq, anodes = D["H"], D["freq"], D["anodes"]
    disc = freq < backbone_thr
    hmax = float(H[atom].max())
    if hmax <= 0:
        return []
    cand = [j for j in range(len(anodes)) if disc[j] and H[atom, j] >= load_frac * hmax]
    cand.sort(key=lambda j: H[atom, j], reverse=True)
    return cand


def matched_random_cols(D, atom_cand_all, m, rng, backbone_thr, knn=8):
    """m non-backbone nodes matched to the freq of the top-m atom nodes, drawn
    from outside the atom's candidate set."""
    freq, anodes = D["freq"], D["anodes"]
    forbidden = set(atom_cand_all)
    pool = [j for j in range(len(anodes)) if freq[j] < backbone_thr and j not in forbidden]
    chosen = []
    for j in atom_cand_all[:m]:
        f = freq[j]
        order = sorted([p for p in pool if p not in chosen], key=lambda p: abs(freq[p] - f))
        pick = order[:knn] or order
        if not pick:
            break
        chosen.append(int(rng.choice(pick)))
    return chosen


def cols_to_nodes(D, cols):
    return [[D["anodes"][j][0], int(D["anodes"][j][1])] for j in cols]


# ---------------------------------------------------------------------------
# Immediate ablation (no re-prune): tau-pin nodes off on the SOURCE circuits
# ---------------------------------------------------------------------------
def immediate_phase(args, D, conditions, exp_dir, src_dir):
    import torch
    from contextlib import nullcontext
    from transformers import AutoTokenizer
    from sparse_pretrain.src.pruning.config import PruningConfig
    from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
    from sparse_pretrain.src.pruning.run_pruning import load_model
    from sparse_pretrain.scripts.pronoun_split import tasks_from_split_info

    device = args.device
    print("  [immediate] loading model + held-out task ...", flush=True)
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
    print(f"  [immediate] {len(circuits)} source circuits; base held-out 2AFC={base_2afc:.3f} "
          f"loss={np.mean([l for l, _ in base]):.4f}", flush=True)

    out = []
    for cond in conditions:
        excl = set((k, int(i)) for k, i in cond["excluded"])
        dloss, d2afc, overlap = [], [], []
        for C, (bl, ba) in zip(circuits, base):
            force(C); off(excl)
            al, aa = ev()
            dloss.append(al - bl); d2afc.append(ba - aa)
            overlap.append(len(excl & C))
        out.append({**{k: cond[k] for k in ("kind", "dose", "rep", "iterN")},
                    "n_excised": len(excl), "mean_nodes_hit_per_circuit": round(float(np.mean(overlap)), 2),
                    "imm_mean_loss_increase": round(float(np.mean(dloss)), 4),
                    "imm_mean_2afc_drop": round(float(np.mean(d2afc)), 4)})
    del mm, model
    torch.cuda.empty_cache()
    return {"base_held_out_2afc": round(base_2afc, 4), "n_source_circuits": len(circuits),
            "conditions": out}


# ---------------------------------------------------------------------------
# Recoverable (re-prune) phase: reuse the ORIGINAL worker
# ---------------------------------------------------------------------------
def collect(it_dir, args):
    rs = []
    for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
        rp = it_dir / f"seed{seed}_result.json"
        if rp.exists():
            rs.append(json.load(open(rp)))
    ok = [r for r in rs if r.get("target_achieved")]
    agg = {"n": len(rs), "n_success": len(ok),
           "feasibility": round(len(ok) / max(len(rs), 1), 3)}

    def m(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    if ok:
        agg.update(mean_circuit_size=m("circuit_size"), mean_circuit_loss=m("circuit_loss"),
                   mean_test_loss=m("test_loss"), mean_test_2afc=m("test_2afc"),
                   generalize_rate=round(float(np.mean([bool(r.get("generalizes")) for r in ok])), 3))
    return agg


def reprune_condition(cond, args, exp_dir):
    N = cond["iterN"]
    it_dir = exp_dir / f"iter{N:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    have = list(it_dir.glob("seed*_result.json"))
    expected = args.num_seeds
    if len(have) >= expected:
        return collect(it_dir, args), 0.0  # resumed

    json.dump([list(x) for x in cond["excluded"]], open(it_dir / "excluded_input.json", "w"))
    for p in list(it_dir.glob("seed*_result.json")) + list(it_dir.glob("seed*_circuit.pt")):
        p.unlink()

    t0 = time.time()
    # Cap each worker's CPU intra-op math to 1 thread. With many data-parallel workers
    # the torch/MKL/OpenBLAS default (one thread PER CORE, PER PROCESS) oversubscribes
    # the box (num_workers x num_cores threads), thrashing the scheduler and starving
    # the GPU (this dominated the runtime; MPS shares the GPU but not the CPU). The
    # GPU work is unaffected (CUDA), so circuits are unchanged. The orchestrator keeps
    # its default thread count, so its NMF re-factoring stays parallel.
    wenv = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    procs = []
    for j, (s, e) in enumerate(_seed_chunks(args.num_seeds, args.num_workers)):
        cmd = [sys.executable, ORIG, "--worker", "--iter", str(N),
               "--seed-start", str(s + args.seed_offset), "--seed-end", str(e + args.seed_offset),
               "--model", args.model, "--tokenizer", args.tokenizer, "--task", args.task,
               "--target-loss", str(args.target_loss), "--num-steps", str(args.num_steps),
               "--batch-size", str(args.batch_size), "--eval-batches", str(args.eval_batches),
               "--bisect-iters", str(args.bisect_iters), "--exp-dir", str(exp_dir),
               "--device", args.device, "--split-over", args.split_over,
               "--split-seed", str(args.split_seed), "--test-frac", str(args.test_frac),
               "--heldout-fold", str(args.heldout_fold), "--name-pool", str(args.name_pool)]
        wlog = open(it_dir / f"worker{j}.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=wlog, stderr=subprocess.STDOUT, env=wenv), wlog))
    for p, wlog in procs:
        p.wait(); wlog.close()
    return collect(it_dir, args), time.time() - t0


# ---------------------------------------------------------------------------
# Loadings-cosine: do re-routed circuits recruit a DIFFERENT atom? (structural,
# weight-based -- NOT node Jaccard)
# ---------------------------------------------------------------------------
def loadings_profile(D, it_dir):
    """Mean NNLS loading vector (onto the fixed dictionary) over a condition's
    successful re-pruned circuits."""
    try:
        from scipy.optimize import nnls
    except Exception:
        return None
    import torch
    H, col = D["H"], D["col"]
    A = H.T  # (n_active, k): x ~ A @ w
    ws = []
    for p in sorted(it_dir.glob("seed*_circuit.pt")):
        cm = torch.load(p, map_location="cpu", weights_only=True)
        x = np.zeros(len(D["anodes"]))
        for (key, idx) in circuit_nodes(cm):
            c = col.get(f"{key}#{idx}")
            if c is not None:
                x[c] = 1.0
        if x.sum() == 0:
            continue
        w, _ = nnls(A, x)
        ws.append(w)
    if not ws:
        return None
    return np.mean(ws, axis=0)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return round(float(u @ v / (nu * nv)), 3) if nu > 0 and nv > 0 else None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(report, out_png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skipping plot: {e})"); return

    def series(items, key, kind):
        pts = {}
        for r in items:
            if r.get("kind") != kind or r.get(key) is None:
                continue
            pts.setdefault(r["dose"], []).append(r[key])
        xs = sorted(pts)
        return xs, [float(np.mean(pts[d])) for d in xs]

    imm = report["immediate"]["conditions"]
    rep = report["reprune"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for kind, mk in [("atom", "o-"), ("random", "s--")]:
        xs, ys = series(imm, "imm_mean_2afc_drop", kind)
        if xs:
            ax.plot(xs, ys, mk, label=kind)
    ax.set_xlabel("dose (# nodes excised)"); ax.set_ylabel("held-out 2AFC drop")
    ax.set_title("IMMEDIATE (no re-prune)\ndependence of universal circuit on the motif")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for kind, mk in [("atom", "o-"), ("random", "s--")]:
        xs, ys = series(rep, "feasibility", kind)
        if xs:
            ax.plot(xs, ys, mk, label=kind)
    ax.set_xlabel("dose (# nodes forbidden)"); ax.set_ylabel("re-prune feasibility (success rate)")
    ax.set_title("RECOVERABLE (re-prune)\ncan the model route around the motif?")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    for kind, mk in [("atom", "o-"), ("random", "s--")]:
        xs, ys = series(rep, "mean_circuit_size", kind)
        if xs:
            ax.plot(xs, ys, mk, label=kind)
    ax.set_xlabel("dose (# nodes forbidden)"); ax.set_ylabel("recovered circuit size (nodes)")
    ax.set_title("RECOVERY COST\nnodes needed to recover behaviour")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True,
                    help="Existing dictionary run (has iter*/seed*_circuit.pt, motif_summary.json, "
                         "hparams.json, split_info.json, run_args.json). Pruning config is read from "
                         "its run_args.json so circuits are discovered byte-identically.")
    ap.add_argument("--exp-dir", required=True, help="Fresh output dir for this experiment.")
    ap.add_argument("--iter", type=int, default=0, help="Source iteration to build the dictionary on.")
    ap.add_argument("--doses", default="1,2,4,8", help="Comma list of excision sizes (capped to "
                    "#atom-specific nodes; the max is always appended as 'all').")
    ap.add_argument("--rand-reps", type=int, default=3, dest="rand_reps",
                    help="Matched-frequency random replicate sets per dose.")
    ap.add_argument("--backbone-thr", type=float, default=0.6, dest="backbone_thr")
    ap.add_argument("--load-frac", type=float, default=0.1, dest="load_frac",
                    help="A node is in the target atom's footprint if its loading is >= this "
                         "fraction of the atom's peak loading.")
    ap.add_argument("--target-atom", type=int, default=-1, dest="target_atom",
                    help="-1 = auto (atom recovering the dominant disc core).")
    ap.add_argument("--num-seeds", type=int, default=40, dest="num_seeds")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset")
    ap.add_argument("--num-workers", type=int, default=8, dest="num_workers")
    ap.add_argument("--rng-seed", type=int, default=0, dest="rng_seed")
    ap.add_argument("--skip-immediate", action="store_true", dest="skip_immediate")
    ap.add_argument("--skip-reprune", action="store_true", dest="skip_reprune")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny end-to-end check: num_seeds=4, doses=2, rand_reps=1.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # pruning identity comes from the source run -> byte-identical circuit discovery
    src_args = json.load(open(src_dir / "run_args.json"))
    for k in PASS_THROUGH:
        setattr(args, k, src_args[k])
    args.name_pool = str(Path(args.name_pool).resolve())
    if args.split_over != "names_templates":
        raise SystemExit("this experiment expects the names_templates split (held-out 2AFC readout).")

    if args.smoke:
        args.num_seeds, args.rand_reps, args.doses, args.num_steps = 4, 1, "2", 200

    # workers need these in exp_dir; copy from source so no CARBS / re-split
    for fn in ("hparams.json", "split_info.json"):
        if not (exp_dir / fn).exists():
            shutil.copy(src_dir / fn, exp_dir / fn)
    json.dump(vars(args), open(exp_dir / "run_args.json", "w"), indent=2, default=str)

    # --- dictionary + target atom + excision sets ---
    print(f"Building fixed dictionary from {src_dir.name}/iter{args.iter:02d} ...", flush=True)
    D = build_dictionary(src_dir, args.iter)
    if args.target_atom >= 0:
        atom, sel = args.target_atom, {"method": "user"}
    else:
        atom, sel = pick_target_atom(D, src_dir, args.iter, args.backbone_thr)
    cand = atom_candidate_cols(D, atom, args.backbone_thr, args.load_frac)
    print(f"  K={D['k']} atoms over {len(D['anodes'])} active nodes, {D['n_circuits']} circuits", flush=True)
    print(f"  TARGET atom {atom} ({sel}); {len(cand)} atom-specific non-backbone nodes", flush=True)
    if not cand:
        raise SystemExit("target atom has no specific non-backbone nodes to excise.")

    doses = sorted({d for d in (int(x) for x in args.doses.split(",")) if 1 <= d <= len(cand)} | {len(cand)})
    rng = np.random.default_rng(args.rng_seed)
    conditions = [{"kind": "control", "dose": 0, "rep": 0, "iterN": 0, "excluded": []}]
    N = 1
    for m in doses:
        conditions.append({"kind": "atom", "dose": m, "rep": 0, "iterN": N,
                           "excluded": cols_to_nodes(D, cand[:m])}); N += 1
        for r in range(args.rand_reps):
            conditions.append({"kind": "random", "dose": m, "rep": r, "iterN": N,
                               "excluded": cols_to_nodes(D, matched_random_cols(
                                   D, cand, m, rng, args.backbone_thr))}); N += 1
    print(f"  doses={doses}; {len(conditions)} conditions "
          f"(1 control + {len(doses)} atom + {len(doses)*args.rand_reps} random)", flush=True)

    # control re-prune results = source iteration (copy in so collect() works uniformly)
    ctrl_dir = exp_dir / "iter00"; ctrl_dir.mkdir(exist_ok=True)
    if not list(ctrl_dir.glob("seed*_result.json")):
        for p in (src_dir / f"iter{args.iter:02d}").glob("seed*_result.json"):
            shutil.copy(p, ctrl_dir / p.name)
        for p in (src_dir / f"iter{args.iter:02d}").glob("seed*_circuit.pt"):
            shutil.copy(p, ctrl_dir / p.name)

    target_info = {"atom": atom, "selection": sel, "load_frac": args.load_frac, "n_candidates": len(cand),
                   "candidate_nodes": [f"{D['anodes'][j][0]}#{D['anodes'][j][1]}" for j in cand],
                   "candidate_loadings": [round(float(D["H"][atom, j]), 4) for j in cand],
                   "candidate_freq": [round(float(D["freq"][j]), 3) for j in cand]}
    report = {"src_dir": str(src_dir), "exp_dir": str(exp_dir), "iter": args.iter,
              "k": D["k"], "backbone_thr": args.backbone_thr, "num_seeds": args.num_seeds,
              "doses": doses, "rand_reps": args.rand_reps, "target": target_info,
              "conditions": [{k: c[k] for k in ("kind", "dose", "rep", "iterN")} |
                             {"n_excised": len(c["excluded"])} for c in conditions]}

    # --- immediate phase ---
    if not args.skip_immediate:
        print("\n=== IMMEDIATE ablation phase ===", flush=True)
        report["immediate"] = immediate_phase(args, D, conditions, exp_dir, src_dir)
        json.dump(report, open(exp_dir / "atom_excision.json", "w"), indent=2, default=str)

    # --- recoverable (re-prune) phase ---
    if not args.skip_reprune:
        print("\n=== RECOVERABLE (re-prune) phase ===", flush=True)
        rep_rows, load_ctrl = [], None
        for ci, cond in enumerate(conditions):
            agg, secs = reprune_condition(cond, args, exp_dir)
            lp = loadings_profile(D, exp_dir / f"iter{cond['iterN']:02d}")
            if cond["kind"] == "control":
                load_ctrl = lp
            agg_full = {**{k: cond[k] for k in ("kind", "dose", "rep", "iterN")}, **agg,
                        "loadings_cosine_to_control": cos(lp, load_ctrl) if (lp is not None and load_ctrl is not None) else None}
            rep_rows.append(agg_full)
            tag = f"{cond['kind']}@{cond['dose']}" + (f".r{cond['rep']}" if cond["kind"] == "random" else "")
            print(f"  [{ci+1}/{len(conditions)}] {tag:<14} feas={agg.get('feasibility')} "
                  f"size={agg.get('mean_circuit_size')} 2afc={agg.get('mean_test_2afc')} "
                  f"loadcos={agg_full['loadings_cosine_to_control']} ({secs/60:.1f}m)", flush=True)
            report["reprune"] = rep_rows
            json.dump(report, open(exp_dir / "atom_excision.json", "w"), indent=2, default=str)

    json.dump(report, open(exp_dir / "atom_excision.json", "w"), indent=2, default=str)
    print(f"\nwrote {exp_dir / 'atom_excision.json'}", flush=True)
    if "immediate" in report and "reprune" in report:
        make_plot(report, exp_dir / "atom_excision.png")


if __name__ == "__main__":
    main()
