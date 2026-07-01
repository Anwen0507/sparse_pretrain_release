#!/usr/bin/env python3
"""Exclude the d1024 peel's terminal universal kernel and re-prune.

The v2 atom-peel collapsed d1024 to a 15-node consensus kernel (all freq=1.0 at the
terminal level) -- 10 iter-0 backbone nodes + 5 nodes promoted to obligatory once
their discriminative alternatives were peeled. This asks: is that kernel NECESSARY,
or can the fresh model route around even it? Forbid the kernel on the full model and
re-prune (100 seeds), vs a frequency-matched random 15-node control. Also reports how
much the kernel-excluded circuits RECRUIT previously-peeled discriminative nodes
(falling back from kernel -> motifs). Reuses the original worker via reprune_condition.
"""
import sys, json, argparse, shutil
from pathlib import Path
import numpy as np
from sparse_pretrain.scripts.motif_dictionary import global_node_space, load_iteration
from sparse_pretrain.scripts.atom_excision import reprune_condition, collect
from sparse_pretrain.scripts.universality_pruning_experiment import circuit_nodes

PASS_THROUGH = ["model", "task", "tokenizer", "target_loss", "num_steps", "batch_size",
                "eval_batches", "bisect_iters", "split_over", "split_seed", "test_frac",
                "heldout_fold", "name_pool"]


def matched_random(active_cols, freq, target_cols, m, exclude, rng, knn=6):
    """m iter-0-active nodes whose freq matches the target nodes', excluding `exclude`."""
    pool = [j for j in active_cols if j not in exclude]
    chosen = []
    for j in target_cols[:m]:
        f = freq[j]
        order = sorted([p for p in pool if p not in chosen], key=lambda p: abs(freq[p] - f))
        pick = order[:knn] or order
        if not pick:
            break
        chosen.append(int(rng.choice(pick)))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--peel-dir", required=True)
    ap.add_argument("--kernel-iter", type=int, default=34)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--num-seeds", type=int, default=100, dest="num_seeds")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset")
    ap.add_argument("--num-workers", type=int, default=16, dest="num_workers")
    ap.add_argument("--rand-reps", type=int, default=2, dest="rand_reps")
    ap.add_argument("--rng-seed", type=int, default=0, dest="rng_seed")
    ap.add_argument("--backbone-thr", type=float, default=0.6, dest="backbone_thr")
    ap.add_argument("--mode", default="split", choices=["split", "leaveoneout"],
                    help="split = backbone/promoted/full; leaveoneout = drop each core node alone.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src, peel = Path(args.src_dir).resolve(), Path(args.peel_dir).resolve()
    exp = Path(args.exp_dir).resolve(); exp.mkdir(parents=True, exist_ok=True)
    for k in PASS_THROUGH:
        setattr(args, k, json.load(open(src / "run_args.json"))[k])
    args.name_pool = str(Path(args.name_pool).resolve())
    for fn in ("hparams.json", "split_info.json"):
        if not (exp / fn).exists():
            shutil.copy(src / fn, exp / fn)

    nodes, index = global_node_space(src)
    Xk, _ = load_iteration(peel / f"iter{args.kernel_iter:02d}", index, len(nodes))
    kcols = list(np.where(Xk.sum(0) > 0)[0])
    kernel = [[nodes[j][0], int(nodes[j][1])] for j in kcols]
    X0, _ = load_iteration(src / "iter00", index, len(nodes))
    freq = X0.mean(0); active = list(np.where(X0.sum(0) > 0)[0])
    # previously-peeled discriminative set = the kernel level's cumulative forbidden set
    peeled = set()
    ep = peel / f"iter{args.kernel_iter:02d}" / "excluded_input.json"
    if ep.exists():
        peeled = {(k, int(i)) for k, i in json.load(open(ep))}
    print(f"kernel = {len(kernel)} nodes; previously-peeled discriminative nodes forbidden at "
          f"depth {args.kernel_iter}: {len(peeled)}", flush=True)

    # All conditions forbid the depth-34 cumulative set (198) as a base.
    base = [list(x) for x in sorted(peeled)]
    if args.mode == "leaveoneout":
        # drop each individual core node alone (on top of the 198): which nodes are
        # individually necessary (articulation points) vs substitutable while the rest stay.
        conditions = [{"kind": "control_depth34", "iterN": 0, "excluded": base}]
        for i, kn in enumerate(kernel):
            tag = "bb" if freq[kcols[i]] >= args.backbone_thr else "pr"
            conditions.append({"kind": f"drop_{tag}_{kn[0]}#{kn[1]}", "iterN": 1 + i,
                               "excluded": base + [kn]})
        print(f"  leave-one-out: control + {len(kernel)} single-node drops", flush=True)
    else:  # split: backbone / promoted / full
        bb = [[nodes[j][0], int(nodes[j][1])] for j in kcols if freq[j] >= args.backbone_thr]
        pr = [[nodes[j][0], int(nodes[j][1])] for j in kcols if freq[j] < args.backbone_thr]
        print(f"  kernel split: {len(bb)} backbone (freq>={args.backbone_thr}) + {len(pr)} promoted", flush=True)
        conditions = [{"kind": "control_depth34", "iterN": 0, "excluded": base},
                      {"kind": f"backbone_removed_{len(bb)}", "iterN": 1, "excluded": base + bb},
                      {"kind": f"promoted_removed_{len(pr)}", "iterN": 2, "excluded": base + pr},
                      {"kind": "kernel_removed_15", "iterN": 3, "excluded": base + kernel}]

    # control = copy the peel's depth-34 level itself (already re-pruned with the 198 forbidden)
    ctrl = exp / "iter00"; ctrl.mkdir(exist_ok=True)
    if not list(ctrl.glob("seed*_result.json")):
        for p in (peel / f"iter{args.kernel_iter:02d}").glob("seed*_result.json"):
            shutil.copy(p, ctrl / p.name)
        for p in (peel / f"iter{args.kernel_iter:02d}").glob("seed*_circuit.pt"):
            shutil.copy(p, ctrl / p.name)

    report = {"src": str(src), "peel": str(peel), "kernel_iter": args.kernel_iter,
              "kernel_nodes": [f"{k}#{i}" for k, i in kernel], "conditions": []}
    for cond in conditions:
        agg, secs = reprune_condition(cond, args, exp)
        rec = {**{k: cond.get(k) for k in ("kind", "rep", "iterN")}, **agg}
        # node composition of the re-pruned circuits: how much is the kernel vs NOVEL
        # (nodes outside both the 198 peeled motifs and the 15 kernel = genuinely new substrate)
        it_dir = exp / f"iter{cond['iterN']:02d}"
        circs = [circuit_nodes(__import__("torch").load(p, map_location="cpu", weights_only=True))
                 for p in sorted(it_dir.glob("seed*_circuit.pt"))]
        kset = {(k, int(i)) for k, i in kernel}; bset = set(peeled)
        if circs:
            fk = [len(c & kset) / max(len(c), 1) for c in circs]
            fn = [sum(1 for nd in c if nd not in kset and nd not in bset) / max(len(c), 1) for c in circs]
            rec["frac_kernel"] = round(float(np.mean(fk)), 3)
            rec["frac_novel"] = round(float(np.mean(fn)), 3)
        report["conditions"].append(rec)
        print(f"  [{cond['kind']:>16}] feas={agg.get('feasibility')} size={agg.get('mean_circuit_size')} "
              f"2afc={agg.get('mean_test_2afc')} test_loss={agg.get('mean_test_loss')} "
              f"frac_kernel={rec.get('frac_kernel')} frac_novel={rec.get('frac_novel')} ({secs/60:.1f}m)", flush=True)
        json.dump(report, open(exp / "exclude_kernel.json", "w"), indent=2, default=str)
    print(f"\nwrote {exp / 'exclude_kernel.json'}", flush=True)


if __name__ == "__main__":
    main()
