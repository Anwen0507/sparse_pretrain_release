#!/usr/bin/env python3
"""Recompute accuracy as 2-alternative forced choice (2AFC): correct_logit > incorrect_logit,
for the full-vs-circuit comparison and the cross-task ablation. (compute_task_loss returns this
as binary_accuracy when use_binary_loss=True; logit_diff is unchanged.)

Default (no args): legacy behavior on the ss_d128_f1_pronoun experiment -- superval TEMPLATES
with ALL task names (random batches), iter00 circuits, plus the core-node cross-task ablation.

--exp-dir pointing at a names_templates run (the dir contains split_info.json, written by
universality_pruning_experiment.py --split-over names_templates): full-vs-circuit 2AFC is
instead computed on SUPERVAL_TEMPLATES x the HELD-OUT 20% names -- one deterministic batch
over EVERY held-out (template, name) pair -- for every iteration's circuits, and saved to
<exp-dir>/accuracy_2afc_heldout.json. bf16 autocast is fine here: 2AFC/logit_diff only
compare logits (calibrated probabilities would need fp32).
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json, argparse
from pathlib import Path
from collections import Counter
import numpy as np, torch
from transformers import AutoTokenizer
from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--exp-dir", dest="exp_dir",
                default=str(OUTPUTS / "ss_d128_f1_pronoun"))
ap.add_argument("--model", default="jacobcd52/ss_d128_f1")
ap.add_argument("--iters", default="auto",
                help="Which iterNN circuit sets to evaluate: 'auto' (legacy exp: iter00 only; "
                     "names_templates exp: all), 'all', or a comma list like '0,3'.")
args = ap.parse_args()

EXP = Path(args.exp_dir)
device = "cuda"; torch.set_grad_enabled(False)
model, _ = load_model(args.model, device)
tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
pc = PruningConfig(device=device, ablation_type="zero", mask_token_embeds=False, batch_size=64, seq_length=0)
mm = MaskedSparseGPT(model, pc); mm.to(device)
autocast = torch.autocast("cuda", dtype=torch.bfloat16)
key_dim = {k: mm.masks.masks[k].num_nodes for k in mm.masks.masks}

# names_templates experiments publish their exact name/template split here
si_path = EXP / "split_info.json"
split_info = json.load(open(si_path)) if si_path.exists() else None


def set_off(nodes):
    st = mm.get_mask_state()
    for k in st: st[k].fill_(1.0)
    for loc, idx in nodes: st[loc][idx] = -1.0
    mm.load_mask_state(st)


def apply_circuit(cm):
    st = mm.get_mask_state()
    for k in st: st[k].copy_(cm[k].to(device) * 2 - 1)
    mm.load_mask_state(st)


def batches_for(task, split):
    bs = []
    for sd in (0, 1, 2):
        t = get_task(task, tok, seed=sd, split=split)
        for _ in range(11):
            bs.append([x.to(device) for x in t.generate_batch(batch_size=64, max_length=0)])
    return bs[:30]


def ev(batches):  # -> (logit_diff, 2AFC accuracy)
    ld = afc = 0.0
    for b in batches:
        with autocast:
            _, m = mm.compute_task_loss(*b, use_binary_loss=True)
        ld += m["logit_diff"]; afc += m["binary_accuracy"]
    n = len(batches); return ld / n, afc / n


def iter_dirs():
    have = sorted(p for p in EXP.glob("iter*") if p.is_dir() and any(p.glob("seed*_circuit.pt")))
    if args.iters == "all" or (args.iters == "auto" and split_info is not None):
        return have
    if args.iters == "auto":
        return [EXP / "iter00"]
    return [EXP / f"iter{int(x):02d}" for x in args.iters.split(",")]


def eval_circuits(itd, batches):
    rows, seeds = [], []
    for fp in sorted(itd.glob("seed*_circuit.pt")):
        apply_circuit(torch.load(fp, map_location=device, weights_only=True))
        rows.append(ev(batches)); seeds.append(fp.stem.replace("_circuit", ""))
    return np.array(rows), seeds


if split_info is not None:
    # ---- names_templates: 2AFC on SUPERVAL_TEMPLATES x HELD-OUT names (every pair, once) ----
    from sparse_pretrain.scripts.pronoun_split import PronounSplitTask
    pairs = [(t, n) for t in split_info["superval_templates"] for n in split_info["test_names"]]
    ht = PronounSplitTask(tok, pairs, label="superval20", male_names=set(split_info["male_names"]))
    heldout = [tuple(x.to(device) for x in ht.full_batch())]
    print(f"=== FULL vs CIRCUIT (held-out 2AFC: {len(split_info['superval_templates'])} superval "
          f"templates x {len(split_info['test_names'])} held-out names = {len(pairs)} pairs, "
          f"fold {split_info['heldout_fold']}) ===")
    set_off([]); f = ev(heldout)
    print(f"  full model:  logit_diff={f[0]:.2f}   2AFC acc={f[1]:.1%}")
    iters_out = {}
    for itd in iter_dirs():
        C, seeds = eval_circuits(itd, heldout)
        if not len(C):
            continue
        iters_out[itd.name] = {
            "n_circuits": len(C),
            "afc_mean": float(C[:, 1].mean()), "afc_min": float(C[:, 1].min()),
            "afc_max": float(C[:, 1].max()), "logit_diff_mean": float(C[:, 0].mean()),
            "per_seed": {s: {"logit_diff": float(r[0]), "afc": float(r[1])}
                         for s, r in zip(seeds, C)},
        }
        print(f"  {itd.name} ({len(C):>3} circuits): 2AFC acc={C[:,1].mean():.1%}  "
              f"[{C[:,1].min():.0%},{C[:,1].max():.0%}]   logit_diff={C[:,0].mean():.2f}")
    json.dump({"eval": "superval_templates x heldout names (full batch, deterministic)",
               "heldout_fold": split_info["heldout_fold"], "n_pairs": len(pairs),
               "test_names": split_info["test_names"],
               "full_logit_diff": f[0], "full_2afc": f[1], "iterations": iters_out},
              open(EXP / "accuracy_2afc_heldout.json", "w"), indent=2)
    print("\nsaved", EXP / "accuracy_2afc_heldout.json")
else:
    # ---- legacy: superval templates, ALL names, iter00; plus core cross-task ablation ----
    print("=== FULL vs CIRCUIT (dummy_pronoun, held-out superval) ===")
    sv = batches_for("dummy_pronoun", "superval")
    set_off([]); f = ev(sv)
    C, _ = eval_circuits(iter_dirs()[0], sv)
    print(f"  full model:  logit_diff={f[0]:.2f}   2AFC acc={f[1]:.1%}")
    print(f"  circuit({len(C)}): logit_diff={C[:,0].mean():.2f}   2AFC acc={C[:,1].mean():.1%}  [{C[:,1].min():.0%},{C[:,1].max():.0%}]")

    out = {}
    core_path = EXP / "core_nodes.json"
    if core_path.exists():
        core = [(n["location"], n["index"]) for n in json.load(open(core_path))["nodes"]]
        core_by_key = Counter(loc for loc, _ in core)
        print("\n=== CROSS-TASK ablation (val), 2AFC accuracy: base / core / random(15) ===")
        for t in ["dummy_pronoun", "dummy_tense", "dummy_article", "ioi_relaxed"]:
            vb = batches_for(t, "val")
            set_off([]); b = ev(vb)
            set_off(core); c = ev(vb)
            r = []
            for rr in range(15):
                rng = np.random.default_rng(rr); off = []
                for k, n in core_by_key.items():
                    for i in rng.choice(key_dim[k], size=n, replace=False): off.append((k, int(i)))
                set_off(off); r.append(ev(vb)[1])
            r = np.array(r)
            out[t] = dict(base=b[1], core=c[1], rand_mean=float(r.mean()), rand_std=float(r.std()),
                          frac_rand_le_core=float((r <= c[1]).mean()))
            print(f"  {t:>15}: base={b[1]:.0%}  core={c[1]:.0%}  rand={r.mean():.0%}±{r.std():.0%}   "
                  f"(rand at/under core: {(r<=c[1]).mean():.0%})")
    else:
        print(f"\n(no {core_path.name}; skipping cross-task ablation)")
    json.dump({"full_vs_circuit": {"full_2afc": f[1], "circuit_2afc_mean": float(C[:,1].mean()),
                                   "circuit_2afc_min": float(C[:,1].min())}, "crosstask_2afc": out},
              open(EXP / "accuracy_2afc.json", "w"), indent=2)
    print("\nsaved", EXP / "accuracy_2afc.json")
