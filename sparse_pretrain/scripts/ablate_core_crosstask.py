#!/usr/bin/env python3
"""Ablate the 84-node pronoun core in the dense model; test cross-task generalization.

The raw dense model does NOT hit the 0.15 CE target (that's a trained-circuit property),
so we measure BEHAVIOR on each task: logit-diff (correct - incorrect) and accuracy
(does the model still prefer the right answer). A core-SPECIFIC effect = core ablation
collapses task performance far more than a matched random-84 ablation.

Conditions (same fixed batches): baseline (full) / core-ablated / random-84 (15 draws).
dummy_pronoun is the positive control.
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
NB, SEEDS, NRAND = 30, (0, 1, 2), 15
TASKS = ["dummy_pronoun", "dummy_tense", "dummy_article", "ioi_relaxed"]   # quote dropped: model at chance
device = "cuda"; torch.set_grad_enabled(False)

model, _ = load_model("jacobcd52/ss_d128_f1", device)
tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
core = [(n["location"], n["index"]) for n in json.load(open(EXP / "core_nodes.json"))["nodes"]]
core_by_key = Counter(loc for loc, _ in core)

pc = PruningConfig(device=device, ablation_type="zero", mask_token_embeds=False, batch_size=64, seq_length=0)
mm = MaskedSparseGPT(model, pc); mm.to(device)
autocast = torch.autocast("cuda", dtype=torch.bfloat16)
key_dim = {k: mm.masks.masks[k].num_nodes for k in mm.masks.masks}


def set_ablation(nodes):
    st = mm.get_mask_state()
    for k in st: st[k].fill_(1.0)
    for loc, idx in nodes: st[loc][idx] = -1.0
    mm.load_mask_state(st)


def fixed_batches(task_name):
    bs = []
    for sd in SEEDS:
        task = get_task(task_name, tok, seed=sd, split="val")
        for _ in range(NB // len(SEEDS) + 1):
            bs.append([x.to(device) for x in task.generate_batch(batch_size=64, max_length=0)])
    return bs[:NB]


def eval_on(batches):
    ld = acc = 0.0
    for b in batches:
        with autocast:
            _, m = mm.compute_task_loss(*b)
        ld += m["logit_diff"]; acc += m["accuracy"]
    return ld / len(batches), acc / len(batches)


def random_nodes(rng):
    out = []
    for k, c in core_by_key.items():
        for i in rng.choice(key_dim[k], size=c, replace=False):
            out.append((k, int(i)))
    return out


rows = []
print(f"{'task':>15} | {'logit_diff (corr-incorr)':^34} | {'accuracy':^24}")
print(f"{'':>15} | {'base':>8} {'core':>8} {'rand(µ±σ)':>15} | {'base':>7} {'core':>7} {'rand':>7}")
for t in TASKS:
    fb = fixed_batches(t)
    set_ablation([]);   b_ld, b_ac = eval_on(fb)
    set_ablation(core); c_ld, c_ac = eval_on(fb)
    r_ld, r_ac = [], []
    for r in range(NRAND):
        set_ablation(random_nodes(np.random.default_rng(r))); x = eval_on(fb); r_ld.append(x[0]); r_ac.append(x[1])
    r_ld, r_ac = np.array(r_ld), np.array(r_ac)
    # how core compares to the random distribution (lower logit_diff = more damage)
    frac_rand_worse = float((r_ld <= c_ld).mean())   # random draws at least as damaging as core
    rows.append(dict(task=t, base_ld=b_ld, core_ld=c_ld, rand_ld_mean=float(r_ld.mean()), rand_ld_std=float(r_ld.std()),
                     base_acc=b_ac, core_acc=c_ac, rand_acc_mean=float(r_ac.mean()), frac_rand_worse=frac_rand_worse))
    print(f"{t:>15} | {b_ld:>8.2f} {c_ld:>8.2f} {r_ld.mean():>7.2f}±{r_ld.std():<6.2f} | "
          f"{b_ac:>6.0%} {c_ac:>6.0%} {r_ac.mean():>6.0%}   (rand>=core damage: {frac_rand_worse:.0%})")

json.dump(dict(core_size=len(core), rows=rows), open(EXP / "crosstask_ablation.json", "w"), indent=2)

# ---- plot: logit-diff retained (higher = task still works) ----
fig, ax = plt.subplots(figsize=(10.5, 6))
x = np.arange(len(rows)); w = 0.27
ax.bar(x - w, [r["base_ld"] for r in rows], w, label="baseline (full model)", color="C0")
ax.bar(x, [r["core_ld"] for r in rows], w, label="core-ablated (−84 core)", color="C3")
ax.bar(x + w, [r["rand_ld_mean"] for r in rows], w, yerr=[r["rand_ld_std"] for r in rows],
       capsize=4, label="random-84 control (matched)", color="C7")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels([r["task"] for r in rows], rotation=12)
ax.set_ylabel("logit-diff: correct − incorrect  (higher = task still works)")
ax.set_title("Ablating the 84-node pronoun core across tasks (ss_d128_f1)\n"
             "core-specific = red collapses vs blue AND vs grey")
ax.legend()
fig.tight_layout(); fig.savefig(EXP / "crosstask_ablation.png", dpi=140)
print("\nsaved", EXP / "crosstask_ablation.png", "and crosstask_ablation.json")
