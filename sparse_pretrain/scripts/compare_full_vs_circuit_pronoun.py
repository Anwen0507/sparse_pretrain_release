#!/usr/bin/env python3
"""Compare the FULL dense model vs the discovered pruned CIRCUITS on dummy_pronoun.

Full model  = all 2816 nodes active.
Circuit     = one seed's discretized boolean mask (~63 active, rest zero-ablated),
              from iteration 0 (no exclusions); 99 of them.
All evaluated on the SAME held-out superval batches. Metrics from compute_task_loss:
  logit_diff (correct-incorrect), accuracy (full-vocab top-1), task_loss (full-vocab CE).
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json, glob
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
device = "cuda"; torch.set_grad_enabled(False)
NB, SEEDS = 30, (0, 1, 2)

model, _ = load_model("jacobcd52/ss_d128_f1", device)
tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
pc = PruningConfig(device=device, ablation_type="zero", mask_token_embeds=False, batch_size=64, seq_length=0)
mm = MaskedSparseGPT(model, pc); mm.to(device)
autocast = torch.autocast("cuda", dtype=torch.bfloat16)

# fixed held-out batches (superval = not used in training/discretization)
batches = []
for sd in SEEDS:
    task = get_task("dummy_pronoun", tok, seed=sd, split="superval")
    for _ in range(NB // len(SEEDS) + 1):
        batches.append([x.to(device) for x in task.generate_batch(batch_size=64, max_length=0)])
batches = batches[:NB]


def eval_on():
    ld = acc = ce = 0.0
    for b in batches:
        with autocast:
            _, m = mm.compute_task_loss(*b)
        ld += m["logit_diff"]; acc += m["accuracy"]; ce += m["task_loss"]
    n = len(batches); return ld / n, acc / n, ce / n


# full model
st = mm.get_mask_state()
for k in st: st[k].fill_(1.0)
mm.load_mask_state(st)
f_ld, f_acc, f_ce = eval_on()
total_nodes = mm.masks.get_total_nodes()

# circuits (iteration 0)
C = {"ld": [], "acc": [], "ce": [], "size": []}
for f in sorted(glob.glob(str(EXP / "iter00" / "seed*_circuit.pt"))):
    cm = torch.load(f, map_location=device, weights_only=True)
    st = mm.get_mask_state()
    for k in st: st[k].copy_(cm[k].to(device) * 2 - 1)   # 1->+1 (keep), 0->-1 (zero-ablate)
    mm.load_mask_state(st)
    ld, acc, ce = eval_on()
    C["ld"].append(ld); C["acc"].append(acc); C["ce"].append(ce)
    C["size"].append(int(sum(int(v.sum()) for v in cm.values())))
for k in C: C[k] = np.array(C[k], float)

print(f"{'':>22} {'nodes':>7} {'logit_diff':>11} {'accuracy':>9} {'CE loss':>9}")
print(f"{'FULL model':>22} {total_nodes:>7} {f_ld:>11.2f} {f_acc:>9.1%} {f_ce:>9.3f}")
print(f"{'pruned circuit (n=99)':>22} {C['size'].mean():>7.0f} "
      f"{C['ld'].mean():>6.2f}±{C['ld'].std():<4.2f} {C['acc'].mean():>8.1%} {C['ce'].mean():>8.3f}")
print(f"{'   circuit range':>22} {'':>7} [{C['ld'].min():.2f},{C['ld'].max():.2f}]  "
      f"[{C['acc'].min():.0%},{C['acc'].max():.0%}]  [{C['ce'].min():.3f},{C['ce'].max():.3f}]")
print(f"\ncircuits beating full model:  logit_diff {100*(C['ld']>f_ld).mean():.0f}%   "
      f"accuracy {100*(C['acc']>f_acc).mean():.0f}%   lower CE {100*(C['ce']<f_ce).mean():.0f}%")
print(f"circuits at/under target CE 0.15 on held-out superval: {100*(C['ce']<=0.15).mean():.0f}%")

# plot
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for a, key, lab, fv in [(ax[0], "ld", "logit-diff (correct−incorrect)", f_ld),
                        (ax[1], "acc", "accuracy (full-vocab top-1)", f_acc),
                        (ax[2], "ce", "CE loss (full-vocab)", f_ce)]:
    a.hist(C[key], bins=18, color="C2", alpha=.8, edgecolor="k")
    a.axvline(fv, color="C0", lw=2.5, label=f"full model = {fv:.2f}" if key != "acc" else f"full model = {fv:.0%}")
    a.axvline(C[key].mean(), color="C3", ls="--", lw=2, label=f"circuit mean = {C[key].mean():.2f}" if key != "acc" else f"circuit mean = {C[key].mean():.0%}")
    if key == "ce": a.axvline(0.15, color="k", ls=":", lw=1.5, label="target 0.15")
    a.set_xlabel(lab); a.set_ylabel("# circuits"); a.legend(fontsize=8)
fig.suptitle("Full dense model (2816 nodes) vs pruned pronoun circuits (~63 nodes) — held-out superval", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(EXP / "full_vs_circuit_pronoun.png", dpi=140)
json.dump({"full": {"nodes": total_nodes, "logit_diff": f_ld, "accuracy": f_acc, "ce": f_ce},
           "circuit": {k: [float(C[k].mean()), float(C[k].std()), float(C[k].min()), float(C[k].max())] for k in C}},
          open(EXP / "full_vs_circuit_pronoun.json", "w"), indent=2)
print("\nsaved", EXP / "full_vs_circuit_pronoun.png")
