#!/usr/bin/env python3
"""Causal cross-check of the DLA shares (crosstask_attn_vs_mlp_dla.py).

Zero-ablate each sublayer's residual write (L0-attn, L0-mlp, L1-attn, L1-mlp) one at a time
and measure 2AFC accuracy + logit gap per task. Tests whether the DLA-dominant sublayer is
causally necessary:
  - pronoun/ioi_mixed (attention-dominant): L1-attn ablation should break it; L1-mlp should not.
  - ioi_relaxed/strict (MLP-dominant):      L1-mlp ablation should break it.
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
torch.set_grad_enabled(False); device = "cuda"
model, _ = load_model("jacobcd52/ss_d128_f1", device)
tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

MODULES = {"L0attn": model.blocks[0].attn, "L0mlp": model.blocks[0].mlp,
           "L1attn": model.blocks[1].attn, "L1mlp": model.blocks[1].mlp}
def zero_hook(m, a, o):
    if isinstance(o, tuple): return (torch.zeros_like(o[0]),) + tuple(o[1:])
    return torch.zeros_like(o)

TASKS = ["dummy_pronoun", "ioi_mixed", "dummy_tense", "ioi_relaxed", "ioi_strict"]
CONDS = ["baseline", "L0attn", "L0mlp", "L1attn", "L1mlp"]

def eval_task(name, cond, n_seeds=4):
    handle = None if cond == "baseline" else MODULES[cond].register_forward_hook(zero_hook)
    afc, gap = [], []
    for sd in range(n_seeds):
        b = get_task(name, tok, seed=sd, split="val").generate_batch(64, 0)
        pos, _, ct, it, ep = [x.to(device) for x in b]
        bi = torch.arange(pos.shape[0], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(pos)
        g = (logits[bi, ep, ct] - logits[bi, ep, it]).float()
        gap.append(g.cpu().numpy()); afc.append((g > 0).float().cpu().numpy())
    if handle: handle.remove()
    return float(np.concatenate(afc).mean()), float(np.concatenate(gap).mean())

R = {}
print(f"{'task':>15} | " + " | ".join(f"{c:>14}" for c in CONDS))
print("-" * 95)
for t in TASKS:
    R[t] = {}
    cells = []
    for c in CONDS:
        a, g = eval_task(t, c); R[t][c] = {"afc": a, "gap": g}
        cells.append(f"{a:>4.0%}/{g:>+6.2f}")
    print(f"{t:>15} | " + " | ".join(f"{x:>14}" for x in cells))
print("\n(cell = 2AFC accuracy / mean logit gap;  baseline vs each sublayer zero-ablated)")
print("'breaks' = 2AFC drops toward/below 50% chance.")
json.dump(R, open(EXP / "crosstask_ablation_check.json", "w"), indent=2)
print(f"saved {EXP/'crosstask_ablation_check.json'}")
