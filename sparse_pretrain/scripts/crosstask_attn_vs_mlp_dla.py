#!/usr/bin/env python3
"""Cross-task test of arXiv:2512.22471's 'attention routes, FFN does the update' claim.

For each task, decompose the correct-incorrect logit gap at the eval position into the five
additive residual writes {embedding, L0-attn, L0-mlp, L1-attn, L1-mlp} (DLA with the final
RMSNorm gain folded in). Report each sublayer's SHARE of the total gap, and attention-total
vs MLP-total. Tasks are ordered by rough 'update demand' (pure copy -> comparison/elimination).
The paper predicts the MLP share should grow with update demand.
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
torch.set_grad_enabled(False); device = "cuda"; EPS = 1e-6
model, _ = load_model("jacobcd52/ss_d128_f1", device)
tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
WU = model.lm_head.weight.detach().float(); g_lnf = model.ln_f.weight.detach().float()

cap = {}
def fpre(n):
    def h(m, a): cap[n] = a[0].detach().float()
    return h
def fout(n):
    def h(m, a, o): cap[n] = (o[0] if isinstance(o, tuple) else o).detach().float()
    return h
H = [model.blocks[0].register_forward_pre_hook(fpre("emb")),
     model.blocks[0].attn.register_forward_hook(fout("attn0")),
     model.blocks[0].mlp.register_forward_hook(fout("mlp0")),
     model.blocks[1].attn.register_forward_hook(fout("attn1")),
     model.blocks[1].mlp.register_forward_hook(fout("mlp1"))]
COMPS = ["emb", "attn0", "mlp0", "attn1", "mlp1"]

# ordered by rough update demand: pure copy/lookup -> syntactic prior -> agreement -> comparison
TASKS = ["dummy_pronoun", "dummy_quote", "dummy_article", "dummy_tense",
         "ioi_relaxed", "ioi_strict", "ioi_mixed"]

def run_task(name, n_seeds=4):
    contrib = {c: [] for c in COMPS}; afc = []; gap = []; recon_err = []
    for sd in range(n_seeds):
        try:
            b = get_task(name, tok, seed=sd, split="val").generate_batch(64, 0)
        except Exception as e:
            return None
        pos, _, ct, it, ep = [x.to(device) for x in b]
        cap.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(pos)
        bi = torch.arange(pos.shape[0], device=device)
        comp = {c: cap[c][bi, ep] for c in COMPS}                  # (B,128) each, at eval pos
        r_final = sum(comp[c] for c in COMPS)
        rms = torch.sqrt((r_final * r_final).mean(-1) + EPS)       # (B,)
        d = g_lnf * (WU[ct] - WU[it])                              # (B,128) toward correct
        for c in COMPS:
            contrib[c].append(((comp[c] * d).sum(-1) / rms).cpu().numpy())
        decomp_gap = sum(((comp[c] * d).sum(-1) / rms) for c in COMPS)
        actual_gap = (logits[bi, ep, ct] - logits[bi, ep, it]).float()
        recon_err.append((decomp_gap - actual_gap).abs().mean().item())
        gap.append(actual_gap.cpu().numpy()); afc.append((actual_gap > 0).float().cpu().numpy())
    contrib = {c: np.concatenate(contrib[c]) for c in COMPS}
    gap = np.concatenate(gap); afc = np.concatenate(afc)
    means = {c: float(contrib[c].mean()) for c in COMPS}
    total = sum(means.values())
    shares = {c: means[c] / total for c in COMPS}
    return {"n": len(gap), "afc": float(afc.mean()), "gap": float(gap.mean()),
            "recon_err": float(np.mean(recon_err)), "contrib_mean": means, "share": shares,
            "attn_share": shares["attn0"] + shares["attn1"], "mlp_share": shares["mlp0"] + shares["mlp1"]}

R = {}
print(f"{'task':>16} | {'2AFC':>5} | {'gap':>6} || {'emb':>6} {'L0a':>6} {'L0m':>6} {'L1a':>6} {'L1m':>6} || {'ATTN':>6} {'MLP':>6}  (share of correct-incorrect gap)")
print("-" * 118)
for t in TASKS:
    r = run_task(t)
    if r is None:
        print(f"{t:>16} | (skipped)"); continue
    R[t] = r; s = r["share"]
    print(f"{t:>16} | {r['afc']:>4.0%} | {r['gap']:>6.2f} || "
          f"{s['emb']:>6.0%} {s['attn0']:>6.0%} {s['mlp0']:>6.0%} {s['attn1']:>6.0%} {s['mlp1']:>6.0%} || "
          f"{r['attn_share']:>6.0%} {r['mlp_share']:>6.0%}")
for h in H: h.remove()
errs = ", ".join("%s=%.2f" % (t, R[t]["recon_err"]) for t in R)
print("\n(reconstruction error |decomp-actual| per task: " + errs + ")")
print("\nPaper (2512.22471) predicts MLP share grows with update demand (copy<agreement<comparison).")
json.dump(R, open(EXP / "crosstask_attn_vs_mlp_dla.json", "w"), indent=2)
print(f"saved {EXP/'crosstask_attn_vs_mlp_dla.json'}")
