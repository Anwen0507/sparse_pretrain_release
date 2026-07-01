#!/usr/bin/env python3
"""Does the L1-MLP emulate the natural next-token prior? + gender-split steering control.

a1. natural-prior tests (overall MLP output, gender-common part):
    - word-start vs '##'-continuation: mean per-token contribution by class
    - breadth: top-2 share of |contribution| for L1-attn (narrow promoter) vs L1-MLP (broad prior)
    - overlap of MLP-boosted tokens with the model's natural next-token prediction
a2. gender-split steering: sweep α per gender; track L1-MLP·(she−he) and suppressor neurons 510/471
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json, math
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sparse_pretrain.src.pruning.tasks import get_task, gender_name_token_ids
from sparse_pretrain.src.pruning.run_pruning import load_model

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
torch.set_grad_enabled(False); device = "cuda"
HE, SHE = 103, 106
model, _ = load_model("jacobcd52/ss_d128_f1", device); mc = model.config
tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
FEMALE, MALE = gender_name_token_ids(tok)
WU = model.lm_head.weight.detach().float().cpu(); g_lnf = model.ln_f.weight.detach().float().cpu()
d_she_he = g_lnf * (WU[SHE] - WU[HE])
g_dir = (model.wte.weight.detach().float()[list(FEMALE)].mean(0) - model.wte.weight.detach().float()[list(MALE)].mean(0))
V = WU.shape[0]

cap = {}
def fout(n):
    def h(m, a, o): cap[n] = (o[0] if isinstance(o, tuple) else o).detach().float()
    return h
def fpre(n):
    def h(m, a): cap[n] = a[0].detach().float()
    return h
H = [model.blocks[1].attn.register_forward_hook(fout("L1_attn")),
     model.blocks[1].mlp.register_forward_hook(fout("L1_mlp")),
     model.blocks[1].mlp.c_proj.register_forward_pre_hook(fpre("neuron"))]

batches = [[x.to(device) for x in get_task("dummy_pronoun", tok, seed=sd, split="val").generate_batch(64, 0)] for sd in range(6)]
A_attn, A_mlp, A_full, gender = [], [], [], []
for b in batches:
    ii, _, _, _, ep = b
    cap.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, _ = model(ii)
    bi = torch.arange(ii.shape[0], device=device)
    A_attn.append(cap["L1_attn"][bi, ep].cpu()); A_mlp.append(cap["L1_mlp"][bi, ep].cpu())
    A_full.append(logits[bi, ep].float().cpu())
    gender += ["F" if int(ii[j, 1]) in FEMALE else "M" for j in range(ii.shape[0])]
A_attn = torch.cat(A_attn); A_mlp = torch.cat(A_mlp); A_full = torch.cat(A_full); gender = np.array(gender)
R = {}

# ---------------- a1. natural-prior tests ----------------
toks = tok.convert_ids_to_tokens(list(range(V)))
is_cont = np.array([t.startswith("##") for t in toks])
is_special = np.array([t.startswith("[") and t.endswith("]") for t in toks])
is_word = ~is_cont & ~is_special
tl_mlp = (WU @ (g_lnf * A_mlp.mean(0))).numpy()      # MLP per-token logit contribution (gender-common)
tl_attn = (WU @ (g_lnf * A_attn.mean(0))).numpy()
print("=== a1. MLP output vs natural next-token prior ===")
print(f"  mean MLP contribution:  word-starts={tl_mlp[is_word].mean():+.3f}   '##'-continuations={tl_mlp[is_cont].mean():+.3f}")
print(f"    → MLP boosts word-starts and suppresses subword continuations (Δ={tl_mlp[is_word].mean()-tl_mlp[is_cont].mean():+.3f})")
conc = lambda tl: float(np.sort(np.abs(tl))[::-1][:2].sum() / np.abs(tl).sum())
print(f"  breadth (top-2 share of |contribution|):  L1-attn={conc(tl_attn):.2f} (narrow)   L1-MLP={conc(tl_mlp):.2f} (broad)")
nat = A_full.mean(0).numpy()
nat_top = np.argsort(nat)[::-1][:15]; mlp_top = np.argsort(tl_mlp)[::-1][:15]
ov = len(set(nat_top.tolist()) & set(mlp_top.tolist()))
print(f"  model's natural top-15 next tokens: {[toks[i] for i in nat_top]}")
print(f"  MLP's top-15 boosted tokens:        {[toks[i] for i in mlp_top]}")
print(f"  overlap of MLP-boosted with natural top-15: {ov}/15")
R["a1_word_start_vs_continuation"] = {"mlp_word_start_mean": float(tl_mlp[is_word].mean()),
                                      "mlp_continuation_mean": float(tl_mlp[is_cont].mean())}
R["a1_breadth_top2_share"] = {"L1_attn": conc(tl_attn), "L1_mlp": conc(tl_mlp)}
R["a1_natural_top15"] = [toks[i] for i in nat_top]
R["a1_mlp_top15"] = [toks[i] for i in mlp_top]
R["a1_overlap_mlp_with_natural_top15"] = ov

# ---------------- a2. gender-split steering ----------------
print("\n=== a2. gender-split steering (does the MLP track the head within gender?) ===")
ii = batches[0][0]; ep = batches[0][4]; bi = torch.arange(ii.shape[0], device=device)
isF = np.array([int(ii[j, 1]) in FEMALE for j in range(ii.shape[0])])
def _add(o, a):
    o = o.clone(); o[:, 1, :] = o[:, 1, :] + a * g_dir; return o
st = {"a": 0.0}
hw = model.wte.register_forward_hook(lambda m, a, o: _add(o, st["a"]))
rows = []
print(f"  {'α':>4} | {'attn·(she-he) F/M':>20} | {'MLP·(she-he) F/M':>20} | {'neuron510 F/M':>16} | {'neuron471 F/M':>16}")
for a in [-2, -1, 0, 1, 2, 3]:
    st["a"] = a; cap.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16): model(ii)
    at = (cap["L1_attn"][bi, ep].cpu() * d_she_he).sum(1).numpy()
    mp = (cap["L1_mlp"][bi, ep].cpu() * d_she_he).sum(1).numpy()
    n510 = cap["neuron"][bi, ep, 510].cpu().numpy(); n471 = cap["neuron"][bi, ep, 471].cpu().numpy()
    row = {"alpha": a, "attn_F": float(at[isF].mean()), "attn_M": float(at[~isF].mean()),
           "mlp_F": float(mp[isF].mean()), "mlp_M": float(mp[~isF].mean()),
           "n510_F": float(n510[isF].mean()), "n510_M": float(n510[~isF].mean()),
           "n471_F": float(n471[isF].mean()), "n471_M": float(n471[~isF].mean())}
    rows.append(row)
    print(f"  {a:>4} | {row['attn_F']:>9.2f}/{row['attn_M']:<9.2f} | {row['mlp_F']:>9.2f}/{row['mlp_M']:<9.2f} | "
          f"{row['n510_F']:>7.2f}/{row['n510_M']:<7.2f} | {row['n471_F']:>7.2f}/{row['n471_M']:<7.2f}")
hw.remove()
R["a2_gender_split_steering"] = rows
for h in H: h.remove()
json.dump(R, open(EXP / "mlp_function.json", "w"), indent=2)

# figure: gender-split steering
al = [r["alpha"] for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(al, [r["attn_F"] for r in rows], "-o", color="C0", label="attn (F)")
ax[0].plot(al, [r["attn_M"] for r in rows], "--o", color="C0", label="attn (M)")
ax[0].plot(al, [r["mlp_F"] for r in rows], "-s", color="C3", label="MLP (F)")
ax[0].plot(al, [r["mlp_M"] for r in rows], "--s", color="C3", label="MLP (M)")
ax[0].axhline(0, color="k", lw=.8); ax[0].set_xlabel("α"); ax[0].set_ylabel("she−he contribution")
ax[0].set_title("Attention swings ±10 with α; net MLP she−he stays flat"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(al, [r["n510_F"] for r in rows], "-o", color="C2", label="neuron510 (F)")
ax[1].plot(al, [r["n510_M"] for r in rows], "--o", color="C2", label="neuron510 (M)")
ax[1].plot(al, [r["n471_F"] for r in rows], "-s", color="C4", label="neuron471 (F)")
ax[1].plot(al, [r["n471_M"] for r in rows], "--s", color="C4", label="neuron471 (M)")
ax[1].set_xlabel("α"); ax[1].set_ylabel("neuron activation (post-GELU)")
ax[1].set_title("...yet neurons #510/#471 DO respond to α (510↑ writes 'he', 471↓ writes 'she')"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("L1-MLP's NET she−he is not proportional feedback on the head (left), though its neurons respond to gender (right)")
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(EXP / "mlp_gendersplit_steering.png", dpi=140)
print(f"\nsaved {EXP/'mlp_function.json'} and {EXP/'mlp_gendersplit_steering.png'}")
