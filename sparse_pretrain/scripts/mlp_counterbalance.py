#!/usr/bin/env python3
"""What is the layer-1 MLP's counterbalancing/suppression doing, and what function does it emulate?

A. gender-split she−he contribution of L1-attn vs L1-MLP   -> input-dependent suppression vs fixed bias
B. steering dose-response: L1-attn & L1-MLP she−he vs α     -> proportional negative feedback?
C. full-vocab projection of the L1-MLP output              -> what it boosts/suppresses (the "target" function)
D. neuron-level: which L1-MLP neurons suppress, gender-gated?
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json, math
from pathlib import Path
from collections import defaultdict
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
WE = model.wte.weight.detach().float(); WU = model.lm_head.weight.detach().float()
g_lnf = model.ln_f.weight.detach().float(); d_she_he = (g_lnf * (WU[SHE] - WU[HE])).cpu()
g_dir = (WE[list(FEMALE)].mean(0) - WE[list(MALE)].mean(0))
W_out = model.blocks[1].mlp.c_proj.weight.detach().float().cpu()    # (d_model, d_mlp)

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
A_attn, A_mlp, A_neuron, gender, corr, incorr = [], [], [], [], [], []
for b in batches:
    ii, _, ct, it, ep = b
    cap.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16): model(ii)
    bi = torch.arange(ii.shape[0], device=device)
    A_attn.append(cap["L1_attn"][bi, ep].cpu()); A_mlp.append(cap["L1_mlp"][bi, ep].cpu())
    A_neuron.append(cap["neuron"][bi, ep].cpu())
    gender += ["F" if int(ii[j, 1]) in FEMALE else "M" for j in range(ii.shape[0])]
    corr.append(ct.cpu()); incorr.append(it.cpu())
A_attn = torch.cat(A_attn); A_mlp = torch.cat(A_mlp); A_neuron = torch.cat(A_neuron)
gender = np.array(gender); corr = torch.cat(corr); incorr = torch.cat(incorr)
d_ex = g_lnf.cpu() * (WU.cpu()[corr] - WU.cpu()[incorr])     # (N,128) toward correct
R = {}

# ---- A. gender-split she−he contribution ----
attn_sh = (A_attn * d_she_he).sum(1); mlp_sh = (A_mlp * d_she_he).sum(1)
R["A_gender_split_she_minus_he"] = {
    "L1_attn": {"female": float(attn_sh[gender=="F"].mean()), "male": float(attn_sh[gender=="M"].mean())},
    "L1_mlp":  {"female": float(mlp_sh[gender=="F"].mean()),  "male": float(mlp_sh[gender=="M"].mean())}}
print("=== A. she−he contribution by gender (pre-RMS) ===")
print(f"  L1_attn:  female={float(attn_sh[gender=='F'].mean()):+.2f}  male={float(attn_sh[gender=='M'].mean()):+.2f}   (writes the promoted pronoun)")
print(f"  L1_mlp :  female={float(mlp_sh[gender=='F'].mean()):+.2f}  male={float(mlp_sh[gender=='M'].mean()):+.2f}   (opposite sign ⇒ suppresses the promoted pronoun)")

# ---- B. steering dose-response ----
print("\n=== B. steering dose-response: she−he contribution vs α ===")
ii = batches[0][0]; ep = batches[0][4]; bi = torch.arange(ii.shape[0], device=device)
def _add(o, a):
    o = o.clone(); o[:, 1, :] = o[:, 1, :] + a * g_dir; return o
st = {"a": 0.0}
hw = model.wte.register_forward_hook(lambda m, a, o: _add(o, st["a"]))
doseB = []
print(f"  {'α':>5} | {'L1_attn·(she−he)':>16} | {'L1_mlp·(she−he)':>16}")
for a in [-2, -1, 0, 1, 2, 3]:
    st["a"] = a; cap.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16): model(ii)
    at = float((cap["L1_attn"][bi, ep].cpu() * d_she_he).sum(1).mean())
    mp = float((cap["L1_mlp"][bi, ep].cpu() * d_she_he).sum(1).mean())
    doseB.append({"alpha": a, "L1_attn": at, "L1_mlp": mp})
    print(f"  {a:>5} | {at:>16.2f} | {mp:>16.2f}")
hw.remove()
R["B_steering_dose_response"] = doseB
# slope (feedback gain) of mlp vs attn
xs = np.array([r["L1_attn"] for r in doseB]); ys = np.array([r["L1_mlp"] for r in doseB])
gain = float(np.polyfit(xs, ys, 1)[0])
R["B_feedback_gain_dmlp_per_dattn"] = gain
print(f"  → fitted slope  Δ(L1_mlp)/Δ(L1_attn) = {gain:+.2f}   (negative ⇒ proportional negative feedback)")

# ---- C. full-vocab projection of the L1-MLP output ----
print("\n=== C. what the L1-MLP output boosts / suppresses (full vocab) ===")
def toks(ids): return [tok.convert_ids_to_tokens([int(i)])[0].replace("Ġ"," ").replace("Ċ","\\n") for i in ids]
R["C_full_vocab"] = {}
for grp in ["F", "M"]:
    u = (g_lnf.cpu() * A_mlp[gender == grp].mean(0))           # MLP output dir (LN gain folded)
    tl = (WU.cpu() @ u)                                        # per-token logit contribution
    boost = torch.argsort(tl, descending=True)[:10].tolist()
    supp = torch.argsort(tl)[:10].tolist()
    R["C_full_vocab"][grp] = {"boosted": list(zip(toks(boost), [round(float(tl[i]),2) for i in boost])),
                              "suppressed": list(zip(toks(supp), [round(float(tl[i]),2) for i in supp])),
                              "she_rank": int((tl < tl[SHE]).sum()), "he_rank": int((tl < tl[HE]).sum()),
                              "she_val": float(tl[SHE]), "he_val": float(tl[HE])}
    print(f"  [{grp} names] MLP SUPPRESSES: {toks(supp)}")
    print(f"  [{grp} names] MLP BOOSTS:     {toks(boost)}")
    print(f"     she(106) contribution={float(tl[SHE]):+.2f} (rank {int((tl<tl[SHE]).sum())}/4096),  he(103)={float(tl[HE]):+.2f} (rank {int((tl<tl[HE]).sum())}/4096)")

# ---- D. neuron-level suppressors ----
print("\n=== D. which L1-MLP neurons suppress, and are they gender-gated? ===")
# W_out is (d_model, d_mlp); d_ex (N, d_model); d_ex @ W_out -> (N, d_mlp) = each neuron's toward-correct write alignment
contrib = (A_neuron * (d_ex @ W_out)).mean(0)
supp_neurons = torch.argsort(contrib)[:6].tolist()
R["D_top_suppressor_neurons"] = []
for n in supp_neurons:
    fa, ma = float(A_neuron[gender=="F", n].mean()), float(A_neuron[gender=="M", n].mean())
    align = float(W_out[:, n] @ d_she_he)
    R["D_top_suppressor_neurons"].append({"neuron": int(n), "toward_correct_contrib": float(contrib[n]),
                                          "W_out_dot_she_he": align, "act_female": fa, "act_male": ma})
    print(f"  neuron {n:>3}: contrib(toward-correct)={float(contrib[n]):+.3f}  W_out·(she−he)={align:+.2f}  act[F]={fa:+.2f} act[M]={ma:+.2f}")

for h in H: h.remove()
json.dump(R, open(EXP / "mlp_counterbalance.json", "w"), indent=2)

# figure B
fig, ax = plt.subplots(figsize=(7.5, 5))
al = [r["alpha"] for r in doseB]
ax.plot(al, [r["L1_attn"] for r in doseB], "-o", color="C0", label="L1 attention (writes pronoun)")
ax.plot(al, [r["L1_mlp"] for r in doseB], "-s", color="C3", label="L1 MLP (suppresses pronoun)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("α  (gender direction added at the name → drives the head's output)")
ax.set_ylabel("contribution to she−he logit (pre-RMS)")
ax.set_title("L1 MLP is proportional negative feedback on the attention's output\nss_d128_f1 / dummy_pronoun")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(EXP / "mlp_counterbalance_steering.png", dpi=140)
print(f"\nsaved {EXP/'mlp_counterbalance.json'} and {EXP/'mlp_counterbalance_steering.png'}")
