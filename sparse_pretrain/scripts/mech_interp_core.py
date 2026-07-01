#!/usr/bin/env python3
"""Mechanistic interpretability of the 84-node pronoun core (ss_d128_f1 / dummy_pronoun).

A. Causal decomposition  — ablate each functional sub-group in the full model.
B. Residual decomposition / direct logit attribution — which component writes the she−he answer.
C. Layer-0 MLP as gender detector — selectivity of core neurons at the NAME position.
D. Head-3 OV circuit — does head-3's value at the name, via W_O, write the pronoun direction.
E. Residual read/write channel map + unembedding alignment (static).
"""
from sparse_pretrain.paths import OUTPUTS
import sys, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task, gender_name_token_ids
from sparse_pretrain.src.pruning.run_pruning import load_model

EXP = (OUTPUTS / "ss_d128_f1_pronoun")
device = "cuda"; torch.set_grad_enabled(False)
HE, SHE = 103, 106
model, _ = load_model("jacobcd52/ss_d128_f1", device); mc = model.config
tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
FEMALE, MALE = gender_name_token_ids(tok)
core = [(n["location"], n["index"]) for n in json.load(open(EXP / "core_nodes.json"))["nodes"]]
by_key = defaultdict(list)
for loc, idx in core: by_key[loc].append(idx)

WU = model.lm_head.weight.detach().float()          # (V, d_model)
g = model.ln_f.weight.detach().float()              # (d_model,)
d_she_he = g * (WU[SHE] - WU[HE])                    # residual-space direction "toward she"
autocast = torch.autocast("cuda", dtype=torch.bfloat16)

batches = []
for sd in range(10):
    t = get_task("dummy_pronoun", tok, seed=sd, split="val")
    batches.append([x.to(device) for x in t.generate_batch(batch_size=64, max_length=0)])

# ============================ A. CAUSAL DECOMPOSITION ============================
pc = PruningConfig(device=device, ablation_type="zero", mask_token_embeds=False, batch_size=64, seq_length=0)
mm = MaskedSparseGPT(model, pc); mm.to(device)

def set_off(nodes):
    st = mm.get_mask_state()
    for k in st: st[k].fill_(1.0)
    for loc, idx in nodes: st[loc][idx] = -1.0
    mm.load_mask_state(st)

def eval_pron():
    ld = afc = 0.0
    for b in batches:
        with autocast:
            _, m = mm.compute_task_loss(*b, use_binary_loss=True)
        ld += m["logit_diff"]; afc += m["binary_accuracy"]
    return ld / len(batches), afc / len(batches)

groups = {
    "(none) baseline": [],
    "FULL core (84)": core,
    "L0 MLP (8)": [(k, i) for k, i in core if k.startswith("layer0")],
    "L1 attn_in (24)": [(k, i) for k, i in core if k == "layer1_attn_in"],
    "L1 attn Q/K/V (19)": [(k, i) for k, i in core if k in ("layer1_attn_q", "layer1_attn_k", "layer1_attn_v")],
    "  - head3 Q/K/V only (13)": [(k, i) for k, i in core if k in ("layer1_attn_q", "layer1_attn_k", "layer1_attn_v") and 48 <= i < 64],
    "L1 attn_out (18)": [(k, i) for k, i in core if k == "layer1_attn_out"],
    "L1 MLP (15)": [(k, i) for k, i in core if k.startswith("layer1_mlp")],
}
print("=== A. CAUSAL DECOMPOSITION (ablate group in FULL model; dummy_pronoun val) ===")
print(f"{'group':>27} | {'logit_diff':>10} | {'2AFC':>6} | n_off")
R = {"model": "jacobcd52/ss_d128_f1", "task": "dummy_pronoun", "causal_decomposition": {}}
for name, nodes in groups.items():
    set_off(nodes); ld, afc = eval_pron()
    R["causal_decomposition"][name.strip()] = {"logit_diff": float(ld), "twoAFC": float(afc), "n_off": len(nodes)}
    print(f"{name:>27} | {ld:>10.2f} | {afc:>6.1%} | {len(nodes)}")
del mm; torch.cuda.empty_cache()

# ============================ B-E: hooked raw model ============================
cap = {}
def fpre(n):
    def h(m, a): cap[n] = a[0].detach().float()
    return h
def fout(n):
    def h(m, a, o): cap[n] = (o[0] if isinstance(o, tuple) else o).detach().float()
    return h
H = [model.blocks[0].attn.register_forward_hook(fout("L0_attn")),
     model.blocks[0].mlp.register_forward_hook(fout("L0_mlp")),
     model.blocks[0].mlp.c_proj.register_forward_pre_hook(fpre("l0_neuron")),
     model.blocks[1].attn.register_forward_hook(fout("L1_attn")),
     model.blocks[1].attn.c_attn.register_forward_hook(fout("qkv")),
     model.blocks[1].mlp.register_forward_hook(fout("L1_mlp")),
     model.ln_f.register_forward_pre_hook(fpre("resid_final"))]

attn_fn = getattr(model.blocks[1].attn, "attn_fn", None)
sink3 = float(attn_fn.sink_logit[3]) if attn_fn is not None else None
scale = 1.0 / math.sqrt(mc.d_head)
comp = defaultdict(list); neuron_name = []; v3_name = []
gender = []; corr = []; incorr = []; attn_name = []
for b in batches:
    pos_ids, _, ct, it, ep = b
    cap.clear()
    with autocast: model(pos_ids)
    B, T = pos_ids.shape; bi = torch.arange(B, device=device)
    emb = model.wte(pos_ids).float()[bi, ep]
    comp["emb"].append(emb.cpu()); comp["L0_attn"].append(cap["L0_attn"][bi, ep].cpu())
    comp["L0_mlp"].append(cap["L0_mlp"][bi, ep].cpu()); comp["L1_attn"].append(cap["L1_attn"][bi, ep].cpu())
    comp["L1_mlp"].append(cap["L1_mlp"][bi, ep].cpu()); comp["resid_final"].append(cap["resid_final"][bi, ep].cpu())
    neuron_name.append(cap["l0_neuron"][:, 1, :].cpu())
    qkv = cap["qkv"]
    v3_name.append(qkv[:, 1, 256 + 48:256 + 64].cpu())
    q3 = qkv[:, :, 0:128].view(B, T, 8, 16)[:, :, 3]; k3 = qkv[:, :, 128:256].view(B, T, 8, 16)[:, :, 3]
    for j in range(B):
        e = int(ep[j]); lg = (k3[j, :e + 1] @ q3[j, e]) * scale
        cols = torch.cat([torch.tensor([sink3], device=device), lg]) if sink3 is not None else lg
        w = torch.softmax(cols, 0); ar = (w[1:] if sink3 is not None else w)
        attn_name.append(float(ar[1]))
        gender.append("F" if int(pos_ids[j, 1]) in FEMALE else "M")
    corr.append(ct.cpu()); incorr.append(it.cpu())
for h in H: h.remove()

comp = {k: torch.cat(v).float() for k, v in comp.items()}
neuron_name = torch.cat(neuron_name).float(); v3_name = torch.cat(v3_name).float()
corr = torch.cat(corr); incorr = torch.cat(incorr); gender = np.array(gender)
d_ex = g.cpu() * (WU.cpu()[corr] - WU.cpu()[incorr])     # (N,128) toward-correct direction

# ---- B. residual decomposition / logit attribution (pre-RMS units, toward correct) ----
print("\n=== B. RESIDUAL DECOMPOSITION (mean contribution to correct−incorrect logit) ===")
contribs = {k: float((comp[k] * d_ex).sum(1).mean()) for k in ["emb", "L0_attn", "L0_mlp", "L1_attn", "L1_mlp"]}
tot = float((comp["resid_final"] * d_ex).sum(1).mean())
for k in ["emb", "L0_attn", "L0_mlp", "L1_attn", "L1_mlp"]:
    print(f"  {k:>10}: {contribs[k]:+7.3f}   ({100*contribs[k]/tot:5.1f}% of total)")
print(f"  {'TOTAL':>10}: {tot:+7.3f}   (sum of parts={sum(contribs.values()):+.3f})")
print("  core's share within last-layer write components:")
for comp_name, key in [("L1_attn", "layer1_attn_out"), ("L1_mlp", "layer1_mlp_out"), ("L0_mlp", "layer0_mlp_out")]:
    dims = by_key[key]; full = float((comp[comp_name] * d_ex).sum(1).mean())
    corep = float((comp[comp_name][:, dims] * d_ex[:, dims]).sum(1).mean())
    print(f"    {comp_name} core dims {dims}: {corep:+.3f} / {full:+.3f}  ({100*corep/full if full else 0:4.0f}% of that component)")

# ---- C. layer-0 MLP gender detector (at name position) ----
print("\n=== C. LAYER-0 MLP GENDER SELECTIVITY (neuron activations at NAME position) ===")
F = neuron_name[gender == "F"]; M = neuron_name[gender == "M"]
sel = (F.mean(0) - M.mean(0)) / (0.5 * (F.std(0) + M.std(0)) + 1e-6)   # d-prime-ish, +=female
order = torch.argsort(sel.abs(), descending=True)
print("  most gender-selective L0 neurons (|d'|):", [(int(i), round(float(sel[i]), 2)) for i in order[:6]])
for n in by_key["layer0_mlp_neuron"]:
    rank = int((sel.abs() > abs(sel[n])).sum())
    print(f"    CORE neuron {n}: d'={float(sel[n]):+.2f}  (rank {rank+1} of 512;  female_mean={float(F[:,n].mean()):+.2f} male_mean={float(M[:,n].mean()):+.2f})")

# ---- D. head-3 OV circuit (value at name -> W_O -> she/he direction) ----
print("\n=== D. HEAD-3 OV CIRCUIT (value at NAME pos, through W_O, onto she−he dir) ===")
WO = model.blocks[1].attn.c_proj.weight.detach().float().cpu()    # (d_model, d_v)
write = v3_name @ WO[:, 48:64].T                                  # (N,128) what head-3 writes if it attends here
proj = write @ d_she_he.cpu()                                     # (N,) >0 => pushes "she"
print(f"  head-3 value→W_O projection onto (she−he):  female={float(proj[gender=='F'].mean()):+.3f}  male={float(proj[gender=='M'].mean()):+.3f}")
print(f"  (positive = pushes 'she'); separation works iff female>0>male")
print(f"  head-3 attention from pronoun pos to NAME (pos1): mean={np.mean(attn_name):.2f} (rest goes to sink/other)")

# ---- E. residual read/write channel map + unembedding alignment ----
print("\n=== E. RESIDUAL CHANNEL MAP + UNEMBEDDING ALIGNMENT ===")
read_dims = sorted(set(by_key["layer1_attn_in"]) | set(by_key["layer1_mlp_in"]) | set(by_key["layer0_mlp_in"]))
write_dims = sorted(set(by_key["layer1_attn_out"]) | set(by_key["layer1_mlp_out"]) | set(by_key["layer0_mlp_out"]))
absd = d_she_he.abs().cpu()
rank_of = {int(c): int((absd > absd[c]).sum()) for c in range(128)}
top_she_he = torch.argsort(absd, descending=True)[:15].tolist()
print(f"  top-15 she−he-aligned residual dims: {top_she_he}")
l1_out_core = by_key["layer1_attn_out"]
print(f"  L1 attn_out CORE write dims (18): {sorted(l1_out_core)}")
print(f"    of these, in top-15 she−he dims: {sorted(set(l1_out_core)&set(top_she_he))}")
print(f"    mean she−he-|alignment| rank of L1-attn_out core dims: {np.mean([rank_of[c] for c in l1_out_core]):.0f}/128  (vs 64 if random)")
print(f"  L0 mlp_out core write dims: {sorted(by_key['layer0_mlp_out'])}  | L1 attn_in core read dims (24): {sorted(by_key['layer1_attn_in'])}")
print(f"  dims BOTH written by L0-mlp_out core AND read by L1-attn_in core: {sorted(set(by_key['layer0_mlp_out'])&set(by_key['layer1_attn_in']))}")

# ---------------------------- persist results ----------------------------
R["logit_attribution_toward_correct"] = {
    "components": {k: contribs[k] for k in contribs}, "total": tot,
    "core_share": {cn: {"core": float((comp[cn][:, by_key[key]] * d_ex[:, by_key[key]]).sum(1).mean()),
                        "component_total": float((comp[cn] * d_ex).sum(1).mean())}
                   for cn, key in [("L1_attn", "layer1_attn_out"), ("L1_mlp", "layer1_mlp_out"), ("L0_mlp", "layer0_mlp_out")]}}
R["L0_gender_selectivity_at_name"] = {
    "top_detectors_dprime": [[int(i), round(float(sel[i]), 2)] for i in order[:8]],
    "core_neurons": {int(n): {"dprime": round(float(sel[n]), 2), "rank_of_512": int((sel.abs() > abs(sel[n])).sum()) + 1}
                     for n in by_key["layer0_mlp_neuron"]}}
R["head3_OV"] = {"female_value_proj_she_he": float(proj[gender == "F"].mean()),
                 "male_value_proj_she_he": float(proj[gender == "M"].mean()),
                 "attn_to_name_mean": float(np.mean(attn_name))}
R["residual_channel_map"] = {"top15_she_he_dims": top_she_he, "L1_attn_out_core_dims": sorted(l1_out_core),
                             "core_dims_in_top15": sorted(set(l1_out_core) & set(top_she_he)),
                             "mean_she_he_alignment_rank_of_core_dims": float(np.mean([rank_of[c] for c in l1_out_core]))}
json.dump(R, open(EXP / "mech_interp_core.json", "w"), indent=2)

cd = R["causal_decomposition"]; gn = list(cd.keys()); gv = [cd[n]["logit_diff"] for n in gn]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(range(len(gn)), gv, color="C0")
ax.axvline(cd["(none) baseline"]["logit_diff"], color="k", ls="--", lw=1, label="baseline (none ablated)")
ax.set_yticks(range(len(gn))); ax.set_yticklabels(gn, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("pronoun logit-diff after ablating the group  (lower ⇒ more necessary)")
ax.set_title("Causal decomposition of the 84-node core (ablation in the full model)")
ax.legend(); fig.tight_layout(); fig.savefig(EXP / "core_causal_decomposition.png", dpi=140)
print(f"\nsaved {EXP/'mech_interp_core.json'} and {EXP/'core_causal_decomposition.png'}")
print("\nDONE")
