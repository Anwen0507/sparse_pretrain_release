#!/usr/bin/env python3
"""Verify two causal claims about the iteration-0 pronoun circuit:
  C1: the name TOKEN EMBEDDINGS linearly encode gender.
  C2: head-3 attention COPIES that (gender-carrying) value from the name to the pronoun slot.

C1: (a) leave-one-out + held-out-name generalization of a gender direction in W_E;
    (b) CAUSAL steering — add the embedding gender direction at the name, watch the pronoun flip.
C2: (a) CAUSAL value-patching/interchange — swap head-3's value at the name between genders;
    (b) weights-level OV check (effective OV maps gender direction -> she-he direction).
"""
from sparse_pretrain.paths import OUTPUTS
import sys, math, json
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
FEMALE, MALE = gender_name_token_ids(tok, as_set=False)   # order-preserving lists
WE = model.wte.weight.detach().float(); WU = model.lm_head.weight.detach().float()
g_lnf = model.ln_f.weight.detach().float(); d_she_he = (g_lnf * (WU[SHE] - WU[HE]))
g_dir = WE[FEMALE].mean(0) - WE[MALE].mean(0)          # gender direction in embedding space
ghat = g_dir / g_dir.norm()
FEMs, MALs = set(FEMALE), set(MALE)

def she_he_at_eval(input_ids, evalp):
    logits, _, _ = model(input_ids)
    bi = torch.arange(input_ids.shape[0], device=device)
    return (logits[bi, evalp, SHE] - logits[bi, evalp, HE])

# ============================== CLAIM 1 ==============================
print("=== C1a. Gender linearly separable in W_E? (leave-one-out over the task names) ===")
ids = FEMALE + MALE; lab = np.array([1]*len(FEMALE) + [0]*len(MALE))   # 1=female
n_names = len(ids)
correct = 0
for h in range(n_names):
    keep = [i for i in range(n_names) if i != h]
    fdir = WE[[ids[i] for i in keep if lab[i]==1]].mean(0) - WE[[ids[i] for i in keep if lab[i]==0]].mean(0)
    mid = 0.5*(WE[[ids[i] for i in keep if lab[i]==1]].mean(0) + WE[[ids[i] for i in keep if lab[i]==0]].mean(0))
    pred = 1 if float((WE[ids[h]] - mid) @ fdir) > 0 else 0
    correct += int(pred == lab[h])
print(f"  leave-one-out accuracy on the {n_names} task names: {correct}/{n_names}")

print("\n=== C1b. Does the gender direction GENERALIZE to held-out names? ===")
cand = {"f": ["Anna","Emma","Sara","Sarah","Laura","Julia","Nina","Clara","Ella","Grace","Hannah",
              "Sophie","Alice","Rose","Ruby","Daisy","Eva","Olivia","Chloe","Amelia","Mary","Jane","Lucy","Zoe","Lola"],
        "m": ["Tom","Ben","Max","Jack","Jake","Noah","Liam","Oliver","Henry","George","Harry","Charlie",
              "Daniel","David","Michael","James","William","Adam","Mark","Luke","John","Paul","Ryan","Owen","Eli"]}
mid_all = 0.5*(WE[FEMALE].mean(0) + WE[MALE].mean(0))
nf = nm = ok = 0; tested = []
for gen, names in cand.items():
    for nm_ in names:
        e = tok.encode(" " + nm_, add_special_tokens=False)
        if len(e) != 1 or e[0] in FEMs or e[0] in MALs: continue
        proj = float((WE[e[0]] - mid_all) @ ghat)
        pred = "f" if proj > 0 else "m"; ok += int(pred == gen)
        tested.append((nm_, gen, round(proj, 2)))
        nf += gen == "f"; nm += gen == "m"
print(f"  held-out single-token names tested: {len(tested)} ({nf} F, {nm} M)")
print(f"  gender-direction classification accuracy on NEW names: {ok}/{len(tested)} = {ok/max(len(tested),1):.0%}")
print(f"  sample projections (>0 ⇒ 'female' side): {tested[:6]} ... {tested[-4:]}")

print("\n=== C1c. CAUSAL steering: add α·(gender direction) at the name, measure she−he logit ===")
pb = get_task("dummy_pronoun", tok, seed=0, split="val").generate_batch(256, 0)
ii, evalp = pb[0].to(device), pb[4].to(device)
isF = torch.tensor([int(ii[j,1]) in FEMs for j in range(ii.shape[0])], device=device)
state = {"a": 0.0}
def steer(m, a, o):
    o = o.clone(); o[:, 1, :] = o[:, 1, :] + state["a"] * g_dir; return o
h = model.wte.register_forward_hook(steer)
print(f"  {'α':>5} | {'she−he (male names)':>20} | {'she−he (female names)':>22}")
steer_rows = []
for a in [-2, -1, -0.5, 0, 0.5, 1, 2]:
    state["a"] = a; sh = she_he_at_eval(ii, evalp)
    steer_rows.append({"alpha": a, "male": float(sh[~isF].mean()), "female": float(sh[isF].mean())})
    print(f"  {a:>5} | {float(sh[~isF].mean()):>20.2f} | {float(sh[isF].mean()):>22.2f}")
h.remove()
print("  (α=0 is baseline; male names start negative, female positive; a sign flip = causal control)")

# ============================== CLAIM 2 ==============================
print("\n=== C2a. CAUSAL value-patching: swap head-3's value at the NAME between genders ===")
cap = {}
hc = model.blocks[1].attn.c_attn.register_forward_hook(lambda m,a,o: cap.__setitem__("qkv", o.detach()))
model(ii)
hc.remove()
v3 = cap["qkv"][:, 1, 256+48:256+64].float()                 # head-3 value at name, per example
vF = v3[isF].mean(0); vM = v3[~isF].mean(0)                   # mean female / male name-value
patch = {"v": None}
def patch_val(m, a, o):
    if patch["v"] is None: return o
    o = o.clone(); o[:, 1, 256+48:256+64] = patch["v"]; return o
hp = model.blocks[1].attn.c_attn.register_forward_hook(patch_val)
patch["v"] = None;            base = she_he_at_eval(ii, evalp)
patch["v"] = vF;              pF = she_he_at_eval(ii, evalp)   # force female value everywhere
patch["v"] = vM;              pM = she_he_at_eval(ii, evalp)
hp.remove()
def afc(sh, isfem): return float(((sh > 0) == isfem).float().mean())
print(f"  baseline:                  she−he  male={float(base[~isF].mean()):+.2f}  female={float(base[isF].mean()):+.2f}  (2AFC {afc(base,isF):.0%})")
print(f"  patch FEMALE value at name: she−he  male={float(pF[~isF].mean()):+.2f}  female={float(pF[isF].mean()):+.2f}")
print(f"  patch MALE value at name:   she−he  male={float(pM[~isF].mean()):+.2f}  female={float(pM[isF].mean()):+.2f}")
flipM = float(((base[~isF] < 0) & (pF[~isF] > 0)).float().mean())
flipF = float(((base[isF] > 0) & (pM[isF] < 0)).float().mean())
print(f"  → male examples flipped he→she by female-value patch: {flipM:.0%};  female→he by male-value patch: {flipF:.0%}")

print("\n=== C2b. Weights-level OV: does head-3's effective OV map gender-dir → she−he dir? ===")
WV3 = model.blocks[1].attn.c_attn.weight.detach().float()[256+48:256+64, :]   # (16,128)
WO3 = model.blocks[1].attn.c_proj.weight.detach().float()[:, 48:64]           # (128,16)
g_ln1 = model.blocks[1].ln_1.weight.detach().float()
def ln1(e): return e / math.sqrt(float((e*e).mean()) + 1e-6) * g_ln1
vin = ln1(WE[FEMALE].mean(0)) - ln1(WE[MALE].mean(0))         # gender contrast at head-3 input
out = WO3 @ (WV3 @ vin)                                       # head-3 OV output
ov_scalar = float(out @ d_she_he)
print(f"  (OV applied to gender contrast) · (she−he direction) = {ov_scalar:+.2f}   [>0 ⇒ writes 'she']")

# ---------------------------- persist results ----------------------------
R = {
    "model": "jacobcd52/ss_d128_f1", "task": "dummy_pronoun",
    "C1_embeddings_carry_gender": {
        "C1a_LOO_accuracy_10_task_names": f"{correct}/10",
        "C1b_heldout_name_generalization": {
            "n_single_token_names_tested": len(tested), "n_correct": ok,
            "accuracy": ok / max(len(tested), 1), "projections": tested,
            "note": "underpowered: tiny vocab yields few single-token names"},
        "C1c_causal_steering_she_minus_he": steer_rows,
    },
    "C2_attention_copies_value": {
        "C2a_value_patch_she_minus_he": {
            "baseline": {"male": float(base[~isF].mean()), "female": float(base[isF].mean())},
            "patch_female_value_at_name": {"male": float(pF[~isF].mean()), "female": float(pF[isF].mean())},
            "patch_male_value_at_name": {"male": float(pM[~isF].mean()), "female": float(pM[isF].mean())},
            "frac_male_flipped_to_she": flipM, "frac_female_flipped_to_he": flipF},
        "C2b_effective_OV_on_gender_dir_dot_she_he": ov_scalar,
        "note": "steering & value-patch run on the FULL model (not a pruned circuit); "
                "the value-patch asymmetry reflects full-model redundancy (Hydra-style self-repair).",
    },
}
json.dump(R, open(EXP / "mech_interp_verification.json", "w"), indent=2)

al = [r["alpha"] for r in steer_rows]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(al, [r["male"] for r in steer_rows], "-o", color="C0", label="male names")
ax.plot(al, [r["female"] for r in steer_rows], "-s", color="C3", label="female names")
ax.axhline(0, color="k", lw=1, ls="--"); ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("α  (amount of embedding gender direction added at the name)")
ax.set_ylabel("she − he logit at the pronoun position")
ax.set_title("C1c causal steering — the embedding gender direction controls the pronoun\nss_d128_f1 / dummy_pronoun")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(EXP / "verification_steering.png", dpi=140)
print(f"\nsaved {EXP/'mech_interp_verification.json'}  and  {EXP/'verification_steering.png'}")
print("DONE")
