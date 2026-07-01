#!/usr/bin/env python3
"""Plot the universality-pruning trajectory and dump the excluded 'universal core'."""
from sparse_pretrain.paths import OUTPUTS
import sys, json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(sys.argv[1] if len(sys.argv) > 1 else
           str(OUTPUTS / "ss_d128_f1_pronoun"))
D_HEAD = 16  # n_heads=8, d_head=16 for ss_d128_f1
LOC_ORDER = ["attn_in", "attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_neuron", "mlp_out"]

st = json.load(open(EXP / "state.json"))
H = st["history"]


def col(key, sub="mean"):
    return [((h.get(key) or {}).get(sub) if isinstance(h.get(key), dict) else None) for h in H]


its = [h["iteration"] for h in H]
succ = [h["n_success"] for h in H]
size = col("circuit_size_nodes")
njac, eunw, ewt = col("node_jaccard"), col("edge_jaccard_unweighted"), col("edge_jaccard_weighted")
excl_before = [h["excluded_before"] for h in H]
newly = [h.get("n_rank1_nodes", 0) for h in H]


def xy(y):  # drop None (e.g. the exhausted iteration)
    return zip(*[(x, v) for x, v in zip(its, y) if v is not None]) if any(v is not None for v in y) else ([], [])


# ----------------------------- trajectory plot -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

a = ax[0, 0]
a.plot(its, succ, "-o", color="C3", label="success")
a.axhline(0, color="grey", lw=.8, ls=":")
a.set_xlabel("iteration"); a.set_ylabel("circuits at target loss / 100", color="C3")
a.set_ylim(-4, 104); a.set_title("Success rate & mean circuit size"); a.grid(alpha=.3)
a2 = a.twinx(); xs, ys = xy(size); a2.plot(list(xs), list(ys), "-s", color="C0")
a2.set_ylabel("mean circuit size (nodes)", color="C0")

a = ax[0, 1]
for y, lab, m in [(njac, "node", "-o"), (eunw, "edge (unweighted)", "-^"), (ewt, "edge (|W|-weighted)", "-s")]:
    xs, ys = xy(y); a.plot(list(xs), list(ys), m, label=lab)
a.set_xlabel("iteration"); a.set_ylabel("mean pairwise Jaccard")
a.set_title("Circuit agreement (similarity)"); a.legend(); a.grid(alpha=.3); a.set_ylim(0, .7)

a = ax[1, 0]
a.plot(its, excl_before, "-o", color="C2")
a.set_xlabel("iteration"); a.set_ylabel("cumulative nodes excluded (at iter start)")
a.set_title("Cumulative excluded universal nodes"); a.grid(alpha=.3)

a = ax[1, 1]
a.bar(its, newly, color="C4")
a.set_xlabel("iteration"); a.set_ylabel("rank-1 nodes excluded this iter")
a.set_title("Newly excluded per iteration"); a.grid(alpha=.3)

fig.suptitle("Universality pruning — ss_d128_f1 / dummy_pronoun (target loss 0.15)\n"
             f"exhausted at iter {its[-1]}; {len(st['excluded'])} nodes form the task 'universal core'",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(EXP / "trajectory.png", dpi=140)
print(f"Saved plot -> {EXP/'trajectory.png'}")

# ----------------------- held-out generalization (80% prune/select vs 20% test) -----------------------
# Present only when the experiment was run with a prune/test split (--split-over != none).
tloss, sloss = col("heldout_test_loss"), col("select_loss")
grate = [h.get("generalize_rate") for h in H]
if any(v is not None for v in tloss):
    print("\nHeld-out generalization (select on 80% prune set, report on 20% test set):")
    print(f"  {'iter':>4}  {'n_succ':>6}  {'select_loss':>11}  {'test_loss':>9}  {'generalize':>10}")
    for h, sl, tl, gr in zip(H, sloss, tloss, grate):
        if tl is None:
            continue
        sl_s = f"{sl:.4f}" if sl is not None else "n/a"
        gr_s = f"{100*gr:.0f}%" if gr is not None else "n/a"
        print(f"  {h['iteration']:>4}  {h['n_success']:>6}  {sl_s:>11}  {tl:>9.4f}  {gr_s:>10}")

# ----------------------------- core dump -----------------------------
excl = [tuple(x) for x in st["excluded"]]
node_iter = {}
for h in H:
    for s in h.get("rank1_nodes", []):
        k, i = s.rsplit("#", 1); node_iter[(k, int(i))] = h["iteration"]

groups = defaultdict(list)
for (k, i) in excl:
    groups[k].append(i)


def loc_sort(k):
    layer = int(k.split("_")[0][5:]); loc = k.split("_", 1)[1]
    return (layer, LOC_ORDER.index(loc))


lines = [f"UNIVERSAL CORE for dummy_pronoun on ss_d128_f1 : {len(excl)} nodes (of 2816 maskable, ~{100*len(excl)/2816:.1f}%)",
         "Removing these makes a target-loss circuit unrecoverable (exhaustion at iter %d)." % its[-1], ""]
by_layer = defaultdict(int); by_loc = defaultdict(int)
for key in sorted(groups, key=loc_sort):
    idxs = sorted(groups[key]); layer, loc = key.split("_")[0], key.split("_", 1)[1]
    by_layer[layer] += len(idxs); by_loc[loc] += len(idxs)
    iters = sorted(set(node_iter.get((key, i), -1) for i in idxs))
    if loc in ("attn_q", "attn_k", "attn_v"):
        byhead = defaultdict(list)
        for i in idxs:
            byhead[i // D_HEAD].append(i % D_HEAD)
        detail = " | ".join(f"head{h}:{sorted(v)}" for h, v in sorted(byhead.items()))
    else:
        detail = str(idxs)
    lines.append(f"{key:>22} ({len(idxs):>2})  [excl@iter {iters}]  {detail}")

lines += ["", "by layer:    " + ", ".join(f"{k}={v}" for k, v in sorted(by_layer.items())),
          "by location: " + ", ".join(f"{k}={v}" for k, v in sorted(by_loc.items(), key=lambda x: -x[1]))]
text = "\n".join(lines)
print("\n" + text)
(EXP / "core_nodes.txt").write_text(text + "\n")
json.dump({"n_core": len(excl),
           "nodes": [{"location": k, "index": i, "excluded_at_iter": node_iter.get((k, i))} for (k, i) in sorted(excl, key=lambda x: (loc_sort(x[0]), x[1]))],
           "by_layer": dict(by_layer), "by_location": dict(by_loc)},
          open(EXP / "core_nodes.json", "w"), indent=2)
print(f"\nSaved -> {EXP/'core_nodes.txt'}  and  {EXP/'core_nodes.json'}")
