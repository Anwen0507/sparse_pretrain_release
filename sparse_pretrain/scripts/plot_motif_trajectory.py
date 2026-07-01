#!/usr/bin/env python3
"""Trajectory plot for a motif-excision universality-pruning run
(universality_motif_excision.py). 2x2 panels, mirroring plot_peel_trajectory.py
style but using motif-specific fields (silhouette vs Bernoulli null, survival
reappearance Jaccard). Optionally overlays a frequency-peel baseline.

Usage: plot_motif_trajectory.py MOTIF_EXP_DIR [PEEL_BASELINE_DIR]
"""
import sys, json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(sys.argv[1])
PEEL = Path(sys.argv[2]) if len(sys.argv) > 2 else None
MATCH_J = 0.3; MODEL = "?"
ra = EXP / "run_args.json"
if ra.exists():
    _ra = json.load(open(ra)); MATCH_J = _ra.get("match_j", 0.3); MODEL = _ra.get("model", "?")
BLABEL = sys.argv[3] if len(sys.argv) > 3 else "frequency-peel"


def fnum(v):
    try:
        f = float(v)
        return f if f == f else None       # drop NaN
    except (TypeError, ValueError):
        return None


# --- load motif run ---
H = []
for p in sorted(glob.glob(str(EXP / "iter*/motif_summary.json"))):
    s = json.load(open(p)); d = s.get("decision") or {}; sv = s.get("survival") or {}
    H.append({
        "it": s["iteration"], "succ": s.get("n_success"),
        "k": s.get("k"), "sil": fnum(s.get("silhouette")), "null95": fnum(s.get("null95")),
        "n_excise": d.get("n_excise", 0) or 0, "motif_id": d.get("motif_id"),
        "stop": d.get("stop"), "reason": d.get("reason"),
        "excl_before": s.get("excluded_before", 0),
        "reap_j": fnum(sv.get("max_core_jaccard_to_current")),
        "reappeared": sv.get("reappeared"), "excised": sv.get("excised_motif"),
    })
its = [h["it"] for h in H]
cum = [h["excl_before"] + h["n_excise"] for h in H]          # nodes excluded through iter t
n_motifs = sum(1 for h in H if h["motif_id"])
tot_excl = cum[-1] if cum else 0
reap = sum(1 for h in H if h["reappeared"] is True)
surv_tot = sum(1 for h in H if h["reap_j"] is not None)


def xy(key):
    return ([h["it"] for h in H if h[key] is not None],
            [h[key] for h in H if h[key] is not None])


# --- optional peel baseline ---
pb = None
if PEEL and (PEEL / "state.json").exists():
    bst = json.load(open(PEEL / "state.json")); bh = bst["history"]
    pb = {"it": [h["iteration"] for h in bh], "succ": [h["n_success"] for h in bh],
          "cum": [h.get("excluded_before", 0) for h in bh],
          "n": len(bst["excluded"]), "iters": bst["next_iter"]}

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A: success rate vs iteration (motif vs peel)
a = ax[0, 0]
a.plot(its, [h["succ"] for h in H], "-o", color="C3", label="motif-excision")
if pb: a.plot(pb["it"], pb["succ"], "--x", color="grey", label=BLABEL)
a.axhline(0, color="grey", lw=.8, ls=":")
a.set_ylim(-4, 104); a.set_xlabel("iteration"); a.set_ylabel("circuits at target loss / 100")
a.set_title("Success rate (task survives until collapse)"); a.legend(); a.grid(alpha=.3)
stop = next((h for h in H if h["stop"] and h["reason"] == "no_circuits"), None)
if stop: a.annotate("collapse", (stop["it"], 0), textcoords="offset points", xytext=(-4, 14),
                    ha="right", color="C3", fontsize=9, arrowprops=dict(arrowstyle="->", color="C3"))

# B: clustering significance -- silhouette vs Bernoulli null; k on twinx
a = ax[0, 1]
xs, ys = xy("sil"); a.plot(xs, ys, "-o", color="C0", label="silhouette (observed)")
xs2, ys2 = xy("null95"); a.plot(xs2, ys2, "--", color="grey", label="null 95th pct")
a.fill_between(xs, ys, [dict(zip(xs2, ys2)).get(x, 0) for x in xs],
               where=[y > dict(zip(xs2, ys2)).get(x, 0) for x, y in zip(xs, ys)],
               color="C0", alpha=.15, label="real motif structure")
a.set_xlabel("iteration"); a.set_ylabel("silhouette", color="C0")
a.set_title("Clustering significance vs product-Bernoulli null"); a.grid(alpha=.3); a.legend(loc="upper right", fontsize=8)
a2 = a.twinx(); a2.plot(its, [h["k"] for h in H], "-s", color="C2", alpha=.6)
a2.set_ylabel("# clusters k", color="C2")

# C: SURVIVAL -- did the excised motif stay gone? (headline)
a = ax[1, 0]
for h in H:
    if h["reap_j"] is None: continue
    c = "#d62728" if h["reappeared"] else "#2ca02c"
    a.plot(h["it"], h["reap_j"], "o", color=c, ms=9)
    a.annotate(h["excised"] or "", (h["it"], h["reap_j"]), textcoords="offset points",
               xytext=(0, 6), ha="center", fontsize=7, color=c)
a.axhline(MATCH_J, color="k", lw=1, ls="--")
a.annotate(f"reappear threshold (J={MATCH_J})", (its[1], MATCH_J), textcoords="offset points",
           xytext=(4, 4), fontsize=8)
a.set_ylim(0, 1); a.set_xlabel("iteration")
a.set_ylabel("max core-Jaccard of prev. excised motif\nto a current cluster")
a.set_title(f"SURVIVAL: did the excised motif stay gone?  ({reap}/{surv_tot} REAPPEARED)")
a.grid(alpha=.3)
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([], [], marker="o", ls="", color="#d62728", label="reappeared (re-formed)"),
                  Line2D([], [], marker="o", ls="", color="#2ca02c", label="gone")], fontsize=8)

# D: cumulative nodes excluded (motif vs peel) + per-iter bars
a = ax[1, 1]
a.bar(its, [h["n_excise"] for h in H], color="C1", alpha=.45, label="excised this iter")
a.set_xlabel("iteration"); a.set_ylabel("nodes excised this iter", color="C1")
a.set_title("Exclusion trajectory (motif vs frequency-peel)"); a.grid(alpha=.3)
a2 = a.twinx()
a2.plot(its, cum, "-o", color="C0", label=f"motif cumulative ({tot_excl})")
if pb: a2.plot(pb["it"], pb["cum"], "--x", color="grey", label=f"{BLABEL} cumulative ({pb['n']})")
a2.set_ylabel("cumulative nodes excluded")
a2.legend(loc="lower right", fontsize=8)

cmp = f"   |   {BLABEL}: {pb['n']} nodes / {pb['iters']} iters" if pb else ""
fig.suptitle(f"Motif-excision trajectory — {EXP.name} — {MODEL.split('/')[-1]} / dummy_pronoun (target 0.15)\n"
             f"exhausted at iter {its[-1]}; {n_motifs} motifs, {tot_excl} nodes excluded; "
             f"survival {reap}/{surv_tot} reappeared{cmp}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = EXP / "motif_trajectory.png"; fig.savefig(out, dpi=140)
print("Saved ->", out)
print(f"motifs={n_motifs} excluded={tot_excl} iters={its[-1]} survival_reappeared={reap}/{surv_tot}")
