#!/usr/bin/env python3
"""Show the NMF H (atom x node) dictionary for iter0 of two atom-peel runs.

Recovered generator for SPAR final-report Figure 2 (figures/fig_nmf_dictionary.png).
Renders, per atom-peel run, the row-normalised NMF dictionary H as a heatmap with a
node-frequency strip on top. Both the H heatmap (magma) and the freq strip (viridis)
now carry their own colorbar.
"""
from sparse_pretrain.paths import OUTPUTS, FIGURES
import sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sparse_pretrain.scripts.motif_dictionary import global_node_space, load_iteration, describe_atoms

BASE = OUTPUTS
RUNS = [("sparse_d1024_atom_peel", "d1024 weight-sparse"),
        ("d128_atom_peel", "d128 dense")]
ITER = 0
DEFAULT_OUT = str(FIGURES / "fig_nmf_dictionary.png")


def factor(run):
    d = BASE / run
    a = json.load(open(d / "run_args.json"))
    K = next(x["k"] for x in json.load(open(d / "atom_peel.json"))["atom_trajectory"] if x["depth"] == ITER)
    nodes, index = global_node_space(d)
    X, _ = load_iteration(d / f"iter{ITER:02d}", index, len(nodes))
    amask = X.sum(0) > 0
    anodes = [nodes[j] for j in np.where(amask)[0]]
    Xa = X[:, amask]; freq = Xa.mean(0)
    atoms, relerr, H = describe_atoms(Xa, K, anodes, freq, 0, 12)
    return dict(a=a, K=K, anodes=anodes, Xa=Xa, freq=freq, atoms=atoms, H=H, relerr=relerr,
                bb=float(a["backbone_thr"]))


def report(run, label, F):
    H, freq, atoms = F["H"], F["freq"], F["atoms"]
    n_circ, n_active = F["Xa"].shape
    print(f"\n{'='*78}\n{label}  ({run})\n{'='*78}")
    print(f"X = W H   ->   X:({n_circ} circuits x {n_active} active nodes)  "
          f"W:({n_circ} x {F['K']})  H:({F['K']} x {n_active})   relerr={F['relerr']:.3f}")
    print(f"active nodes: {int((freq>=F['bb']).sum())} backbone (freq>={F['bb']}) + "
          f"{int((freq<F['bb']).sum())} motif (freq<{F['bb']})")
    print(f"\n  H rows (atoms), sorted by mean circuit-share:")
    print(f"  {'atom':>4} {'role':>15} {'share':>6} {'peak_H':>7} {'footprint':>9}  top nodes (loading)")
    for at in atoms:
        a = at["atom"]; hr = H[a]; pk = hr.max()
        foot = int((hr >= 0.1 * pk).sum())  # nodes loaded >=10% of peak
        tops = ", ".join(f"{t['node']}={t['weight']}" for t in at["top_nodes"][:4])
        print(f"  a{a:<3} {at['role']:>15} {at['mean_share']:>6} {pk:>7.2f} {foot:>9}  {tops}")


def heat(ax, axf, F, label):
    """row-normalised H heatmap; columns sorted by owning atom; freq strip on top."""
    H, freq = F["H"], F["freq"]
    order = [at["atom"] for at in F["atoms"]]            # rows by share
    Hs = H[order]
    rn = Hs / np.maximum(Hs.max(1, keepdims=True), 1e-12)  # per-atom peak -> [0,1]
    own = np.argmax(Hs, 0)                                # which (sorted) atom owns each col
    col = sorted(range(Hs.shape[1]), key=lambda j: (own[j], -Hs[own[j], j]))
    im = ax.imshow(rn[:, col], aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"a{a} ({'BB' if at['role'][0]=='b' else 'mot'})"
                        for a, at in zip(order, F["atoms"])], fontsize=7)
    ax.set_xlabel(f"{Hs.shape[1]} active nodes (sorted by owning atom)", fontsize=8)
    ax.set_ylabel("atoms (by share)", fontsize=8)
    ax.set_title(f"{label}: H = {F['K']}x{Hs.shape[1]}   (rows=atoms, row-normalised to peak)", fontsize=9)
    fim = axf.imshow(freq[col][None, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axf.set_yticks([0]); axf.set_yticklabels(["freq"], fontsize=7); axf.set_xticks([])
    return im, fim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help="output PNG path")
    args = ap.parse_args()

    Fs = [(run, label, factor(run)) for run, label in RUNS]
    for run, label, F in Fs:
        report(run, label, F)

    fig = plt.figure(figsize=(13, 9))
    # Reserve a right margin for two SHARED colorbars: both panels use the identical
    # freq (viridis 0..1) and H (magma 0..1) scales, so we draw each scale once rather
    # than four bars total. GridSpec stops at right=0.80; the bars live beyond it.
    gs = GridSpec(4, 1, height_ratios=[1, 9, 1, 9], hspace=0.35,
                  left=0.09, right=0.80, top=0.92, bottom=0.08)
    im0 = fim0 = None
    for i, (run, label, F) in enumerate(Fs):
        axf = fig.add_subplot(gs[2 * i]); ax = fig.add_subplot(gs[2 * i + 1])
        im, fim = heat(ax, axf, F, label)
        if im0 is None:
            im0, fim0 = im, fim
    # Two full-height shared colorbars in the right margin. The inner (node freq) bar's
    # ticks/label face LEFT and the outer (H norm) bar's face RIGHT, so they point away
    # from each other and never overlap.
    cax_f = fig.add_axes([0.845, 0.08, 0.015, 0.84])
    cax_h = fig.add_axes([0.925, 0.08, 0.015, 0.84])
    cbf = fig.colorbar(fim0, cax=cax_f)
    cbf.set_label("node freq")
    cbf.ax.yaxis.set_ticks_position("left")
    cbf.ax.yaxis.set_label_position("left")
    cbh = fig.colorbar(im0, cax=cax_h)
    cbh.set_label("H (norm)")
    fig.suptitle("NMF dictionary H per atom-peel run (iter 0, full model)", fontsize=12)
    fig.savefig(args.out, dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
