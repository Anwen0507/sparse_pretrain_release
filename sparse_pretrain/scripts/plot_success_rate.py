"""Figure: per-iteration success counts N_t/100 for the exhaustion runs (report Fig. 1).

Reads the aggregate produced by ``iteration_jaccard_all.py`` (``iteration_jaccard_all.json``
under $SP_OUTPUTS) and draws one panel per exhaustion run: the observed success
trajectory N_t over exclusion iterations, with the orange "cascade" band (N_t <= 10)
where one exclusion round dumps tens-to-hundreds of nodes.

Runs are discovered from the JSON (keyed by their --exp-dir name), so this plots
whatever exhaustion runs you have aggregated -- no hard-coded run list.

    python -m sparse_pretrain.scripts.iteration_jaccard_all   # writes the aggregate
    python -m sparse_pretrain.scripts.plot_success_rate        # writes figures/fig_success_rate.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sparse_pretrain.paths import OUTPUTS, ensure_figures

d = json.load(open(OUTPUTS / 'iteration_jaccard_all.json'))

runs = {}
for run, v in d.items():
    ns = v['n_success']
    its = sorted(int(k) for k in ns)
    runs[run] = (np.array(its, float), np.array([ns[str(i)] for i in its], float))

names = sorted(runs)
n = len(names)
ncol = 4
nrow = max(1, (n + ncol - 1) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4 * nrow), squeeze=False)
axes = axes.ravel()
for ax, run in zip(axes, names):
    t, N = runs[run]
    ax.plot(t, N, 'o-', color='k', ms=5, lw=1, zorder=3, label='observed $N_t$')
    ax.axhspan(0, 10, color='orange', alpha=.12)          # cascade region N_t <= 10
    ax.set_title(run, fontsize=9)
    ax.set_ylim(-3, 108)
    ax.set_xlabel('iteration $t$')
    ax.set_ylabel('successes $N_t$ /100')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=.25)
for ax in axes[n:]:
    ax.axis('off')

plt.tight_layout()
out = ensure_figures() / 'fig_success_rate.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'wrote {out}  ({n} runs)')
