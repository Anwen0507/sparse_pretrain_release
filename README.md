# Exhausting Backup Circuits

Code to reproduce **"Exhausting Backup Circuits: Redundancy, Universality, and
Necessity in Dense and Weight-Sparse Transformer Circuits"** (Anwen Hao & Rick
Goldstein, SPAR Spring 2026).

We study circuit **redundancy** by *exhausting* the backup circuits of a
pronoun-gender task in two checkpoints from Gao et al. (2025): a fully **dense**
2-layer transformer (`ss_d128_f1`, `d=128`) and a **weight-sparse** one
(`ss_bridges_d1024_f0.015625`, `d=1024`, 1.56 % nonzero weights). We iteratively
prune minimal task circuits from 100 random seeds, forcibly exclude the
most-shared node each round, and re-prune until the task can no longer be solved.
We then decompose circuits into NMF **atoms** and peel them to exhaustion. Main
findings: backup circuits are **finite** and **exhaustible**; the discovered core
is **task-selective** and **generalizes** to held-out names; the exclusion
dynamics obey a probabilistic **cascade** law with a seed-dependent endpoint; and
**weight-sparsity buys deep, cheap, modular redundancy** over a small irreducible
core, whereas dense redundancy is finite and collapses through a phase transition.

---

## What's in this repository

This is the **interpretability / analysis** pipeline only. The two model
checkpoints are Gao et al.'s and are downloaded from the HuggingFace Hub at
runtime, so **nothing large ships here** — no model weights, datasets, or
precomputed experiment outputs. Model *pretraining* code is out of scope.

```
sparse_pretrain/
├── src/
│   ├── config.py, model.py          # SparseGPT model + config (loads dense & sparse ckpts)
│   └── pruning/                      # node-mask circuit pruning (Gao et al. 2025)
│       ├── config.py  tasks.py       # PruningConfig; dummy_pronoun / _tense / _article tasks
│       ├── node_mask.py  masked_model.py  trainer.py
│       └── discretize.py  calibrate.py  run_pruning.py   # load_model() lives here
├── scripts/                          # experiment + plotting scripts (see Reproduction)
├── data/name_pools/                  # curated pronoun-task name pools (cast15, ...)
└── paths.py                          # portable output/figure paths (env-overridable)
```

## Installation

```bash
git clone <this-repo> && cd <this-repo>
pip install -e .                      # installs deps from pyproject.toml
# optional, only for re-tuning pruning hyperparameters:
pip install "carbs @ git+https://github.com/imbue-ai/carbs.git"
```

Requires Python ≥ 3.10 and a **CUDA GPU**. Set `HF_TOKEN` if you hit HuggingFace
rate limits.

## Testing

```bash
pip install pytest pytest-cov
pytest                                   # full suite (~1 min on a GPU box)
pytest --cov=sparse_pretrain             # with line coverage (~94%)
```

The suite is hermetic — no network, no HuggingFace downloads, no experiment
outputs required. Models are tiny randomly-initialized `SparseGPT`s, tasks use
a deterministic whitespace tokenizer, and the analysis/plot scripts run
against synthetic experiment directories built in `tmp_path`. Multi-process
seed workers are rerouted in-process, so even the iterative exhaustion
protocol (`universality_pruning_experiment.py`) is exercised end to end on
CPU. One file, `tests/test_analysis_scripts_cuda.py`, execs the nine
mech-interp scripts that hardcode `device="cuda"`; those tests skip
automatically on CPU-only machines, everything else is CPU-only by design.

## Models & data (auto-downloaded)

| Kind | HuggingFace id | Used for |
|---|---|---|
| Dense model | `jacobcd52/ss_d128_f1` | dense results (core, selectivity, mechanism, d128 atom-peel) |
| Weight-sparse model | `jacobcd52/ss_bridges_d1024_f0.015625` | sparse results (d1024 atom-peel, NMF) |
| Tokenizer | `SimpleStories/SimpleStories-1.25M` | all tasks |

Name pools ship in `sparse_pretrain/data/name_pools/` (`name_pool_cast15.json` is
the 15-name SimpleStories "cast", gender-validated in fp32).

## Where outputs go

Experiment scripts write per-run directories, and plot scripts write figures.
Both locations are repo-relative and overridable:

```bash
export SP_OUTPUTS=/path/to/outputs     # default: ./outputs/universality_pruning
export SP_FIGURES=/path/to/figures     # default: ./figures
```

Each exhaustion run creates `<SP_OUTPUTS>/<run-name>/` containing `iterNN/`
(per-seed circuit masks + results), `iteration_summary.json`, `state.json`, and
`run_args.json`. These are **not** version-controlled (`.gitignore`).

## Compute notes

* Each **exhaustion run** prunes 100 seeds × ~2000 steps for each exclusion
  iteration (9–37 iterations depending on the run) — hours on one GPU.
* Multi-seed pruning parallelizes across worker processes (`--num-workers`). For a
  large (~9×) speedup with several workers sharing one GPU, start the CUDA MPS
  daemon first: `nvidia-cuda-mps-control -d`.
* Pruning is **deterministic given the seed set**; all probabilistic statements
  (the cascade model, the seed-dependent terminal core) concern counterfactual
  seed ensembles. A different `--seed-offset` gives a different core.
* All commands below default to `--skip-carbs` (fixed, CARBS-tuned center
  hyperparameters). Drop it to re-run the CARBS sweep (needs the `carbs` extra).

---

## Reproducing the report

Set a shorthand for the outputs base (matches the script defaults):

```bash
export OUT=${SP_OUTPUTS:-outputs/universality_pruning}
```

**Run naming.** Most steps below take an explicit `--exp-dir`, but a few analysis
scripts read specific run-directory *names*: the selectivity/mechanism scripts read
the dense core from `ss_d128_f1_pronoun`, the NMF/atom-peel figures read
`{cast15_fold0_motif, sparse_d1024_motif, d128_atom_peel, sparse_d1024_atom_peel}`,
and the cascade-model script reads the fixed list at the top of
`test_avalanche_model_all.py`. Use the exact `--exp-dir` names shown in each step
(or edit the run list at the top of the consuming script to point at your dirs).

### Step 1 — Exhaustion runs → **Fig. 1 (success rate)**, and the dense core

Run the exhaustion loop for each configuration. The **dense baseline** must live
at `ss_d128_f1_pronoun` because the selectivity/mechanism scripts read its core
from there.

```bash
# O  — dense 10-name baseline (single family; exhausts ~iter 9, 84-node core)
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs \
    --exp-dir $OUT/ss_d128_f1_pronoun

# O' — exact rerun (bit-identical), and a fresh seed block 100–199 (different core)
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs \
    --exp-dir $OUT/ss_d128_f1_pronoun_repeat2
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs \
    --seed-offset 100 --exp-dir $OUT/ss_d128_f1_pronoun_seeds100_199

# cast15 fold-0 with a names×templates split (also yields held-out 2-AFC)
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs \
    --split-over names_templates --heldout-fold 0 \
    --exp-dir $OUT/ss_d128_f1_pronoun_cast15_fold0
```

The report's Fig. 1 overlays seven such runs; add further variants by changing
`--seed-offset`, `--name-pool` (e.g. `name_pool_cast15_orig10plus2.json`), and
`--split-over`. Then aggregate every run under `$OUT` and plot:

```bash
python -m sparse_pretrain.scripts.iteration_jaccard_all   # -> $OUT/iteration_jaccard_all.json
python -m sparse_pretrain.scripts.plot_success_rate       # -> $SP_FIGURES/fig_success_rate.png
```

### Step 2 — **Table 1 (task selectivity)**

Extract the dense core from the baseline run, then ablate it vs. matched random
ablation across tasks:

```bash
python -m sparse_pretrain.scripts.universality_pruning_report   # -> ss_d128_f1_pronoun/core_nodes.json
python -m sparse_pretrain.scripts.ablate_core_crosstask         # -> crosstask_ablation.{json,png}
python -m sparse_pretrain.scripts.plot_crosstask_2afc           # 2-AFC bars
# lighter cross-check:
python -m sparse_pretrain.scripts.crosstask_ablation_check
```

### Step 3 — Mechanism (copy-from-embedding; MLP counterbalance)

```bash
python -m sparse_pretrain.scripts.mech_interp_core            # head ablation, value-patch flip, OV projection
python -m sparse_pretrain.scripts.mlp_counterbalance          # DLA: attention +142% / MLP −42% of she−he gap
python -m sparse_pretrain.scripts.mlp_function
python -m sparse_pretrain.scripts.crosstask_attn_vs_mlp_dla
python -m sparse_pretrain.scripts.verify_embedding_and_copying
python -m sparse_pretrain.scripts.compare_full_vs_circuit_pronoun
```

### Step 4 — Held-out generalization (2-AFC)

```bash
python -m sparse_pretrain.scripts.recompute_2afc \
    --exp-dir $OUT/ss_d128_f1_pronoun_cast15_fold0
```

### Step 5 — Cascade / probabilistic model (validated across nine runs)

`test_avalanche_model_all.py` consumes a fixed set of exhaustion runs, named at
the top of the file (`STD_RUNS` / `PEEL_RUNS`): six standard runs
`{og10names, og10names_repeat, og10names_seeds100_199, cast15_nosplit,
cast15_fold0, cast15_og10plus2}` and three `*_multiexclude` peel runs. Produce
them by varying the exhaustion command (name pool, `--split-over`,
`--seed-offset`) into `--exp-dir`s with those names — e.g.:

```bash
# O / O' / O'' — dense 10-name, identical config, seeds 0–99 (twice) and 100–199
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs --exp-dir $OUT/og10names
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs --exp-dir $OUT/og10names_repeat
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs \
    --seed-offset 100 --exp-dir $OUT/og10names_seeds100_199
# B / fold0 / og10plus2 — 15-name variants via --split-over and --name-pool
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs \
    --split-over names_templates --heldout-fold 0 --exp-dir $OUT/cast15_fold0
```

(edit `STD_RUNS`/`PEEL_RUNS` if you prefer different names), then:

```bash
python -m sparse_pretrain.scripts.test_avalanche_model_all    # log-corr ≈ 0.98; -> model_test_all_runs.json
python -m sparse_pretrain.scripts.dump_law_check
python -m sparse_pretrain.scripts.dump_law_check2
```

### Step 6 — **Fig. 2 (NMF dictionary)** and **Fig. 3 (atom-peel)**

First produce the two *source* exhaustion runs (dense cast15 + weight-sparse),
then peel atoms to exhaustion:

```bash
# source ensembles (iter00 supplies the initial 100-circuit pool)
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_d128_f1 --task dummy_pronoun --skip-carbs \
    --split-over names_templates --heldout-fold 0 --exp-dir $OUT/cast15_fold0_motif
python -m sparse_pretrain.scripts.universality_pruning_experiment \
    --model jacobcd52/ss_bridges_d1024_f0.015625 --task dummy_pronoun --skip-carbs \
    --split-over names_templates --heldout-fold 0 --exp-dir $OUT/sparse_d1024_motif

# atom-peel  -> each writes <exp-dir>/atom_peel.png  == report Fig. 3(b)/3(a)
python -m sparse_pretrain.scripts.universality_atom_peel \
    --src-dir $OUT/cast15_fold0_motif --exp-dir $OUT/d128_atom_peel \
    --model jacobcd52/ss_d128_f1 --split-over names_templates --heldout-fold 0 \
    --max-depth 40 --feas-eps 0.01
python -m sparse_pretrain.scripts.universality_atom_peel \
    --src-dir $OUT/sparse_d1024_motif --exp-dir $OUT/sparse_d1024_atom_peel \
    --model jacobcd52/ss_bridges_d1024_f0.015625 --split-over names_templates --heldout-fold 0 \
    --max-depth 40 --feas-eps 0.01

# NMF dictionary figure (reads iter00 of both atom-peel dirs)
python -m sparse_pretrain.scripts.plot_nmf_dictionary          # -> $SP_FIGURES/fig_nmf_dictionary.png
```

`fig_atompeel_d1024.png` / `fig_atompeel_d128.png` are the `atom_peel.png` files
written into `sparse_d1024_atom_peel/` and `d128_atom_peel/`.

The NMF decomposition itself (rank selection, backbone vs. discriminative atoms)
lives in `motif_dictionary.py`, imported by the peel and plot scripts; run it
standalone on any run dir with
`python -m sparse_pretrain.scripts.motif_dictionary --exp-dir $OUT/<run> --kmax 16`.

### Step 7 — Necessity / articulation, and the negative results

```bash
# irreducible-core necessity + leave-one-out articulation points (weight-sparse)
python -m sparse_pretrain.scripts.exclude_kernel
python -m sparse_pretrain.scripts.epistasis_test      # NMF atoms are functionally atomic
python -m sparse_pretrain.scripts.atomicity_test
python -m sparse_pretrain.scripts.atom_excision

# negative results: node-Jaccard is backbone-confounded (retired);
# "one clean motif per iteration" fails its survival check;
# within-iteration clustering beats a Bernoulli null
python -m sparse_pretrain.scripts.iteration_jaccard
python -m sparse_pretrain.scripts.universality_motif_excision --model jacobcd52/ss_d128_f1
python -m sparse_pretrain.scripts.family_clustering
python -m sparse_pretrain.scripts.plot_family_clustering
python -m sparse_pretrain.scripts.metric_clustering_robustness
```

---

## Citation

```bibtex
@techreport{hao2026backup,
  title  = {Exhausting Backup Circuits: Redundancy, Universality, and Necessity
            in Dense and Weight-Sparse Transformer Circuits},
  author = {Hao, Anwen and Goldstein, Rick},
  year   = {2026},
  note   = {SPAR Spring 2026}
}
```

Built on Gao et al. (2025), *Weight-sparse transformers have interpretable
circuits*, whose checkpoints and node-mask pruning method this work analyzes.

## License

MIT — see [LICENSE](LICENSE).
