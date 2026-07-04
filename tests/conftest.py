"""Shared fixtures for the sparse_pretrain test suite.

Everything here is hermetic: no network, no GPU, no HuggingFace downloads.
Models are tiny (2 layers, d_model=16) so the whole suite runs on CPU in a
couple of minutes; tasks use a deterministic whitespace tokenizer instead of
the SimpleStories WordPiece tokenizer.
"""
import json
import os
import tempfile

# Must be set before matplotlib is imported anywhere (several scripts also
# call matplotlib.use("Agg") themselves, but tests import pyplot directly too).
os.environ.setdefault("MPLBACKEND", "Agg")
# Keep any module-level OUTPUTS/FIGURES references pointed at a throwaway dir
# so an accidentally-unpatched script can never write into the repo.
_SESSION_SCRATCH = tempfile.mkdtemp(prefix="sparse_pretrain_tests_")
os.environ.setdefault("SP_OUTPUTS", os.path.join(_SESSION_SCRATCH, "outputs"))
os.environ.setdefault("SP_FIGURES", os.path.join(_SESSION_SCRATCH, "figures"))
# Never let a test talk to the HuggingFace Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    torch.set_grad_enabled(True)  # some exec'd scripts disable it globally
    import numpy as np
    import random
    np.random.seed(0)
    random.seed(0)


# ---------------------------------------------------------------------------
# Fake tokenizer: whitespace word-level, ids assigned on first sight
# ---------------------------------------------------------------------------
class FakeTokenizer:
    """Duck-typed stand-in for the SimpleStories tokenizer.

    Implements exactly the surface the library uses: encode/decode,
    pad/unk/eos token ids, and settable pad_token. Splitting on whitespace
    keeps every name/pronoun/article a single distinct token, which is the
    property the binary tasks rely on.
    """

    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.inv = {0: "<pad>", 1: "<unk>", 2: "<eos>"}

    def _id(self, word):
        if word not in self.vocab:
            idx = len(self.vocab)
            self.vocab[word] = idx
            self.inv[idx] = word
        return self.vocab[word]

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def decode(self, ids):
        return " ".join(self.inv.get(int(i), "<unk>") for i in ids)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.inv.get(ids, "<unk>")
        return [self.inv.get(int(i), "<unk>") for i in ids]

    def get_vocab(self):
        return dict(self.vocab)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


# ---------------------------------------------------------------------------
# Tiny models
# ---------------------------------------------------------------------------
TINY_VOCAB = 512


def make_tiny_config(**overrides):
    from sparse_pretrain.src.config import ModelConfig
    kwargs = dict(
        n_layer=2, d_model=16, n_ctx=32, d_head=4, d_mlp=24,
        vocab_size=TINY_VOCAB, use_rms_norm=True, tie_embeddings=False,
        use_positional_embeddings=False, use_bigram_table=True,
        use_attention_sinks=True, activation="gelu", dropout=0.0,
        use_bias=True, use_flash_attention=True,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def make_tiny_model(sparsity=None, **config_overrides):
    """A fresh 2-layer SparseGPT in eval mode. sparsity=None disables
    weight/activation sparsity (the safe-inference default)."""
    from sparse_pretrain.src.model import SparseGPT
    model = SparseGPT(make_tiny_config(**config_overrides), sparsity)
    model.eval()
    return model


@pytest.fixture
def tiny_config():
    return make_tiny_config()


@pytest.fixture
def tiny_model():
    return make_tiny_model()


@pytest.fixture
def tiny_pruning_config():
    from sparse_pretrain.src.pruning.config import PruningConfig
    return PruningConfig(device="cpu", batch_size=4, seq_length=0,
                         num_steps=3, log_every=10 ** 9)


def make_masked(model=None, **pc_overrides):
    """MaskedSparseGPT around a tiny model, on CPU."""
    from sparse_pretrain.src.pruning.config import PruningConfig
    from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
    if model is None:
        model = make_tiny_model()
    kwargs = dict(device="cpu", batch_size=4, seq_length=0, num_steps=3,
                  log_every=10 ** 9)
    kwargs.update(pc_overrides)
    return MaskedSparseGPT(model, PruningConfig(**kwargs))


def set_all_taus(masked_model, value):
    """Force every node mask tau to a constant (+1 = fully active)."""
    with torch.no_grad():
        for mask in masked_model.masks.masks.values():
            mask.tau.fill_(value)
        if masked_model.token_mask is not None:
            masked_model.token_mask.tau.fill_(value)


# ---------------------------------------------------------------------------
# A deterministic binary task that needs no tokenizer at all
# ---------------------------------------------------------------------------
class ToyTask:
    """Minimal BinaryTask stand-in: fixed contexts over a tiny id range.

    generate_batch matches the (pos, neg, correct, incorrect, eval_positions)
    contract used by MaskedSparseGPT/PruningTrainer/discretize/calibrate.
    """

    def __init__(self, seq_len=6, vocab=TINY_VOCAB, seed=0):
        self.seq_len = seq_len
        self.vocab = vocab
        self.gen = torch.Generator().manual_seed(seed)
        self.name = "toy"

    def generate_batch(self, batch_size, max_length=0):
        L = self.seq_len if max_length <= 0 else min(self.seq_len, max_length)
        pos = torch.randint(3, min(64, self.vocab), (batch_size, L),
                            generator=self.gen)
        correct = torch.randint(3, min(64, self.vocab), (batch_size,),
                                generator=self.gen)
        incorrect = (correct + 1) % min(64, self.vocab)
        eval_pos = torch.full((batch_size,), L - 1, dtype=torch.long)
        return pos, pos.clone(), correct, incorrect, eval_pos


@pytest.fixture
def toy_task():
    return ToyTask()


# ---------------------------------------------------------------------------
# Model-dim-consistent circuit fixtures (for scripts that re-apply masks to a
# MaskedSparseGPT: keys and dims must match the wrapped model)
# ---------------------------------------------------------------------------
MASK_LOCS = ["attn_in", "attn_q", "attn_k", "attn_v", "attn_out", "mlp_in",
             "mlp_neuron", "mlp_out"]


def model_mask_dims(model):
    dims = {}
    for layer in range(model.config.n_layer):
        for loc in MASK_LOCS:
            if loc in ("attn_q", "attn_k", "attn_v"):
                d = model.config.n_heads * model.config.d_head
            elif loc == "mlp_neuron":
                d = model.config.d_mlp
            else:
                d = model.config.d_model
            dims[f"layer{layer}_{loc}"] = d
    return dims


def write_model_circuits(it_dir, model, n_circuits=3, frac=0.5, seed=0,
                         block_pattern=False):
    """seed*_circuit.pt + passing seed*_result.json with model-true dims.

    block_pattern=True plants two disjoint discriminative node blocks on top
    of a shared backbone (even seeds get block A, odd seeds block B), so NMF
    decompositions find exactly two specific motifs.
    """
    import json as _json
    it_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    dims = model_mask_dims(model)
    keys = sorted(dims)
    for s in range(n_circuits):
        if block_pattern:
            mask = {k: torch.zeros(d) for k, d in dims.items()}
            for k in keys:  # backbone: node 0 of every location
                mask[k][0] = 1.0
            block_key = keys[0]
            if s % 2 == 0:
                mask[block_key][1:4] = 1.0  # block A
            else:
                mask[block_key][4:7] = 1.0  # block B
        else:
            mask = {k: (torch.rand(d, generator=g) < frac).float()
                    for k, d in dims.items()}
        torch.save(mask, it_dir / f"seed{s}_circuit.pt")
        size = int(sum(v.sum() for v in mask.values()))
        (it_dir / f"seed{s}_result.json").write_text(_json.dumps(
            {"seed": s, "target_achieved": True, "circuit_size": size,
             "circuit_loss": 0.1, "test_loss": 0.12, "test_2afc": 0.9,
             "generalizes": True}))


# ---------------------------------------------------------------------------
# Synthetic universality-pruning experiment directories
# ---------------------------------------------------------------------------
# Node keys follow the real layout (layer<N>_<loc>) so scripts that parse the
# key names (universality_pruning_report.loc_sort etc.) accept them.
EXP_KEYS = ["layer0_attn_in", "layer0_attn_q", "layer0_mlp_neuron",
            "layer1_attn_out", "layer1_mlp_out"]
EXP_DIM = 8  # nodes per key -> 40-node global space


def make_circuit_mask(active_nodes):
    """circuit_mask dict {key: float tensor of 0/1} from (key, idx) pairs."""
    mask = {k: torch.zeros(EXP_DIM) for k in EXP_KEYS}
    for key, idx in active_nodes:
        mask[key][idx] = 1.0
    return mask


def make_exp_dir(base, n_iters=3, seeds_per_iter=4, name="exp",
                 extra_history_keys=None, fail_last_seed=False):
    """Build a self-consistent fake experiment directory.

    Layout matches universality_pruning_experiment.py output:
      state.json, hparams.json, run_args.json,
      iterNN/{excluded_input.json, iteration_summary.json,
              seedK_result.json, seedK_circuit.pt}

    Circuits are deterministic: seed s of iteration t activates nodes
    {(EXP_KEYS[j], (t + j) % EXP_DIM) for j} plus one seed-specific node, so
    every iteration has a large shared core (every node appears in >= 2
    circuits except the seed-specific ones).
    """
    exp = base / name
    exp.mkdir(parents=True, exist_ok=True)
    history = []
    excluded = []
    for t in range(n_iters):
        it_dir = exp / f"iter{t:02d}"
        it_dir.mkdir(exist_ok=True)
        (it_dir / "excluded_input.json").write_text(json.dumps(
            [list(x) for x in excluded]))
        seed_results = []
        node_sets = []
        for s in range(seeds_per_iter):
            ok = not (fail_last_seed and s == seeds_per_iter - 1)
            core = {(EXP_KEYS[j], (t + j) % EXP_DIM) for j in range(len(EXP_KEYS))}
            extra = {(EXP_KEYS[s % len(EXP_KEYS)], (t + 3 + s) % EXP_DIM)}
            nodes = core | extra
            rec = {"seed": s, "target_achieved": ok,
                   "loss_at_all_active": 0.05 + 0.01 * s if ok else 0.9,
                   "num_active_after_training": len(nodes),
                   "total_nodes": len(EXP_KEYS) * EXP_DIM,
                   "circuit_size": len(nodes) if ok else None,
                   "circuit_loss": 0.1 + 0.01 * s if ok else None,
                   "test_loss": 0.12 + 0.01 * s if ok else None,
                   "test_2afc": 0.9 - 0.01 * s if ok else None,
                   "generalizes": ok or None}
            seed_results.append(rec)
            if ok:
                torch.save(make_circuit_mask(nodes), it_dir / f"seed{s}_circuit.pt")
                node_sets.append(nodes)
            (it_dir / f"seed{s}_result.json").write_text(json.dumps(rec))
        n_succ = len(node_sets)
        rank1 = sorted(set.intersection(*node_sets)) if node_sets else []
        summary = {
            "iteration": t, "num_seeds": seeds_per_iter, "n_success": n_succ,
            "success_rate": n_succ / seeds_per_iter, "elapsed_sec": 1.0,
            "excluded_before": len(excluded), "seed_results": seed_results,
            "circuit_size_nodes": {"mean": 6.0, "std": 0.0, "min": 6.0,
                                   "max": 6.0, "n": n_succ},
            "node_jaccard": {"mean": 0.8, "std": 0.05, "min": 0.7,
                             "max": 0.9, "n": 6},
            "edge_jaccard_unweighted": {"mean": 0.6, "std": 0.0, "min": 0.6,
                                        "max": 0.6, "n": 6},
            "edge_jaccard_weighted": {"mean": 0.5, "std": 0.0, "min": 0.5,
                                      "max": 0.5, "n": 6},
            "max_universality": 1.0,
            "n_rank1_nodes": len(rank1),
            "rank1_nodes": [f"{k}#{i}" for k, i in rank1],
            "heldout_test_loss": {"mean": 0.12, "std": 0.0, "min": 0.12,
                                  "max": 0.12, "n": n_succ},
            "select_loss": {"mean": 0.1, "std": 0.0, "min": 0.1, "max": 0.1,
                            "n": n_succ},
            "n_generalize": n_succ, "generalize_rate": 1.0,
        }
        if extra_history_keys:
            summary.update(extra_history_keys)
        (it_dir / "iteration_summary.json").write_text(json.dumps(summary))
        history.append({k: v for k, v in summary.items() if k != "seed_results"})
        excluded.extend(rank1)
    state = {"next_iter": n_iters, "excluded": [list(x) for x in excluded],
             "history": history, "exhausted": False}
    (exp / "state.json").write_text(json.dumps(state))
    (exp / "hparams.json").write_text(json.dumps(
        {"k_coef": 1e-3, "weight_decay": 1e-3, "lr": 1e-2, "beta2": 0.95,
         "heaviside_temp": 1.0}))
    (exp / "run_args.json").write_text(json.dumps(
        {"model": "tiny", "task": "dummy_pronoun", "tokenizer": "fake",
         "target_loss": 0.15, "num_steps": 3, "batch_size": 4,
         "eval_batches": 1, "bisect_iters": 3, "split_over": "names_templates",
         "split_seed": 0, "test_frac": 0.2, "heldout_fold": 0,
         "name_pool": "pool.json", "backbone_thr": 0.6, "seed_offset": 0}))
    return exp
