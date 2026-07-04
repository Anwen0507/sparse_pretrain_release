"""Correctness tests for the full SparseGPT model: forward/loss semantics,
bigram table, embedding options, bridge-site forward variants, generation,
parameter counting, and checkpoint loading."""
import json

import pytest
import torch
import torch.nn.functional as F

from sparse_pretrain.src.config import SparsityConfig
from sparse_pretrain.src.model import SparseGPT, create_model
from tests.conftest import TINY_VOCAB, make_tiny_config, make_tiny_model


def rand_ids(batch=2, seq=8, high=TINY_VOCAB):
    return torch.randint(0, high, (batch, seq))


class TestForward:
    def test_logits_shape_and_no_loss_without_labels(self, tiny_model):
        ids = rand_ids()
        logits, loss, hidden = tiny_model(ids)
        assert logits.shape == (2, 8, TINY_VOCAB)
        assert loss is None and hidden is None

    def test_loss_is_shifted_cross_entropy(self, tiny_model):
        ids = rand_ids()
        logits, loss, _ = tiny_model(ids, labels=ids)
        expected = F.cross_entropy(
            logits[:, :-1].reshape(-1, TINY_VOCAB), ids[:, 1:].reshape(-1))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_loss_ignore_index(self, tiny_model):
        ids = rand_ids()
        labels = ids.clone()
        labels[:, -1] = -100  # ignored target
        logits, loss, _ = tiny_model(ids, labels=labels)
        expected = F.cross_entropy(
            logits[:, :-1].reshape(-1, TINY_VOCAB), labels[:, 1:].reshape(-1),
            ignore_index=-100)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_hidden_states_returned_per_block(self, tiny_model):
        ids = rand_ids()
        _, _, hidden = tiny_model(ids, return_hidden_states=True)
        # embedding output + one state per block
        assert len(hidden) == 1 + tiny_model.config.n_layer
        assert all(h.shape == (2, 8, 16) for h in hidden)

    def test_sequence_longer_than_context_asserts(self, tiny_model):
        with pytest.raises(AssertionError):
            tiny_model(rand_ids(seq=tiny_model.config.n_ctx + 1))

    def test_bigram_table_added_to_logits(self):
        torch.manual_seed(0)
        with_bigram = make_tiny_model()
        without = make_tiny_model(use_bigram_table=False)
        without.load_state_dict(
            {k: v for k, v in with_bigram.state_dict().items()
             if k != "bigram_table"})
        ids = rand_ids()
        lg_with, _, _ = with_bigram(ids)
        lg_without, _, _ = without(ids)
        expected_extra = F.embedding(ids, with_bigram.bigram_table)
        assert torch.allclose(lg_with - lg_without, expected_extra, atol=1e-5)

    def test_positional_embeddings_change_output(self):
        torch.manual_seed(0)
        model = make_tiny_model(use_positional_embeddings=True)
        assert model.wpe is not None
        ids = torch.full((1, 6), 7)
        logits, _, _ = model(ids)
        # with positions, identical tokens at different positions get
        # different next-token distributions even at position>0
        assert not torch.allclose(logits[0, 1], logits[0, 2], atol=1e-4)

    def test_activation_sparsity_applied_in_forward(self):
        sc = SparsityConfig(enable_weight_sparsity=False,
                            enable_activation_sparsity=True,
                            activation_topk_fraction=0.25,
                            activation_sparsity_locations="mlp_neuron")
        torch.manual_seed(0)
        sparse = make_tiny_model(sparsity=sc)
        dense = make_tiny_model()
        dense.load_state_dict(sparse.state_dict())
        ids = rand_ids()
        lg_sparse, _, _ = sparse(ids)
        lg_dense, _, _ = dense(ids)
        assert not torch.allclose(lg_sparse, lg_dense, atol=1e-4)
        # AbsTopK modules are cached per dimension
        fn = sparse._get_activation_sparsity_fn()
        x = torch.randn(1, 1, sparse.config.d_mlp)
        out = fn(x, "mlp_neuron")
        k = max(1, int(sparse.config.d_mlp * 0.25))
        assert (out != 0).sum() == k
        # non-listed location passes through untouched
        assert fn(x, "attn_in") is x

    def test_sparsity_disabled_by_default(self, tiny_model):
        assert tiny_model._get_activation_sparsity_fn() is None
        assert not tiny_model.sparsity_config.enable_activation_sparsity
        assert not tiny_model.sparsity_config.enable_weight_sparsity


class TestEmbeddingOptions:
    def test_untied_by_default(self, tiny_model):
        assert tiny_model.lm_head.weight is not tiny_model.wte.weight

    def test_tied_embeddings_share_storage(self):
        model = make_tiny_model(tie_embeddings=True)
        assert model.lm_head.weight is model.wte.weight

    def test_get_num_params(self):
        model = make_tiny_model(use_positional_embeddings=True)
        total = sum(p.numel() for p in model.parameters())
        assert model.get_num_params(non_embedding=False) == total
        assert model.get_num_params(non_embedding=True) == \
            total - model.wpe.weight.numel()
        no_pos = make_tiny_model()
        total2 = sum(p.numel() for p in no_pos.parameters())
        assert no_pos.get_num_params(non_embedding=True) == total2


class TestBridgeSites:
    def test_bridge_site_count_and_final_logits_match_forward(self, tiny_model):
        ids = rand_ids()
        logits_direct, _, _ = tiny_model(ids)
        logits, pre, post = tiny_model.forward_with_bridge_sites(ids)
        L = tiny_model.config.n_layer
        assert len(pre) == len(post) == 2 * L + 1
        assert torch.allclose(logits, logits_direct, atol=1e-5)
        # without activation sparsity, pre and post are the same tensors
        for p, q in zip(pre, post):
            assert p is q

    def test_forward_from_every_site_reproduces_logits(self, tiny_model):
        ids = rand_ids()
        logits, _, post = tiny_model.forward_with_bridge_sites(ids)
        for site_idx, h in enumerate(post):
            resumed = tiny_model.forward_from_site(h, site_idx, input_ids=ids)
            assert torch.allclose(resumed, logits, atol=1e-5), \
                f"site {site_idx} does not reproduce final logits"

    def test_forward_from_site_without_input_ids_drops_bigram(self, tiny_model):
        ids = rand_ids()
        logits, _, post = tiny_model.forward_with_bridge_sites(ids)
        resumed = tiny_model.forward_from_site(post[0], 0, input_ids=None)
        expected_gap = F.embedding(ids, tiny_model.bigram_table)
        assert torch.allclose(logits - resumed, expected_gap, atol=1e-5)


class TestGenerate:
    def test_appends_requested_tokens(self, tiny_model):
        out = tiny_model.generate(rand_ids(1, 4), max_new_tokens=5)
        assert out.shape == (1, 9)

    def test_top_k_1_is_greedy(self, tiny_model):
        ids = rand_ids(1, 4)
        out = tiny_model.generate(ids.clone(), max_new_tokens=1, top_k=1)
        logits, _, _ = tiny_model(ids)
        assert out[0, -1] == logits[0, -1].argmax()

    def test_crops_context_when_input_exceeds_n_ctx(self):
        model = make_tiny_model(n_ctx=8)
        ids = rand_ids(1, 8)
        out = model.generate(ids, max_new_tokens=2)
        assert out.shape == (1, 10)


class TestFactoryAndPretrained:
    def test_create_model(self, tiny_config):
        model = create_model(tiny_config)
        assert isinstance(model, SparseGPT)
        assert model.config is tiny_config

    def test_from_pretrained_local_files(self, tmp_path, monkeypatch):
        """from_pretrained round-trips config+weights via hf_hub_download,
        which we point at local files."""
        src = make_tiny_model()
        cfg = {"model_config": {"n_layer": 2, "d_model": 16, "n_ctx": 32,
                                "d_head": 4, "d_mlp": 24,
                                "vocab_size": TINY_VOCAB},
               "sparsity_config": {"enable_weight_sparsity": False,
                                   "enable_activation_sparsity": False}}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        torch.save(src.state_dict(), tmp_path / "pytorch_model.bin")

        def fake_download(repo_id, filename):
            return str(tmp_path / filename)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        loaded = SparseGPT.from_pretrained("user/tiny", device="cpu")
        assert not loaded.training  # eval mode
        ids = rand_ids()
        a, _, _ = src(ids)
        b, _, _ = loaded(ids)
        assert torch.allclose(a, b, atol=1e-6)


def test_init_weights_statistics():
    """Linear/embedding weights ~ N(0, 0.02); c_proj gets the scaled init;
    biases start at zero."""
    torch.manual_seed(0)
    model = make_tiny_model(d_model=64, d_mlp=256, n_layer=2)
    assert model.blocks[0].mlp.c_fc.bias.abs().max() == 0
    std = model.wte.weight.std().item()
    assert 0.015 < std < 0.025
    import math
    expected_proj_std = 0.02 / math.sqrt(2 * 2)
    proj_std = model.blocks[0].attn.c_proj.weight.std().item()
    assert abs(proj_std - expected_proj_std) / expected_proj_std < 0.25
