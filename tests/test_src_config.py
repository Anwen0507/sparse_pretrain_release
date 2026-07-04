"""Tests for the training-side config dataclasses (sparse_pretrain.src.config)
and the pruning config (sparse_pretrain.src.pruning.config)."""
import pytest

from sparse_pretrain.src.config import (
    Config, ModelConfig, OptimizerConfig, SparsityConfig, TrainingConfig,
)
from sparse_pretrain.src.pruning.config import PruningConfig


class TestModelConfig:
    def test_d_mlp_defaults_to_4x_d_model(self):
        assert ModelConfig(d_model=32).d_mlp == 128

    def test_explicit_d_mlp_kept(self):
        assert ModelConfig(d_model=32, d_mlp=48).d_mlp == 48

    def test_n_heads(self):
        assert ModelConfig(d_model=64, d_head=16).n_heads == 4

    def test_n_heads_requires_divisibility(self):
        with pytest.raises(AssertionError):
            _ = ModelConfig(d_model=30, d_head=16).n_heads

    def test_paper_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_layer == 8
        assert cfg.use_rms_norm and cfg.use_bigram_table and cfg.use_attention_sinks
        assert not cfg.tie_embeddings and not cfg.use_positional_embeddings
        assert cfg.dropout == 0.0


class TestSparsityConfig:
    def test_defaults(self):
        sc = SparsityConfig()
        assert sc.target_l0_fraction == pytest.approx(1 / 64)
        assert sc.activation_topk_fraction == 0.25
        locs = set(sc.activation_sparsity_locations.split(","))
        assert locs == {"attn_in", "attn_out", "mlp_in", "mlp_out",
                        "mlp_neuron", "attn_v", "attn_k", "attn_q"}


def test_optimizer_and_training_defaults():
    oc = OptimizerConfig()
    assert oc.eps == 0.1  # the paper's unusually large epsilon
    assert oc.enable_grad_clip and oc.grad_clip_rms == 1.0
    tc = TrainingConfig()
    assert tc.mixed_precision == "bf16"
    assert tc.seed == 0


class TestConfigYaml:
    def test_round_trip(self, tmp_path):
        cfg = Config(
            model=ModelConfig(n_layer=3, d_model=48, d_head=16, vocab_size=99),
            sparsity=SparsityConfig(target_l0_fraction=0.5),
            optimizer=OptimizerConfig(learning_rate=1e-3),
            training=TrainingConfig(total_tokens=1234, batch_size=2),
        )
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(str(path))
        loaded = Config.from_yaml(str(path))
        assert loaded == cfg

    def test_from_yaml_partial_sections_use_defaults(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("model:\n  n_layer: 5\n")
        cfg = Config.from_yaml(str(path))
        assert cfg.model.n_layer == 5
        assert cfg.training == TrainingConfig()

    def test_to_dict_structure(self):
        d = Config().to_dict()
        assert set(d) == {"model", "sparsity", "optimizer", "training"}
        assert d["model"]["n_layer"] == 8


class TestPruningConfig:
    def test_default_mask_locations(self):
        pc = PruningConfig()
        assert pc.mask_locations == ["attn_in", "attn_q", "attn_k", "attn_v",
                                     "attn_out", "mlp_in", "mlp_neuron",
                                     "mlp_out"]

    def test_defaults(self):
        pc = PruningConfig()
        assert pc.target_loss == 0.15
        assert pc.ablation_type == "mean_pretrain"
        assert pc.lr_warmup_frac == 0.0  # "no warmup" per the paper text
        assert not pc.use_binary_loss and not pc.mask_token_embeds

    def test_yaml_round_trip(self, tmp_path):
        pc = PruningConfig(target_loss=0.3, k_coef=2e-4, device="cpu",
                           mask_locations=["attn_in", "mlp_out"])
        path = tmp_path / "pc.yaml"
        pc.to_yaml(str(path))
        assert PruningConfig.from_yaml(str(path)) == pc

    def test_to_dict(self):
        d = PruningConfig().to_dict()
        assert d["heaviside_temp"] == 1.0
        assert isinstance(d["mask_locations"], list)
