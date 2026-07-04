"""Correctness tests for the model building blocks in sparse_pretrain.src.model:
AbsTopK, SDPAWithSink (vs a hand-rolled softmax reference), CausalSelfAttention
(flash vs manual path), MLP, and TransformerBlock forward variants."""
import math

import pytest
import torch
import torch.nn.functional as F

from sparse_pretrain.src.model import (
    AbsTopK, CausalSelfAttention, MLP, SDPAWithSink, TransformerBlock,
)
from tests.conftest import make_tiny_config


class TestAbsTopK:
    def test_keeps_k_largest_by_magnitude(self):
        x = torch.tensor([[1.0, -5.0, 3.0, -2.0, 0.5]])
        out = AbsTopK(k=2)(x)
        assert torch.equal(out, torch.tensor([[0.0, -5.0, 3.0, 0.0, 0.0]]))

    def test_sign_preserved_and_values_untouched(self):
        x = torch.randn(4, 7, 32)
        out = AbsTopK(k=5)(x)
        kept = out != 0
        assert kept.sum(-1).eq(5).all()
        assert torch.equal(out[kept], x[kept])
        # every zeroed entry is <= the smallest kept magnitude, per row
        min_kept = out.abs().masked_fill(~kept, float("inf")).amin(-1)
        max_dropped = x.abs().masked_fill(kept, 0.0).amax(-1)
        assert (max_dropped <= min_kept + 1e-12).all()

    def test_k_geq_dim_is_identity(self):
        x = torch.randn(3, 4)
        assert AbsTopK(k=4)(x) is x
        assert AbsTopK(k=10)(x) is x

    def test_gradient_flows_only_through_kept_entries(self):
        x = torch.tensor([1.0, -5.0, 3.0, -2.0], requires_grad=True)
        AbsTopK(k=2)(x).sum().backward()
        assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_extra_repr(self):
        assert AbsTopK(k=3).extra_repr() == "k=3"


def reference_sink_attention(q, k, v, sink_logit, scale):
    """Direct softmax implementation of causal attention with a per-head
    learnable sink slot (zero value vector, always visible)."""
    B, H, L, D = q.shape
    out = torch.zeros_like(v[..., :q.shape[2], :])
    for b in range(B):
        for h in range(H):
            for i in range(L):
                logits = [sink_logit[h]]
                for j in range(i + 1):  # causal: keys 0..i
                    logits.append((q[b, h, i] * k[b, h, j]).sum() * scale)
                probs = torch.softmax(torch.stack(logits), dim=0)
                acc = torch.zeros(v.shape[-1])
                for j in range(i + 1):
                    acc = acc + probs[j + 1] * v[b, h, j]
                out[b, h, i] = acc  # sink contributes a zero vector
    return out


class TestSDPAWithSink:
    def test_matches_reference_softmax(self):
        torch.manual_seed(3)
        B, H, L, D = 2, 3, 5, 4
        q, k, v = torch.randn(3, B, H, L, D).unbind(0)
        attn = SDPAWithSink(n_heads=H)
        with torch.no_grad():
            attn.sink_logit.copy_(torch.tensor([0.3, -0.7, 1.2]))
        scale = 1.0 / math.sqrt(D)
        out = attn(q, k, v, is_causal=True, scale=scale)
        ref = reference_sink_attention(q, k, v, attn.sink_logit, scale)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_very_negative_sink_recovers_plain_causal_sdpa(self):
        B, H, L, D = 1, 2, 6, 4
        q, k, v = torch.randn(3, B, H, L, D).unbind(0)
        attn = SDPAWithSink(n_heads=H, init_logit=-1e4)
        out = attn(q, k, v, is_causal=True, scale=1.0 / math.sqrt(D))
        plain = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               scale=1.0 / math.sqrt(D))
        assert torch.allclose(out, plain, atol=1e-5)

    def test_large_sink_absorbs_probability_mass(self):
        B, H, L, D = 1, 1, 4, 4
        q, k, v = torch.randn(3, B, H, L, D).unbind(0)
        big = SDPAWithSink(n_heads=H, init_logit=30.0)
        out = big(q, k, v, is_causal=True, scale=1.0)
        # nearly all mass on the zero-valued sink -> output ~ 0
        assert out.abs().max() < 1e-3

    def test_causality(self):
        """Changing a future key/value must not change earlier outputs."""
        B, H, L, D = 1, 2, 5, 4
        q, k, v = torch.randn(3, B, H, L, D).unbind(0)
        attn = SDPAWithSink(n_heads=H)
        out1 = attn(q, k, v, is_causal=True, scale=0.5)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, -1], v2[:, :, -1] = 100.0, 100.0
        out2 = attn(q, k2, v2, is_causal=True, scale=0.5)
        assert torch.allclose(out1[:, :, :-1], out2[:, :, :-1], atol=1e-5)
        assert not torch.allclose(out1[:, :, -1], out2[:, :, -1], atol=1e-3)

    def test_sink_logit_is_learnable(self):
        attn = SDPAWithSink(n_heads=2)
        q, k, v = torch.randn(3, 1, 2, 3, 4).unbind(0)
        attn(q, k, v, is_causal=True, scale=0.5).sum().backward()
        assert attn.sink_logit.grad is not None
        assert attn.sink_logit.grad.abs().sum() > 0


class TestCausalSelfAttention:
    def test_flash_and_manual_paths_agree_without_sinks(self):
        cfg = make_tiny_config(use_attention_sinks=False)
        torch.manual_seed(0)
        flash = CausalSelfAttention(cfg)
        cfg_manual = make_tiny_config(use_attention_sinks=False,
                                      use_flash_attention=False)
        manual = CausalSelfAttention(cfg_manual)
        # the manual variant has an extra causal_mask buffer
        manual.load_state_dict(flash.state_dict(), strict=False)
        flash.eval(), manual.eval()
        x = torch.randn(2, 7, cfg.d_model)
        assert torch.allclose(flash(x), manual(x), atol=1e-5)

    def test_sink_module_created_only_when_enabled(self):
        assert isinstance(
            CausalSelfAttention(make_tiny_config()).attn_fn, SDPAWithSink)
        assert CausalSelfAttention(
            make_tiny_config(use_attention_sinks=False)).attn_fn is None
        # use_sinks=False argument overrides the config
        assert CausalSelfAttention(make_tiny_config(), use_sinks=False).attn_fn is None

    def test_output_shape_and_causality(self):
        cfg = make_tiny_config()
        attn = CausalSelfAttention(cfg).eval()
        x = torch.randn(2, 9, cfg.d_model)
        y = attn(x)
        assert y.shape == x.shape
        x2 = x.clone()
        x2[:, -1] += 10.0
        y2 = attn(x2)
        assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5)

    def test_activation_sparsity_fn_called_on_qkv(self):
        cfg = make_tiny_config()
        attn = CausalSelfAttention(cfg).eval()
        seen = []

        def spy(x, loc):
            seen.append(loc)
            return x

        attn(torch.randn(1, 4, cfg.d_model), activation_sparsity_fn=spy)
        assert seen == ["attn_q", "attn_k", "attn_v"]


class TestMLP:
    def test_matches_manual_computation_gelu(self):
        cfg = make_tiny_config()
        mlp = MLP(cfg).eval()
        x = torch.randn(2, 5, cfg.d_model)
        expected = mlp.c_proj(F.gelu(mlp.c_fc(x)))
        assert torch.allclose(mlp(x), expected, atol=1e-6)

    def test_relu_activation_selected(self):
        mlp = MLP(make_tiny_config(activation="relu"))
        assert isinstance(mlp.act_fn, torch.nn.ReLU)

    def test_sparsity_fn_applied_to_neurons(self):
        cfg = make_tiny_config()
        mlp = MLP(cfg).eval()
        x = torch.randn(1, 3, cfg.d_model)

        def zero_neurons(t, loc):
            assert loc == "mlp_neuron"
            return torch.zeros_like(t)

        out = mlp(x, activation_sparsity_fn=zero_neurons)
        # all neurons zeroed -> output is just the projection bias
        expected = mlp.c_proj.bias.expand_as(out)
        assert torch.allclose(out, expected, atol=1e-6)


class TestTransformerBlock:
    def test_forward_with_intermediate_consistency(self):
        cfg = make_tiny_config()
        block = TransformerBlock(cfg).eval()
        x = torch.randn(2, 6, cfg.d_model)
        out = block(x)
        out2, post_attn = block.forward_with_intermediate(x)
        assert torch.allclose(out, out2, atol=1e-6)
        # post_attn is the state after the attention residual add
        normed = block.ln_1(x)
        assert torch.allclose(post_attn, x + block.attn(normed), atol=1e-6)

    def test_forward_from_post_attn_completes_the_block(self):
        cfg = make_tiny_config()
        block = TransformerBlock(cfg).eval()
        x = torch.randn(2, 6, cfg.d_model)
        full, post_attn = block.forward_with_intermediate(x)
        resumed = block.forward_from_post_attn(post_attn)
        assert torch.allclose(full, resumed, atol=1e-6)

    def test_layernorm_variant(self):
        block = TransformerBlock(make_tiny_config(use_rms_norm=False))
        assert isinstance(block.ln_1, torch.nn.LayerNorm)
        rms_block = TransformerBlock(make_tiny_config())
        assert isinstance(rms_block.ln_1, torch.nn.RMSNorm)

    def test_sparsity_fn_locations_in_order(self):
        cfg = make_tiny_config()
        block = TransformerBlock(cfg).eval()
        seen = []

        def spy(t, loc):
            seen.append(loc)
            return t

        block(torch.randn(1, 4, cfg.d_model), activation_sparsity_fn=spy)
        assert seen == ["resid_pre", "attn_in", "attn_q", "attn_k", "attn_v",
                        "attn_out", "resid_mid", "mlp_in", "mlp_neuron",
                        "mlp_out"]
