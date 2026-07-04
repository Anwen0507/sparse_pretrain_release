"""Correctness tests for MaskedSparseGPT.

The strongest checks are exact equivalences:
  * fully-active masks  ==  the wrapped base model (with and without AbsTopK),
  * a masked-out token embedding  ==  a base model whose wte row was replaced,
  * frozen-LN forward with fully-active masks  ==  the ordinary forward,
plus formula-level checks of the task/binary losses and the mean cache.
"""
import pytest
import torch
import torch.nn.functional as F

from sparse_pretrain.src.config import SparsityConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from tests.conftest import (
    TINY_VOCAB, ToyTask, make_masked, make_tiny_model, set_all_taus,
)


def rand_ids(batch=2, seq=8):
    return torch.randint(0, TINY_VOCAB, (batch, seq))


class TestForwardEquivalence:
    def test_fully_active_equals_base_model(self):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        ids = rand_ids()
        base_logits, _, _ = mm.model(ids)
        assert torch.allclose(mm(ids), base_logits, atol=1e-5)

    def test_fully_active_equals_base_with_activation_sparsity(self):
        sc = SparsityConfig(enable_weight_sparsity=False,
                            enable_activation_sparsity=True)
        model = make_tiny_model(sparsity=sc)
        mm = make_masked(model)
        assert mm._activation_sparsity_enabled
        set_all_taus(mm, 1.0)
        ids = rand_ids()
        base_logits, _, _ = model(ids)
        assert torch.allclose(mm(ids), base_logits, atol=1e-5)

    def test_base_parameters_are_frozen(self):
        mm = make_masked()
        assert all(not p.requires_grad for p in mm.model.parameters())
        assert all(p.requires_grad for p in mm.masks.parameters())

    def test_masking_changes_output_zero_ablation(self):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        ids = rand_ids()
        full = mm(ids)
        with torch.no_grad():
            mm.masks.get_mask(0, "mlp_neuron").tau.fill_(-1.0)
        pruned = mm(ids)
        assert not torch.allclose(full, pruned, atol=1e-4)

    def test_mean_ablation_differs_from_zero_ablation(self):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        with torch.no_grad():
            mm.masks.get_mask(0, "mlp_neuron").tau.fill_(-1.0)
        ids = rand_ids()
        zero_out = mm(ids)
        mm.masks.get_mask(0, "mlp_neuron").set_mean(
            torch.full((mm.d_mlp,), 3.0))
        mean_out = mm(ids)
        assert not torch.allclose(zero_out, mean_out, atol=1e-4)

    def test_return_logits_only_flag(self):
        mm = make_masked()
        ids = rand_ids()
        out = mm.forward(ids, return_logits_only=False)
        assert isinstance(out, tuple) and out[1] is None and out[2] is None

    def test_abstopk_matches_base_model_function(self):
        sc = SparsityConfig(enable_activation_sparsity=True,
                            activation_topk_fraction=0.25)
        model = make_tiny_model(sparsity=sc)
        mm = make_masked(model)
        x = torch.randn(2, 3, 16)
        base_fn = model._get_activation_sparsity_fn()
        assert torch.equal(mm._apply_abstopk(x, "attn_in"),
                           base_fn(x, "attn_in"))
        # location not under sparsity passes through
        assert mm._apply_abstopk(x, "resid_pre") is x


class TestTokenMask:
    def test_disabled_by_default(self):
        assert make_masked().token_mask is None

    def test_masked_token_equals_wte_row_substitution(self):
        """Masking token T with mean m is exactly a model with wte[T] := m."""
        model = make_tiny_model()
        mm = make_masked(model, mask_token_embeds=True)
        set_all_taus(mm, 1.0)
        T = 7
        mean_embed = torch.randn(16)
        with torch.no_grad():
            mm.token_mask.tau.fill_(1.0)
            mm.token_mask.tau[T] = -1.0
        mm.token_mask.register_buffer("mean_activation", mean_embed.clone())
        mm.token_mask.mean_set = True

        ids = torch.tensor([[3, T, 5, T, 9]])
        out = mm(ids)

        swapped = make_tiny_model()
        swapped.load_state_dict(model.state_dict())
        with torch.no_grad():
            swapped.wte.weight[T] = mean_embed
        mm2 = make_masked(swapped)
        set_all_taus(mm2, 1.0)
        assert torch.allclose(out, mm2(ids), atol=1e-5)

    def test_zero_ablation_when_no_mean(self):
        model = make_tiny_model()
        mm = make_masked(model, mask_token_embeds=True)
        set_all_taus(mm, 1.0)
        T = 4
        with torch.no_grad():
            mm.token_mask.tau.fill_(1.0)
            mm.token_mask.tau[T] = -1.0
        ids = torch.tensor([[T, 3]])
        out = mm(ids)
        zeroed = make_tiny_model()
        zeroed.load_state_dict(model.state_dict())
        with torch.no_grad():
            zeroed.wte.weight[T] = 0.0
        mm2 = make_masked(zeroed)
        set_all_taus(mm2, 1.0)
        assert torch.allclose(out, mm2(ids), atol=1e-5)


class TestTaskLoss:
    def test_matches_manual_cross_entropy(self):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        mm.eval()
        task = ToyTask(seq_len=6)
        pos, neg, corr, inc, ep = task.generate_batch(4)
        loss, metrics = mm.compute_task_loss(pos, neg, corr, inc, ep)
        with torch.no_grad():
            logits = mm(pos)
            final = logits[torch.arange(4), ep]
            expected = F.cross_entropy(final, corr)
            expected_acc = (final.argmax(-1) == corr).float().mean().item()
            expected_diff = (final[torch.arange(4), corr]
                             - final[torch.arange(4), inc]).mean().item()
        assert torch.allclose(loss, expected, atol=1e-5)
        assert metrics["task_loss"] == pytest.approx(expected.item(), abs=1e-5)
        assert metrics["accuracy"] == pytest.approx(expected_acc)
        assert metrics["logit_diff"] == pytest.approx(expected_diff, abs=1e-4)
        assert "binary_accuracy" not in metrics

    def test_eval_positions_are_honored(self):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        pos = rand_ids(2, 6)
        corr = torch.tensor([3, 4])
        inc = torch.tensor([5, 6])
        l_last, _ = mm.compute_task_loss(pos, pos, corr, inc,
                                         torch.tensor([5, 5]))
        l_default, _ = mm.compute_task_loss(pos, pos, corr, inc, None)
        assert torch.allclose(l_last, l_default, atol=1e-6)
        l_mid, _ = mm.compute_task_loss(pos, pos, corr, inc,
                                        torch.tensor([2, 3]))
        assert not torch.allclose(l_mid, l_last, atol=1e-5)

    def test_binary_loss_is_softplus_of_logit_gap(self):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        task = ToyTask()
        pos, neg, corr, inc, ep = task.generate_batch(5)
        loss, metrics = mm.compute_task_loss(pos, neg, corr, inc, ep,
                                             use_binary_loss=True)
        with torch.no_grad():
            final = mm(pos)[torch.arange(5), ep]
            c = final[torch.arange(5), corr]
            i = final[torch.arange(5), inc]
            expected = F.softplus(-(c - i)).mean()
            expected_bin_acc = (c > i).float().mean().item()
        assert torch.allclose(loss, expected, atol=1e-5)
        assert metrics["binary_accuracy"] == pytest.approx(expected_bin_acc)

    def test_binary_loss_defaults_to_config(self):
        mm = make_masked(use_binary_loss=True)
        set_all_taus(mm, 1.0)
        task = ToyTask()
        pos, neg, corr, inc, ep = task.generate_batch(3)
        _, metrics = mm.compute_task_loss(pos, neg, corr, inc, ep)
        assert "binary_accuracy" in metrics

    def test_compute_full_loss_arithmetic(self):
        mm = make_masked()
        set_all_taus(mm, -1.0)
        with torch.no_grad():
            mm.masks.get_mask(0, "attn_in").tau[:6] = 1.0
        task = ToyTask()
        pos, neg, corr, inc, ep = task.generate_batch(3)
        k_coef = 0.01
        total, metrics = mm.compute_full_loss(pos, neg, corr, inc, k_coef, ep)
        assert metrics["num_active_nodes"] == 6
        assert metrics["sparsity_loss"] == pytest.approx(6.0)
        assert metrics["total_loss"] == pytest.approx(
            metrics["task_loss"] + k_coef * 6.0, rel=1e-5)
        assert metrics["total_nodes"] == mm.masks.get_total_nodes()

    def test_compute_full_loss_includes_token_mask(self):
        mm = make_masked(mask_token_embeds=True)
        set_all_taus(mm, -1.0)
        with torch.no_grad():
            mm.token_mask.tau.fill_(-1.0)
            mm.token_mask.tau[:10] = 1.0
            mm.masks.get_mask(0, "attn_in").tau[:2] = 1.0
        task = ToyTask()
        pos, neg, corr, inc, ep = task.generate_batch(2)
        _, metrics = mm.compute_full_loss(pos, neg, corr, inc, 0.0, ep)
        assert metrics["num_active_tokens"] == 10
        assert metrics["total_tokens"] == TINY_VOCAB
        assert metrics["token_sparsity_loss"] == pytest.approx(10.0)
        assert metrics["sparsity_loss"] == pytest.approx(12.0)  # 2 nodes + 10 tokens


class TestFrozenLayerNorm:
    def test_rms_scale_formula(self):
        mm = make_masked()
        ln = torch.nn.RMSNorm(16, eps=1e-5)
        x = torch.randn(2, 3, 16)
        scale = mm._compute_ln_rms_scale(x, ln)
        expected = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        assert torch.allclose(scale, expected, atol=1e-6)

    def test_layernorm_scale_formula(self):
        mm = make_masked()
        ln = torch.nn.LayerNorm(16)
        x = torch.randn(2, 3, 16)
        scale = mm._compute_ln_rms_scale(x, ln)
        expected = 1.0 / torch.sqrt(
            x.var(-1, keepdim=True, unbiased=False) + ln.eps)
        assert torch.allclose(scale, expected, atol=1e-6)

    def test_apply_frozen_scale_matches_live_norm_on_same_input(self):
        mm = make_masked()
        ln = torch.nn.RMSNorm(16, eps=1e-6)
        with torch.no_grad():
            ln.weight.copy_(torch.rand(16) + 0.5)
        x = torch.randn(4, 16)
        frozen = mm._apply_ln_with_frozen_scale(
            x, ln, mm._compute_ln_rms_scale(x, ln))
        assert torch.allclose(frozen, ln(x), atol=1e-5)

    def test_apply_frozen_scale_layernorm_matches_live(self):
        mm = make_masked()
        ln = torch.nn.LayerNorm(16)
        with torch.no_grad():
            ln.weight.copy_(torch.rand(16) + 0.5)
            ln.bias.copy_(torch.randn(16))
        x = torch.randn(4, 16)
        frozen = mm._apply_ln_with_frozen_scale(
            x, ln, mm._compute_ln_rms_scale(x, ln))
        assert torch.allclose(frozen, ln(x), atol=1e-5)

    def test_freeze_with_fully_active_masks_equals_plain_forward(self):
        # The frozen path assumes eps=1e-6 when RMSNorm has eps=None; pin the
        # model's norms to that value so the equivalence is exact.
        model = make_tiny_model()
        for module in model.modules():
            if isinstance(module, torch.nn.RMSNorm):
                module.eps = 1e-6
        mm_frozen = make_masked(model, freeze_layernorm_scale=True)
        set_all_taus(mm_frozen, 1.0)
        ids = rand_ids()
        base_logits, _, _ = model(ids)
        # unpruned residuals == pruned residuals -> frozen scales identical
        assert torch.allclose(mm_frozen(ids), base_logits, atol=1e-5)

    def test_freeze_default_eps_mismatch_is_small(self):
        """With eps=None RMSNorm (the shipped default), the frozen path uses
        1e-6 instead of finfo.eps; on untrained-scale activations this is a
        small approximation, not an exact identity."""
        model = make_tiny_model()
        mm_frozen = make_masked(model, freeze_layernorm_scale=True)
        set_all_taus(mm_frozen, 1.0)
        ids = rand_ids()
        base_logits, _, _ = model(ids)
        frozen = mm_frozen(ids)
        rel = (frozen - base_logits).abs().max() / base_logits.abs().max()
        assert rel < 0.05

    def test_frozen_scales_captured_per_layer(self):
        mm = make_masked(freeze_layernorm_scale=True)
        ids = rand_ids(1, 5)
        scales = mm._compute_frozen_ln_scales(ids)
        assert set(scales) == {"layer0_ln_1", "layer0_ln_2", "layer1_ln_1",
                               "layer1_ln_2", "ln_f"}
        assert all(v.shape == (1, 5, 1) for v in scales.values())

    def test_gradients_flow_with_frozen_scales(self):
        mm = make_masked(freeze_layernorm_scale=True)
        task = ToyTask()
        pos, neg, corr, inc, ep = task.generate_batch(2)
        loss, _ = mm.compute_task_loss(pos, neg, corr, inc, ep)
        loss.backward()
        grads = [m.tau.grad for m in mm.masks.masks.values()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


class TestMeanCache:
    @staticmethod
    def manual_attn_in_mean(model, batches):
        acts = []
        with torch.no_grad():
            for ids in batches:
                x = model.wte(ids)
                acts.append(model.blocks[0].ln_1(x).reshape(-1, 16))
        return torch.cat(acts).mean(0)

    def test_attn_in_mean_matches_manual(self):
        model = make_tiny_model()
        mm = make_masked(model)
        batches = [rand_ids(2, 6), rand_ids(3, 6)]
        cache = mm.compute_mean_cache(iter(batches), num_batches=2,
                                      show_progress=False)
        expected_keys = {f"layer{l}_{loc}" for l in range(2)
                         for loc in mm.config.mask_locations}
        assert set(cache) == expected_keys
        expected = self.manual_attn_in_mean(model, batches)
        assert torch.allclose(cache["layer0_attn_in"], expected, atol=1e-5)

    def test_running_mean_equals_concat_mean(self):
        model = make_tiny_model()
        mm = make_masked(model)
        b1, b2 = rand_ids(2, 4), rand_ids(2, 4)
        two = mm.compute_mean_cache(iter([b1, b2]), 2, show_progress=False)
        one = mm.compute_mean_cache(iter([torch.cat([b1, b2])]), 1,
                                    show_progress=False)
        for key in two:
            assert torch.allclose(two[key], one[key], atol=1e-5), key

    def test_dict_batches_accepted(self):
        mm = make_masked()
        cache = mm.compute_mean_cache(
            iter([{"input_ids": rand_ids(2, 4)}]), 1, show_progress=False)
        assert "layer0_mlp_neuron" in cache

    def test_token_embed_mean(self):
        model = make_tiny_model()
        mm = make_masked(model, mask_token_embeds=True)
        ids = rand_ids(2, 5)
        cache = mm.compute_mean_cache(iter([ids]), 1, show_progress=False)
        expected = model.wte(ids).reshape(-1, 16).mean(0)
        assert torch.allclose(cache["token_embed"], expected, atol=1e-5)
        mm.set_means_from_dict(cache)
        assert mm.token_mask.mean_set
        assert torch.allclose(mm.token_mask.mean_activation, expected,
                              atol=1e-5)

    def test_num_batches_limit_respected(self):
        mm = make_masked()

        def endless():
            while True:
                yield rand_ids(1, 4)

        cache = mm.compute_mean_cache(endless(), num_batches=3,
                                      show_progress=False)
        assert cache  # terminates


class TestMaskState:
    def test_round_trip(self):
        mm = make_masked(mask_token_embeds=True)
        state = mm.get_mask_state()
        assert "token_embed" in state
        set_all_taus(mm, -1.0)
        mm.load_mask_state(state)
        for key, tau in mm.get_mask_state().items():
            assert torch.equal(tau, state[key])

    def test_state_is_a_copy(self):
        mm = make_masked()
        state = mm.get_mask_state()
        set_all_taus(mm, -1.0)
        assert any(not torch.equal(v, mm.masks.masks[k].tau)
                   for k, v in state.items() if k != "token_embed")

    def test_load_ignores_unknown_keys(self):
        mm = make_masked()
        mm.load_mask_state({"bogus_key": torch.zeros(3)})  # no error

    def test_get_circuit_mask_binary(self):
        mm = make_masked(mask_token_embeds=True)
        set_all_taus(mm, -1.0)
        with torch.no_grad():
            mm.masks.get_mask(1, "attn_q").tau[2] = 0.3
        circuit = mm.get_circuit_mask()
        assert circuit["layer1_attn_q"][2] == 1.0
        assert circuit["layer1_attn_q"].sum() == 1.0
        assert circuit["token_embed"].sum() == 0.0
        for value in circuit.values():
            assert set(value.unique().tolist()) <= {0.0, 1.0}

    def test_clamp_mask_parameters_covers_token_mask(self):
        mm = make_masked(mask_token_embeds=True)
        with torch.no_grad():
            mm.token_mask.tau.fill_(5.0)
        mm.clamp_mask_parameters()
        assert mm.token_mask.tau.max() == 1.0

    def test_count_edges_is_node_count_proxy(self):
        mm = make_masked()
        assert mm.count_edges() == mm.masks.get_total_active_nodes()
