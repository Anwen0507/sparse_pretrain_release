"""Correctness tests for the mask machinery in
sparse_pretrain.src.pruning.node_mask: the Heaviside/STE autograd function,
single NodeMask semantics (zero vs mean ablation), and NodeMaskCollection
bookkeeping (dims, top-k discretization, sparsity loss)."""
import pytest
import torch

from sparse_pretrain.src.pruning.node_mask import (
    HeavisideSTE, NodeLocation, NodeMask, NodeMaskCollection, heaviside_ste,
)


class TestHeavisideSTE:
    def test_forward_is_step_at_zero(self):
        x = torch.tensor([-2.0, -1e-9, 0.0, 1e-9, 3.0])
        assert torch.equal(heaviside_ste(x),
                           torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    def test_backward_is_sigmoid_derivative(self, temperature):
        x = torch.linspace(-3, 3, 13, requires_grad=True)
        upstream = torch.randn(13)
        heaviside_ste(x, temperature).backward(upstream)
        s = torch.sigmoid(x.detach() / temperature)
        expected = upstream * s * (1 - s) / temperature
        assert torch.allclose(x.grad, expected, atol=1e-6)

    def test_apply_alias(self):
        x = torch.tensor([1.0])
        assert HeavisideSTE.apply(x, 1.0).item() == 1.0


def test_node_location_dataclass():
    loc = NodeLocation(layer=3, location_type="mlp_neuron", dim=128)
    assert (loc.layer, loc.location_type, loc.dim) == (3, "mlp_neuron", 128)


class TestNodeMask:
    def test_init_clamped_and_biased_positive(self):
        torch.manual_seed(0)
        m = NodeMask(1000, init_noise_scale=0.01, init_noise_bias=0.1)
        assert m.tau.min() >= -1.0 and m.tau.max() <= 1.0
        assert 0.05 < m.tau.mean() < 0.15
        assert not m.mean_set

    def test_clamp_parameters(self):
        m = NodeMask(4)
        with torch.no_grad():
            m.tau.copy_(torch.tensor([-5.0, -0.5, 0.5, 5.0]))
        m.clamp_parameters()
        assert torch.equal(m.tau, torch.tensor([-1.0, -0.5, 0.5, 1.0]))

    def test_binary_mask_and_num_active(self):
        m = NodeMask(4)
        with torch.no_grad():
            m.tau.copy_(torch.tensor([-1.0, -0.1, 0.0, 0.7]))
        assert torch.equal(m.get_binary_mask(), torch.tensor([0.0, 0.0, 1.0, 1.0]))
        assert m.get_num_active() == 2

    def test_forward_zero_ablation_when_mean_unset(self):
        m = NodeMask(3)
        with torch.no_grad():
            m.tau.copy_(torch.tensor([1.0, -1.0, 1.0]))
        x = torch.tensor([[10.0, 20.0, 30.0]])
        assert torch.equal(m(x), torch.tensor([[10.0, 0.0, 30.0]]))

    def test_forward_mean_ablation(self):
        m = NodeMask(3)
        with torch.no_grad():
            m.tau.copy_(torch.tensor([1.0, -1.0, -1.0]))
        m.set_mean(torch.tensor([5.0, 6.0, 7.0]))
        assert m.mean_set
        x = torch.tensor([[10.0, 20.0, 30.0]])
        # active keeps x, masked gets the mean
        assert torch.equal(m(x), torch.tensor([[10.0, 6.0, 7.0]]))

    def test_forward_broadcasts_over_leading_dims(self):
        m = NodeMask(4)
        with torch.no_grad():
            m.tau.fill_(1.0)
        x = torch.randn(2, 3, 4)
        assert torch.equal(m(x), x)

    def test_gradient_reaches_tau_through_forward(self):
        m = NodeMask(4)
        x = torch.randn(5, 4)
        m(x).sum().backward()
        assert m.tau.grad is not None and m.tau.grad.abs().sum() > 0


def make_collection(**kwargs):
    defaults = dict(n_layers=2, d_model=16, d_mlp=24, n_heads=4, d_head=4,
                    mask_locations=["attn_in", "attn_q", "attn_k", "attn_v",
                                    "attn_out", "mlp_in", "mlp_neuron",
                                    "mlp_out"])
    defaults.update(kwargs)
    return NodeMaskCollection(**defaults)


class TestNodeMaskCollection:
    def test_dims_per_location(self):
        col = make_collection()
        assert col.get_mask(0, "attn_in").num_nodes == 16
        assert col.get_mask(1, "attn_q").num_nodes == 16  # n_heads * d_head
        assert col.get_mask(0, "mlp_neuron").num_nodes == 24
        # 2 layers x (4*16 + 3*16 + 24) = 2 * 136
        assert col.get_total_nodes() == 2 * (4 * 16 + 3 * 16 + 24)

    def test_unknown_location_raises(self):
        with pytest.raises(ValueError, match="Unknown location"):
            make_collection(mask_locations=["attn_in", "nonsense"])

    def test_total_active_and_sparsity_loss_agree(self):
        col = make_collection()
        with torch.no_grad():
            for mask in col.masks.values():
                mask.tau.fill_(-1.0)
            col.get_mask(0, "attn_in").tau[:5] = 1.0
            col.get_mask(1, "mlp_out").tau[:3] = 1.0
        assert col.get_total_active_nodes() == 8
        assert col.get_sparsity_loss().item() == 8.0

    def test_sparsity_loss_is_differentiable(self):
        col = make_collection()
        loss = col.get_sparsity_loss()
        loss.backward()
        grads = [m.tau.grad for m in col.masks.values()]
        assert all(g is not None for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0

    def test_clamp_all(self):
        col = make_collection()
        with torch.no_grad():
            col.get_mask(0, "attn_in").tau.fill_(9.0)
        col.clamp_all_parameters()
        assert col.get_mask(0, "attn_in").tau.max() == 1.0

    def test_apply_threshold(self):
        col = make_collection(mask_locations=["attn_in"], n_layers=1)
        mask = col.get_mask(0, "attn_in")
        with torch.no_grad():
            mask.tau.copy_(torch.linspace(-1, 1, 16))
        col.apply_threshold(0.5)
        expected_active = (torch.linspace(-1, 1, 16) >= 0.5)
        assert torch.equal(mask.tau == 1.0, expected_active)
        assert torch.equal(mask.tau == -1.0, ~expected_active)

    def test_keep_top_k_selects_globally_highest_taus(self):
        col = make_collection(mask_locations=["attn_in", "mlp_neuron"],
                              n_layers=1)
        a = col.get_mask(0, "attn_in")
        b = col.get_mask(0, "mlp_neuron")
        with torch.no_grad():
            a.tau.copy_(torch.linspace(-1.0, 0.5, 16))
            b.tau.copy_(torch.linspace(-0.9, 0.9, 24))
        originals = {("attn_in", i): a.tau[i].item() for i in range(16)}
        originals.update({("mlp_neuron", i): b.tau[i].item() for i in range(24)})
        col.keep_top_k(7)
        kept = [k for k in originals
                if (a if k[0] == "attn_in" else b).tau[k[1]].item() == 1.0]
        assert len(kept) == 7
        assert col.get_total_active_nodes() == 7
        kept_vals = sorted((originals[k] for k in kept), reverse=True)
        dropped_vals = [originals[k] for k in originals if k not in kept]
        assert min(kept_vals) >= max(dropped_vals)
        # non-kept are exactly -1
        assert set(a.tau.tolist()) | set(b.tau.tolist()) <= {-1.0, 1.0}

    def test_keep_top_k_includes_currently_inactive_nodes(self):
        """keep_top_k ranks ALL taus, so a negative-tau node can re-enter."""
        col = make_collection(mask_locations=["attn_in"], n_layers=1)
        mask = col.get_mask(0, "attn_in")
        with torch.no_grad():
            mask.tau.fill_(-1.0)
            mask.tau[3] = -0.2  # inactive but highest tau
        col.keep_top_k(1)
        assert mask.tau[3] == 1.0
        assert mask.get_num_active() == 1

    def test_keep_top_k_edge_cases(self):
        col = make_collection(mask_locations=["attn_in"], n_layers=1)
        mask = col.get_mask(0, "attn_in")
        before = mask.tau.clone()
        col.keep_top_k(16)  # k == total: unchanged
        assert torch.equal(mask.tau, before)
        col.keep_top_k(100)  # k > total: unchanged
        assert torch.equal(mask.tau, before)
        col.keep_top_k(0)  # deactivate everything
        assert mask.tau.eq(-1.0).all()

    def test_set_means_from_cache_and_warning(self, capsys):
        col = make_collection(mask_locations=["attn_in"], n_layers=1)
        col.set_means_from_cache({"layer0_attn_in": torch.full((16,), 2.0)})
        mask = col.get_mask(0, "attn_in")
        assert mask.mean_set
        assert mask.mean_activation.eq(2.0).all()
        col2 = make_collection(mask_locations=["attn_in", "mlp_out"], n_layers=1)
        col2.set_means_from_cache({"layer0_attn_in": torch.zeros(16)})
        out = capsys.readouterr().out
        assert "No mean cache for layer0_mlp_out" in out
        assert not col2.get_mask(0, "mlp_out").mean_set

    def test_get_mask_summary_layout(self):
        col = make_collection()
        summary = col.get_mask_summary()
        assert set(summary) == {"layer0", "layer1"}
        assert set(summary["layer0"]) == {"attn_in", "attn_q", "attn_k",
                                          "attn_v", "attn_out", "mlp_in",
                                          "mlp_neuron", "mlp_out"}
        assert summary["layer0"]["attn_in"] == \
            col.get_mask(0, "attn_in").get_num_active()

    def test_get_all_tau_values_concatenates_in_mask_order(self):
        col = make_collection(mask_locations=["attn_in", "mlp_neuron"],
                              n_layers=1)
        taus = col.get_all_tau_values()
        assert taus.shape == (40,)
        expected = torch.cat([col.get_mask(0, "attn_in").tau,
                              col.get_mask(0, "mlp_neuron").tau])
        assert torch.equal(taus, expected)
