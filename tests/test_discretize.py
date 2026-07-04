"""Tests for discretization (bisection over circuit size k).

The bisection logic itself is verified against a deterministic synthetic loss
curve (monkeypatched evaluate_at_k), so the smallest-k-achieving-target
guarantee is checked exactly; the evaluate_at_k helpers are checked for
state restoration and batch-format handling on a real tiny model.
"""
import pytest
import torch

import sparse_pretrain.src.pruning.discretize as D
from sparse_pretrain.src.pruning.config import PruningConfig
from tests.conftest import ToyTask, make_masked, set_all_taus


def small_config(**kw):
    kwargs = dict(device="cpu", batch_size=2, seq_length=0,
                  discretization_max_iters=50, discretization_tolerance=0.0)
    kwargs.update(kw)
    return PruningConfig(**kwargs)


class TestEvaluateAtK:
    def test_restores_mask_state(self):
        mm = make_masked()
        before = mm.get_mask_state()
        D.evaluate_at_k(mm, ToyTask(), k=5, config=small_config(),
                        num_batches=1)
        after = mm.get_mask_state()
        for key in before:
            assert torch.equal(before[key], after[key])

    def test_loss_is_mean_over_batches(self, monkeypatch):
        mm = make_masked()
        losses = iter([1.0, 3.0])

        def fake_task_loss(*a, **k):
            return None, {"task_loss": next(losses)}

        monkeypatch.setattr(mm, "compute_task_loss", fake_task_loss)
        out = D.evaluate_at_k(mm, ToyTask(), k=5, config=small_config(),
                              num_batches=2)
        assert out == pytest.approx(2.0)

    def test_fixed_batches_tuple_and_dict_agree(self):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        task = ToyTask()
        batch = task.generate_batch(3)
        as_tuple = [batch]
        keys = ["positive_ids", "negative_ids", "correct_tokens",
                "incorrect_tokens", "eval_positions"]
        as_dict = [dict(zip(keys, batch))]
        l1 = D.evaluate_at_k_fixed_batches(mm, as_tuple, k=10, device="cpu")
        l2 = D.evaluate_at_k_fixed_batches(mm, as_dict, k=10, device="cpu")
        assert l1 == pytest.approx(l2)

    def test_fixed_batches_restores_state(self):
        mm = make_masked()
        before = mm.get_mask_state()
        D.evaluate_at_k_fixed_batches(mm, [ToyTask().generate_batch(2)], k=3,
                                      device="cpu")
        for key, tau in mm.get_mask_state().items():
            assert torch.equal(tau, before[key])


def synthetic_loss_curve(k_star):
    """loss(k) strictly decreasing, crossing the 0.15 target exactly at
    k >= k_star."""
    def fake_evaluate(masked_model, task, k, config, num_batches=20):
        return 0.15 if k >= k_star else 0.15 + (k_star - k) * 0.1
    return fake_evaluate


class TestDiscretizeMasks:
    def _model_with_ordered_taus(self):
        mm = make_masked()
        taus = mm.masks.get_all_tau_values()
        n = taus.numel()
        with torch.no_grad():
            i = 0
            for mask in mm.masks.masks.values():
                for j in range(mask.num_nodes):
                    # descending, top 100 active (tau>=0)
                    mask.tau[j] = 1.0 - 2.0 * (i / n)
                    i += 1
        return mm

    @pytest.mark.parametrize("k_star", [2, 17, 60])
    def test_bisection_finds_smallest_k(self, monkeypatch, k_star):
        mm = self._model_with_ordered_taus()
        num_active = mm.masks.get_total_active_nodes()
        assert num_active >= k_star
        monkeypatch.setattr(D, "evaluate_at_k", synthetic_loss_curve(k_star))
        k, loss, circuit = D.discretize_masks(
            mm, ToyTask(), small_config(), target_loss=0.15,
            show_progress=False)
        assert k == k_star
        assert loss == pytest.approx(0.15)
        active = sum(int(v.sum()) for v in circuit.values())
        assert active == k_star

    def test_unreachable_target_returns_all_active(self, monkeypatch, capsys):
        mm = self._model_with_ordered_taus()
        num_active = mm.masks.get_total_active_nodes()
        monkeypatch.setattr(
            D, "evaluate_at_k", lambda *a, **kw: 9.9)  # never achieves target
        k, loss, circuit = D.discretize_masks(
            mm, ToyTask(), small_config(), target_loss=0.15,
            show_progress=False)
        assert k == num_active
        assert loss == pytest.approx(9.9)
        assert "Cannot achieve target loss" in capsys.readouterr().out

    def test_single_node_shortcut(self, monkeypatch, capsys):
        mm = self._model_with_ordered_taus()
        monkeypatch.setattr(D, "evaluate_at_k", lambda *a, **kw: 0.01)
        k, loss, _ = D.discretize_masks(mm, ToyTask(), small_config(),
                                        target_loss=0.15, show_progress=False)
        assert k == 1 and loss == pytest.approx(0.01)
        assert "just 1 node" in capsys.readouterr().out

    def test_default_target_loss_from_config(self, monkeypatch):
        mm = self._model_with_ordered_taus()
        seen = []

        def spy(masked_model, task, k, config, num_batches=20):
            seen.append(k)
            return 0.0  # instantly achieves any target

        monkeypatch.setattr(D, "evaluate_at_k", spy)
        cfg = small_config(target_loss=0.42)
        k, loss, _ = D.discretize_masks(mm, ToyTask(), cfg,
                                        show_progress=False)
        assert k == 1  # trivially achievable

    def test_tolerance_loosens_acceptance(self, monkeypatch):
        """With tolerance t, k is accepted when loss <= target + t
        (boundary inclusive), so the found k shrinks accordingly."""
        mm = self._model_with_ordered_taus()
        k_star = 9
        monkeypatch.setattr(D, "evaluate_at_k", synthetic_loss_curve(k_star))
        cfg = small_config(discretization_tolerance=0.1)
        # loss(8) = 0.25 == 0.15+0.1 accepted; loss(7) = 0.35 rejected
        k, _, _ = D.discretize_masks(mm, ToyTask(), cfg, target_loss=0.15,
                                     show_progress=False)
        assert k == k_star - 1

    def test_real_model_end_to_end_smoke(self):
        """No mocks: on the real tiny model a huge target is achievable and
        the returned circuit has k active nodes with restored consistency."""
        mm = make_masked()
        cfg = small_config(discretization_tolerance=0.01)
        k, loss, circuit = D.discretize_masks(
            mm, ToyTask(), cfg, target_loss=1e6, num_eval_batches=1,
            show_progress=False)
        assert k == 1  # everything satisfies a huge target
        assert sum(int(v.sum()) for v in circuit.values()) == 1


class TestDiscretizeToEdgeCount:
    def test_keeps_requested_node_count(self):
        mm = make_masked()
        k, loss, circuit = D.discretize_to_edge_count(
            mm, ToyTask(), small_config(), target_edges=12,
            num_eval_batches=1, show_progress=False)
        assert k == 12
        assert sum(int(v.sum()) for v in circuit.values()) == 12
        assert loss > 0
