"""Tests for PruningTrainer: LR schedule math, RMS gradient clipping, the
train step contract (clamping, state, history), the diagnostic statistics
(verified against known tau values), checkpointing, and the run_pruning
convenience wrapper."""
import numpy as np
import pytest
import torch

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.trainer import (
    PruningTrainer, TrainingState, run_pruning,
)
from tests.conftest import ToyTask, make_masked


def make_trainer(num_steps=4, **pc_over):
    pc_kwargs = dict(device="cpu", batch_size=2, seq_length=0,
                     num_steps=num_steps, log_every=10 ** 9, lr=1e-2,
                     k_coef=1e-3)
    pc_kwargs.update(pc_over)
    mm = make_masked(**pc_kwargs)
    return PruningTrainer(mm, ToyTask(), mm.config, use_wandb=False)


class TestLRSchedule:
    def test_no_warmup_linear_decay(self):
        tr = make_trainer(num_steps=10, lr_warmup_frac=0.0)
        assert tr._get_lr_multiplier(0) == pytest.approx(1.0)
        assert tr._get_lr_multiplier(5) == pytest.approx(0.5)
        assert tr._get_lr_multiplier(10) == pytest.approx(0.0)

    def test_warmup_then_decay(self):
        tr = make_trainer(num_steps=100, lr_warmup_frac=0.1)
        # warmup: (step+1)/warmup_steps for step < 10
        assert tr._get_lr_multiplier(0) == pytest.approx(0.1)
        assert tr._get_lr_multiplier(9) == pytest.approx(1.0)
        # decay from 1 to 0 over remaining 90 steps
        assert tr._get_lr_multiplier(10) == pytest.approx(1.0)
        assert tr._get_lr_multiplier(55) == pytest.approx(0.5)

    def test_update_lr_applies_multiplier(self):
        tr = make_trainer(num_steps=10)
        tr._update_lr(5)
        assert tr.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2 * 0.5)


class TestGradClipping:
    def test_rms_above_threshold_is_scaled_down(self):
        tr = make_trainer(grad_clip_norm=1.0)
        params = list(tr.masked_model.masks.parameters())
        for p in params:
            p.grad = torch.full_like(p, 3.0)  # RMS = 3
        tr._clip_gradients()
        rms = np.sqrt(np.mean(np.concatenate(
            [p.grad.numpy().ravel() ** 2 for p in params])))
        assert rms == pytest.approx(1.0, rel=1e-4)

    def test_rms_below_threshold_untouched(self):
        tr = make_trainer(grad_clip_norm=1.0)
        params = list(tr.masked_model.masks.parameters())
        for p in params:
            p.grad = torch.full_like(p, 0.5)
        tr._clip_gradients()
        assert all(p.grad.eq(0.5).all() for p in params)

    def test_no_grads_is_noop(self):
        tr = make_trainer()
        tr._clip_gradients()  # must not raise


class TestTrainStep:
    def test_step_updates_state_and_clamps(self):
        tr = make_trainer()
        metrics = tr.train_step()
        assert tr.state.step == 1
        assert metrics["step"] == 1
        taus = tr.masked_model.masks.get_all_tau_values()
        assert taus.min() >= -1.0 and taus.max() <= 1.0
        assert {"task_loss", "accuracy", "logit_diff", "total_loss",
                "sparsity_loss", "num_active_nodes", "total_nodes",
                "lr"} <= set(metrics)

    def test_parameters_actually_move(self):
        tr = make_trainer()
        before = tr.masked_model.masks.get_all_tau_values().detach().clone()
        tr.train_step()
        after = tr.masked_model.masks.get_all_tau_values().detach()
        assert not torch.equal(before, after)

    def test_best_loss_tracking(self):
        tr = make_trainer()
        metrics = tr.train_step()
        assert tr.state.best_step == 1
        assert tr.state.best_loss == metrics["task_loss"]

    def test_train_loop_populates_history(self):
        tr = make_trainer(num_steps=3)
        final = tr.train(show_progress=False)
        assert tr.state.step == 3
        assert len(tr.history) == 3
        assert final["step"] == 3
        assert all(not k.startswith("_") for h in tr.history for k in h)


class TestDiagnostics:
    def _trainer_with_known_taus(self):
        tr = make_trainer()
        with torch.no_grad():
            for mask in tr.masked_model.masks.masks.values():
                mask.tau.fill_(-1.0)
            first = next(iter(tr.masked_model.masks.masks.values()))
            first.tau[0:2] = 1.0
            first.tau[2] = 0.95
            first.tau[3] = -0.3
        return tr

    def test_tau_statistics(self):
        tr = self._trainer_with_known_taus()
        stats = tr._compute_tau_statistics()
        taus = tr.masked_model.masks.get_all_tau_values().detach().numpy()
        assert stats["tau/mean"] == pytest.approx(float(taus.mean()))
        assert stats["tau/max"] == pytest.approx(1.0)
        assert stats["tau/frac_positive"] == pytest.approx(
            float((taus >= 0).mean()))
        assert stats["tau/frac_near_plus1"] == pytest.approx(
            float((taus > 0.9).mean()))

    def test_per_location_and_per_layer_stats(self):
        tr = self._trainer_with_known_taus()
        loc = tr._compute_per_location_stats()
        first_key = next(iter(tr.masked_model.masks.masks))
        loc_type = first_key.split("_", 1)[1]
        assert loc[f"location/{loc_type}/active_nodes"] == 3
        layer = tr._compute_per_layer_stats()
        assert layer["layer/0/active_nodes"] == 3
        assert layer["layer/1/active_nodes"] == 0
        total_l0 = sum(m.num_nodes for k, m in
                       tr.masked_model.masks.masks.items()
                       if k.startswith("layer0"))
        assert layer["layer/0/frac_active"] == pytest.approx(3 / total_l0)

    def test_threshold_analysis(self):
        tr = self._trainer_with_known_taus()
        stats = tr._compute_threshold_analysis()
        # taus >= 0.0 are the three active nodes
        assert stats["threshold/p0/circuit_size"] == 3
        assert stats["threshold/p90/circuit_size"] == 3  # 0.95 included
        assert stats["threshold/m50/circuit_size"] == 4  # includes -0.3

    def test_bimodality_metrics(self):
        tr = self._trainer_with_known_taus()
        stats = tr._compute_bimodality_metrics()
        assert stats["bimodal/gap"] == pytest.approx(0.95 - (-0.3))
        assert stats["bimodal/middle_zone_count"] == 1  # only -0.3
        assert 0 < stats["bimodal/entropy"] < 1

    def test_tau_velocity_and_sign_flips(self):
        tr = self._trainer_with_known_taus()
        assert tr._compute_tau_velocity() == {}  # first call: no previous
        with torch.no_grad():
            first = next(iter(tr.masked_model.masks.masks.values()))
            first.tau[0] = -1.0  # one sign flip
        stats = tr._compute_tau_velocity()
        assert stats["velocity/sign_flips"] == 1
        assert stats["velocity/max_abs_delta"] == pytest.approx(2.0)

    def test_topk_stability(self):
        tr = self._trainer_with_known_taus()
        stats = tr._compute_topk_stability(k_values=[2])
        assert "stability/top2_cutoff_tau" in stats
        assert stats["stability/top2_cutoff_tau"] == pytest.approx(1.0)
        stats2 = tr._compute_topk_stability(k_values=[2])
        assert stats2["stability/top2_overlap"] == 2
        assert stats2["stability/top2_overlap_frac"] == pytest.approx(1.0)
        # k larger than the node count is skipped
        assert tr._compute_topk_stability(k_values=[10 ** 6]) == {}

    def test_gradient_stats(self):
        tr = make_trainer()
        assert tr._compute_gradient_stats() == {}  # no grads yet
        for p in tr.masked_model.masks.parameters():
            p.grad = torch.ones_like(p)
        stats = tr._compute_gradient_stats()
        assert stats["grad/mean"] == pytest.approx(1.0)
        assert stats["grad/rms"] == pytest.approx(1.0)
        assert stats["grad/frac_near_zero"] == 0.0

    def test_val_metrics(self):
        mm = make_masked(batch_size=2)
        tr = PruningTrainer(mm, ToyTask(), mm.config,
                            val_task=ToyTask(seed=99), use_wandb=False)
        vm = tr._compute_val_metrics(num_batches=2)
        assert set(vm) == {"val/task_loss", "val/accuracy", "val/logit_diff"}
        tr_noval = make_trainer()
        assert tr_noval._compute_val_metrics() == {}


class TestEvaluate:
    def test_averages_full_loss_metrics(self):
        tr = make_trainer()
        metrics = tr.evaluate(num_batches=2)
        assert metrics["num_active_nodes"] == \
            tr.masked_model.masks.get_total_active_nodes()
        assert metrics["task_loss"] > 0


class TestCheckpointing:
    def test_round_trip(self, tmp_path):
        tr = make_trainer(num_steps=2)
        tr.train(show_progress=False)
        path = tmp_path / "sub" / "ckpt.pt"
        tr.save_checkpoint(str(path))
        assert path.exists()

        tr2 = make_trainer(num_steps=2)
        tr2.load_checkpoint(str(path))
        assert tr2.state.step == tr.state.step
        assert tr2.state.best_loss == tr.state.best_loss
        assert tr2.state.best_step == tr.state.best_step
        assert len(tr2.history) == len(tr.history)
        s1, s2 = tr.masked_model.get_mask_state(), tr2.masked_model.get_mask_state()
        for key in s1:
            assert torch.equal(s1[key], s2[key])


def test_training_state_defaults():
    st = TrainingState()
    assert (st.step, st.best_loss, st.best_step) == (0, float("inf"), 0)


class TestRunPruningWrapper:
    def test_full_pipeline_without_mean_cache(self, capsys):
        model = __import__("tests.conftest", fromlist=["make_tiny_model"]).make_tiny_model()
        cfg = PruningConfig(device="cpu", batch_size=2, seq_length=0,
                            num_steps=2, log_every=10 ** 9)
        mm, metrics, trainer = run_pruning(model, ToyTask(), cfg,
                                           show_progress=False)
        assert trainer.state.step == 2
        assert metrics["num_active_nodes"] >= 0
        assert "Final results" in capsys.readouterr().out

    def test_mean_cache_computed_when_iterator_given(self):
        model = __import__("tests.conftest", fromlist=["make_tiny_model"]).make_tiny_model()
        cfg = PruningConfig(device="cpu", batch_size=2, seq_length=0,
                            num_steps=1, log_every=10 ** 9,
                            mean_cache_num_batches=1)
        data = iter([torch.randint(0, 512, (2, 6))])
        mm, _, _ = run_pruning(model, ToyTask(), cfg, data_iterator=data,
                               show_progress=False)
        assert all(m.mean_set for m in mm.masks.masks.values())
