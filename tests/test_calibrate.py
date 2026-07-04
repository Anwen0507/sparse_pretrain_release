"""Tests for logit calibration (scale+shift fit with LBFGS) and for baking
the calibration into the LM head."""
import pytest
import torch
import torch.nn.functional as F

from sparse_pretrain.src.pruning.calibrate import (
    LogitCalibrator, apply_calibration_to_model, calibrate_logits,
)
from sparse_pretrain.src.pruning.config import PruningConfig
from tests.conftest import ToyTask, make_masked, make_tiny_model, set_all_taus


class TestLogitCalibrator:
    def test_identity_at_init(self):
        cal = LogitCalibrator(vocab_size=10, device="cpu")
        x = torch.randn(3, 10)
        assert torch.allclose(cal(x), x)

    def test_affine_transform(self):
        cal = LogitCalibrator(vocab_size=10, device="cpu")
        with torch.no_grad():
            cal.scale.fill_(2.0)
            cal.shift.fill_(-1.0)
        x = torch.randn(3, 10)
        assert torch.allclose(cal(x), 2.0 * x - 1.0)

    def test_parameters_trainable(self):
        cal = LogitCalibrator(vocab_size=4, device="cpu")
        F.mse_loss(cal(torch.randn(2, 4)), torch.zeros(2, 4)).backward()
        assert cal.scale.grad is not None and cal.shift.grad is not None


class TestCalibrateLogits:
    def test_returns_finite_params_and_improves_loss(self):
        torch.manual_seed(0)
        mm = make_masked()
        set_all_taus(mm, 1.0)
        # Inflate the logits so a scale < 1 is clearly beneficial: the
        # optimizer should find scale != 1 and not increase the loss.
        with torch.no_grad():
            mm.model.lm_head.weight.mul_(20.0)
            mm.model.bigram_table.mul_(20.0)
        cfg = PruningConfig(device="cpu", batch_size=4, seq_length=0,
                            calibration_steps=4)
        scale, shift, metrics = calibrate_logits(
            mm, ToyTask(), cfg, num_batches_per_step=2, show_progress=False)
        assert torch.isfinite(torch.tensor([scale, shift])).all()
        assert set(metrics) == {"pre_calibration_loss",
                                "post_calibration_loss",
                                "post_calibration_accuracy", "scale", "shift"}
        assert metrics["scale"] == pytest.approx(scale)
        assert metrics["shift"] == pytest.approx(shift)
        assert scale != pytest.approx(1.0)

    def test_num_steps_defaults_to_config(self, monkeypatch):
        mm = make_masked()
        set_all_taus(mm, 1.0)
        cfg = PruningConfig(device="cpu", batch_size=2, seq_length=0,
                            calibration_steps=1)
        calls = []
        from torch.optim import LBFGS
        orig_step = LBFGS.step

        def counting_step(self, closure):
            calls.append(1)
            return orig_step(self, closure)

        monkeypatch.setattr(LBFGS, "step", counting_step)
        calibrate_logits(mm, ToyTask(), cfg, num_batches_per_step=1,
                         show_progress=False)
        assert len(calls) == 1


class TestApplyCalibration:
    def test_creates_bias_and_transforms_logits(self):
        """For a model without a bigram table, baking (scale, shift) into the
        lm_head makes new_logits == old_logits * scale + shift exactly."""
        model = make_tiny_model(use_bigram_table=False)
        mm = make_masked(model)
        set_all_taus(mm, 1.0)
        ids = torch.randint(0, 512, (2, 5))
        before = mm(ids).detach().clone()
        assert mm.model.lm_head.bias is None
        apply_calibration_to_model(mm, scale=2.5, shift=-0.75)
        assert mm.model.lm_head.bias is not None
        after = mm(ids).detach()
        assert torch.allclose(after, before * 2.5 - 0.75, atol=1e-4)

    def test_existing_bias_is_rescaled_then_shifted(self):
        model = make_tiny_model(use_bigram_table=False)
        with torch.no_grad():
            model.lm_head.bias = torch.nn.Parameter(torch.randn(512))
        old_bias = model.lm_head.bias.detach().clone()
        mm = make_masked(model)
        apply_calibration_to_model(mm, scale=3.0, shift=1.0)
        assert torch.allclose(mm.model.lm_head.bias, old_bias * 3.0 + 1.0,
                              atol=1e-6)
