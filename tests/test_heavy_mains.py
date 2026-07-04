"""Coverage of the remaining heavyweight paths, all on CPU:

* the wandb-instrumented trainer diagnostics (against a fake wandb module),
* activation-sparsity branches of the bridge/intermediate forward variants,
* MaskedSparseGPT over manual-attention and positional-embedding models,
* run_pruning CLI with live mean cache + discretization + calibration,
* universality experiment: names_templates end-to-end, split snapshot guard,
  CARBS hparam bootstrap,
* universality_motif_excision main (workers rerouted in-process),
* epistasis_test main, exclude_kernel main, universality_atom_peel main
  (re-prune stubbed to circuit copies; oracle/scoring run for real),
* run_carbs_clean: run_single_pruning, superval Pareto, tiny real CARBS sweep.
"""
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import sparse_pretrain.paths as paths
from sparse_pretrain.src.config import SparsityConfig
from sparse_pretrain.src.pruning.config import PruningConfig
from tests.conftest import (
    FakeTokenizer, ToyTask, make_masked, make_tiny_model, model_mask_dims,
    set_all_taus, write_model_circuits,
)


# ---------------------------------------------------------------------------
# Trainer: wandb-instrumented paths
# ---------------------------------------------------------------------------
class FakeWandb(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.logged, self.init_kwargs, self.finished = [], None, False

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def log(self, data, step=None, commit=None):
        self.logged.append(dict(data))

    def Histogram(self, data, num_bins=50):
        return ("hist", len(data))

    def Image(self, fig):
        return "img"

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    import sparse_pretrain.src.pruning.trainer as T
    fw = FakeWandb()
    monkeypatch.setattr(T, "wandb", fw)
    monkeypatch.setattr(T, "WANDB_AVAILABLE", True)
    return fw


class TestTrainerWandbInstrumentation:
    def test_full_instrumented_training(self, fake_wandb):
        from sparse_pretrain.src.pruning.trainer import PruningTrainer
        mm = make_masked(batch_size=2, num_steps=2, lr=1e-2)
        tr = PruningTrainer(mm, ToyTask(), mm.config,
                            val_task=ToyTask(seed=7), use_wandb=True,
                            wandb_run_name="t", wandb_config={"extra": 1})
        assert fake_wandb.init_kwargs["config"]["extra"] == 1
        final = tr.train(num_steps=2, show_progress=False, histogram_every=1,
                         detailed_log_every=1, pareto_probe_every=2)
        assert final["step"] == 2
        keys = set().union(*(d.keys() for d in fake_wandb.logged))
        # step-0 evaluation, detailed diagnostics, pareto probe, final block
        assert "train/task_loss" in keys
        assert "val/task_loss" in keys
        assert any(k.startswith("tau/") for k in keys)
        assert any(k.startswith("bimodal/") for k in keys)
        assert any(k.startswith("location/") for k in keys)
        assert "pareto/evolution" in keys
        assert "final/task_loss" in keys
        assert len(tr._pareto_history) >= 1
        assert len(tr._pareto_history_val) >= 1

    def test_pareto_probe_restores_mask_state(self, fake_wandb):
        from sparse_pretrain.src.pruning.trainer import PruningTrainer
        mm = make_masked(batch_size=2)
        tr = PruningTrainer(mm, ToyTask(), mm.config, use_wandb=True)
        before = mm.get_mask_state()
        stats = tr._run_pareto_probes(k_values=[1, 5])
        after = mm.get_mask_state()
        for key in before:
            assert torch.equal(before[key], after[key])
        assert stats["pareto/num_curves"] == 1
        assert stats["pareto/train_min_loss"] > 0

    def test_pareto_probe_zero_active_short_circuits(self, fake_wandb):
        from sparse_pretrain.src.pruning.trainer import PruningTrainer
        mm = make_masked(batch_size=2)
        set_all_taus(mm, -1.0)
        tr = PruningTrainer(mm, ToyTask(), mm.config, use_wandb=True)
        assert tr._run_pareto_probes() == {"pareto/num_active": 0}

    def test_evaluate_step0_metrics(self):
        from sparse_pretrain.src.pruning.trainer import PruningTrainer
        mm = make_masked(batch_size=2)
        tr = PruningTrainer(mm, ToyTask(), mm.config, use_wandb=False)
        m = tr._evaluate_step0(num_batches=2)
        assert set(m) >= {"task_loss", "accuracy", "logit_diff",
                          "num_active_nodes", "total_nodes", "lr"}
        assert 0 <= m["accuracy"] <= 100
        assert m["total_loss"] == pytest.approx(
            m["task_loss"] + mm.config.k_coef * m["num_active_nodes"])

    def test_run_pruning_wrapper_logs_and_finishes(self, fake_wandb, capsys):
        from sparse_pretrain.src.pruning.trainer import run_pruning
        model = make_tiny_model()
        cfg = PruningConfig(device="cpu", batch_size=2, seq_length=0,
                            num_steps=1, log_every=10 ** 9)
        run_pruning(model, ToyTask(), cfg, show_progress=False,
                    use_wandb=True)
        assert fake_wandb.finished
        assert any("eval/task_loss" in d for d in fake_wandb.logged)


# ---------------------------------------------------------------------------
# Model: activation-sparsity branches of the alternate forwards
# ---------------------------------------------------------------------------
class TestModelSparsityBranches:
    def sparse_model(self, locations=None):
        sc = SparsityConfig(
            enable_weight_sparsity=False, enable_activation_sparsity=True,
            activation_topk_fraction=0.5,
            activation_sparsity_locations=locations or (
                "attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,"
                "attn_q,attn_k,attn_v"))
        return make_tiny_model(sparsity=sc)

    def test_bridge_and_from_site_consistent_with_default_locations(self):
        """With the shipped sparsity locations (no resid sites) all three
        forward variants agree exactly."""
        model = self.sparse_model()
        ids = torch.randint(0, 512, (2, 6))
        logits, pre, post = model.forward_with_bridge_sites(ids)
        direct, _, _ = model(ids)
        assert torch.allclose(logits, direct, atol=1e-5)
        # resid sites untouched -> pre and post are identical tensors
        assert all(p is q for p, q in zip(pre, post))
        for site_idx, h in enumerate(post):
            resumed = model.forward_from_site(h, site_idx, input_ids=ids)
            assert torch.allclose(resumed, logits, atol=1e-5), site_idx

    def test_resid_pre_final_site_divergence_is_pinned(self):
        """Known quirk (unused by shipped configs): with resid_pre in the
        sparsity locations, forward_with_bridge_sites applies AbsTopK to the
        final residual before ln_f while forward() does not, so their logits
        legitimately differ."""
        model = self.sparse_model(
            locations="resid_pre,attn_in,mlp_neuron")
        ids = torch.randint(0, 512, (2, 6))
        logits_bridge, pre, post = model.forward_with_bridge_sites(ids)
        direct, _, _ = model(ids)
        assert any(not torch.equal(p, q) for p, q in zip(pre, post))
        assert not torch.allclose(logits_bridge, direct, atol=1e-4)

    def test_block_intermediate_variants_with_sparsity(self):
        model = self.sparse_model()
        fn = model._get_activation_sparsity_fn()
        block = model.blocks[0]
        x = torch.randn(2, 5, 16)
        out = block(x, fn)
        out2, post_attn = block.forward_with_intermediate(x, fn)
        assert torch.allclose(out, out2, atol=1e-6)
        resumed = block.forward_from_post_attn(post_attn, fn)
        assert torch.allclose(out, resumed, atol=1e-6)


class TestMaskedModelVariants:
    def test_manual_attention_path_equivalence(self):
        model = make_tiny_model(use_flash_attention=False,
                                use_attention_sinks=False)
        mm = make_masked(model)
        set_all_taus(mm, 1.0)
        ids = torch.randint(0, 512, (2, 6))
        base, _, _ = model(ids)
        assert torch.allclose(mm(ids), base, atol=1e-5)

    def test_flash_without_sinks_path(self):
        model = make_tiny_model(use_attention_sinks=False)
        mm = make_masked(model)
        set_all_taus(mm, 1.0)
        ids = torch.randint(0, 512, (2, 6))
        base, _, _ = model(ids)
        assert torch.allclose(mm(ids), base, atol=1e-5)

    def test_positional_embeddings_flow_through_everything(self):
        model = make_tiny_model(use_positional_embeddings=True)
        for module in model.modules():  # align eps with the frozen-LN path
            if isinstance(module, torch.nn.RMSNorm):
                module.eps = 1e-6
        mm = make_masked(model, freeze_layernorm_scale=True,
                         mask_token_embeds=True)
        set_all_taus(mm, 1.0)
        ids = torch.randint(0, 512, (2, 6))
        base, _, _ = model(ids)
        assert torch.allclose(mm(ids), base, atol=1e-4)
        cache = mm.compute_mean_cache(iter([ids]), 1, show_progress=False)
        assert "token_embed" in cache and "layer0_attn_in" in cache


# ---------------------------------------------------------------------------
# run_pruning CLI: full pipeline incl. mean cache, discretize, calibrate
# ---------------------------------------------------------------------------
class TestRunPruningMainFull:
    def test_main_with_mean_cache_discretize_calibrate(self, tmp_path,
                                                       monkeypatch, capsys):
        import sparse_pretrain.src.pruning.run_pruning as RP
        from tests.test_run_pruning_cli import (
            FakeStreamingDataset, save_checkpoint_dir,
        )
        ckpt = tmp_path / "model"
        save_checkpoint_dir(ckpt, make_tiny_model())
        out_dir = tmp_path / "out"
        tok = FakeTokenizer()
        texts = [f"tok{i} " * 50 for i in range(20)]
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        monkeypatch.setattr("datasets.load_dataset",
                            lambda *a, **k: FakeStreamingDataset(texts))
        monkeypatch.setattr(sys, "argv", [
            "run_pruning", "--model_path", str(ckpt),
            "--task", "dummy_pronoun", "--output_dir", str(out_dir),
            "--num_steps", "2", "--batch_size", "2", "--seq_length", "16",
            "--device", "cpu", "--mean_cache_batches", "1",
            "--target_loss", "1e6"])
        orig_eval = RP.PruningTrainer.evaluate
        monkeypatch.setattr(RP.PruningTrainer, "evaluate",
                            lambda self, num_batches=10: orig_eval(self, 1))
        RP.main()
        assert (out_dir / "mean_cache.pt").exists()
        assert (out_dir / "binary_masks.pt").exists()
        cal = json.loads((out_dir / "calibration.json").read_text())
        assert {"scale", "shift"} <= set(cal)
        masks = torch.load(out_dir / "binary_masks.pt", weights_only=True)
        # huge target -> discretization hits the 1-node shortcut
        assert sum(int(v.sum()) for v in masks.values()) == 1
        assert "PRUNING COMPLETE" in capsys.readouterr().out


class TestVerboseDiscretizeCalibrate:
    def test_discretize_show_progress_prints(self, monkeypatch, capsys):
        import sparse_pretrain.src.pruning.discretize as D
        mm = make_masked()
        monkeypatch.setattr(
            D, "evaluate_at_k",
            lambda m, t, k, c, num_batches=20: 0.15 if k >= 5 else 0.5)
        D.discretize_masks(mm, ToyTask(),
                           PruningConfig(device="cpu", batch_size=2,
                                         seq_length=0),
                           target_loss=0.15, show_progress=True)
        out = capsys.readouterr().out
        assert "Bisecting" in out and "Discretization complete" in out

    def test_discretize_to_edge_count_verbose(self, capsys):
        import sparse_pretrain.src.pruning.discretize as D
        mm = make_masked()
        D.discretize_to_edge_count(mm, ToyTask(),
                                   PruningConfig(device="cpu", batch_size=2,
                                                 seq_length=0),
                                   target_edges=5, num_eval_batches=1,
                                   show_progress=True)
        assert "Discretized to 5 nodes" in capsys.readouterr().out

    def test_calibrate_show_progress(self, capsys):
        from sparse_pretrain.src.pruning.calibrate import calibrate_logits
        mm = make_masked()
        set_all_taus(mm, 1.0)
        cfg = PruningConfig(device="cpu", batch_size=2, seq_length=0)
        calibrate_logits(mm, ToyTask(), cfg, num_steps=4,
                         num_batches_per_step=1, show_progress=True)
        out = capsys.readouterr().out
        assert "Calibrating logits" in out
        assert "Calibration complete" in out
        assert "Step 4" in out  # the every-4-steps progress line


# ---------------------------------------------------------------------------
# Universality experiment: names_templates + CARBS bootstrap
# ---------------------------------------------------------------------------
class TestUniversalityNamesTemplates:
    def _argv(self, ckpt, exp, pool):
        return ["universality_pruning_experiment.py",
                "--model", str(ckpt), "--tokenizer", "fake",
                "--task", "dummy_pronoun", "--target-loss", "1e6",
                "--num-seeds", "2", "--num-steps", "2", "--batch-size", "2",
                "--eval-batches", "1", "--bisect-iters", "2",
                "--num-workers", "1", "--max-iters", "1", "--skip-carbs",
                "--exp-dir", str(exp), "--device", "cpu",
                "--split-over", "names_templates", "--name-pool", str(pool)]

    def test_main_names_templates_end_to_end(self, tmp_path, monkeypatch):
        import sparse_pretrain.scripts.universality_pruning_experiment as U
        from tests.test_run_pruning_cli import save_checkpoint_dir
        from tests.test_universality_experiment import FakePopen
        ckpt = tmp_path / "model"
        save_checkpoint_dir(ckpt, make_tiny_model())
        exp = tmp_path / "exp"
        pool = paths.NAME_POOLS / "name_pool_cast15.json"
        tok = FakeTokenizer()
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        monkeypatch.setattr(U.subprocess, "Popen", FakePopen)
        monkeypatch.setattr(sys, "argv", self._argv(ckpt, exp, pool))
        U.main()
        si = json.loads((exp / "split_info.json").read_text())
        assert si["split_over"] == "names_templates"
        assert not set(si["train_names"]) & set(si["test_names"])
        state = json.loads((exp / "state.json").read_text())
        assert state["next_iter"] == 1
        rec = json.loads((exp / "iter00" / "seed0_result.json").read_text())
        assert rec["target_achieved"] is True
        assert rec["test_loss"] is not None  # held-out reporting active
        summary = json.loads(
            (exp / "iter00" / "iteration_summary.json").read_text())
        assert "heldout_test_loss" in summary
        assert "heldout_test_2afc" in summary

    def test_main_refuses_regenerated_pool(self, tmp_path, monkeypatch):
        import sparse_pretrain.scripts.universality_pruning_experiment as U
        from tests.test_run_pruning_cli import save_checkpoint_dir
        ckpt = tmp_path / "model"
        save_checkpoint_dir(ckpt, make_tiny_model())
        exp = tmp_path / "exp"
        exp.mkdir()
        tok = FakeTokenizer()
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        # a snapshot from a previous launch with different names
        (exp / "split_info.json").write_text(json.dumps(
            {"train_names": ["ghost"], "test_names": ["zork"]}))
        pool = paths.NAME_POOLS / "name_pool_cast15.json"
        argv = self._argv(ckpt, exp, pool)
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit, match="pool regenerated"):
            U.main()

    def test_get_hparams_paths(self, tmp_path, monkeypatch, capsys):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        import sparse_pretrain.scripts.universality_pruning_experiment as U
        args = types.SimpleNamespace(
            skip_carbs=False, model="user/tiny", task="dummy_pronoun",
            carbs_runs=1, num_steps=2, target_loss=0.15, device="cpu")

        # CARBS run that produces a best checkpoint
        def fake_sweep(cfg):
            ckpt = Path(cfg.output_base_dir) / "tiny_zero_noembed" / \
                "best_checkpoint"
            ckpt.mkdir(parents=True)
            (ckpt / "hparams.json").write_text(json.dumps(
                {"k_coef": 1.0, "lr": 2.0, "weight_decay": 3.0,
                 "beta2": 0.9, "heaviside_temp": 1.0,
                 "suggestion_uuid": "drop-me"}))

        monkeypatch.setattr(RC, "run_carbs_sweep", fake_sweep)
        exp1 = tmp_path / "e1"
        exp1.mkdir()
        hp = U.get_hparams(args, exp1)
        assert hp["k_coef"] == 1.0
        assert "suggestion_uuid" not in hp
        # cached on second call (no sweep)
        monkeypatch.setattr(RC, "run_carbs_sweep",
                            lambda cfg: pytest.fail("must use cache"))
        assert U.get_hparams(args, exp1) == hp

        # CARBS run that fails to produce a checkpoint -> center fallback
        exp2 = tmp_path / "e2"
        exp2.mkdir()
        monkeypatch.setattr(RC, "run_carbs_sweep", lambda cfg: None)
        hp2 = U.get_hparams(args, exp2)
        assert hp2 == U.CENTER_HPARAMS


# ---------------------------------------------------------------------------
# Motif excision main (workers in-process)
# ---------------------------------------------------------------------------
class TestMotifExcisionMain:
    def _run_main(self, tmp_path, monkeypatch, null_sil, max_iters=1):
        import sparse_pretrain.scripts.universality_motif_excision as ME
        from tests.test_run_pruning_cli import save_checkpoint_dir
        from tests.test_universality_experiment import FakePopen
        ckpt = tmp_path / "model"
        save_checkpoint_dir(ckpt, make_tiny_model())
        exp = tmp_path / "exp"
        tok = FakeTokenizer()
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        monkeypatch.setattr(ME.subprocess, "Popen", FakePopen)
        # pin the significance verdict so both decision branches are testable
        monkeypatch.setattr(ME, "null_best_sils",
                            lambda rng, q, N, kmax, reps: np.array([null_sil]))
        monkeypatch.setattr(sys, "argv", [
            "universality_motif_excision.py",
            "--model", str(ckpt), "--tokenizer", "fake",
            "--task", "dummy_pronoun", "--target-loss", "1e6",
            "--num-seeds", "5", "--num-steps", "2", "--batch-size", "2",
            "--eval-batches", "1", "--bisect-iters", "2",
            "--num-workers", "1", "--max-iters", str(max_iters),
            "--skip-carbs", "--exp-dir", str(exp), "--device", "cpu",
            "--split-over", "none", "--min-n-cluster", "3", "--kmax", "3",
            "--null-reps", "2"])
        ME.main()
        return exp

    def test_stop_when_clustering_not_significant(self, tmp_path,
                                                  monkeypatch):
        exp = self._run_main(tmp_path, monkeypatch, null_sil=1.0)
        state = json.loads((exp / "state.json").read_text())
        summary = json.loads(
            (exp / "iter00" / "motif_summary.json").read_text())
        assert summary["n_success"] == 5
        assert summary["decision"]["stop"] is True
        assert summary["decision"]["reason"] == "not_clustered"
        assert state["exhausted"] is True and state["next_iter"] == 0

    def test_excision_path_advances_and_tracks_survival(self, tmp_path,
                                                        monkeypatch):
        exp = self._run_main(tmp_path, monkeypatch, null_sil=-1.0,
                             max_iters=2)
        state = json.loads((exp / "state.json").read_text())
        s0 = json.loads((exp / "iter00" / "motif_summary.json").read_text())
        assert s0["decision"]["stop"] is False
        assert s0["decision"]["motif_id"] == "M0"
        assert s0["decision"]["n_excise"] >= 1
        assert state["next_iter"] >= 1
        assert state["motifs"][0]["id"] == "M0"
        assert state["excluded"]
        # second iteration performs the survival check on M0
        s1_path = exp / "iter01" / "motif_summary.json"
        if s1_path.exists():
            s1 = json.loads(s1_path.read_text())
            if s1["n_success"] and "survival" in s1 and s1["survival"]:
                assert s1["survival"]["excised_motif"] == "M0"
                assert "reappeared" in s1["survival"]


# ---------------------------------------------------------------------------
# Shared names_templates source-run fixture for the excision-family mains
# ---------------------------------------------------------------------------
def make_names_templates_src(tmp_path, model, tok, n_circuits=4, name="src"):
    from sparse_pretrain.scripts.pronoun_split import make_pronoun_fold_split
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / "run_args.json").write_text(json.dumps({
        "model": "tiny", "task": "dummy_pronoun", "tokenizer": "fake",
        "target_loss": 0.15, "num_steps": 2, "batch_size": 2,
        "eval_batches": 1, "bisect_iters": 2, "split_over": "names_templates",
        "split_seed": 0, "test_frac": 0.2, "heldout_fold": 0,
        "name_pool": str(paths.NAME_POOLS / "name_pool_cast15.json"),
        "seed_offset": 0}))
    (src / "hparams.json").write_text(json.dumps(
        {"k_coef": 1e-3, "weight_decay": 1e-3, "lr": 1e-2, "beta2": 0.95,
         "heaviside_temp": 1.0}))
    _, _, _, info = make_pronoun_fold_split(tok, heldout_fold=0)
    (src / "split_info.json").write_text(json.dumps(info))
    write_model_circuits(src / "iter00", model, n_circuits=n_circuits,
                         block_pattern=True)
    (src / "motif_dictionary.json").write_text(json.dumps(
        {"iterations": [{"iteration": 0, "k_parsimonious": 2}]}))
    return src


@pytest.fixture
def excision_env(tmp_path, monkeypatch):
    model = make_tiny_model()
    tok = FakeTokenizer()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        lambda *a, **k: tok)
    monkeypatch.setattr("sparse_pretrain.src.pruning.run_pruning.load_model",
                        lambda path, device="cuda": (model, {}))
    src = make_names_templates_src(tmp_path, model, tok)
    return tmp_path, model, tok, src


class TestEpistasisMain:
    def test_requires_names_templates_split(self, excision_env, monkeypatch):
        import sparse_pretrain.scripts.epistasis_test as ET
        tmp_path, model, tok, src = excision_env
        bad = json.loads((src / "run_args.json").read_text())
        bad["split_over"] = "none"
        other = tmp_path / "src_none"
        other.mkdir()
        (other / "run_args.json").write_text(json.dumps(bad))
        monkeypatch.setattr(sys, "argv", [
            "epistasis_test.py", "--src-dir", str(other),
            "--exp-dir", str(tmp_path / "e2"), "--device", "cpu"])
        with pytest.raises(SystemExit, match="names_templates"):
            ET.main()

    def test_requires_two_testable_atoms(self, excision_env, monkeypatch):
        import sparse_pretrain.scripts.epistasis_test as ET
        tmp_path, model, tok, src = excision_env
        monkeypatch.setattr(sys, "argv", [
            "epistasis_test.py", "--src-dir", str(src),
            "--exp-dir", str(tmp_path / "e3"), "--iter", "0",
            "--min-size", "999", "--smoke", "--device", "cpu"])
        with pytest.raises(SystemExit, match="testable atoms"):
            ET.main()

    def test_end_to_end_cpu(self, excision_env, monkeypatch, capsys):
        import sparse_pretrain.scripts.epistasis_test as ET
        tmp_path, model, tok, src = excision_env
        exp = tmp_path / "epi"
        monkeypatch.setattr(sys, "argv", [
            "epistasis_test.py", "--src-dir", str(src), "--exp-dir", str(exp),
            "--iter", "0", "--min-share", "0.25", "--min-size", "1",
            "--smoke", "--device", "cpu", "--rng-seed", "0"])
        ET.main()
        report = json.loads((exp / "epistasis_test.json").read_text())
        assert report["k"] == 2
        assert len(report["testable_atoms"]) >= 2
        assert report["pairs"]
        assert "verdict" in report
        assert (exp / "epistasis_test.png").exists()
        assert "VERDICT" in capsys.readouterr().out


class TestAtomPeelMain:
    def test_peel_loop_with_stubbed_reprune(self, excision_env, monkeypatch):
        import sparse_pretrain.scripts.universality_atom_peel as AP
        tmp_path, model, tok, src = excision_env
        exp = tmp_path / "peel"

        def stub_reprune(cond, args, exp_dir):
            it_dir = Path(exp_dir) / f"iter{cond['iterN']:02d}"
            if not list(it_dir.glob("seed*_result.json")):
                it_dir.mkdir(parents=True, exist_ok=True)
                for p in (src / "iter00").glob("seed*"):
                    shutil.copy(p, it_dir / p.name)
            return AP.collect(it_dir, args), 0.0

        monkeypatch.setattr(AP, "reprune_condition", stub_reprune)
        monkeypatch.setattr(sys, "argv", [
            "universality_atom_peel.py", "--src-dir", str(src),
            "--exp-dir", str(exp), "--iter", "0", "--max-depth", "1",
            "--k", "2", "--num-seeds", "4", "--rand-reps", "1", "--knn", "1",
            "--device", "cpu"])
        AP.main()
        report = json.loads((exp / "atom_peel.json").read_text())
        assert report["atom_trajectory"]
        depth0 = report["atom_trajectory"][0]
        assert depth0["depth"] == 0
        assert depth0["k"] == 2
        assert depth0["peeled"] is not None  # a unit was scored and peeled
        assert report["random_trajectory"]  # matched-random arm ran
        assert "atom_collapse_depth" in report["summary"]
        assert (exp / "atom_peel.png").exists()

    def test_make_plot_from_synthetic_report(self, tmp_path):
        import sparse_pretrain.scripts.universality_atom_peel as AP
        report = {
            "exp_dir": "x", "feas_eps": 0.1,
            "atom_trajectory": [
                {"depth": 0, "feasibility": 1.0, "mean_circuit_size": 30,
                 "mean_test_2afc": 0.9, "n_excluded": 0,
                 "peeled": {"damage_loss": 0.2, "support": 0.5,
                            "nodes": ["a#1"]}},
                {"depth": 1, "feasibility": 0.05, "mean_circuit_size": 50,
                 "mean_test_2afc": 0.6, "n_excluded": 3, "peeled": None}],
            "random_trajectory": [[
                {"depth": 1, "feasibility": 0.8, "mean_circuit_size": 35,
                 "mean_test_2afc": 0.85, "n_excluded": 3}]],
            "summary": {"atom_collapse_depth": 1,
                        "random_collapse_depth": None},
        }
        out = tmp_path / "peel.png"
        AP.make_plot(report, out)
        assert out.exists()


class TestExcludeKernelMain:
    @pytest.mark.parametrize("mode", ["split", "leaveoneout"])
    def test_modes_with_stubbed_reprune(self, excision_env, monkeypatch,
                                        mode):
        import sparse_pretrain.scripts.exclude_kernel as EK
        tmp_path, model, tok, src = excision_env
        peel = tmp_path / "peel_src"
        write_model_circuits(peel / "iter00", model, n_circuits=3,
                             block_pattern=True)
        (peel / "iter00" / "excluded_input.json").write_text(json.dumps(
            [["layer0_attn_in", 8], ["layer0_attn_in", 9]]))
        exp = tmp_path / f"kernel_{mode}"

        def stub_reprune(cond, args, exp_dir):
            it_dir = Path(exp_dir) / f"iter{cond['iterN']:02d}"
            if not list(it_dir.glob("seed*_result.json")):
                it_dir.mkdir(parents=True, exist_ok=True)
                write_model_circuits(it_dir, model, n_circuits=2,
                                     block_pattern=True)
            from sparse_pretrain.scripts.atom_excision import collect
            return collect(it_dir, args), 0.0

        monkeypatch.setattr(EK, "reprune_condition", stub_reprune)
        monkeypatch.setattr(sys, "argv", [
            "exclude_kernel.py", "--src-dir", str(src),
            "--peel-dir", str(peel), "--kernel-iter", "0",
            "--exp-dir", str(exp), "--num-seeds", "2", "--mode", mode,
            "--device", "cpu"])
        EK.main()
        report = json.loads((exp / "exclude_kernel.json").read_text())
        assert report["kernel_nodes"]
        kinds = [c["kind"] for c in report["conditions"]]
        assert kinds[0] == "control_depth34"
        if mode == "split":
            assert any(k.startswith("kernel_removed") for k in kinds)
        else:
            assert any(k.startswith("drop_") for k in kinds)
        for c in report["conditions"]:
            assert "frac_kernel" in c and "frac_novel" in c


class TestExcludeKernelMatchedRandomHelper:
    def test_deterministic_given_rng(self):
        import sparse_pretrain.scripts.exclude_kernel as EK
        freq = np.linspace(0.1, 0.5, 10)
        a = EK.matched_random(list(range(10)), freq, [0, 1], 2, set(),
                              np.random.default_rng(3), knn=3)
        b = EK.matched_random(list(range(10)), freq, [0, 1], 2, set(),
                              np.random.default_rng(3), knn=3)
        assert a == b


# ---------------------------------------------------------------------------
# run_carbs_clean heavy paths
# ---------------------------------------------------------------------------
def carbs_cpu_config(tmp_path, **over):
    import sparse_pretrain.scripts.run_carbs_clean as RC
    kwargs = dict(model_path="user/tiny", task_name="dummy_pronoun",
                  num_runs=2, parallel_suggestions=1, num_steps=2,
                  batch_size=2, task_max_length=0, target_loss=1e6,
                  use_autocast=False, bisection_eval_batches=1,
                  bisection_max_iters=2, output_base_dir=str(tmp_path / "carbs"),
                  device="cpu", use_wandb=False, ablation_type="zero",
                  mask_token_embeds=False, k_coef_center=1e-3)
    kwargs.update(over)
    return RC.CleanSweepConfig(**kwargs)


class TestRunCarbsHeavy:
    def test_run_single_pruning_cpu(self, tmp_path, fake_tokenizer):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        from sparse_pretrain.src.pruning.tasks import get_task
        cfg = carbs_cpu_config(tmp_path)
        model = make_tiny_model()
        train = get_task("dummy_pronoun", fake_tokenizer, split="train")
        val = get_task("dummy_pronoun", fake_tokenizer, split="val")
        hp = {"k_coef": 1e-3, "weight_decay": 1e-3, "lr": 1e-2,
              "beta2": 0.95, "heaviside_temp": 1.0}
        res = RC.run_single_pruning(hp, model, fake_tokenizer, train, val,
                                    None, cfg, run_id=0)
        assert res["success"] is True
        assert res["target_achieved"] is True  # huge target
        assert res["circuit_size"] >= 1
        assert "mask_state" in res

    def test_create_pareto_plot_superval(self, tmp_path, fake_tokenizer,
                                         monkeypatch):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        cfg = carbs_cpu_config(tmp_path)
        # the hardcoded 0.001..0.5 targets are unreachable for an untrained
        # model (and an empty Pareto set would crash the log-scaled plot);
        # use reachable targets so the sweep/plot logic is exercised
        monkeypatch.setattr(RC.np, "geomspace",
                            lambda *a, **k: np.array([10.0, 20.0]))
        model = make_tiny_model()
        mm = make_masked(model)
        hp = {"k_coef": 1e-3, "weight_decay": 1e-3, "lr": 1e-2,
              "beta2": 0.95, "heaviside_temp": 1.0}
        best = {"hparams": hp, "mask_state": mm.get_mask_state()}
        out = tmp_path / "pareto"
        out.mkdir()
        data = RC.create_pareto_plot_superval(best, model, fake_tokenizer,
                                              None, cfg, out)
        assert (out / "pareto_superval_data.json").exists()
        assert (out / "pareto_superval.png").exists()
        assert isinstance(data, list)

    def test_run_carbs_sweep_tiny(self, tmp_path, monkeypatch):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        monkeypatch.chdir(tmp_path)  # carbs snapshots ./checkpoints in CWD
        model = make_tiny_model()
        tok = FakeTokenizer()
        monkeypatch.setattr(RC.np, "geomspace",
                            lambda *a, **k: np.array([10.0, 20.0]))
        monkeypatch.setattr(RC, "load_model",
                            lambda path, device="cuda": (model, {}))
        monkeypatch.setattr(RC.AutoTokenizer, "from_pretrained",
                            staticmethod(lambda *a, **k: tok))
        cfg = carbs_cpu_config(tmp_path, num_runs=2)
        results = RC.run_carbs_sweep(cfg)
        out_dir = tmp_path / "carbs" / "tiny_zero_noembed"
        assert (out_dir / "sweep_config.json").exists()
        assert (out_dir / "final_results.json").exists()
        assert (out_dir / "best_checkpoint" / "hparams.json").exists()
        assert (out_dir / "pareto_superval.png").exists()

    def test_main_cli(self, tmp_path, monkeypatch):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        monkeypatch.chdir(tmp_path)  # carbs snapshots ./checkpoints in CWD
        model = make_tiny_model()
        tok = FakeTokenizer()
        monkeypatch.setattr(RC.np, "geomspace",
                            lambda *a, **k: np.array([10.0, 20.0]))
        monkeypatch.setattr(RC, "load_model",
                            lambda path, device="cuda": (model, {}))
        monkeypatch.setattr(RC.AutoTokenizer, "from_pretrained",
                            staticmethod(lambda *a, **k: tok))
        # force CPU + tiny knobs through the CLI surface
        orig_init = RC.CleanSweepConfig.__init__

        def tiny_init(self, **kw):
            kw.update(dict(num_steps=2, batch_size=2, task_max_length=0,
                           use_autocast=False, bisection_eval_batches=1,
                           bisection_max_iters=2, target_loss=1e6))
            orig_init(self, **kw)

        monkeypatch.setattr(RC.CleanSweepConfig, "__init__", tiny_init)
        monkeypatch.setattr(sys, "argv", [
            "run_carbs_clean", "--model", "user/tiny",
            "--task", "dummy_pronoun", "--num-runs", "1", "--steps", "2",
            "--device", "cpu", "--no-wandb", "--ablation", "zero",
            "--output-dir", str(tmp_path / "cli")])
        RC.main()
        assert (tmp_path / "cli" / "tiny_zero_noembed"
                / "final_results.json").exists()
