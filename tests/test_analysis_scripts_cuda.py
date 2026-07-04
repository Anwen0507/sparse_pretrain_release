"""End-to-end exec tests for the nine mech-interp analysis scripts that run
top-to-bottom at import time with device="cuda" hardcoded:

    mech_interp_core, recompute_2afc, verify_embedding_and_copying,
    mlp_counterbalance, mlp_function, crosstask_attn_vs_mlp_dla,
    crosstask_ablation_check, ablate_core_crosstask,
    compare_full_vs_circuit_pronoun

Each is executed against a tiny CUDA model injected via a patched
run_pruning.load_model and the whitespace FakeTokenizer, over synthetic
experiment inputs. The assertions check that each script completes and writes
its declared artifacts with sane structure — i.e. the published analysis
pipeline is runnable end to end.

These tests require a CUDA device (the scripts allocate tensors on "cuda"
unconditionally) and are skipped elsewhere; the rest of the suite is CPU-only.
"""
import importlib
import json
import sys

import pytest
import torch

import sparse_pretrain.paths as paths
from sparse_pretrain.src.pruning.config import PruningConfig
from tests.conftest import FakeTokenizer, make_tiny_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="these analysis scripts hardcode device='cuda'")

MASK_LOCS = PruningConfig().mask_locations


def model_dims(model):
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


def write_circuits(it_dir, model, n_circuits=3, frac=0.5, seed=0):
    """Model-dim-consistent seed*_circuit.pt files + passing result jsons."""
    it_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    dims = model_dims(model)
    for s in range(n_circuits):
        mask = {k: (torch.rand(d, generator=g) < frac).float()
                for k, d in dims.items()}
        torch.save(mask, it_dir / f"seed{s}_circuit.pt")
        (it_dir / f"seed{s}_result.json").write_text(json.dumps(
            {"seed": s, "target_achieved": True, "circuit_size": 10,
             "circuit_loss": 0.1}))


def write_core_nodes(exp, model, n_per_key=2):
    dims = model_dims(model)
    nodes = []
    for key in list(dims)[:6]:
        for i in range(n_per_key):
            nodes.append({"location": key, "index": i})
    (exp / "core_nodes.json").write_text(json.dumps({"nodes": nodes}))
    return nodes


@pytest.fixture
def analysis_env(tmp_path, monkeypatch):
    """Patch OUTPUTS + load_model + AutoTokenizer; return (exp_dir, model, tok)."""
    def _setup(**model_overrides):
        outputs = tmp_path / "outputs"
        exp = outputs / "ss_d128_f1_pronoun"
        exp.mkdir(parents=True, exist_ok=True)
        model = make_tiny_model(**model_overrides).to("cuda").eval()
        tok = FakeTokenizer()
        monkeypatch.setattr(paths, "OUTPUTS", outputs)
        monkeypatch.setattr(
            "sparse_pretrain.src.pruning.run_pruning.load_model",
            lambda path, device="cuda": (model, {}))
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        return exp, model, tok

    return _setup


def exec_script(name, monkeypatch=None, argv=None):
    if argv is not None:
        monkeypatch.setattr(sys, "argv", argv)
    mod_name = f"sparse_pretrain.scripts.{name}"
    sys.modules.pop(mod_name, None)
    grad_was_enabled = torch.is_grad_enabled()
    try:
        return importlib.import_module(mod_name)
    finally:
        sys.modules.pop(mod_name, None)
        # several of these scripts call torch.set_grad_enabled(False) at
        # module level; do not let that leak into the rest of the suite
        torch.set_grad_enabled(grad_was_enabled)


class TestMechInterpCore:
    def test_runs_and_writes_report(self, analysis_env, monkeypatch):
        # the script hardcodes the ss_d128_f1 geometry (qkv.view(B,T,8,16),
        # WO[:,48:64], 128-dim directions) -> inject a d128-shaped model
        exp, model, tok = analysis_env(d_model=128, d_head=16, d_mlp=512)
        write_core_nodes(exp, model)
        exec_script("mech_interp_core", monkeypatch,
                    argv=["mech_interp_core.py"])
        report = json.loads((exp / "mech_interp_core.json").read_text())
        assert report  # non-empty analysis dict
        assert (exp / "core_causal_decomposition.png").exists()


class TestRecompute2afc:
    def test_legacy_branch(self, analysis_env, monkeypatch):
        exp, model, tok = analysis_env()
        write_circuits(exp / "iter00", model, n_circuits=2)
        write_core_nodes(exp, model)
        exec_script("recompute_2afc", monkeypatch,
                    argv=["recompute_2afc.py", "--exp-dir", str(exp),
                          "--model", "tiny", "--iters", "0"])
        out = json.loads((exp / "accuracy_2afc.json").read_text())
        assert out

    def test_names_templates_branch(self, analysis_env, monkeypatch):
        from sparse_pretrain.scripts.pronoun_split import (
            make_pronoun_fold_split,
        )
        exp, model, tok = analysis_env()
        _, _, _, info = make_pronoun_fold_split(tok, heldout_fold=0)
        (exp / "split_info.json").write_text(json.dumps(info))
        write_circuits(exp / "iter00", model, n_circuits=2)
        exec_script("recompute_2afc", monkeypatch,
                    argv=["recompute_2afc.py", "--exp-dir", str(exp),
                          "--model", "tiny", "--iters", "0"])
        out = json.loads((exp / "accuracy_2afc_heldout.json").read_text())
        assert out


class TestVerifyEmbeddingAndCopying:
    def test_runs_and_writes_verification(self, analysis_env, monkeypatch):
        exp, model, tok = analysis_env()
        exec_script("verify_embedding_and_copying", monkeypatch,
                    argv=["verify_embedding_and_copying.py"])
        report = json.loads(
            (exp / "mech_interp_verification.json").read_text())
        assert report
        assert (exp / "verification_steering.png").exists()


class TestMlpCounterbalance:
    def test_runs_and_writes_report(self, analysis_env, monkeypatch):
        exp, model, tok = analysis_env()
        exec_script("mlp_counterbalance", monkeypatch,
                    argv=["mlp_counterbalance.py"])
        assert json.loads((exp / "mlp_counterbalance.json").read_text())
        assert (exp / "mlp_counterbalance_steering.png").exists()


class TestMlpFunction:
    def test_runs_with_wide_mlp(self, analysis_env, monkeypatch):
        # the script inspects hardcoded neurons 510/471 -> needs d_mlp > 510
        exp, model, tok = analysis_env(d_mlp=520)
        exec_script("mlp_function", monkeypatch, argv=["mlp_function.py"])
        assert json.loads((exp / "mlp_function.json").read_text())
        assert (exp / "mlp_gendersplit_steering.png").exists()


class TestCrosstaskDLA:
    def test_runs_and_reports_all_tasks(self, analysis_env, monkeypatch):
        exp, model, tok = analysis_env()
        exec_script("crosstask_attn_vs_mlp_dla", monkeypatch,
                    argv=["crosstask_attn_vs_mlp_dla.py"])
        report = json.loads(
            (exp / "crosstask_attn_vs_mlp_dla.json").read_text())
        assert "dummy_pronoun" in report


class TestCrosstaskAblationCheck:
    def test_runs_all_conditions(self, analysis_env, monkeypatch):
        exp, model, tok = analysis_env()
        exec_script("crosstask_ablation_check", monkeypatch,
                    argv=["crosstask_ablation_check.py"])
        report = json.loads(
            (exp / "crosstask_ablation_check.json").read_text())
        assert report


class TestAblateCoreCrosstask:
    def test_runs_and_writes_report(self, analysis_env, monkeypatch):
        exp, model, tok = analysis_env()
        write_core_nodes(exp, model)
        exec_script("ablate_core_crosstask", monkeypatch,
                    argv=["ablate_core_crosstask.py"])
        report = json.loads((exp / "crosstask_ablation.json").read_text())
        assert report
        assert (exp / "crosstask_ablation.png").exists()


class TestCompareFullVsCircuit:
    def test_runs_over_iter00_circuits(self, analysis_env, monkeypatch):
        exp, model, tok = analysis_env()
        write_circuits(exp / "iter00", model, n_circuits=2)
        exec_script("compare_full_vs_circuit_pronoun", monkeypatch,
                    argv=["compare_full_vs_circuit_pronoun.py"])
        report = json.loads(
            (exp / "full_vs_circuit_pronoun.json").read_text())
        assert report
        assert (exp / "full_vs_circuit_pronoun.png").exists()
