"""Tests for the run_pruning CLI module: local checkpoint loading (standalone
and bridge formats), the packed-sequence data iterator, and a full main()
run against a tiny local checkpoint with mocked tokenizer/dataset."""
import json
import sys

import pytest
import torch

import sparse_pretrain.src.pruning.run_pruning as RP
from tests.conftest import TINY_VOCAB, FakeTokenizer, make_tiny_model


def save_checkpoint_dir(path, model, weights_name="pytorch_model.bin",
                        with_tokenizer=True):
    path.mkdir(parents=True, exist_ok=True)
    cfg = {"model_config": {"n_layer": 2, "d_model": 16, "n_ctx": 32,
                            "d_head": 4, "d_mlp": 24, "vocab_size": TINY_VOCAB},
           "sparsity_config": {"enable_weight_sparsity": False,
                               "enable_activation_sparsity": False}}
    if with_tokenizer:
        cfg["training_config"] = {"tokenizer_name": "fake-tokenizer"}
    (path / "config.json").write_text(json.dumps(cfg))
    torch.save(model.state_dict(), path / weights_name)


class TestLoadModel:
    def test_standalone_checkpoint(self, tmp_path):
        src = make_tiny_model()
        save_checkpoint_dir(tmp_path / "ckpt", src)
        model, cfg = RP.load_model(str(tmp_path / "ckpt"), device="cpu")
        assert not model.training
        ids = torch.randint(0, TINY_VOCAB, (1, 5))
        a, _, _ = src(ids)
        b, _, _ = model(ids)
        assert torch.allclose(a, b, atol=1e-6)
        assert cfg["training_config"]["tokenizer_name"] == "fake-tokenizer"

    def test_bridge_checkpoint_uses_sparse_model_bin(self, tmp_path, capsys):
        src = make_tiny_model()
        save_checkpoint_dir(tmp_path / "bridge", src,
                            weights_name="sparse_model.bin")
        model, _ = RP.load_model(str(tmp_path / "bridge"), device="cpu")
        assert "bridge checkpoint" in capsys.readouterr().out
        ids = torch.randint(0, TINY_VOCAB, (1, 4))
        assert torch.allclose(src(ids)[0], model(ids)[0], atol=1e-6)

    def test_checkpoint_subdir_falls_back_to_parent_config(self, tmp_path):
        src = make_tiny_model()
        run_dir = tmp_path / "run"
        save_checkpoint_dir(run_dir, src)  # config at run level
        sub = run_dir / "checkpoint-100"
        sub.mkdir()
        torch.save(src.state_dict(), sub / "pytorch_model.bin")
        (run_dir / "config.json").exists()
        model, _ = RP.load_model(str(sub), device="cpu")
        assert isinstance(model, type(src))

    def test_missing_config_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="config.json"):
            RP.load_model(str(empty), device="cpu")

    def test_missing_weights_raises(self, tmp_path):
        d = tmp_path / "noweights"
        save_checkpoint_dir(d, make_tiny_model())
        (d / "pytorch_model.bin").unlink()
        with pytest.raises(FileNotFoundError, match="model weights"):
            RP.load_model(str(d), device="cpu")

    def test_hub_path_config_error_is_wrapped(self, monkeypatch):
        def boom(repo_id, filename):
            raise RuntimeError("offline")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
        with pytest.raises(ValueError, match="Failed to download config.json"):
            RP.load_model("someone/nonexistent-model", device="cpu")

    def test_hub_bridge_fallback(self, tmp_path, monkeypatch):
        """Hub loading: pytorch_model.bin missing -> falls back to
        sparse_model.bin."""
        src = make_tiny_model()
        save_checkpoint_dir(tmp_path, src, weights_name="sparse_model.bin")

        def fake_download(repo_id, filename):
            p = tmp_path / filename
            if not p.exists():
                raise FileNotFoundError(filename)
            return str(p)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        model, _ = RP.load_model("user/bridge-model", device="cpu")
        ids = torch.randint(0, TINY_VOCAB, (1, 4))
        assert torch.allclose(src(ids)[0], model(ids)[0], atol=1e-6)

    def test_hub_no_weights_at_all_raises(self, tmp_path, monkeypatch):
        save_checkpoint_dir(tmp_path, make_tiny_model())
        (tmp_path / "pytorch_model.bin").unlink()

        def fake_download(repo_id, filename):
            p = tmp_path / filename
            if not p.exists():
                raise FileNotFoundError(filename)
            return str(p)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(ValueError, match="Neither pytorch_model.bin"):
            RP.load_model("user/broken", device="cpu")


class FakeStreamingDataset:
    def __init__(self, texts):
        self.texts = texts

    def shuffle(self, seed=None, buffer_size=None):
        return self

    def __iter__(self):
        for t in self.texts:
            yield {"story": t}


class TestCreateDataIterator:
    def test_packed_batches_shape_and_eos(self, monkeypatch):
        tok = FakeTokenizer()
        texts = [f"word{i} " * 40 for i in range(30)]
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        monkeypatch.setattr("datasets.load_dataset",
                            lambda *a, **k: FakeStreamingDataset(texts))
        it = RP.create_data_iterator("fake", batch_size=2, seq_length=16,
                                     num_batches=3)
        batches = list(it)
        assert len(batches) == 3
        assert all(b.shape == (2, 16) for b in batches)
        flat = torch.cat([b.reshape(-1) for b in batches])
        # sequence packing inserts EOS between documents
        assert (flat == tok.eos_token_id).any()

    def test_skips_empty_texts(self, monkeypatch):
        tok = FakeTokenizer()
        texts = ["", "   ", "alpha " * 100]
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        monkeypatch.setattr("datasets.load_dataset",
                            lambda *a, **k: FakeStreamingDataset(texts))
        batches = list(RP.create_data_iterator("fake", batch_size=1,
                                               seq_length=8, num_batches=2))
        assert len(batches) == 2


class TestMainCLI:
    def test_end_to_end_tiny_run(self, tmp_path, monkeypatch, capsys):
        ckpt = tmp_path / "model"
        save_checkpoint_dir(ckpt, make_tiny_model())
        out_dir = tmp_path / "out"
        tok = FakeTokenizer()
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                            lambda *a, **k: tok)
        argv = ["run_pruning",
                "--model_path", str(ckpt),
                "--task", "dummy_pronoun",
                "--output_dir", str(out_dir),
                "--num_steps", "2",
                "--batch_size", "2",
                "--seq_length", "0",
                "--device", "cpu",
                "--skip_mean_cache",
                "--skip_discretization",
                "--skip_calibration"]
        monkeypatch.setattr(sys, "argv", argv)
        # shrink the final evaluation (main() asks for 50 batches) to one
        orig_eval = RP.PruningTrainer.evaluate
        monkeypatch.setattr(RP.PruningTrainer, "evaluate",
                            lambda self, num_batches=10: orig_eval(self, 1))

        RP.main()

        out = capsys.readouterr().out
        assert "PRUNING COMPLETE" in out
        assert (out_dir / "config.yaml").exists()
        assert (out_dir / "checkpoint.pt").exists()
        results = json.loads((out_dir / "results.json").read_text())
        assert results["task"] == "dummy_pronoun"
        assert results["total_nodes"] == 2 * (4 * 16 + 3 * 16 + 24)
        assert "mask_summary" in results

    def test_main_requires_tokenizer_somewhere(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "model"
        save_checkpoint_dir(ckpt, make_tiny_model(), with_tokenizer=False)
        monkeypatch.setattr(sys, "argv", [
            "run_pruning", "--model_path", str(ckpt),
            "--output_dir", str(tmp_path / "o"), "--device", "cpu"])
        with pytest.raises(ValueError, match="No tokenizer found"):
            RP.main()
