"""Tests for scripts/universality_pruning_experiment.py.

Covers the pure analysis functions (node space, weighted edges, Jaccard/
universality statistics, exclusion pinning, seed chunking, state files) with
known-answer checks, plus in-process integration tests of run_single_seed,
worker_main, run_iteration and main() on a tiny CPU model — subprocess worker
launches are rerouted to in-process calls so the real orchestration state
machine runs end to end.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import sparse_pretrain.scripts.universality_pruning_experiment as U
from tests.conftest import FakeTokenizer, make_tiny_model
from tests.test_run_pruning_cli import save_checkpoint_dir


# ---------------------------------------------------------------------------
# Node space & weighted edges
# ---------------------------------------------------------------------------
LOCS = ["attn_in", "attn_q", "attn_k", "attn_v", "attn_out", "mlp_in",
        "mlp_neuron", "mlp_out"]


class TestNodeSpace:
    def test_dim_for_loc(self, tiny_model):
        mc = tiny_model.config
        assert U._dim_for_loc("attn_in", mc) == 16
        assert U._dim_for_loc("attn_q", mc) == 16
        assert U._dim_for_loc("mlp_neuron", mc) == 24
        with pytest.raises(ValueError):
            U._dim_for_loc("resid_pre", mc)

    def test_offsets_are_contiguous_and_total_correct(self, tiny_model):
        ns = U.NodeSpace(tiny_model, LOCS)
        per_layer = 4 * 16 + 3 * 16 + 24
        assert ns.total == 2 * per_layer
        assert ns.offset["layer0_attn_in"] == 0
        assert ns.offset["layer0_attn_q"] == 16
        assert ns.offset["layer1_attn_in"] == per_layer
        assert ns.gid("layer0_attn_q", 3) == 19
        # keys enumerate every (layer, loc) once, in order
        assert len(ns.keys) == 16
        assert ns.keys[0] == "layer0_attn_in"


class TestWeightMaps:
    def test_families_and_slicing(self, tiny_model):
        maps = U.build_weight_maps(tiny_model, LOCS)
        assert len(maps) == 2 * 6
        by_pair = {(s, d): W for s, d, W in maps}
        blk = tiny_model.blocks[0]
        HD = 16
        Wqkv = blk.attn.c_attn.weight.detach()
        assert torch.equal(by_pair[("layer0_attn_in", "layer0_attn_q")],
                           Wqkv[0:HD].float())
        assert torch.equal(by_pair[("layer0_attn_in", "layer0_attn_k")],
                           Wqkv[HD:2 * HD].float())
        assert torch.equal(by_pair[("layer0_attn_in", "layer0_attn_v")],
                           Wqkv[2 * HD:3 * HD].float())
        assert torch.equal(by_pair[("layer0_attn_v", "layer0_attn_out")],
                           blk.attn.c_proj.weight.detach().float())
        assert torch.equal(by_pair[("layer0_mlp_in", "layer0_mlp_neuron")],
                           blk.mlp.c_fc.weight.detach().float())
        assert torch.equal(by_pair[("layer0_mlp_neuron", "layer0_mlp_out")],
                           blk.mlp.c_proj.weight.detach().float())

    def test_families_filtered_to_masked_locations(self, tiny_model):
        maps = U.build_weight_maps(tiny_model, ["mlp_in", "mlp_neuron"])
        assert {(s.split("_", 1)[1], d.split("_", 1)[1]) for s, d, _ in maps} \
            == {("mlp_in", "mlp_neuron")}


class TestCircuitNodesAndEdges:
    def test_circuit_nodes(self):
        mask = {"layer0_attn_in": torch.tensor([1.0, 0.0, 1.0]),
                "layer0_mlp_out": torch.tensor([0.0, 1.0])}
        assert U.circuit_nodes(mask) == {("layer0_attn_in", 0),
                                         ("layer0_attn_in", 2),
                                         ("layer0_mlp_out", 1)}

    def test_circuit_edges_values_and_ids(self, tiny_model):
        ns = U.NodeSpace(tiny_model, LOCS)
        maps = U.build_weight_maps(tiny_model, LOCS)
        mask = {key: torch.zeros(ns.dim[key]) for key in ns.keys}
        mask["layer0_mlp_in"][2] = 1.0
        mask["layer0_mlp_in"][5] = 1.0
        mask["layer0_mlp_neuron"][7] = 1.0
        edges = U.circuit_edges(mask, maps, ns)
        # only the mlp_in -> mlp_neuron family has both endpoints active
        assert len(edges) == 2
        Wfc = tiny_model.blocks[0].mlp.c_fc.weight.detach().float()
        for src_idx in (2, 5):
            eid = ns.gid("layer0_mlp_in", src_idx) * ns.total + \
                ns.gid("layer0_mlp_neuron", 7)
            assert edges[eid] == pytest.approx(abs(Wfc[7, src_idx].item()),
                                               rel=1e-6)

    def test_no_active_endpoints_no_edges(self, tiny_model):
        ns = U.NodeSpace(tiny_model, LOCS)
        maps = U.build_weight_maps(tiny_model, LOCS)
        mask = {key: torch.zeros(ns.dim[key]) for key in ns.keys}
        mask["layer0_attn_q"][0] = 1.0  # source side inactive
        assert U.circuit_edges(mask, maps, ns) == {}


# ---------------------------------------------------------------------------
# Similarity statistics
# ---------------------------------------------------------------------------
class TestStats:
    def test_stats_empty(self):
        assert U._stats([]) == {"mean": None, "std": None, "min": None,
                                "max": None, "n": 0}

    def test_stats_values(self):
        s = U._stats([1.0, 3.0])
        assert s == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0, "n": 2}

    def test_node_jaccard_stats(self):
        sets = [{1, 2, 3}, {2, 3, 4}, {9}]
        stats, vals = U.node_jaccard_stats(sets)
        assert sorted(vals) == [0.0, 0.0, pytest.approx(0.5)]
        assert stats["n"] == 3
        assert stats["max"] == pytest.approx(0.5)

    def test_node_jaccard_skips_empty_unions(self):
        stats, vals = U.node_jaccard_stats([set(), set()])
        assert vals == [] and stats["n"] == 0

    def test_edge_jaccard_unweighted_and_weighted(self):
        ea = {1: 2.0, 2: 1.0}
        eb = {2: 3.0, 3: 4.0}
        unw, wt, uv, wv = U.edge_jaccard_stats([ea, eb])
        assert uv == [pytest.approx(1 / 3)]
        # weighted: num = min over intersection = min(1,3)=1
        #           den = max over union = max(2,0)+max(1,3)+max(0,4) = 9
        assert wv == [pytest.approx(1 / 9)]
        assert unw["mean"] == pytest.approx(1 / 3)
        assert wt["mean"] == pytest.approx(1 / 9)

    def test_node_universality_and_rank1(self):
        sets = [{("a", 0), ("b", 1)}, {("a", 0)}, {("a", 0), ("c", 2)}]
        freq = U.node_universality(sets)
        assert freq[("a", 0)] == pytest.approx(1.0)
        assert freq[("b", 1)] == pytest.approx(1 / 3)
        r1, mx = U.rank1_nodes(freq)
        assert r1 == [("a", 0)] and mx == pytest.approx(1.0)

    def test_rank1_ties(self):
        r1, mx = U.rank1_nodes({("a", 0): 0.5, ("b", 1): 0.5, ("c", 2): 0.25})
        assert set(r1) == {("a", 0), ("b", 1)} and mx == 0.5

    def test_rank1_empty(self):
        assert U.rank1_nodes({}) == ([], 0.0)


class TestSeedChunks:
    def test_even_split(self):
        assert U._seed_chunks(10, 2) == [(0, 5), (5, 10)]

    def test_uneven_split(self):
        assert U._seed_chunks(10, 3) == [(0, 4), (4, 8), (8, 10)]

    def test_more_workers_than_seeds(self):
        assert U._seed_chunks(2, 8) == [(0, 1), (1, 2)]

    def test_zero_workers_clamped(self):
        assert U._seed_chunks(3, 0) == [(0, 3)]


class TestApplyNodeExclusion:
    def test_pins_and_repins_after_clamp(self):
        from tests.conftest import make_masked
        mm = make_masked()
        excluded = {("layer0_attn_in", 0), ("layer0_attn_in", 5),
                    ("layer1_mlp_out", 3)}
        U.apply_node_exclusion(mm, excluded)
        assert mm.masks.masks["layer0_attn_in"].tau[0] == -1.0
        assert mm.masks.masks["layer1_mlp_out"].tau[3] == -1.0
        # after an optimizer-style perturbation + clamp, still pinned
        with torch.no_grad():
            mm.masks.masks["layer0_attn_in"].tau.fill_(0.9)
        mm.clamp_mask_parameters()
        assert mm.masks.masks["layer0_attn_in"].tau[0].item() == -1.0
        assert mm.masks.masks["layer0_attn_in"].tau[5].item() == -1.0
        assert mm.masks.masks["layer0_attn_in"].tau[1].item() == pytest.approx(0.9)

    def test_empty_exclusion_is_noop(self):
        from tests.conftest import make_masked
        mm = make_masked()
        U.apply_node_exclusion(mm, set())
        # the clamp hook is only replaced when nodes are excluded
        assert "clamp_mask_parameters" not in mm.__dict__


class TestStateFiles:
    def test_default_state(self, tmp_path):
        state = U.load_state(tmp_path)
        assert state == {"next_iter": 0, "excluded": [], "history": [],
                         "exhausted": False}

    def test_round_trip(self, tmp_path):
        state = {"next_iter": 3, "excluded": [["a", 1]], "history": [{}],
                 "exhausted": True}
        U.save_state(tmp_path, state)
        assert U.load_state(tmp_path) == state


# ---------------------------------------------------------------------------
# Integration: run_single_seed / worker_main / run_iteration / main on CPU
# ---------------------------------------------------------------------------
def base_args(model_dir, exp_dir, **over):
    ns = argparse.Namespace(
        model=str(model_dir), task="dummy_pronoun", tokenizer="fake",
        target_loss=0.15, num_seeds=2, seed_offset=0, num_steps=2,
        batch_size=2, eval_batches=1, bisect_iters=2, carbs_runs=1,
        skip_carbs=True, num_workers=1, max_iters=1, exp_dir=str(exp_dir),
        device="cpu", split_over="none", split_seed=0, test_frac=0.2,
        heldout_fold=0, name_pool=str(U.NAME_POOLS / "name_pool_cast15.json"),
        worker=False, iter=0, seed_start=0, seed_end=0,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def tiny_ckpt(tmp_path):
    ckpt = tmp_path / "model"
    save_checkpoint_dir(ckpt, make_tiny_model())
    return ckpt


@pytest.fixture
def patched_tokenizer(monkeypatch):
    tok = FakeTokenizer()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        lambda *a, **k: tok)
    return tok


class TestRunSingleSeed:
    def test_result_contract_split_none(self, tiny_ckpt, patched_tokenizer,
                                        tmp_path):
        model, _ = U.load_model(str(tiny_ckpt), "cpu")
        args = base_args(tiny_ckpt, tmp_path / "exp", target_loss=1e6)
        res = U.run_single_seed(0, dict(U.CENTER_HPARAMS), model,
                                patched_tokenizer, set(), args, None, None)
        assert res["seed"] == 0
        assert res["target_achieved"] is True  # huge target always achieved
        assert res["circuit_mask"] is not None
        assert res["circuit_size"] >= 1
        assert res["test_loss"] is None  # no held-out split in "none" mode
        active = sum(int(v.sum()) for v in res["circuit_mask"].values())
        assert active == res["circuit_size"]

    def test_failure_path_records_loss(self, tiny_ckpt, patched_tokenizer,
                                       tmp_path):
        model, _ = U.load_model(str(tiny_ckpt), "cpu")
        args = base_args(tiny_ckpt, tmp_path / "exp", target_loss=-1.0)
        res = U.run_single_seed(1, dict(U.CENTER_HPARAMS), model,
                                patched_tokenizer, set(), args, None, None)
        assert res["target_achieved"] is False
        assert res["circuit_mask"] is None
        assert res["loss_at_all_active"] > 0

    def test_excluded_nodes_stay_out_of_circuit(self, tiny_ckpt,
                                                patched_tokenizer, tmp_path):
        model, _ = U.load_model(str(tiny_ckpt), "cpu")
        args = base_args(tiny_ckpt, tmp_path / "exp", target_loss=1e6)
        excluded = {("layer0_attn_in", i) for i in range(8)}
        res = U.run_single_seed(0, dict(U.CENTER_HPARAMS), model,
                                patched_tokenizer, excluded, args, None, None)
        assert res["target_achieved"]
        mask = res["circuit_mask"]["layer0_attn_in"]
        assert mask[:8].sum() == 0

    def test_examples_split_reports_heldout(self, tiny_ckpt, patched_tokenizer,
                                            tmp_path):
        model, _ = U.load_model(str(tiny_ckpt), "cpu")
        args = base_args(tiny_ckpt, tmp_path / "exp", target_loss=1e6,
                         split_over="examples")
        res = U.run_single_seed(0, dict(U.CENTER_HPARAMS), model,
                                patched_tokenizer, set(), args, None, None)
        assert res["test_loss"] is not None
        assert 0.0 <= res["test_2afc"] <= 1.0
        assert res["generalizes"] is True  # target is huge


def run_worker_inprocess(cmd, patched_tok):
    """Parse a worker command line exactly like main() would and run
    worker_main in-process."""
    argv_backup = sys.argv
    sys.argv = [cmd[1]] + cmd[2:]
    try:
        parser_args = _parse_worker_argv()
        U.worker_main(parser_args)
    finally:
        sys.argv = argv_backup


def _parse_worker_argv():
    """Re-parse sys.argv using the script's own argument definitions by
    calling main() would recurse; replicate the worker-relevant namespace."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    for flag, typ, default in [
            ("--iter", int, 0), ("--seed-start", int, 0), ("--seed-end", int, 0),
            ("--target-loss", float, 0.15), ("--num-steps", int, 2000),
            ("--batch-size", int, 64), ("--eval-batches", int, 5),
            ("--bisect-iters", int, 15), ("--split-seed", int, 0),
            ("--test-frac", float, 0.2), ("--heldout-fold", int, 0)]:
        ap.add_argument(flag, type=typ, default=default,
                        dest=flag.lstrip("-").replace("-", "_"))
    for flag in ["--model", "--tokenizer", "--task", "--exp-dir", "--device",
                 "--split-over", "--name-pool"]:
        ap.add_argument(flag, dest=flag.lstrip("-").replace("-", "_"))
    args, _ = ap.parse_known_args()
    return args


class FakePopen:
    """Stands in for subprocess.Popen inside run_iteration: executes the
    worker synchronously in-process (the fake tokenizer patch stays active)."""

    def __init__(self, cmd, stdout=None, stderr=None, env=None):
        self.returncode = 0
        try:
            run_worker_inprocess(cmd, None)
        except SystemExit as e:  # argparse failure inside worker
            self.returncode = int(e.code or 1)

    def wait(self):
        return self.returncode


class TestWorkerAndIteration:
    def _setup_exp(self, tmp_path, tiny_ckpt, **over):
        exp = tmp_path / "exp"
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "hparams.json").write_text(json.dumps(U.CENTER_HPARAMS))
        args = base_args(tiny_ckpt, exp, **over)
        return exp, args

    def test_worker_main_writes_results(self, tmp_path, tiny_ckpt,
                                        patched_tokenizer):
        exp, args = self._setup_exp(tmp_path, tiny_ckpt, target_loss=1e6)
        it_dir = exp / "iter00"
        it_dir.mkdir()
        (it_dir / "excluded_input.json").write_text("[]")
        args.worker, args.iter, args.seed_start, args.seed_end = True, 0, 0, 2
        U.worker_main(args)
        for seed in (0, 1):
            rec = json.loads((it_dir / f"seed{seed}_result.json").read_text())
            assert rec["seed"] == seed
            assert rec["target_achieved"] is True
            assert (it_dir / f"seed{seed}_circuit.pt").exists()

    def test_run_iteration_end_to_end(self, tmp_path, tiny_ckpt,
                                      patched_tokenizer, monkeypatch, capsys):
        exp, args = self._setup_exp(tmp_path, tiny_ckpt, target_loss=1e6)
        monkeypatch.setattr(U.subprocess, "Popen", FakePopen)
        model, _ = U.load_model(str(tiny_ckpt), "cpu")
        ns = U.NodeSpace(model, U.PruningConfig().mask_locations)
        maps = U.build_weight_maps(model, U.PruningConfig().mask_locations)
        summary, new_excl = U.run_iteration(0, set(), args, maps, ns, exp)
        assert summary["n_success"] == 2
        assert summary["success_rate"] == 1.0
        assert summary["n_rank1_nodes"] == len(new_excl) > 0
        assert summary["node_jaccard"]["n"] == 1  # one pair
        saved = json.loads((exp / "iter00" / "iteration_summary.json").read_text())
        assert saved["n_success"] == 2
        # rank-1 nodes are exactly the ties at max universality
        sets = [U.circuit_nodes(torch.load(exp / "iter00" / f"seed{s}_circuit.pt",
                                           map_location="cpu", weights_only=True))
                for s in (0, 1)]
        freq = U.node_universality(sets)
        r1, _ = U.rank1_nodes(freq)
        assert new_excl == set(r1)

    def test_run_iteration_zero_success(self, tmp_path, tiny_ckpt,
                                        patched_tokenizer, monkeypatch):
        exp, args = self._setup_exp(tmp_path, tiny_ckpt, target_loss=-1.0)
        monkeypatch.setattr(U.subprocess, "Popen", FakePopen)
        model, _ = U.load_model(str(tiny_ckpt), "cpu")
        ns = U.NodeSpace(model, U.PruningConfig().mask_locations)
        maps = U.build_weight_maps(model, U.PruningConfig().mask_locations)
        summary, new_excl = U.run_iteration(0, set(), args, maps, ns, exp)
        assert summary["n_success"] == 0
        assert new_excl == set()
        assert summary["rank1_nodes"] == []


class TestMainOrchestration:
    def _argv(self, tiny_ckpt, exp, extra=()):
        return (["universality_pruning_experiment.py",
                 "--model", str(tiny_ckpt), "--tokenizer", "fake",
                 "--task", "dummy_pronoun", "--target-loss", "1e6",
                 "--num-seeds", "2", "--num-steps", "2", "--batch-size", "2",
                 "--eval-batches", "1", "--bisect-iters", "2",
                 "--num-workers", "1", "--max-iters", "1", "--skip-carbs",
                 "--exp-dir", str(exp), "--device", "cpu"] + list(extra))

    def test_main_runs_one_iteration_and_saves_state(
            self, tmp_path, tiny_ckpt, patched_tokenizer, monkeypatch):
        exp = tmp_path / "exp"
        monkeypatch.setattr(U.subprocess, "Popen", FakePopen)
        monkeypatch.setattr(sys, "argv", self._argv(tiny_ckpt, exp))
        U.main()
        state = json.loads((exp / "state.json").read_text())
        assert state["next_iter"] == 1
        assert len(state["history"]) == 1
        assert state["excluded"]  # rank-1 nodes accumulated
        assert not state["exhausted"]
        assert (exp / "hparams.json").exists()
        assert (exp / "run_args.json").exists()

    def test_main_marks_exhausted_when_no_success(
            self, tmp_path, tiny_ckpt, patched_tokenizer, monkeypatch):
        exp = tmp_path / "exp"
        argv = self._argv(tiny_ckpt, exp)
        argv[argv.index("--target-loss") + 1] = "-1.0"
        monkeypatch.setattr(U.subprocess, "Popen", FakePopen)
        monkeypatch.setattr(sys, "argv", argv)
        U.main()
        state = json.loads((exp / "state.json").read_text())
        assert state["exhausted"] is True
        assert state["next_iter"] == 0

    def test_main_refuses_identity_mismatch_on_resume(
            self, tmp_path, tiny_ckpt, patched_tokenizer, monkeypatch, capsys):
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "run_args.json").write_text(json.dumps(
            {"model": "OTHER", "task": "dummy_pronoun"}))
        monkeypatch.setattr(sys, "argv", self._argv(tiny_ckpt, exp))
        with pytest.raises(SystemExit, match="use a fresh --exp-dir"):
            U.main()

    def test_main_noop_when_already_exhausted(
            self, tmp_path, tiny_ckpt, patched_tokenizer, monkeypatch, capsys):
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "state.json").write_text(json.dumps(
            {"next_iter": 5, "excluded": [], "history": [],
             "exhausted": True}))
        (exp / "hparams.json").write_text(json.dumps(U.CENTER_HPARAMS))
        monkeypatch.setattr(sys, "argv", self._argv(tiny_ckpt, exp))
        U.main()
        assert "already marked exhausted" in capsys.readouterr().out
        assert json.loads((exp / "state.json").read_text())["next_iter"] == 5
