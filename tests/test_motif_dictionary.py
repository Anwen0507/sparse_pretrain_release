"""Tests for scripts/motif_dictionary.py: the circuit-incidence loader, the
frequency backbone, held-out NMF rank selection against a column-shuffled
null (on a matrix with planted two-block structure), atom description, the
vs_hard comparison, and a full main() run over a synthetic experiment dir."""
import json
import sys

import numpy as np
import pytest
import torch

import sparse_pretrain.scripts.motif_dictionary as MD
from tests.conftest import EXP_DIM, EXP_KEYS, make_circuit_mask, make_exp_dir


def planted_two_block_matrix(n_per=8, width=6, noise=0):
    """Circuits are one of two disjoint node blocks -> rank-2 structure."""
    X = np.zeros((2 * n_per, 2 * width), dtype=np.float32)
    X[:n_per, :width] = 1.0
    X[n_per:, width:] = 1.0
    return X


class TestLoaders:
    def test_circuit_node_set_matches_experiment_convention(self):
        mask = make_circuit_mask([(EXP_KEYS[0], 1), (EXP_KEYS[2], 4)])
        assert MD.circuit_node_set(mask) == {(EXP_KEYS[0], 1), (EXP_KEYS[2], 4)}

    def test_global_node_space_order_and_index(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=2)
        nodes, index = MD.global_node_space(exp)
        assert len(nodes) == len(EXP_KEYS) * EXP_DIM
        assert nodes[0] == (EXP_KEYS[0], 0)
        assert all(index[nd] == j for j, nd in enumerate(nodes))

    def test_global_node_space_requires_circuits(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit, match="no seed"):
            MD.global_node_space(empty)

    def test_load_iteration_filters_failures(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=4,
                           fail_last_seed=True)
        nodes, index = MD.global_node_space(exp)
        X, seeds = MD.load_iteration(exp / "iter00", index, len(nodes))
        assert X.shape == (3, len(nodes))  # failed seed excluded
        assert seeds == [0, 1, 2]
        assert set(np.unique(X)) <= {0.0, 1.0}
        # row content matches the stored circuit
        cm = torch.load(exp / "iter00" / "seed0_circuit.pt",
                        map_location="cpu", weights_only=True)
        expected = np.zeros(len(nodes), dtype=np.float32)
        for nd in MD.circuit_node_set(cm):
            expected[index[nd]] = 1.0
        assert np.array_equal(X[0], expected)

    def test_load_iteration_empty_dir(self, tmp_path):
        d = tmp_path / "iterXX"
        d.mkdir()
        X, seeds = MD.load_iteration(d, {}, 5)
        assert X.shape == (0, 5) and seeds == []


class TestBackboneAndRelerr:
    def test_frequency_backbone_counts(self):
        freq = np.array([1.0, 0.95, 0.8, 0.6, 0.4, 0.1])
        anodes = [("k", i) for i in range(6)]
        bb = MD.frequency_backbone(freq, anodes, topn=3)
        assert bb["n_freq_ge_90"] == 2
        assert bb["n_freq_ge_70"] == 3
        assert bb["n_freq_ge_50"] == 4
        assert bb["n_freq_ge_30"] == 5
        # top_nodes: top-3 by freq, only those >= 0.5
        assert [d["node"] for d in bb["top_nodes"]] == ["k#0", "k#1", "k#2"]

    def test_relerr(self):
        X = np.array([[3.0, 4.0]])
        assert MD.relerr(X, np.zeros_like(X)) == pytest.approx(1.0)
        assert MD.relerr(X, X) == pytest.approx(0.0)
        # ||X - 0.5X|| / ||X|| = 0.5
        assert MD.relerr(X, 0.5 * X) == pytest.approx(0.5)


class TestHeldoutSelection:
    def test_real_beats_null_on_planted_structure(self):
        Xa = planted_two_block_matrix()
        ks = [1, 2, 3]
        real, null = MD.heldout_curves(Xa, ks, reps=3, heldout_frac=0.25,
                                       base_seed=0)
        assert set(real) == set(ks)
        for k in ks:
            assert {"mean", "se", "n"} <= set(real[k])
        # two perfectly disjoint blocks: rank 2 reconstructs the real data
        # nearly exactly, while the column-shuffled null cannot
        assert real[2]["mean"] < 0.05
        assert null[2]["mean"] > real[2]["mean"]

    def test_select_k_from_crafted_curves(self):
        ks = [1, 2, 3]
        real = {1: {"mean": 0.5, "se": 0.01, "n": 3},
                2: {"mean": 0.10, "se": 0.02, "n": 3},
                3: {"mean": 0.09, "se": 0.02, "n": 3}}
        null = {1: {"mean": 0.5, "se": 0.01, "n": 3},
                2: {"mean": 0.45, "se": 0.02, "n": 3},
                3: {"mean": 0.44, "se": 0.02, "n": 3}}
        sel = MD.select_k(real, null, ks)
        assert sel["k_best"] == 3  # lowest mean
        assert sel["k_parsimonious"] == 2  # within 1 SE of best
        assert sel["k_cooccur"] == 2  # first k with gap > 2 combined SE
        assert sel["saturated"] is False  # k_best == max(ks)

    def test_select_k_all_nan(self):
        ks = [1]
        real = {1: {"mean": float("nan"), "se": float("nan"), "n": 0}}
        sel = MD.select_k(real, real, ks)
        assert sel == {"k_best": None, "k_parsimonious": None,
                       "k_cooccur": None, "saturated": None}

    def test_ks_larger_than_train_yield_nan(self):
        Xa = planted_two_block_matrix(n_per=2, width=3)  # 4 circuits
        real, null = MD.heldout_curves(Xa, [1, 3], reps=2, heldout_frac=0.25,
                                       base_seed=0)
        # with 3 train rows, k=3 > n_tr-1=2 is nan-skipped
        assert np.isnan(real[3]["mean"]) or real[3]["n"] == 0


class TestDescribeAtoms:
    def test_atoms_recover_planted_blocks(self):
        Xa = planted_two_block_matrix(n_per=6, width=5)
        anodes = [("blk", i) for i in range(10)]
        freq = Xa.mean(0)  # every node 0.5
        atoms, recon, H = MD.describe_atoms(Xa, 2, anodes, freq, seed=0,
                                            topn=5)
        assert len(atoms) == 2
        assert recon < 0.05
        assert H.shape == (2, 10)
        tops = [{d["node"] for d in a["top_nodes"]} for a in atoms]
        blocks = [{f"blk#{i}" for i in range(5)},
                  {f"blk#{i}" for i in range(5, 10)}]
        assert tops[0] in blocks and tops[1] in blocks and tops[0] != tops[1]
        for a in atoms:
            assert a["role"] == "specific-motif"  # top freq 0.5 < 0.6
            assert a["mean_share"] == pytest.approx(0.5, abs=0.05)

    def test_backbone_aligned_role(self):
        X = np.ones((6, 4), dtype=np.float32)  # every node in every circuit
        atoms, _, _ = MD.describe_atoms(X, 1, [("k", i) for i in range(4)],
                                        X.mean(0), seed=0, topn=4)
        assert atoms[0]["role"] == "backbone-aligned"
        assert atoms[0]["top_node_mean_freq"] == pytest.approx(1.0)


class TestVsHard:
    def test_missing_summary_returns_none(self, tmp_path):
        assert MD.vs_hard(tmp_path, [], np.zeros((1, 2)), [], np.zeros(2)) is None

    def test_dominant_cluster_matching(self, tmp_path):
        anodes = [("blk", i) for i in range(10)]
        freq = np.full(10, 0.5)
        Xa = planted_two_block_matrix(n_per=6, width=5)
        atoms, _, H = MD.describe_atoms(Xa, 2, anodes, freq, 0, 5)
        summary = {"k": 2, "clustered": True,
                   "cluster_sizes": {"1": 8, "2": 4},
                   "silhouette": 0.9,
                   "cores": {"1": [f"blk#{i}" for i in range(5)],
                             "2": [f"blk#{i}" for i in range(5, 10)]}}
        (tmp_path / "motif_summary.json").write_text(json.dumps(summary))
        out = MD.vs_hard(tmp_path, atoms, H, anodes, freq)
        assert out["hard_k"] == 2
        assert out["dominant_hard_cluster"] == "1"  # largest cluster
        assert out["dominant_core_disc"] == 5  # all core nodes below thr
        assert out["best_match_cosine_disc"] > 0.9  # atom recovers the core
        assert out["jaccard_top15_sanity"] > 0.9


class TestMain:
    def test_full_run_on_synthetic_dir(self, tmp_path, monkeypatch, capsys):
        exp = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=6)
        monkeypatch.setattr(sys, "argv", [
            "motif_dictionary", "--exp-dir", str(exp), "--kmax", "3",
            "--reps", "2", "--min-n", "4", "--seed", "0"])
        MD.main()
        report = json.loads((exp / "motif_dictionary.json").read_text())
        assert report["N_nodes"] == len(EXP_KEYS) * EXP_DIM
        assert len(report["iterations"]) == 2
        rec = report["iterations"][0]
        assert rec["n_success"] == 6
        assert rec["k_parsimonious"] >= 1
        assert "atoms" in rec and "frequency_backbone" in rec
        assert (exp / "motif_dictionary.png").exists()
        assert "wrote" in capsys.readouterr().out

    def test_min_n_skips_small_iterations(self, tmp_path, monkeypatch):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=3)
        monkeypatch.setattr(sys, "argv", [
            "motif_dictionary", "--exp-dir", str(exp), "--kmax", "2",
            "--reps", "2", "--min-n", "10"])
        MD.main()
        report = json.loads((exp / "motif_dictionary.json").read_text())
        assert report["iterations"][0]["skipped"] is True

    def test_iters_filter(self, tmp_path, monkeypatch):
        exp = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=6)
        monkeypatch.setattr(sys, "argv", [
            "motif_dictionary", "--exp-dir", str(exp), "--kmax", "2",
            "--reps", "2", "--min-n", "4", "--iters", "1"])
        MD.main()
        report = json.loads((exp / "motif_dictionary.json").read_text())
        assert [r["iteration"] for r in report["iterations"]] == [1]
