"""Tests for scripts/family_clustering.py: incidence/Jaccard math, silhouette
and average-linkage model selection on planted clusters, cluster cores, the
Bernoulli null, run loading, and a full main() over two synthetic runs."""
import json

import numpy as np
import pytest
import torch

import sparse_pretrain.scripts.family_clustering as FC
from tests.conftest import make_circuit_mask, make_exp_dir


class TestBasics:
    def test_circuit_nodes_string_form(self):
        mask = {"layer0_attn_in": torch.tensor([1.0, 0.0, 1.0])}
        assert FC.circuit_nodes(mask) == frozenset(
            {"layer0_attn_in#0", "layer0_attn_in#2"})

    def test_bool_matrix(self):
        sets = [frozenset({"a", "b"}), frozenset({"b", "c"})]
        X, nodes = FC.bool_matrix(sets)
        assert nodes == ["a", "b", "c"]
        assert X.tolist() == [[True, True, False], [False, True, True]]

    def test_jaccard_from_bool_known_answer(self):
        X = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
        J = FC.jaccard_from_bool(X)
        assert J[0, 0] == pytest.approx(1.0)
        assert J[0, 1] == pytest.approx(1 / 3)
        assert J[0, 2] == pytest.approx(2 / 3)
        assert np.allclose(J, J.T)

    def test_jacc(self):
        assert FC.jacc({1, 2}, {2, 3}) == pytest.approx(1 / 3)
        assert np.isnan(FC.jacc(set(), set()))


class TestSilhouetteAndClustering:
    def planted(self, n_per=5, sep=0.9):
        """Distance matrix with two tight clusters far apart."""
        n = 2 * n_per
        D = np.full((n, n), sep)
        D[:n_per, :n_per] = 0.1
        D[n_per:, n_per:] = 0.1
        np.fill_diagonal(D, 0.0)
        return D

    def test_silhouette_perfect_clusters_near_one(self):
        D = self.planted()
        labels = np.array([1] * 5 + [2] * 5)
        sil = FC.silhouette(D, labels)
        # b=0.9, a=0.1 -> (0.9-0.1)/0.9
        assert sil == pytest.approx((0.9 - 0.1) / 0.9, abs=1e-6)

    def test_silhouette_single_cluster_zero(self):
        D = self.planted()
        assert FC.silhouette(D, np.ones(10, dtype=int)) == 0.0

    def test_best_clustering_finds_planted_k2(self):
        J = 1.0 - self.planted()
        k, labels, sil = FC.best_clustering(J, kmax=5)
        assert k == 2
        assert sil > 0.8
        # the two blocks land in different clusters
        assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
        assert labels[0] != labels[-1]

    def test_cluster_cores_threshold(self):
        X = np.array([[1, 1, 0, 1],
                      [1, 1, 0, 0],
                      [0, 0, 1, 0]], dtype=bool)
        nodes = ["a", "b", "c", "d"]
        labels = np.array([1, 1, 2])
        cores = FC.cluster_cores(X, nodes, labels, frac=0.8)
        assert cores[1] == {"a", "b"}  # d only in 50% of cluster 1
        assert cores[2] == {"c"}

    def test_null_best_sils_shape_and_range(self):
        q = np.full(20, 0.5)
        sils = FC.null_best_sils(q, N=8, kmax=4, reps=5)
        assert sils.shape == (5,)
        assert np.all(sils <= 1.0)


class TestLoadRun:
    def test_load_run_reads_successful_circuits(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=4,
                           fail_last_seed=True)
        run = FC.load_run(exp)
        assert set(run) == {0, 1}
        assert run[0]["N"] == 3
        assert run[0]["seeds"] == [0, 1, 2]
        assert len(run[0]["sets"]) == 3
        assert all(isinstance(s, frozenset) for s in run[0]["sets"])
        # rank1 comes from the summary
        summary = json.loads(
            (exp / "iter00" / "iteration_summary.json").read_text())
        assert run[0]["rank1"] == set(summary["rank1_nodes"])


class TestMain:
    def test_full_run_two_synthetic_experiments(self, tmp_path, monkeypatch,
                                                capsys):
        orig = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=6,
                            name="orig")
        rep = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=6,
                           name="repeat")
        monkeypatch.setattr(FC, "RUNS", [("original", orig),
                                         ("repeat1", rep)])
        monkeypatch.setattr(FC, "OUT_DIR", rep)
        monkeypatch.setattr(FC, "NULL_REPS", 3)
        monkeypatch.setattr(FC, "KMAX", 4)
        FC.main()
        report = json.loads((rep / "family_clustering.json").read_text())
        assert set(report) == {"original", "repeat1", "transition"}
        its = report["original"]["iterations"]
        assert set(its) == {"0", "1"}
        rec = its["0"]
        assert rec["N"] == 6
        assert "silhouette" in rec and "cores" in rec
        assert sum(rec["cluster_sizes"].values()) == 6
        # families assigned to every cluster
        assert set(rec["families"]) == set(rec["cores"])
        assert report["original"]["lineage"]
        # transition analysis needs iters 3..6; with 2 iters it stays empty
        assert report["transition"] == {}
        npz = np.load(rep / "family_clustering.npz")
        assert "original_iter00_J" in npz
        assert npz["original_iter00_J"].shape == (6, 6)
        assert "repeat1_iter01_labels" in npz
        out = capsys.readouterr().out
        assert "wrote" in out

    def test_transition_analysis_with_seven_iterations(self, tmp_path,
                                                       monkeypatch):
        orig = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=6, name="o7")
        rep = make_exp_dir(tmp_path, n_iters=7, seeds_per_iter=6, name="r7")
        monkeypatch.setattr(FC, "RUNS", [("original", orig), ("repeat1", rep)])
        monkeypatch.setattr(FC, "OUT_DIR", rep)
        monkeypatch.setattr(FC, "NULL_REPS", 2)
        monkeypatch.setattr(FC, "KMAX", 3)
        FC.main()
        report = json.loads((rep / "family_clustering.json").read_text())
        trans = report["transition"]
        # iterations 3..6 exist in the repeat run -> the survivor/core
        # transition analysis populates
        assert "cores3_vs_cores6" in trans
        assert any(k.startswith("iter4_survivor") for k in trans)

    def test_small_iteration_skips_clustering(self, tmp_path, monkeypatch):
        orig = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=2, name="o")
        rep = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=2, name="r")
        monkeypatch.setattr(FC, "RUNS", [("original", orig), ("repeat1", rep)])
        monkeypatch.setattr(FC, "OUT_DIR", rep)
        FC.main()
        report = json.loads((rep / "family_clustering.json").read_text())
        rec = report["original"]["iterations"]["0"]
        assert rec["k"] == 1  # below MIN_N_CLUSTER -> single cluster
        assert rec["clustered"] is False
