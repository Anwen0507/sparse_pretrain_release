"""Tests for the atom/motif intervention toolchain:

* atom_excision.py       — dictionary building, target-atom pick, dose sets,
                           matched-frequency randoms, result aggregation,
                           NNLS loading profiles, resume path, plotting
* epistasis_test.py      — disjoint atom ownership, union-find components
* exclude_kernel.py      — frequency-matched random control
* universality_atom_peel — per-level rank selection, matched random extension
* universality_motif_excision — discriminative cores, state/motif round trips
* atomicity_test.py      — X building, NMF factor padding, Hungarian cosines,
                           and a full main() over a synthetic excision dir
"""
import json
import sys

import numpy as np
import pytest
import torch

import sparse_pretrain.scripts.atom_excision as AE
import sparse_pretrain.scripts.atomicity_test as AT
import sparse_pretrain.scripts.epistasis_test as ET
import sparse_pretrain.scripts.exclude_kernel as EK
import sparse_pretrain.scripts.universality_atom_peel as AP
import sparse_pretrain.scripts.universality_motif_excision as ME
from tests.conftest import EXP_DIM, EXP_KEYS, make_exp_dir


def make_D(freq, H=None, anodes=None):
    """Minimal dictionary dict as produced by atom_excision.build_dictionary."""
    freq = np.asarray(freq, float)
    n = len(freq)
    anodes = anodes or [("layer0_attn_in", i) for i in range(n)]
    H = np.eye(min(2, n), n) if H is None else np.asarray(H, float)
    return dict(anodes=anodes, freq=freq, H=H,
                atoms=[], k=H.shape[0], n_circuits=4,
                col={f"{k}#{i}": c for c, (k, i) in enumerate(anodes)})


class TestAtomExcisionDictionary:
    def test_read_parsimonious_k(self, tmp_path):
        assert AE.read_parsimonious_k(tmp_path, 0, fallback=5) == 5
        (tmp_path / "motif_dictionary.json").write_text(json.dumps(
            {"iterations": [{"iteration": 0, "k_parsimonious": 3},
                            {"iteration": 1, "k_parsimonious": None}]}))
        assert AE.read_parsimonious_k(tmp_path, 0) == 3
        assert AE.read_parsimonious_k(tmp_path, 1, fallback=7) == 7

    def test_build_dictionary_from_synthetic_run(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=5)
        (exp / "motif_dictionary.json").write_text(json.dumps(
            {"iterations": [{"iteration": 0, "k_parsimonious": 2}]}))
        D = AE.build_dictionary(exp, 0)
        assert D["k"] == 2
        assert D["n_circuits"] == 5
        assert D["H"].shape == (2, len(D["anodes"]))
        assert len(D["freq"]) == len(D["anodes"])
        # col maps every active node label to its column
        for label, c in D["col"].items():
            key, idx = label.rsplit("#", 1)
            assert D["anodes"][c] == (key, int(idx))

    def test_build_dictionary_empty_iteration_exits(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=2)
        for p in (exp / "iter00").glob("seed*_result.json"):
            rec = json.loads(p.read_text())
            rec["target_achieved"] = False
            p.write_text(json.dumps(rec))
        with pytest.raises(SystemExit, match="no successful circuits"):
            AE.build_dictionary(exp, 0)

    def test_pick_target_atom_prefers_dominant_disc_core(self, tmp_path):
        # atom 1 loads on the dominant cluster's discriminative core
        freq = np.array([0.9, 0.9, 0.3, 0.3, 0.3])
        H = np.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.0, 0.8]])
        D = make_D(freq, H)
        it_dir = tmp_path / "iter00"
        it_dir.mkdir(parents=True)
        (it_dir / "motif_summary.json").write_text(json.dumps({
            "cluster_sizes": {"1": 5, "2": 2},
            "cores": {"1": ["layer0_attn_in#2", "layer0_attn_in#3"],
                      "2": ["layer0_attn_in#0"]}}))
        atom, sel = AE.pick_target_atom(D, tmp_path, 0, backbone_thr=0.6)
        assert atom == 1
        assert sel["method"] == "cos_disc_to_dominant_core"
        assert sel["dominant_cluster"] == "1"

    def test_pick_target_atom_fallback(self, tmp_path):
        D = make_D([0.3, 0.3])
        D["atoms"] = [{"atom": 1, "role": "specific-motif"},
                      {"atom": 0, "role": "backbone-aligned"}]
        atom, sel = AE.pick_target_atom(D, tmp_path, 0, backbone_thr=0.6)
        assert atom == 1
        assert sel["method"] == "fallback_top_share_specific"

    def test_atom_candidate_cols_ranked_and_filtered(self):
        freq = np.array([0.9, 0.5, 0.5, 0.5, 0.5])
        H = np.array([[5.0, 4.0, 0.6, 3.0, 0.04]])
        D = make_D(freq, H)
        cand = AE.atom_candidate_cols(D, 0, backbone_thr=0.6, load_frac=0.1)
        # col0 is backbone; col4 below 10% of the peak (0.5); ranked by loading
        assert cand == [1, 3, 2]
        assert AE.atom_candidate_cols(make_D([0.5], [[0.0]]), 0, 0.6) == []

    def test_matched_random_cols_freq_matched_disjoint(self):
        rng = np.random.default_rng(0)
        freq = np.array([0.5, 0.5, 0.49, 0.51, 0.1, 0.9])
        D = make_D(freq, np.zeros((1, 6)))
        cand = [0, 1]
        rand = AE.matched_random_cols(D, cand, m=2, rng=rng, backbone_thr=0.6)
        assert len(rand) == 2
        assert not set(rand) & set(cand)
        assert all(D["freq"][j] < 0.6 for j in rand)
        # nearest-frequency pool for 0.5 targets is {2, 3}
        assert set(rand) <= {2, 3, 4}

    def test_cols_to_nodes(self):
        D = make_D([0.1, 0.2])
        assert AE.cols_to_nodes(D, [1, 0]) == [["layer0_attn_in", 1],
                                               ["layer0_attn_in", 0]]

    def test_cos(self):
        assert AE.cos(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
        assert AE.cos(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
        assert AE.cos(np.zeros(2), np.ones(2)) is None


class TestAtomExcisionCollect:
    def test_collect_aggregates_seed_results(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=4,
                           fail_last_seed=True)
        args = type("A", (), {"seed_offset": 0, "num_seeds": 4})
        agg = AE.collect(exp / "iter00", args)
        assert agg["n"] == 4 and agg["n_success"] == 3
        assert agg["feasibility"] == pytest.approx(0.75)
        recs = [json.loads((exp / "iter00" / f"seed{s}_result.json").read_text())
                for s in range(3)]
        assert agg["mean_circuit_size"] == pytest.approx(
            np.mean([r["circuit_size"] for r in recs]))
        assert agg["mean_test_2afc"] == pytest.approx(
            np.mean([r["test_2afc"] for r in recs]))
        assert agg["generalize_rate"] == 1.0

    def test_collect_empty_dir(self, tmp_path):
        (tmp_path / "iter09").mkdir()
        args = type("A", (), {"seed_offset": 0, "num_seeds": 4})
        agg = AE.collect(tmp_path / "iter09", args)
        assert agg == {"n": 0, "n_success": 0, "feasibility": 0.0}

    def test_reprune_condition_resumes_without_subprocess(self, tmp_path,
                                                          monkeypatch):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=4)
        args = type("A", (), {"seed_offset": 0, "num_seeds": 4})

        def explode(*a, **k):
            raise AssertionError("subprocess must not launch on resume")

        monkeypatch.setattr(AE.subprocess, "Popen", explode)
        agg, secs = AE.reprune_condition({"iterN": 0, "excluded": []}, args,
                                         exp)
        assert secs == 0.0
        assert agg["n_success"] == 4

    def test_loadings_profile_projects_onto_dictionary(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=5)
        (exp / "motif_dictionary.json").write_text(json.dumps(
            {"iterations": [{"iteration": 0, "k_parsimonious": 2}]}))
        D = AE.build_dictionary(exp, 0)
        w = AE.loadings_profile(D, exp / "iter00")
        assert w is not None
        assert w.shape == (D["k"],)
        assert (w >= 0).all() and w.sum() > 0
        assert AE.loadings_profile(D, tmp_path / "nowhere") is None

    def test_make_plot_writes_png(self, tmp_path):
        report = {
            "immediate": {"conditions": [
                {"kind": "atom", "dose": 2, "imm_mean_2afc_drop": 0.2},
                {"kind": "random", "dose": 2, "imm_mean_2afc_drop": 0.05}]},
            "reprune": [
                {"kind": "atom", "dose": 2, "feasibility": 0.5,
                 "mean_circuit_size": 40.0},
                {"kind": "random", "dose": 2, "feasibility": 0.9,
                 "mean_circuit_size": 30.0}],
        }
        out = tmp_path / "plot.png"
        AE.make_plot(report, out)
        assert out.exists() and out.stat().st_size > 0


class TestEpistasisHelpers:
    def test_atom_owned_cols_disjoint_ownership(self):
        freq = np.array([0.9, 0.4, 0.4, 0.4, 0.4])
        H = np.array([[9.0, 3.0, 0.1, 0.0, 1.0],
                      [9.0, 1.0, 3.0, 3.0, 1.0]])
        D = make_D(freq, H)
        testable, detail = ET.atom_owned_cols(D, backbone_thr=0.6,
                                              min_share=0.5, min_size=1)
        # col0 is backbone; col1 owned by atom0 (3/4=0.75); cols 2,3 by atom1;
        # col4 tied 1/2 share=0.5 -> argmax atom0 at exactly min_share
        assert set(testable[0]) == {1, 4}
        assert set(testable[1]) == {2, 3}
        owned_all = [c for cols in testable.values() for c in cols]
        assert len(owned_all) == len(set(owned_all))  # disjoint
        assert detail[1] == (0, pytest.approx(0.75))
        # ranked by loading within each atom
        assert testable[0] == [1, 4]

    def test_atom_owned_cols_min_size_filter(self):
        freq = np.array([0.4, 0.4])
        H = np.array([[3.0, 0.0], [0.0, 3.0]])
        testable, _ = ET.atom_owned_cols(make_D(freq, H), 0.6, 0.25,
                                         min_size=2)
        assert testable == {}

    def test_node_label(self):
        D = make_D([0.1, 0.2])
        assert ET.node_label(D, 1) == "layer0_attn_in#1"

    def test_connected_components(self):
        comps = ET.connected_components([1, 2, 3, 4, 5],
                                        [(1, 2), (2, 3), (4, 5)])
        assert sorted(map(tuple, comps)) == [(1, 2, 3), (4, 5)]
        singletons = ET.connected_components([1, 2], [])
        assert sorted(map(tuple, singletons)) == [(1,), (2,)]


class TestExcludeKernelMatchedRandom:
    def test_freq_matched_excluding_targets(self):
        rng = np.random.default_rng(1)
        freq = np.array([0.5, 0.5, 0.48, 0.52, 0.1, 0.95])
        active = list(range(6))
        target = [0, 1]
        chosen = EK.matched_random(active, freq, target, m=2,
                                   exclude=set(target), rng=rng, knn=1)
        assert len(chosen) == 2
        assert not set(chosen) & set(target)
        # knn=1 -> deterministic nearest-frequency picks: 0.48 then 0.52
        assert set(chosen) == {2, 3}


class TestAtomPeelHelpers:
    def test_select_level_k_fixed_mode(self):
        Xa = np.ones((6, 8), dtype=np.float32)
        k, method = AP.select_level_k(Xa, {"mode": "fixed", "fixed": 3,
                                           "fallback": 5, "min_n": 2,
                                           "heldout_frac": 0.25, "kmax": 8,
                                           "reps": 2, "seed": 0})
        assert (k, method) == (3, "user_fixed")

    def test_select_level_k_min_n_fallback(self):
        Xa = np.ones((2, 8), dtype=np.float32)
        k, method = AP.select_level_k(Xa, {"mode": "auto", "fixed": 0,
                                           "fallback": 4, "min_n": 10,
                                           "heldout_frac": 0.25, "kmax": 8,
                                           "reps": 2, "seed": 0})
        assert method == "min_n_fallback"
        assert k == 1  # fallback clamped to n-1 = 1

    def test_select_level_k_heldout_path(self):
        from tests.test_motif_dictionary import planted_two_block_matrix
        Xa = planted_two_block_matrix(n_per=8, width=6)
        k, method = AP.select_level_k(Xa, {"mode": "auto", "fixed": 0,
                                           "fallback": 4, "min_n": 4,
                                           "heldout_frac": 0.25, "kmax": 4,
                                           "reps": 3, "seed": 0})
        assert method == "heldout_parsimonious"
        assert 1 <= k <= 4

    def test_matched_random_extend(self):
        rng = np.random.default_rng(0)
        freq = np.array([0.5, 0.5, 0.49, 0.51, 0.9])
        D0 = make_D(freq, np.zeros((1, 5)))
        targets = [("layer0_attn_in", 0), ("layer0_attn_in", 1)]
        out = AP.matched_random_extend(D0, targets, set(targets), rng,
                                       backbone_thr=0.6)
        chosen = {tuple(x) for x in map(tuple, out)}
        assert len(chosen) == 2
        assert not chosen & {("layer0_attn_in", 0), ("layer0_attn_in", 1)}
        # backbone col4 excluded by threshold
        assert ("layer0_attn_in", 4) not in chosen

    def test_factor_level_on_synthetic_iteration(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=5)
        import sparse_pretrain.scripts.motif_dictionary as MD
        nodes, index = MD.global_node_space(exp)
        ksel = {"mode": "fixed", "fixed": 2, "fallback": 2, "min_n": 2,
                "heldout_frac": 0.25, "kmax": 4, "reps": 2, "seed": 0}
        D, units, detail, kinfo = AP.factor_level(
            exp / "iter00", nodes, index, ksel, backbone_thr=0.6,
            min_share=0.25, min_size=1)
        assert kinfo["k"] == 2 and kinfo["k_method"] == "user_fixed"
        assert D["n_circuits"] == 5
        assert isinstance(units, dict)

    def test_factor_level_empty_dir(self, tmp_path):
        (tmp_path / "iterZZ").mkdir()
        out = AP.factor_level(tmp_path / "iterZZ", [], {}, {}, 0.6, 0.25, 1)
        assert out == (None, None, None, None)

    def test_load_circuit_sets(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=3)
        sets = AP.load_circuit_sets(exp / "iter00")
        assert len(sets) == 3
        assert all(isinstance(s, set) for s in sets)
        assert all(isinstance(nd, tuple) for s in sets for nd in s)


class TestMotifExcisionHelpers:
    def test_discriminative_core_strict_level(self):
        X = np.array([[1, 1, 0, 1],
                      [1, 1, 0, 0],
                      [0, 1, 1, 1],
                      [0, 1, 1, 0]], dtype=float)
        nodes = [("k", i) for i in range(4)]
        labels = np.array([1, 1, 2, 2])
        sel, info = ME.discriminative_core(X, nodes, labels, target=1,
                                           core_frac=0.9, margin=0.5)
        # node0: q_in=1, q_out=0 -> selected; node1 not discriminative;
        # node3: q_in=0.5 < core_frac
        assert sel == [("k", 0)]
        assert info["level"] == (0.9, 0.5)
        assert info["selected"][0]["node"] == "k#0"
        assert info["selected"][0]["disc"] == pytest.approx(1.0)

    def test_discriminative_core_relaxes_core_frac(self):
        """RELAX_LADDER keeps the margin and halves core_frac: a node in only
        half the family passes at the second ladder level."""
        X = np.array([[1, 1], [1, 0], [1, 0], [1, 0]], dtype=float)
        nodes = [("k", 0), ("k", 1)]
        labels = np.array([1, 1, 2, 2])
        sel, info = ME.discriminative_core(X, nodes, labels, target=1,
                                           core_frac=1.0, margin=0.4)
        # node1: q_in=0.5, q_out=0, disc=0.5 >= margin but fails core_frac=1.0
        # -> selected once core_frac relaxes to 0.5
        assert sel == [("k", 1)]
        assert info["level"] == (0.5, 0.4)

    def test_state_round_trip_with_motifs(self, tmp_path):
        default = ME.load_state(tmp_path)
        assert default["motifs"] == [] and default["next_iter"] == 0
        motifs = [{"id": "M0", "core": {("a", 1), ("b", 2)},
                   "excised_iter": 3, "excised_nodes": {("a", 1)},
                   "target_cluster": 2, "size": 5}]
        state = {"next_iter": 4, "excluded": [], "history": [],
                 "motifs": ME.motifs_to_state(motifs), "exhausted": False}
        ME.save_state(tmp_path, state)
        loaded = ME.load_state(tmp_path)
        rebuilt = ME.motifs_from_state(loaded)
        assert rebuilt[0]["core"] == {("a", 1), ("b", 2)}
        assert rebuilt[0]["excised_nodes"] == {("a", 1)}
        assert rebuilt[0]["id"] == "M0" and rebuilt[0]["size"] == 5

    def test_null_best_sils_rng_first_signature(self):
        rng = np.random.default_rng(0)
        out = ME.null_best_sils(rng, np.full(10, 0.5), N=6, kmax=3, reps=4)
        assert out.shape == (4,)


class TestAtomicityTest:
    def test_build_X(self):
        index = {("a", 0): 0, ("b", 1): 1, ("c", 2): 2}
        sets = [{("a", 0), ("c", 2)}, {("b", 1), ("zzz", 9)}]
        X = AT.build_X(sets, index)
        assert X.tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

    def test_factor_pads_inactive_columns(self):
        X = np.array([[1, 0, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 1, 1]], dtype=np.float32)
        X[:, 1] = 0.0  # column 1 never active
        H = AT.factor(X, k=2, seed=0)
        assert H.shape == (2, 4)
        assert np.all(H[:, 1] == 0.0)

    def test_factor_clamps_k(self):
        X = np.ones((3, 5), dtype=np.float32)
        H = AT.factor(X, k=10)
        assert H.shape[0] == 2  # min(k, n-1=2, active=5)

    def test_cos_matrix_and_hungarian_identity(self):
        Hs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Ht = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # permuted
        C = AT.cos_matrix(Hs, Ht)
        assert C[0, 1] == pytest.approx(1.0) and C[0, 0] == 0.0
        rec = AT.matched_cosines(Hs, Ht)
        assert np.allclose(rec, [1.0, 1.0])

    def test_load_circuits(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=1, seeds_per_iter=3)
        circuits = AT.load_circuits(exp / "iter00")
        assert len(circuits) == 3
        assert all(isinstance(c, set) for c in circuits)

    def test_main_end_to_end(self, tmp_path, monkeypatch, capsys):
        exp = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=5)
        candidates = [f"{EXP_KEYS[0]}#{i}" for i in range(4)]
        (exp / "atom_excision.json").write_text(json.dumps({
            "k": 2,
            "conditions": [
                {"kind": "control", "dose": 0, "iterN": 0},
                {"kind": "atom", "dose": 4, "iterN": 1},
                {"kind": "random", "dose": 4, "iterN": 2},
            ],
            "target": {"candidate_nodes": candidates},
        }))
        monkeypatch.setattr(sys, "argv", [
            "atomicity_test", "--exp-dir", str(exp), "--dose", "4",
            "--boot", "2", "--seed", "0"])
        AT.main()
        out = json.loads((exp / "atomicity_test.json").read_text())
        assert out["k"] == 2 and out["dose"] == 4
        assert out["n_survivors"] == 1  # k=2 minus the target atom
        assert len(out["rec_atom"]) == out["n_survivors"]
        assert "verdict" in out
        assert "VERDICT" in capsys.readouterr().out
