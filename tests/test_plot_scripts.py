"""Tests for the plotting/reporting scripts.

Most of these scripts execute at import time (no main()), reading experiment
dirs resolved from sparse_pretrain.paths.OUTPUTS / sys.argv and writing PNGs
or JSON. The harness here re-imports each module freshly with OUTPUTS/FIGURES
patched to a tmp dir populated by the synthetic experiment builder, then
asserts on the written artifacts. Numeric helpers (jaccard, dump-law
predictions, fold construction) get direct known-answer checks.
"""
import importlib
import json
import sys

import numpy as np
import pytest
import torch

import sparse_pretrain.paths as paths
from tests.conftest import EXP_KEYS, make_circuit_mask, make_exp_dir


@pytest.fixture
def exec_script(monkeypatch):
    """Freshly import (= execute) a scripts module with patched paths/argv."""
    loaded = []

    def _run(name, outputs=None, figures=None, argv=None):
        if outputs is not None:
            monkeypatch.setattr(paths, "OUTPUTS", outputs)
        if figures is not None:
            monkeypatch.setattr(paths, "FIGURES", figures)
        if argv is not None:
            monkeypatch.setattr(sys, "argv", argv)
        mod_name = f"sparse_pretrain.scripts.{name}"
        sys.modules.pop(mod_name, None)
        module = importlib.import_module(mod_name)
        loaded.append(mod_name)
        return module

    yield _run
    for mod_name in loaded:
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# iteration_jaccard (+ _all): importable modules with main()
# ---------------------------------------------------------------------------
class TestIterationJaccard:
    def test_jaccard_helper(self):
        import sparse_pretrain.scripts.iteration_jaccard as IJ
        assert IJ.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
        assert np.isnan(IJ.jaccard(set(), set()))

    def test_iteration_sets_count_filter(self, tmp_path):
        import sparse_pretrain.scripts.iteration_jaccard as IJ
        exp = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=4)
        sets = IJ.iteration_sets(exp)
        assert set(sets) == {0, 1}
        nodes, n_succ = sets[0]
        assert n_succ == 4
        # the shared 5-node core appears in all 4 circuits (count >= 2);
        # seed-specific extras appear once and are filtered out
        assert len(nodes) == len(EXP_KEYS)

    def test_main_writes_outputs(self, tmp_path, monkeypatch, capsys):
        import sparse_pretrain.scripts.iteration_jaccard as IJ
        a = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=4, name="a")
        b = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=4, name="b")
        monkeypatch.setattr(IJ, "RUNS", [("A", a, "tab:blue"),
                                         ("B", b, "tab:orange")])
        IJ.main()
        assert (b / "iteration_jaccard.png").exists()
        results = json.loads((b / "iteration_jaccard.json").read_text())
        assert set(results) == {"A", "B"}
        assert results["A"]["iterations"] == [0, 1, 2]
        M = np.array(results["A"]["pairwise_jaccard"])
        assert M.shape == (3, 3)
        assert np.allclose(np.diag(M), 1.0)

    def test_iteration_jaccard_all_main(self, tmp_path, monkeypatch):
        import sparse_pretrain.scripts.iteration_jaccard_all as IJA
        base = tmp_path / "outputs"
        make_exp_dir(base, n_iters=2, seeds_per_iter=4, name="run1")
        make_exp_dir(base, n_iters=2, seeds_per_iter=4, name="run2")
        monkeypatch.setattr(IJA, "BASE", base)
        IJA.main()
        assert (base / "iteration_jaccard_all.png").exists()
        assert (base / "iteration_jaccard_all_heatmaps.png").exists()
        data = json.loads((base / "iteration_jaccard_all.json").read_text())
        assert set(data) == {"run1", "run2"}
        assert "consecutive_jaccard" in data["run1"]


# ---------------------------------------------------------------------------
# dump_law_check / dump_law_check2: main() + pure prediction math
# ---------------------------------------------------------------------------
class TestDumpLawCheck:
    def test_pred_dump_formula(self):
        import sparse_pretrain.scripts.dump_law_check as DL
        counts = {"a#0": 4, "b#1": 2, "c#2": 4}
        # exclude c#2 (already rank1); q = {a:1.0, b:0.5}; n=3
        pred = DL.pred_dump(counts, N_prev=4, rank1_prev={"c#2"}, n=3)
        assert pred == pytest.approx(1.0 ** 3 + 0.5 ** 3)

    def test_main_prints_tables(self, tmp_path, monkeypatch, capsys):
        import sparse_pretrain.scripts.dump_law_check as DL
        a = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=4, name="a")
        b = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=4, name="b")
        monkeypatch.setattr(DL, "RUNS", [("original", a), ("repeat1", b)])
        DL.main()
        out = capsys.readouterr().out
        assert "original" in out and "repeat1" in out
        assert "counterfactual" in out.lower() or "N'" in out or "pred" in out

    def test_check2_binomial_matches_avalanche_implementation(self):
        import sparse_pretrain.scripts.dump_law_check2 as DL2
        import sparse_pretrain.scripts.test_avalanche_model_all as AV
        q = np.array([0.9, 0.5, 0.3])
        for n in (2, 5, 9):
            ties2, pmn2 = DL2.expected_ties_at_max(q, n)
            assert ties2 == pytest.approx(AV.expected_ties_at_max(q, n),
                                          rel=1e-9)
            assert pmn2 == pytest.approx(AV.p_max_eq_n(q, n), rel=1e-9)

    def test_check2_q_profile(self):
        import sparse_pretrain.scripts.dump_law_check2 as DL2
        prev = {"N": 4, "rank1": {"a#0"},
                "counts": {"a#0": 4, "b#1": 3, "c#2": 1}}
        q = DL2.q_profile(prev)
        assert sorted(q.tolist()) == [0.25, 0.75]

    def test_check2_main(self, tmp_path, monkeypatch, capsys):
        import sparse_pretrain.scripts.dump_law_check2 as DL2
        a = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=4, name="a")
        b = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=4, name="b")
        monkeypatch.setattr(DL2, "RUNS", [("original", a), ("repeat1", b)])
        DL2.main()
        out = capsys.readouterr().out
        assert "new-ties@max" in out or "ties" in out


# ---------------------------------------------------------------------------
# module-level plot scripts
# ---------------------------------------------------------------------------
class TestPlotCrosstask2afc:
    def test_writes_png(self, tmp_path, exec_script, capsys):
        outputs = tmp_path / "outputs"
        exp = outputs / "ss_d128_f1_pronoun"
        exp.mkdir(parents=True)
        tasks = ["dummy_pronoun", "ioi_relaxed", "dummy_tense",
                 "dummy_article"]
        (exp / "accuracy_2afc.json").write_text(json.dumps({
            "crosstask_2afc": {t: {"base": 0.9, "core": 0.6,
                                   "rand_mean": 0.85, "rand_std": 0.02}
                               for t in tasks}}))
        exec_script("plot_crosstask_2afc", outputs=outputs)
        assert (exp / "crosstask_ablation.png").exists()
        assert "regenerated" in capsys.readouterr().out


class TestPlotSuccessRate:
    def test_writes_figure(self, tmp_path, exec_script, capsys):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        figures = tmp_path / "figs"
        (outputs / "iteration_jaccard_all.json").write_text(json.dumps({
            "runA": {"n_success": {"0": 90, "1": 60, "2": 0}},
            "runB": {"n_success": {"0": 80, "1": 10}}}))
        exec_script("plot_success_rate", outputs=outputs, figures=figures)
        assert (figures / "fig_success_rate.png").exists()
        assert "2 runs" in capsys.readouterr().out


class TestPlotLossTrajectory:
    def test_writes_png(self, tmp_path, exec_script):
        outputs = tmp_path / "outputs"
        make_exp_dir(outputs, n_iters=3, seeds_per_iter=4,
                     name="ss_d128_f1_pronoun")
        exec_script("plot_loss_trajectory", outputs=outputs)
        assert (outputs / "ss_d128_f1_pronoun"
                / "mean_loss_trajectory.png").exists()


class TestPlotPeelTrajectory:
    def test_requires_argv_and_writes_png(self, tmp_path, exec_script):
        exp = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=3,
                           extra_history_keys={
                               "n_excluded_this_iter": 2,
                               "n_peel_ablation_steps": 1,
                               "peel_sequence": [1, 1]})
        mod = exec_script("plot_peel_trajectory",
                          argv=["plot_peel_trajectory.py", str(exp)])
        assert (exp / "peel_trajectory.png").exists()
        # helper functions on the loaded module
        assert mod.col("node_jaccard") == [0.8, 0.8]
        xs, ys = mod.xy([None, 5])
        assert (xs, ys) == ([1], [5])

    def test_with_baseline_run(self, tmp_path, exec_script):
        exp = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=3, name="peel")
        base = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=3, name="std")
        exec_script("plot_peel_trajectory",
                    argv=["plot_peel_trajectory.py", str(exp), str(base)])
        assert (exp / "peel_trajectory.png").exists()


class TestPlotMostUniversalCount:
    def test_load_helper_and_pngs(self, tmp_path, exec_script):
        base = tmp_path / "base"
        make_exp_dir(base, n_iters=2, seeds_per_iter=3, name="expA")
        mod = exec_script("plot_most_universal_count",
                          argv=["plot_most_universal_count.py", str(base)])
        assert (base / "expA" / "most_universal_node_count_per_iter.png").exists()
        assert (base / "most_universal_node_count_all.png").exists()
        its, cnt, nsucc, nseed, nr = mod.load(base / "expA")
        assert its == [0, 1]
        assert nsucc == [3, 3]
        assert cnt[0] == round(1.0 * 3)  # max_universality * n_success
        assert nseed == [3, 3]


class TestPlotMotifTrajectory:
    def make_motif_exp(self, base):
        exp = base / "motif"
        for t in range(2):
            d = exp / f"iter{t:02d}"
            d.mkdir(parents=True)
            (d / "motif_summary.json").write_text(json.dumps({
                "iteration": t, "n_success": 5, "k": 2, "silhouette": 0.4,
                "null95": 0.3, "excluded_before": 2 * t,
                "decision": {"n_excise": 3, "motif_id": f"M{t}",
                             "stop": False, "reason": "ok"},
                "survival": {"max_core_jaccard_to_current": 0.5,
                             "reappeared": t == 1, "excised_motif": "M0"},
            }))
        (exp / "run_args.json").write_text(json.dumps(
            {"match_j": 0.3, "model": "tiny"}))
        return exp

    def test_writes_png(self, tmp_path, exec_script):
        exp = self.make_motif_exp(tmp_path)
        mod = exec_script("plot_motif_trajectory",
                          argv=["plot_motif_trajectory.py", str(exp)])
        assert (exp / "motif_trajectory.png").exists()
        assert mod.fnum(float("nan")) is None
        assert mod.fnum("3.5") == 3.5
        assert mod.fnum(None) is None


class TestUniversalityPruningReport:
    def test_writes_all_three_artifacts(self, tmp_path, exec_script, capsys):
        exp = make_exp_dir(tmp_path, n_iters=3, seeds_per_iter=4)
        mod = exec_script("universality_pruning_report",
                          argv=["universality_pruning_report.py", str(exp)])
        assert (exp / "trajectory.png").exists()
        assert (exp / "core_nodes.txt").exists()
        core = json.loads((exp / "core_nodes.json").read_text())
        state = json.loads((exp / "state.json").read_text())
        assert len(core["nodes"]) == len(state["excluded"])
        # loc_sort orders by (layer, location order)
        assert mod.loc_sort("layer1_mlp_out") == (1, 7)
        assert mod.loc_sort("layer0_attn_q") == (0, 1)
        out = capsys.readouterr().out
        assert "held-out generalization" in out.lower() or "held-out" in out


class TestPlotFamilyClustering:
    def test_writes_png(self, tmp_path, exec_script):
        outputs = tmp_path / "outputs"
        out_dir = outputs / "ss_d128_f1_pronoun_repeat1"
        out_dir.mkdir(parents=True)
        heat = {"original": [0, 3, 6, 8], "repeat1": [2, 3, 6, 13]}
        report, npz = {}, {}
        for run, iters in heat.items():
            report[run] = {"iterations": {}}
            for t in iters:
                n = 4
                report[run]["iterations"][str(t)] = {
                    "N": n, "k": 2, "silhouette": 0.5, "null95": 0.4,
                    "clustered": True, "within_J": 0.7, "between_J": 0.2,
                    "families": {"1": "F0", "2": "F1"},
                    "cluster_sizes": {"1": 2, "2": 2},
                    "cores": {"1": ["a#0"], "2": ["b#1"]},
                }
                J = np.full((n, n), 0.2)
                np.fill_diagonal(J, 1.0)
                npz[f"{run}_iter{t:02d}_J"] = J
                npz[f"{run}_iter{t:02d}_labels"] = np.array([1, 1, 2, 2])
        (out_dir / "family_clustering.json").write_text(json.dumps(report))
        np.savez_compressed(out_dir / "family_clustering.npz", **npz)
        exec_script("plot_family_clustering", outputs=outputs)
        assert (out_dir / "family_clustering.png").exists()


class TestPlotNmfDictionary:
    def test_main_over_patched_runs(self, tmp_path, monkeypatch, capsys):
        import sparse_pretrain.scripts.plot_nmf_dictionary as PN
        base = tmp_path / "outputs"
        for name in ("runA", "runB"):
            exp = make_exp_dir(base, n_iters=1, seeds_per_iter=5, name=name)
            (exp / "atom_peel.json").write_text(json.dumps(
                {"atom_trajectory": [{"depth": 0, "k": 2}]}))
        monkeypatch.setattr(PN, "BASE", base)
        monkeypatch.setattr(PN, "RUNS", [("runA", "A label"),
                                         ("runB", "B label")])
        out_png = tmp_path / "fig.png"
        monkeypatch.setattr(sys, "argv", ["plot_nmf_dictionary",
                                          "--out", str(out_png)])
        PN.main()
        assert out_png.exists()
        text = capsys.readouterr().out
        assert "A label" in text and "wrote" in text


class TestBuildCastNamePool:
    FEMALE = ["mia", "kim", "rita", "lily", "alice", "maria", "lena", "anne"]
    MALE = ["leo", "alex", "samuel", "jose", "emmanuel", "peter", "luis"]

    def make_inputs(self, outputs):
        exp = outputs / "ss_d128_f1_pronoun"
        exp.mkdir(parents=True)
        names = self.FEMALE + self.MALE + ["jean"]
        mined = [{"word": n, "freq": 100000 + i, "female_share": 0.5,
                  "ntok": 1} for i, n in enumerate(names)]
        (exp / "mined_multitoken_names.json").write_text(json.dumps(mined))
        per_name = {n: {"gap_mean": 5.0 if n != "jean" else 0.6,
                        "consistency": 1.0, "binary_ce": 0.01}
                    for n in names}
        (outputs / "name_pool_gap1.json").write_text(json.dumps({
            "per_name": per_name, "below_threshold": [],
            "model": "tiny", "n_templates": 45}))

    def test_pool_construction(self, tmp_path, exec_script, capsys):
        outputs = tmp_path / "outputs"
        self.make_inputs(outputs)
        mod = exec_script("build_cast_name_pool", outputs=outputs)
        pool = json.loads((outputs / "name_pool_cast15.json").read_text())
        assert pool["female"] == self.FEMALE
        assert pool["male"] == self.MALE
        assert pool["balanced_female"] == self.FEMALE
        assert pool["balanced_male"] == self.MALE
        assert len(pool["folds"]) == 5
        assert sorted(n for f in pool["folds"] for n in f) == \
            sorted(self.FEMALE + self.MALE)
        assert all(len(f) == 3 for f in pool["folds"])
        # every fold is gender-mixed: at least one female per fold
        for fold in pool["folds"]:
            assert any(n in self.FEMALE for n in fold)
        assert pool["control_no_gender_signal"]["name"] == "jean"
        # jean is the control, kept out of the labelled pool
        assert len(pool["per_name"]) == 15
        assert "jean" not in pool["per_name"]
        # stats() helper pulls from both inputs
        s = mod.stats("mia")
        assert s["name"] == "mia" and s["gap_mean"] == 5.0
        assert "wrote" in capsys.readouterr().out
