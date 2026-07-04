"""Tests for:

* test_avalanche_model_all.py — binomial ties-at-max mathematics (checked
  against Monte-Carlo and closed forms), q profiles, peel replay, metric
  formatting, and a full main() over synthetic runs under a fake OUTPUTS.
* metric_clustering_robustness.py — representation builders (typed edges,
  WL kernel, Ruzicka), clustering wrappers, and main() with a mocked model.
* run_carbs_clean.py — config dataclass, CARBS parameter spaces, and
  best-checkpoint saving.
"""
import json
import sys

import numpy as np
import pytest
import torch

import sparse_pretrain.scripts.metric_clustering_robustness as MR
import sparse_pretrain.scripts.test_avalanche_model_all as AV
from tests.conftest import make_exp_dir, make_tiny_model


# ---------------------------------------------------------------------------
# Avalanche model math
# ---------------------------------------------------------------------------
class TestBinomialTies:
    def test_pmf_cdf_match_scipy(self):
        from scipy.stats import binom
        q = np.array([0.2, 0.7])
        n = 6
        pmf, cdf = AV._binom_pmf_cdf(q, n)
        assert pmf.shape == (2, 7)
        for vi, qq in enumerate(q):
            assert np.allclose(pmf[vi], binom.pmf(np.arange(7), n, qq),
                               atol=1e-10)
            assert np.allclose(cdf[vi], binom.cdf(np.arange(7), n, qq),
                               atol=1e-10)

    def test_expected_ties_single_node_is_one(self):
        # with one node, it is always the (unique) max
        assert AV.expected_ties_at_max(np.array([0.4]), 5) == pytest.approx(
            1.0, abs=1e-9)

    def test_expected_ties_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        q = np.array([0.9, 0.5, 0.5, 0.2])
        n = 4
        pred = AV.expected_ties_at_max(q, n)
        draws = rng.binomial(n, q[None, :].repeat(200_000, 0))
        ties = (draws == draws.max(1, keepdims=True)).sum(1)
        assert pred == pytest.approx(ties.mean(), rel=0.02)

    def test_p_max_eq_n_closed_form(self):
        q = np.array([0.5, 0.25])
        n = 2
        expected = 1 - (1 - 0.5 ** 2) * (1 - 0.25 ** 2)
        assert AV.p_max_eq_n(q, n) == pytest.approx(expected, rel=1e-9)

    def test_lower_bound(self):
        assert AV.lower_bound(np.array([0.5, 1.0]), 3) == pytest.approx(
            0.5 ** 3 + 1.0 - 3e-12, abs=1e-6)

    def test_empty_and_degenerate_guards(self):
        assert AV.expected_ties_at_max(np.array([]), 5) == 0.0
        assert AV.expected_ties_at_max(np.array([0.5]), 0) == 0.0
        assert AV.p_max_eq_n(np.array([]), 3) == 0.0
        assert AV.lower_bound(np.array([]), 3) == 0.0

    def test_q_profile_excludes_rank1(self):
        rec = {"N": 4, "rank1": ["a#0"],
               "counts": {"a#0": 4, "b#1": 2, "c#2": 1}}
        q = AV.q_profile(rec)
        assert sorted(q.tolist()) == [0.25, 0.5]


class TestAvalancheLoaders:
    def test_circuit_nodes_string_list(self):
        mask = {"k": torch.tensor([1.0, 0.0, 1.0])}
        assert AV.circuit_nodes(mask) == ["k#0", "k#2"]

    def test_load_run(self, tmp_path):
        exp = make_exp_dir(tmp_path, n_iters=2, seeds_per_iter=3,
                           fail_last_seed=True)
        run = AV.load_run(exp, want_circuits=True)
        assert set(run) == {0, 1}
        rec = run[0]
        assert rec["N"] == 2
        assert rec["obs_dump"] == len(rec["rank1"])
        assert rec["max_u"] == 1.0
        assert set(rec["circuits"]) == {0, 1}
        # counts sum equals total node occurrences across circuits
        assert sum(rec["counts"].values()) == sum(
            len(v) for v in rec["circuits"].values())

    def test_fmt_metrics(self):
        rows = [{"pred_dump": 2.0, "obs_dump": 2},
                {"pred_dump": 4.0, "obs_dump": 8}]
        m = AV.fmt_metrics(rows)
        assert m["n"] == 2
        assert m["mae"] == pytest.approx((0 + 4) / 2)
        assert m["rmse"] == pytest.approx(np.sqrt((0 + 16) / 2))
        assert m["median_pred_over_obs"] == pytest.approx(
            np.median([1.0, 0.5]))

    def test_reconstruct_peel_replays_pool(self):
        rec = {
            "peel_steps": [
                {"step": 0, "top_freq": 1.0, "n_top_nodes": 1,
                 "top_nodes": ["a#0"]},
                {"step": 1, "top_freq": 1.0, "n_top_nodes": 1,
                 "top_nodes": ["b#1"]},
            ],
            "circuits": {0: ["a#0", "b#1"], 1: ["a#0", "c#2"],
                         2: ["b#1", "c#2"]},
        }
        steps = AV.reconstruct_peel(rec)
        assert len(steps) == 2
        assert steps[0]["pool_n"] == 3
        # circuits containing a#0 are set aside -> pool shrinks to 1
        assert steps[1]["pool_n"] == 1
        assert AV.reconstruct_peel({"peel_steps": None}) == []


class TestAvalancheMain:
    def _make_cache(self, base):
        """Synthetic per-run records for all 9 expected runs via the cache."""
        base.mkdir(parents=True, exist_ok=True)
        note = base / "avalanche_model_note"
        note.mkdir(exist_ok=True)

        def std_rec(t):
            return {"N": 4, "counts": {"a#0": 4, "b#1": 2, "c#2": 2,
                                       f"d#{t}": 1},
                    "rank1": ["a#0"], "obs_dump": 1, "max_u": 1.0,
                    "peel_steps": None}

        def peel_rec(t):
            rec = std_rec(t)
            rec["peel_steps"] = [{"step": 0, "top_freq": 1.0,
                                  "n_top_nodes": 1, "top_nodes": ["a#0"]}]
            rec["circuits"] = {s: ["a#0", "b#1", f"s#{s}"] for s in range(4)}
            return rec

        cache = {}
        for run, _ in AV.STD_RUNS:
            cache[run] = {str(t): std_rec(t) for t in range(3)}
        for run, _ in AV.PEEL_RUNS:
            cache[run] = {str(t): peel_rec(t) for t in range(2)}
        (note / "_cache_counts.json").write_text(json.dumps(cache))
        # seed_ensemble reads two state.json files directly
        for name, excl in [("og10names", [["a", 0], ["b", 1]]),
                           ("og10names_seeds100_199", [["a", 0], ["c", 2]])]:
            d = base / name
            d.mkdir(exist_ok=True)
            (d / "state.json").write_text(json.dumps(
                {"next_iter": 2, "excluded": excl, "history": []}))
        return note

    def test_load_all_builds_cache_from_run_dirs(self, tmp_path, monkeypatch):
        base = tmp_path / "outputs"
        for run, _ in AV.STD_RUNS + AV.PEEL_RUNS:
            make_exp_dir(base, n_iters=1, seeds_per_iter=2, name=run)
        note = base / "avalanche_model_note"
        note.mkdir()
        monkeypatch.setattr(AV, "BASE", base)
        monkeypatch.setattr(AV, "NOTE", note)
        monkeypatch.setattr(AV, "CACHE", note / "_cache_counts.json")
        data = AV.load_all()
        assert (note / "_cache_counts.json").exists()
        assert set(data) == {r for r, _ in AV.STD_RUNS + AV.PEEL_RUNS}
        assert data["og10names"][0]["N"] == 2
        # peel runs keep per-seed circuits
        assert "circuits" in data["cast15_nosplit_multiexclude"][0]
        # second call reads the cache (iter keys restored as ints)
        again = AV.load_all()
        assert 0 in again["og10names"]

    def test_main_full_report(self, tmp_path, monkeypatch, capsys):
        base = tmp_path / "outputs"
        note = self._make_cache(base)
        monkeypatch.setattr(AV, "BASE", base)
        monkeypatch.setattr(AV, "NOTE", note)
        monkeypatch.setattr(AV, "CACHE", note / "_cache_counts.json")
        out = AV.main()
        assert (note / "model_test_all_runs.json").exists()
        assert out["seed_ensemble"]["core_overlap"] == 1
        assert out["seed_ensemble"]["core_jaccard"] == pytest.approx(1 / 3)
        assert out["standard"]  # transitions with N>=2 present
        assert out["metrics"]["std_all"]["n"] == len(out["standard"])
        text = capsys.readouterr().out
        assert "DUMP LAW" in text and "PREDICTION 6.1" in text


# ---------------------------------------------------------------------------
# Metric / clustering robustness
# ---------------------------------------------------------------------------
LOCS = ["attn_in", "attn_q", "attn_k", "attn_v", "attn_out", "mlp_in",
        "mlp_neuron", "mlp_out"]


def model_consistent_mask(model, active):
    """Circuit mask keyed like the real node space, dims from the model."""
    import sparse_pretrain.scripts.universality_pruning_experiment as U
    ns = U.NodeSpace(model, LOCS)
    mask = {k: torch.zeros(ns.dim[k]) for k in ns.keys}
    for key, idx in active:
        mask[key][idx] = 1.0
    return mask


class TestMetricRepresentations:
    def test_node_bool_and_jaccard_D(self):
        sets = [{("a", 0), ("b", 1)}, {("b", 1)}]
        X, nodes = MR.node_bool(sets)
        assert nodes == [("a", 0), ("b", 1)]
        D = MR.jaccard_D(X)
        assert D[0, 1] == pytest.approx(1 - 1 / 2)
        assert D[0, 0] == pytest.approx(0.0)

    def test_typed_edges_values(self):
        W = torch.tensor([[1.0, -2.0], [0.0, 3.0]])
        maps = [("src", "dst", W)]
        mask = {"src": torch.tensor([1.0, 1.0]),
                "dst": torch.tensor([1.0, 1.0])}
        edges = MR.typed_edges(mask, maps)
        assert edges[("src", 0, "dst", 0)] == pytest.approx(1.0)
        assert edges[("src", 1, "dst", 0)] == pytest.approx(2.0)
        assert edges[("src", 1, "dst", 1)] == pytest.approx(3.0)
        assert ("src", 0, "dst", 1) not in edges  # zero weight dropped

    def test_ruzicka_D_formula(self):
        e1 = {("s", 0, "d", 0): 2.0, ("s", 1, "d", 0): 1.0}
        e2 = {("s", 0, "d", 0): 1.0, ("s", 2, "d", 0): 1.0}
        D = MR.ruzicka_D([e1, e2])
        # min-sum = 1, max-sum = 2+1+1 = 4 -> distance 0.75
        assert D[0, 1] == pytest.approx(0.75)
        assert D[0, 0] == 0.0

    def test_wl_kernel_identical_graphs_similarity_one(self):
        relabel = {}
        W = torch.ones(2, 2)
        maps = [("src", "dst", W)]
        mask = {"src": torch.tensor([1.0, 0.0]),
                "dst": torch.tensor([0.0, 1.0])}
        g1 = MR.build_graph(mask, maps, relabel)
        g2 = MR.build_graph(mask, maps, relabel)
        K = MR.wl_kernel([g1, g2], h=2, relabel=relabel)
        assert K[0, 1] == pytest.approx(1.0)
        assert np.allclose(np.diag(K), 1.0)

    def test_wl_kernel_distinguishes_structure(self):
        relabel = {}
        W = torch.ones(2, 2)
        maps = [("src", "dst", W)]
        connected = {"src": torch.tensor([1.0, 0.0]),
                     "dst": torch.tensor([1.0, 0.0])}
        lonely = {"src": torch.tensor([1.0, 0.0]),
                  "dst": torch.tensor([0.0, 0.0])}
        g1 = MR.build_graph(connected, maps, relabel)
        g2 = MR.build_graph(lonely, maps, relabel)
        K = MR.wl_kernel([g1, g2], h=1, relabel=relabel)
        assert K[0, 1] < 1.0

    def test_upgma_and_spectral_on_planted_clusters(self):
        n = 8
        D = np.full((n, n), 0.9)
        D[:4, :4] = 0.05
        D[4:, 4:] = 0.05
        np.fill_diagonal(D, 0.0)
        k, lab, sil = MR.upgma(D, kmax=4)
        assert k == 2 and sil > 0.8
        K = 1.0 - D  # similarity
        ks, labs, sils = MR.spectral(K, kmax=4)
        assert ks == 2

    def test_upgma_small_input(self):
        D = np.zeros((2, 2))
        k, lab, sil = MR.upgma(D, kmax=4)
        assert (k, sil) == (1, -1.0)

    def test_silhouette_single_cluster_is_minus_one(self):
        D = np.zeros((3, 3))
        assert MR.silhouette(D, np.ones(3, dtype=int)) == -1.0

    def test_cores_and_jacc(self):
        sets = [{("a", 0), ("b", 1)}, {("a", 0)}, {("c", 2)}]
        labels = np.array([1, 1, 2])
        cores = MR.cores(sets, labels, frac=0.9)
        assert cores[1] == {("a", 0)}
        assert cores[2] == {("c", 2)}
        assert MR.jacc(set(), set()) == 0.0


class TestMetricMain:
    def test_end_to_end_with_mocked_model(self, tmp_path, monkeypatch, capsys):
        model = make_tiny_model()
        run = tmp_path / "run"
        it_dir = run / "iter01"
        it_dir.mkdir(parents=True)
        (run / "run_args.json").write_text(json.dumps({"model": "tiny"}))
        prior_core = [["layer0_attn_in", 0], ["layer0_attn_in", 1]]
        (run / "state.json").write_text(json.dumps(
            {"motifs": [{"id": "M0", "excised_iter": 0, "core": prior_core}]}))
        (it_dir / "excluded_input.json").write_text(json.dumps(
            [["layer0_attn_in", 1]]))
        rng = np.random.default_rng(0)
        for s in range(6):
            active = [("layer0_attn_in", 0),
                      ("layer0_mlp_neuron", int(rng.integers(0, 24))),
                      ("layer1_attn_q", int(rng.integers(0, 16)))]
            torch.save(model_consistent_mask(model, active),
                       it_dir / f"seed{s}_circuit.pt")
        monkeypatch.setattr(MR, "load_model", lambda name, dev: (model, {}))
        monkeypatch.setattr(sys, "argv", [
            "metric_clustering_robustness", str(run), "1",
            "--null-reps", "2", "--wl-h", "1", "--kmax", "3"])
        MR.main()
        out = json.loads((it_dir / "metric_robustness.json").read_text())
        assert out["n_circuits"] == 6
        assert out["prior_motif"] == "M0"
        assert "node-Jaccard / UPGMA" in out["results"]
        for row in out["results"].values():
            assert {"k", "silhouette", "null95", "clustered",
                    "ari_vs_node_jaccard", "survival"} <= set(row)
        assert "metric_robustness.json" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_carbs_clean pure pieces
# ---------------------------------------------------------------------------
class TestRunCarbsClean:
    def test_config_defaults_and_helpers(self):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        cfg = RC.CleanSweepConfig()
        assert cfg.target_loss == 0.15
        assert cfg.get_model_short_name() == "ss_bridges_d1024_f0.015625"
        d = cfg.to_dict()
        assert d["num_runs"] == 32 and "k_coef_center" in d

    def test_param_spaces(self):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        params = RC.get_carbs_param_spaces(k_coef_center=5e-4, lr_center=2e-2)
        names = [p.name for p in params]
        assert names == ["k_coef", "weight_decay", "lr", "beta2",
                         "heaviside_temp"]
        by_name = {p.name: p for p in params}
        assert by_name["k_coef"].search_center == 5e-4
        assert by_name["lr"].search_center == 2e-2

    def test_save_best_checkpoint(self, tmp_path):
        import sparse_pretrain.scripts.run_carbs_clean as RC
        cfg = RC.CleanSweepConfig(output_base_dir=str(tmp_path))
        result = {"mask_state": {"layer0_attn_in": torch.ones(4)},
                  "hparams": {"lr": 0.01}, "circuit_size": 42,
                  "achieved_loss_val": 0.12, "success": True}
        out = tmp_path / "run"
        out.mkdir()
        # stale checkpoint contents are replaced wholesale
        stale = out / "best_checkpoint"
        stale.mkdir()
        (stale / "junk.txt").write_text("old")
        RC.save_best_checkpoint(result, out, cfg)
        ckpt = out / "best_checkpoint"
        assert not (ckpt / "junk.txt").exists()
        masks = torch.load(ckpt / "masks.pt", weights_only=True)
        assert torch.equal(masks["layer0_attn_in"], torch.ones(4))
        saved_cfg = json.loads((ckpt / "config.json").read_text())
        assert saved_cfg["best_circuit_size"] == 42
        assert saved_cfg["best_val_loss"] == 0.12
        assert json.loads((ckpt / "hparams.json").read_text()) == {"lr": 0.01}
        summary = json.loads((ckpt / "summary.json").read_text())
        assert "mask_state" not in summary and summary["success"] is True
