"""Offline smoke tests: imports resolve, shipped data is intact, config is sane.

These deliberately avoid loading models or datasets (no GPU / network), so they
run anywhere and just guard against import/packaging regressions.
"""
import importlib
import json

import pytest


CORE_MODULES = [
    "sparse_pretrain",
    "sparse_pretrain.paths",
    "sparse_pretrain.src.config",
    "sparse_pretrain.src.model",
    "sparse_pretrain.src.pruning.config",
    "sparse_pretrain.src.pruning.tasks",
    "sparse_pretrain.src.pruning.node_mask",
    "sparse_pretrain.src.pruning.masked_model",
    "sparse_pretrain.src.pruning.discretize",
    "sparse_pretrain.src.pruning.calibrate",
    "sparse_pretrain.src.pruning.trainer",
    "sparse_pretrain.src.pruning.run_pruning",
]


@pytest.mark.parametrize("mod", CORE_MODULES)
def test_core_imports(mod):
    importlib.import_module(mod)


def test_paths_resolve():
    from sparse_pretrain import paths
    assert paths.REPO_ROOT.is_dir()
    assert paths.NAME_POOLS.is_dir()


def test_name_pools_present_and_valid():
    from sparse_pretrain.paths import NAME_POOLS
    pool = json.loads((NAME_POOLS / "name_pool_cast15.json").read_text())
    # 8 female + 7 male single-token cast names, 5 cross-validation folds
    assert len(pool["female"]) == 8
    assert len(pool["male"]) == 7
    assert len(pool["folds"]) == 5
    assert len(pool["per_name"]) == 15


def test_pruning_config_defaults():
    from sparse_pretrain.src.pruning.config import PruningConfig
    cfg = PruningConfig()
    # zero-ablation, contrastive-loss pruning as used throughout the report
    assert cfg.mask_locations, "PruningConfig must define maskable node locations"


def test_task_registry_has_report_tasks():
    from sparse_pretrain.src.pruning.run_pruning import TASK_REGISTRY
    for name in ("dummy_pronoun", "dummy_tense", "dummy_article"):
        assert name in TASK_REGISTRY
