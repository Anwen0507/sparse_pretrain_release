"""Tests for sparse_pretrain.paths: layout, env overrides, dir creation."""
import os
import subprocess
import sys

from sparse_pretrain import paths


def test_package_and_repo_roots():
    assert paths.PACKAGE_ROOT.name == "sparse_pretrain"
    assert paths.REPO_ROOT == paths.PACKAGE_ROOT.parent
    assert (paths.PACKAGE_ROOT / "paths.py").is_file()


def test_name_pools_location_and_contents():
    assert paths.NAME_POOLS == paths.PACKAGE_ROOT / "data" / "name_pools"
    assert (paths.NAME_POOLS / "name_pool_cast15.json").is_file()


def test_outputs_and_figures_honor_env(tmp_path):
    # OUTPUTS/FIGURES are resolved at import time from SP_OUTPUTS/SP_FIGURES;
    # check the override in a clean interpreter so this test is independent of
    # the session-wide values conftest installed.
    env = dict(os.environ)
    env["SP_OUTPUTS"] = str(tmp_path / "o")
    env["SP_FIGURES"] = str(tmp_path / "f")
    code = ("from sparse_pretrain import paths;"
            "print(paths.OUTPUTS); print(paths.FIGURES)")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    assert lines[0] == str(tmp_path / "o")
    assert lines[1] == str(tmp_path / "f")


def test_defaults_without_env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("SP_OUTPUTS", "SP_FIGURES")}
    code = ("from sparse_pretrain import paths;"
            "print(paths.OUTPUTS); print(paths.FIGURES)")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    assert lines[0] == str(paths.REPO_ROOT / "outputs" / "universality_pruning")
    assert lines[1] == str(paths.REPO_ROOT / "figures")


def test_ensure_outputs_and_figures_create_dirs():
    # Session env points these at a scratch dir (see conftest).
    out = paths.ensure_outputs()
    fig = paths.ensure_figures()
    assert out.is_dir() and out == paths.OUTPUTS
    assert fig.is_dir() and fig == paths.FIGURES
