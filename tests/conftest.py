"""
tests/conftest.py — RUL-Bench leakage-red-team subagent (Phase 2)

Shared pytest fixtures for the automated leakage test suite. Nothing here
modifies any file under src/ or results/ — read-only fixtures plus one
carefully-scoped patch-restore helper for src/track_b_pipeline.py's
pd.read_csv guard (see `track_b_pipeline_module` fixture below for why that
needs special handling).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SKILLS_DIR = os.path.join(REPO_ROOT, ".claude", "skills")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _load_module_from_path(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def train_csv_path() -> str:
    return os.path.join(REPO_ROOT, "data", "processed", "train.csv")


@pytest.fixture(scope="session")
def test_csv_path() -> str:
    return os.path.join(REPO_ROOT, "data", "processed", "test.csv")


# ---------------------------------------------------------------------------
# Real processed data (Track B's actual input files) — loaded ONCE per
# session with the real pandas.read_csv (captured before any module in this
# suite has a chance to monkeypatch it — see track_b_pipeline_module fixture).
# ---------------------------------------------------------------------------

_REAL_READ_CSV = pd.read_csv  # captured at collection time, before any test runs


@pytest.fixture(scope="session")
def train_df(train_csv_path) -> pd.DataFrame:
    return _REAL_READ_CSV(train_csv_path)


@pytest.fixture(scope="session")
def test_df(test_csv_path) -> pd.DataFrame:
    return _REAL_READ_CSV(test_csv_path)


# ---------------------------------------------------------------------------
# Tiny synthetic dataset — same shape convention as
# .claude/skills/pgts-split/pgts.py's own __main__ sanity check: 10 engines,
# 20 cycles each, contiguous by engine, rng.default_rng(0).
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_grouped_data() -> dict:
    rng = np.random.default_rng(0)
    n_engines, n_cycles = 10, 20
    engine_id = np.repeat(np.arange(1, n_engines + 1), n_cycles)
    cycle = np.tile(np.arange(1, n_cycles + 1), n_engines)
    X = rng.normal(size=(len(engine_id), 4))
    y = 100.0 - cycle + rng.normal(scale=1.0, size=len(engine_id))
    return {"engine_id": engine_id, "cycle": cycle, "X": X, "y": y}


# ---------------------------------------------------------------------------
# Skill modules, loaded directly from their file paths (not reimplemented).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pgts_module():
    path = os.path.join(SKILLS_DIR, "pgts-split", "pgts.py")
    return _load_module_from_path(path, "pgts_module_for_tests")


@pytest.fixture(scope="session")
def assert_no_leakage(pgts_module):
    return pgts_module._assert_no_leakage


@pytest.fixture(scope="session")
def phm08_score_fn():
    path = os.path.join(SKILLS_DIR, "phm08-scoring", "score.py")
    return _load_module_from_path(path, "phm08_score_module_for_tests").phm08_score


# ---------------------------------------------------------------------------
# src/ modules used by multiple test files.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def nested_cv_module():
    import nested_cv  # noqa
    return nested_cv


@pytest.fixture(scope="session")
def nested_stacking_module():
    import nested_stacking  # noqa
    return nested_stacking


@pytest.fixture(scope="session")
def fold_safe_pipeline_module():
    import fold_safe_pipeline  # noqa
    return fold_safe_pipeline


@pytest.fixture
def track_b_pipeline_module():
    """Import src/track_b_pipeline.py, which monkeypatches pandas.read_csv
    at import time to reject any */test.csv path (its whole "hard-stop"
    safety mechanism, see that module's docstring). We test that guard
    function directly (calling it as a plain function, not via the live
    pd.read_csv global) and then IMMEDIATELY restore pd.read_csv to the
    true, unpatched function, both here and via a finalizer, so this
    fixture's side effect can never leak into other tests regardless of
    test ordering or repeated (cached) imports.
    """
    import track_b_pipeline as tbp

    pd.read_csv = tbp._original_read_csv  # undo the module-level monkeypatch immediately
    yield tbp
    pd.read_csv = tbp._original_read_csv  # belt-and-suspenders: restore again on teardown
