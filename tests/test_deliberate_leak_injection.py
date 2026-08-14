"""
tests/test_deliberate_leak_injection.py — RUL-Bench leakage-red-team subagent
(Phase 2)

Proves the leakage test suite has teeth: builds a DELIBERATELY-BROKEN
variant of Track B's CV protocol -- row-level `KFold(shuffle=True)` in place
of `GroupKFold`, on the SAME data (data/processed/train.csv) -- and confirms:

  (a) the group-overlap assertion (`pgts._assert_no_leakage`, the same
      function test_leakage.py uses to certify the real pipeline) FAILS
      LOUDLY on this broken variant, i.e. it correctly detects the injected
      leak rather than passing tautologically; and
  (b) the resulting R2 is MEASURABLY HIGHER than the real, leakage-safe
      Track B R2 recorded in results/tables/clean_grouped_cv_metrics.csv --
      a live, reproduced demonstration (in this codebase, on this data) of
      the exact leak mechanism the Phase 2 plan is guarding against: rows
      from the same engine (here, via adjacent/nearby cycles of the same
      engine trajectory) ending up on both sides of a "train/test" split.

This mirrors the mechanism Track A independently demonstrates on the raw,
engine_id-inclusive feature set (see src/track_a_reproduction.py,
results/tables/track_a_groupkfold_collapse.csv) -- but here the leak is
injected directly into Track B's own already-leakage-safe artifact for a
head-to-head, same-codebase comparison, not just cited from the paper.

Uses LightGBM specifically (fast, deterministic given fixed seeds) so the
comparison is apples-to-apples against clean_grouped_cv_metrics.csv's own
LightGBM row.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from conftest import TABLES_DIR

SEED = 42
N_SPLITS = 5


def _leaky_row_level_kfold_r2(train_df: pd.DataFrame, fold_safe_pipeline_module) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Deliberately-broken variant: row-level KFold(shuffle=True) instead of
    GroupKFold, on the exact same feature matrix nested_cv.py uses. Returns
    (per_fold_r2, groups, one_example_train_idx, one_example_test_idx)."""
    fsp = fold_safe_pipeline_module
    feature_cols = [c for c in train_df.columns if c not in ("engine_id", "RUL")]
    X = train_df[feature_cols].values
    y = train_df["RUL"].values.astype(float)
    groups = train_df["engine_id"].values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)  # NOT grouped -- the injected bug
    r2s = []
    first_train_idx = first_test_idx = None
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        if fold_idx == 0:
            first_train_idx, first_test_idx = train_idx, test_idx
        pipe = fsp.build_model_pipeline(
            "LightGBM", LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1)
        )
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        r2s.append(r2_score(y[test_idx], preds))
    return r2s, groups, first_train_idx, first_test_idx


def test_group_overlap_assertion_fails_loudly_on_row_level_kfold(train_df, assert_no_leakage, fold_safe_pipeline_module):
    """(a) The SAME _assert_no_leakage used to certify the real GroupKFold
    pipeline must FAIL (raise AssertionError) when handed a row-level
    KFold(shuffle=True) split of the same engine-grouped data -- proving the
    check has teeth, not just tautological agreement with whatever split it's
    given."""
    groups = train_df["engine_id"].values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(kf.split(groups))

    # Sanity precondition: confirm the injected split really DOES put some
    # engine's rows on both sides (otherwise this test would be vacuous).
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    assert overlap, (
        "test setup problem: row-level KFold(shuffle=True) on this data "
        "happened to produce zero engine overlap, which would make this "
        "leak-detection test meaningless -- re-check N_SPLITS/SEED"
    )

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(groups, train_idx, test_idx)


def test_row_level_kfold_r2_measurably_higher_than_real_track_b_r2(train_df, fold_safe_pipeline_module):
    """(b) The R2 achieved under the broken (row-level KFold) protocol must
    be measurably higher than Track B's real, leakage-safe GroupKFold R2 for
    the same model (LightGBM), read from
    results/tables/clean_grouped_cv_metrics.csv -- a live demonstration, not
    a cited number, of the exact leak mechanism (same-engine rows split
    across train/test) inflating the apparent score."""
    clean_path = os.path.join(TABLES_DIR, "clean_grouped_cv_metrics.csv")
    if not os.path.exists(clean_path):
        pytest.skip(f"{clean_path} not found -- run src/nested_cv.py or "
                     f"src/track_b_pipeline.py first")
    clean_df = pd.read_csv(clean_path)
    clean_row = clean_df[clean_df["model"] == "LightGBM"]
    assert not clean_row.empty, "expected a 'LightGBM' row in clean_grouped_cv_metrics.csv"
    clean_r2_mean = float(clean_row["R2_mean"].iloc[0])

    leaky_r2s, *_ = _leaky_row_level_kfold_r2(train_df, fold_safe_pipeline_module)
    leaky_r2_mean = float(np.mean(leaky_r2s))

    print(f"\n[deliberate-leak] real Track B (GroupKFold) LightGBM R2_mean = {clean_r2_mean:.4f}")
    print(f"[deliberate-leak] broken (row-level KFold)  LightGBM R2_mean = {leaky_r2_mean:.4f}")
    print(f"[deliberate-leak] gap (leaky - clean) = {leaky_r2_mean - clean_r2_mean:+.4f}")

    MIN_MEASURABLE_GAP = 0.005  # small, deliberately conservative floor -- the
    # observed gap on this data is roughly an order of magnitude larger; this
    # threshold exists only so the test fails loudly if the gap were to
    # shrink to noise level, not to require a huge effect size
    assert leaky_r2_mean > clean_r2_mean + MIN_MEASURABLE_GAP, (
        f"expected the leaky row-level-KFold R2 ({leaky_r2_mean:.4f}) to be "
        f"measurably higher than the real, leakage-safe Track B R2 "
        f"({clean_r2_mean:.4f}) by at least {MIN_MEASURABLE_GAP} -- if this "
        f"fails, either the leak mechanism stopped inflating R2 on this data "
        f"(worth investigating) or clean_grouped_cv_metrics.csv is stale"
    )
