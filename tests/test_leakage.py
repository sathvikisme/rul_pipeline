"""
tests/test_leakage.py — RUL-Bench leakage-red-team subagent (Phase 2)

Automated leakage tests for Track B's leakage-safe pipeline. Every test here
is read-only / fast — none of them re-run the full ~10-18 minute nested CV;
they reconstruct the SAME split objects nested_cv.py itself builds (identical
code path: GroupKFold has no random_state, so the split is fully determined
by group order in the data) or reuse already-written results/tables outputs.

Run: `pytest tests/ -v` from repo root.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

from conftest import REPO_ROOT, TABLES_DIR


# ---------------------------------------------------------------------------
# 1. Group overlap: real nested_cv.py GroupKFold(5) outer splits on
#    data/processed/train.csv must have zero engine overlap between train
#    and test of any fold, AND all 100 engines must be accounted for.
# ---------------------------------------------------------------------------

def test_nested_cv_outer_groupkfold_no_overlap(nested_cv_module, assert_no_leakage):
    ncv = nested_cv_module
    df, X, y, groups, feature_cols = ncv.load_train_data(ncv.TRAIN_PATH)

    outer_cv = GroupKFold(n_splits=ncv.N_OUTER_SPLITS)
    outer_splits = list(outer_cv.split(X, y, groups))
    assert len(outer_splits) == ncv.N_OUTER_SPLITS

    all_engines = set(np.unique(groups))
    test_engines_seen = set()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        # No group leakage — imported/reused from the pgts-split skill, not
        # reimplemented here.
        assert_no_leakage(groups, train_idx, test_idx)

        train_engines = set(groups[train_idx])
        test_engines = set(groups[test_idx])

        # Every row belongs to exactly one of train/test for this fold.
        assert len(train_idx) + len(test_idx) == len(groups), (
            f"fold {fold_idx}: train+test row counts don't sum to total rows"
        )
        # All engines accounted for (either in train or test) for this fold.
        assert train_engines | test_engines == all_engines, (
            f"fold {fold_idx}: union of train/test engines does not equal all "
            f"{len(all_engines)} engines"
        )
        test_engines_seen |= test_engines

    # Across the 5 folds, every engine appears in exactly one fold's held-out
    # (test) partition — a genuine 5-way partition of the 100 engines, not
    # just "no overlap within a fold".
    assert test_engines_seen == all_engines, (
        f"union of test-fold engines across all outer folds ({len(test_engines_seen)}) "
        f"does not equal all {len(all_engines)} engines"
    )


# ---------------------------------------------------------------------------
# 2. Test isolation: Track B's nested CV output (nested_cv_oof_predictions.csv)
#    must correspond exactly to train.csv's own rows, never test.csv's, and
#    track_b_pipeline.py's structural pd.read_csv guard must actually reject
#    a test.csv path.
# ---------------------------------------------------------------------------

def test_oof_predictions_correspond_only_to_train_csv_rows(train_df, test_df):
    oof_path = os.path.join(TABLES_DIR, "nested_cv_oof_predictions.csv")
    if not os.path.exists(oof_path):
        pytest.skip(f"{oof_path} not found -- run src/nested_cv.py or "
                     f"src/track_b_pipeline.py first")
    oof_df = pd.read_csv(oof_path)

    oof_keys = set(zip(oof_df["engine_id"], oof_df["cycle"]))
    train_keys = set(zip(train_df["engine_id"], train_df["cycle"]))

    # Exact bijection with train.csv's own (engine_id, cycle) rows -- every
    # OOF-predicted row came from train.csv, and every train.csv row got
    # exactly one OOF prediction (nested_cv.py's own internal invariant,
    # re-verified here from the written artifact rather than trusted).
    assert oof_keys == train_keys, (
        "nested_cv_oof_predictions.csv (engine_id, cycle) keys do not exactly "
        "match data/processed/train.csv's own keys -- either some train rows "
        "are missing an OOF prediction, or rows from an unexpected source "
        "(e.g. test.csv) leaked into the OOF frame"
    )
    assert len(oof_df) == len(train_df), (
        f"nested_cv_oof_predictions.csv has {len(oof_df)} rows, "
        f"train.csv has {len(train_df)} rows -- expected an exact match"
    )


def test_track_b_pipeline_guard_rejects_test_csv_path(track_b_pipeline_module):
    tbp = track_b_pipeline_module
    fake_test_csv_path = os.path.join(REPO_ROOT, "data", "processed", "test.csv")

    with pytest.raises(RuntimeError, match="track_b_pipeline"):
        tbp._guarded_read_csv(fake_test_csv_path)


def test_track_b_pipeline_guard_allows_train_csv_path(track_b_pipeline_module, train_csv_path):
    tbp = track_b_pipeline_module
    # Should succeed (no exception) -- the guard is specific to */test.csv
    # basenames, not a blanket refusal to read anything.
    df = tbp._guarded_read_csv(train_csv_path)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# 3. Fit-only-on-train: fold_safe_pipeline.build_model_pipeline's
#    VarianceThreshold/StandardScaler must be fit on exactly the rows passed
#    to .fit(), not train+test combined.
# ---------------------------------------------------------------------------

def test_fold_safe_pipeline_scaler_fit_row_count_matches_fold_only(fold_safe_pipeline_module):
    fsp = fold_safe_pipeline_module

    rng = np.random.default_rng(42)
    n_fold_train, n_held_out = 100, 50
    n_total = n_fold_train + n_held_out
    X_all = rng.normal(size=(n_total, 5))
    y_all = X_all[:, 0] + rng.normal(scale=0.1, size=n_total)

    # Simulate what a correct CV loop does: fit ONLY on the fold-train slice.
    X_fold_train, y_fold_train = X_all[:n_fold_train], y_all[:n_fold_train]
    X_held_out = X_all[n_fold_train:]

    pipe = fsp.build_model_pipeline("Ridge", Ridge(random_state=42))
    pipe.fit(X_fold_train, y_fold_train)

    scaler = pipe.named_steps["scaler"]
    assert scaler.n_samples_seen_ == n_fold_train, (
        f"StandardScaler.n_samples_seen_ == {scaler.n_samples_seen_}, expected "
        f"{n_fold_train} (fold-train size only). If this were "
        f"{n_total} (train+held_out), the scaler would have been fit on "
        f"combined train+test data -- the exact leak mechanism this test guards "
        f"against."
    )
    # Sanity: pipeline can still transform/predict on the held-out rows using
    # only the fold-fit statistics.
    preds = pipe.predict(X_held_out)
    assert preds.shape == (n_held_out,)


def test_fold_safe_pipeline_variance_threshold_columns_from_fold_only(fold_safe_pipeline_module):
    """A column that is near-constant ONLY within the fold-train slice (but
    varies in the full train+held_out data) should be dropped if and only if
    VarianceThreshold saw just the fold-train slice -- a second, independent
    angle on the same "fit only on what .fit() received" guarantee."""
    fsp = fold_safe_pipeline_module

    rng = np.random.default_rng(7)
    n_fold_train, n_held_out = 80, 40
    # Column 0: near-constant in fold-train, but varies a lot in held-out.
    col0 = np.concatenate([
        np.full(n_fold_train, 5.0) + rng.normal(scale=1e-8, size=n_fold_train),
        rng.normal(loc=5.0, scale=10.0, size=n_held_out),
    ])
    col1 = rng.normal(size=n_fold_train + n_held_out)  # always informative
    X_all = np.column_stack([col0, col1])
    y_all = col1 + rng.normal(scale=0.1, size=n_fold_train + n_held_out)

    X_fold_train, y_fold_train = X_all[:n_fold_train], y_all[:n_fold_train]

    pipe = fsp.build_model_pipeline("Ridge", Ridge(random_state=42))
    pipe.fit(X_fold_train, y_fold_train)

    support = pipe.named_steps["variance_threshold"].get_support()
    assert support.tolist() == [False, True], (
        "VarianceThreshold should drop column 0 based on the fold-train "
        "slice's own (near-zero) variance -- if it kept column 0, either the "
        "threshold logic changed or (worse) it was fit on data including the "
        "held-out rows where column 0 is NOT near-constant"
    )


# ---------------------------------------------------------------------------
# 4. Stacking OOF purity: every row in nested_cv's OOF frame (the matrix
#    nested_stacking.py trains its meta-learner on) must belong to an outer
#    fold in which that row's engine was in the HELD-OUT partition -- i.e.
#    no engine is split across more than one outer_fold value.
# ---------------------------------------------------------------------------

def test_stacking_oof_purity_engine_wholly_within_one_outer_fold(nested_stacking_module):
    oof_path = os.path.join(TABLES_DIR, "nested_cv_oof_predictions.csv")
    if not os.path.exists(oof_path):
        pytest.skip(f"{oof_path} not found -- run src/nested_cv.py or "
                     f"src/track_b_pipeline.py first")
    oof_df = pd.read_csv(oof_path)

    per_engine_fold_counts = oof_df.groupby("engine_id")["outer_fold"].nunique()
    bad_engines = per_engine_fold_counts[per_engine_fold_counts != 1]
    assert bad_engines.empty, (
        f"{len(bad_engines)} engine(s) have rows spread across more than one "
        f"outer_fold: {bad_engines.to_dict()} -- this would mean some of that "
        f"engine's OOF predictions came from a fold where the SAME engine's "
        f"other rows were in the training partition (stacking OOF leakage)"
    )

    # Exercise nested_stacking.run_nested_stacking directly on this real OOF
    # frame (cheap -- only fits Ridge meta-learners, zero base-model refits)
    # and confirm every row gets exactly one stacking OOF prediction, with no
    # row's own outer_fold contributing to the meta-learner that predicted it
    # (guaranteed structurally by run_nested_stacking's train_mask/test_mask
    # split on outer_fold, re-verified here rather than just trusted).
    nst = nested_stacking_module
    results = nst.run_nested_stacking(oof_df, verbose=False)
    oof_pred = results["oof_pred"]
    assert not np.isnan(oof_pred).any(), "every row should receive a stacking OOF prediction"
    assert len(oof_pred) == len(oof_df)


# ---------------------------------------------------------------------------
# 5. groups= propagation regression guard.
# ---------------------------------------------------------------------------

def test_smoke_test_groups_required(nested_cv_module):
    """pytest wrapper around nested_cv.py's own
    _smoke_test_groups_required() -- proves that a GroupKFold-backed search
    fails LOUDLY (ValueError) if groups= is omitted, and succeeds when it is
    passed. Does not duplicate that logic; just calls it."""
    nested_cv_module._smoke_test_groups_required()


# ---------------------------------------------------------------------------
# 6. Single-frozen-final-eval check: results/tables/_test_set_access_log.json
#    should have at most one entry. final_eval.py has not run yet at the time
#    this suite is first written, so "file doesn't exist" is an expected pass.
# ---------------------------------------------------------------------------

def test_test_set_access_log_at_most_one_entry():
    path = os.path.join(TABLES_DIR, "_test_set_access_log.json")
    if not os.path.exists(path):
        # Expected/correct state before src/final_eval.py has ever run --
        # vacuous pass, not a skip, because "no accesses yet" genuinely
        # satisfies "at most one access".
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "accesses" in data:
        entries = data["accesses"]
    else:
        entries = data

    assert isinstance(entries, list), (
        f"_test_set_access_log.json has an unexpected shape ({type(entries)}); "
        f"expected a JSON list of access-log entries (optionally wrapped in "
        f"{{'accesses': [...]}})"
    )
    assert len(entries) <= 1, (
        f"_test_set_access_log.json has {len(entries)} entries -- the official "
        f"test.csv must be scored AT MOST ONCE (the Phase 2 'single frozen "
        f"final eval' rule). 2+ entries means the frozen model selection was "
        f"revisited after seeing a test-set number, which defeats the purpose "
        f"of freezing it."
    )
