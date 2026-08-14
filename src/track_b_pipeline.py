"""
track_b_pipeline.py — RUL-Bench model-trainer subagent (Phase 2, Track B)

Orchestrator: data-engineer's integrity checks -> nested_cv.py ->
nested_stacking.py -> results/tables/{clean_grouped_cv_metrics,
nested_cv_metrics, nested_stacking_metrics}.csv.

HARD STOP: this script (and every module it imports/drives -- nested_cv.py,
nested_stacking.py, fold_safe_pipeline.py, data_integrity.py) must NEVER
load or touch data/processed/test.csv. This is the "freeze checkpoint"
boundary from the Phase 2 plan: Track B's whole point is producing
leakage-safe hyperparameter-search and model-selection numbers WITHOUT ever
letting the official test set influence any decision, so a later subagent
(leakage-red-team) can score the frozen result on test.csv exactly once.

This is enforced at the CODE level, not just by omission/comment: importing
this module monkey-patches `pandas.read_csv` for the remainder of the
process so that any attempt -- by this file OR by anything it imports
(nested_cv.py, nested_stacking.py, data_integrity.py, fold_safe_pipeline.py,
or any future addition) -- to read a path whose basename is `test.csv`
raises `RuntimeError` immediately. `data_integrity.py`'s own checks DO
legitimately read test.csv (that's their whole job -- verifying both splits
independently of Track B) so this script calls a test.csv-free SUBSET of
those checks directly (train.csv-only assertions) rather than
`data_integrity.run_all_checks()`, which reads both files.

GATE: `pytest tests/ -v` must be green before anyone proceeds past this
stage (a later subagent, leakage-red-team, builds that suite; this script
does not build or run it -- it only documents the gate here so the next
subagent's own entry point references the same requirement).

Run: `python src/track_b_pipeline.py` from repo root.
"""

from __future__ import annotations

import os
import sys
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Hard-stop guard: patch pandas.read_csv BEFORE importing anything else in
# this process, so nested_cv.py / nested_stacking.py (imported below) are
# covered by the guard for their entire lifetime in this process too.
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

_FORBIDDEN_BASENAME = "test.csv"
_original_read_csv = pd.read_csv


def _guarded_read_csv(filepath_or_buffer, *args, **kwargs):
    try:
        path_str = os.fspath(filepath_or_buffer)
    except TypeError:
        path_str = None  # e.g. an in-memory buffer, not a path -- not our concern
    if path_str is not None and os.path.basename(str(path_str)).lower() == _FORBIDDEN_BASENAME:
        raise RuntimeError(
            f"track_b_pipeline.py hard-stop: refused to read '{path_str}' -- this "
            "orchestrator (and everything it imports) must NEVER touch any "
            "*/test.csv path before the Phase 2 freeze checkpoint. Track B's "
            "nested-CV numbers must be produced with zero test-set influence. "
            "If a later, explicitly test.csv-aware stage needs this file "
            "(leakage-red-team's final_eval.py, post-freeze), that lives in a "
            "separate module that does not import track_b_pipeline."
        )
    return _original_read_csv(filepath_or_buffer, *args, **kwargs)


pd.read_csv = _guarded_read_csv

# ---------------------------------------------------------------------------
# Now safe to import the rest -- any accidental test.csv read anywhere in
# this call graph will raise immediately, not silently succeed.
# ---------------------------------------------------------------------------
import data_integrity as di  # noqa: E402
import nested_cv as ncv  # noqa: E402
import nested_stacking as nst  # noqa: E402

REPO_ROOT = ncv.REPO_ROOT
TABLES_DIR = ncv.TABLES_DIR


def run_train_only_integrity_checks() -> list[dict]:
    """Subset of data_integrity.py's checks that only ever touch train.csv
    -- deliberately NOT data_integrity.run_all_checks(), which also loads
    test.csv (legitimately, for its own purpose) and would trip the guard
    above if called from inside this hard-stopped module."""
    train_df = _original_read_csv(di.TRAIN_PATH)  # data_integrity's own path constant
    checks = [
        ("train_no_duplicate_engine_cycle",
         lambda: di.assert_no_duplicate_engine_cycle(train_df, "train.csv")),
        ("train_sorted_by_engine_cycle",
         lambda: di.assert_sorted_by_engine_cycle(train_df, "train.csv")),
    ]
    results = []
    for name, fn in checks:
        try:
            detail = fn()
            results.append({"check": name, "passed": True, "detail": detail})
        except AssertionError as e:
            results.append({"check": name, "passed": False, "detail": str(e)})
    n_train = int(train_df["engine_id"].nunique())
    results.append({
        "check": "train_engine_count_100",
        "passed": n_train == 100,
        "detail": {"train_engines": n_train},
    })
    return results


def main():
    t_start = time.time()

    print("=" * 78)
    print("[track_b_pipeline] STEP 1/3 -- train-only data integrity checks")
    print("=" * 78)
    integrity_results = run_train_only_integrity_checks()
    n_pass = sum(r["passed"] for r in integrity_results)
    for r in integrity_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}: {r['detail']}")
    if n_pass != len(integrity_results):
        raise SystemExit(
            f"[track_b_pipeline] {len(integrity_results) - n_pass} train-only integrity "
            f"check(s) failed -- refusing to proceed to nested CV."
        )
    print(f"[track_b_pipeline] {n_pass}/{len(integrity_results)} train-only integrity checks passed.")

    print("\n" + "=" * 78)
    print("[track_b_pipeline] STEP 2/3 -- nested_cv.run_nested_cv (all 10 models)")
    print("=" * 78)
    ncv._smoke_test_groups_required()
    ncv_results = ncv.run_nested_cv(train_path=ncv.TRAIN_PATH, models=ncv.ALL_MODELS, verbose=True)
    ncv.write_outputs(ncv_results, tables_dir=TABLES_DIR)

    print("\n" + "=" * 78)
    print("[track_b_pipeline] STEP 3/3 -- nested_stacking (reusing nested_cv's OOF predictions, "
          "zero additional base-model fits)")
    print("=" * 78)
    stacking_results = nst.run_nested_stacking(ncv_results["oof_df"], verbose=True)
    final_meta, final_meta_params = nst.fit_final_meta_learner(ncv_results["oof_df"])
    nst.write_outputs(stacking_results, final_meta_params, tables_dir=TABLES_DIR)

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"[track_b_pipeline] DONE in {total_elapsed/60:.2f} min. "
          f"test.csv was never read (hard-stop guard active throughout).")
    print("[track_b_pipeline] GATE: pytest tests/ -v must be green before anyone proceeds "
          "past this stage (leakage-red-team subagent builds that suite).")
    print("=" * 78)

    clean_df = ncv.summarize_clean_grouped_cv(ncv_results["fold_metrics_df"])
    print("\nclean_grouped_cv_metrics.csv (mean +/- std across 5 outer folds):")
    print(clean_df.to_string(index=False))
    print("\nnested_stacking_metrics.csv (mean +/- std across 5 outer folds):")
    sm = stacking_results["fold_metrics_df"]
    print(f"  R2   mean={sm['R2'].mean():.4f} std={sm['R2'].std():.4f}")
    print(f"  MSE  mean={sm['MSE'].mean():.2f} std={sm['MSE'].std():.2f}")
    print(f"  MAE  mean={sm['MAE'].mean():.2f} std={sm['MAE'].std():.2f}")
    print(f"  PHM08 mean={sm['PHM08_RUL_Score'].mean():.1f} std={sm['PHM08_RUL_Score'].std():.1f}")


if __name__ == "__main__":
    main()
