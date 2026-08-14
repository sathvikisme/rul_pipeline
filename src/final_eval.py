"""
final_eval.py — RUL-Bench leakage-red-team subagent (Phase 2, post-freeze)

THE single frozen evaluation of `data/processed/test.csv`. Per the Phase 2
plan (`we-are-moving-to-goofy-rivest.md`, step 4) and
`results/FREEZE_DECISION.md`, two candidates were selected using ONLY
training-engine evidence (nested-GroupKFold CV + paired-stats + engine-level
bootstrap, all pre-registered before this script ever ran):

    1. LightGBM        — single model, mode-selected nested-CV hyperparams.
    2. StackingEnsemble — Ridge meta-learner over 5 base learners (LightGBM,
                           CatBoost, XGBoost, GradientBoosting, MLP), each
                           refit on the full 100-engine training set with
                           their own mode-selected nested-CV hyperparams; the
                           meta-learner is trained on the ALREADY-GENUINE
                           nested out-of-fold predictions in
                           `results/tables/nested_cv_oof_predictions.csv`
                           (nothing here regenerates that OOF matrix).

No decision is made in this script based on what it produces — everything
that determined WHICH two candidates to evaluate, and WHICH hyperparameters
to give them, was fixed before `data/processed/test.csv` is read. This
script enforces (in code, not just by convention) that:

  (a) `data/processed/test.csv` is read exactly once, at a single clearly
      marked point (`STEP 4`), only after every model has already been
      fit on the full training set with hyperparameters chosen from
      training-only evidence.
  (b) `results/tables/_test_set_access_log.json` never accumulates more
      than one entry. Before doing ANY work, this script checks that file;
      if it already has an entry, that means a previous run of this (or
      some other) script already spent the one-shot test-set look, and this
      run refuses to proceed (RuntimeError) rather than silently scoring the
      test set a second time. `tests/test_leakage.py::
      test_test_set_access_log_at_most_one_entry` re-checks this invariant
      independently.

Mode-hyperparameter selection (documented in code, not just by assertion):
for every one of the 5 base learners, `results/tables/nested_cv_best_params.json`
records the hyperparameters `nested_cv.py` chose (via nested RandomizedSearchCV/
GridSearchCV, or Phase 1's fixed Tier-2 params) independently in each of the
5 outer folds. This script canonicalizes each fold's chosen-params dict
(order-independent, list-safe) and takes the most common one. If two or more
param sets are tied for most-common, the tie is broken by picking whichever
tied set has the higher (less negative) MEAN `inner_cv_best_score_neg_mse`
across the folds that chose it — i.e. the tied option nested CV liked best on
average, not an arbitrary/first-seen pick. See `_mode_hyperparams` below.
The one place this tie-break actually fires (of the 5 base learners) is
XGBoost, documented in the printed summary and `final_test_metrics.csv`.

Random seed: 42 throughout (every base learner's `random_state`, the
meta-learner's `random_state`, `GroupKFold`'s split is itself deterministic
and takes no seed).

Run: `python src/final_eval.py` from repo root. Writes:
  - results/tables/final_test_metrics.csv
  - results/tables/_test_set_access_log.json   (created here; exactly 1 entry)
  - results/FINAL_TEST_RESULT.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import nested_cv as ncv  # noqa: E402 -- reuse _base_estimator, load_train_data, _score, SEED
import nested_stacking as nst  # noqa: E402 -- reuse STACK_BASE_LEARNERS, fit_final_meta_learner
from fold_safe_pipeline import build_model_pipeline  # noqa: E402

REPO_ROOT = ncv.REPO_ROOT
TABLES_DIR = ncv.TABLES_DIR
RESULTS_DIR = os.path.dirname(TABLES_DIR)
TRAIN_PATH = ncv.TRAIN_PATH
TEST_PATH = os.path.join(REPO_ROOT, "data", "processed", "test.csv")
NESTED_CV_BEST_PARAMS_PATH = os.path.join(TABLES_DIR, "nested_cv_best_params.json")
NESTED_CV_OOF_PATH = os.path.join(TABLES_DIR, "nested_cv_oof_predictions.csv")
OFFICIAL_SPLIT_METRICS_PATH = os.path.join(TABLES_DIR, "official_split_metrics.csv")
ACCESS_LOG_PATH = os.path.join(TABLES_DIR, "_test_set_access_log.json")
FINAL_METRICS_PATH = os.path.join(TABLES_DIR, "final_test_metrics.csv")
FINAL_REPORT_PATH = os.path.join(RESULTS_DIR, "FINAL_TEST_RESULT.md")

_score = ncv._score
SEED = ncv.SEED

STACK_BASE_LEARNERS = nst.STACK_BASE_LEARNERS  # ["LightGBM","CatBoost","XGBoost","GradientBoosting","MLP"]
SINGLE_MODEL_NAME = "LightGBM"
ALL_MODELS_TO_FIT = STACK_BASE_LEARNERS  # LightGBM is in this list already; reused for both candidates


# ---------------------------------------------------------------------------
# Single-frozen-final-eval discipline (checked BEFORE any real work happens)
# ---------------------------------------------------------------------------

def _read_access_log_entries(path: str = ACCESS_LOG_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data["accesses"] if isinstance(data, dict) and "accesses" in data else data
    if not isinstance(entries, list):
        raise RuntimeError(
            f"{path} has an unexpected shape ({type(entries)}); expected a JSON "
            f"list of access-log entries (optionally wrapped in {{'accesses': [...]}})."
        )
    return entries


def _assert_test_set_not_yet_accessed(path: str = ACCESS_LOG_PATH) -> None:
    entries = _read_access_log_entries(path)
    if len(entries) >= 1:
        raise RuntimeError(
            f"REFUSING TO RUN: {path} already has {len(entries)} entry/entries. "
            f"data/processed/test.csv has already been scored once under the Phase 2 "
            f"'freeze then evaluate once' discipline (see results/FREEZE_DECISION.md). "
            f"Running final_eval.py again would mean reading the official test set a "
            f"second time and would defeat the entire purpose of freezing the model "
            f"selection before ever looking at it. Existing entries: {entries}"
        )


def _write_access_log_entry(models_evaluated: list[str], script_name: str = "final_eval.py",
                             path: str = ACCESS_LOG_PATH) -> dict:
    # Re-check immediately before writing too (paranoia: guards against this
    # function ever being called a second time within a long-lived process).
    _assert_test_set_not_yet_accessed(path)
    entries = _read_access_log_entries(path)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_evaluated": models_evaluated,
        "script": script_name,
    }
    entries.append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entry


# ---------------------------------------------------------------------------
# Mode-hyperparameter selection from nested_cv_best_params.json
# ---------------------------------------------------------------------------

def _canonical(obj):
    """Order-independent, list-safe canonical form for equality/hashing of a
    hyperparameter dict (handles nested lists like MLP's hidden_layer_sizes,
    which are unhashable as-is)."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _canonical(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(_canonical(v) for v in obj)
    return obj


def _mode_hyperparams(model_name: str, log_path: str = NESTED_CV_BEST_PARAMS_PATH) -> dict:
    """Most-common per-outer-fold chosen hyperparameter set for `model_name`
    across nested_cv.py's 5 outer folds. Ties are broken by the tied option
    with the higher (less negative) mean inner_cv_best_score_neg_mse among
    the folds that chose it. Returns a dict with the chosen (pipeline-
    prefixed, "estimator__...") params plus full documentation of the
    grouping/tie-break decision.
    """
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    records = [r for r in log["per_outer_fold_chosen_params"] if r["model"] == model_name]
    if not records:
        raise ValueError(f"no chosen_params records found for model '{model_name}' in {log_path}")

    groups: dict = {}
    for rec in records:
        params = json.loads(rec["chosen_params"])
        key = _canonical(params)
        if key not in groups:
            groups[key] = {"params": params, "scores": [], "folds": []}
        groups[key]["scores"].append(rec.get("inner_cv_best_score_neg_mse"))
        groups[key]["folds"].append(rec["outer_fold"])

    counts = {k: len(v["folds"]) for k, v in groups.items()}
    max_count = max(counts.values())
    tied_keys = [k for k, c in counts.items() if c == max_count]

    tie_broken = len(tied_keys) > 1
    if not tie_broken:
        chosen_key = tied_keys[0]
        tie_break_detail = None
    else:
        means = {}
        for k in tied_keys:
            scores = [s for s in groups[k]["scores"] if s is not None]
            means[k] = float(np.mean(scores)) if scores else float("-inf")
        chosen_key = max(means, key=means.get)
        tie_break_detail = {
            "tied_options": [
                {"params": groups[k]["params"], "folds": groups[k]["folds"],
                 "mean_inner_cv_score_neg_mse": means[k]}
                for k in tied_keys
            ],
            "winner_params": groups[chosen_key]["params"],
            "rule": "higher (less negative) mean inner_cv_best_score_neg_mse among tied-count options",
        }

    chosen_params = groups[chosen_key]["params"]
    detail = {
        "model": model_name,
        "n_folds": len(records),
        "n_distinct_param_sets_across_folds": len(groups),
        "mode_count": max_count,
        "chosen_params": chosen_params,
        "chosen_from_folds": groups[chosen_key]["folds"],
        "tie_broken": tie_broken,
        "tie_break_detail": tie_break_detail,
        "all_groups": [
            {"params": v["params"], "folds": v["folds"], "count": len(v["folds"]),
             "scores": v["scores"]}
            for v in groups.values()
        ],
    }
    return chosen_params, detail


def _strip_prefix(params: dict) -> dict:
    return {k.replace("estimator__", "", 1): v for k, v in params.items()}


# ---------------------------------------------------------------------------
# STEP 1-2: mode hyperparams + full-train-set fit of the 5 base learners
# ---------------------------------------------------------------------------

def fit_all_base_learners(train_path: str = TRAIN_PATH, models: list[str] = ALL_MODELS_TO_FIT,
                           seed: int = SEED, verbose: bool = True) -> dict:
    df, X, y, groups, feature_cols = ncv.load_train_data(train_path)
    if verbose:
        print("=" * 78)
        print(f"[final_eval] STEP 1-2: mode-hyperparam selection + full-train fit "
              f"({len(models)} base learners) on {train_path}")
        print(f"[final_eval] n_rows={len(y)}  n_engines={len(np.unique(groups))}  "
              f"n_features={len(feature_cols)}")
        print("=" * 78)

    fitted = {}
    hp_details = {}
    fit_seconds = {}
    for name in models:
        chosen_params, detail = _mode_hyperparams(name)
        hp_details[name] = detail

        base_est = ncv._base_estimator(name)
        pipe = build_model_pipeline(name, base_est)
        pipe.set_params(**chosen_params)

        t0 = time.time()
        pipe.fit(X, y)
        elapsed = time.time() - t0
        fit_seconds[name] = elapsed
        fitted[name] = pipe

        if verbose:
            tie_note = " (TIE-BROKEN by mean inner score)" if detail["tie_broken"] else ""
            print(f"    [{name:18s}] mode_count={detail['mode_count']}/{detail['n_folds']} "
                  f"folds{tie_note}  params={_strip_prefix(chosen_params)}  "
                  f"fit={elapsed:.2f}s")

    return {
        "fitted": fitted, "hp_details": hp_details, "fit_seconds": fit_seconds,
        "feature_cols": feature_cols, "train_df": df,
    }


# ---------------------------------------------------------------------------
# STEP 3: fit the final Ridge meta-learner on the ALREADY-GENUINE OOF matrix
# ---------------------------------------------------------------------------

def fit_meta_learner(oof_path: str = NESTED_CV_OOF_PATH, base_learners: list[str] = STACK_BASE_LEARNERS,
                      seed: int = SEED, verbose: bool = True):
    if verbose:
        print("\n" + "=" * 78)
        print(f"[final_eval] STEP 3: fit final Ridge meta-learner on {oof_path}")
        print("=" * 78)
    oof_df = pd.read_csv(oof_path)

    # Sanity: this file covers the full 100-engine training set, never test.csv
    # (re-verified here rather than just trusted -- same invariant
    # tests/test_leakage.py checks independently against train.csv itself).
    train_df, _, _, _, _ = ncv.load_train_data(TRAIN_PATH)
    oof_keys = set(zip(oof_df["engine_id"], oof_df["cycle"])) if "cycle" in oof_df.columns \
        else set(oof_df["engine_id"])
    train_keys = set(zip(train_df["engine_id"], train_df["cycle"])) if "cycle" in train_df.columns \
        else set(train_df["engine_id"])
    assert oof_keys == train_keys, (
        "nested_cv_oof_predictions.csv does not exactly match data/processed/train.csv's "
        "own (engine_id, cycle) rows -- refusing to fit the meta-learner on a matrix that "
        "may include rows outside the true 100-engine training set."
    )

    meta, meta_params = nst.fit_final_meta_learner(
        oof_df, base_learners=base_learners, alpha_grid=nst.META_ALPHA_GRID,
        n_splits=5, seed=seed,
    )
    if verbose:
        print(f"    [Ridge meta-learner] alpha={meta_params['alpha']:g}  "
              f"(GroupKFold(5) on the pooled nested-OOF matrix, "
              f"grid={nst.META_ALPHA_GRID})")
    return meta, meta_params, oof_df


# ---------------------------------------------------------------------------
# STEP 4: read test.csv EXACTLY ONCE, generate predictions, score
# ---------------------------------------------------------------------------

def evaluate_on_test_set(fitted_base_learners: dict, meta_learner, feature_cols: list[str],
                          test_path: str = TEST_PATH, verbose: bool = True) -> dict:
    if verbose:
        print("\n" + "=" * 78)
        print(f"[final_eval] STEP 4: reading {test_path} -- THE ONE AND ONLY TIME "
              f"this file may be read by any Phase 2 script.")
        print("=" * 78)

    # The single pd.read_csv call on test.csv for this entire run.
    test_df, X_test, y_test, test_groups, test_feature_cols = ncv.load_train_data(test_path)

    assert test_feature_cols == feature_cols, (
        "Feature-column mismatch between data/processed/train.csv and "
        "data/processed/test.csv -- refusing to score predictions built on "
        "mismatched columns.\n"
        f"train feature_cols: {feature_cols}\ntest feature_cols: {test_feature_cols}"
    )

    n_test_rows = len(y_test)
    n_test_engines = len(np.unique(test_groups))
    if verbose:
        print(f"[final_eval] test.csv: n_rows={n_test_rows}  n_engines={n_test_engines}")

    # -- LightGBM (single model) --
    lgbm_pred = fitted_base_learners[SINGLE_MODEL_NAME].predict(X_test)
    lgbm_r2, lgbm_mse, lgbm_mae, lgbm_phm08 = _score(y_test, lgbm_pred)

    # -- StackingEnsemble --
    meta_X_test = np.column_stack([fitted_base_learners[m].predict(X_test) for m in STACK_BASE_LEARNERS])
    stack_pred = meta_learner.predict(meta_X_test)
    stack_r2, stack_mse, stack_mae, stack_phm08 = _score(y_test, stack_pred)

    if verbose:
        print(f"    [LightGBM]        R2={lgbm_r2:.4f} MSE={lgbm_mse:8.2f} "
              f"MAE={lgbm_mae:6.2f} PHM08={lgbm_phm08:10.1f}")
        print(f"    [StackingEnsemble] R2={stack_r2:.4f} MSE={stack_mse:8.2f} "
              f"MAE={stack_mae:6.2f} PHM08={stack_phm08:10.1f}")

    return {
        "n_test_rows": n_test_rows, "n_test_engines": n_test_engines,
        "LightGBM": {"R2": lgbm_r2, "MSE": lgbm_mse, "MAE": lgbm_mae, "PHM08_RUL_Score": lgbm_phm08},
        "StackingEnsemble": {"R2": stack_r2, "MSE": stack_mse, "MAE": stack_mae, "PHM08_RUL_Score": stack_phm08},
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_final_test_metrics(results: dict, hp_details: dict, meta_params: dict,
                              path: str = FINAL_METRICS_PATH) -> pd.DataFrame:
    lgbm_hp = _strip_prefix(hp_details[SINGLE_MODEL_NAME]["chosen_params"])
    stack_hp = {m: _strip_prefix(hp_details[m]["chosen_params"]) for m in STACK_BASE_LEARNERS}

    rows = [
        {
            "model": "LightGBM", "is_ensemble": False,
            "R2": results["LightGBM"]["R2"], "MSE": results["LightGBM"]["MSE"],
            "MAE": results["LightGBM"]["MAE"], "PHM08_RUL_Score": results["LightGBM"]["PHM08_RUL_Score"],
            "n_test_rows": results["n_test_rows"], "n_test_engines": results["n_test_engines"],
            "hyperparameters_json": json.dumps(lgbm_hp),
            "base_learners": "", "meta_learner_alpha": "",
            "hyperparam_selection": "mode across 5 nested-CV outer folds (tie-broken by mean inner CV score if needed)",
        },
        {
            "model": "StackingEnsemble", "is_ensemble": True,
            "R2": results["StackingEnsemble"]["R2"], "MSE": results["StackingEnsemble"]["MSE"],
            "MAE": results["StackingEnsemble"]["MAE"],
            "PHM08_RUL_Score": results["StackingEnsemble"]["PHM08_RUL_Score"],
            "n_test_rows": results["n_test_rows"], "n_test_engines": results["n_test_engines"],
            "hyperparameters_json": json.dumps(stack_hp),
            "base_learners": ",".join(STACK_BASE_LEARNERS),
            "meta_learner_alpha": meta_params["alpha"],
            "hyperparam_selection": "each base learner: mode across 5 nested-CV outer folds; "
                                     "meta-learner: GroupKFold(5) alpha tuning on full pooled nested-OOF matrix",
        },
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(path, index=False)
    return out_df


def _load_phase1_row(model_name: str, path: str = OFFICIAL_SPLIT_METRICS_PATH):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    match = df[df["model"] == model_name]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def write_final_report(results: dict, hp_details: dict, meta_params: dict,
                        access_log_entry: dict, path: str = FINAL_REPORT_PATH) -> None:
    lgbm = results["LightGBM"]
    stack = results["StackingEnsemble"]

    p1_lgbm = _load_phase1_row("LightGBM")
    p1_stack = _load_phase1_row("Stacking_Ridge")

    def _fmt_row(name, m):
        return (f"| {name} | {m['R2']:.4f} | {m['MSE']:.2f} | {m['MAE']:.2f} | "
                f"{m['PHM08_RUL_Score']:.1f} |")

    def _delta(a, b):
        return a - b if (a is not None and b is not None) else None

    lines = []
    lines.append("# Final Frozen Test-Set Result (Phase 2, single-shot evaluation)\n")
    lines.append(
        "This is the terminal measurement of the Phase 2 'freeze then evaluate once' "
        "process. Model selection (which two candidates, which hyperparameters) was "
        "decided entirely from training-engine evidence in "
        "`results/FREEZE_DECISION.md` and `results/tables/ensemble_selection.csv` "
        "*before* `data/processed/test.csv` was read by this script. No further "
        "model, hyperparameter, or feature decisions may be made based on the numbers "
        "below -- this is a report of what happened, not a selection step.\n"
    )
    lines.append(f"`data/processed/test.csv` access log entry written: "
                  f"`{json.dumps(access_log_entry)}`\n")

    lines.append("## Headline numbers (official held-out test set)\n")
    lines.append(f"n_test_rows={results['n_test_rows']}, "
                  f"n_test_engines={results['n_test_engines']}\n")
    lines.append("| Model | R2 | MSE | MAE | PHM08 RUL Score |")
    lines.append("|---|---|---|---|---|")
    lines.append(_fmt_row("LightGBM", lgbm))
    lines.append(_fmt_row("StackingEnsemble", stack))
    lines.append("")

    lines.append("## Hyperparameters used\n")
    lines.append("Each was the mode (most common) hyperparameter set chosen for that model "
                  "across nested_cv.py's 5 outer GroupKFold folds "
                  "(`results/tables/nested_cv_best_params.json`); ties broken by the mean "
                  "inner-CV score among tied options.\n")
    for name in STACK_BASE_LEARNERS:
        d = hp_details[name]
        tie_note = " -- **TIE-BROKEN** by mean inner CV score" if d["tie_broken"] else ""
        lines.append(f"- **{name}** (mode {d['mode_count']}/{d['n_folds']} folds{tie_note}): "
                      f"`{json.dumps(_strip_prefix(d['chosen_params']))}`")
    lines.append(f"- **Ridge meta-learner alpha**: {meta_params['alpha']:g} "
                  f"(GroupKFold(5) tuning on the pooled nested-OOF matrix, "
                  f"grid={nst.META_ALPHA_GRID})")
    lines.append("")
    xgb_detail = hp_details["XGBoost"]
    if xgb_detail["tie_broken"]:
        lines.append(
            "**Note on the XGBoost tie-break** (the only base learner where the mode was "
            "ambiguous): two hyperparameter sets each won 2 of the 5 outer folds. The tied "
            f"option was resolved to the one with the better mean inner-CV score: "
            f"{json.dumps(xgb_detail['tie_break_detail'], default=str)}\n"
        )

    lines.append("## Comparison to Phase 1's official-split numbers\n")
    lines.append(
        "**Caveat up front: these are NOT the same methodology**, so a raw before/after "
        "delta partly reflects methodology changes, not just 'the leakage-safe process "
        "picked a worse/better model.' Differences include: Phase 1's LightGBM hyperparameters "
        "came from a single `GridSearchCV` under a plain (non-grouped, row-level) `KFold`; "
        "Phase 2's LightGBM hyperparameters are the mode across 5 genuinely engine-grouped "
        "nested-CV outer folds. Phase 1's `Stacking_Ridge` used `cross_val_predict(KFold(5, "
        "shuffle=True))` (row-level, non-grouped) OOF predictions for its meta-learner and its "
        "base learners' hyperparameters also came from plain-KFold `GridSearchCV`; Phase 2's "
        "`StackingEnsemble` uses genuinely engine-grouped nested-CV OOF predictions throughout, "
        "and its base learners are also mode-selected from the same nested-CV process, not "
        "independently tuned. Phase 1's ensemble selection (`FixedWeighted_XGBoost70_MLP30` "
        "and, implicitly, which 5 models fed `Stacking_Ridge`) was itself chosen by looking "
        "at official test-set R2 (`results/tables/ensembling_config.json`'s "
        "`\"selection_metric\": \"official test-set R2\"`) -- the exact test-set-contamination "
        "problem Phase 2 exists to fix. Phase 2's freeze decision was made with zero test-set "
        "influence.\n"
    )

    if p1_lgbm is not None:
        lines.append("### LightGBM: Phase 1 (official split) vs Phase 2 (frozen, official test set)\n")
        lines.append("| | R2 | MSE | MAE | PHM08 |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| Phase 1 | {p1_lgbm['R2']:.4f} | {p1_lgbm['MSE']:.2f} | "
                      f"{p1_lgbm['MAE']:.2f} | {p1_lgbm['PHM08_RUL_Score']:.1f} |")
        lines.append(f"| Phase 2 (frozen) | {lgbm['R2']:.4f} | {lgbm['MSE']:.2f} | "
                      f"{lgbm['MAE']:.2f} | {lgbm['PHM08_RUL_Score']:.1f} |")
        d_r2 = _delta(lgbm['R2'], p1_lgbm['R2'])
        d_mse = _delta(lgbm['MSE'], p1_lgbm['MSE'])
        lines.append(f"\nDelta (Phase 2 - Phase 1): R2 {d_r2:+.4f}, MSE {d_mse:+.2f}. ")
    else:
        lines.append("Phase 1 LightGBM row not found in "
                      "`results/tables/official_split_metrics.csv` -- comparison skipped.\n")

    if p1_stack is not None:
        lines.append("\n### Stacking: Phase 1 `Stacking_Ridge` vs Phase 2 `StackingEnsemble` "
                      "(frozen, official test set)\n")
        lines.append("Not the same base-learner-tuning or OOF-generation methodology (see caveat "
                      "above) -- reported side by side for transparency, not as an apples-to-apples "
                      "ablation.\n")
        lines.append("| | R2 | MSE | MAE | PHM08 |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| Phase 1 `Stacking_Ridge` | {p1_stack['R2']:.4f} | {p1_stack['MSE']:.2f} | "
                      f"{p1_stack['MAE']:.2f} | {p1_stack['PHM08_RUL_Score']:.1f} |")
        lines.append(f"| Phase 2 `StackingEnsemble` (frozen) | {stack['R2']:.4f} | {stack['MSE']:.2f} | "
                      f"{stack['MAE']:.2f} | {stack['PHM08_RUL_Score']:.1f} |")
        d_r2 = _delta(stack['R2'], p1_stack['R2'])
        d_mse = _delta(stack['MSE'], p1_stack['MSE'])
        lines.append(f"\nDelta (Phase 2 - Phase 1): R2 {d_r2:+.4f}, MSE {d_mse:+.2f}. ")
    else:
        lines.append("\nPhase 1 `Stacking_Ridge` row not found in "
                      "`results/tables/official_split_metrics.csv` -- comparison skipped.\n")

    lines.append("\n## Did the leakage-safe frozen selection process change the final numbers "
                 "meaningfully from Phase 1's test-set-selected ones?\n")
    if p1_lgbm is not None and p1_stack is not None:
        lgbm_close = abs(lgbm['R2'] - p1_lgbm['R2']) < 0.01
        stack_close = abs(stack['R2'] - p1_stack['R2']) < 0.01
        lines.append(
            f"LightGBM R2 moved by {lgbm['R2'] - p1_lgbm['R2']:+.4f} and the stacking R2 moved by "
            f"{stack['R2'] - p1_stack['R2']:+.4f} relative to Phase 1's numbers for the "
            f"(non-identical) closest-analogue architectures. "
            + ("Both are within a small margin of Phase 1's numbers, " if (lgbm_close and stack_close)
               else "At least one moved by more than a small margin, ")
            + "which is the expected shape of this result: this dataset's official split places "
            "cycles from every engine's early life in train and only each engine's LAST cycles in "
            "test, so it is far less prone to the catastrophic collapse the PGTS/leakage-red-team "
            "re-evaluation demonstrates for genuinely leaky protocols (e.g. Track A's row-level, "
            "engine_id-as-feature reproduction of the paper's own headline number). The methodology "
            "changes here (genuinely grouped nested CV instead of plain KFold, hyperparameters not "
            "picked by looking at the test set) mainly change WHICH hyperparameters/ensemble members "
            "get used, not whether the official-split evaluation itself is leaky -- so a large jump "
            "was never expected here, unlike the PGTS re-evaluation elsewhere in this project. See "
            "`results/tables/pgts_comparison.csv` and `results/LEAKAGE_REPORT.md` for the split-level "
            "(not selection-level) leakage story."
        )
    else:
        lines.append("Comparison incomplete -- see missing-row notes above.")
    lines.append("")

    lines.append("## Source paper citation\n")
    lines.append(
        "Özcan, H. \"Interpretable ensemble remaining useful life prediction enables dynamic "
        "maintenance scheduling for aircraft engines.\" *Scientific Reports* 15, 39795 (2025). "
        "https://doi.org/10.1038/s41598-025-23473-2 -- this project reproduces and stress-tests "
        "that paper's methodology; the numbers above are this project's own frozen, leakage-safe "
        "re-evaluation, not the paper's own reported numbers (see `results/REPRODUCTION_REPORT.md` "
        "/ `README.md` for the direct paper comparison)."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 78)
    print("[final_eval] STEP 0: checking results/tables/_test_set_access_log.json "
          "has zero prior entries")
    print("=" * 78)
    _assert_test_set_not_yet_accessed()
    print("[final_eval] OK -- test.csv has not been read by any prior final_eval.py run.")

    base_fit = fit_all_base_learners(train_path=TRAIN_PATH, models=ALL_MODELS_TO_FIT,
                                      seed=SEED, verbose=True)
    fitted = base_fit["fitted"]
    hp_details = base_fit["hp_details"]
    feature_cols = base_fit["feature_cols"]

    meta_learner, meta_params, _oof_df = fit_meta_learner(
        oof_path=NESTED_CV_OOF_PATH, base_learners=STACK_BASE_LEARNERS, seed=SEED, verbose=True,
    )

    # ---- STEP 4: the one and only test.csv read for this entire run ----
    results = evaluate_on_test_set(fitted, meta_learner, feature_cols,
                                    test_path=TEST_PATH, verbose=True)

    print("\n" + "=" * 78)
    print("[final_eval] STEP 5-6: writing outputs")
    print("=" * 78)
    final_df = write_final_test_metrics(results, hp_details, meta_params)
    print(final_df.to_string(index=False))

    access_entry = _write_access_log_entry(
        models_evaluated=["LightGBM", "StackingEnsemble"], script_name="final_eval.py",
    )
    print(f"\n[final_eval] wrote {ACCESS_LOG_PATH}: {access_entry}")

    write_final_report(results, hp_details, meta_params, access_entry)
    print(f"[final_eval] wrote {FINAL_REPORT_PATH}")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"[final_eval] DONE in {total_elapsed/60:.2f} min. "
          f"data/processed/test.csv was read exactly once (STEP 4).")
    print("=" * 78)


if __name__ == "__main__":
    main()
