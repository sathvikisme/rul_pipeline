"""
nested_stacking.py — RUL-Bench model-trainer subagent (Phase 2, Track B)

Leakage-safe replacement for Phase 1's `src/ensembling.py` stacking
variant. Phase 1's stacking meta-learner was trained on OOF predictions
generated via `cross_val_predict(cv=KFold(5, shuffle=True, random_state=42))`
— row-level, non-grouped folds. Even though `cross_val_predict` genuinely
avoids in-fold leakage in the "did this row's own label leak into its own
prediction" sense, the underlying base-learner hyperparameters were
THEMSELVES selected under that same leaky row-level KFold in `models.py`,
and cycles from the same engine could sit in both the "training" fold that
fit the base learner and the "held-out" fold that produced that engine's
OOF prediction elsewhere. That is a weaker leakage guarantee than what this
module provides.

This module instead builds the meta-learner's OOF training matrix DIRECTLY
from `nested_cv.py`'s per-outer-fold predictions — every base learner was
tuned (Tier 1: RandomizedSearchCV; Tier 3: n/a here) or fit (Tier 2) using
ONLY the 4/5 of engines in that outer fold's training split, and its
prediction for the held-out 1/5 of engines is a genuine, engine-disjoint
out-of-fold prediction. Reusing that matrix here means **zero additional
base-model fits** are needed for stacking — this module only ever fits (and
nested-CV-evaluates) the cheap Ridge meta-learner.

Base learners (same 5-model subset as Phase 1's stacking ensemble, and same
rationale — top performance tier by CV R2, documented in
results/tables/ensembling_config.json): LightGBM, CatBoost, XGBoost,
GradientBoosting, MLP.

Two things this module produces:

1. `run_nested_stacking(oof_df, ...)` — a genuinely nested evaluation of the
   STACKING ENSEMBLE ITSELF (not just of the base learners it's built from).
   The base-learner OOF predictions are already leakage-free per-row, but
   fitting Ridge on ALL of them and then scoring on the SAME rows would
   still be an in-sample evaluation of the ridge stage. So this function
   reuses `oof_df`'s existing `outer_fold` tag (the same 5-way engine
   partition `nested_cv.py` used) as its OWN outer loop: for each outer fold
   k, the ridge meta-learner (with alpha tuned via a further inner
   GroupKFold(3) on the *other* 4 outer folds' pooled OOF rows) is fit on
   OOF rows from folds != k and evaluated on fold k's OOF rows. This yields
   5 genuinely held-out ensemble-level metrics, at the cost of only cheap
   Ridge refits (never re-touches the base learners).
2. `fit_final_meta_learner(oof_df, ...)` — one Ridge, alpha tuned by
   GroupKFold(5) on the FULL pooled OOF matrix, fit on all of it. This is a
   deployable artifact (for whichever subagent needs a single frozen
   meta-learner later), not itself a performance claim — the performance
   claim is (1) above.

Random seed: 42, fixed for Ridge and for GridSearchCV's own (unused, since
Ridge/GroupKFold have no stochastic component) determinism.

Run: `python src/nested_stacking.py` from repo root. This re-runs
`nested_cv.run_nested_cv` itself (needed to get `oof_df` if not already in
memory) unless imported and called with an existing `oof_df` from a prior
`nested_cv` run (e.g. from `track_b_pipeline.py`, which passes the OOF
frame through directly and therefore performs zero extra base-model fits).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import nested_cv as ncv  # noqa: E402 -- reuse REPO_ROOT/TABLES_DIR/_assert_no_leakage/_score/SEED

REPO_ROOT = ncv.REPO_ROOT
TABLES_DIR = ncv.TABLES_DIR
_assert_no_leakage = ncv._assert_no_leakage
_score = ncv._score
SEED = ncv.SEED

STACK_BASE_LEARNERS = ["LightGBM", "CatBoost", "XGBoost", "GradientBoosting", "MLP"]
META_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
N_INNER_SPLITS = 3


def _pred_cols(base_learners: list[str]) -> list[str]:
    return [f"pred_{m}" for m in base_learners]


def run_nested_stacking(oof_df: pd.DataFrame, base_learners: list[str] = STACK_BASE_LEARNERS,
                         alpha_grid: list[float] = META_ALPHA_GRID,
                         n_inner_splits: int = N_INNER_SPLITS, seed: int = SEED,
                         verbose: bool = True) -> dict:
    """Nested-CV evaluation of the Ridge stacking ensemble, reusing
    nested_cv.py's outer-fold OOF predictions as the ensemble's own inputs.

    Returns dict with `fold_metrics_df` (per outer fold R2/MSE/MAE/PHM08),
    `alpha_log_df` (chosen alpha + inner CV score per outer fold), and
    `oof_pred` (the stacking ensemble's own OOF predictions, row-aligned
    with `oof_df`, for stats-auditor's paired bootstrap).
    """
    pred_cols = _pred_cols(base_learners)
    missing = [c for c in pred_cols + ["outer_fold", "y_true", "engine_id"] if c not in oof_df.columns]
    if missing:
        raise ValueError(f"oof_df missing required columns: {missing}")

    outer_folds = sorted(oof_df["outer_fold"].unique())
    fold_rows = []
    alpha_log = []
    stack_oof_pred = np.full(len(oof_df), np.nan)

    if verbose:
        print("=" * 78)
        print(f"[nested_stacking] base_learners={base_learners}  "
              f"n_outer_folds={len(outer_folds)}  alpha_grid={alpha_grid}")
        print("=" * 78)

    for k in outer_folds:
        t0 = time.time()
        test_mask = (oof_df["outer_fold"] == k).values
        train_mask = ~test_mask

        X_tr = oof_df.loc[train_mask, pred_cols].values
        y_tr = oof_df.loc[train_mask, "y_true"].values
        g_tr = oof_df.loc[train_mask, "engine_id"].values
        X_te = oof_df.loc[test_mask, pred_cols].values
        y_te = oof_df.loc[test_mask, "y_true"].values

        inner_cv = GroupKFold(n_splits=n_inner_splits)
        for in_tr, in_te in inner_cv.split(X_tr, y_tr, g_tr):
            _assert_no_leakage(g_tr, in_tr, in_te)

        grid = GridSearchCV(
            estimator=Ridge(random_state=seed), param_grid={"alpha": alpha_grid},
            cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1, refit=True,
        )
        grid.fit(X_tr, y_tr, groups=g_tr)  # groups= REQUIRED for inner GroupKFold
        meta = grid.best_estimator_

        y_pred = meta.predict(X_te)
        stack_oof_pred[test_mask] = y_pred
        r2, mse, mae, score = _score(y_te, y_pred)
        elapsed = time.time() - t0

        fold_rows.append({
            "outer_fold": int(k), "n_train_rows": int(train_mask.sum()),
            "n_test_rows": int(test_mask.sum()), "R2": r2, "MSE": mse, "MAE": mae,
            "PHM08_RUL_Score": score, "fit_seconds": elapsed,
        })
        alpha_log.append({
            "outer_fold": int(k), "best_alpha": float(grid.best_params_["alpha"]),
            "inner_cv_best_score_neg_mse": float(grid.best_score_),
        })
        if verbose:
            print(f"    [Stacking_Ridge] outer fold {k}: alpha={grid.best_params_['alpha']:g} "
                  f"R2={r2:.4f} MSE={mse:8.2f} MAE={mae:6.2f} PHM08={score:9.1f} ({elapsed:.2f}s)")

    assert not np.isnan(stack_oof_pred).any(), "every OOF row should receive a stacking prediction"

    fold_metrics_df = pd.DataFrame(fold_rows)
    alpha_log_df = pd.DataFrame(alpha_log)

    if verbose:
        print("-" * 78)
        print(f"[nested_stacking] R2 mean={fold_metrics_df['R2'].mean():.4f} "
              f"std={fold_metrics_df['R2'].std():.4f}  "
              f"MSE mean={fold_metrics_df['MSE'].mean():.2f} std={fold_metrics_df['MSE'].std():.2f}")
        print("-" * 78)

    return {
        "fold_metrics_df": fold_metrics_df,
        "alpha_log_df": alpha_log_df,
        "oof_pred": stack_oof_pred,
        "base_learners": base_learners,
    }


def fit_final_meta_learner(oof_df: pd.DataFrame, base_learners: list[str] = STACK_BASE_LEARNERS,
                            alpha_grid: list[float] = META_ALPHA_GRID, n_splits: int = 5,
                            seed: int = SEED) -> tuple[Ridge, dict]:
    """One Ridge meta-learner, alpha tuned via GroupKFold(5) on the FULL
    pooled OOF matrix, fit on all of it -- a deployable artifact, not a
    performance claim (see run_nested_stacking for the honest nested-CV
    metrics)."""
    pred_cols = _pred_cols(base_learners)
    X_meta = oof_df[pred_cols].values
    y_meta = oof_df["y_true"].values
    g_meta = oof_df["engine_id"].values

    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(X_meta, y_meta, g_meta):
        _assert_no_leakage(g_meta, tr, te)

    grid = GridSearchCV(
        estimator=Ridge(random_state=seed), param_grid={"alpha": alpha_grid},
        cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, refit=True,
    )
    grid.fit(X_meta, y_meta, groups=g_meta)
    return grid.best_estimator_, grid.best_params_


def write_outputs(stacking_results: dict, final_meta_params: dict, tables_dir: str = TABLES_DIR) -> None:
    fold_metrics_df = stacking_results["fold_metrics_df"]
    alpha_log_df = stacking_results["alpha_log_df"]

    fold_metrics_df.to_csv(os.path.join(tables_dir, "nested_stacking_metrics.csv"), index=False)
    alpha_log_df.to_csv(os.path.join(tables_dir, "nested_stacking_alpha_log.csv"), index=False)

    config = {
        "base_learners": stacking_results["base_learners"],
        "meta_learner": "sklearn.linear_model.Ridge",
        "meta_learner_alpha_grid": META_ALPHA_GRID,
        "meta_learner_training_data": (
            "Genuine nested out-of-fold predictions of the 5 base learners from "
            "nested_cv.py's outer GroupKFold(5) -- NOT cross_val_predict (Phase 1's "
            "approach), and NOT raw features. For each outer fold k, the ridge "
            "meta-learner is trained ONLY on OOF rows from the other 4 outer folds "
            "(with its own alpha tuned by a further inner GroupKFold(3) on those "
            "rows) and evaluated on fold k's OOF rows -- a genuinely nested "
            "evaluation of the stacking ensemble itself, not just of the base "
            "learners feeding it."
        ),
        "zero_additional_base_model_fits": True,
        "final_deployable_meta_learner_alpha": final_meta_params,
        "final_deployable_meta_learner_note": (
            "Fit on the FULL pooled OOF matrix (alpha tuned via GroupKFold(5) on "
            "that same matrix) -- an artifact for downstream use, not itself the "
            "nested-CV performance claim (see nested_stacking_metrics.csv for that)."
        ),
        "random_seed": SEED,
    }
    with open(os.path.join(tables_dir, "nested_stacking_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[nested_stacking] wrote nested_stacking_metrics.csv, nested_stacking_alpha_log.csv, "
          f"nested_stacking_config.json -> {tables_dir}")


if __name__ == "__main__":
    print("[nested_stacking] no oof_df passed in -- running nested_cv.run_nested_cv() first "
          "to obtain one (use track_b_pipeline.py to avoid this recompute if you already have "
          "a nested_cv result in memory).")
    ncv_results = ncv.run_nested_cv(train_path=ncv.TRAIN_PATH, models=ncv.ALL_MODELS, verbose=True)
    ncv.write_outputs(ncv_results, tables_dir=TABLES_DIR)

    stacking_results = run_nested_stacking(ncv_results["oof_df"], verbose=True)
    final_meta, final_meta_params = fit_final_meta_learner(ncv_results["oof_df"])
    write_outputs(stacking_results, final_meta_params, tables_dir=TABLES_DIR)

    print("\n[nested_stacking] final deployable meta-learner best alpha:", final_meta_params)
