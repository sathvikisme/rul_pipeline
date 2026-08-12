"""
models.py — RUL-Bench model-trainer subagent

Trains and tunes 10 individual regression models on the processed C-MAPSS
FD001 data (data/processed/train.csv, data/processed/test.csv — produced by
the data-engineer subagent's src/preprocessing.py + src/features.py).

Models: LightGBM, CatBoost, Gradient Boosting (sklearn), XGBoost, SVM (SVR),
KNN, Linear Regression, Ridge, Bayesian Ridge, MLP.

Pipeline per model:
  1. GridSearchCV on the TRAINING set only, cv=KFold(5, shuffle=True,
     random_state=42), scoring='neg_mean_squared_error' (see SCORING_METRIC
     below for why this was chosen over 'r2').
  2. With the selected best hyperparameters, compute a *plain* 5-fold
     cross_val_score (both R2 and MSE) using the SAME KFold(5, shuffle=True,
     random_state=42) split, and save the per-fold arrays — this is the
     input stats-auditor needs for Shapiro-Wilk/Levene/ANOVA/Tukey HSD, not
     just a mean.
  3. Fit best-params model on the FULL training set (GridSearchCV's
     refit=True already does this — grid.best_estimator_ IS the full-train
     fit, so it is reused directly rather than re-fit a second time).
  4. Evaluate ONCE on the official held-out test.csv: R2, MSE, MAE (sklearn)
     and PHM08 RUL Score (imported from .claude/skills/phm08-scoring/score.py
     — never re-implemented here, per CLAUDE.md skills rule).
  5. Save the fitted model (joblib) to results/models/, log grids + best
     params, and append the test metrics to
     results/tables/official_split_metrics.csv.

Feature set decision (documented per PROJECT_BRIEF.md instruction): `cycle`
IS used as a model feature. Rationale: at prediction time in any real
deployment, the current cycle count for an engine is always known (it's
just "how many flights/cycles has this engine done so far") — it carries no
information from the future and is not derived from the per-engine
lifetime (max_cycle), which is exactly the unknown quantity we are trying
to predict. Excluding it would discard a genuinely predictive, leakage-free
signal, and Heimes (2008) and related C-MAPSS RUL literature commonly
include it. `engine_id` is excluded — it's a bookkeeping identifier, not a
physical signal, and including it would let models memorize per-engine
mappings that don't generalize.

Random seed: random_state=42 fixed everywhere a model/GridSearchCV/KFold
accepts one (documented per-model below for the handful that have no
meaningful source of randomness, e.g. plain OLS LinearRegression, KNN,
BayesianRidge with default solver).

Run: `python src/models.py` from repo root (or via -m from src/).
"""

from __future__ import annotations

import importlib.util
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Repo-relative paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(REPO_ROOT, "data", "processed", "test.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "results", "models")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")
PHM08_SCRIPT = os.path.join(REPO_ROOT, ".claude", "skills", "phm08-scoring", "score.py")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Import phm08_score from the skill script without reimplementing it, per
# CLAUDE.md ("Skills — load automatically, don't reimplement").
_spec = importlib.util.spec_from_file_location("phm08_score_module", PHM08_SCRIPT)
_phm08_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phm08_mod)
phm08_score = _phm08_mod.phm08_score

# ---------------------------------------------------------------------------
# Global constants — seeds and CV protocol
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
KFOLD = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# GridSearchCV scoring metric. Choosing 'neg_mean_squared_error' (sklearn
# requires "higher is better" scorers, hence the negation) rather than 'r2':
# MSE is the metric the downstream PHM08 RUL Score and MAE most directly
# track (squared-error minimization matches how these regressors are
# actually fit), and it keeps the grid-search objective on the same footing
# as the final official-split evaluation metrics. Documented here, not a
# silent default — 'r2' would have been an equally defensible choice, and a
# different SCORING_METRIC would generally select different best_params_
# for models with a nontrivial hyperparameter tradeoff (e.g. GBM tree
# depth).
SCORING_METRIC = "neg_mean_squared_error"

N_JOBS = -1  # outer GridSearchCV/cross_val_score parallelism (8 CPUs available)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    feature_cols = [c for c in train_df.columns if c not in ("engine_id", "RUL")]
    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["RUL"].values
    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------------------------
# Model registry: name -> (base_estimator, param_grid)
#
# Grids are deliberately small (2-4 values per hyperparameter) so the full
# suite runs in low minutes on CPU, per PROJECT_BRIEF.md. random_state=42 is
# set on every estimator that accepts one; estimators with no stochastic
# fitting procedure (LinearRegression, Ridge w/ default 'svd'/'cholesky'
# solver, BayesianRidge, KNN, SVR w/ default solver) do not take a seed —
# noted inline.
# ---------------------------------------------------------------------------

def build_model_registry():
    registry = {}

    registry["LightGBM"] = (
        LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        },
    )

    registry["CatBoost"] = (
        CatBoostRegressor(random_state=SEED, thread_count=1, verbose=False, allow_writing_files=False),
        {
            "iterations": [100, 200],
            "depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        },
    )

    registry["GradientBoosting"] = (
        GradientBoostingRegressor(random_state=SEED),
        {
            "n_estimators": [100, 200],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.05, 0.1],
        },
    )

    registry["XGBoost"] = (
        XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        },
    )

    # SVR (RBF kernel): libsvm's solver is a deterministic QP solve given
    # fixed data/params — no random_state parameter exists on SVR because
    # there is no stochasticity to seed.
    # NOTE: gamma='auto' (1/n_features, ignoring feature variance) was
    # empirically found to make libsvm's QP solve pathologically slow on
    # this data (a timed exploratory fit exceeded 5 minutes and was killed
    # — see training log), so it is excluded from the grid; gamma='scale'
    # (1/(n_features * X.var())) is the sklearn default and fits in ~13-15s
    # regardless of C in the same exploratory timing. max_iter is capped so
    # any other unexpectedly-slow combination fails fast instead of hanging
    # the whole grid search.
    registry["SVM"] = (
        SVR(kernel="rbf", gamma="scale", max_iter=20000),
        {
            "C": [1, 10, 100],
            "epsilon": [0.1, 0.5],
        },
    )

    # KNN: purely deterministic given the data (brute/kd-tree neighbor
    # lookup) — no random_state parameter.
    registry["KNN"] = (
        KNeighborsRegressor(),
        {
            "n_neighbors": [5, 10, 15, 20],
            "weights": ["uniform", "distance"],
        },
    )

    # Plain OLS has no regularization hyperparameter to tune; GridSearchCV
    # is run anyway (per the brief's model list) over the only two knobs
    # LinearRegression exposes. No random_state — closed-form solve.
    registry["LinearRegression"] = (
        LinearRegression(),
        {
            "fit_intercept": [True, False],
            "positive": [False, True],
        },
    )

    # Ridge: default solver ('auto' -> typically 'cholesky'/'svd' for dense
    # data) is deterministic, so random_state is accepted but only matters
    # for the sag/saga/lsqr-with-randomization solvers — set anyway for
    # reproducibility if GridSearchCV ever picks a stochastic solver.
    registry["Ridge"] = (
        Ridge(random_state=SEED),
        {
            "alpha": [0.1, 1.0, 10.0, 100.0],
            "solver": ["auto"],
        },
    )

    # BayesianRidge: deterministic iterative fit (no random_state param).
    registry["BayesianRidge"] = (
        BayesianRidge(),
        {
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4],
        },
    )

    registry["MLP"] = (
        MLPRegressor(random_state=SEED, max_iter=500, early_stopping=True, n_iter_no_change=15),
        {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001, 0.01],
        },
    )

    return registry


def train_and_evaluate_all(X_train, y_train, X_test, y_test, feature_cols):
    registry = build_model_registry()

    r2_fold_records = {}
    mse_fold_records = {}
    official_metrics_rows = []
    best_hyperparams_rows = []
    fitted_models = {}
    test_predictions = {"y_true": y_test}

    for name, (base_estimator, param_grid) in registry.items():
        print("\n" + "=" * 70)
        print(f"[model-trainer] {name}: GridSearchCV starting (scoring={SCORING_METRIC})")
        print(f"[model-trainer] {name}: param_grid = {param_grid}")
        t0 = time.time()

        grid = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=KFOLD,
            scoring=SCORING_METRIC,
            n_jobs=N_JOBS,
            refit=True,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"[model-trainer] {name}: GridSearchCV done in {elapsed:.1f}s")
        print(f"[model-trainer] {name}: best_params_ = {grid.best_params_}")
        print(f"[model-trainer] {name}: best_score_ ({SCORING_METRIC}) = {grid.best_score_:.4f}")

        best_estimator = grid.best_estimator_  # already refit on FULL training set

        # --- Plain 5-fold cross_val_score with best-params estimator ---
        # cross_val_score clones the passed estimator internally for every
        # fold (it does not reuse the already-fitted state), so this is a
        # genuine fresh 5-fold CV with the selected hyperparameters, using
        # the identical KFold(5, shuffle=True, random_state=42) split.
        r2_scores = cross_val_score(best_estimator, X_train, y_train, cv=KFOLD, scoring="r2", n_jobs=N_JOBS)
        neg_mse_scores = cross_val_score(best_estimator, X_train, y_train, cv=KFOLD, scoring="neg_mean_squared_error", n_jobs=N_JOBS)
        mse_scores = -neg_mse_scores
        print(f"[model-trainer] {name}: CV R2 per fold  = {np.round(r2_scores, 4).tolist()}")
        print(f"[model-trainer] {name}: CV MSE per fold = {np.round(mse_scores, 4).tolist()}")

        r2_fold_records[name] = r2_scores
        mse_fold_records[name] = mse_scores

        # --- Official-split test evaluation (ONE shot, full-train-fit model) ---
        y_pred_test = best_estimator.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rul_score, _ = phm08_score(y_test, y_pred_test)
        print(f"[model-trainer] {name}: TEST R2={test_r2:.4f} MSE={test_mse:.4f} MAE={test_mae:.4f} PHM08={test_rul_score:.2f}")

        official_metrics_rows.append({
            "model": name,
            "R2": test_r2,
            "MSE": test_mse,
            "MAE": test_mae,
            "PHM08_RUL_Score": test_rul_score,
        })

        best_hyperparams_rows.append({
            "model": name,
            "param_grid": json.dumps(param_grid),
            "best_params": json.dumps(grid.best_params_),
            "grid_scoring_metric": SCORING_METRIC,
            "cv_best_score": grid.best_score_,
            "cv_r2_mean": float(np.mean(r2_scores)),
            "cv_r2_std": float(np.std(r2_scores)),
            "cv_mse_mean": float(np.mean(mse_scores)),
            "cv_mse_std": float(np.std(mse_scores)),
        })

        # --- Save fitted model ---
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(best_estimator, model_path)
        print(f"[model-trainer] {name}: saved fitted model -> {model_path}")

        fitted_models[name] = best_estimator
        test_predictions[name] = y_pred_test

    # --- Write results/tables outputs ---
    r2_df = pd.DataFrame(r2_fold_records)
    r2_df.index = [f"fold_{i+1}" for i in range(N_SPLITS)]
    r2_df.index.name = "fold"
    r2_df.to_csv(os.path.join(TABLES_DIR, "cv_scores_r2.csv"))

    mse_df = pd.DataFrame(mse_fold_records)
    mse_df.index = [f"fold_{i+1}" for i in range(N_SPLITS)]
    mse_df.index.name = "fold"
    mse_df.to_csv(os.path.join(TABLES_DIR, "cv_scores_mse.csv"))

    official_df = pd.DataFrame(official_metrics_rows)
    official_df.to_csv(os.path.join(TABLES_DIR, "official_split_metrics.csv"), index=False)

    best_hp_df = pd.DataFrame(best_hyperparams_rows)
    best_hp_df.to_csv(os.path.join(TABLES_DIR, "best_hyperparams.csv"), index=False)
    with open(os.path.join(TABLES_DIR, "best_hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump(best_hyperparams_rows, f, indent=2)

    return {
        "fitted_models": fitted_models,
        "test_predictions": test_predictions,
        "official_df": official_df,
        "best_hp_df": best_hp_df,
        "r2_df": r2_df,
        "mse_df": mse_df,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    print(f"[model-trainer] Loaded train {X_train.shape}, test {X_test.shape}")
    print(f"[model-trainer] Feature columns ({len(feature_cols)}): {feature_cols}")

    results = train_and_evaluate_all(X_train, y_train, X_test, y_test, feature_cols)

    print("\n" + "=" * 70)
    print("[model-trainer] OFFICIAL-SPLIT TEST METRICS SUMMARY")
    print("=" * 70)
    print(results["official_df"].to_string(index=False))

    # Persist test_predictions (individual models) for ensembling.py to
    # extend with ensemble columns and for downstream subagents to reuse.
    preds_df = pd.DataFrame(results["test_predictions"])
    preds_df.to_csv(os.path.join(TABLES_DIR, "test_predictions_individual.csv"), index=False)
    print(f"\n[model-trainer] Individual model test predictions saved -> "
          f"{os.path.join(TABLES_DIR, 'test_predictions_individual.csv')}")
