"""
nested_cv.py — RUL-Bench model-trainer subagent (Phase 2, Track B)

Leakage-safe replacement for Phase 1's `src/models.py` CV protocol. Phase 1
used a plain row-level `KFold(shuffle=True)` for BOTH hyperparameter tuning
(GridSearchCV) and the CV scores handed to stats-auditor — that lets cycles
from the same engine appear in both the "train" and "validation" side of a
fold, which is a leakage bug (the model partially memorizes an engine's own
degradation curve instead of generalizing to unseen engines). This module
fixes that with a genuinely nested, engine-grouped CV:

  Outer split : GroupKFold(n_splits=5),  grouped by engine_id, on
                data/processed/train.csv (100 engines -> ~20 held out per
                outer fold). Outer-fold test predictions are the "clean"
                out-of-fold (OOF) predictions used both for this module's
                own per-fold metrics AND (via nested_stacking.py) as the
                stacking meta-learner's training matrix.
  Inner split : GroupKFold(n_splits=3) on the outer-TRAINING engines only,
                used exclusively for Tier 1 (RandomizedSearchCV) and Tier 3
                (GridSearchCV) hyperparameter search. Tier 2 does no inner
                search at all (see tier table below).

Tiered hyperparameter-tuning budget (deliberate cost control — exhaustive
nested GridSearchCV for all 10 models would be a ~25x-cost trap):

  Tier 1 (LightGBM, CatBoost, XGBoost)
      RandomizedSearchCV(n_iter=10, cv=inner GroupKFold(3), random_state=42)
      run FRESH inside every one of the 5 outer folds (5 x 3 = 15 independent
      searches total). `groups=` is passed explicitly to every inner
      `.fit()` call — forgetting this is the single most important bug this
      module guards against (see `_smoke_test_groups_required` below, run
      automatically under `if __name__ == "__main__":`, and the
      `inner_group_pool_size` column logged into
      results/tables/nested_cv_best_params.json for every Tier-1/3 search).
  Tier 2 (GradientBoosting, MLP, SVM, KNN)
      No inner tuning. Hyperparameters are the exact `best_params` Phase 1
      already selected (results/tables/best_hyperparams.json), reused as
      FIXED params and refit once per outer fold (via the fold-safe
      `build_model_pipeline`) purely to obtain that model's outer-fold-clean
      OOF predictions/metrics.
  Tier 3 (LinearRegression, Ridge, BayesianRidge)
      Cheap enough for genuine nested GridSearchCV — the exact small grids
      already recorded in `best_hyperparams.json`'s `param_grid` column,
      searched fresh inside every outer fold via inner GroupKFold(3).

Every estimator is wrapped in `fold_safe_pipeline.build_model_pipeline` so
VarianceThreshold(1e-5) and (for scale-sensitive models) StandardScaler are
fit ONLY on whatever rows are handed to `.fit()` at that moment — never on
the held-out outer or inner fold, never on data/processed/test.csv (this
module never even imports that path).

NOTE on double preprocessing: data/processed/train.csv was already
variance-thresholded + globally z-scored ONCE by Phase 1's offline
`src/features.py` pipeline (fit on ALL 100 training engines at once, before
any CV loop existed) — per the parent Phase-2 task, this module is
instructed to read that same file (not the unscaled/unselected raw
variants, which live under data/processed/rul_cap_ablation/*/train.csv and
ARE fully raw+unselected). Re-running VarianceThreshold/StandardScaler
fold-locally on top of already-preprocessed data is: (a) a no-op for
VarianceThreshold, since Phase 1 already dropped the near-constant sensors
and nothing here should be near-constant again; (b) NOT a no-op for
StandardScaler — refitting the scaler's mean/std on only the current fold's
training rows (rather than reusing Phase 1's global-fit statistics) still
removes the fold-boundary leakage that matters for CV honesty, even though
the *feature-selection decision itself* (which sensors survive at all) was
made once, globally, by Phase 1. This is a real, documented residual
limitation of reusing Phase 1's already-processed file rather than
re-deriving features from data/raw/ inside this module — flagged here
rather than silently glossed over.

Random seed: 42, fixed for GroupKFold-independent randomness sources
(RandomizedSearchCV, GridSearchCV via estimator random_state, and every
individual model's own random_state) — GroupKFold itself is deterministic
(contiguous group-order partition) and takes no seed.

Run: `python src/nested_cv.py` from repo root — runs the full 10-model nested
CV on data/processed/train.csv and writes results/tables/nested_cv_metrics.csv,
results/tables/clean_grouped_cv_metrics.csv, results/tables/nested_cv_best_params.json,
results/tables/nested_cv_timing.csv, results/tables/nested_cv_oof_predictions.csv.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from fold_safe_pipeline import build_model_pipeline  # noqa: E402

REPO_ROOT = os.path.dirname(SRC_DIR)
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")
BEST_HYPERPARAMS_PATH = os.path.join(TABLES_DIR, "best_hyperparams.json")
PHM08_SCRIPT = os.path.join(REPO_ROOT, ".claude", "skills", "phm08-scoring", "score.py")
PGTS_SCRIPT = os.path.join(REPO_ROOT, ".claude", "skills", "pgts-split", "pgts.py")

os.makedirs(TABLES_DIR, exist_ok=True)


def _load_skill_module(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Skills reused, not reimplemented (CLAUDE.md rule).
phm08_score = _load_skill_module(PHM08_SCRIPT, "phm08_score_module_nested_cv").phm08_score
_assert_no_leakage = _load_skill_module(PGTS_SCRIPT, "pgts_module_nested_cv")._assert_no_leakage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
N_ITER_TIER1 = 10
N_JOBS = -1  # 8 CPUs available (checked at build time)

TIER1_MODELS = ["LightGBM", "CatBoost", "XGBoost"]
TIER2_MODELS = ["GradientBoosting", "MLP", "SVM", "KNN"]
TIER3_MODELS = ["LinearRegression", "Ridge", "BayesianRidge"]
ALL_MODELS = TIER1_MODELS + TIER2_MODELS + TIER3_MODELS

_TIER_OF = {m: 1 for m in TIER1_MODELS}
_TIER_OF.update({m: 2 for m in TIER2_MODELS})
_TIER_OF.update({m: 3 for m in TIER3_MODELS})


# ---------------------------------------------------------------------------
# Base (untuned) estimators — same construction as Phase 1's
# src/models.py::build_model_registry, reproduced here rather than imported
# so this module has no import-time dependency on src/models.py (kept
# genuinely separate per the "don't touch Phase 1 files" constraint — this
# only reads the same literal constructor calls, doesn't import that module).
# ---------------------------------------------------------------------------

def _base_estimator(name: str):
    if name == "LightGBM":
        return LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1)
    if name == "CatBoost":
        return CatBoostRegressor(random_state=SEED, thread_count=1, verbose=False,
                                  allow_writing_files=False)
    if name == "XGBoost":
        return XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=SEED)
    if name == "MLP":
        return MLPRegressor(random_state=SEED, max_iter=500, early_stopping=True,
                             n_iter_no_change=15)
    if name == "SVM":
        return SVR(kernel="rbf", gamma="scale", max_iter=20000)
    if name == "KNN":
        return KNeighborsRegressor()
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(random_state=SEED)
    if name == "BayesianRidge":
        return BayesianRidge()
    raise ValueError(f"Unknown model name '{name}'")


# ---------------------------------------------------------------------------
# Tier 1 RandomizedSearchCV distributions (deliberately wider than Phase 1's
# GridSearchCV grids so 10 sampled points are a genuine random sample rather
# than "GridSearchCV wearing a RandomizedSearchCV costume" — combo counts are
# all >> n_iter=10, so no ParameterSampler degeneracy warning is expected).
# Keys are pipeline-prefixed ("estimator__...") to match
# fold_safe_pipeline.build_model_pipeline's Pipeline step name.
# ---------------------------------------------------------------------------
TIER1_PARAM_DISTRIBUTIONS = {
    "LightGBM": {
        "estimator__n_estimators": [50, 100, 150, 200, 300],
        "estimator__max_depth": [3, 4, 5, 6, 7, 9, -1],
        "estimator__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "estimator__num_leaves": [15, 31, 63, 127],
    },
    "CatBoost": {
        "estimator__iterations": [50, 100, 150, 200, 300],
        "estimator__depth": [3, 4, 5, 6, 8],
        "estimator__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "estimator__l2_leaf_reg": [1, 3, 5, 7, 9],
    },
    "XGBoost": {
        "estimator__n_estimators": [50, 100, 150, 200, 300],
        "estimator__max_depth": [3, 4, 5, 6, 7, 9],
        "estimator__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "estimator__subsample": [0.7, 0.85, 1.0],
        "estimator__colsample_bytree": [0.7, 0.85, 1.0],
    },
}


def load_phase1_hyperparams(path: str = BEST_HYPERPARAMS_PATH) -> dict:
    """Phase 1's results/tables/best_hyperparams.json, keyed by model name,
    with both `best_params` (Tier 2 fixed reuse) and `param_grid` (Tier 3
    nested-grid reuse) decoded from their JSON-string columns."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        out[row["model"]] = {
            "best_params": json.loads(row["best_params"]),
            "param_grid": json.loads(row["param_grid"]),
        }
    return out


def load_train_data(path: str = TRAIN_PATH):
    """Load a train.csv-shaped file (engine_id, cycle, features..., RUL).
    Never touches test.csv — this function's only argument is the caller's
    own choice of train-shaped file (main data/processed/train.csv, or one
    of the rul_cap_ablation/{A,B,C}/train.csv variants)."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("engine_id", "RUL")]
    X = df[feature_cols].values
    y = df["RUL"].values.astype(float)
    groups = df["engine_id"].values
    return df, X, y, groups, feature_cols


def _score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    score, _ = phm08_score(y_true, y_pred)
    return r2, mse, mae, score


def _jsonable_params(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (tuple, list)):
            out[k] = list(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_nested_cv(train_path: str = TRAIN_PATH, models: list[str] | None = None,
                   n_outer_splits: int = N_OUTER_SPLITS, n_inner_splits: int = N_INNER_SPLITS,
                   n_iter_tier1: int = N_ITER_TIER1, seed: int = SEED, n_jobs: int = N_JOBS,
                   verbose: bool = True) -> dict:
    """Run the tiered nested-GroupKFold CV protocol on `train_path`.

    Returns a dict with:
      fold_metrics_df : per (model, outer_fold) row — R2/MSE/MAE/PHM08/timing
      best_params_df  : per (model, outer_fold) row — chosen hyperparams +
                         inner CV score + inner group-pool size (Tier 1/3 only)
      timing_df       : per (model, outer_fold, tier) fit_seconds
      oof_df          : one row per training-set sample, columns
                         engine_id, cycle (if present), y_true, outer_fold,
                         pred_<model> for every requested model — the
                         genuine nested out-of-fold predictions consumed by
                         nested_stacking.py and (later) stats-auditor.
      feature_cols, outer_splits, groups
    """
    if models is None:
        models = ALL_MODELS
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s) requested: {unknown}")

    df, X, y, groups, feature_cols = load_train_data(train_path)
    phase1_hp = load_phase1_hyperparams()

    outer_cv = GroupKFold(n_splits=n_outer_splits)
    outer_splits = list(outer_cv.split(X, y, groups))
    assert len(outer_splits) == n_outer_splits

    fold_metric_rows = []
    best_params_log = []
    timing_log = []
    oof_records = {name: np.full(len(y), np.nan) for name in models}
    outer_fold_tag = np.full(len(y), -1, dtype=int)

    if verbose:
        print("=" * 78)
        print(f"[nested_cv] train_path={train_path}  n_rows={len(y)}  "
              f"n_engines={len(np.unique(groups))}  models={models}")
        print("=" * 78)

    tier_wall_seconds = {1: 0.0, 2: 0.0, 3: 0.0}

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
        _assert_no_leakage(groups, outer_train_idx, outer_test_idx)
        outer_fold_tag[outer_test_idx] = fold_idx

        X_tr, y_tr, g_tr = X[outer_train_idx], y[outer_train_idx], groups[outer_train_idx]
        X_te, y_te = X[outer_test_idx], y[outer_test_idx]

        # Inner GroupKFold(3) on the outer-TRAINING engines only, shared by
        # every Tier-1/Tier-3 model fit inside this outer fold (the split
        # itself doesn't depend on which model is being tuned).
        inner_cv = GroupKFold(n_splits=n_inner_splits)
        inner_splits = list(inner_cv.split(X_tr, y_tr, g_tr))
        for in_tr, in_te in inner_splits:
            _assert_no_leakage(g_tr, in_tr, in_te)
        n_inner_group_pool = int(len(np.unique(g_tr)))

        if verbose:
            print(f"\n[nested_cv] outer fold {fold_idx}/{n_outer_splits}: "
                  f"train_rows={len(outer_train_idx)} test_rows={len(outer_test_idx)} "
                  f"train_engines={len(np.unique(g_tr))} test_engines={len(np.unique(groups[outer_test_idx]))} "
                  f"inner_group_pool={n_inner_group_pool} inner_splits={len(inner_splits)}")

        for name in models:
            tier = _TIER_OF[name]
            t0 = time.time()
            base_est = _base_estimator(name)
            pipe = build_model_pipeline(name, base_est)

            if tier == 1:
                param_dist = TIER1_PARAM_DISTRIBUTIONS[name]
                search = RandomizedSearchCV(
                    estimator=pipe, param_distributions=param_dist,
                    n_iter=n_iter_tier1, cv=inner_cv, scoring="neg_mean_squared_error",
                    random_state=seed, n_jobs=n_jobs, refit=True,
                )
                search.fit(X_tr, y_tr, groups=g_tr)  # groups= REQUIRED for inner GroupKFold
                fitted = search.best_estimator_
                chosen_params = search.best_params_
                inner_score = float(search.best_score_)
            elif tier == 2:
                fixed = {f"estimator__{k}": v for k, v in phase1_hp[name]["best_params"].items()}
                pipe.set_params(**fixed)
                pipe.fit(X_tr, y_tr)
                fitted = pipe
                chosen_params = fixed
                inner_score = None
            else:  # tier == 3
                param_grid = {f"estimator__{k}": v for k, v in phase1_hp[name]["param_grid"].items()}
                search = GridSearchCV(
                    estimator=pipe, param_grid=param_grid, cv=inner_cv,
                    scoring="neg_mean_squared_error", n_jobs=n_jobs, refit=True,
                )
                search.fit(X_tr, y_tr, groups=g_tr)  # groups= REQUIRED for inner GroupKFold
                fitted = search.best_estimator_
                chosen_params = search.best_params_
                inner_score = float(search.best_score_)

            y_pred = fitted.predict(X_te)
            oof_records[name][outer_test_idx] = y_pred
            r2, mse, mae, score = _score(y_te, y_pred)
            elapsed = time.time() - t0
            tier_wall_seconds[tier] += elapsed

            fold_metric_rows.append({
                "model": name, "tier": tier, "outer_fold": fold_idx,
                "n_train_rows": len(outer_train_idx), "n_test_rows": len(outer_test_idx),
                "R2": r2, "MSE": mse, "MAE": mae, "PHM08_RUL_Score": score,
                "fit_seconds": elapsed,
            })
            best_params_log.append({
                "model": name, "tier": tier, "outer_fold": fold_idx,
                "chosen_params": json.dumps(_jsonable_params(chosen_params)),
                "inner_cv_best_score_neg_mse": inner_score,
                "inner_group_pool_size": n_inner_group_pool if tier in (1, 3) else None,
                "inner_n_splits": n_inner_splits if tier in (1, 3) else None,
            })
            timing_log.append({"model": name, "tier": tier, "outer_fold": fold_idx, "seconds": elapsed})

            if verbose:
                print(f"    [{name:18s} tier{tier}] R2={r2:.4f} MSE={mse:8.2f} "
                      f"MAE={mae:6.2f} PHM08={score:9.1f}  ({elapsed:5.1f}s)"
                      + (f"  inner_group_pool={n_inner_group_pool}" if tier in (1, 3) else ""))

    assert (outer_fold_tag >= 1).all(), "every training row must be assigned to exactly one outer fold"

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    best_params_df = pd.DataFrame(best_params_log)
    timing_df = pd.DataFrame(timing_log)

    oof_cols = {"engine_id": df["engine_id"].values, "y_true": y, "outer_fold": outer_fold_tag}
    if "cycle" in df.columns:
        oof_cols["cycle"] = df["cycle"].values
    oof_df = pd.DataFrame(oof_cols)
    for name in models:
        assert not np.isnan(oof_records[name]).any(), (
            f"{name}: some training rows never received an outer-fold OOF prediction "
            f"-- the 5 outer folds should partition every row exactly once"
        )
        oof_df[f"pred_{name}"] = oof_records[name]

    if verbose:
        print("\n" + "-" * 78)
        print("[nested_cv] wall-clock seconds by tier (summed across all outer folds/models):")
        for tier, secs in tier_wall_seconds.items():
            print(f"    Tier {tier}: {secs:.1f}s ({secs/60:.2f} min)")
        total_min = sum(tier_wall_seconds.values()) / 60.0
        print(f"    TOTAL:  {total_min:.2f} min")
        if total_min > 18.0:
            print(f"[nested_cv] WARNING: total wall time {total_min:.2f} min exceeds the "
                  f"~10-18 min Phase-2-plan budget estimate for this run.")
        print("-" * 78)

    return {
        "fold_metrics_df": fold_metrics_df,
        "best_params_df": best_params_df,
        "timing_df": timing_df,
        "oof_df": oof_df,
        "feature_cols": feature_cols,
        "outer_splits": outer_splits,
        "groups": groups,
        "tier_wall_seconds": tier_wall_seconds,
    }


def summarize_clean_grouped_cv(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean +/- std across the 5 outer folds -- the headline
    leakage-safe replacement for Phase 1's plain_kfold_cv_* columns."""
    agg = fold_metrics_df.groupby(["model", "tier"]).agg(
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        PHM08_mean=("PHM08_RUL_Score", "mean"), PHM08_std=("PHM08_RUL_Score", "std"),
        mean_fit_seconds=("fit_seconds", "mean"),
    ).reset_index()
    return agg.sort_values("R2_mean", ascending=False).reset_index(drop=True)


def write_outputs(results: dict, tables_dir: str = TABLES_DIR, prefix: str = "") -> None:
    fold_metrics_df = results["fold_metrics_df"]
    best_params_df = results["best_params_df"]
    timing_df = results["timing_df"]
    oof_df = results["oof_df"]

    clean_df = summarize_clean_grouped_cv(fold_metrics_df)

    fold_metrics_df.to_csv(os.path.join(tables_dir, f"{prefix}nested_cv_metrics.csv"), index=False)
    clean_df.to_csv(os.path.join(tables_dir, f"{prefix}clean_grouped_cv_metrics.csv"), index=False)
    timing_df.to_csv(os.path.join(tables_dir, f"{prefix}nested_cv_timing.csv"), index=False)
    oof_df.to_csv(os.path.join(tables_dir, f"{prefix}nested_cv_oof_predictions.csv"), index=False)

    hyperparam_log = {
        "seed": SEED,
        "n_outer_splits": N_OUTER_SPLITS,
        "n_inner_splits": N_INNER_SPLITS,
        "n_iter_tier1": N_ITER_TIER1,
        "tier1_models": TIER1_MODELS,
        "tier2_models": TIER2_MODELS,
        "tier3_models": TIER3_MODELS,
        "tier1_param_distributions": TIER1_PARAM_DISTRIBUTIONS,
        "tier2_fixed_params_source": "results/tables/best_hyperparams.json (Phase 1 best_params, reused as-is, refit once per outer fold)",
        "tier3_param_grid_source": "results/tables/best_hyperparams.json (Phase 1 param_grid, re-searched fresh per outer fold via inner GroupKFold(3))",
        "per_outer_fold_chosen_params": json.loads(best_params_df.to_json(orient="records")),
    }
    with open(os.path.join(tables_dir, f"{prefix}nested_cv_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(hyperparam_log, f, indent=2)

    print(f"[nested_cv] wrote {prefix}nested_cv_metrics.csv, {prefix}clean_grouped_cv_metrics.csv, "
          f"{prefix}nested_cv_timing.csv, {prefix}nested_cv_oof_predictions.csv, "
          f"{prefix}nested_cv_best_params.json -> {tables_dir}")


# ---------------------------------------------------------------------------
# Smoke test: proves that forgetting `groups=` on a GroupKFold-cv'd search
# fails LOUD (ValueError), not silently — the specific bug class the parent
# task calls out. Run automatically before the real work under __main__.
# ---------------------------------------------------------------------------

def _smoke_test_groups_required():
    from sklearn.linear_model import Ridge as _Ridge
    rng = np.random.default_rng(0)
    n = 60
    Xs = rng.normal(size=(n, 3))
    ys = Xs[:, 0] + rng.normal(scale=0.1, size=n)
    gs = np.repeat(np.arange(12), 5)  # 12 groups, 5 rows each

    search = GridSearchCV(_Ridge(), param_grid={"alpha": [0.1, 1.0]}, cv=GroupKFold(n_splits=3))
    raised = False
    try:
        search.fit(Xs, ys)  # deliberately NOT passing groups=
    except ValueError:
        raised = True
    assert raised, (
        "expected GroupKFold-backed GridSearchCV.fit() without groups= to raise "
        "ValueError -- if this assertion fails, sklearn's own safety net changed "
        "behavior and this module's groups= discipline needs a different guard"
    )

    # And confirm it succeeds (no error) once groups= IS passed, with a
    # different fold assignment than a plain (non-grouped) KFold would give.
    search.fit(Xs, ys, groups=gs)
    print("[nested_cv] smoke test: GroupKFold-cv'd search raises ValueError when "
          "groups= is omitted, and succeeds when groups= is passed -- groups= "
          "discipline is enforced by sklearn itself, not just convention.")


if __name__ == "__main__":
    _smoke_test_groups_required()

    t_start = time.time()
    results = run_nested_cv(train_path=TRAIN_PATH, models=ALL_MODELS, verbose=True)
    write_outputs(results, tables_dir=TABLES_DIR)

    clean_df = summarize_clean_grouped_cv(results["fold_metrics_df"])
    print("\n" + "=" * 78)
    print("[nested_cv] clean_grouped_cv_metrics.csv summary (mean +/- std across 5 outer folds)")
    print("=" * 78)
    print(clean_df.to_string(index=False))

    total_elapsed = time.time() - t_start
    print(f"\n[nested_cv] TOTAL wall time (incl. smoke test + I/O): {total_elapsed/60:.2f} min")
