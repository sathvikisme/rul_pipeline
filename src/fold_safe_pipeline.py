"""
fold_safe_pipeline.py — RUL-Bench data-engineer subagent (Phase 2)

Provides `build_model_pipeline(name, estimator)`, a factory that wraps a
raw sklearn/lightgbm/catboost/xgboost estimator in a
`sklearn.pipeline.Pipeline` of:

    VarianceThreshold(threshold=1e-5)  -->  [StandardScaler]  -->  estimator

Why this exists (Phase 2 methodological repair): Phase 1's variance-
threshold feature selection (`src/features.py::variance_threshold_report`)
was computed ONCE on the full official train.csv and baked into
data/processed/train.csv / test.csv before any cross-validation ever ran.
Inside a proper nested/grouped CV loop, that is itself a (mild) leakage
source — the "which sensors are near-constant" decision saw rows that later
act as validation data in some other fold. The fix is to make variance
thresholding, like scaling, a `Pipeline` step that gets `.fit()` (or
`.fit_transform()`) called only on whatever rows are handed to the pipeline
at fit time. When `nested_cv.py` calls `pipeline.fit(X_fold_train, y_fold_train)`
inside an outer (or inner) CV fold, `VarianceThreshold` only ever sees that
fold's training rows — never the held-out fold, never the official test set.
This module does not import or touch data/processed/train.csv or test.csv;
it only builds pipeline objects for a caller (nested_cv.py) to fit on
whatever fold-specific arrays it passes in.

WHY VarianceThreshold MUST come BEFORE StandardScaler in the pipeline order
(not after, not as a separate offline step applied to already-scaled data):
StandardScaler transforms every surviving column to zero mean, unit
variance (variance == 1.0) by construction. If you scaled first and then
ran VarianceThreshold with a small threshold like 1e-5, every column would
have variance ~1.0 post-scaling and nothing would ever be dropped —
the near-constant-sensor signal that VarianceThreshold is supposed to catch
only exists in the RAW (pre-scaling) feature distribution. So the pipeline
order encodes a real methodological requirement, not just an arbitrary
step ordering:

    raw features -> VarianceThreshold (sees real, informative variance)
                 -> StandardScaler (only for scale-sensitive models)
                 -> estimator

Model-to-scaling-step decision table (per PROJECT_BRIEF.md's model list):

    Tree-based (no scaling — scale-invariant, splits on raw thresholds):
        LightGBM, CatBoost, XGBoost, GradientBoosting
    Scale-sensitive (StandardScaler inserted after VarianceThreshold):
        SVM, KNN, LinearRegression, Ridge, BayesianRidge, MLP

This module has no random/stochastic steps of its own (VarianceThreshold
and StandardScaler are both deterministic given input data), so there is no
seed to fix here; any seed requirement belongs to the estimator the caller
passes in.
"""

from __future__ import annotations

import re

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same threshold value Phase 1 used (src/features.py::VARIANCE_THRESHOLD),
# reused here deliberately for continuity — the actual per-sensor variance
# numbers that justified it are in data/processed/DATA_DICTIONARY.md
# (computed on train-only, ~3-orders-of-magnitude natural gap between
# near-constant sensors and the rest). This module re-applies the SAME
# threshold value fold-by-fold rather than reusing Phase 1's fixed dropped-
# sensor list, so the selection itself is never fit on data outside the
# current fold.
VARIANCE_THRESHOLD = 1e-5

# Canonical model-name registry. Keys are normalized (lowercased,
# non-alphanumeric stripped) so callers can pass "Linear Regression",
# "linear_regression", "LinearRegression", etc. and get the same answer.
TREE_BASED_MODELS = {"lightgbm", "catboost", "xgboost", "gradientboosting"}
SCALE_SENSITIVE_MODELS = {"svm", "knn", "linearregression", "ridge", "bayesianridge", "mlp"}
KNOWN_MODELS = TREE_BASED_MODELS | SCALE_SENSITIVE_MODELS


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_model_pipeline(name: str, estimator, needs_scaling: bool | None = None) -> Pipeline:
    """Build a fold-safe sklearn Pipeline for one model.

    Parameters
    ----------
    name : str
        Model name, matched case-/separator-insensitively against the
        registry above (e.g. "LightGBM", "Linear Regression", "svm" all
        resolve correctly). Used only to decide whether to insert a
        StandardScaler step — it is not otherwise stored or validated
        against the estimator's actual type.
    estimator : sklearn-compatible regressor
        Any object implementing .fit()/.predict() (sklearn, lightgbm,
        catboost, xgboost estimators all qualify). Not modified — passed
        through as the final pipeline step.
    needs_scaling : bool or None, default None
        Explicit override. If None (default), looked up from `name` via
        the TREE_BASED_MODELS / SCALE_SENSITIVE_MODELS registry above, and
        a ValueError is raised if `name` isn't recognized (fail loud rather
        than silently guessing for an unknown model). Pass True/False
        explicitly to bypass the registry for a model not yet listed here.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps: ("variance_threshold", VarianceThreshold(threshold=1e-5))
        [, ("scaler", StandardScaler())] , ("estimator", estimator)

        Fit this pipeline with pipeline.fit(X_fold_train, y_fold_train)
        inside each CV fold — NEVER pre-fit VarianceThreshold/StandardScaler
        on the full dataset and pass already-transformed arrays in, or the
        entire point of this module (fold-safety) is defeated.
    """
    key = _normalize(name)
    if needs_scaling is None:
        if key in SCALE_SENSITIVE_MODELS:
            needs_scaling = True
        elif key in TREE_BASED_MODELS:
            needs_scaling = False
        else:
            raise ValueError(
                f"Unrecognized model name '{name}' (normalized: '{key}'). "
                f"Known models: {sorted(KNOWN_MODELS)}. Pass needs_scaling=True/False "
                f"explicitly to use build_model_pipeline with a model not in this registry."
            )

    steps = [("variance_threshold", VarianceThreshold(threshold=VARIANCE_THRESHOLD))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", estimator))
    return Pipeline(steps)


def model_scaling_table() -> list[dict]:
    """Return the full model -> needs_scaling decision table as data
    (used by the __main__ sanity check below and available for any caller
    that wants to print/report it, e.g. nested_cv.py's run log)."""
    rows = []
    for m in sorted(TREE_BASED_MODELS):
        rows.append({"model": m, "needs_scaling": False, "category": "tree-based"})
    for m in sorted(SCALE_SENSITIVE_MODELS):
        rows.append({"model": m, "needs_scaling": True, "category": "scale-sensitive"})
    return rows


if __name__ == "__main__":
    # Sanity check, same convention as the project's .claude/skills/*.py
    # scripts: synthetic data with one near-constant column and one
    # genuinely-varying column, confirm (a) VarianceThreshold drops the
    # near-constant column, (b) scaling is inserted/omitted per the
    # registry, and (c) the pipeline actually fits and predicts.
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(42)
    n = 200
    X = np.column_stack([
        rng.normal(loc=50.0, scale=10.0, size=n),   # informative
        np.full(n, 3.14159) + rng.normal(scale=1e-8, size=n),  # near-constant
        rng.normal(loc=0.0, scale=1.0, size=n),      # informative
    ])
    y = 2.0 * X[:, 0] - 0.5 * X[:, 2] + rng.normal(scale=0.1, size=n)

    print("=" * 78)
    print("fold_safe_pipeline.py — synthetic sanity check")
    print("=" * 78)
    print(f"Input X shape: {X.shape} (column 1 is near-constant, variance ~1e-16)")
    print(f"Per-column variance: {X.var(axis=0)}")

    ridge_pipe = build_model_pipeline("Ridge", Ridge(random_state=42))
    ridge_pipe.fit(X, y)
    n_kept_ridge = ridge_pipe.named_steps["variance_threshold"].get_support().sum()
    has_scaler = "scaler" in ridge_pipe.named_steps
    assert n_kept_ridge == 2, f"expected 2 surviving columns, got {n_kept_ridge}"
    assert has_scaler, "Ridge should get a StandardScaler step"
    preds = ridge_pipe.predict(X)
    assert preds.shape == (n,)
    print(f"[Ridge]  columns kept={n_kept_ridge}/3, scaler_present={has_scaler}, "
          f"fit+predict OK")

    gbr_pipe = build_model_pipeline("GradientBoosting", GradientBoostingRegressor(random_state=42))
    gbr_pipe.fit(X, y)
    n_kept_gbr = gbr_pipe.named_steps["variance_threshold"].get_support().sum()
    has_scaler_gbr = "scaler" in gbr_pipe.named_steps
    assert n_kept_gbr == 2, f"expected 2 surviving columns, got {n_kept_gbr}"
    assert not has_scaler_gbr, "GradientBoosting should NOT get a StandardScaler step"
    preds_gbr = gbr_pipe.predict(X)
    assert preds_gbr.shape == (n,)
    print(f"[GradientBoosting] columns kept={n_kept_gbr}/3, scaler_present={has_scaler_gbr}, "
          f"fit+predict OK")

    try:
        build_model_pipeline("some_unknown_model", Ridge())
        raise AssertionError("expected ValueError for unrecognized model name")
    except ValueError as e:
        print(f"[unrecognized-name guard] correctly raised ValueError: {e}")

    print()
    print("Model -> scaling-step decision table:")
    for row in model_scaling_table():
        print(f"  {row['model']:18s} needs_scaling={row['needs_scaling']!s:5s} "
              f"({row['category']})")

    print("-" * 78)
    print("All fold_safe_pipeline.py sanity checks passed.")
    print("=" * 78)
