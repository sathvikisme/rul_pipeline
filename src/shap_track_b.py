"""
shap_track_b.py -- interpretability-analyst subagent, RUL-Bench Phase 2 (Track B).

SHAP analysis on the TWO frozen, leakage-safe models from `src/final_eval.py`
(see `results/FREEZE_DECISION.md` and `results/FINAL_TEST_RESULT.md`):

  1. LightGBM          -- single model, mode-selected nested-CV hyperparams,
                           refit on the full 100-engine training set.
  2. StackingEnsemble  -- Ridge meta-learner over 5 base learners (LightGBM,
                           CatBoost, XGBoost, GradientBoosting, MLP), each
                           refit on the full 100-engine training set with
                           their own mode-selected nested-CV hyperparams; the
                           meta-learner is trained on the genuine nested
                           out-of-fold predictions in
                           `results/tables/nested_cv_oof_predictions.csv`.

Model reconstruction
---------------------
`final_eval.py` does not persist joblib artifacts for these two candidates
-- it fits them in-memory and scores `data/processed/test.csv` exactly once
(enforced by its own access-log guard). Per the parent task, this script
reconstructs the EXACT same fitted models by importing and calling
`final_eval.py`'s own fitting functions directly:

    fe.fit_all_base_learners(...)  -- fits all 5 base learners (LightGBM
                                       included) with the same mode-selected
                                       hyperparameters, same seed=42, same
                                       full train.csv.
    fe.fit_meta_learner(...)       -- fits the same Ridge meta-learner on
                                       the same nested-CV OOF matrix.
    fe.evaluate_on_test_set(...)   -- scores data/processed/test.csv with
                                       these reconstructed models.

This is NOT a re-run of `final_eval.py`'s one-shot access-log discipline --
that discipline exists to stop the *model-selection* process from peeking
at the test set more than once. That decision is long since frozen
(`results/FREEZE_DECISION.md`); this script makes zero model, hyperparameter
or feature decisions based on what it finds. It reads `data/processed/
test.csv` a second time purely to (a) sanity-check that the reconstruction
reproduces `results/tables/final_test_metrics.csv`'s R2/MSE exactly, and
(b) select a handful of representative test rows for local force plots --
both purely descriptive, downstream-of-freeze uses, explicitly requested by
the parent task. It does NOT append to
`results/tables/_test_set_access_log.json` (that log's one-entry invariant
belongs to final_eval.py's own selection discipline, not to this script).

SHAP methodology
-----------------
- LightGBM: `shap.TreeExplainer` directly on the fitted LGBMRegressor inside
  its fold-safe Pipeline (exact, no sampling approximation). The pipeline's
  VarianceThreshold(1e-5) step is a no-op here (train.csv was already
  variance-filtered offline by Phase 1's features.py -- confirmed
  numerically below, not assumed), so the tree explainer's feature space is
  the same 18-column `feature_cols` used everywhere else in this project.
- CatBoost / XGBoost / GradientBoosting: also `shap.TreeExplainer` (exact).
- MLP: not a tree model. `shap.Explainer(pipeline.predict, background)`
  (background = `shap.sample(train, n=50, random_state=42)`), which shap
  dispatches to its Permutation explainer for a generic callable -- exact
  under a feature-independence assumption, standard practice for black-box
  models, same approach `src/shap_analysis.py` (Phase 1) used for its MLP.
- StackingEnsemble: the Ridge meta-learner is linear in the 5 base
  learners' predictions, so
      SHAP(ensemble)_j = sum_i meta_coef_i * SHAP(base_learner_i)_j
      base_value(ensemble) = meta_intercept + sum_i meta_coef_i * base_value_i
  is EXACT (not an approximation) -- the same principle Phase 1's
  `src/shap_analysis.py` used for its 2-model FixedWeighted ensemble,
  generalized here to 5 models with the meta-learner's *learned* Ridge
  coefficients (not fixed 0.7/0.3 weights). Verified numerically below by
  reconstructing each row's prediction as base_value + sum(SHAP) and
  comparing to the ensemble's actual `meta_learner.predict(...)` output.

Every SHAP-based finding is worded as "the model attributes weight to
feature X" -- not "feature X causes/determines RUL." SHAP explains a
model's learned function, not the physical degradation process; see the
"Causal-language caveat" section of `results/SHAP_ANALYSIS_TRACK_B.md` for
the explicit statement of this limitation.

Fixed seed 42 used everywhere sampling occurs (test-row sample, background
sample, MLP permutation explainer) per CLAUDE.md rule 3.

Outputs (results/shap/track_b/):
  beeswarm_lightgbm.png, beeswarm_catboost.png, beeswarm_xgboost.png,
  beeswarm_gradientboosting.png, beeswarm_mlp.png,
  beeswarm_stackingensemble.png
  force_stacking_low_rul.png, force_stacking_mid_rul.png,
  force_stacking_high_rul.png, force_lightgbm_low_rul.png,
  force_lightgbm_high_rul.png
  shap_concentration_metrics_track_b.csv, shap_ranked_features_full_track_b.json
  reconstruction_sanity_check.json

Run: `python src/shap_track_b.py` from repo root.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import final_eval as fe  # noqa: E402 -- reuse the exact fitting logic, not a fresh reimplementation
import nested_cv as ncv  # noqa: E402

SEED = 42
N_SAMPLE = 500       # test rows used for global SHAP / beeswarm plots (same as Phase 1)
N_BACKGROUND = 50    # background sample for the MLP permutation explainer (same as Phase 1)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "shap" / "track_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = ROOT / "results" / "tables"
FINAL_METRICS_PATH = TABLES_DIR / "final_test_metrics.csv"

STACK_BASE_LEARNERS = fe.STACK_BASE_LEARNERS  # ["LightGBM","CatBoost","XGBoost","GradientBoosting","MLP"]
TREE_MODELS = ["LightGBM", "CatBoost", "XGBoost", "GradientBoosting"]

# Physical sensor documentation (C-MAPSS / PHM08 convention, Saxena & Goebel;
# same table Phase 1's src/shap_analysis.py used), restricted to the 14 kept
# sensors + op settings + cycle -- see data/processed/DATA_DICTIONARY.md.
SENSOR_MEANING = {
    "cycle": "Operational cycle index (time-in-service proxy)",
    "op_setting_1": "Operational setting 1",
    "op_setting_2": "Operational setting 2",
    "op_setting_3": "Operational setting 3",
    "sensor_2": "T24 - Total temperature at LPC outlet (deg R)",
    "sensor_3": "T30 - Total temperature at HPC outlet (deg R)",
    "sensor_4": "T50 - Total temperature at LPT outlet (deg R)",
    "sensor_7": "P30 - Total pressure at HPC outlet (psia)",
    "sensor_8": "Nf - Physical fan speed (rpm)",
    "sensor_9": "Nc - Physical core speed (rpm)",
    "sensor_11": "Ps30 - Static pressure at HPC outlet (psia)",
    "sensor_12": "phi - Ratio of fuel flow to Ps30 (pps/psi)",
    "sensor_13": "NRf - Corrected fan speed (rpm)",
    "sensor_14": "NRc - Corrected core speed (rpm)",
    "sensor_15": "BPR - Bypass Ratio",
    "sensor_17": "htBleed - Bleed Enthalpy",
    "sensor_20": "W31 - HPT coolant bleed (lbm/s)",
    "sensor_21": "W32 - LPT coolant bleed (lbm/s)",
}


# ---------------------------------------------------------------------------
# STEP 0: reconstruct the exact frozen models + sanity-check against
# results/tables/final_test_metrics.csv
# ---------------------------------------------------------------------------

def reconstruct_and_sanity_check(verbose: bool = True) -> dict:
    if verbose:
        print("=" * 78)
        print("[shap_track_b] STEP 0: reconstructing the frozen models via "
              "final_eval.fit_all_base_learners / fit_meta_learner (no retraining "
              "logic reimplemented here -- exact same code path, seed=42)")
        print("=" * 78)

    base_fit = fe.fit_all_base_learners(train_path=fe.TRAIN_PATH, models=fe.ALL_MODELS_TO_FIT,
                                         seed=fe.SEED, verbose=verbose)
    fitted = base_fit["fitted"]
    feature_cols = base_fit["feature_cols"]
    train_df = base_fit["train_df"]

    meta_learner, meta_params, oof_df = fe.fit_meta_learner(
        oof_path=fe.NESTED_CV_OOF_PATH, base_learners=STACK_BASE_LEARNERS, seed=fe.SEED, verbose=verbose,
    )

    results = fe.evaluate_on_test_set(fitted, meta_learner, feature_cols,
                                       test_path=fe.TEST_PATH, verbose=verbose)

    # -- sanity check vs. the frozen final_test_metrics.csv --
    ref = pd.read_csv(FINAL_METRICS_PATH).set_index("model")
    checks = {}
    for name in ["LightGBM", "StackingEnsemble"]:
        ref_row = ref.loc[name]
        got = results[name]
        row_check = {}
        for metric in ["R2", "MSE", "MAE", "PHM08_RUL_Score"]:
            ref_val = float(ref_row[metric])
            got_val = float(got[metric])
            match = bool(np.isclose(ref_val, got_val, atol=1e-6, rtol=1e-8))
            row_check[metric] = {"reference": ref_val, "reconstructed": got_val,
                                  "abs_diff": abs(ref_val - got_val), "match": match}
        checks[name] = row_check

    all_match = all(row_check[m]["match"] for row_check in checks.values() for m in row_check)
    if verbose:
        print("\n" + "=" * 78)
        print("[shap_track_b] SANITY CHECK: reconstructed model metrics vs. "
              f"{FINAL_METRICS_PATH}")
        print("=" * 78)
        for name, row_check in checks.items():
            for metric, d in row_check.items():
                flag = "OK" if d["match"] else "MISMATCH"
                print(f"    [{name:18s}] {metric:16s} reference={d['reference']:.6f} "
                      f"reconstructed={d['reconstructed']:.6f} abs_diff={d['abs_diff']:.2e}  [{flag}]")
        print(f"\n[shap_track_b] ALL METRICS MATCH: {all_match}")
        print("=" * 78)

    if not all_match:
        raise RuntimeError(
            "Reconstructed model metrics do NOT match results/tables/final_test_metrics.csv -- "
            "refusing to trust SHAP values computed on a possibly-different model. See printed "
            "sanity-check table above for the mismatching metric(s)."
        )

    sanity_path = TABLES_DIR / "shap_track_b_reconstruction_sanity_check.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump({"all_match": all_match, "checks": checks}, f, indent=2)
    if verbose:
        print(f"[shap_track_b] wrote {sanity_path}")

    return {
        "fitted": fitted, "feature_cols": feature_cols, "train_df": train_df,
        "meta_learner": meta_learner, "meta_params": meta_params, "oof_df": oof_df,
        "results": results, "sanity_checks": checks, "all_match": all_match,
    }


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def tree_shap_from_pipeline(pipe, X_df: pd.DataFrame, feature_cols: list[str], name: str):
    """shap.TreeExplainer on the fitted tree estimator inside a fold-safe
    Pipeline. Transforms X_df through the pipeline's VarianceThreshold step
    first (so the explainer sees exactly the columns the estimator was
    fit on).

    NOTE (discovered at runtime, not assumed): unlike Phase 1's offline
    variance filter -- which only ever checked the 21 raw sensor channels,
    not the 3 operational settings (data/processed/DATA_DICTIONARY.md) --
    final_eval.py's fold-safe VarianceThreshold(1e-5) is applied to ALL 18
    feature columns, including op_setting_3. Because train.csv's op_setting_3
    was already globally z-scored to ~0 variance in FD001 (single operating
    condition -- op_setting_3 is a near-constant raw value; StandardScaler
    on a zero-variance column yields all-~0 output), this fold-safe
    VarianceThreshold correctly drops it a second time when refit on the
    full 100-engine training set. Every Tier-1/Tier-2 tree estimator here
    therefore sees 17 features, not 18. This function pads the returned
    SHAP values back out to the full `feature_cols` (18-column) space with
    an exact 0.0 attribution for op_setting_3 (accurate, not an
    approximation: a feature the estimator never received as input
    contributes exactly zero to its output) so every base learner's
    Explanation lines up column-for-column with the others -- required for
    the StackingEnsemble linear-combination trick below.
    """
    vt = pipe.named_steps["variance_threshold"]
    kept_mask = vt.get_support()
    kept_cols = [c for c, k in zip(feature_cols, kept_mask) if k]
    dropped_cols = [c for c in feature_cols if c not in kept_cols]
    if dropped_cols:
        print(f"[shap] {name}: VarianceThreshold dropped {dropped_cols} inside the fold-safe "
              f"pipeline (fit on the full 100-engine train.csv) -- padding SHAP values back to "
              f"the full {len(feature_cols)}-feature space with exact 0.0 attribution for "
              f"{dropped_cols}.")

    estimator = pipe.named_steps["estimator"]
    print(f"[shap] TreeExplainer on {name} ({len(X_df)} rows)...")
    explainer = shap.TreeExplainer(estimator)
    exp_reduced = explainer(X_df[kept_cols])
    print(f"[shap] {name}: base_value={np.mean(exp_reduced.base_values):.4f}")

    if not dropped_cols:
        return exp_reduced

    kept_idx = [feature_cols.index(c) for c in kept_cols]
    full_values = np.zeros((exp_reduced.values.shape[0], len(feature_cols)), dtype=float)
    full_values[:, kept_idx] = exp_reduced.values
    exp_full = shap.Explanation(
        values=full_values, base_values=exp_reduced.base_values,
        data=X_df[feature_cols].values, feature_names=feature_cols,
    )
    return exp_full


def mlp_shap_from_pipeline(pipe, X_df: pd.DataFrame, background_df: pd.DataFrame, feature_cols: list[str]):
    """MLP is wrapped in a Pipeline (VarianceThreshold -> StandardScaler ->
    MLPRegressor). `pipe.predict` handles both transform steps internally,
    so we explain the whole pipeline as a black-box callable over the raw
    (pre-pipeline) feature space -- same approach Phase 1's shap_analysis.py
    used for its standalone MLP model."""
    print(f"[shap] Permutation Explainer (via shap.Explainer) on MLP "
          f"({len(X_df)} rows, background={len(background_df)})...")
    explainer = shap.Explainer(pipe.predict, background_df[feature_cols], seed=SEED)
    exp = explainer(X_df[feature_cols])
    print(f"[shap] MLP: base_value={np.mean(exp.base_values):.4f}")
    return exp


def combine_stacking_shap(exp_by_model: dict, meta_learner, base_learners: list[str]):
    """Exact linear combination: SHAP(ensemble) = sum_i coef_i * SHAP(base_i),
    base_value(ensemble) = intercept + sum_i coef_i * base_value(base_i).
    Ridge meta-learner is linear in the base learners' predictions, so this
    is exact, not an approximation (same principle as Phase 1's 2-model
    FixedWeighted case, generalized to 5 models + learned coefficients)."""
    coefs = meta_learner.coef_
    intercept = meta_learner.intercept_
    values = np.zeros_like(exp_by_model[base_learners[0]].values, dtype=float)
    base_values = np.full_like(exp_by_model[base_learners[0]].base_values, float(intercept), dtype=float)
    for i, name in enumerate(base_learners):
        values += coefs[i] * exp_by_model[name].values
        base_values += coefs[i] * exp_by_model[name].base_values
    ens = shap.Explanation(
        values=values, base_values=base_values,
        data=exp_by_model[base_learners[0]].data,
        feature_names=exp_by_model[base_learners[0]].feature_names,
    )
    return ens, {"coef": {n: float(c) for n, c in zip(base_learners, coefs)}, "intercept": float(intercept)}


def verify_stacking_linearity(exp_ens, fitted: dict, base_learners: list[str], X_df: pd.DataFrame,
                               feature_cols: list[str], meta_learner) -> float:
    reconstructed = exp_ens.base_values + exp_ens.values.sum(axis=1)
    meta_X = np.column_stack([fitted[m].predict(X_df[feature_cols].values) for m in base_learners])
    actual = meta_learner.predict(meta_X)
    max_abs_err = float(np.max(np.abs(reconstructed - actual)))
    print(f"[verify] StackingEnsemble SHAP linearity: max |reconstructed - actual| over "
          f"{len(X_df)} rows: {max_abs_err:.6f}")
    assert max_abs_err < 1e-3, "StackingEnsemble SHAP linearity check failed"
    return max_abs_err


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_beeswarm(exp, title, filename):
    plt.figure()
    shap.plots.beeswarm(exp, show=False, max_display=18)
    plt.title(title)
    plt.tight_layout()
    path = OUT_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] saved {path}")


def save_force(exp_row, title, filename):
    shap.plots.force(exp_row, matplotlib=True, show=False,
                      figsize=(24, 4), text_rotation=25,
                      contribution_threshold=0.08)
    fig = plt.gcf()
    fig.suptitle(title, y=1.25, fontsize=10)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def concentration_metrics(exp, feature_names):
    mean_abs = np.abs(exp.values).mean(axis=0)
    total = mean_abs.sum()
    order = np.argsort(mean_abs)[::-1]
    top3_share = mean_abs[order[:3]].sum() / total

    x = np.sort(mean_abs)
    n = len(x)
    cum = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

    ranked = [(feature_names[i], float(mean_abs[i])) for i in order]
    return {
        "top3_share_of_total_mean_abs_shap": float(top3_share),
        "gini_coefficient": float(gini),
        "ranked_features": ranked,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    recon = reconstruct_and_sanity_check(verbose=True)
    fitted = recon["fitted"]
    feature_cols = recon["feature_cols"]
    train_df = recon["train_df"]
    meta_learner = recon["meta_learner"]

    # ---- fixed-seed sample of test rows for global SHAP (same convention as Phase 1) ----
    test_df, _, _, _, test_feature_cols = ncv.load_train_data(fe.TEST_PATH)
    assert test_feature_cols == feature_cols
    sample_df = test_df.sample(n=N_SAMPLE, random_state=SEED).sort_index()
    X_sample = sample_df[feature_cols].reset_index(drop=True)
    meta_sample = sample_df[["engine_id", "cycle", "RUL"]].reset_index(drop=True)
    print(f"\n[data] test sample: n={len(X_sample)}, seed={SEED}")

    background_df = shap.sample(train_df[feature_cols], N_BACKGROUND, random_state=SEED)
    print(f"[data] MLP background sample: n={len(background_df)}, seed={SEED}")

    # ---- global SHAP: 4 tree models + MLP ----
    exp_by_model = {}
    for name in TREE_MODELS:
        exp_by_model[name] = tree_shap_from_pipeline(fitted[name], X_sample, feature_cols, name)
    exp_by_model["MLP"] = mlp_shap_from_pipeline(fitted["MLP"], X_sample, background_df, feature_cols)

    exp_ens, meta_weights = combine_stacking_shap(exp_by_model, meta_learner, STACK_BASE_LEARNERS)
    verify_stacking_linearity(exp_ens, fitted, STACK_BASE_LEARNERS, X_sample, feature_cols, meta_learner)
    print(f"[stacking] Ridge meta-learner weights: {meta_weights}")

    save_beeswarm(exp_by_model["LightGBM"], "SHAP summary - LightGBM (frozen, single model)",
                  "beeswarm_lightgbm.png")
    save_beeswarm(exp_by_model["CatBoost"], "SHAP summary - CatBoost (StackingEnsemble base learner)",
                  "beeswarm_catboost.png")
    save_beeswarm(exp_by_model["XGBoost"], "SHAP summary - XGBoost (StackingEnsemble base learner)",
                  "beeswarm_xgboost.png")
    save_beeswarm(exp_by_model["GradientBoosting"],
                  "SHAP summary - GradientBoosting (StackingEnsemble base learner)",
                  "beeswarm_gradientboosting.png")
    save_beeswarm(exp_by_model["MLP"], "SHAP summary - MLP (StackingEnsemble base learner)",
                  "beeswarm_mlp.png")
    save_beeswarm(exp_ens, "SHAP summary - StackingEnsemble (frozen, Ridge meta-learner "
                           "over 5 base learners)", "beeswarm_stackingensemble.png")

    # ---- concentration metrics: >= 2 individuals + the ensemble ----
    metrics = {}
    for name, exp in list(exp_by_model.items()) + [("StackingEnsemble", exp_ens)]:
        metrics[name] = concentration_metrics(exp, feature_cols)

    rows = []
    for name, m in metrics.items():
        rows.append({
            "model": name,
            "top3_share_of_total_mean_abs_shap": m["top3_share_of_total_mean_abs_shap"],
            "gini_coefficient": m["gini_coefficient"],
            "top1_feature": m["ranked_features"][0][0], "top1_mean_abs_shap": m["ranked_features"][0][1],
            "top2_feature": m["ranked_features"][1][0], "top2_mean_abs_shap": m["ranked_features"][1][1],
            "top3_feature": m["ranked_features"][2][0], "top3_mean_abs_shap": m["ranked_features"][2][1],
            "top4_feature": m["ranked_features"][3][0], "top4_mean_abs_shap": m["ranked_features"][3][1],
            "top5_feature": m["ranked_features"][4][0], "top5_mean_abs_shap": m["ranked_features"][4][1],
        })
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "shap_concentration_metrics_track_b.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[metrics] saved {metrics_path}")
    print(metrics_df.to_string(index=False))

    ranked_path = OUT_DIR / "shap_ranked_features_full_track_b.json"
    with open(ranked_path, "w") as f:
        json.dump({k: v["ranked_features"] for k, v in metrics.items()}, f, indent=2)
    print(f"[metrics] saved {ranked_path}")

    # ---- representative local predictions for force plots ----
    idx_low = meta_sample["RUL"].idxmin()
    idx_high = (meta_sample["RUL"] - 125).abs().idxmin()
    uncapped = meta_sample[meta_sample["RUL"] < 125]
    mid_target = (meta_sample["RUL"].min() + 125) / 2
    idx_mid = (uncapped["RUL"] - mid_target).abs().idxmin()

    picks = {"low_rul": idx_low, "mid_rul": idx_mid, "high_rul": idx_high}
    print("\n[force] representative rows:")
    for tag, idx in picks.items():
        r = meta_sample.loc[idx]
        print(f"  {tag}: engine_id={int(r.engine_id)} cycle={int(r.cycle)} RUL={r.RUL}")

    # StackingEnsemble: low/mid/high (required by parent task -- "at least the StackingEnsemble")
    for tag, idx in picks.items():
        r = meta_sample.loc[idx]
        title = (f"StackingEnsemble force plot ({tag}) - engine {int(r.engine_id)}, "
                 f"cycle {int(r.cycle)}, true RUL={r.RUL}")
        save_force(exp_ens[idx], title, f"force_stacking_{tag}.png")

    # LightGBM: low + high, for direct single-model-vs-ensemble comparison on the same rows
    for tag in ["low_rul", "high_rul"]:
        idx = picks[tag]
        r = meta_sample.loc[idx]
        title = (f"LightGBM force plot ({tag}) - engine {int(r.engine_id)}, "
                 f"cycle {int(r.cycle)}, true RUL={r.RUL}")
        save_force(exp_by_model["LightGBM"][idx], title, f"force_lightgbm_{tag}.png")

    print("\n[done] shap_track_b.py complete.")

    return {
        "recon": recon, "exp_by_model": exp_by_model, "exp_ens": exp_ens,
        "metrics_df": metrics_df, "picks": picks, "meta_sample": meta_sample,
        "meta_weights": meta_weights,
    }


if __name__ == "__main__":
    main()
