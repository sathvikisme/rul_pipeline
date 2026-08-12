"""
shap_analysis.py -- interpretability-analyst subagent, RUL-Bench.

Reproduces Sec.7 of PROJECT_BRIEF.md: SHAP summary (beeswarm) plots, SHAP
force plots on a handful of representative predictions, and an attribution
concentration comparison (the source paper's claim that its ensemble spreads
credit more evenly across features than any single base model).

Models analyzed (chosen from results/tables/official_split_metrics.csv,
already-trained artifacts in results/models/ -- nothing here is retrained):

  - XGBoost   (individual, top-tier by test R^2 = 0.6776)
  - CatBoost  (individual, top-tier by test R^2 = 0.6759)
  - MLP       (individual, top-tier by test R^2 = 0.6775; computed because it
               is a required ingredient of the FixedWeighted ensemble below,
               and reported standalone too since it's "free" once computed)
  - FixedWeighted ensemble = 0.7 * XGBoost + 0.3 * MLP (results/models/
    FixedWeightedEnsemble.joblib, weights confirmed against
    results/tables/ensembling_config.json)

Why XGBoost + CatBoost as the two individuals: both are top-tier by official
test R^2, and both support shap.TreeExplainer, which is fast and *exact*
(no sampling approximation) for tree ensembles -- see PHM08/official metrics
table for the R^2 ranking that motivated this pick.

Why the FixedWeighted ensemble (over Stacking_Ridge): SHAP is linear under
linear combination. FixedWeighted is a literal linear combination of two
already-fitted models (0.7*XGBoost + 0.3*MLP), so
    SHAP(ensemble) = 0.7 * SHAP(XGBoost) + 0.3 * SHAP(MLP)
is EXACT -- not an approximation -- with zero extra explainer calls beyond
what XGBoost + MLP already need. (Stacking_Ridge would require the same
trick across 5 base learners including 2 more non-tree-friendly models for
no analytical gain; FixedWeighted gets the identical exactness guarantee the
brief describes, for less compute.) This is verified numerically below by
checking the reconstructed ensemble prediction (base_value + sum(SHAP))
against the ensemble's actual weighted-average prediction.

MLP is not a tree model, so its SHAP values are computed with
shap.Explainer(model.predict, background) with a small fixed background
sample (shap.sample, n=50, seed=42) -- shap dispatches this to its
Permutation explainer for a generic callable, which is exact under a
feature-independence assumption (documented limitation, standard practice
for black-box models).

Fixed seed 42 used everywhere sampling occurs (test-row sample, background
sample) per CLAUDE.md rule 3.

Outputs (results/shap/):
  beeswarm_xgboost.png, beeswarm_catboost.png, beeswarm_mlp.png,
  beeswarm_ensemble_fixedweighted.png
  force_ensemble_low_rul.png, force_ensemble_mid_rul.png,
  force_ensemble_high_rul.png, force_xgboost_low_rul.png,
  force_catboost_low_rul.png
  shap_concentration_metrics.csv
"""
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

SEED = 42
N_SAMPLE = 500          # test rows used for global SHAP / beeswarm plots
N_BACKGROUND = 50       # background sample for the MLP permutation explainer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "results" / "models"
OUT_DIR = ROOT / "results" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]

# Physical sensor documentation (C-MAPSS / PHM08 convention, Saxena & Goebel),
# restricted to the 14 kept sensors + op settings.
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


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def load_models():
    xgb = joblib.load(MODELS_DIR / "XGBoost.joblib")
    cat = joblib.load(MODELS_DIR / "CatBoost.joblib")
    mlp = joblib.load(MODELS_DIR / "MLP.joblib")
    fw = joblib.load(MODELS_DIR / "FixedWeightedEnsemble.joblib")
    assert fw["models"] == ("XGBoost", "MLP"), fw
    assert fw["weights"] == (0.7, 0.3), fw
    return xgb, cat, mlp, fw


def compute_tree_shap(model, X_df, name):
    print(f"[shap] TreeExplainer on {name} ({len(X_df)} rows)...")
    explainer = shap.TreeExplainer(model)
    exp = explainer(X_df)
    print(f"[shap] {name}: base_value={np.mean(exp.base_values):.4f}")
    return exp


def compute_mlp_shap(model, X_df, background_df):
    print(f"[shap] Permutation Explainer (via shap.Explainer) on MLP "
          f"({len(X_df)} rows, background={len(background_df)})...")
    explainer = shap.Explainer(model.predict, background_df, seed=SEED)
    exp = explainer(X_df)
    print(f"[shap] MLP: base_value={np.mean(exp.base_values):.4f}")
    return exp


def combine_fixed_weighted(exp_xgb, exp_mlp, w_xgb=0.7, w_mlp=0.3):
    """Exact linear combination of two SHAP Explanation objects (same rows,
    same feature order) into the FixedWeighted ensemble's attribution."""
    values = w_xgb * exp_xgb.values + w_mlp * exp_mlp.values
    base_values = w_xgb * exp_xgb.base_values + w_mlp * exp_mlp.base_values
    ens = shap.Explanation(
        values=values,
        base_values=base_values,
        data=exp_xgb.data,
        feature_names=exp_xgb.feature_names,
    )
    return ens


def verify_linearity(exp_ens, xgb_model, mlp_model, X_df):
    """Sanity check: base_value + sum(SHAP) should reconstruct the actual
    ensemble prediction (0.7*XGBoost_pred + 0.3*MLP_pred) for each row."""
    reconstructed = exp_ens.base_values + exp_ens.values.sum(axis=1)
    actual = 0.7 * xgb_model.predict(X_df) + 0.3 * mlp_model.predict(X_df)
    max_abs_err = np.max(np.abs(reconstructed - actual))
    print(f"[verify] max |reconstructed - actual| over {len(X_df)} rows: "
          f"{max_abs_err:.6f}")
    assert max_abs_err < 1e-3, "Ensemble SHAP linearity check failed"
    return max_abs_err


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
    # contribution_threshold groups low-impact features into "N other
    # features" so the plot stays legible with 18 candidate features;
    # a wide figsize + rotated labels avoids the label-overlap that a
    # default-sized force plot produces with this many features.
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
    """Mean |SHAP| per feature -> top-3 share of total, and Gini coefficient
    of the mean-|SHAP| distribution across features (0 = perfectly even
    credit across all features, 1 = all credit on one feature)."""
    mean_abs = np.abs(exp.values).mean(axis=0)
    total = mean_abs.sum()
    order = np.argsort(mean_abs)[::-1]
    top3_share = mean_abs[order[:3]].sum() / total

    # Gini coefficient over the mean_abs distribution
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


def main():
    train, test = load_data()
    xgb, cat, mlp, fw = load_models()

    # ---- draw fixed-seed sample of test rows for global SHAP ----
    sample_df = test.sample(n=N_SAMPLE, random_state=SEED).sort_index()
    X_sample = sample_df[FEATURE_COLS].reset_index(drop=True)
    meta_sample = sample_df[["engine_id", "cycle", "RUL"]].reset_index(drop=True)
    print(f"[data] test sample: n={len(X_sample)}, seed={SEED}")

    background_df = shap.sample(train[FEATURE_COLS], N_BACKGROUND, random_state=SEED)
    print(f"[data] MLP background sample: n={len(background_df)}, seed={SEED}")

    # ---- global SHAP ----
    exp_xgb = compute_tree_shap(xgb, X_sample, "XGBoost")
    exp_cat = compute_tree_shap(cat, X_sample, "CatBoost")
    exp_mlp = compute_mlp_shap(mlp, X_sample, background_df)
    exp_ens = combine_fixed_weighted(exp_xgb, exp_mlp, 0.7, 0.3)
    verify_linearity(exp_ens, xgb, mlp, X_sample)

    save_beeswarm(exp_xgb, "SHAP summary - XGBoost (individual)", "beeswarm_xgboost.png")
    save_beeswarm(exp_cat, "SHAP summary - CatBoost (individual)", "beeswarm_catboost.png")
    save_beeswarm(exp_mlp, "SHAP summary - MLP (individual)", "beeswarm_mlp.png")
    save_beeswarm(exp_ens, "SHAP summary - FixedWeighted Ensemble (0.7*XGBoost + 0.3*MLP)",
                  "beeswarm_ensemble_fixedweighted.png")

    # ---- concentration metrics ----
    metrics = {}
    for name, exp in [("XGBoost", exp_xgb), ("CatBoost", exp_cat),
                       ("MLP", exp_mlp), ("FixedWeighted_Ensemble", exp_ens)]:
        metrics[name] = concentration_metrics(exp, FEATURE_COLS)

    rows = []
    for name, m in metrics.items():
        row = {
            "model": name,
            "top3_share_of_total_mean_abs_shap": m["top3_share_of_total_mean_abs_shap"],
            "gini_coefficient": m["gini_coefficient"],
            "top1_feature": m["ranked_features"][0][0],
            "top1_mean_abs_shap": m["ranked_features"][0][1],
            "top2_feature": m["ranked_features"][1][0],
            "top2_mean_abs_shap": m["ranked_features"][1][1],
            "top3_feature": m["ranked_features"][2][0],
            "top3_mean_abs_shap": m["ranked_features"][2][1],
        }
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "shap_concentration_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[metrics] saved {metrics_path}")
    print(metrics_df.to_string(index=False))

    # full ranked-feature lists for the report
    ranked_path = OUT_DIR / "shap_ranked_features_full.json"
    with open(ranked_path, "w") as f:
        json.dump({k: v["ranked_features"] for k, v in metrics.items()}, f, indent=2)
    print(f"[metrics] saved {ranked_path}")

    # ---- representative local predictions for force plots ----
    # low RUL: near end-of-life; mid RUL: near median; high RUL: capped plateau (125)
    idx_low = meta_sample["RUL"].idxmin()
    idx_high = (meta_sample["RUL"] - 125).abs().idxmin()  # a capped/early-life row
    # "mid-life": target a RUL roughly halfway between near-failure and the
    # capped plateau, restricted to *uncapped* rows (RUL < 125) so it's a
    # genuinely distinct, actively-degrading mid-life point rather than
    # colliding with the early-life/capped row picked above.
    uncapped = meta_sample[meta_sample["RUL"] < 125]
    mid_target = (meta_sample["RUL"].min() + 125) / 2
    idx_mid = (uncapped["RUL"] - mid_target).abs().idxmin()

    picks = {"low_rul": idx_low, "mid_rul": idx_mid, "high_rul": idx_high}
    print("[force] representative rows:")
    for tag, idx in picks.items():
        r = meta_sample.loc[idx]
        print(f"  {tag}: engine_id={int(r.engine_id)} cycle={int(r.cycle)} RUL={r.RUL}")

    for tag, idx in picks.items():
        r = meta_sample.loc[idx]
        title = (f"Ensemble force plot ({tag}) - engine {int(r.engine_id)}, "
                  f"cycle {int(r.cycle)}, true RUL={r.RUL}")
        save_force(exp_ens[idx], title, f"force_ensemble_{tag}.png")

    # comparison: same low-RUL (near-failure) row, individual models
    idx = picks["low_rul"]
    r = meta_sample.loc[idx]
    save_force(exp_xgb[idx],
               f"XGBoost force plot (low_rul) - engine {int(r.engine_id)}, "
               f"cycle {int(r.cycle)}, true RUL={r.RUL}",
               "force_xgboost_low_rul.png")
    save_force(exp_cat[idx],
               f"CatBoost force plot (low_rul) - engine {int(r.engine_id)}, "
               f"cycle {int(r.cycle)}, true RUL={r.RUL}",
               "force_catboost_low_rul.png")

    print("[done] shap_analysis.py complete.")


if __name__ == "__main__":
    main()
