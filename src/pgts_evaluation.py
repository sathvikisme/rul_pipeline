"""
pgts_evaluation.py — RUL-Bench leakage-red-team subagent

PGTS re-evaluation script. Run as `python src/pgts_evaluation.py` from the
repo root (or `python pgts_evaluation.py` from within src/).

Retrains the 4 strongest official-split individual models (XGBoost, MLP,
CatBoost, LightGBM) fresh on each PGTS fold (embargo=10 and embargo=0),
using the SAME hyperparameters model-trainer already selected via
GridSearchCV on the official split (results/tables/best_hyperparams.json) --
we are not re-tuning, only re-validating under a leakage-safe split.

Also computes:
  - a null baseline (predict training-fold mean RUL) for every PGTS fold
    and for the official test.csv split, for direct comparison.
  - reproduces model-trainer's plain-KFold CV numbers as a comparison
    column pulled straight from results/tables/cv_scores_r2.csv /
    cv_scores_mse.csv (already computed, not re-run).

Writes results/tables/pgts_comparison.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, ".claude", "skills", "pgts-split"))
from pgts import purged_group_time_series_split, _assert_no_leakage  # noqa: E402

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

SEED = 42
N_SPLITS = 5

TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(REPO_ROOT, "data", "processed", "test.csv")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Confirm sort order (must be true for PGTS group-contiguity assumption).
assert train_df.reset_index(drop=True).equals(
    train_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
), "train.csv is not sorted by engine_id,cycle -- PGTS requires contiguous groups"

feature_cols = [c for c in train_df.columns if c not in ("engine_id", "RUL")]
X_train_full = train_df[feature_cols].values
y_train_full = train_df["RUL"].values
groups = train_df["engine_id"].values

X_test_official = test_df[feature_cols].values
y_test_official = test_df["RUL"].values

print(f"[pgts-eval] train.csv: {train_df.shape}, {train_df['engine_id'].nunique()} engines")
print(f"[pgts-eval] feature_cols ({len(feature_cols)}): {feature_cols}")

# ---------------------------------------------------------------------------
# Load best hyperparams selected by model-trainer (reuse, do not re-tune)
# ---------------------------------------------------------------------------
with open(os.path.join(TABLES_DIR, "best_hyperparams.json"), encoding="utf-8") as f:
    best_hp = {row["model"]: json.loads(row["best_params"]) for row in json.load(f)}

print("[pgts-eval] Reusing best_hyperparams.json:")
for name in ["XGBoost", "MLP", "CatBoost", "LightGBM"]:
    print(f"  {name}: {best_hp[name]}")


def make_estimator(name):
    p = best_hp[name]
    if name == "XGBoost":
        return XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0, **p)
    if name == "LightGBM":
        return LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1, **p)
    if name == "CatBoost":
        return CatBoostRegressor(random_state=SEED, thread_count=1, verbose=False,
                                  allow_writing_files=False, **p)
    if name == "MLP":
        return MLPRegressor(random_state=SEED, max_iter=500, early_stopping=True,
                             n_iter_no_change=15, **p)
    raise ValueError(name)


MODEL_NAMES = ["XGBoost", "MLP", "CatBoost", "LightGBM"]

# ---------------------------------------------------------------------------
# Official-split null baseline (reference point)
# ---------------------------------------------------------------------------
official_null_pred = np.full_like(y_test_official, fill_value=y_train_full.mean(), dtype=float)
official_null_r2 = r2_score(y_test_official, official_null_pred)
official_null_mse = mean_squared_error(y_test_official, official_null_pred)
print(f"\n[pgts-eval] OFFICIAL-SPLIT null baseline (predict train-set mean RUL): "
      f"R2={official_null_r2:.4f} MSE={official_null_mse:.2f}")

# ---------------------------------------------------------------------------
# PGTS re-evaluation, embargo=10 and embargo=0
# ---------------------------------------------------------------------------
results = {}  # results[embargo][model] = list of per-fold dicts

for embargo in (10, 0):
    print("\n" + "=" * 70)
    print(f"[pgts-eval] PGTS embargo={embargo}")
    print("=" * 70)

    fold_num = 0
    per_model_folds = {name: [] for name in MODEL_NAMES}
    null_folds = []

    for train_idx, test_idx in purged_group_time_series_split(groups, n_splits=N_SPLITS, embargo=embargo):
        fold_num += 1
        _assert_no_leakage(groups, train_idx, test_idx)  # hard assertion, per skill instructions

        X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
        X_te, y_te = X_train_full[test_idx], y_train_full[test_idx]

        n_test_engines = len(set(groups[test_idx]))
        print(f"\n  fold {fold_num}: train_rows={len(train_idx)} test_rows={len(test_idx)} "
              f"test_engines={n_test_engines}")

        # Null baseline for this fold
        null_pred = np.full_like(y_te, fill_value=y_tr.mean(), dtype=float)
        null_r2 = r2_score(y_te, null_pred)
        null_mse = mean_squared_error(y_te, null_pred)
        null_folds.append({"fold": fold_num, "R2": null_r2, "MSE": null_mse})
        print(f"    null baseline: R2={null_r2:.4f} MSE={null_mse:.2f}")

        for name in MODEL_NAMES:
            est = make_estimator(name)
            est.fit(X_tr, y_tr)
            y_pred = est.predict(X_te)
            r2 = r2_score(y_te, y_pred)
            mse = mean_squared_error(y_te, y_pred)
            per_model_folds[name].append({"fold": fold_num, "R2": r2, "MSE": mse})
            print(f"    {name:16s}: R2={r2:.4f} MSE={mse:.2f}")

    results[embargo] = {"models": per_model_folds, "null": null_folds}

# ---------------------------------------------------------------------------
# Build the full comparison table
# ---------------------------------------------------------------------------
official_df = pd.read_csv(os.path.join(TABLES_DIR, "official_split_metrics.csv")).set_index("model")
cv_r2_df = pd.read_csv(os.path.join(TABLES_DIR, "cv_scores_r2.csv"))
cv_mse_df = pd.read_csv(os.path.join(TABLES_DIR, "cv_scores_mse.csv"))

rows = []
for name in MODEL_NAMES:
    row = {"model": name}
    row["official_split_R2"] = official_df.loc[name, "R2"]
    row["official_split_MSE"] = official_df.loc[name, "MSE"]
    row["plain_kfold_cv_R2_mean"] = cv_r2_df[name].mean()
    row["plain_kfold_cv_R2_std"] = cv_r2_df[name].std()
    row["plain_kfold_cv_MSE_mean"] = cv_mse_df[name].mean()
    row["plain_kfold_cv_MSE_std"] = cv_mse_df[name].std()

    for embargo in (10, 0):
        fold_recs = results[embargo]["models"][name]
        r2_vals = [r["R2"] for r in fold_recs]
        mse_vals = [r["MSE"] for r in fold_recs]
        row[f"pgts_embargo{embargo}_R2_mean"] = float(np.mean(r2_vals))
        row[f"pgts_embargo{embargo}_R2_std"] = float(np.std(r2_vals))
        row[f"pgts_embargo{embargo}_MSE_mean"] = float(np.mean(mse_vals))
        row[f"pgts_embargo{embargo}_MSE_std"] = float(np.std(mse_vals))

    rows.append(row)

# Null baseline rows (once per embargo + official), reported as its own pseudo-model row too
null_row = {"model": "NULL_BASELINE_mean_RUL"}
null_row["official_split_R2"] = official_null_r2
null_row["official_split_MSE"] = official_null_mse
null_row["plain_kfold_cv_R2_mean"] = np.nan
null_row["plain_kfold_cv_R2_std"] = np.nan
null_row["plain_kfold_cv_MSE_mean"] = np.nan
null_row["plain_kfold_cv_MSE_std"] = np.nan
for embargo in (10, 0):
    r2_vals = [r["R2"] for r in results[embargo]["null"]]
    mse_vals = [r["MSE"] for r in results[embargo]["null"]]
    null_row[f"pgts_embargo{embargo}_R2_mean"] = float(np.mean(r2_vals))
    null_row[f"pgts_embargo{embargo}_R2_std"] = float(np.std(r2_vals))
    null_row[f"pgts_embargo{embargo}_MSE_mean"] = float(np.mean(mse_vals))
    null_row[f"pgts_embargo{embargo}_MSE_std"] = float(np.std(mse_vals))
rows.append(null_row)

comparison_df = pd.DataFrame(rows)
out_path = os.path.join(TABLES_DIR, "pgts_comparison.csv")
comparison_df.to_csv(out_path, index=False)
print("\n" + "=" * 70)
print(f"[pgts-eval] Wrote comparison table -> {out_path}")
print("=" * 70)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(comparison_df.to_string(index=False))

# Also dump raw per-fold numbers for the report / sanity-checking.
raw_out = os.path.join(TABLES_DIR, "pgts_perfold_raw.json")
with open(raw_out, "w", encoding="utf-8") as f:
    json.dump({
        "official_null": {"R2": official_null_r2, "MSE": official_null_mse},
        "embargo_10": {
            "models": results[10]["models"],
            "null": results[10]["null"],
        },
        "embargo_0": {
            "models": results[0]["models"],
            "null": results[0]["null"],
        },
    }, f, indent=2)
print(f"[pgts-eval] Wrote per-fold raw numbers -> {raw_out}")
