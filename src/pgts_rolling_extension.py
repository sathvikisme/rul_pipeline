"""
pgts_rolling_extension.py — RUL-Bench leakage-red-team subagent

Extension experiment: does adding rolling-window sensor features
(features.py:add_rolling_features, window=5, per-engine so it never crosses
engine boundaries) narrow the gap between plain-KFold CV (mildly leaky,
row-level) and PGTS (embargo=10, leakage-safe) for the strongest model
(XGBoost)?

One clean experiment, reported honestly either way. Run as
`python src/pgts_rolling_extension.py` from the repo root.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".claude", "skills", "pgts-split"))

import features as feat  # noqa: E402
from pgts import purged_group_time_series_split, _assert_no_leakage  # noqa: E402

from xgboost import XGBRegressor

SEED = 42
N_SPLITS = 5
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train.csv")

train_df = pd.read_csv(TRAIN_PATH)
sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
print(f"[ext] sensor_cols ({len(sensor_cols)}): {sensor_cols}")

# --- Baseline feature set (no rolling features) ---
base_feature_cols = [c for c in train_df.columns if c not in ("engine_id", "RUL")]

# --- Rolling-window feature set ---
train_roll = feat.add_rolling_features(train_df, sensor_cols, window=5)
roll_feature_cols = [c for c in train_roll.columns if c not in ("engine_id", "RUL")]
print(f"[ext] base feature count: {len(base_feature_cols)}, with rolling: {len(roll_feature_cols)}")

groups = train_roll["engine_id"].values
y = train_roll["RUL"].values

xgb_params = dict(learning_rate=0.05, max_depth=5, n_estimators=200, random_state=SEED, n_jobs=1, verbosity=0)


def eval_kfold(X, y):
    est = XGBRegressor(**xgb_params)
    r2 = cross_val_score(est, X, y, cv=KFold(N_SPLITS, shuffle=True, random_state=SEED), scoring="r2", n_jobs=-1)
    est2 = XGBRegressor(**xgb_params)
    neg_mse = cross_val_score(est2, X, y, cv=KFold(N_SPLITS, shuffle=True, random_state=SEED), scoring="neg_mean_squared_error", n_jobs=-1)
    return r2, -neg_mse


def eval_pgts(X, y, groups, embargo=10):
    r2s, mses = [], []
    for train_idx, test_idx in purged_group_time_series_split(groups, n_splits=N_SPLITS, embargo=embargo):
        _assert_no_leakage(groups, train_idx, test_idx)
        est = XGBRegressor(**xgb_params)
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])
        r2s.append(r2_score(y[test_idx], pred))
        mses.append(mean_squared_error(y[test_idx], pred))
    return np.array(r2s), np.array(mses)


results = {}
for label, cols in [("baseline_no_rolling", base_feature_cols), ("with_rolling_w5", roll_feature_cols)]:
    X = train_roll[cols].values
    print(f"\n[ext] === {label} (n_features={len(cols)}) ===")

    kf_r2, kf_mse = eval_kfold(X, y)
    print(f"  plain KFold: R2 mean={kf_r2.mean():.4f} std={kf_r2.std():.4f}  MSE mean={kf_mse.mean():.2f}")

    pg_r2, pg_mse = eval_pgts(X, y, groups, embargo=10)
    print(f"  PGTS(embargo=10): R2 mean={pg_r2.mean():.4f} std={pg_r2.std():.4f}  MSE mean={pg_mse.mean():.2f}")

    gap_r2 = kf_r2.mean() - pg_r2.mean()
    gap_mse_pct = (pg_mse.mean() - kf_mse.mean()) / kf_mse.mean() * 100
    print(f"  KFold-PGTS R2 gap = {gap_r2:.4f}, PGTS MSE is {gap_mse_pct:+.1f}% vs KFold MSE")

    results[label] = {
        "n_features": len(cols),
        "kfold_r2_mean": float(kf_r2.mean()), "kfold_r2_std": float(kf_r2.std()),
        "kfold_mse_mean": float(kf_mse.mean()),
        "pgts10_r2_mean": float(pg_r2.mean()), "pgts10_r2_std": float(pg_r2.std()),
        "pgts10_mse_mean": float(pg_mse.mean()),
        "kfold_pgts_r2_gap": float(gap_r2),
        "pgts_mse_pct_vs_kfold": float(gap_mse_pct),
    }

out_path = os.path.join(REPO_ROOT, "results", "tables", "pgts_rolling_extension.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[ext] Wrote -> {out_path}")

print("\n[ext] SUMMARY")
b = results["baseline_no_rolling"]
r = results["with_rolling_w5"]
print(f"  baseline PGTS(10) R2: {b['pgts10_r2_mean']:.4f}  ->  with rolling: {r['pgts10_r2_mean']:.4f}  "
      f"(delta {r['pgts10_r2_mean']-b['pgts10_r2_mean']:+.4f})")
print(f"  baseline PGTS(10) MSE: {b['pgts10_mse_mean']:.2f}  ->  with rolling: {r['pgts10_mse_mean']:.2f}  "
      f"(delta {r['pgts10_mse_mean']-b['pgts10_mse_mean']:+.2f})")
print(f"  baseline KFold-PGTS R2 gap: {b['kfold_pgts_r2_gap']:.4f}  ->  with rolling: {r['kfold_pgts_r2_gap']:.4f}")
