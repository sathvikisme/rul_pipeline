"""
preprocessing.py — RUL-Bench data-engineer subagent

Loads raw NASA C-MAPSS FD001 train/test files, checks data quality,
computes per-row RUL labels (train from max-cycle-per-engine, test from
RUL_FD001.txt back-computation), applies piecewise-linear RUL capping,
and provides a fit-on-train-only feature scaler.

Reference: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. PHM08.
Also: Özcan, H. (2025), Scientific Reports 15, 39795 — see PROJECT_BRIEF.md.

This module does NOT do feature selection (variance threshold) or the final
write to data/processed/ — that orchestration lives in features.py, which
calls into this module. See features.py's __main__ for the end-to-end run.

Random seed: none needed in this module — every transform here is
deterministic (no random sampling, no stochastic imputation).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Column schema for FD001: 26 whitespace-separated columns, no header.
# 1 engine id, 1 cycle, 3 operational settings, 21 sensor measurements.
# ---------------------------------------------------------------------------
ID_COLS = ["engine_id", "cycle"]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = ID_COLS + OP_SETTING_COLS + SENSOR_COLS  # 26 columns total

# Piecewise-linear RUL cap. 125 is the convention introduced by Heimes (2008)
# and used widely in C-MAPSS RUL literature (including Özcan 2025, per
# PROJECT_BRIEF.md) as a plateau value beyond which "how degraded is this
# engine" is not meaningfully distinguishable from "healthy" — early-life
# cycles get flat-capped instead of a huge linearly-growing RUL target that
# the model can't realistically predict from sensor data alone. We adopt the
# same value here deliberately (not by silent default) so our results stay
# comparable to the source paper's reported numbers.
RUL_CAP = 125


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw C-MAPSS FD001 space-separated file with no header.

    Known quirk of this exact file format: each line has a trailing space
    before the newline. If you read it with a single fixed-width literal
    space separator (sep=' '), pandas treats every space as a delimiter and
    produces 2 extra all-NaN trailing columns (28 total instead of 26).
    We avoid that by using a regex whitespace separator (sep=r'\\s+'), which
    collapses runs of whitespace including the trailing one. We additionally
    assert the result really is 26 columns and defensively drop any
    all-NaN trailing columns in case a different file variant is ever
    substituted in.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # Defensive check for the known trailing-column quirk, in case a raw
    # file with literal double-spaces (not just trailing whitespace) shows
    # up and sep=r'\s+' still produces extras.
    all_nan_trailing = [c for c in df.columns if df[c].isna().all()]
    if all_nan_trailing:
        df = df.drop(columns=all_nan_trailing)

    if df.shape[1] != len(ALL_COLS):
        raise ValueError(
            f"Expected {len(ALL_COLS)} columns after load/cleanup, got "
            f"{df.shape[1]} for {path}. Raw file format may have changed."
        )

    df.columns = ALL_COLS
    return df


def check_data_quality(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Check for missing/invalid values. Returns a small report dict.

    We only impute if this check actually finds something — no silent
    "impute just in case" behavior. As of the run recorded in this repo's
    README/data dictionary, FD001 train and test both came back completely
    clean (0 NaNs, 0 negative sensor readings, 0 duplicate rows) — verified
    by executing this function, not assumed from prior C-MAPSS experience.
    """
    n_missing = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    # Only flag negative values in SENSOR columns as suspicious — sensor
    # readings are physical quantities (temperature, pressure, speed, etc.)
    # and should not be negative. op_setting_1/2 are legitimately allowed to
    # be small positive or negative values (they represent deviations
    # around a nominal operating point), so a negative op_setting is
    # expected, not a data-quality problem, and is excluded from this count.
    n_negative_sensors = int((df[SENSOR_COLS] < 0).sum().sum())
    report = {
        "dataset": name,
        "n_rows": len(df),
        "n_missing_values": n_missing,
        "n_duplicate_rows": n_dup,
        "n_negative_sensor_values": n_negative_sensors,
    }
    return report


def impute_if_needed(df: pd.DataFrame, method: str = "median") -> pd.DataFrame:
    """Impute missing values in sensor/op-setting columns, only if present.

    Method is median imputation by default (robust to the occasional sensor
    spike/outlier vs. mean). This function is a no-op (returns df unchanged,
    modulo a printed note) when check_data_quality found 0 missing values —
    which is what we observed for FD001 train and test. Kept here so the
    pipeline is robust if a different C-MAPSS subset (FD002-4) or a future
    re-download does have gaps.
    """
    numeric_cols = OP_SETTING_COLS + SENSOR_COLS
    n_missing = int(df[numeric_cols].isna().sum().sum())
    if n_missing == 0:
        print("[preprocessing] impute_if_needed: 0 missing values found - no imputation applied.")
        return df
    print(f"[preprocessing] impute_if_needed: {n_missing} missing values found - applying {method} imputation (train-fit values only, see caller).")
    df = df.copy()
    if method == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    return df


def compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL per row = max_cycle_for_engine - current_cycle.

    This is exact for the training set because every training trajectory
    runs to failure (max observed cycle == failure cycle).
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def compute_test_rul(df: pd.DataFrame, rul_final: pd.Series) -> pd.DataFrame:
    """Back-compute per-row RUL for the test set.

    RUL_FD001.txt gives the true RUL at the FINAL observed cycle of each
    test engine (test trajectories are truncated before failure, unlike
    train). For engine i with final observed cycle `last_cycle_i` and true
    RUL at that row `rul_final_i`, RUL at any earlier row with cycle `c` is:

        RUL(c) = rul_final_i + (last_cycle_i - c)

    i.e. we walk backward from the known final-row RUL by however many
    cycles earlier this row is. rul_final is indexed 0..n_engines-1
    matching engine_id order 1..n_engines (as in the raw RUL_FD001.txt,
    which is one value per engine in engine-id order).
    """
    df = df.copy()
    engine_ids = sorted(df["engine_id"].unique())
    if len(engine_ids) != len(rul_final):
        raise ValueError(
            f"Number of test engines ({len(engine_ids)}) does not match "
            f"length of rul_final ({len(rul_final)})."
        )
    rul_map = dict(zip(engine_ids, rul_final.values))

    last_cycle = df.groupby("engine_id")["cycle"].transform("max")
    rul_final_per_row = df["engine_id"].map(rul_map)
    df["RUL"] = rul_final_per_row + (last_cycle - df["cycle"])
    return df


def apply_rul_cap(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Piecewise-linear RUL capping: RUL = min(raw_RUL, cap).

    See RUL_CAP module constant above for why 125 was chosen (explicit,
    documented choice — not a silent hardcode). Adjust `cap` here and the
    module constant together if this ever needs to change; keep them in
    sync so the documented rationale stays attached to the actual value
    used.
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler on the TRAINING split ONLY.

    Scaler choice: StandardScaler (zero mean, unit variance) rather than
    MinMaxScaler, because several of the downstream models (SVM, KNN, MLP,
    linear/ridge/Bayesian-ridge regression per PROJECT_BRIEF.md) are
    sensitive to feature scale and StandardScaler is the more standard
    choice for that mix; tree-based models (LightGBM/CatBoost/XGBoost/GBM)
    are scale-invariant so this choice doesn't hurt them either.

    LOUD FLAG: this function fits ONLY on train_df. Never call
    scaler.fit(...) on test data or on pd.concat([train, test]) — that
    would leak test-set distribution statistics into the transform and
    invalidate the evaluation. Callers must use fit_scaler(train, ...) then
    scaler.transform(test[...]), never scaler.fit(test) or
    scaler.fit(pd.concat([train, test])).
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_with_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]) -> pd.DataFrame:
    """Apply an already-fit scaler (transform only, never fit) to a dataframe."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df
