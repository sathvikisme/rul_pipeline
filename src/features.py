"""
features.py — RUL-Bench data-engineer subagent

Feature engineering for C-MAPSS FD001:
  1. Variance-threshold check on the 21 raw sensor channels (computed on
     TRAIN ONLY, never on test or train+test combined) and drop near-constant
     ones, with the actual variance number for every sensor reported.
  2. Optional rolling-window statistics per sensor (implemented, off by
     default in the main pipeline — see note near add_rolling_features).

Also contains the end-to-end orchestration (__main__) that ties together
preprocessing.py + this module and writes data/processed/{train,test}.csv
plus data/processed/DATA_DICTIONARY.md.

Random seed: none needed — variance computation and rolling stats are
deterministic.
"""

from __future__ import annotations

import pandas as pd

import preprocessing as prep

# Variance threshold for dropping near-constant sensor channels. Chosen
# after actually inspecting the FD001 train-set per-sensor variance (see
# variance_threshold_report / the run in __main__): there is a natural
# ~3-orders-of-magnitude gap in this data between the near-constant sensors
# (max variance ~1.93e-06, sensor_6) and the next-lowest genuinely-varying
# sensor (sensor_15, variance ~1.41e-03). A threshold of 1e-5 sits cleanly
# in that gap, so the cutoff is not sensitive to its exact value.
VARIANCE_THRESHOLD = 1e-5


def variance_threshold_report(train_df: pd.DataFrame, sensor_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute variance of each sensor channel on the TRAINING split only.

    Returns a DataFrame with columns [sensor, variance, n_unique, dropped]
    sorted by variance ascending. Computed on train only (never test/combined)
    so the feature-selection decision itself doesn't leak test-set
    information, consistent with the fit-on-train-only rule used for
    scaling.
    """
    if sensor_cols is None:
        sensor_cols = prep.SENSOR_COLS
    var = train_df[sensor_cols].var()
    nunique = train_df[sensor_cols].nunique()
    report = pd.DataFrame({
        "sensor": sensor_cols,
        "variance": [var[c] for c in sensor_cols],
        "n_unique": [nunique[c] for c in sensor_cols],
    })
    report["dropped"] = report["variance"] < VARIANCE_THRESHOLD
    report = report.sort_values("variance").reset_index(drop=True)
    return report


def drop_low_variance_sensors(df: pd.DataFrame, dropped_sensors: list[str]) -> pd.DataFrame:
    """Drop the given (already-identified-on-train) sensor columns from df."""
    return df.drop(columns=[c for c in dropped_sensors if c in df.columns])


def add_rolling_features(df: pd.DataFrame, sensor_cols: list[str], window: int = 5) -> pd.DataFrame:
    """OPTIONAL rolling-window statistics (mean, std, slope) per sensor,
    computed per-engine so windows never cross engine boundaries.

    Implemented because it was straightforward with pandas groupby+rolling,
    but NOT applied in the default processed output (see __main__) — kept
    opt-in so the baseline handoff to model-trainer has a simple, well-
    understood schema first. model-trainer (or a future data-engineer pass)
    can call this directly on data/processed/{train,test}.csv if rolling
    features turn out to help.

    "slope" here is a simple finite-difference approximation
    (last value in window - first value in window) / (window - 1), not a
    full rolling linear regression — cheap and adequate as a first pass.
    """
    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"])
    grouped = df.groupby("engine_id")
    for col in sensor_cols:
        roll = grouped[col].rolling(window=window, min_periods=1)
        df[f"{col}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_roll_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)

        def _slope(s: pd.Series) -> pd.Series:
            first = s.rolling(window=window, min_periods=1).apply(lambda x: x[0], raw=True)
            last = s.rolling(window=window, min_periods=1).apply(lambda x: x[-1], raw=True)
            n = s.rolling(window=window, min_periods=1).count()
            denom = (n - 1).replace(0, 1)
            return (last - first) / denom

        df[f"{col}_roll_slope"] = grouped[col].transform(_slope)
    return df


# ---------------------------------------------------------------------------
# End-to-end pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(raw_dir: str = "data/raw", out_dir: str = "data/processed") -> dict:
    """Run the full data-engineer pipeline end to end and write outputs.

    Order (deliberate):
      1. Load raw train/test (handles the trailing-NaN-column quirk).
      2. Data-quality check (missing/negative/duplicate) — impute only if
         something is actually found.
      3. Compute RUL labels (train: max-cycle-per-engine; test: back-computed
         from RUL_FD001.txt) and apply the piecewise-linear cap (RUL_CAP=125,
         see preprocessing.py for rationale).
      4. Variance-threshold sensor selection, computed on TRAIN ONLY, applied
         to both train and test.
      5. StandardScaler fit on TRAIN ONLY (post feature-selection, so the
         scaler only ever sees the surviving sensor+op-setting columns),
         transform applied to train and test.
      6. Write data/processed/train.csv, data/processed/test.csv, and
         data/processed/DATA_DICTIONARY.md.

    Returns a dict of stats actually computed during this run (row counts,
    variance report, quality-check results) for the caller to print/report —
    nothing here is a placeholder; every value comes from this execution.
    """
    import os

    train_path = os.path.join(raw_dir, "train_FD001.txt")
    test_path = os.path.join(raw_dir, "test_FD001.txt")
    rul_path = os.path.join(raw_dir, "RUL_FD001.txt")

    # --- 1. Load ---
    train_raw = prep.load_raw(train_path)
    test_raw = prep.load_raw(test_path)
    rul_final = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])["RUL"]

    # --- 2. Data quality check + conditional imputation ---
    train_quality = prep.check_data_quality(train_raw, "train_FD001")
    test_quality = prep.check_data_quality(test_raw, "test_FD001")
    train_clean = prep.impute_if_needed(train_raw)
    test_clean = prep.impute_if_needed(test_raw)

    # --- 3. RUL labels + capping ---
    train_labeled = prep.compute_train_rul(train_clean)
    test_labeled = prep.compute_test_rul(test_clean, rul_final)
    train_capped = prep.apply_rul_cap(train_labeled)
    test_capped = prep.apply_rul_cap(test_labeled)

    # --- 4. Variance-threshold sensor selection (train-only decision) ---
    var_report = variance_threshold_report(train_capped, prep.SENSOR_COLS)
    dropped_sensors = var_report.loc[var_report["dropped"], "sensor"].tolist()
    kept_sensors = [c for c in prep.SENSOR_COLS if c not in dropped_sensors]

    train_selected = drop_low_variance_sensors(train_capped, dropped_sensors)
    test_selected = drop_low_variance_sensors(test_capped, dropped_sensors)

    # --- 5. Scale (fit train only, transform train+test) ---
    feature_cols = prep.OP_SETTING_COLS + kept_sensors
    scaler = prep.fit_scaler(train_selected, feature_cols)
    train_final = prep.transform_with_scaler(train_selected, scaler, feature_cols)
    test_final = prep.transform_with_scaler(test_selected, scaler, feature_cols)

    # --- 6. Write outputs ---
    os.makedirs(out_dir, exist_ok=True)
    train_out_path = os.path.join(out_dir, "train.csv")
    test_out_path = os.path.join(out_dir, "test.csv")
    train_final.to_csv(train_out_path, index=False)
    test_final.to_csv(test_out_path, index=False)

    write_data_dictionary(
        out_dir=out_dir,
        feature_cols=feature_cols,
        dropped_sensors=dropped_sensors,
        kept_sensors=kept_sensors,
        var_report=var_report,
        train_quality=train_quality,
        test_quality=test_quality,
        train_shape=train_final.shape,
        test_shape=test_final.shape,
    )

    return {
        "train_quality": train_quality,
        "test_quality": test_quality,
        "variance_report": var_report,
        "dropped_sensors": dropped_sensors,
        "kept_sensors": kept_sensors,
        "train_shape": train_final.shape,
        "test_shape": test_final.shape,
        "train_out_path": train_out_path,
        "test_out_path": test_out_path,
        "scaler_mean_": dict(zip(feature_cols, scaler.mean_)),
        "scaler_scale_": dict(zip(feature_cols, scaler.scale_)),
    }


def write_data_dictionary(out_dir, feature_cols, dropped_sensors, kept_sensors,
                            var_report, train_quality, test_quality, train_shape, test_shape):
    import os
    lines = []
    lines.append("# data/processed — Data Dictionary\n")
    lines.append(
        "Generated by `src/preprocessing.py` + `src/features.py` (data-engineer "
        "subagent) for RUL-Bench, reproducing/extending Özcan (2025, Scientific "
        "Reports 15, 39795) on NASA C-MAPSS FD001. This file documents every "
        "column in `train.csv` / `test.csv` so the model-trainer subagent can "
        "load these files without needing to know preprocessing internals.\n"
    )
    lines.append("## Files\n")
    lines.append(f"- `train.csv` — shape {train_shape[0]} rows x {train_shape[1]} cols\n")
    lines.append(f"- `test.csv` — shape {test_shape[0]} rows x {test_shape[1]} cols\n")
    lines.append("\n## Columns\n")
    lines.append("| Column | Meaning | Notes |\n|---|---|---|\n")
    lines.append("| `engine_id` | Engine/unit number (1-100 per split) | Identifier, not a model feature |\n")
    lines.append("| `cycle` | Operational cycle number for this row | Identifier/time index, not scaled |\n")
    for c in prep.OP_SETTING_COLS:
        lines.append(f"| `{c}` | Operational setting {c[-1]} | StandardScaler-transformed (train-fit) |\n")
    for c in kept_sensors:
        lines.append(f"| `{c}` | Raw sensor measurement {c.split('_')[1]} (see NASA readme.txt) | StandardScaler-transformed (train-fit); kept — variance above threshold |\n")
    lines.append("| `RUL` | Remaining Useful Life label (target) | Piecewise-linear capped at "
                  f"{prep.RUL_CAP} cycles; NOT scaled |\n")
    lines.append("\n## Dropped sensor channels (near-constant, variance-threshold check)\n")
    lines.append(
        f"Variance threshold = {VARIANCE_THRESHOLD:.0e}, computed on **train split only** "
        "(see features.py:variance_threshold_report). Full per-sensor variance from "
        "the actual run:\n\n"
    )
    lines.append("| Sensor | Variance (train) | n_unique (train) | Dropped |\n|---|---|---|---|\n")
    for _, row in var_report.iterrows():
        lines.append(f"| `{row['sensor']}` | {row['variance']:.6e} | {int(row['n_unique'])} | {'YES' if row['dropped'] else 'no'} |\n")
    lines.append(f"\nDropped: {', '.join(dropped_sensors)} ({len(dropped_sensors)} of 21).\n")
    lines.append(f"Kept: {', '.join(kept_sensors)} ({len(kept_sensors)} of 21).\n")
    lines.append("\n## Preprocessing decisions summary\n")
    lines.append(f"- **Missing values**: train had {train_quality['n_missing_values']} NaNs, "
                  f"test had {test_quality['n_missing_values']} NaNs (checked directly, not assumed) "
                  "-> no imputation was applied to either split (see preprocessing.py:impute_if_needed, "
                  "a documented no-op in this case).\n")
    lines.append(f"- **RUL capping**: piecewise-linear cap at RUL={prep.RUL_CAP} cycles "
                  "(literature convention, e.g. Heimes 2008 and Özcan 2025 — see preprocessing.py "
                  "module docstring for the reasoning, chosen deliberately not hardcoded silently).\n")
    lines.append("- **Scaling**: `sklearn.preprocessing.StandardScaler`, `fit()` called on the "
                  "TRAINING split only (post feature-selection), `transform()` applied to both "
                  "train and test. Test was never used to fit the scaler.\n")
    lines.append("- **Feature selection**: variance-threshold on the 21 raw sensor channels, "
                  "computed on train only; op-setting columns and `cycle`/`engine_id` were not "
                  "subject to this check.\n")
    lines.append("- **Rolling-window features**: implemented in features.py:add_rolling_features "
                  "(mean/std/slope per sensor per engine) but NOT applied by default — opt-in "
                  "stretch feature, not present in train.csv/test.csv as shipped.\n")
    lines.append("- **Random seed**: none required in this stage — no stochastic steps.\n")

    with open(os.path.join(out_dir, "DATA_DICTIONARY.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    results = run_pipeline()

    print("=" * 70)
    print("RUL-Bench data-engineer pipeline — run summary")
    print("=" * 70)
    print(f"train quality check: {results['train_quality']}")
    print(f"test quality check:  {results['test_quality']}")
    print()
    print("Per-sensor variance (train-only), sorted ascending:")
    print(results["variance_report"].to_string(index=False))
    print()
    print(f"Dropped sensors ({len(results['dropped_sensors'])}): {results['dropped_sensors']}")
    print(f"Kept sensors ({len(results['kept_sensors'])}): {results['kept_sensors']}")
    print()
    print(f"train.csv shape: {results['train_shape']}  -> {results['train_out_path']}")
    print(f"test.csv shape:  {results['test_shape']}  -> {results['test_out_path']}")
