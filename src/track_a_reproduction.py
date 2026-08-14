"""
track_a_reproduction.py — RUL-Bench leakage-red-team subagent (Phase 2, Track A)

Faithful, standalone reproduction of Özcan (2025)'s own reference
implementation protocol (github.com/hkmtcn/interpretable-rul-maintenance,
notebooks/LGMB_CatBoost.ipynb), confirmed by direct inspection of that
notebook (see PROJECT_BRIEF.md / CLAUDE.md Phase-2 addendum for the full
provenance trail; summarized here so this module is self-contained).

GENUINELY SEPARATE CODE PATH FROM TRACK B. This module:
  - does NOT import src/fold_safe_pipeline.py, src/nested_cv.py,
    src/nested_stacking.py, or src/track_b_pipeline.py;
  - does NOT read data/processed/train.csv or data/processed/test.csv —
    those are Phase 1's/Track B's already-transformed files (variance-
    selected sensors, StandardScaler, RUL capped at 125 on BOTH splits).
    Using them here would not be a faithful reproduction of the paper's own
    feature matrix, which uses ALL 21 raw sensors (no variance filtering),
    engine_id AND cycle as raw model features, MinMaxScaler (not
    StandardScaler), and (per the notebook cells inspected) NO RUL capping
    at all;
  - reuses ONLY src/preprocessing.py's raw-parsing utility (`load_raw`) —
    everything else (RUL computation, feature matrix, splits, models,
    metrics aggregation) is implemented fresh in this file.

CONFIRMED FACTS FROM THE PAPER'S OWN NOTEBOOK (not re-derived, replicated):
  - Headline FD001 result (LightGBM R2=0.9894, CatBoost R2=0.9872,
    Ensemble R2=0.9904, RUL Score~=2951, matching the paper's reported
    RMSE~=6.62 / R2~=0.99) comes from
    `train_test_split(X, y, test_size=0.2, random_state=42)` — a PLAIN
    ROW-LEVEL split — applied to `train_FD001.txt` ONLY (20,631 rows;
    their own split_80_20_proof_FD001.txt confirms train=16504/test=4127).
  - X includes `engine_id` ('id' in their column naming) AND `cycle` as raw
    model features (no exclusion), ALL 21 raw sensors (no variance-
    threshold filtering), 3 op settings, scaled with `MinMaxScaler` (fit on
    the 80% train partition, applied to the 20% test partition) — NOT
    StandardScaler.
  - Fixed (not tuned) hyperparameters — see LGB_PARAMS / CAT_PARAMS below.
    Ensemble = simple 0.5/0.5 average of the two models' predictions.
  - The official NASA test_FD001.txt/RUL_FD001.txt IS loaded and RUL-
    labeled in their notebook but is NEVER passed to `.predict()` anywhere
    — their headline number has nothing to do with the standard
    held-out-engine benchmark. We reproduce that same fact here explicitly
    (loaded, labeled, never scored) rather than silently omitting it.
  - The same notebook's own `GroupKFold(5)` diagnostic on the SAME features
    collapses LightGBM Overall R2 to ~0.4372. An ablation removing
    id/cycle under GroupKFold improves R2 from ~0.4356 to ~0.5964
    (mean+/-std form: All features R2=0.4356+/-0.1078, Without id/cycle
    R2=0.5964+/-0.0463) — evidence `id` (not a coding bug) is the leakage
    vector specific to same-engine-on-both-sides splits. A permutation-null
    test in their code returns R2~=0 as expected, confirming their pipeline
    mechanics are sound (not just "any number in, any positive R2 out").

RUL LABELING — NO CAPPING (explicit discrepancy vs. this repo's own Track B
methodology, flagged rather than silently harmonized): the paper's reference
notebook computes train RUL as `max_cycle_per_engine - cycle` with NO
piecewise-linear capping step anywhere in the cells we inspected. This
DIFFERS from PROJECT_BRIEF.md's discussion of a 125-cycle cap, which this
repo's own Phase 1 preprocessing (`src/preprocessing.py::apply_rul_cap`)
adopts for Track B. Track A follows the paper's literal, uncapped code
because Track A's entire purpose is a *faithful* reproduction of what the
paper's own repository actually does, not this project's own improved
methodology (that's Track B's job). If a discrepancy between our numbers and
the paper's shows up, this uncapped-vs-capped choice is investigated FIRST,
not assumed away.

Random seed: 42 everywhere — train_test_split, LightGBM, CatBoost,
np.random.default_rng for the permutation-null shuffle. GroupKFold itself is
deterministic (contiguous group-order partition) and needs no seed.

Run: `python src/track_a_reproduction.py` from repo root. Writes:
  results/tables/paper_reproduction_metrics.csv
  results/tables/track_a_groupkfold_collapse.csv
  results/tables/track_a_id_cycle_ablation.csv
  results/tables/track_a_permutation_null.csv
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import preprocessing as prep  # noqa: E402 -- ONLY load_raw is reused from this module

REPO_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
TRAIN_RAW_PATH = os.path.join(RAW_DIR, "train_FD001.txt")
TEST_RAW_PATH = os.path.join(RAW_DIR, "test_FD001.txt")
RUL_RAW_PATH = os.path.join(RAW_DIR, "RUL_FD001.txt")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")
PHM08_SCRIPT = os.path.join(REPO_ROOT, ".claude", "skills", "phm08-scoring", "score.py")
PGTS_SCRIPT = os.path.join(REPO_ROOT, ".claude", "skills", "pgts-split", "pgts.py")

os.makedirs(TABLES_DIR, exist_ok=True)


def _load_skill_module(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Skills reused, not reimplemented (CLAUDE.md rule).
phm08_score = _load_skill_module(PHM08_SCRIPT, "phm08_score_module_track_a")
phm08_score = phm08_score.phm08_score
_assert_no_leakage = _load_skill_module(PGTS_SCRIPT, "pgts_module_track_a")._assert_no_leakage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42

# Exact fixed hyperparameters confirmed from the paper's own notebook.
LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.1, max_depth=-1, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1,
)
CAT_PARAMS = dict(
    iterations=500, learning_rate=0.1, depth=6, random_seed=SEED,
    verbose=0, loss_function="RMSE",
)

# The paper's reported headline numbers (Özcan 2025 / their own notebook
# output) — used ONLY as a comparison reference printed alongside our own
# reproduced numbers, never substituted for an actual run.
PAPER_REPORTED_R2 = {"LightGBM": 0.9894, "CatBoost": 0.9872, "Ensemble": 0.9904}


# ---------------------------------------------------------------------------
# Data loading / RUL labeling (own implementation — no compute_train_rul or
# apply_rul_cap reuse from preprocessing.py, deliberately, per the module
# docstring's "uncapped, own RUL computation" requirement).
# ---------------------------------------------------------------------------

def compute_uncapped_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL per row = max_cycle_for_engine - current_cycle. NO capping —
    matches the paper's reference notebook's literal code (see module
    docstring's "RUL LABELING" section)."""
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def load_paper_style_data() -> dict:
    """Load train/test/RUL_FD001 raw files exactly as the paper's notebook
    does. Test + RUL_FD001 are loaded and RUL-labeled for parity with the
    notebook's own cells, but (matching the confirmed fact that the
    notebook never scores them) are NEVER passed to any .predict() call
    anywhere in this module."""
    train_raw = prep.load_raw(TRAIN_RAW_PATH)
    test_raw = prep.load_raw(TEST_RAW_PATH)
    rul_final = pd.read_csv(RUL_RAW_PATH, sep=r"\s+", header=None, names=["RUL"])["RUL"]

    train_labeled = compute_uncapped_train_rul(train_raw)

    # Test RUL labeled for parity/documentation only -- never used downstream
    # in this module. Uses the same back-computation logic as the paper's
    # notebook (final observed RUL + cycles-back-from-end), implemented
    # locally rather than via preprocessing.compute_test_rul to keep this
    # module's RUL logic self-contained and independently auditable.
    engine_ids = sorted(test_raw["engine_id"].unique())
    if len(engine_ids) != len(rul_final):
        raise ValueError(
            f"test engines ({len(engine_ids)}) != len(RUL_FD001.txt) ({len(rul_final)})"
        )
    rul_map = dict(zip(engine_ids, rul_final.values))
    last_cycle = test_raw.groupby("engine_id")["cycle"].transform("max")
    test_labeled = test_raw.copy()
    test_labeled["RUL"] = test_raw["engine_id"].map(rul_map) + (last_cycle - test_raw["cycle"])

    return {
        "train": train_labeled,
        "test_never_scored": test_labeled,
        "rul_final_never_scored": rul_final,
    }


def build_feature_matrix(df: pd.DataFrame, drop_id_cycle: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Paper's exact feature set: engine_id + cycle + 3 op settings + 21 raw
    sensors (prep.ALL_COLS, 26 columns), no variance filtering. `RUL` is the
    target, excluded from X. `drop_id_cycle=True` builds the ablation
    feature set (engine_id/cycle removed) for the id/cycle-removal check."""
    feature_cols = list(prep.ALL_COLS)
    if drop_id_cycle:
        feature_cols = [c for c in feature_cols if c not in ("engine_id", "cycle")]
    X = df[feature_cols].values.astype(float)
    y = df["RUL"].values.astype(float)
    return X, y, feature_cols


def _metrics(y_true, y_pred) -> tuple[float, float, float, float]:
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    score, _ = phm08_score(y_true, y_pred)
    return r2, mse, mae, score


def permute_rul_within_engine(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Shuffle RUL values WITHIN each engine (keeps the set of RUL values
    per engine the same, destroys the cycle<->RUL relationship inside each
    engine) — a permutation-null sanity check: if the pipeline's headline
    R2 reflects a genuine (if leaky) signal rather than a code bug, this
    should collapse R2 toward 0."""
    rng = np.random.default_rng(seed)
    rul = df["RUL"].values.copy()
    for _eid, idx in df.groupby("engine_id").indices.items():
        idx = np.asarray(idx)
        permuted = rul[idx].copy()
        rng.shuffle(permuted)
        rul[idx] = permuted
    df_perm = df.copy()
    df_perm["RUL"] = rul
    return df_perm


# ---------------------------------------------------------------------------
# 1. Headline reproduction: row-level 80/20 split of train_FD001.txt only.
# ---------------------------------------------------------------------------

def run_headline_reproduction(train_df: pd.DataFrame, label: str = "paper_reproduction") -> pd.DataFrame:
    X, y, feature_cols = build_feature_matrix(train_df, drop_id_cycle=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)  # fit on TRAIN split only, per the paper's notebook
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    lgb = LGBMRegressor(**LGB_PARAMS)
    lgb.fit(X_train_s, y_train)
    pred_lgb = lgb.predict(X_test_s)

    cat = CatBoostRegressor(**CAT_PARAMS)
    cat.fit(X_train_s, y_train)
    pred_cat = cat.predict(X_test_s)

    pred_ens = 0.5 * pred_lgb + 0.5 * pred_cat

    rows = []
    for name, preds in [("LightGBM", pred_lgb), ("CatBoost", pred_cat), ("Ensemble", pred_ens)]:
        r2, mse, mae, score = _metrics(y_test, preds)
        rows.append({
            "model": name,
            f"{label}_R2": r2,
            f"{label}_MSE": mse,
            f"{label}_MAE": mae,
            f"{label}_PHM08_RUL_Score": score,
            "paper_reported_R2": PAPER_REPORTED_R2[name],
            "R2_gap_vs_paper": r2 - PAPER_REPORTED_R2[name],
            "n_train_rows": len(y_train),
            "n_test_rows": len(y_test),
            "split_protocol": "row_level_train_test_split_80_20_random_state_42_TRAIN_FD001_ONLY",
            "feature_set": "engine_id+cycle+3_op_settings+21_raw_sensors_MinMaxScaler_no_variance_filter",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2/3. GroupKFold(5) diagnostic — same features, then id/cycle ablation.
# ---------------------------------------------------------------------------

def run_groupkfold_diagnostic(train_df: pd.DataFrame, drop_id_cycle: bool, prefix: str) -> pd.DataFrame:
    X, y, feature_cols = build_feature_matrix(train_df, drop_id_cycle=drop_id_cycle)
    groups = train_df["engine_id"].values

    gkf = GroupKFold(n_splits=5)
    rows = []
    pooled_true = {"LightGBM": [], "CatBoost": []}
    pooled_pred = {"LightGBM": [], "CatBoost": []}

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        _assert_no_leakage(groups, tr_idx, te_idx)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        scaler = MinMaxScaler()
        scaler.fit(X_tr)  # fold-local fit -- no leakage of held-out engines
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lgb = LGBMRegressor(**LGB_PARAMS)
        lgb.fit(X_tr_s, y_tr)
        pred_lgb = lgb.predict(X_te_s)

        cat = CatBoostRegressor(**CAT_PARAMS)
        cat.fit(X_tr_s, y_tr)
        pred_cat = cat.predict(X_te_s)

        for name, preds in [("LightGBM", pred_lgb), ("CatBoost", pred_cat)]:
            r2, mse, mae, score = _metrics(y_te, preds)
            rows.append({
                "model": name, "fold": fold_idx,
                f"{prefix}_R2": r2, f"{prefix}_MSE": mse, f"{prefix}_MAE": mae,
                f"{prefix}_PHM08_RUL_Score": score,
                "n_train_rows": len(tr_idx), "n_test_rows": len(te_idx),
                "n_train_engines": int(len(np.unique(groups[tr_idx]))),
                "n_test_engines": int(len(np.unique(groups[te_idx]))),
            })
            pooled_true[name].append(y_te)
            pooled_pred[name].append(preds)

    # Pooled/"Overall" R2 across all 5 folds' concatenated OOF predictions --
    # the same quantity the paper's notebook reports as "Overall R2" for its
    # GroupKFold diagnostic.
    for name in ["LightGBM", "CatBoost"]:
        yt = np.concatenate(pooled_true[name])
        yp = np.concatenate(pooled_pred[name])
        r2, mse, mae, score = _metrics(yt, yp)
        rows.append({
            "model": name, "fold": "overall_pooled",
            f"{prefix}_R2": r2, f"{prefix}_MSE": mse, f"{prefix}_MAE": mae,
            f"{prefix}_PHM08_RUL_Score": score,
            "n_train_rows": None, "n_test_rows": len(yt),
            "n_train_engines": None, "n_test_engines": None,
        })

    df_out = pd.DataFrame(rows)
    df_out["feature_set"] = "id_cycle_dropped" if drop_id_cycle else "id_cycle_included"
    return df_out


# ---------------------------------------------------------------------------
# 4. Permutation-null sanity check.
# ---------------------------------------------------------------------------

def run_permutation_null(train_df: pd.DataFrame, label: str = "track_a_permutation_null") -> pd.DataFrame:
    df_perm = permute_rul_within_engine(train_df, seed=SEED)
    metrics_df = run_headline_reproduction(df_perm, label=label)
    metrics_df["protocol"] = "row_level_80_20_split_RUL_permuted_within_engine"
    metrics_df = metrics_df.drop(columns=["paper_reported_R2", "R2_gap_vs_paper"])
    return metrics_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("[track_a_reproduction] loading raw FD001 data (train_FD001.txt only feeds "
          "any model.fit()/predict() call in this module)")
    print("=" * 78)
    data = load_paper_style_data()
    train_df = data["train"]
    print(f"train rows={len(train_df)} engines={train_df['engine_id'].nunique()} "
          f"RUL range=[{train_df['RUL'].min()}, {train_df['RUL'].max()}] (uncapped, per paper's code)")
    print(f"test rows={len(data['test_never_scored'])} engines="
          f"{data['test_never_scored']['engine_id'].nunique()} -- loaded/labeled for parity "
          f"with the paper's notebook, NEVER passed to .predict() anywhere in this module.")

    print("\n" + "=" * 78)
    print("[track_a_reproduction] STEP 1/4 -- headline reproduction "
          "(row-level 80/20 split, train_FD001.txt only)")
    print("=" * 78)
    headline_df = run_headline_reproduction(train_df, label="paper_reproduction")
    headline_df.to_csv(os.path.join(TABLES_DIR, "paper_reproduction_metrics.csv"), index=False)
    print(headline_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("[track_a_reproduction] STEP 2/4 -- GroupKFold(5) collapse check "
          "(same features: id+cycle+settings+sensors)")
    print("=" * 78)
    gkf_full_df = run_groupkfold_diagnostic(train_df, drop_id_cycle=False, prefix="track_a_groupkfold")
    gkf_full_df.to_csv(os.path.join(TABLES_DIR, "track_a_groupkfold_collapse.csv"), index=False)
    print(gkf_full_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("[track_a_reproduction] STEP 3/4 -- id/cycle-removal ablation under GroupKFold(5)")
    print("=" * 78)
    gkf_ablation_df = run_groupkfold_diagnostic(train_df, drop_id_cycle=True, prefix="track_a_ablation")
    gkf_ablation_df.to_csv(os.path.join(TABLES_DIR, "track_a_id_cycle_ablation.csv"), index=False)
    print(gkf_ablation_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("[track_a_reproduction] STEP 4/4 -- permutation-null sanity check "
          "(RUL shuffled within engine, headline row-level 80/20 protocol)")
    print("=" * 78)
    perm_df = run_permutation_null(train_df)
    perm_df.to_csv(os.path.join(TABLES_DIR, "track_a_permutation_null.csv"), index=False)
    print(perm_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("[track_a_reproduction] SUMMARY")
    print("=" * 78)
    for _, row in headline_df.iterrows():
        print(f"  {row['model']:10s} headline R2={row['paper_reproduction_R2']:.4f}  "
              f"paper-reported R2={row['paper_reported_R2']:.4f}  "
              f"gap={row['R2_gap_vs_paper']:+.4f}")
    overall_full = gkf_full_df[gkf_full_df["fold"] == "overall_pooled"]
    overall_ablation = gkf_ablation_df[gkf_ablation_df["fold"] == "overall_pooled"]
    for _, row in overall_full.iterrows():
        print(f"  {row['model']:10s} GroupKFold(5) overall R2={row['track_a_groupkfold_R2']:.4f} "
              f"(paper's own notebook reports ~0.4372 for LightGBM)")
    for _, row in overall_ablation.iterrows():
        print(f"  {row['model']:10s} GroupKFold(5) id/cycle-dropped overall R2="
              f"{row['track_a_ablation_R2']:.4f} (paper's own notebook reports ~0.5964 recovery)")
    for _, row in perm_df.iterrows():
        print(f"  {row['model']:10s} permutation-null R2={row['track_a_permutation_null_R2']:.4f} "
              f"(expected ~0 if pipeline mechanics are sound)")
    print("=" * 78)
    print(f"[track_a_reproduction] wrote paper_reproduction_metrics.csv, "
          f"track_a_groupkfold_collapse.csv, track_a_id_cycle_ablation.csv, "
          f"track_a_permutation_null.csv -> {TABLES_DIR}")


if __name__ == "__main__":
    main()
