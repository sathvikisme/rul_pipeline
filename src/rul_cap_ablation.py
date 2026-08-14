"""
rul_cap_ablation.py — RUL-Bench model-trainer subagent (Phase 2, Issue 7 ablation)

Runs `nested_cv.py`'s Tier-1-only protocol (LightGBM, CatBoost, XGBoost --
cost-bounded per the Phase 2 plan, NOT all 10 models) across the 3 parallel
RUL-cap variants data-engineer built in `data/processed/rul_cap_ablation/`:

    A : train RUL capped at 125,   test RUL uncapped   (only train.csv used here)
    B : train RUL capped at 125,   test RUL capped      (matches Phase 1 exactly; only train.csv used here)
    C : train RUL uncapped,        test RUL uncapped   (only train.csv used here)

Only each variant's `train.csv` is read -- this ablation is itself a
nested-CV protocol over engine-held-out folds of the TRAINING data (same as
nested_cv.py generally), so no variant's test.csv is ever touched.

IMPORTANT CAVEAT, already flagged in data/processed/rul_cap_ablation/MANIFEST.md
and repeated here because it changes how to read this ablation's own output:
Variant A's train.csv and Variant B's train.csv are BYTE-IDENTICAL (both are
"capped at 125" on the train side -- the A vs B distinction is entirely
about whether the *test* RUL is capped, which this train-only ablation never
observes). So this ablation's A and B rows are expected to be numerically
identical to each other (same data, same seed, same protocol) -- that is
NOT a bug, it's the correct, honest consequence of an ablation that (by
design, to stay inside the "never touch test.csv before the freeze
checkpoint" rule) only ever sees each variant's training split. Variant C
(train RUL uncapped, range roughly 0-361 instead of 0-125) is the only row
that differs mechanically from A/B here.

A second, orthogonal caveat (from the Phase 2 plan and MANIFEST.md): PHM08
score and R2/MSE are only meaningfully comparable WITHIN a cap regime (A vs
A, B vs B, C vs C) -- capping changes the scale/skew of the RUL target
itself, so a raw R2 or PHM08 comparison across variant C (uncapped) vs A/B
(capped) conflates "the model got better" with "the target distribution
changed." This is recorded as an explicit column in the output CSV, not
just prose here.

Output: results/tables/ablation_matrix.csv, one row per (variant, model)
with R2/MSE/MAE/PHM08 mean+std across the 5 outer folds.

Run: `python src/rul_cap_ablation.py` from repo root.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import nested_cv as ncv  # noqa: E402

REPO_ROOT = ncv.REPO_ROOT
TABLES_DIR = ncv.TABLES_DIR
ABLATION_DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "rul_cap_ablation")

VARIANT_TRAIN_PATHS = {
    "A": os.path.join(ABLATION_DATA_DIR, "A", "train.csv"),
    "B": os.path.join(ABLATION_DATA_DIR, "B", "train.csv"),
    "C": os.path.join(ABLATION_DATA_DIR, "C", "train.csv"),
}

CROSS_VARIANT_CAVEAT = (
    "R2/MSE/MAE/PHM08 are only directly comparable WITHIN a cap regime "
    "(A vs A, B vs B, C vs C). Comparing across variants (e.g. A vs C) "
    "conflates model quality with a change in the RUL target's scale/skew "
    "caused by capping. Variant A and B rows are numerically identical here "
    "because this ablation only reads each variant's train.csv, and A/B "
    "share the same (capped) train-side RUL -- their only difference is "
    "test-side capping, which this train-only nested-CV ablation never sees."
)


def run_ablation(models: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    if models is None:
        models = ncv.TIER1_MODELS

    rows = []
    variant_wall_seconds = {}

    for variant, path in VARIANT_TRAIN_PATHS.items():
        if verbose:
            print("\n" + "#" * 78)
            print(f"[rul_cap_ablation] variant {variant}: {path}")
            print("#" * 78)
        t0 = time.time()
        result = ncv.run_nested_cv(train_path=path, models=models, verbose=verbose)
        elapsed = time.time() - t0
        variant_wall_seconds[variant] = elapsed

        fm = result["fold_metrics_df"]
        agg = fm.groupby("model").agg(
            R2_mean=("R2", "mean"), R2_std=("R2", "std"),
            MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
            MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
            PHM08_mean=("PHM08_RUL_Score", "mean"), PHM08_std=("PHM08_RUL_Score", "std"),
        ).reset_index()
        agg.insert(0, "variant", variant)
        agg["variant_wall_seconds"] = elapsed
        agg["cross_variant_comparability_caveat"] = CROSS_VARIANT_CAVEAT
        rows.append(agg)

        if verbose:
            print(f"[rul_cap_ablation] variant {variant} done in {elapsed/60:.2f} min")

    matrix = pd.concat(rows, ignore_index=True).sort_values(["model", "variant"]).reset_index(drop=True)

    total_min = sum(variant_wall_seconds.values()) / 60.0
    if verbose:
        print("\n" + "=" * 78)
        print(f"[rul_cap_ablation] TOTAL wall time across all 3 variants: {total_min:.2f} min")
        if total_min > 25.0:
            print(f"[rul_cap_ablation] WARNING: total wall time {total_min:.2f} min exceeds the "
                  f"~15-25 min Phase-2-plan budget estimate for this ablation.")
        print("=" * 78)

    return matrix


def write_outputs(matrix: pd.DataFrame, tables_dir: str = TABLES_DIR) -> None:
    out_path = os.path.join(tables_dir, "ablation_matrix.csv")
    matrix.to_csv(out_path, index=False)
    print(f"[rul_cap_ablation] wrote ablation_matrix.csv -> {out_path}")


if __name__ == "__main__":
    matrix = run_ablation(models=ncv.TIER1_MODELS, verbose=True)
    write_outputs(matrix)
    print("\n[rul_cap_ablation] ablation_matrix.csv contents:")
    print(matrix.drop(columns=["cross_variant_comparability_caveat"]).to_string(index=False))
