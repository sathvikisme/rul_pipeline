"""
rul_cap_variants.py — RUL-Bench data-engineer subagent (Phase 2)

Builds three parallel RUL-labeled datasets from the SAME raw C-MAPSS FD001
files, differing ONLY in how the piecewise-linear RUL cap is applied to the
train vs. test target:

    Variant A — train-cap-only : train RUL capped at 125, test RUL RAW/uncapped
    Variant B — train+test-cap : both capped at 125 (== current Phase 1 behavior,
                                  data/processed/train.csv & test.csv)
    Variant C — no-cap         : neither train nor test RUL capped

Why this exists: Phase 1's `features.py::run_pipeline` applies
`preprocessing.py::apply_rul_cap` to BOTH train and test (Variant B). Capping
the TEST set's ground-truth RUL is methodologically debatable — it means a
model is never evaluated on its ability to predict "this engine has 300
cycles of life left," only "this engine has >=125 cycles left, capped to
look identical to one with exactly 125." Variant A (test uncapped) tests
against the true remaining life. Variant C (nothing capped) is the fully
uncapped ablation baseline. This module does not decide which variant is
"correct" — that's an empirical question for model-trainer's
`rul_cap_ablation.py` to answer; this module just builds the three datasets
honestly so that experiment is possible.

Reused, not reimplemented: `load_raw`, `compute_train_rul`, `compute_test_rul`,
and `apply_rul_cap` are imported directly from `src/preprocessing.py` (same
module Phase 1 used) — this file does not redefine RUL-computation logic.
The cap VALUE (125) is also reused from `preprocessing.RUL_CAP`, not
re-hardcoded here, so the two can never silently drift apart. See
preprocessing.py's module/RUL_CAP docstring for the full rationale for why
125 was chosen (Heimes 2008 convention, also used by Özcan 2025).

Feature scope (deliberate, per Phase 2 plan): op settings + ALL 21 raw
sensor channels, UNSCALED, with NO variance-threshold feature selection
applied here. Phase 1 computed variance-threshold selection once, offline,
on the full train.csv before any CV ran — that's a mild leakage source this
Phase 2 effort is repairing elsewhere (see `src/fold_safe_pipeline.py`,
`VarianceThreshold` as a fold-fit Pipeline step). So this module intentionally
ships the full unfiltered, unscaled raw feature set; feature selection and
scaling both happen fold-safely, later, inside Track B's `nested_cv.py` — NOT
here at data-prep time. This also has the useful side effect of making it
trivial to verify A/B/C's feature columns are byte-identical (no per-variant
selection decision that could differ).

Random seed: none needed — every transform here is deterministic (identical
to preprocessing.py's own no-seed-needed rationale).

Outputs:
    data/processed/rul_cap_ablation/A/{train,test}.csv
    data/processed/rul_cap_ablation/B/{train,test}.csv
    data/processed/rul_cap_ablation/C/{train,test}.csv
    data/processed/rul_cap_ablation/MANIFEST.md
"""

from __future__ import annotations

import os
import sys

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import preprocessing as prep  # noqa: E402  (load_raw, compute_train_rul, compute_test_rul, apply_rul_cap, RUL_CAP)

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "rul_cap_ablation")

# Full raw feature schema shared identically by every variant: identifiers
# + op settings + all 21 sensors + RUL target. No dropping, no scaling.
FEATURE_COLS = prep.ID_COLS + prep.OP_SETTING_COLS + prep.SENSOR_COLS
OUTPUT_COLS = FEATURE_COLS + ["RUL"]


def _load_labeled_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train/test and attach UNCAPPED RUL labels (reusing
    preprocessing.py's load/compute functions verbatim). Capping is applied
    per-variant by the caller, never inside this helper."""
    train_path = os.path.join(RAW_DIR, "train_FD001.txt")
    test_path = os.path.join(RAW_DIR, "test_FD001.txt")
    rul_path = os.path.join(RAW_DIR, "RUL_FD001.txt")

    train_raw = prep.load_raw(train_path)
    test_raw = prep.load_raw(test_path)
    rul_final = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])["RUL"]

    train_labeled = prep.compute_train_rul(train_raw)
    test_labeled = prep.compute_test_rul(test_raw, rul_final)
    return train_labeled[OUTPUT_COLS], test_labeled[OUTPUT_COLS]


def build_variants() -> dict:
    """Build Variant A / B / C train+test dataframes.

    Returns a dict: {"A": (train_df, test_df), "B": (...), "C": (...)}
    Each train_df/test_df has columns OUTPUT_COLS (26 identifier/feature
    columns + RUL). Only the RUL column's cap treatment differs between
    variants and between train/test within a variant; every feature column
    is byte-identical to the raw (uncapped-source) load across all three.
    """
    train_uncapped, test_uncapped = _load_labeled_raw()
    cap = prep.RUL_CAP  # reused, not re-hardcoded — see module docstring

    train_capped = prep.apply_rul_cap(train_uncapped, cap=cap)
    test_capped = prep.apply_rul_cap(test_uncapped, cap=cap)

    variants = {
        # A: train-cap-only -- test RUL left raw/uncapped
        "A": (train_capped.copy(), test_uncapped.copy()),
        # B: train+test-cap -- matches current Phase 1 data/processed/*.csv exactly
        "B": (train_capped.copy(), test_capped.copy()),
        # C: no-cap -- neither split capped
        "C": (train_uncapped.copy(), test_uncapped.copy()),
    }
    return variants, cap


def write_variants(variants: dict, out_dir: str = OUT_DIR) -> dict:
    """Write each variant's train.csv/test.csv, return actual on-disk stats
    (row counts, RUL ranges) computed from what was actually written — no
    placeholders."""
    stats = {}
    for label, (train_df, test_df) in variants.items():
        variant_dir = os.path.join(out_dir, label)
        os.makedirs(variant_dir, exist_ok=True)
        train_path = os.path.join(variant_dir, "train.csv")
        test_path = os.path.join(variant_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        stats[label] = {
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "train_RUL_min": float(train_df["RUL"].min()),
            "train_RUL_max": float(train_df["RUL"].max()),
            "test_RUL_min": float(test_df["RUL"].min()),
            "test_RUL_max": float(test_df["RUL"].max()),
            "train_path": train_path,
            "test_path": test_path,
        }
    return stats


def verify_feature_columns_identical(variants: dict) -> dict:
    """Confirm feature columns (everything except RUL) are byte-identical
    across A/B/C for train and, separately, for test -- the only thing that
    should ever differ between variants is the RUL column's cap treatment.

    Also confirms the specific pairwise equalities implied by the variant
    definitions:
      - train_A == train_B (both cap train the same way)
      - test_A == test_C (neither caps test)
    """
    feat_cols = FEATURE_COLS
    train_A, test_A = variants["A"]
    train_B, test_B = variants["B"]
    train_C, test_C = variants["C"]

    checks = {
        "train_feature_cols_A_vs_B": train_A[feat_cols].equals(train_B[feat_cols]),
        "train_feature_cols_A_vs_C": train_A[feat_cols].equals(train_C[feat_cols]),
        "test_feature_cols_A_vs_B": test_A[feat_cols].equals(test_B[feat_cols]),
        "test_feature_cols_A_vs_C": test_A[feat_cols].equals(test_C[feat_cols]),
        # RUL-column implications of the variant definitions themselves:
        "train_RUL_A_equals_B (both capped)": train_A["RUL"].equals(train_B["RUL"]),
        "test_RUL_A_equals_C (both uncapped)": test_A["RUL"].equals(test_C["RUL"]),
        "train_RUL_A_equals_C (should be False -- A capped, C not)": train_A["RUL"].equals(train_C["RUL"]),
        "test_RUL_A_equals_B (should be False -- A uncapped, B capped)": test_A["RUL"].equals(test_B["RUL"]),
    }
    return checks


def write_manifest(stats: dict, cap: int, checks: dict, out_dir: str = OUT_DIR) -> str:
    lines = []
    lines.append("# data/processed/rul_cap_ablation — MANIFEST\n\n")
    lines.append(
        "Three parallel RUL-labeled datasets built from the SAME raw NASA "
        "C-MAPSS FD001 files (`data/raw/train_FD001.txt`, `test_FD001.txt`, "
        "`RUL_FD001.txt`) by `src/rul_cap_variants.py` (data-engineer "
        "subagent, Phase 2). **The only difference between variants A/B/C is "
        "the RUL target's cap treatment — every feature column "
        "(engine_id, cycle, op_setting_1-3, sensor_1-21) is identical in "
        "value across all three variants**, verified programmatically at "
        "build time (see checks below), not just claimed.\n\n"
    )
    lines.append(
        f"RUL cap value used where capping is applied: **{cap}** cycles, reused "
        "directly from `preprocessing.RUL_CAP` (not re-hardcoded) — same value "
        "and same rationale as Phase 1 (Heimes 2008 plateau convention, also "
        "used by Özcan 2025; see `src/preprocessing.py` module docstring).\n\n"
    )
    lines.append("## Feature scope\n\n")
    lines.append(
        "Op settings (3) + ALL 21 raw sensor channels, UNSCALED, with NO "
        "variance-threshold feature selection applied (unlike Phase 1's "
        "data/processed/train.csv, which drops 7 near-constant sensors and "
        "z-score-scales the rest). Feature selection and scaling are applied "
        "fold-safely later, inside Track B's `nested_cv.py` / "
        "`src/fold_safe_pipeline.py`, not at data-prep time — see that "
        "module's docstring for why.\n\n"
    )
    lines.append("## Variants\n\n")
    lines.append("| Variant | Train RUL | Test RUL | Notes |\n|---|---|---|---|\n")
    lines.append(f"| A | capped at {cap} | **raw / uncapped** | tests against true remaining life |\n")
    lines.append(f"| B | capped at {cap} | capped at {cap} | matches current Phase 1 `data/processed/{{train,test}}.csv` exactly |\n")
    lines.append("| C | raw / uncapped | raw / uncapped | fully uncapped baseline |\n\n")

    lines.append("## Actual row counts / RUL ranges from this run\n\n")
    lines.append("| Variant | train shape | train RUL range | test shape | test RUL range |\n|---|---|---|---|---|\n")
    for label in ("A", "B", "C"):
        s = stats[label]
        lines.append(
            f"| {label} | {s['train_shape'][0]} x {s['train_shape'][1]} | "
            f"[{s['train_RUL_min']:.0f}, {s['train_RUL_max']:.0f}] | "
            f"{s['test_shape'][0]} x {s['test_shape'][1]} | "
            f"[{s['test_RUL_min']:.0f}, {s['test_RUL_max']:.0f}] |\n"
        )

    lines.append("\n## Feature-column identity checks (actually run, not assumed)\n\n")
    lines.append("| Check | Result |\n|---|---|\n")
    for name, val in checks.items():
        lines.append(f"| {name} | {val} |\n")

    lines.append(
        "\n## Files\n\n"
        "```\n"
        "data/processed/rul_cap_ablation/\n"
        "  A/train.csv  A/test.csv\n"
        "  B/train.csv  B/test.csv\n"
        "  C/train.csv  C/test.csv\n"
        "  MANIFEST.md  (this file)\n"
        "```\n\n"
        "Columns in every train.csv/test.csv: `engine_id, cycle, op_setting_1, "
        "op_setting_2, op_setting_3, sensor_1..sensor_21, RUL` (26 raw feature "
        "columns + RUL target, unscaled, unselected).\n\n"
        "Caveat for downstream consumers (per Phase 2 plan §6): PHM08 score "
        "and R2/MSE are only meaningfully comparable **within** a cap regime "
        "(A vs A, B vs B, C vs C) — comparing raw metric values ACROSS "
        "variants conflates \"model got better\" with \"target distribution "
        "changed,\" since capping directly changes the scale/skew of the "
        "target.\n"
    )

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "MANIFEST.md")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return manifest_path


if __name__ == "__main__":
    print("=" * 78)
    print("rul_cap_variants.py — building RUL-cap ablation datasets (A/B/C)")
    print("=" * 78)

    variants, cap = build_variants()
    print(f"RUL cap value in use (reused from preprocessing.RUL_CAP): {cap}")

    stats = write_variants(variants)
    checks = verify_feature_columns_identical(variants)

    print("\nPer-variant actual shapes / RUL ranges:")
    for label in ("A", "B", "C"):
        s = stats[label]
        print(f"  Variant {label}: train {s['train_shape']} RUL[{s['train_RUL_min']:.0f},{s['train_RUL_max']:.0f}]  "
              f"test {s['test_shape']} RUL[{s['test_RUL_min']:.0f},{s['test_RUL_max']:.0f}]")
        print(f"      -> {s['train_path']}")
        print(f"      -> {s['test_path']}")

    print("\nFeature-column identity checks:")
    all_expected = True
    for name, val in checks.items():
        expect_true = "should be False" not in name
        ok = (val == expect_true)
        all_expected &= ok
        print(f"  [{'OK' if ok else 'UNEXPECTED'}] {name}: {val}")

    manifest_path = write_manifest(stats, cap, checks)
    print(f"\nWrote manifest -> {manifest_path}")

    print("-" * 78)
    if not all_expected:
        print("WARNING: at least one feature-column identity check did not match expectation.")
        raise SystemExit(1)
    print("All variants built; feature columns match expectations across A/B/C.")
    print("=" * 78)
