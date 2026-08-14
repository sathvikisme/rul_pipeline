"""
data_integrity.py — RUL-Bench data-engineer subagent (Phase 2)

Assertion functions that check `data/processed/train.csv` / `test.csv` (the
Phase 1 output, left untouched by Phase 2) for the invariants the rest of the
pipeline silently assumes:

  1. Exactly 100 unique engines in train, 100 in test.
  2. No duplicate (engine_id, cycle) row pairs in either file.
  3. Both files sorted by (engine_id, cycle), contiguous per engine.
  4. RUL cap sanity — train/test RUL min/max match what
     data/processed/DATA_DICTIONARY.md documents (cap=125).
  5. Train/test schema (column set) match.

Design intent: every `assert_*` function below raises AssertionError with a
descriptive message on failure and returns a small diagnostics dict on
success. That makes them directly reusable as pytest test bodies (a later
subagent's `tests/` suite can `from data_integrity import assert_*` and wrap
each in a `test_*` function, or call them as-is) AND usable standalone here
for a human-readable pass/fail report — no separate "report" vs. "test"
logic to keep in sync.

The sort-check in `assert_sorted_by_engine_cycle` intentionally reuses the
exact pattern already used in `src/pgts_evaluation.py` (lines ~50-52:
`df.reset_index(drop=True).equals(df.sort_values([...]).reset_index(drop=True))`)
rather than reinventing a different sortedness check.

This module does not modify data/processed/ in any way — read-only checks.
"""

from __future__ import annotations

import os
import re

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(REPO_ROOT, "data", "processed", "test.csv")
DATA_DICT_PATH = os.path.join(REPO_ROOT, "data", "processed", "DATA_DICTIONARY.md")


# ---------------------------------------------------------------------------
# Individual assertion functions
# ---------------------------------------------------------------------------

def assert_engine_counts(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          expected_train: int = 100, expected_test: int = 100) -> dict:
    """Exactly `expected_train`/`expected_test` unique engines in each split.

    We compute the actual nunique() rather than assuming the documented
    100/100 is still true — the whole point of this module is to verify,
    not trust, prior claims.
    """
    n_train = int(train_df["engine_id"].nunique())
    n_test = int(test_df["engine_id"].nunique())
    assert n_train == expected_train, (
        f"train.csv has {n_train} unique engines, expected {expected_train}"
    )
    assert n_test == expected_test, (
        f"test.csv has {n_test} unique engines, expected {expected_test}"
    )
    return {"train_engines": n_train, "test_engines": n_test}


def assert_no_duplicate_engine_cycle(df: pd.DataFrame, name: str = "dataset") -> dict:
    """No duplicate (engine_id, cycle) row pairs — each engine-cycle should
    appear exactly once (that's the whole identity of a row in this data)."""
    dup_mask = df.duplicated(subset=["engine_id", "cycle"], keep=False)
    n_dup = int(dup_mask.sum())
    assert n_dup == 0, (
        f"{name}: found {n_dup} rows involved in duplicate (engine_id, cycle) pairs"
    )
    return {"dataset": name, "n_duplicate_engine_cycle_rows": n_dup}


def assert_sorted_by_engine_cycle(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Rows sorted by (engine_id, cycle), contiguous per engine.

    Reuses the exact sortedness-check pattern already used in
    src/pgts_evaluation.py rather than a different implementation, since
    PGTS's group-contiguity assumption depends on this same property.
    """
    is_sorted = df.reset_index(drop=True).equals(
        df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    )
    assert is_sorted, (
        f"{name} is not sorted by (engine_id, cycle) -- required for the "
        f"group-contiguity assumption used by PGTS (see pgts_evaluation.py) "
        f"and by GroupKFold-style splitters generally"
    )
    return {"dataset": name, "sorted_by_engine_cycle": is_sorted}


def _documented_rul_cap(data_dict_path: str = DATA_DICT_PATH) -> int:
    """Parse the RUL cap value out of DATA_DICTIONARY.md's own prose
    ("Piecewise-linear capped at 125 cycles") rather than hardcoding 125
    here a second time -- if the documented value ever drifts from the
    actual pipeline constant, we want THIS check to read the doc, not
    silently re-assert a stale literal.
    """
    with open(data_dict_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"capped at (\d+) cycles", text)
    if not m:
        raise ValueError(
            f"Could not find a 'capped at N cycles' statement in {data_dict_path} "
            f"-- DATA_DICTIONARY.md format may have changed."
        )
    return int(m.group(1))


def assert_rul_cap_sanity(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           data_dict_path: str = DATA_DICT_PATH) -> dict:
    """Confirm train/test RUL min/max are consistent with the cap value
    DATA_DICTIONARY.md documents for the CURRENT data/processed/ files.

    Phase 1's pipeline (src/preprocessing.py::apply_rul_cap, called from
    features.py::run_pipeline) applies the cap to BOTH train and test RUL
    (this is the "Variant B" behavior that src/rul_cap_variants.py's
    ablation exists to interrogate -- capping test ground truth is
    debatable methodology, not a bug in this check). So for the data as it
    actually exists on disk today, both splits should have:
      - RUL >= 0 (no negative remaining life)
      - RUL <= documented_cap
      - RUL == documented_cap achieved by at least one row in TRAIN (every
        engine's early cycles get flat-capped, so the cap value should
        actually appear, not just bound the range from above)
    """
    cap = _documented_rul_cap(data_dict_path)

    train_min, train_max = float(train_df["RUL"].min()), float(train_df["RUL"].max())
    test_min, test_max = float(test_df["RUL"].min()), float(test_df["RUL"].max())

    assert train_min >= 0, f"train RUL min {train_min} < 0"
    assert test_min >= 0, f"test RUL min {test_min} < 0"
    assert train_max <= cap, f"train RUL max {train_max} exceeds documented cap {cap}"
    assert test_max <= cap, f"test RUL max {test_max} exceeds documented cap {cap}"
    assert train_max == cap, (
        f"train RUL max is {train_max}, expected it to equal the cap ({cap}) -- "
        f"at least one engine's early cycles should be flat-capped at the plateau"
    )

    return {
        "documented_cap": cap,
        "train_RUL_min": train_min,
        "train_RUL_max": train_max,
        "test_RUL_min": test_min,
        "test_RUL_max": test_max,
    }


def assert_schema_match(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train and test column sets match exactly (same feature schema)."""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    only_train = train_cols - test_cols
    only_test = test_cols - train_cols
    assert not only_train and not only_test, (
        f"schema mismatch -- columns only in train: {sorted(only_train)}; "
        f"columns only in test: {sorted(only_test)}"
    )
    return {"n_columns": len(train_cols), "columns": sorted(train_cols)}


# ---------------------------------------------------------------------------
# Runner — executes every check against the real processed files and prints
# a pass/fail report. This is what makes the checks "actually run" rather
# than just defined-but-never-called.
# ---------------------------------------------------------------------------

def run_all_checks(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH,
                    data_dict_path: str = DATA_DICT_PATH) -> list[dict]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    checks = [
        ("engine_counts (100 train / 100 test)",
         lambda: assert_engine_counts(train_df, test_df)),
        ("no_duplicate_engine_cycle (train)",
         lambda: assert_no_duplicate_engine_cycle(train_df, "train.csv")),
        ("no_duplicate_engine_cycle (test)",
         lambda: assert_no_duplicate_engine_cycle(test_df, "test.csv")),
        ("sorted_by_engine_cycle (train)",
         lambda: assert_sorted_by_engine_cycle(train_df, "train.csv")),
        ("sorted_by_engine_cycle (test)",
         lambda: assert_sorted_by_engine_cycle(test_df, "test.csv")),
        ("rul_cap_sanity (cap=125 per DATA_DICTIONARY.md)",
         lambda: assert_rul_cap_sanity(train_df, test_df, data_dict_path)),
        ("schema_match (train vs test columns)",
         lambda: assert_schema_match(train_df, test_df)),
    ]

    results = []
    for name, fn in checks:
        try:
            detail = fn()
            results.append({"check": name, "passed": True, "detail": detail})
        except AssertionError as e:
            results.append({"check": name, "passed": False, "detail": str(e)})
    return results


if __name__ == "__main__":
    print("=" * 78)
    print("data_integrity.py — checking data/processed/train.csv, test.csv")
    print("=" * 78)

    results = run_all_checks()

    n_pass = sum(r["passed"] for r in results)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['check']}")
        print(f"        {r['detail']}")

    print("-" * 78)
    print(f"{n_pass}/{len(results)} checks passed")
    print("=" * 78)

    if n_pass != len(results):
        raise SystemExit(1)
