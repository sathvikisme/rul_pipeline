---
name: pgts-split
description: Generate leakage-safe Purged Group Time Series splits for engine-grouped C-MAPSS data. Use whenever building any train/test split for the leakage-safe re-evaluation, or when investigating whether a reported metric is inflated by cross-engine or cross-cycle leakage.
---

# Purged Group Time Series Split (PGTS)

The official C-MAPSS train/test split (and even a plain GroupKFold) can look
much better than a model actually deserves. This skill implements the
leakage-safe alternative used to stress-test the official-split results.

## Usage

```python
from pgts import purged_group_time_series_split

# `groups` = engine ID per row, sorted so each engine's rows are contiguous
for train_idx, test_idx in purged_group_time_series_split(
    groups, n_splits=5, embargo=10
):
    X_train, X_test = X[train_idx], X[test_idx]
    ...
```

## Non-negotiable checks before using results from this split

1. Run `python pgts.py` first — it asserts zero group overlap between train
   and test on every fold using synthetic data. If that assertion fails,
   something is broken; do not proceed to real data.
2. After splitting real data, explicitly verify (don't just assume) that no
   engine ID appears in both `train_idx` and `test_idx` for any fold — use
   `_assert_no_leakage` from `pgts.py` as a helper, or write an equivalent
   check inline in the analysis notebook.
3. Expect PGTS metrics to look meaningfully worse than official-split
   metrics. That gap is the finding, not a bug to "fix" by loosening the
   embargo until the numbers look better again.
4. Report both `embargo=10` and `embargo=0` results side by side (per the
   project brief) — the delta between them is itself informative about how
   much of the leakage is boundary-adjacency vs. something structural.
