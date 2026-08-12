"""
Purged Group Time Series Split (PGTS).

Standard k-fold or GroupKFold splitting on run-to-failure sensor data leaks
information: if you split randomly by row, cycles from the same engine end up
in both train and test, and the model effectively memorizes that engine's
degradation curve. If you split by engine (GroupKFold) but don't respect the
implicit time-ordering across folds, you can still get subtly optimistic
results depending on how folds are constructed.

PGTS combines group-based splitting (so no engine's cycles appear in both
train and test) with an embargo: cycles adjacent to a group boundary are
dropped from training entirely, so the model can't exploit near-duplicate
sensor windows that sit just across the split boundary.

This is deliberately conservative. Expect metrics under PGTS to look WORSE
than under a naive split or even a plain GroupKFold — that gap is itself a
diagnostic, not a bug. If your official-split R^2 is ~0.99 and your PGTS R^2
is strongly negative, that tells you the official-split number was leaking,
not that PGTS is "too harsh."
"""
import numpy as np


def purged_group_time_series_split(groups, n_splits=5, embargo=10):
    """
    Generate (train_idx, test_idx) pairs, splitting by group (engine ID) with
    an embargo zone around each test group to prevent near-boundary leakage.

    Parameters
    ----------
    groups : array-like, shape (n_samples,)
        Group/engine identifier per row. Rows for the same engine MUST be
        contiguous in the array (typical after sorting by engine then cycle).
    n_splits : int
        Number of folds.
    embargo : int
        Number of rows adjacent to each test fold's boundary (on both sides,
        within the same engine group) to drop from training.

    Yields
    ------
    train_idx, test_idx : np.ndarray, np.ndarray
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_splits > n_groups:
        raise ValueError(f"n_splits ({n_splits}) cannot exceed number of groups ({n_groups})")

    fold_boundaries = np.array_split(np.arange(n_groups), n_splits)

    for fold_group_idx in fold_boundaries:
        test_groups = set(unique_groups[fold_group_idx])
        test_mask = np.isin(groups, list(test_groups))
        test_idx = np.where(test_mask)[0]

        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]

        if embargo > 0 and len(test_idx) > 0:
            # Drop rows within `embargo` positions of any test-block boundary,
            # per contiguous test block, so we don't purge unrelated engines.
            boundaries = []
            block_start = test_idx[0]
            prev = test_idx[0]
            for idx in test_idx[1:]:
                if idx != prev + 1:
                    boundaries.append((block_start, prev))
                    block_start = idx
                prev = idx
            boundaries.append((block_start, prev))

            embargo_idx = set()
            for start, end in boundaries:
                lo = max(0, start - embargo)
                hi = min(len(groups) - 1, end + embargo)
                embargo_idx.update(range(lo, hi + 1))

            train_idx = np.array([i for i in train_idx if i not in embargo_idx])

        yield train_idx, test_idx


def _assert_no_leakage(groups, train_idx, test_idx):
    train_groups = set(np.asarray(groups)[train_idx])
    test_groups = set(np.asarray(groups)[test_idx])
    overlap = train_groups & test_groups
    assert not overlap, f"leakage: groups {overlap} appear in both train and test"


if __name__ == "__main__":
    # Synthetic check: 10 engines, 20 cycles each, contiguous by engine.
    rng = np.random.default_rng(0)
    engine_ids = np.repeat(np.arange(10), 20)

    fold_count = 0
    for train_idx, test_idx in purged_group_time_series_split(engine_ids, n_splits=5, embargo=3):
        _assert_no_leakage(engine_ids, train_idx, test_idx)
        fold_count += 1
        print(f"fold {fold_count}: train={len(train_idx)} rows, test={len(test_idx)} rows, "
              f"test_engines={sorted(set(engine_ids[test_idx]))}")

    assert fold_count == 5
    print("no group leakage across any fold — sanity checks passed")
