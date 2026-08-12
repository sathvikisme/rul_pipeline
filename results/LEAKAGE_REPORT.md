# Leakage Red-Team Report — PGTS Re-evaluation

RUL-Bench, `leakage-red-team` subagent. Reproduces PROJECT_BRIEF.md §8 using
`.claude/skills/pgts-split/pgts.py` (window/embargo protocol: n_splits=5,
embargo=10 and embargo=0). All numbers below come from actually executing
`src/pgts_evaluation.py` and `src/pgts_rolling_extension.py` against
`data/processed/train.csv` / `data/processed/test.csv` and the models/
hyperparameters model-trainer already produced — nothing here is estimated
or reused from a prior run of a different pipeline (CLAUDE.md rule 1).

`python .claude/skills/pgts-split/pgts.py` sanity check passed before this
work started, and `_assert_no_leakage` was called explicitly on every one of
the 10 real folds generated below (5 for embargo=10, 5 for embargo=0) — all
passed, zero engine-ID overlap between train and test in every fold.

## 1. Confirmed leak: plain row-level KFold on grouped time-series data

**Mechanism.** `src/models.py:99` defines
`KFOLD = KFold(n_splits=5, shuffle=True, random_state=42)` and uses it both
for `GridSearchCV` hyperparameter selection (line 266-274) and for the
per-fold R²/MSE arrays written to `results/tables/cv_scores_r2.csv` /
`cv_scores_mse.csv` (line 288-295) — the exact numbers stats-auditor used
for significance testing. This is **row-level** shuffled K-fold, not grouped
by `engine_id`. Because `train.csv` is sorted by `engine_id` then `cycle`
and each of the 100 engines contributes ~206 contiguous rows of a smooth
degradation trajectory, plain shuffled KFold routinely puts cycle *t* of an
engine in one fold and cycle *t±1, t±2, …* of the **same engine** in
another. Adjacent cycles differ by tiny sensor deltas, so the model is
effectively being asked to interpolate a near-duplicate row it has already
seen the neighbors of — an easier task than genuine held-out generalization
to an unseen engine.

**Quantified.** Re-validating the four strongest official-split models
(XGBoost, MLP, CatBoost, LightGBM) with `purged_group_time_series_split`
(same train.csv, same hyperparameters, no re-tuning) instead of plain KFold:

| Model | Plain-KFold CV R² (leaky) | PGTS(embargo=10) R² (clean) | R² gap | Plain-KFold MSE | PGTS(embargo=10) MSE | MSE inflation |
|---|---|---|---|---|---|---|
| XGBoost | 0.8581 | 0.8420 | 0.0161 | 246.3 | 273.2 | +10.9% |
| MLP | 0.8604 | 0.8450 | 0.0154 | 242.3 | 268.0 | +10.6% |
| CatBoost | 0.8608 | 0.8417 | 0.0191 | 241.5 | 273.7 | +13.3% |
| LightGBM | 0.8586 | 0.8402 | 0.0185 | 245.3 | 276.3 | +12.6% |

This confirms the mechanism CLAUDE.md rule 4 and this subagent's mandate
warn about: the plain-KFold numbers model-trainer reported (and stats-auditor
tested significance on) are **modestly but consistently optimistic** — about
1.5-2 R² points, 11-13% MSE — because of same-engine adjacent-cycle leakage.
This is real and should be flagged, but it is **not** the "collapses to
strongly negative R²" effect the source paper (Özcan 2025) reports for its
own headline number. See §2 for why.

`embargo=10` vs `embargo=0` made almost no difference here (R² 0.840-0.845
either way, deltas ≤0.002). Reason: `purged_group_time_series_split`
partitions engines into contiguous *blocks by engine_id order* — since
FD001 engine IDs are arbitrary labels for independently-simulated runs, not
a real chronology, the embargo purges a handful of training rows belonging
to *different, unrelated* engines that happen to sit next to a test block's
boundary in the sorted array. It does not (and structurally cannot, since
GroupKFold-style group splitting already keeps every engine wholly on one
side) purge same-engine boundary rows, which is where the real leakage risk
would be. The embargo is doing effectively nothing to remove signal here —
that's a property of this dataset's engine ordering, not a bug in the
implementation.

## 2. What did NOT reproduce: official-split R² is not leaking upward

PROJECT_BRIEF.md and CLAUDE.md both flag the expectation that "the official
split R² of ~0.99 collapses to strongly negative R² under PGTS." That
expectation does not hold for this reproduction, and forcing it into the
report would itself be a fabrication (CLAUDE.md rule 1). Actual comparison,
official test.csv vs PGTS(embargo=10) on train.csv, same models:

| Model | Official-split R² | Official-split MSE | PGTS(embargo=10) R² | PGTS(embargo=10) MSE |
|---|---|---|---|---|
| XGBoost | 0.6776 | 245.2 | 0.8420 | 273.2 |
| MLP | 0.6775 | 245.3 | 0.8450 | 268.0 |
| CatBoost | 0.6759 | 246.5 | 0.8417 | 273.7 |
| LightGBM | 0.6754 | 246.9 | 0.8402 | 276.3 |

PGTS R² is *higher* than the official split's, not lower — the opposite of
the expected direction. Before writing this up as "the model is just
robust," I checked for a split bug (per this subagent's own mandate to
double-check surprising non-collapses): sort order confirmed, zero
group leakage confirmed via `_assert_no_leakage` on every fold, hyperparameters
matched exactly to model-trainer's selection. No bug found. Instead there is
a real, explainable, non-leakage mechanism:

**MSE tells the true story; R² is being distorted by unequal target
variance between train.csv and test.csv.** Because R² = 1 − MSE / Var(y):

| Split | RUL mean | RUL std | RUL variance | % rows at RUL cap (125, "easy") |
|---|---|---|---|---|
| `train.csv` (full run-to-failure trajectories) | 86.8 | 41.67 | 1736.5 | 39.4% |
| `test.csv` (official, truncated trajectories) | 108.9 | 27.58 | 760.6 | 61.4% |

`test.csv` engines are truncated at an earlier, more homogeneous point in
their life than `train.csv`'s full run-to-failure trajectories, so the
official test set's RUL target has **less than half the variance** of the
training set's. A model can have *worse absolute error* (MSE 245-247 on
`test.csv` vs 268-276 on PGTS `train.csv` folds — official-split MSE is
actually the **better** number) and still show a *lower R²*, purely because
R²'s denominator (target variance) shrank more than the numerator (error)
did. Sanity check: `1 − 245.2/760.6 = 0.678` — matches the reported official
XGBoost R² of 0.6776 almost exactly, confirming this is the dominant effect,
not an artifact of my computation.

**Conclusion for this section:** the official train/test split in this
reproduction is not leaking in the direction the paper's critique describes.
If anything, MSE-for-MSE, the officially-split test performance is
genuinely *better* than the leakage-safe PGTS estimate on train.csv, which
is consistent with a model that generalizes to unseen engines reasonably
well — the low R² is a target-variance artifact of comparing a truncated,
easier test distribution against R²'s variance-normalized scale, not
evidence of memorization. This also means our official R² (~0.68) is
already far below the paper's reported ~0.99 for reasons unrelated to PGTS
(likely preprocessing/feature choices — out of scope for this subagent,
flagged for model-trainer/README) — there was much less headline number to
"collapse" in the first place.

## 3. Null baseline (predict training-fold mean RUL)

| Reference split | R² | MSE |
|---|---|---|
| Official test.csv (trained on train.csv mean) | −0.6416 | 1248.6 |
| PGTS embargo=10, mean across 5 folds | −0.0060 | 1741.4 |
| PGTS embargo=0, mean across 5 folds | −0.0060 | 1741.4 |

All four real models comfortably and consistently beat the null baseline
under every evaluation regime (official split, PGTS embargo=10, PGTS
embargo=0) by a wide margin — R² 0.68-0.86 vs. a null baseline that is at or
below 0 everywhere, as expected (0 by construction on the PGTS folds where
train and test come from the same distribution; the official-split null is
actually negative because `train.csv`'s mean RUL, 86.8, is a poor predictor
of `test.csv`'s systematically higher/tighter RUL distribution, 108.9 mean —
another symptom of the truncation-driven distribution shift from §2). This
rules out "the models aren't doing anything real" as an alternative
explanation for either the official-split or PGTS numbers.

## 4. Extension experiment: do rolling-window features close the KFold→PGTS gap?

One clean, honestly-reported experiment (`src/pgts_rolling_extension.py`):
added `add_rolling_features` (mean/std/finite-difference-slope, window=5,
computed per-engine so windows never cross engine boundaries — the existing,
previously-unused function in `src/features.py`) to XGBoost's feature set
(18 → 60 features) and re-ran both plain KFold and PGTS(embargo=10).

| Feature set | Plain-KFold R² | Plain-KFold MSE | PGTS(10) R² | PGTS(10) MSE | KFold−PGTS R² gap |
|---|---|---|---|---|---|
| Baseline (18 features) | 0.8581 | 246.3 | 0.8420 | 273.2 | 0.0161 |
| + rolling window features (60 features) | 0.8833 | 202.5 | 0.8441 | 269.6 | 0.0392 |

**Result: rolling features did not meaningfully close the gap, and made the
plain-KFold estimate more misleading, not less.** The honest,
leakage-safe PGTS number barely moved (R² +0.0021, MSE −3.6, ~1.3%
improvement — within PGTS's own fold-to-fold noise, std≈0.039). Meanwhile
the plain-KFold "CV" number jumped dramatically (R² +0.025, MSE −43.9,
~18% improvement), which *widened* the KFold-vs-PGTS gap from 0.016 to
0.039 R² points — more than double.

**Why, mechanistically:** rolling mean/std/slope features are computed from
a window of nearby cycles *within the same engine*. Under plain row-level
KFold, a test-fold row's rolling features are highly correlated with
training-fold rows from the very same engine's adjacent cycles (which is
exactly the leakage mechanism in §1) — so these features hand the leaky
evaluation an even easier way to "recognize" a trajectory it has already
partially seen, inflating the plain-KFold number further without adding
real forecasting information. Under PGTS, no engine's rows are ever split
between train and test, so the same rolling features can only carry
information that generalizes to genuinely unseen engines — which turns out
to be marginal for this window size/feature set. This is a negative result,
reported honestly: simple rolling-window features are not a fix for the
PGTS-measured generalization gap in this pipeline, and the experiment
incidentally provides a second, independent confirmation of the KFold
leakage mechanism from §1 (the same leaky-feature-under-leaky-split
interaction that inflates cv_scores_r2.csv).

## Bottom line

1. **Real, modest leak confirmed**: `src/models.py`'s plain
   `KFold(5, shuffle=True, random_state=42)` (used for both GridSearchCV
   tuning and the CV scores stats-auditor tested) overstates train-set
   generalization by ~1.5-2 R² points / ~11-13% MSE relative to grouped,
   embargoed PGTS on the same data. Fix: re-run tuning/CV with
   `GroupKFold`/PGTS grouped by `engine_id` before trusting
   `cv_scores_r2.csv` for significance claims.
2. **The paper's specific "official R² collapses to strongly negative under
   PGTS" narrative does not reproduce** in this codebase. The official-split
   numbers are, if anything, MSE-better than the PGTS estimate; the R² gap
   in the *opposite* direction is a target-variance artifact of `test.csv`
   being truncated/more homogeneous than `train.csv`, not evidence of
   leakage. Verified this isn't a split bug (zero group overlap, correct
   sort order, matched hyperparameters) before reporting it.
3. **Extension experiment (rolling features) is a negative result**: no
   meaningful PGTS improvement, and it makes the naive-KFold evaluation
   *more* overoptimistic, not less — itself a useful cautionary finding for
   any future feature-engineering work on this pipeline.

## Artifacts produced by this subagent

- `results/tables/pgts_comparison.csv` — the full comparison table (per
  model: official-split R²/MSE, plain-KFold CV R²/MSE, PGTS(embargo=10)
  R²/MSE, PGTS(embargo=0) R²/MSE, plus a `NULL_BASELINE_mean_RUL` row for
  all three regimes).
- `results/tables/pgts_perfold_raw.json` — raw per-fold R²/MSE for every
  model x embargo x fold, for anyone who wants to check fold-level variance
  rather than just the means.
- `results/tables/pgts_rolling_extension.json` — raw numbers behind §4.
- `src/pgts_evaluation.py` — the re-evaluation script (reruns cleanly,
  reproducible, `random_state=42` throughout).
- `src/pgts_rolling_extension.py` — the extension experiment script.
