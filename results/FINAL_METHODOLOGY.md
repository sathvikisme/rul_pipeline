# Final Methodology Reference (Phase 2)

A technical reference for the leakage-safe protocol this project settled on,
for anyone reproducing or extending the work. For the narrative/results
write-up, see `results/REPRODUCTION_REPORT.md`. Phase 1's original
methodology is documented separately in the Phase 1 section of `README.md`
and is preserved unmodified as the historical baseline.

## Data

- Source: NASA C-MAPSS FD001, raw files in `data/raw/` (verified
  byte-identical to the original NASA distribution).
- Track B (leakage-safe benchmark) uses `data/processed/train.csv` /
  `test.csv` — Phase 1's cleaned, RUL-labeled, capped-at-125, globally
  variance-thresholded/scaled files (`src/preprocessing.py` +
  `src/features.py`). **Known residual limitation**: the sensor
  variance-threshold *selection* (which 14 of 21 sensors survive) was made
  once, globally, before any Track B CV loop — fold-local re-fitting
  (`src/fold_safe_pipeline.py`) refits the `VarianceThreshold` transform's
  parameters and the `StandardScaler` inside every fold, but cannot undo
  the original global inclusion/exclusion decision. Documented, not hidden.
- Track A (faithful paper reproduction) uses its own feature matrix, built
  directly from `data/raw/*.txt` via `src/preprocessing.py`'s raw-parsing
  utilities only — never touches `data/processed/*.csv` — matching the
  paper's protocol exactly (`engine_id`+`cycle` included, all 21 raw
  sensors, `MinMaxScaler`, no capping). See `src/track_a_reproduction.py`.
- RUL-cap ablation datasets: `data/processed/rul_cap_ablation/{A,B,C}/`,
  built by `src/rul_cap_variants.py`, sharing identical features and
  differing only in train/test RUL-target capping.

## Fold-safe preprocessing pipeline (Track B)

`src/fold_safe_pipeline.py`: `build_model_pipeline(name, estimator)` →
`sklearn.Pipeline([VarianceThreshold(1e-5), StandardScaler() if name in
{SVM, KNN, LinearRegression, Ridge, BayesianRidge, MLP} else passthrough,
estimator])`. `VarianceThreshold` always precedes scaling (scaling first
would normalize all variances to ~1 and defeat the filter). Every pipeline
instance is fit fresh inside each CV fold — never globally.

## Nested cross-validation protocol

`src/nested_cv.py`. Outer: `GroupKFold(n_splits=5)` on `engine_id` (100
engines, ~20 held out per fold). Inner: `GroupKFold(n_splits=3)` on the
outer-training engines, used only where hyperparameter search happens.
`groups=` is passed explicitly to every inner `.fit()` call — a self-check
(`_smoke_test_groups_required()`) runs at the start of every invocation and
asserts a `GroupKFold`-backed search raises `ValueError` if `groups=` is
omitted, guarding against silent degradation to non-grouped folding.

**Tiered tuning budget** (cost control — see `README.md`/`REPRODUCTION_
REPORT.md` §7 for the rationale):
- Tier 1 (LightGBM, CatBoost, XGBoost): `RandomizedSearchCV(n_iter=10)`
  inside every outer fold, `cv=`inner `GroupKFold(3)`.
- Tier 2 (GradientBoosting, MLP, SVM, KNN): fixed hyperparameters (reused
  from Phase 1's `results/tables/best_hyperparams.json`), refit once per
  outer fold, no inner search.
- Tier 3 (LinearRegression, Ridge, BayesianRidge): small exhaustive
  `GridSearchCV` grids, cheap enough for genuine nested tuning.
- Total wall-clock for all 10 models, 5 outer folds: **5.64 minutes**
  (Tier 1: 154.5s, Tier 2: 179.5s, Tier 3: 3.4s) on an 8-core machine.

Per-fold outputs: R²/MSE/MAE/PHM08 per (model, outer fold) →
`results/tables/nested_cv_metrics.csv`; per-row out-of-fold predictions
(tagged `engine_id`) → `results/tables/nested_cv_oof_predictions.csv`; chosen
hyperparameters per fold → `results/tables/nested_cv_best_params.json`; fit
timing → `results/tables/nested_cv_timing.csv`.

Every fold (outer and inner) is checked with `pgts._assert_no_leakage`
(reused from `.claude/skills/pgts-split/pgts.py`, not reimplemented) before
its results are trusted.

## Nested stacking

`src/nested_stacking.py`. Base learners: LightGBM, CatBoost, XGBoost,
GradientBoosting, MLP (the tier that's both CV-strong and test-strong in
Phase 1's own analysis). Meta-learner training data: the pooled, genuine
nested out-of-fold predictions from `nested_cv.py`'s outer-fold refits —
**not** `cross_val_predict` (Phase 1's approach, which used hyperparameters
selected under a leaky split). Zero additional base-model fits required.
Ridge `alpha` tuned via a further `GroupKFold` on the pooled OOF matrix.

## Statistical testing

`.claude/skills/paired-stats/paired_stats.py` (new skill, same convention
as `.claude/skills/stats-suite/`, own `python paired_stats.py` sanity
check). Use this — not the original `stats-suite` skill — whenever
comparing models evaluated on identical folds or on predictions clustered
within the same ~100 engines.

- `paired_permutation_test` / `wilcoxon_signed_rank` — paired per-fold
  score comparisons, with a documented resolution floor
  (min achievable p ≈ 2/2ⁿ for n paired folds).
- `holm_bonferroni_correction` — applied across the full pairwise family
  (55 pairs for 11 entities), not pair-by-pair.
- `practical_significance_flag` — 1% relative-MSE-delta threshold; a
  pair can be statistically non-significant yet practically meaningful, or
  vice versa, and both are reported, never conflated.
- `paired_engine_bootstrap` — resamples the set of ~100 engines (not
  individual rows), includes all of a sampled engine's rows, uses the
  *same* resample draw for both models in a pair (required for a valid
  paired ΔMSE CI). 10,000 resamples, seed 42.

`src/stats_auditor_track_b.py` runs both across all pairs →
`results/tables/model_pairwise_tests.csv`,
`results/tables/engine_bootstrap_metrics.csv`.

## Freeze discipline

`results/FREEZE_DECISION.md` is the authoritative, human-reviewed synthesis
of the statistical evidence into a final model/ensemble choice — a
judgment call (cost, interpretability, confirmed pairwise wins), not an
automated pick. It must be written and finalized **before** any script is
allowed to read `data/processed/test.csv`.

`src/track_b_pipeline.py` hard-stops before ever touching `test.csv` — this
is a code-level guard, not just convention (verified by `tests/
test_leakage.py::test_track_b_pipeline_guard_rejects_test_csv_path`).

`src/final_eval.py` is the only script permitted to read the official test
set, and only once: it checks `results/tables/_test_set_access_log.json`
for any prior entry and refuses to run (raises `RuntimeError`) if one
exists. Verified independently: a second invocation does raise as expected.

## Automated leakage tests

`tests/` (pytest, `pytest tests/ -v`, 11/11 passing):
- Group overlap + full-engine-accounting, reusing `pgts._assert_no_leakage`.
- Test-set isolation (no `test.csv` row ever enters a Track B training index).
- Fold-only-fit assertions for `VarianceThreshold`/`StandardScaler`.
- Stacking OOF purity (every OOF row's engine was genuinely held out that fold).
- `groups=` propagation regression guard.
- Single-frozen-final-eval log check.
- `tests/test_deliberate_leak_injection.py` — a deliberately-broken
  `GroupKFold`→row-level-`KFold` swap, asserting (a) the leakage assertion
  fails loudly on it and (b) its R² is measurably higher than the real
  Track B number — proof the test suite has teeth, not just tautological
  passes on already-correct code.

## Track A — faithful paper reproduction

`src/track_a_reproduction.py`. A genuinely separate code path: does not
import any Track B module, does not read `data/processed/*.csv`. Replicates
the paper's exact protocol (§ Data, above) including `engine_id` as a raw
feature and the row-level 80/20 split — deliberately, to demonstrate the
paper's number is reproducible and to independently confirm its own
`GroupKFold`-collapse and `id`/`cycle`-ablation diagnostics in this
codebase. All Track A output tables are prefixed `track_a_`/
`paper_reproduction_` and must never be blended with Track B numbers.

## Reproducing this end-to-end

```bash
pip install -r requirements.txt          # includes pytest==8.3.4

# Track B (leakage-safe benchmark)
python src/nested_cv.py                  # or via track_b_pipeline.py
python src/track_b_pipeline.py
python src/nested_stacking.py
python src/rul_cap_variants.py && python src/rul_cap_ablation.py
python src/stats_auditor_track_b.py

pytest tests/ -v                         # must be green before proceeding

# Freeze checkpoint is a human-reviewed document, not a script —
# see results/FREEZE_DECISION.md before running the next line.
python src/final_eval.py                 # reads test.csv exactly once
python src/shap_track_b.py

# Track A (independent, faithful reproduction)
python src/track_a_reproduction.py
```

Every stage uses `random_state=42` (or the library's equivalent seed
parameter). `nested_cv.py`/`track_b_pipeline.py` include a determinism
self-check pattern (re-run twice, compare outputs) recommended before
trusting any single run's numbers as final.
