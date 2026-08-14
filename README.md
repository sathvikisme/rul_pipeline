# RUL-Bench

# RUL-Bench

Reproduction and stress-test of an interpretable ensemble Remaining Useful
Life (RUL) prediction pipeline for aircraft turbofan engines, built on the
NASA C-MAPSS FD001 dataset.

**Source paper:** Özcan, H. "Interpretable ensemble remaining useful life
prediction enables dynamic maintenance scheduling for aircraft engines."
*Scientific Reports* 15, 39795 (2025).
[https://doi.org/10.1038/s41598-025-23473-2](https://doi.org/10.1038/s41598-025-23473-2)
Reference implementation (consulted for sanity-checking, not copied):
[github.com/hkmtcn/interpretable-rul-maintenance](https://github.com/hkmtcn/interpretable-rul-maintenance)

**What this is:** a portfolio project demonstrating methodological rigor in
applied ML — reproducible splits, real statistical significance testing (not
just point estimates), and an adversarial audit of the evaluation
methodology itself, including of *this project's own* evaluation. All
numbers in this document come from code in this repository that was
actually executed — see each `results/` artifact referenced inline.

This project has two phases, both preserved in full:

- **Phase 2 (current — start here)**: a methodological audit found Phase 1's
  own model/ensemble selection was itself contaminated by looking at the
  official test set. Phase 2 rebuilt the evaluation to be genuinely
  leakage-safe (nested, engine-grouped CV; train-only selection; paired,
  engine-level statistics; an automated leakage test suite; a single frozen
  final evaluation) and separately tracked down *why* this reproduction's
  numbers differ so sharply from the paper's — by directly inspecting and
  replicating the paper's own reference code. Full narrative:
  [`results/REPRODUCTION_REPORT.md`](results/REPRODUCTION_REPORT.md).
  Technical protocol reference: [`results/FINAL_METHODOLOGY.md`](results/FINAL_METHODOLOGY.md).
- **Phase 1 (below, preserved as historical record)**: the original
  reproduction — real pipeline, real results, but with model/ensemble
  selection contaminated by test-set peeking, which Phase 2 exists to fix.
  Kept unmodified, not deleted, so the before/after is visible.

---

## Phase 2 — Methodological Repair (start here)

### What was wrong with Phase 1

The most serious issue: Phase 1's ensembles were **selected by looking at
official test-set R²** (`results/tables/ensembling_config.json`:
`"selection_metric": "official test-set R2"`) — literally the test-set
contamination this project's own rules exist to prevent. A milder,
secondary issue: Phase 1's hyperparameter tuning used a plain row-level
`KFold` (not grouped by engine), letting same-engine cycles split across
train/validation, inflating apparent CV performance by a real but modest
~1.5–2 R² points.

### What Phase 2 built

A parallel, leakage-safe pipeline in new files (Phase 1's files are
untouched):

- **Fold-safe preprocessing** — variance filtering and scaling fit inside
  every CV fold, never globally (`src/fold_safe_pipeline.py`).
- **Nested, engine-grouped CV** — outer `GroupKFold(5)` / inner
  `GroupKFold(3)`, with a cost-tiered hyperparameter search (full nested
  tuning only for LightGBM/CatBoost/XGBoost; reused fixed hyperparameters
  for the rest) that ran in **5.6 minutes total**, not the naive ~25×
  blowup a fully exhaustive nested search would cost (`src/nested_cv.py`).
- **Genuinely nested stacking** — the meta-learner trains on real
  outer-fold-refit out-of-fold predictions, not `cross_val_predict`
  (`src/nested_stacking.py`).
- **Train-only ensemble/model selection** — every decision about which
  model(s) to freeze was made from training-engine evidence alone, written
  down and reasoned about *before* the test set was ever read
  ([`results/FREEZE_DECISION.md`](results/FREEZE_DECISION.md)).
- **Paired, engine-level statistics** — a new `paired-stats` skill
  (permutation tests, Wilcoxon, Holm correction, and a 100-engine cluster
  bootstrap instead of a naive 13,096-row one) replacing the assumption
  that CV folds or test rows are independent samples.
- **An automated leakage test suite** — 11 pytest tests, including one that
  deliberately breaks the pipeline (swaps grouped CV for row-level CV) and
  confirms the tests actually catch it (`tests/`, `pytest tests/ -v`).
- **A single, logged, frozen final evaluation** — `src/final_eval.py` reads
  `data/processed/test.csv` exactly once, and a runtime guard raises an
  error if anything tries to read it a second time.

### Headline finding: why doesn't this reproduction match the paper?

Rather than speculate, Phase 2 directly inspected the paper's own reference
notebook and replicated its exact protocol
(`src/track_a_reproduction.py`). The result: **the paper's headline
R²≈0.99 is real and independently reproducible — but it is not measured
against the official NASA test set at all.**

| Model | Track A reproduced R² | Paper reported R² |
|---|---:|---:|
| LightGBM | 0.9888 | 0.9894 |
| CatBoost | 0.9864 | 0.9872 |
| Ensemble | 0.9898 | 0.9904 |

This near-exact match comes from replicating the paper's actual protocol: a
row-level 80/20 split of the *training* trajectories only, with `engine_id`
included as a raw model feature — the official test set is loaded in the
paper's notebook but never scored. The same notebook's own `GroupKFold`
diagnostic shows this collapsing to R²≈0.44, and removing `engine_id`/`cycle`
as features recovers it to R²≈0.60 — both independently reproduced in this
codebase (`results/tables/track_a_groupkfold_collapse.csv`,
`track_a_id_cycle_ablation.csv`). Full detail:
`results/REPRODUCTION_REPORT.md` §5.

### Frozen final result (the one official test-set evaluation)

Per the training-only freeze decision, two statistically-tied candidates
were evaluated together, once:

| Model | R² | MSE | MAE | PHM08 |
|---|---:|---:|---:|---:|
| LightGBM | 0.6800 | 243.37 | 9.98 | 71,102 |
| StackingEnsemble | 0.6793 | 243.95 | 10.02 | 74,097 |

Essentially a tie — matching what the training-only statistics predicted.
**Fixing test-set-driven selection did not dramatically move the
official-split numbers** (Phase 1 LightGBM R²=0.6754 → Phase 2 R²=0.6800;
Phase 1 best stacking R²=0.6820 → Phase 2 R²=0.6793) — because the official
NASA engine split itself was never the leaky part of this pipeline; the
contamination was in the *internal selection process*, not the benchmark.
Full detail: `results/FINAL_TEST_RESULT.md`, `results/REPRODUCTION_REPORT.md` §10.

### What the statistics actually support

- The tree/boosting/MLP model cluster is genuinely, significantly better
  than linear models and KNN/SVM (confirmed by a 100-engine paired
  bootstrap).
- **Within that top cluster, almost no pairwise ranking is defensible** —
  including "the stacking ensemble beats the best individual model," which
  does *not* hold up under either Phase 1's or Phase 2's statistical
  testing. The one exception: LightGBM is confirmed better than
  GradientBoosting.
- With only 5 leakage-safe outer folds, per-fold significance tests have a
  hard resolution floor (documented, not glossed over) — the engine-level
  bootstrap is the more informative test and is treated as primary evidence.

Full statistical detail: `results/tables/PAIRED_STATS_SUMMARY.md`.

### Phase 2 repo additions

```
src/
  fold_safe_pipeline.py       fold-fit VarianceThreshold + StandardScaler pipeline factory
  data_integrity.py           engine-count/dup/sort/schema sanity checks
  nested_cv.py                outer/inner GroupKFold, tiered hyperparameter search
  nested_stacking.py          genuinely nested stacking OOF + meta-learner
  track_b_pipeline.py         orchestrator; hard-stops before touching test.csv
  rul_cap_variants.py / rul_cap_ablation.py    3-way RUL-cap A/B/C ablation
  track_a_reproduction.py     faithful paper-protocol reproduction (separate code path)
  final_eval.py               the single frozen official test-set evaluation
  shap_track_b.py             SHAP on the frozen models
.claude/skills/paired-stats/  new skill: paired permutation/Wilcoxon/bootstrap tests
tests/                        pytest leakage test suite (11 tests, incl. a deliberate-leak smoke test)
results/
  FREEZE_DECISION.md          the human-reviewed model-selection synthesis
  FINAL_TEST_RESULT.md        the one frozen test evaluation's numbers
  REPRODUCTION_REPORT.md      full narrative report (15 sections)
  FINAL_METHODOLOGY.md        technical protocol reference
  SHAP_ANALYSIS_TRACK_B.md, shap/track_b/
  tables/clean_grouped_cv_metrics.csv, nested_cv_metrics.csv, nested_cv_oof_predictions.csv,
         model_pairwise_tests.csv, engine_bootstrap_metrics.csv, ensemble_selection.csv,
         ablation_matrix.csv, paper_reproduction_metrics.csv, track_a_*.csv,
         final_test_metrics.csv, _test_set_access_log.json
```

### Reproducing Phase 2

```bash
pip install -r requirements.txt          # now includes pytest==8.3.4
python src/track_b_pipeline.py           # nested CV + nested stacking, ~6 min
python src/rul_cap_variants.py && python src/rul_cap_ablation.py
python src/stats_auditor_track_b.py
pytest tests/ -v                         # must be green — 11 passed
# read results/FREEZE_DECISION.md before proceeding — it's a human-reviewed
# judgment call, not a script, and final_eval.py depends on it being final
python src/final_eval.py                 # reads test.csv exactly once
python src/shap_track_b.py
python src/track_a_reproduction.py       # independent, faithful paper reproduction
```

---

## Phase 1 — Initial Reproduction (historical record, preserved unmodified)

Everything below is exactly as originally written. Where Phase 2 supersedes
a finding, that's noted in the Phase 2 section above, not by editing this
one.

### 1. Dataset — NASA C-MAPSS FD001

Source: NASA Ames Prognostics Data Repository (mirrored copy used here:
[github.com/edwardzjl/CMAPSSData](https://github.com/edwardzjl/CMAPSSData),
verified byte-identical in row/file size to the original NASA distribution).

FD001: single operating condition (sea level), single fault mode (HPC
degradation). 100 training engines (run-to-failure trajectories, 20,631
rows), 100 test engines (truncated trajectories, 13,096 rows), true RUL for
each test engine given separately in `RUL_FD001.txt`. 26 raw columns: engine
ID, cycle, 3 operational settings, 21 sensor readings.

### 2. Preprocessing & feature engineering **[Reproduction, with independent verification]**

Full detail: `data/processed/DATA_DICTIONARY.md`, `src/preprocessing.py`, `src/features.py`.

- **Missing values**: checked directly — 0 NaNs, 0 duplicate rows in both
  train and test. No imputation applied (nothing to impute).
- **RUL labels**: `max_cycle_for_engine − cycle` per row (train); back-computed
  from `RUL_FD001.txt`'s true value at each test engine's last observed cycle
  (test).
- **Piecewise-linear RUL capping** at **125 cycles** (literature convention,
  e.g. Heimes 2008; consistent with Özcan 2025) — chosen and documented
  explicitly, not silently hardcoded. (Phase 2 found the paper's own
  reference code actually applies **no** cap at all — see
  `results/REPRODUCTION_REPORT.md` §5/§11 for the ablation this motivated.)
- **Scaling**: `StandardScaler`, fit on the training split **only**, applied
  (never refit) to test. Verified post-hoc: test's scaled `op_setting_1`
  mean ≈ −0.00106 (non-zero — proof the scaler used train statistics, not
  test's own).
- **Feature selection**: variance-threshold check (threshold 1e-5, computed
  on train only) on the 21 raw sensor channels — **independently verified
  against this data**, not assumed from the paper. 7 of 21 sensors dropped
  as near-constant (`sensor_1,5,6,10,16,18,19`, variance ≤ 1.9e-6); 14 kept.
  Full per-sensor variance table in the data dictionary.
- **Rolling-window features** (mean/std/slope per sensor) implemented in
  `src/features.py` but **not used** in the baseline pipeline below — see
  §7 for why adding them was tested and rejected as an extension.

### 3. Models & ensembles **[Reproduction — ensemble selection later found to be test-set-contaminated, see Phase 2]**

10 individual models tuned via `GridSearchCV` (5-fold `KFold`,
`random_state=42` throughout, `neg_mean_squared_error` scoring): LightGBM,
CatBoost, Gradient Boosting, XGBoost, SVM, KNN, Linear Regression, Ridge,
Bayesian Ridge, MLP. Full grids and selected hyperparameters:
`results/tables/best_hyperparams.json`.

Two ensemble variants, both built on top of the tuned base learners:

- **Fixed weighted average**: 70% XGBoost + 30% MLP (the two strongest
  models by official test R²: 0.6776 vs 0.6775 — a near-tie; CatBoost and
  LightGBM trail by <0.002 R², noted in `results/tables/ensembling_config.json`).
  **Selected by looking at official test-set R² — this is the test-set
  contamination Phase 2 exists to fix; see `results/FREEZE_DECISION.md` for
  the corrected, train-only selection process.**
- **Stacking**: Ridge meta-learner (`alpha=100`, grid-searched), trained
  **only on genuine out-of-fold predictions** of 5 base learners (LightGBM,
  CatBoost, XGBoost, GradientBoosting, MLP — the clear top tier by CV score;
  KNN/SVM and the three near-identical linear models were excluded as
  weaker or redundant) generated via `cross_val_predict` — never on raw
  features, never on in-fold predictions. Mechanics: `results/tables/ensembling_config.json`.
  (Phase 2 found the base learners' own hyperparameters were themselves
  selected under a leaky, non-grouped `KFold` — see Phase 2 section above.)

### 4. Results — official split **[Reproduction]**

Evaluated once on the true held-out test set (100 engines never seen during
training).
(sign convention: `d = predicted − true`; late predictions penalized more

| Model | R² | RMSE | MSE | MAE | PHM08 RUL Score |
|---|---:|---:|---:|---:|---:|
| **Stacking (Ridge)** | **0.6820** | **15.55** | 241.88 | **9.98** | 76,738 |
| **Fixed-Weighted (XGB 70/MLP 30)** | **0.6817** | **15.56** | 242.13 | 10.01 | **72,576** |
| XGBoost | 0.6776 | 15.66 | 245.19 | 10.06 | 71,883 |
| MLP | 0.6775 | 15.66 | 245.32 | 10.31 | 82,074 |
| CatBoost | 0.6759 | 15.70 | 246.50 | 10.02 | 76,569 |
| LightGBM | 0.6754 | 15.71 | 246.92 | 10.02 | 76,549 |
| GradientBoosting | 0.6738 | 15.75 | 248.11 | 10.19 | 73,617 |
| KNN | 0.6350 | 16.66 | 277.59 | 10.73 | 78,847 |
| SVM | 0.6121 | 17.18 | 295.05 | 10.30 | 96,119 |
| Ridge | 0.5205 | 19.10 | 364.74 | 15.07 | 102,900 |
| BayesianRidge | 0.5205 | 19.10 | 364.75 | 15.07 | 102,933 |
| LinearRegression | 0.5205 | 19.10 | 364.75 | 15.07 | 102,988 |

Full table: `results/tables/official_split_metrics.csv`. **See Phase 2's
frozen re-evaluation above for the leakage-safe-selected numbers
(LightGBM R²=0.6800, StackingEnsemble R²=0.6793) — close to, but not
identical to, these Phase-1 numbers, and selected without ever looking at
this table.**

#### Comparison to the source paper — and an honest discrepancy

Özcan (2025) reports, for FD001, a LightGBM+CatBoost ensemble at
**RMSE = 6.62, RUL Score ≈ 2,951**, implying R² in the neighborhood of 0.99.
This reproduction's best result — the Stacking ensemble at **RMSE = 15.55,
RUL Score ≈ 76,738** — falls well short of that. We do **not** paper over
this gap. **Phase 2 fully resolved this discrepancy** by directly
replicating the paper's own protocol — see the Phase 2 section above and
`results/REPRODUCTION_REPORT.md` §5. Original (Phase 1) speculation, kept
for the record:

<<<<<<< HEAD
- **Not our own train/test leakage** — PGTS audit (an adversarial
  re-check specifically designed to catch this) found the official split
  is *not* inflated by grouping issues; if anything, our official-split MSE
  is slightly *better* than the leakage-safe PGTS estimate on the training
  engines. So the gap to the paper isn't explained by leakage in
  *this* codebase's official-split evaluation.
- **Likely explanations, not yet isolated**: differing preprocessing
  choices (e.g. a different RUL cap value, different scaler, or different
  sensor-selection threshold than this reproduction's independently-derived
  choices), a different GridSearchCV search space landing on different
  hyperparameters, or — as is common in the published RUL literature on
  C-MAPSS — an evaluation methodology in the original work that is more
  favorable than a strict held-out-engine test split. Without literally
  re-running the authors' own notebooks against our environment we can't
  attribute the gap definitively, and we're not asserting a specific cause
  we haven't verified.
- **What we can say with confidence**: our result is real, reproducible
  (fixed seeds throughout), and independently statistically validated.
=======
- **Not our own train/test leakage** — §7's PGTS audit (an adversarial
  re-check specifically designed to catch this) found the official split
  is *not* inflated by grouping issues; if anything, our official-split MSE
  is slightly *better* than the leakage-safe PGTS estimate on the training
  engines (see §7). So the gap to the paper isn't explained by leakage in
  *this* codebase's official-split evaluation.
- **Likely explanations, not yet isolated** *(now isolated — see Phase 2)*:
  differing preprocessing choices, a different GridSearchCV search space,
  or an evaluation methodology in the original work more favorable than a
  strict held-out-engine split.
>>>>>>> 4d99f58 (phase 2 complete)

### 5. Statistical validation **[Reproduction of methodology, extension in rigor — superseded by Phase 2's paired analysis for significance claims]**

Full report: `results/tables/STATS_SUMMARY.md`. Suite: `.claude/skills/stats-suite/stats_tests.py`.

- **Shapiro-Wilk** (normality, per model, 5-fold CV R²): 2/10 models violate
  (LinearRegression p=0.045, BayesianRidge p=0.048); Ridge borderline
  (p=0.050). **Levene's test** (variance homogeneity): F=0.109, p=0.999 — OK.
- Because normality was violated for CV R², **Kruskal-Wallis + Dunn's test
  (Holm-corrected)** — not in the base skill, added per its own
  instructions — was used as the *primary* evidence rather than the more
  dramatic-looking parametric Tukey HSD table.
- **One-way ANOVA**: F=142.60, p=1.63×10⁻²⁷ (R²); F=150.72, p=5.59×10⁻²⁸
  (MSE, which passed all assumption checks) — a real difference exists
  among the 10 models.
- **Defensible pairwise claims** (Dunn/Holm on R², Tukey on assumption-clean
  MSE): the three linear models are significantly worse than the
  tree/boosting/MLP cluster; CatBoost and MLP are significantly better than
  the three linear models specifically. **Within the top cluster**
  (LightGBM, CatBoost, GradientBoosting, XGBoost, MLP), **no pairwise
  difference is statistically significant** on either metric.
- **Bootstrap CI on held-out test MSE** (row-level, n=13,096, 10,000
  resamples, seed=0): both ensembles' 95% CIs ([232.9, 251.0] and
  [233.3, 251.2]) overlap substantially with XGBoost's ([236.3, 254.4]) and
  MLP's ([236.3, 254.5]).

**Honest verdict: the ensembles' small edge over the best individual models
in §4 is not statistically supported.** Per the project's own rule against
claiming "model A beats model B" from a point estimate alone, this
reproduction does **not** claim the ensembles are significantly better than
XGBoost/MLP/CatBoost/LightGBM — the apparent gain is consistent with
resampling noise given the CI overlap. **Note (Phase 2): this row-level
bootstrap and per-fold ANOVA both treat correlated observations (rows within
an engine; folds sharing evaluation structure) as independent — Phase 2's
paired, engine-level analysis is the more defensible version of this same
conclusion; see `results/tables/PAIRED_STATS_SUMMARY.md`.**

### 6. Interpretability — SHAP **[Reproduction of methodology + a check on the paper's specific claim]**

Full report: `results/SHAP_ANALYSIS.md`. Plots: `results/shap/`.

Analyzed XGBoost, CatBoost, MLP (individually), and the Fixed-Weighted
ensemble (SHAP is exact under linear combination:
`SHAP(ensemble) = 0.7·SHAP(XGBoost) + 0.3·SHAP(MLP)`, numerically verified
to 6.2×10⁻⁵ RUL-cycles reconstruction error). 500-row fixed-seed sample,
seed 42 throughout.

- **Dominant sensors, tied to physical meaning**: after `cycle` itself, the
  top features across all models are core-gas-path channels —
  `sensor_11` (Ps30, HPC outlet static pressure), `sensor_9`/`sensor_14`
  (Nc/NRc, physical/corrected core speed), `sensor_4` (T50, LPT outlet
  temperature), `sensor_12` (phi, fuel-flow ratio). This is physically
  sensible: FD001's injected fault mode is High-Pressure-Compressor
  degradation, so HPC/core-speed/turbine-temperature sensors dominating the
  models' decisions is the expected, not spurious, result. **This finding
  replicated independently in Phase 2's SHAP analysis on the leakage-safe
  frozen models — see `results/SHAP_ANALYSIS_TRACK_B.md`.**
- **The paper's "ensemble balances feature attribution" claim** — checked
  quantitatively (top-3-share and Gini coefficient of mean |SHAP| across
  features), not just eyeballed:

  | Model | Top-3 share | Gini |
  |---|---:|---:|
  | XGBoost | 0.568 | 0.565 |
  | CatBoost | **0.469** | **0.480** |
  | MLP | 0.526 | 0.519 |
  | Fixed-Weighted Ensemble | 0.521 | 0.536 |

  **Holds only partially.** The ensemble is more balanced than its own
  dominant member (XGBoost) — unsurprising, that's what averaging does —
  but it is *not* more balanced than CatBoost, an individual model not even
  part of this ensemble, which is the most evenly-attributed model measured.
  The paper's framing doesn't hold universally in this reproduction; it
  depends on which individual model you compare against. **Confirmed again
  independently in Phase 2 with a different ensemble architecture.**

### 7. The critical extension — leakage-safe (PGTS) re-evaluation **[Extension]**

Full report: `results/LEAKAGE_REPORT.md`. Skill: `.claude/skills/pgts-split/pgts.py`
(window/embargo protocol: `n_splits=5`, `embargo=10` and `embargo=0`).

This is the project's own contribution beyond reproduction: adversarially
re-checking whether *this reproduction's* evaluation methodology is lying to
us, independent of whatever the original paper's number is.

**1. A real, modest leak was found and quantified** — not in the official
train/test split, but in `src/models.py`'s `KFold(5, shuffle=True,
random_state=42)`, used for both GridSearchCV tuning and the CV scores §5
tested significance on. It is **row-level, not grouped by engine**, so
adjacent cycles from the same engine's trajectory can land in both the
train and validation fold. Quantified by re-validating the 4 strongest
models under grouped, embargoed PGTS instead:

| Model | Plain-KFold R² (leaky) | PGTS(embargo=10) R² (clean) | Gap | MSE inflation |
|---|---:|---:|---:|---:|
| XGBoost | 0.858 | 0.842 | 0.016 | +10.9% |
| MLP | 0.860 | 0.845 | 0.015 | +10.6% |
| CatBoost | 0.861 | 0.842 | 0.019 | +13.3% |
| LightGBM | 0.859 | 0.840 | 0.019 | +12.6% |

A real, consistent effect (~1.5–2 R² points, ~11–13% MSE) — but modest, not
dramatic. `embargo=10` vs `embargo=0` made almost no difference, because
FD001's `engine_id` ordering is arbitrary (independent simulation runs, not
a real chronology), so boundary-embargo purges unrelated engines' rows, not
same-engine ones — a property of this dataset, not an implementation bug
(verified via `_assert_no_leakage` passing on all 10 real folds). **Phase 2
independently reproduced this exact mechanism in a live, in-codebase
deliberate-leak test — see `tests/test_deliberate_leak_injection.py`.**

**2. What did *not* reproduce, reported honestly rather than forced:** the
paper's own reported finding — that official-split R²≈0.99 collapses to
strongly negative under PGTS — **does not reproduce in this codebase.**
PGTS R² (~0.84) on the *training* engines is actually *higher* than this
reproduction's official-split R² (~0.68) on held-out engines — the opposite
direction. Before reporting this as "the model is just robust," the
leakage-red-team check verified it wasn't a split bug (sort order, zero
group overlap, matched hyperparameters — all clean). The real mechanism:
`test.csv`'s truncated trajectories have RUL variance of only 760.6 vs.
`train.csv`'s 1736.5 (test is a more homogeneous, "easier-to-look-flat"
target distribution) — since R² = 1 − MSE/Var(y), the same or better
absolute error (official MSE ~245 is *lower*, i.e. better, than PGTS MSE
~270–276) produces a *lower* R² purely from the smaller denominator. This
also means this reproduction's official R² gap vs. the paper's ~0.99 (§4)
is not attributable to PGTS-style leakage — there was much less headline
number here to "collapse" in the first place. **Phase 2 found the actual
explanation for the paper's ~0.99 — see the Phase 2 section above.**

**3. Null baseline** (predict training-fold mean RUL): R² ≈ −0.64 (official
split) to ≈ −0.006 (PGTS folds) — every real model beats it by a wide
margin everywhere, ruling out "the models aren't doing anything real" as an
alternative explanation.

**4. Extension experiment — do rolling-window features close the gap?**
Added `add_rolling_features` (mean/std/slope, window=5, per-engine) to
XGBoost (18→60 features) and re-ran both plain-KFold and PGTS:

| Feature set | Plain-KFold R² | PGTS(10) R² | KFold−PGTS gap |
|---|---:|---:|---:|
| Baseline (18 feat.) | 0.858 | 0.842 | 0.016 |
| + rolling window (60 feat.) | 0.883 | 0.844 | 0.039 |

**Negative result, reported honestly**: rolling features barely moved the
leakage-safe PGTS number (+0.002 R², within fold-to-fold noise) while
*inflating* the leaky plain-KFold number further (+0.025 R²) — widening the
KFold-vs-PGTS gap by more than 2×. Mechanism: rolling stats computed from
nearby same-engine cycles hand row-level leakage an even easier way to
"recognize" a partially-seen trajectory, without adding real
unseen-engine generalization signal. Simple rolling-window features are not
a fix for this pipeline's KFold leakage — and this experiment incidentally
provides independent confirmation of the leak mechanism in finding 1.

**Practical takeaway**: don't trust `results/tables/cv_scores_r2.csv` /
`cv_scores_mse.csv` as leakage-safe (they were used for §5's significance
testing with this caveat explicitly noted there) — re-tune with
`GroupKFold`/PGTS grouped by `engine_id` before treating small differences
among the top-cluster models as meaningful. **Phase 2's `nested_cv.py` is
exactly this fix, done properly (nested tuning, not just re-evaluation).**

### 8. Repo structure (Phase 1)

```
rul-bench/
  CLAUDE.md, PROJECT_BRIEF.md    project rules and methodology reference
  data/
    raw/                        NASA C-MAPSS FD001 (train/test/RUL/readme)
    processed/                  cleaned, labeled, scaled train.csv/test.csv + DATA_DICTIONARY.md
  src/
    preprocessing.py            imputation check, RUL labeling+capping, scaling
    features.py                 variance-threshold selection, rolling-window features
    models.py                   10 individual models, GridSearchCV, CV scoring
    ensembling.py                fixed-weighted + stacking ensembles
    pgts_evaluation.py          PGTS re-evaluation of trained models
    pgts_rolling_extension.py   rolling-feature extension experiment
    shap_analysis.py            SHAP global/local analysis
  results/
    models/                     saved model artifacts (joblib)
    tables/                     all metrics, CV scores, stats tests, PGTS comparison
    shap/                       SHAP plots + concentration metrics
    STATS_SUMMARY.md, LEAKAGE_REPORT.md, SHAP_ANALYSIS.md
  .claude/
    agents/                     scoped subagent definitions (data-engineer, model-trainer, stats-auditor, leakage-red-team, interpretability-analyst)
    skills/                     phm08-scoring, pgts-split, stats-suite (tested, deterministic)
  requirements.txt
```

(See the Phase 2 section above for everything added since.)

### 9. Reproducing Phase 1

```bash
pip install -r requirements.txt
python src/preprocessing.py       # or run via features.py, which calls it
python src/features.py            # writes data/processed/{train,test}.csv
python src/models.py              # trains 10 models, writes results/models/, results/tables/
python src/ensembling.py          # both ensemble variants
python src/pgts_evaluation.py     # PGTS leakage-safe re-evaluation
python src/pgts_rolling_extension.py
python src/shap_analysis.py       # SHAP plots + concentration metrics
```

Every stage uses `random_state=42` (or the model's equivalent seed
parameter) — re-running should reproduce these numbers exactly, modulo
library-version floating-point differences.

### 10. Non-negotiables this project followed

- No fabricated numbers anywhere, including intermediate/draft artifacts —
  every metric in this README and in `results/` came from an executed run.
- Fixed random seeds everywhere (models, splits, bootstrap resampling).
- Statistical claims (§5) are backed by bootstrap CIs and/or ANOVA+Tukey /
  Kruskal-Wallis+Dunn, not point estimates alone.
- The official split's headline numbers were not trusted until the PGTS
  leakage-red-team audit (§7) ran — and that audit's actual findings are
  reported even where they contradict the source paper's own narrative.
- **Phase 2 extended this same discipline to the project's own selection
  process, which Phase 1 had not yet fully applied it to.**

---

## Citation

If referencing this work, please cite the original paper:

```bibtex
@article{Ozcan2025InterpretableRUL,
  author  = {Özcan, H.},
  title   = {Interpretable ensemble remaining useful life prediction enables dynamic maintenance scheduling for aircraft engines},
  journal = {Scientific Reports},
  year    = {2025},
  volume  = {15},
  pages   = {39795},
  doi     = {10.1038/s41598-025-23473-2}
}
```

This repository is an independent reproduction and extension, not an
official release from the paper's authors.

