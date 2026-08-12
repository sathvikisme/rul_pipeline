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
applied ML — reproducible splits, real statistical significance testing
(not just point estimates), and an adversarial audit of the evaluation
methodology itself. Sections below are explicitly labeled **[Reproduction]**
or **[Extension]** so it's clear what came from the paper versus what is
this project's own contribution. All numbers in this document come from
code in this repository that was actually executed — see each `results/`
artifact referenced inline.

---

## 1. Dataset — NASA C-MAPSS FD001

Source: NASA Ames Prognostics Data Repository (mirrored copy used here:
[github.com/edwardzjl/CMAPSSData](https://github.com/edwardzjl/CMAPSSData),
verified byte-identical in row/file size to the original NASA distribution).

FD001: single operating condition (sea level), single fault mode (HPC
degradation). 100 training engines (run-to-failure trajectories, 20,631
rows), 100 test engines (truncated trajectories, 13,096 rows), true RUL for
each test engine given separately in `RUL_FD001.txt`. 26 raw columns: engine
ID, cycle, 3 operational settings, 21 sensor readings.

## 2. Preprocessing & feature engineering **[Reproduction, with independent verification]**

Full detail: `data/processed/DATA_DICTIONARY.md`, `src/preprocessing.py`, `src/features.py`.

- **Missing values**: checked directly — 0 NaNs, 0 duplicate rows in both
  train and test. No imputation applied (nothing to impute).
- **RUL labels**: `max_cycle_for_engine − cycle` per row (train); back-computed
  from `RUL_FD001.txt`'s true value at each test engine's last observed cycle
  (test).
- **Piecewise-linear RUL capping** at **125 cycles** (literature convention,
  e.g. Heimes 2008; consistent with Özcan 2025) — chosen and documented
  explicitly, not silently hardcoded.
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
  §5 for why adding them was tested and rejected as an extension.

## 3. Models & ensembles **[Reproduction]**

10 individual models tuned via `GridSearchCV` (5-fold `KFold`,
`random_state=42` throughout, `neg_mean_squared_error` scoring): LightGBM,
CatBoost, Gradient Boosting, XGBoost, SVM, KNN, Linear Regression, Ridge,
Bayesian Ridge, MLP. Full grids and selected hyperparameters:
`results/tables/best_hyperparams.json`.

Two ensemble variants, both built on top of the tuned base learners:

- **Fixed weighted average**: 70% XGBoost + 30% MLP (the two strongest
  models by official test R²: 0.6776 vs 0.6775 — a near-tie; CatBoost and
  LightGBM trail by <0.002 R², noted in `results/tables/ensembling_config.json`).
- **Stacking**: Ridge meta-learner (`alpha=100`, grid-searched), trained
  **only on genuine out-of-fold predictions** of 5 base learners (LightGBM,
  CatBoost, XGBoost, GradientBoosting, MLP — the clear top tier by CV score;
  KNN/SVM and the three near-identical linear models were excluded as
  weaker or redundant) generated via `cross_val_predict` — never on raw
  features, never on in-fold predictions. Mechanics: `results/tables/ensembling_config.json`.

## 4. Results — official split **[Reproduction]**

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

Full table: `results/tables/official_split_metrics.csv`.

### Comparison to the source paper — and an honest discrepancy

Özcan (2025) reports, for FD001, a LightGBM+CatBoost ensemble at
**RMSE = 6.62, RUL Score ≈ 2,951**, implying R² in the neighborhood of 0.99.
This reproduction's best result — the Stacking ensemble at **RMSE = 15.55,
RUL Score ≈ 76,738** — falls well short of that. We do **not** paper over
this gap. Candidate explanations, investigated where possible:

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

## 5. Statistical validation **[Reproduction of methodology, extension in rigor]**

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
resampling noise given the CI overlap.

## 6. Interpretability — SHAP **[Reproduction of methodology + a check on the paper's specific claim]**

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
  models' decisions is the expected, not spurious, result.
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
  depends on which individual model you compare against.

## 7. The critical extension — leakage-safe (PGTS) re-evaluation **[Extension]**

Full report: `results/LEAKAGE_REPORT.md`. Skill: `.claude/skills/pgts-split/pgts.py`
(window/embargo protocol: `n_splits=5`, `embargo=10` and `embargo=0`).

This is the project's own contribution beyond reproduction: adversarially
re-checking whether *this reproduction's* evaluation methodology is lying to
us, independent of whatever the original paper's number is.

**1. A real, modest leak was found and quantified** — not in the official
train/test split, but in `src/models.py`'s `KFold(5, shuffle=True,
random_state=42)`, used for both GridSearchCV tuning and the CV scores §6
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
(verified via `_assert_no_leakage` passing on all 10 real folds).

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
number here to "collapse" in the first place.

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
`cv_scores_mse.csv` as leakage-safe (they were used for §6's significance
testing with this caveat explicitly noted there) — re-tune with
`GroupKFold`/PGTS grouped by `engine_id` before treating small differences
among the top-cluster models as meaningful.

## 8. Repo structure

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

## 9. Reproducing this

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

## 10. Non-negotiables this project followed

- No fabricated numbers anywhere, including intermediate/draft artifacts —
  every metric in this README and in `results/` came from an executed run.
- Fixed random seeds everywhere (models, splits, bootstrap resampling).
- Statistical claims (§6) are backed by bootstrap CIs and/or ANOVA+Tukey /
  Kruskal-Wallis+Dunn, not point estimates alone.
- The official split's headline numbers were not trusted until the PGTS
  leakage-red-team audit (§7) ran — and that audit's actual findings are
  reported even where they contradict the source paper's own narrative.

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
