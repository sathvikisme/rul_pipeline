# RUL-Bench Reproduction Report

Reproduction and methodological stress-test of Özcan, H. "Interpretable
ensemble remaining useful life prediction enables dynamic maintenance
scheduling for aircraft engines." *Scientific Reports* 15, 39795 (2025).
https://doi.org/10.1038/s41598-025-23473-2 — on NASA C-MAPSS FD001. This
report covers both Phase 1 (initial reproduction) and Phase 2 (methodological
repair, undertaken after an audit found Phase 1's evaluation was itself
contaminated by test-set-driven model selection). Every number below comes
from code in this repository that was actually executed; `results/tables/`
and `results/*.md` hold the underlying artifacts referenced throughout.

---

## 1. Executive Summary

Phase 1 built a full pipeline — preprocessing, 10 tuned models, two
ensembles, statistical validation, SHAP interpretability, and a PGTS
leakage audit — and got real, reproducible results (LightGBM R²=0.6754,
best Phase-1 ensemble R²=0.6820 on the official split). A follow-up
methodological audit found Phase 1's own evaluation had a serious flaw:
**the ensemble weights and composition were selected by looking at official
test-set R²** — the exact test-set contamination the project's own rules
exist to prevent — plus a milder row-level (non-engine-grouped) `KFold`
leak in hyperparameter tuning. Phase 2 rebuilt the pipeline with genuinely
train-only, engine-grouped nested cross-validation, paired/engine-level
statistics, an automated pytest leakage suite, and a single frozen official
test evaluation. Separately, inspecting the source paper's own reference
notebook revealed why this project's numbers (R²≈0.68) differ so sharply
from the paper's reported R²≈0.99: **the paper's headline number is not
measured against the official NASA test set at all** — it comes from a
row-level 80/20 split of the training data only, with `engine_id` included
as a raw model feature. This project independently reproduced that exact
mechanism (§5) and confirmed it in this codebase.

## 2. Dataset

NASA C-MAPSS FD001: single operating condition (sea level), single fault
mode (HPC degradation). 100 training engines, run-to-failure (20,631 rows).
100 test engines, truncated trajectories (13,096 rows), true RUL given
separately in `RUL_FD001.txt`. Verified byte-identical in file/row counts
to the original NASA distribution (mirror source:
github.com/edwardzjl/CMAPSSData).

## 3. Original Pipeline (the source paper)

Per the paper's own reference implementation
(github.com/hkmtcn/interpretable-rul-maintenance,
`notebooks/LGMB_CatBoost.ipynb`, inspected directly for this project):
LightGBM (`n_estimators=500, learning_rate=0.1, max_depth=-1, num_leaves=31,
subsample=0.8, colsample_bytree=0.8`) + CatBoost
(`iterations=500, learning_rate=0.1, depth=6`), simple 0.5/0.5 ensemble,
`MinMaxScaler`, all 21 raw sensors + `engine_id` + `cycle` as features, no
RUL capping. Evaluated via `train_test_split(test_size=0.2, random_state=42)`
— a **row-level split of `train_FD001.txt` only**. The official
`test_FD001.txt`/`RUL_FD001.txt` is loaded and RUL-labeled in the notebook
but is **never passed to `.predict()` anywhere**, confirmed by a `grep`
across all four FD001–FD004 notebook cells (zero `predict(test` calls).

## 4. Initial Reproduction (Phase 1)

10 models (LightGBM, CatBoost, GradientBoosting, XGBoost, SVM, KNN, Linear
Regression, Ridge, Bayesian Ridge, MLP) tuned via `GridSearchCV`
(`KFold(5, shuffle=True, random_state=42)` — row-level, not
engine-grouped), evaluated once on the official test split. Best individual:
XGBoost R²=0.6776. Best ensemble (`Stacking_Ridge`): R²=0.6820, RMSE=15.55.
Full table: `results/tables/official_split_metrics.csv`. Two ensembles
built: a 70/30 fixed-weighted average (XGBoost/MLP) and a Ridge-stacking
ensemble — **both selected by comparing official test-set R²**
(`results/tables/ensembling_config.json`: `"selection_metric": "official
test-set R2"`), which is the test-set-contamination issue Phase 2 exists
to fix.

## 5. Reproduction Discrepancy — Paper vs. Phase 1

Phase 1's best result (RMSE=15.55, R²=0.682) falls far short of the paper's
reported RMSE≈6.62, R²≈0.99. This project did not accept that gap as
unexplained. `src/track_a_reproduction.py` faithfully replicated the
paper's exact protocol (§3) on this project's own copy of the raw data:

| Model | Track A reproduced R² | Paper reported R² | Gap |
|---|---:|---:|---:|
| LightGBM | 0.9888 | 0.9894 | -0.0006 |
| CatBoost | 0.9864 | 0.9872 | -0.0008 |
| Ensemble (0.5/0.5) | 0.9898 | 0.9904 | -0.0006 |

**Essentially an exact match** (residual gap consistent with minor library
version differences). This confirms the paper's number is real and
reproducible — it is measuring a fundamentally different, easier task (a
row-level holdout of the training trajectories, with `engine_id` available
as a feature) than the standard held-out-engine FD001 benchmark this
project's official-split numbers (§4, §9) are measuring. The paper's own
notebook independently confirms this: running the same data/features
through `GroupKFold(5)` (grouped by engine) collapses LightGBM to
R²≈0.4372; this project reproduced that collapse too (R²=0.4421,
`results/tables/track_a_groupkfold_collapse.csv`). An ablation removing
`engine_id`/`cycle` as features recovers R²≈0.5964 (paper) /
0.5974 (this reproduction, `results/tables/track_a_id_cycle_ablation.csv`)
— confirming `engine_id`-as-feature is the dominant leakage mechanism: it
only helps when the same engine appears on both sides of a split, which is
exactly what the paper's row-level `train_test_split` allows and the
official NASA test-engine split does not.

A permutation-null sanity test (shuffle RUL within each engine) gives
R²≈0.07–0.09 for both models — small but not exactly zero, expected because
`engine_id` retains a per-engine-mean-RUL signal even after the
within-engine cycle↔RUL relationship is destroyed (`results/tables/
track_a_permutation_null.csv`). This is a legitimate residual, not evidence
the permutation test itself is broken.

One additional, previously-undocumented finding: the paper's own code
applies **no RUL capping at all** (train RUL range came out uncapped,
[0, 361]) — differing from this project's own (independently justified)
cap=125 convention. See §11 for the ablation this motivated.

## 6. Leakage Audit

Two independent, complementary leaks were found and quantified in this
project's own pipeline (distinct from the paper's `engine_id`-driven leak
in §5):

1. **Test-set-driven model selection (Phase 1)** — the most serious issue.
   Ensemble composition/weights chosen by looking at official test R².
   Fixed in Phase 2 by a training-only freeze decision (§8, §10).
2. **Row-level `KFold` in hyperparameter tuning/CV scoring (Phase 1)** —
   `src/models.py`'s plain `KFold(shuffle=True)` let same-engine cycles
   split across train/validation. Quantified via the original PGTS
   re-evaluation (`results/LEAKAGE_REPORT.md`): ~1.5–2 R² points / ~11–13%
   MSE inflation — real but modest, not catastrophic. **A live,
   in-this-codebase demonstration of the same mechanism** was added in
   Phase 2 (`tests/test_deliberate_leak_injection.py`): deliberately
   swapping Track B's `GroupKFold` for row-level `KFold` on the same data
   produces LightGBM R²=0.8575 vs. the real Track B R²=0.8457
   (`results/tables/clean_grouped_cv_metrics.csv`) — reproducibly, +0.0119
   R² from the same-engine leak alone, and `pgts._assert_no_leakage`
   correctly raises on the broken variant, confirming the leakage
   assertions have teeth, not just tautological passes.

Notably, the **official train/test engine split itself was never found to
be leaking** — `results/LEAKAGE_REPORT.md`'s PGTS re-evaluation and Phase
2's frozen final evaluation (§10) both show the official split behaving
reasonably; the contamination was in the internal selection process, not
the benchmark split.

## 7. Methodological Repairs (Phase 2)

Full detail: `results/FREEZE_DECISION.md`, `data/processed/rul_cap_ablation/
MANIFEST.md`, `.claude/skills/paired-stats/SKILL.md`. Summary of fixes,
each in new files, with every Phase 1 file preserved unmodified as the
historical record:

- **Fold-safe preprocessing** (`src/fold_safe_pipeline.py`): variance
  threshold and scaling fit inside each CV fold, never globally.
- **Nested, engine-grouped CV** (`src/nested_cv.py`): outer `GroupKFold(5)`
  / inner `GroupKFold(3)`, tiered tuning budget (full `RandomizedSearchCV`
  for LightGBM/CatBoost/XGBoost; fixed Phase-1 hyperparameters refit
  per-fold for GradientBoosting/MLP/SVM/KNN; small exhaustive grids for the
  linear models) — ~5.6 minutes total, well under the ~10–18 min budget
  estimate.
- **Genuinely nested stacking OOF** (`src/nested_stacking.py`): meta-learner
  trained on real outer-fold-refit OOF predictions, not `cross_val_predict`.
- **Train-only ensemble/model selection** (`results/FREEZE_DECISION.md`):
  no decision after this point used `data/processed/test.csv`.
- **Paired, engine-level statistics** (`.claude/skills/paired-stats/`,
  `src/stats_auditor_track_b.py`): permutation/Wilcoxon tests on paired
  per-fold scores with Holm correction; a 100-engine (not 13,096-row)
  cluster bootstrap for ΔMSE confidence intervals.
- **Automated leakage tests** (`tests/`, pytest, 11/11 passing): group
  overlap, test-set isolation, fold-only-fit assertions, stacking OOF
  purity, `groups=` propagation regression guard, single-final-eval-log
  check, and the deliberate-leak-injection smoke test (§6).
- **RUL-cap ablation** (§11).
- **Single frozen official evaluation** (§10).

## 8. Clean Evaluation — Nested GroupKFold Results

`results/tables/clean_grouped_cv_metrics.csv` (mean ± std across 5 outer
folds, engine-grouped, leakage-safe):

| Model | R² | MSE | PHM08 |
|---|---:|---:|---:|
| CatBoost | 0.8461 ± 0.0217 | 267.27 ± 37.64 | 24174 ± 4977 |
| LightGBM | 0.8457 ± 0.0228 | 267.98 ± 39.54 | 24603 ± 4515 |
| XGBoost | 0.8446 ± 0.0243 | 269.94 ± 42.13 | 25269 ± 5319 |
| GradientBoosting | 0.8439 ± 0.0220 | 271.01 ± 38.18 | 25458 ± 4618 |
| MLP | 0.8432 ± 0.0176 | 272.23 ± 30.66 | 28238 ± 6200 |
| SVM | 0.8380 ± 0.0265 | 281.39 ± 46.06 | 30839 ± 7717 |
| KNN | 0.8377 ± 0.0253 | 281.79 ± 43.93 | 31537 ± 6791 |
| Ridge / BayesianRidge / LinearRegression | 0.7676 ± 0.035 | ~403.6 | ~31,570 |

**Sanity cross-check**: these nested-GroupKFold numbers land within ~0.005
R² of the independently-computed PGTS(embargo=0) numbers in
`results/tables/pgts_comparison.csv` for the same three tree models
(e.g. CatBoost: 0.8461 vs. 0.8420) — two structurally different but both
leakage-safe protocols converge on the same answer, a meaningful
consistency check that the new pipeline is measuring the real thing.

## 9. Statistical Analysis

Full detail: `results/tables/PAIRED_STATS_SUMMARY.md`.

- **Per-fold paired tests (permutation + Wilcoxon, Holm-corrected, 55 pairs
  × 2 metrics): 0 of 110 comparisons reach significance.** This is a
  resolution-floor artifact of n=5 outer folds (minimum achievable
  corrected p-value floor confirmed directly in the skill's own sanity
  check), not evidence that nothing differs — 44 of 55 pairs *are*
  practically significant (≥1% ΔMSE) despite not clearing the corrected
  bar. This finding is **more conservative than Phase 1's own already-
  conservative Dunn/Holm fallback** (which found 6/45 significant pairs).
- **Engine-level paired bootstrap (100 engines, 10,000 resamples): 25 of
  45 pairs exclude zero** — the tree/boosting/MLP cluster is genuinely and
  significantly better than the linear models and KNN/SVM. **Within the
  top tier, only 1 of 10 pairs is distinguishable**: LightGBM significantly
  beats GradientBoosting (ΔMSE=-3.03, 95% CI [-5.22, -0.90]). The closest
  pair, LightGBM vs. CatBoost, is not distinguishable (ΔMSE=+0.71, CI
  [-3.50, 4.69]).
- **Does the stacking ensemble beat the best individual model?** No —
  StackingEnsemble edges out CatBoost by ΔMSE=2.48 (0.9%, below the 1%
  practical-significance threshold), permutation p=0.3736, p_corrected=1.0.
  Not distinguishable.

## 10. Ensemble Evaluation and the Freeze Decision

Per Issue 20's rule (don't force a ranking when models are statistically
tied), the freeze decision (`results/FREEZE_DECISION.md`) froze **two**
candidates for the single official test evaluation, chosen entirely from
training-engine evidence:

- **LightGBM** — cheapest model in the statistically-tied top cluster
  (~5.5s/fold vs. CatBoost's ~20s/fold), the one confirmed pairwise winner
  in the whole top tier (vs. GradientBoosting), simplest to interpret, and
  half of the paper's own reference architecture.
- **StackingEnsemble** — numerically best point estimate and the literal
  subject of the paper ("interpretable ensemble RUL prediction"), frozen to
  honestly test whether its added complexity pays off on real held-out data.

**Frozen official test-set result** (`results/tables/final_test_metrics.csv`,
`results/FINAL_TEST_RESULT.md`; the only read of `data/processed/test.csv`
in Phase 2, logged and guarded against a second read):

| Model | R² | MSE | MAE | PHM08 |
|---|---:|---:|---:|---:|
| LightGBM | 0.6800 | 243.37 | 9.98 | 71,102 |
| StackingEnsemble | 0.6793 | 243.95 | 10.02 | 74,097 |

**Essentially a tie**, exactly as the training-only evidence predicted —
the leakage-safe process did not just happen to agree with Phase 1's
test-set-selected choice by luck; it produced a defensible, evidence-based
non-decision that the actual held-out numbers confirm.

Compared to Phase 1's numbers for the closest-analogue architectures (not
identical methodology — see caveat in `FINAL_TEST_RESULT.md`): LightGBM
moved from R²=0.6754 to R²=0.6800 (+0.0047); the best stacking ensemble
moved from R²=0.6820 to R²=0.6793 (-0.0027). Both small. This is the
expected shape: the *official* split was never the leaky part (§6) — fixing
test-set-driven selection mainly changed *which* hyperparameters/ensemble
composition got used, not whether the official-split evaluation itself was
inflated.

## 11. RUL-Cap Ablation

`results/tables/ablation_matrix.csv` (Tier-1 models, nested-CV, 3 variants
sharing identical features): A (train-cap-only, test RUL raw), B (both
capped at 125 — Phase 1's choice), C (neither capped). A and B are
numerically identical in this train-only ablation (expected — both share
the same capped training target; the two variants only differ on the
test-side label, which a train-only nested-CV ablation structurally cannot
see). Variant C (fully uncapped): R² drops from ~0.845–0.848 (A/B) to
~0.708–0.713 — a substantial drop, but **not directly comparable** to A/B's
numbers, since removing the cap changes what "true RUL" means for both the
R² denominator and the PHM08 exponential penalty (explicit caveat embedded
in the CSV itself). This project's cap=125 choice remains independently
justified (§7 of `README.md`'s Phase 1 section; literature convention), and
is now known to differ from the paper's own uncapped convention (§5) — a
genuine, previously-undocumented methodological difference between this
reproduction and the source paper, on top of the split-methodology
difference.

## 12. SHAP / Interpretability

Full detail: `results/SHAP_ANALYSIS_TRACK_B.md` (Phase 2, frozen models) and
`results/SHAP_ANALYSIS.md` (Phase 1, historical, kept unmodified).
Phase 2's SHAP analysis reconstructed the exact frozen models (verified: R²/
MSE/MAE/PHM08 matched `final_test_metrics.csv` to floating-point precision)
and ran `TreeExplainer` on the tree-model components plus the linear
meta-learner decomposition for the ensemble. **Top features replicate
Phase 1's finding**: `cycle`, `sensor_11` (Ps30, HPC outlet static
pressure), `sensor_9`/`sensor_14` (Nc/NRc, core speed), `sensor_4` (T50,
LPT outlet temperature) — all HPC/core-gas-path channels, consistent with
FD001's documented HPC-degradation fault mode, surviving a completely
independent hyperparameter-selection and ensemble-architecture process.
The one ranking shift (ensemble moved from Ps30-led in Phase 1 to
core-speed-led in Phase 2) is fully explained by the change in meta-learner
weights (CatBoost/MLP dominate Phase 2's Ridge coefficients), not a new
physical disagreement. Attribution concentration (Gini coefficient):
StackingEnsemble (0.481) is more balanced than 4 of its 5 base learners but
not CatBoost (0.463) alone — the same partial support for the paper's
"ensembling balances credit" claim found in Phase 1 with a different
ensemble architecture. As in Phase 1, all SHAP findings are reported as
model attribution, not physical causation.

## 13. Final Official Test

See §10 — the single frozen evaluation, executed exactly once, logged, and
guarded (a second invocation of the same script correctly raises
`RuntimeError` and refuses to run; verified independently).

## 14. Limitations

- FD001 is small (100 train / 100 test engines) and single-fault/single-
  condition — results may not generalize to FD002–FD004 or real fleets.
- C-MAPSS is a simulated dataset; RUL labeling conventions (capping, or
  its absence, as this project's own §11/§5 findings show) materially
  change reported metrics, independent of model quality.
- Track B's feature *selection* (which sensors survive variance
  thresholding) was still made once, globally, by Phase 1's offline
  pipeline before any Phase 2 CV loop existed — fold-local re-fitting
  (scaling, threshold refitting) removes the fold-boundary leakage that
  matters most, but the original sensor-inclusion decision is a smaller,
  documented residual limitation, not a full re-derivation from scratch.
  (`src/nested_cv.py`'s own docstring flags this explicitly.)
  - n=5 outer folds is a hard statistical constraint (§9) — with only 100
    engines and a need for grouped, leakage-safe splits, finer-grained
    significance claims are not achievable without more data or a
    different resampling design (the engine-level bootstrap partially
    compensates but cannot fully substitute for more independent folds).
- The stacking ensemble's engine-level bootstrap comparison could not be
  computed (no per-row OOF predictions were retained for it) — its
  significance conclusions rest on the less-powerful per-fold test alone.
- This is a benchmark-specific study; the paper's own abstract notes
  results should be read as benchmark-specific until confirmed on real
  fleets and alternative risk-aware policies.

## 15. Conclusions

Claims this project can defend with evidence:

1. The tree/boosting/MLP model cluster is genuinely and significantly
   better than linear models and KNN/SVM on FD001, under leakage-safe,
   engine-grouped evaluation (§9).
2. Within that top cluster, essentially no pairwise ranking is statistically
   defensible except LightGBM > GradientBoosting — including the specific
   claim "ensembling beats the best individual model," which does **not**
   hold up under either Phase 1's or Phase 2's statistical testing (§9).
3. The paper's headline R²≈0.99 is real and independently reproducible
   (§5) — but it measures a fundamentally different, leakage-prone task
   (row-level holdout of training data, `engine_id` as a feature) than the
   standard held-out-engine FD001 benchmark. This project's R²≈0.68–0.68
   is the answer to the harder, standard question.
4. Fixing test-set-driven model selection (Phase 1's most serious flaw)
   changed *which* model/hyperparameters got frozen, but only modestly
   moved the final official-split numbers (§10) — because the official
   NASA split itself was never the leaky part of this pipeline.
5. Physical plausibility of the dominant sensors (HPC/core-path channels)
   held up across two independently-tuned model generations (§12),
   the strongest interpretability finding of this project.

Claims explicitly **not** made: that this project's numbers "beat" or
"disprove" the source paper (they answer a different, harder question, not
a better answer to the same one); that any single top-tier model or the
stacking ensemble is proven best (statistically indistinguishable, reported
as such); that SHAP attributions establish physical causation.
