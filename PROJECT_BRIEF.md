# RUL-Bench — Reproducing & Stress-Testing a Published Predictive Maintenance Ensemble

## Goal

Reproduce the core results of a peer-reviewed paper on turbofan engine
Remaining Useful Life (RUL) prediction, verify the authors' own statistical
validation, and extend the study by rebuilding their leakage-safe critique of
their own headline result. This is a portfolio project for a Data Scientist
role — the emphasis is on methodological rigor (reproducibility, correct
evaluation splits, statistical significance testing), not just hitting a
target metric.

See `CLAUDE.md` at the repo root for the non-negotiable project rules
(no fabricated numbers, fixed seeds, etc.) and for which subagent/skill
handles which part of this plan. This file is the detailed methodology
reference; CLAUDE.md is the always-loaded rules summary.

## Source paper

Özcan, H. "Interpretable ensemble remaining useful life prediction enables
dynamic maintenance scheduling for aircraft engines." *Scientific Reports*
15, 39795 (2025). https://doi.org/10.1038/s41598-025-23473-2

Their reference implementation (use to sanity-check your pipeline, not to
copy wholesale): https://github.com/hkmtcn/interpretable-rul-maintenance

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.
Source: NASA Ames Prognostics Data Repository —
https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
(also mirrored on Kaggle for convenience).

Start with **FD001** (single operating condition, single fault mode — the
subset the paper's headline number is based on). FD002–FD004 are optional
stretch goals once FD001 is solid.

Format: space-separated columns, no header. 26 columns = engine ID, cycle
number, 3 operational settings, 21 sensor readings. Train file has
run-to-failure trajectories; test file is truncated mid-life with true RUL
given separately in `RUL_FD001.txt`.

## Build plan

### 1. Preprocessing — `data-engineer` subagent
- Handle missing/invalid values (mean/median imputation).
- Scale features (min-max or z-score standardization) — fit on train, apply
  to test only.
- Compute RUL label per row as `max_cycle_for_engine - current_cycle`.
- Apply **piecewise-linear RUL capping**: cap RUL at a plateau (commonly
  ~125 cycles) so early-life cycles aren't penalized as if actively
  degrading. Document whatever cap value is chosen and why.

### 2. Feature engineering — `data-engineer` subagent
- Drop near-constant/uninformative sensor channels (variance threshold
  check — verify against this specific data, don't assume from the paper).
- Optional: rolling-window statistics (mean/std/slope) per sensor.

### 3. Models to train — `model-trainer` subagent
Individual models (scikit-learn / lightgbm / catboost / xgboost):
LightGBM, CatBoost, Gradient Boosting, XGBoost, SVM, KNN, Linear Regression,
Ridge, Bayesian Ridge, MLP.

Hyperparameters via `GridSearchCV`, fixed random seeds throughout.

### 4. Ensembling — `model-trainer` subagent, build both variants
- **Fixed weighted average**: 70% weight to the stronger single model (by
  validation performance), 30% to the other.
- **Stacking**: ridge-regression meta-learner trained on out-of-fold
  predictions of the base learners (not raw features). Proper out-of-fold
  generation is required — no leakage into the meta-learner.

### 5. Evaluation metrics — `model-trainer` subagent, using the `phm08-scoring` skill
- R², MSE, MAE (standard sklearn implementations).
- **PHM08 RUL Score** — use `.claude/skills/phm08-scoring/score.py`, don't
  reimplement inline. See that skill's SKILL.md for the sign-convention
  caveat before comparing against any paper's reported number.

### 6. Statistical validation — `stats-auditor` subagent, using the `stats-suite` skill
This is what makes it a "statistical modeling" project, not just an
ML-fitting exercise:
- 5-fold cross-validation, collect per-fold scores per model.
- Shapiro-Wilk test for normality of the score distributions.
- Levene's test for homogeneity of variance across models.
- Bootstrap confidence intervals on MSE (~10,000 resamples).
- One-way ANOVA across model configurations.
- Tukey HSD post-hoc test to determine which models are *significantly*
  different from each other, not just numerically different.

Use `.claude/skills/stats-suite/stats_tests.py` — don't re-derive this
pipeline from scratch each session.

### 7. Interpretability — `interpretability-analyst` subagent
- SHAP summary plots (global) for each individual model and the ensemble.
- SHAP force plots (local) for a handful of representative predictions.
- Identify and report which sensors dominate the model's decisions, tied to
  physical sensor meaning where the dataset documentation allows it.

### 8. The critical extension — leakage-safe re-evaluation — `leakage-red-team` subagent, using the `pgts-split` skill
- Re-run evaluation using a **Purged Group Time Series Split (PGTS)** instead
  of the official split: window=30, horizon=1, splits=5, embargo=10 (and,
  for comparison, embargo=0). Use `.claude/skills/pgts-split/pgts.py`.
- The paper reports the official-split R² of ~0.99 collapses to strongly
  negative R² under PGTS — reproduce this finding independently.
- Then go further than the paper: investigate *why*, and whether better
  temporal feature engineering, sequence-aware modeling, or a different
  capping strategy can close some of that gap. This is the part of the
  project that's your own contribution, not a reproduction. A negative
  result here (nothing closes the gap) is still a legitimate, useful
  finding — report it honestly rather than forcing a positive spin.

## Suggested repo structure

```
rul-bench/
  CLAUDE.md
  PROJECT_BRIEF.md
  .claude/
    agents/
      data-engineer.md
      model-trainer.md
      stats-auditor.md
      leakage-red-team.md
      interpretability-analyst.md
    skills/
      phm08-scoring/{SKILL.md, score.py}
      pgts-split/{SKILL.md, pgts.py}
      stats-suite/{SKILL.md, stats_tests.py}
  data/                  # raw + processed C-MAPSS files (gitignore raw if large)
  src/
    preprocessing.py
    features.py
    models.py
    ensembling.py
    evaluation.py
    shap_analysis.py
  notebooks/             # exploratory work only
  results/                # actual run outputs — tables, plots, saved models
  README.md
  requirements.txt
```

## Deliverables / success criteria

1. A working pipeline that trains all listed models on FD001 and reports
   real R²/MSE/MAE/RUL Score under the official split.
2. A results table comparing your numbers to the paper's reported numbers,
   with commentary on any discrepancy (seed, library version, preprocessing
   choices).
3. The full statistical test suite implemented and reported (not just point
   estimates) — including explicit assumption checks, not just the headline
   ANOVA/Tukey result.
4. SHAP plots with a short interpretation of which sensors matter.
5. The PGTS leakage-safe re-evaluation reproduced, plus original analysis of
   whether/how the gap can be narrowed.
6. A README that's honest about what's reproduction vs. original
   contribution, and cites the source paper.

## Resume bullet (fill in only once real numbers exist — see CLAUDE.md rule 1)

> Reproduced and statistically validated a peer-reviewed LightGBM/CatBoost
> ensemble for aircraft engine RUL prediction (NASA C-MAPSS), independently
> confirming published R²/RMSE via [N]-fold CV with bootstrap confidence
> intervals, ANOVA, and Tukey HSD significance testing; extended the study
> by reproducing the authors' Purged Group Time Series Split critique and
> investigating [whatever you actually find] to narrow the leakage-safe
> performance gap.
