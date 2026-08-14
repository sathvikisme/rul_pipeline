# Phase 2 Freeze Checkpoint — Ensemble/Model Selection Decision

This is the human-reviewed synthesis step called for in the Phase 2 plan (Issue 20:
final model selection must not be pure lowest-RMSE). It combines
`results/tables/ensemble_selection.csv` (built by stats-auditor, with cost/
interpretability/physical-plausibility columns filled in here) and the
pairwise/bootstrap findings from `results/tables/model_pairwise_tests.csv` and
`results/tables/engine_bootstrap_metrics.csv`. After this document, no further
model/hyperparameter/feature decisions should be made based on anything other
than what's already computed on the training engines — the next step
(`src/final_eval.py`) scores the official test set exactly once.

## The numbers, plainly

Nested-GroupKFold mean MSE, best to worst: Stacking (264.79) < CatBoost (267.27)
< LightGBM (267.98) < XGBoost (269.94) < GradientBoosting (271.01) < MLP (272.23).
If this were the whole story, Stacking would just win. It isn't the whole story.

**Fold-level paired tests (n=5 folds): 0 of 55 pairwise comparisons are
statistically significant after Holm correction**, including Stacking vs. every
individual model. This is a real resolution-floor problem, not a null result to
take at face value — a sign-flip test on 5 paired folds can't produce a
corrected p-value below ~0.06 no matter how large the true effect is.

**Engine-level bootstrap (100 engines, much better resolution): only ONE
pairwise comparison in the entire top tier excludes zero** — LightGBM beats
GradientBoosting (ΔMSE = -3.03, 95% CI [-5.22, -0.90]). Every other top-tier
pair, including Stacking vs. CatBoost (ΔMSE point estimate +0.71, CI
[-3.50, 4.69] — note Stacking is *not even numerically ahead* of CatBoost in
this particular resampling view) and the fold-level Stacking-vs-CatBoost
permutation test (ΔMSE=2.48, 0.9%, below the practical-significance threshold,
p_corrected=1.0), shows no confirmed winner. (Stacking's per-row OOF
predictions weren't available for the engine bootstrap specifically — see
`PAIRED_STATS_SUMMARY.md` — so its bootstrap comparisons are limited to the
fold-level test, which also found no significant difference.)

**Conclusion: {Stacking, CatBoost, LightGBM, XGBoost, GradientBoosting, MLP}
are a statistically indistinguishable cluster**, with the single exception that
LightGBM is confirmed better than GradientBoosting specifically.

## The tie-break

Per Issue 20, when models are statistically tied, cost, stability, and
interpretability are legitimate tie-breakers — but the resulting choice must be
reported as a tie-break, not dressed up as "the winner."

- **Cost**: LightGBM/XGBoost fit in ~5s/fold; CatBoost takes ~20s/fold for a
  statistically unconfirmed edge; Stacking requires fitting all 5 base learners
  every fold plus a meta-learner — strictly the most expensive option.
- **Interpretability**: LightGBM/CatBoost/XGBoost/GradientBoosting all get exact
  `shap.TreeExplainer` attribution. Stacking's attribution requires decomposing
  through a linear meta-learner across 5 components (workable — Phase 1 already
  did an analogous trick for a 2-model fixed-weight ensemble — but more moving
  parts for a marginal, unconfirmed gain).
- **Confirmed pairwise results**: LightGBM is the only model in the top tier
  with an actual confirmed win over anything (GradientBoosting). No other model,
  including Stacking, has one.

## Decision

**Two candidates are frozen for `final_eval.py`, evaluated together in the same
single official-test-set pass — not because either is proven better, but for
two different reasons:**

1. **LightGBM** — the cheapest model in the statistically-tied top cluster, the
   one with a confirmed pairwise win, the simplest to interpret, and one half of
   the paper's own reference architecture (LightGBM + CatBoost). This is the
   "if you can only ship one model, ship the cheap one that's never been beaten"
   candidate.
2. **StackingEnsemble** — the numerically best point estimate, and the actual
   subject of the paper this project reproduces ("interpretable ENSEMBLE RUL
   prediction"). Freezing it lets the final report honestly answer "was
   ensembling worth it here?" with a real held-out number, rather than assuming
   yes because Phase 1's test-set-selected comparison said so.

**Not frozen, kept only as reference points in the report**: CatBoost (tied,
but ~4x LightGBM's cost for no confirmed gain), XGBoost, GradientBoosting
(confirmed worse than LightGBM), MLP, and the weaker tier (SVM/KNN/linear
models — Phase 1 and this analysis agree these are clearly behind).

## Why this differs from Phase 1's frozen choice

Phase 1 froze a 70/30 XGBoost/MLP fixed-weighted ensemble, selected by looking
at **official test-set R²** — the exact test-set-contamination problem this
phase exists to fix. Under the leakage-safe nested-CV comparison, XGBoost/MLP
aren't even the top two candidates by point estimate, and — more importantly —
no train-only evidence would have supported picking that specific pair over
several statistically-tied alternatives. This freeze decision is made entirely
from training-engine evidence; `data/processed/test.csv` has not been read by
any Phase 2 script prior to this document (enforced in code by
`track_b_pipeline.py`'s hard-stop guard, verified by `tests/test_leakage.py`).
