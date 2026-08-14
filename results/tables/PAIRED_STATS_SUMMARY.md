# Paired Statistical Validation — Phase 2 Track B (stats-auditor)

Produced by `src/stats_auditor_track_b.py`, using the new
`.claude/skills/paired-stats/paired_stats.py` skill. This is the paired,
leakage-safe companion to Phase 1's `results/tables/STATS_SUMMARY.md`
(row-level `KFold` ANOVA/Tukey + row-level bootstrap) — **Phase 1's files
are unchanged and remain the historical record**; this document explains
where the two agree and where the paired analysis is more conservative.

All numbers below come from actually executing `src/stats_auditor_track_b.py`
against `results/tables/nested_cv_metrics.csv` (10 models × 5 outer
`GroupKFold` folds, from model-trainer's leakage-safe nested pipeline),
`results/tables/nested_stacking_metrics.csv` (stacking ensemble, same 5
outer folds — verified by matching `n_train_rows`/`n_test_rows` per fold
before treating its scores as paired with the base models'), and
`results/tables/nested_cv_oof_predictions.csv` (20,631 per-row OOF
predictions across 100 engines). Fixed seed 42 throughout (permutation
test, Holm correction is deterministic, and engine bootstrap).

## Why this analysis exists (methodological issue being fixed)

Every model here is scored on the **identical** 5 outer `GroupKFold` folds
(same held-out engines each fold) — a paired design, not independent
samples. Phase 1's ANOVA/Tukey HSD treats per-fold scores as independent
groups, which is the wrong model for this data and distorts the variance
estimate. Separately, Phase 1's bootstrap resampled individual test rows as
if independent when they are really ~131 correlated cycles nested within
each of 100 engines, which understates true resampling uncertainty (CIs too
narrow). This analysis uses `paired_permutation_test` / `wilcoxon_signed_rank`
(on paired per-fold scores) and `paired_engine_bootstrap` (resampling the
100 engines, not the 20,631 rows) to address both.

---

## 1. Per-fold paired tests (`results/tables/model_pairwise_tests.csv`)

11 entities (10 base models + the nested stacking ensemble) → C(11,2) = 55
pairwise comparisons, run separately for R² and MSE (110 rows total).
Holm-Bonferroni correction (see rationale in `paired-stats/SKILL.md` — kept
consistent with Phase 1's Dunn's-test correction choice) applied **within
each metric's full 55-pair family**, not pair-by-pair.

### Headline finding: zero pairs survive Holm correction, at either metric

**0 of 55 pairs are `statistically_significant` (Holm-corrected p < 0.05)
on R², and 0 of 55 on MSE.** This is a mathematical consequence of n=5
outer folds, not a sign that nothing differs: a sign-flip permutation test
(and exact Wilcoxon) on n=5 paired folds has a hard resolution floor of
2/2⁵ = **0.0625** for the minimum achievable two-sided uncorrected p-value —
confirmed directly in `paired_stats.py`'s own sanity check (Scenario 1: a
huge injected 25-MSE-unit effect at n=5 still only reaches p≈0.061). Every
pair here that shows a real, consistent per-fold gap lands exactly at that
floor (uncorrected permutation p = 0.060994, Wilcoxon p = 0.0625), and Holm
correction across 55 comparisons (multiplying the smallest raw p by up to
55) pushes every one of them to p_corrected = 1.0.

Concretely, e.g. **BayesianRidge vs. StackingEnsemble** (MSE): mean_delta =
+138.85, permutation p = 0.060994, wilcoxon p = 0.0625, p_corrected = 1.0,
statistically_significant = False, practically_significant = True (delta is
52% of the stacking ensemble's MSE) → verdict: "not statistically confirmed
despite ≥1% MSE effect (likely underpowered at n=5 folds)."

**44 of 55 pairs (80%) are `practically_significant`** (|ΔMSE| ≥ 1% of the
smaller model's mean MSE) on both metrics, but because none clear the
corrected significance bar, every one of those 44 gets the verdict
"not statistically confirmed despite ≥1% MSE effect (likely underpowered at
n=5 folds)" rather than "significant difference." The 11 pairs that are
*not* practically significant are almost entirely within the
near-identical linear-model trio (LinearRegression/Ridge/BayesianRidge,
which differ from each other only by regularization and produce
near-zero deltas) plus a few of the closest top-tier tree/boosting pairs.

**Honest conclusion:** with only 5 paired folds and 55 comparisons, the
per-fold paired test *cannot* declare any pair "significantly different" —
this isn't evidence that all 11 models/ensembles perform identically, it's
evidence that this specific test is underpowered at this fold count. The
engine-level bootstrap (§2, ~100 units of resolution instead of 5) is the
more informative test for this dataset and is treated as primary evidence
below.

### Comparison to Phase 1's naive ANOVA/Tukey conclusion

Phase 1 (`STATS_SUMMARY.md`, row-level `KFold`, non-grouped) found Tukey HSD
flagged 25/45 pairs significant on R² and a broad set on MSE, though its own
non-parametric Dunn/Holm fallback (added because Shapiro-Wilk flagged 2/10
models' R² distributions as non-normal) cut that down to 6/45 significant
pairs — CatBoost and MLP beating the three linear models. **This paired
Track-B analysis is even more conservative than Phase 1's own
already-conservative Dunn/Holm fallback**: 0/55 pairs reach significance
here, including the linear-vs-tree/boosting gap that Phase 1's Dunn's test
*did* manage to detect. The two results are not contradictory — Track B's
nested `GroupKFold` folds are a genuinely harder, leakage-safe evaluation
with only 5 truly independent-engine-holdout units, while Phase 1's
row-level `KFold` folds (engines could appear in both train and validation
within a fold) inflated apparent precision. The right reading is: Phase 1's
already-narrow "6/45 significant" claim should be read with even more
caution once evaluated on genuinely leakage-safe, correctly-paired folds —
the true "5-fold-of-100-engines" evaluation simply doesn't have the
resolution to support ANY pairwise significance claim on its own.

---

## 2. Engine-level paired bootstrap (`results/tables/engine_bootstrap_metrics.csv`)

45 pairs among the 10 base models (stacking ensemble excluded here — no
per-row OOF prediction file exists for it; only per-fold aggregate metrics
in `nested_stacking_metrics.csv`. Confirmed by inspecting
`nested_stacking_config.json` and the `results/tables/` directory listing;
this is a real, documented limitation, not an oversight). 100 unique
engines resampled with replacement, 10,000 resamples, seed 42, ΔMSE = MSE_a
− MSE_b computed on the identical resampled engine set for both models each
time.

**25 of 45 pairs (56%) have a 95% CI that excludes zero** — a meaningfully
higher hit rate than the per-fold test, because resampling ~100 engines
gives far more resolution than 5 folds. All 25 involve at least one of the
three linear models (LinearRegression/Ridge/BayesianRidge) or KNN/SVM vs.
the tree/boosting/MLP cluster — e.g.:

- **LightGBM vs. LinearRegression**: ΔMSE = −135.66, 95% CI = **[−171.07, −103.59]** (excludes zero)
- **CatBoost vs. BayesianRidge**: ΔMSE = −136.37, 95% CI = **[−170.91, −105.39]** (excludes zero)
- **LightGBM vs. GradientBoosting**: ΔMSE = −3.03, 95% CI = **[−5.22, −0.90]** (excludes zero — the tightest / smallest-magnitude pair that still clears the bar)

**Within the top tier (LightGBM/CatBoost/XGBoost/GradientBoosting/MLP), 9 of
10 pairs do NOT exclude zero.** The closest pair overall is
**LightGBM vs. CatBoost**: ΔMSE = **+0.71**, 95% CI = **[−3.50, 4.69]**
(includes zero — not distinguishable). The one top-tier pair that *does*
exclude zero is **LightGBM vs. GradientBoosting**: ΔMSE = −3.03, 95% CI =
**[−5.22, −0.90]** — a small but real difference (1.1% of GradientBoosting's
mean MSE, which the pairwise-test table's practical-significance flag also
marks as `practically_significant=True`). Every other top-tier pair
(including CatBoost vs. GradientBoosting, CI [−8.88, 1.45]) includes zero.

**Conclusion:** the engine-level bootstrap — the correct, leakage-safe unit
of resampling for this dataset — confirms the tree/boosting/MLP cluster is
genuinely and significantly better than the three linear models and
KNN/SVM, but does **not** support any single top-tier model (LightGBM,
CatBoost, XGBoost, GradientBoosting, MLP) as distinguishably better than
the others except the one narrow LightGBM-vs-GradientBoosting case above.

---

## 3. Does the stacking ensemble beat the best individual model?

**No — not distinguishably, by either test.** StackingEnsemble has the
lowest mean nested-CV MSE (264.79) and highest mean R² (0.8475) of all 11
entities, edging out the best individual model, CatBoost (MSE=267.27,
R²=0.8461) by ΔMSE = 2.48 (0.9%, below the 1% practical threshold) and
ΔR²=0.0014. The per-fold paired test on StackingEnsemble vs. CatBoost:
mean_delta = 2.48, permutation p = 0.3736, wilcoxon p = 0.4375, p_corrected
= 1.0 — nowhere near significant, and the effect itself doesn't even clear
the practical-significance threshold (delta_mse_pct = 0.93%,
practically_significant = False). Verdict: **"no significant difference
detected."** (Note: the engine-level bootstrap could not be run for this
specific comparison — no per-row stacking OOF predictions are available —
so this conclusion rests on the per-fold test only, which is itself
underpowered at n=5; treat "stacking is not distinguishably better than
CatBoost" as the honest, current answer, not "stacking is proven equal to
CatBoost.")

Against the other four top-tier individual models, StackingEnsemble's
advantage is larger in absolute MSE terms (e.g. +138.85 vs.
LinearRegression-family, +17.0 vs. KNN, +16.6 vs. SVM) but every one of
those comparisons also fails to clear the corrected significance bar for
the same n=5-fold resolution reason described in §1.

---

## 4. Draft ensemble selection table

See `results/tables/ensemble_selection.csv` (11 rows: 10 base models +
StackingEnsemble). Columns include nested-CV R²/MSE mean±std, fold-to-fold
stability (MSE std across the 5 outer folds), whether each model is
statistically/practically distinguishable from the best-by-mean-MSE entity
(StackingEnsemble), and placeholder `*_tbd` columns for cost,
interpretability, and physical plausibility — intentionally left as
`"TBD -- human-reviewed synthesis"` per the Phase 2 plan; those are judgment
calls for the freeze-checkpoint synthesis step, not something this script
should auto-fill.

---

## Files produced

- `results/tables/model_pairwise_tests.csv` — 110 rows (55 pairs × 2 metrics), permutation + Wilcoxon p-values, Holm-corrected, practical-significance flag, verdict.
- `results/tables/engine_bootstrap_metrics.csv` — 45 rows, engine-level paired bootstrap ΔMSE point estimate + 95% CI for every base-model pair.
- `results/tables/ensemble_selection.csv` — draft model/ensemble selection table, cost/interpretability/physical-plausibility columns TBD.
- `results/tables/PAIRED_STATS_SUMMARY.md` — this file.
- `.claude/skills/paired-stats/{paired_stats.py, SKILL.md}` — the new skill used to produce all of the above; `python paired_stats.py` sanity check passing (synthetic data, four scenarios + Holm/practical-significance/bootstrap checks).
- `src/stats_auditor_track_b.py` — the script that produced all outputs above.
