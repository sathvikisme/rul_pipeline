---
name: paired-stats
description: Run paired significance tests (sign-flip permutation test, Wilcoxon signed-rank, Holm-Bonferroni correction, practical-significance flag, engine-level cluster bootstrap) for comparing models evaluated on the SAME folds or the SAME held-out engines. Use instead of stats-suite whenever the observations being compared are paired/correlated rather than independent groups -- e.g. every model scored on identical GroupKFold outer folds, or per-row test predictions that are really clustered within ~100 engines.
---

# Paired-Comparison Statistics Suite

Companion to `.claude/skills/stats-suite/stats_tests.py`, not a replacement.
`stats-suite`'s ANOVA/Tukey HSD and row-level bootstrap assume independent
observations. In this project that assumption is false in two places:

1. **Per-fold model scores are paired, not independent groups.** Every model
   in the nested-CV pipeline (`results/tables/nested_cv_metrics.csv`) is
   evaluated on the identical 5 outer `GroupKFold` folds (same held-out
   engines each fold). Model A's fold-3 score and Model B's fold-3 score
   share the same held-out engines and are therefore correlated -- a paired
   design. Feeding these into a between-groups ANOVA/Tukey HSD (as Phase 1's
   `stats-suite` does, by design, as the documented naive/negative-control
   procedure) ignores that correlation.
2. **Test-set rows are clustered within engines, not independent.** Phase
   1's `bootstrap_ci` resampled individual test rows (~13,096 of them) as if
   independent, when they are really ~131 correlated cycles nested within
   each of 100 engines. That understates true resampling uncertainty (CIs
   come out artificially narrow).

Use `paired-stats` whenever you're comparing two models' scores that were
measured on the *same* evaluation units (same folds, same engines).

## Usage

```python
from paired_stats import (
    paired_permutation_test, wilcoxon_signed_rank,
    holm_bonferroni_correction, practical_significance_flag,
    paired_engine_bootstrap,
)

# 1) per-fold paired test (n = number of folds, e.g. 5)
perm = paired_permutation_test(scores_a, scores_b, n_permutations=10_000, seed=42)
wil = wilcoxon_signed_rank(scores_a, scores_b)

# 2) correct across the full family of pairwise comparisons (e.g. 45 pairs
#    for 10 models) -- do this ONCE per metric, across the whole family,
#    not pair-by-pair.
corrected = holm_bonferroni_correction({"A_vs_B": perm["p_value"], ...})

# 3) flag whether a significant difference is big enough to matter
flag = practical_significance_flag(delta_mse=3.2, baseline_mse=250.0, pct_threshold=1.0)

# 4) engine-level cluster bootstrap on delta-MSE, using per-row OOF
#    predictions tagged with engine_id (e.g. nested_cv_oof_predictions.csv)
boot = paired_engine_bootstrap(preds_df, "LightGBM", "CatBoost",
                                engine_col="engine_id", n_resamples=10_000, seed=42)
```

## Required order of operations

1. **Permutation test AND Wilcoxon, both** -- report both, not just one.
   They usually agree; when they disagree, say so rather than picking the
   one that supports the conclusion you want.
2. **Correct for multiple comparisons ACROSS THE WHOLE FAMILY**, not
   pairwise. With 10 models there are 45 pairs; correcting each p-value in
   isolation inflates the false-positive rate across the full table.
   `holm_bonferroni_correction` expects the entire dict of p-values for one
   metric's family of comparisons at once.
3. **Report the practical-significance flag alongside the corrected
   p-value**, always -- a statistically significant p-value with
   `practically_significant=False` should be written up as a "statistical
   tie" (real but negligible difference), not as "model A wins."
4. **Use `paired_engine_bootstrap`, not a row-level bootstrap**, for any CI
   built from per-row predictions in this project. Resampling rows instead
   of engines will produce an artificially narrow (overconfident) interval.

## Known limitation -- read before trusting a p<0.05 on 5 folds

A sign-flip permutation test (and exact Wilcoxon) on `n` paired
observations has a hard resolution floor of `2/2^n` for the minimum
achievable two-sided p-value. At `n=5` (this project's outer-fold count),
that floor is `2/32 = 0.0625` -- **no effect size, however large, can
produce p < 0.05 from 5 paired folds if every fold agrees in direction.**
This is a real mathematical property of the test, not a bug; the
`__main__` sanity check demonstrates it directly (Scenario 1). Practical
implication: expect the per-fold permutation/Wilcoxon tests in this project
to rarely clear p<0.05 on their own even for models that look meaningfully
different; treat mean_delta's sign/magnitude and the engine-level bootstrap
CI (which has ~100 units of resolution, not 5) as the more informative
signal, and say so explicitly rather than reporting "not significant" as if
it means "no real difference."

## Why Holm-Bonferroni (not Benjamini-Hochberg)

Phase 1's Dunn's-test post-hoc correction (`results/tables/dunn_*_pairs.csv`)
already used Holm-Bonferroni for the identical "45 pairwise comparisons
among a small, fixed, complete set of 10 models" problem in this project.
Holm controls the family-wise error rate (probability of ANY false
positive across the family), which is the more conservative, defensible
choice when a single false "model A beats model B" claim could get
over-quoted in the README -- and it keeps this project's multiple-comparison
methodology consistent across skills rather than switching correction
procedures depending on which test produced the p-values.

## Sanity check before trusting any real output

Run `python paired_stats.py`. It runs four synthetic scenarios (n=5 with a
huge injected effect, n=5 with no effect, n=30 with a real effect, n=30 with
no effect) plus the Holm-Bonferroni, practical-significance-flag, and
engine-bootstrap checks, and asserts the expected direction/significance in
each case. If any assertion fails, the pipeline itself is broken -- fix it
before running on real fold/prediction data.
