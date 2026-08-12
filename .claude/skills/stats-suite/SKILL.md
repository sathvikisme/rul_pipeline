---
name: stats-suite
description: Run the full statistical validation suite (Shapiro-Wilk, Levene, bootstrap CI, one-way ANOVA, Tukey HSD) on cross-validated model results. Use whenever comparing two or more models' performance and a claim of "better" or "significantly different" is being made.
---

# Statistical Validation Suite

Turns "model A's number is lower than model B's" into a defensible
significance claim — or an honest "we can't claim that" when the data
doesn't support it.

## Usage

```python
from stats_tests import run_full_suite

# scores_by_model: dict of {model_name: array of per-fold scores}
# e.g. 5-fold CV R^2 for each of LightGBM, CatBoost, GBM, ...
report = run_full_suite(scores_by_model, metric_name="R2")
```

## Required order of operations — do not skip steps

1. **Shapiro-Wilk first.** Check normality of each model's per-fold score
   distribution before running ANOVA. ANOVA assumes normality; if it's
   violated, say so in the writeup.
2. **Levene's test second.** Check variance homogeneity across models. ANOVA
   also assumes this.
3. **If either assumption is violated** (`report["assumptions_violated"]`
   is `True`), do not present the ANOVA/Tukey HSD results as the primary
   evidence. Either report them with an explicit caveat, or additionally run
   a non-parametric alternative (Kruskal-Wallis + Dunn's test — not included
   in `stats_tests.py`, add if needed) and lead with that instead.
4. **Bootstrap CIs** (`bootstrap_ci`) should accompany any point-estimate
   metric reported in the final writeup (e.g. mean MSE) — a bare mean with no
   interval invites over-claiming precision.
5. **Tukey HSD** tells you *which specific pairs* of models differ
   significantly — report the full pairwise table, not just "model A won."

## Sanity check before trusting any real output

Run `python stats_tests.py` — it runs the full pipeline on synthetic data
constructed to have a known significant difference, and asserts the ANOVA
p-value comes back below 0.05. If that assertion fails, the pipeline itself
is broken; fix it before running on real fold results.
