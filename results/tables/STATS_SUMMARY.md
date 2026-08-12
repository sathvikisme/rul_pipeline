# Statistical Validation — RUL-Bench (stats-auditor)

Ran via `.claude/skills/stats-suite/stats_tests.py` (sanity check passed:
synthetic ANOVA p < 0.05 as asserted) plus a hand-written Kruskal-Wallis +
Dunn's test fallback (not in `stats_tests.py`; added per the skill doc and
CLAUDE.md rule 5). All numbers below are computed from the real files
`results/tables/cv_scores_r2.csv`, `cv_scores_mse.csv`, and
`test_predictions.csv` — nothing here is fabricated (CLAUDE.md rule 1).

**Caveat on the CV folds** (per model-trainer's note): the 5-fold CV split is
a plain row-level `KFold(shuffle=True, random_state=42)`, **not grouped by
engine**. Rows from the same engine trajectory can appear in both the train
and validation fold, which inflates apparent performance and is exactly the
kind of leakage CLAUDE.md rule 4 flags for the leakage-red-team subagent to
re-check with PGTS. Treat the CV-based significance results below as
"significance among these numbers as produced," not as proof any model is
leakage-safe.

---

## 1. Assumption checks — 5-fold CV, R² (10 models, n=5 folds each)

**Shapiro-Wilk (normality per model, α=0.05):**

| Model | W | p | Result |
|---|---|---|---|
| LightGBM | 0.8868 | 0.3411 | OK |
| CatBoost | 0.8550 | 0.2110 | OK |
| GradientBoosting | 0.8564 | 0.2157 | OK |
| XGBoost | 0.9030 | 0.4270 | OK |
| SVM | 0.9362 | 0.6393 | OK |
| KNN | 0.8184 | 0.1135 | OK |
| LinearRegression | 0.7702 | 0.0453 | **VIOLATED** |
| Ridge | 0.7752 | 0.0501 | OK (borderline) |
| BayesianRidge | 0.7734 | 0.0483 | **VIOLATED** |
| MLP | 0.9360 | 0.6375 | OK |

**Levene's test (variance homogeneity across all 10 models):** F = 0.1090, p = 0.9993 — OK.

**Verdict:** 2/10 models (LinearRegression, BayesianRidge) fail Shapiro-Wilk
at α=0.05, Ridge is borderline (p=0.0501). Levene passes cleanly. Because
normality is violated for the R² distributions, **ANOVA/Tukey on R² are
reported below but are NOT treated as primary evidence** — Kruskal-Wallis +
Dunn's test is the trustworthy result for R², per the stats-auditor rule to
add a non-parametric fallback rather than proceed as if nothing were wrong.
Note n=5 folds per group also gives Shapiro-Wilk very low power either way,
so even the "OK" results should not be over-read as strong normality
evidence.

**One-way ANOVA (R²):** F = 142.5952, p = 1.6287e-27 (printed as p=0.000000 at 6 decimals by the skill's default formatting; exact scipy value confirmed separately) — significant difference exists among the 10 models.

**Kruskal-Wallis (R², non-parametric, trustworthy given the Shapiro-Wilk violations):** H = 42.0146, p = 0.000003 (2.71e-06) — significant difference exists among the 10 models. This *agrees* with ANOVA on the omnibus question (some model differs), so the overall "not all models are equal" conclusion is solid.

### Where ANOVA/Tukey and Dunn *disagree* — this is the important finding

Tukey HSD (parametric, NOT primary given the violation) flags **25 of 45
pairs** as significant (p-adj < 0.05), including e.g. LightGBM vs
LinearRegression/Ridge/BayesianRidge (p-adj = 0.0).

Dunn's test with Holm correction (non-parametric, the trustworthy result)
flags only **6 of 45 pairs** as significant:

| group1 | group2 | z | p_raw | p_holm | reject |
|---|---|---|---|---|---|
| CatBoost | LinearRegression | 3.6227 | 0.000292 | 0.013118 | True |
| CatBoost | Ridge | 3.5794 | 0.000344 | 0.014811 | True |
| CatBoost | BayesianRidge | 3.6010 | 0.000317 | 0.013945 | True |
| LinearRegression | MLP | -3.5577 | 0.000374 | 0.015715 | True |
| Ridge | MLP | -3.5143 | 0.000441 | 0.017638 | True |
| BayesianRidge | MLP | -3.5360 | 0.000406 | 0.016658 | True |

Everything else — including LightGBM vs the three linear models, which
Tukey called significant — does **not** survive Holm correction under Dunn
(e.g. LightGBM vs LinearRegression: p_holm = 0.0517, just above 0.05).

**Honest conclusion for R²:** with only n=5 folds per model and 45 pairwise
comparisons, the conservative non-parametric test can only defensibly
support that **CatBoost and MLP are significantly better than the three
linear models (LinearRegression, Ridge, BayesianRidge)**. Claims like
"LightGBM significantly beats Ridge" are visible in the parametric Tukey
table but do not survive the more defensible non-parametric correction and
should not be reported as established.

Full tables: `results/tables/anova_tukey_r2.txt` (full text report),
`results/tables/tukey_hsd_r2_pairs.csv` (all 45 pairs),
`results/tables/dunn_r2_pairs.csv` (all 45 pairs).

---

## 2. Secondary check — 5-fold CV, MSE (10 models, n=5 folds each)

Run as a cross-check since R² is a nonlinear (variance-normalized) transform
of MSE per fold and can distort small-sample shape differently.

**Shapiro-Wilk:** all 10/10 models pass (p ranges 0.098–0.688) — no
violation on this metric.

**Levene's test:** F = 0.1565, p = 0.9971 — OK.

**Verdict:** unlike R², the MSE distributions show no detected assumption
violations, so **ANOVA/Tukey on MSE can be treated as primary evidence**.

**One-way ANOVA (MSE):** F = 150.7163, p = 5.5878e-28 — significant.

**Kruskal-Wallis (MSE):** H = 41.96, p = 0.000003 — agrees with ANOVA.

Tukey HSD on MSE (trustworthy here) shows a broader, more defensible set of
significant pairs than the R²/Dunn result above, e.g.: the linear-model trio
(LinearRegression/Ridge/BayesianRidge) is significantly worse (higher MSE,
p-adj = 0.0) than every tree/boosting model and MLP; KNN is significantly
worse than LightGBM, CatBoost, XGBoost, MLP, GradientBoosting (p-adj ≤
0.0002); SVM is significantly worse than every tree/boosting model and MLP
(p-adj = 0.0). Within the top cluster — LightGBM, CatBoost, GradientBoosting,
XGBoost, MLP — **no pair is significant** (all p-adj = 1.0 or close to it),
i.e. these five are statistically indistinguishable from each other on CV
MSE.

Full tables: `results/tables/anova_tukey_mse_cv.txt`,
`results/tables/tukey_hsd_mse_cv_pairs.csv`,
`results/tables/dunn_mse_cv_pairs.csv`.

---

## 3. Bootstrap CI on held-out test-set MSE (row-level, n=13,096 rows, 10,000 resamples, seed=0)

Computed from `results/tables/test_predictions.csv` (the true 100-engine
held-out set, not CV) — per-row squared error, then `bootstrap_ci` from the
stats-suite skill.

| Model | Point estimate (MSE) | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|
| Stacking_Ridge | 241.883 | 232.887 | 251.030 |
| FixedWeighted_XGBoost70_MLP30 | 242.132 | 233.263 | 251.187 |
| XGBoost | 245.189 | 236.253 | 254.400 |
| MLP | 245.318 | 236.323 | 254.480 |
| CatBoost | 246.500 | 237.320 | 255.907 |
| LightGBM | 246.917 | 237.778 | 256.306 |
| GradientBoosting | 248.115 | 238.968 | 257.392 |
| KNN | 277.592 | 267.782 | 287.655 |
| SVM | 295.049 | 283.623 | 306.724 |
| Ridge | 364.743 | 355.146 | 374.470 |
| BayesianRidge | 364.746 | 355.142 | 374.472 |
| LinearRegression | 364.752 | 355.139 | 374.477 |

Full table: `results/tables/bootstrap_ci_mse.csv`.

### Ensembles vs. best individual models — explicit overlap check

- Stacking_Ridge CI: **[232.887, 251.030]**
- FixedWeighted_XGBoost70_MLP30 CI: **[233.263, 251.187]**
- XGBoost CI: **[236.253, 254.400]**
- MLP CI: **[236.323, 254.480]**
- CatBoost CI: **[237.320, 255.907]**
- LightGBM CI: **[237.778, 256.306]**

Both ensemble intervals overlap substantially with XGBoost's, MLP's,
CatBoost's, and LightGBM's intervals (e.g. Stacking_Ridge's upper bound of
251.030 sits well inside XGBoost's interval, and XGBoost's lower bound of
236.253 sits well inside both ensembles' intervals). **There is no
non-overlapping pair here.**

---

## Verdict on the headline claim

**The ensembles' small edge over XGBoost/MLP in `official_split_metrics.csv`
(Stacking_Ridge R²=0.6820 / MSE=241.88 vs. XGBoost R²=0.6776 / MSE=245.19)
is NOT statistically supported by this analysis.** The 95% bootstrap CIs on
row-level test MSE for both ensembles overlap heavily with the CIs for
XGBoost, MLP, CatBoost, and LightGBM. Per the stats-auditor rule ("never
report model A beats model B from a single point-estimate comparison"),
this project should **not** claim the ensembles are significantly better
than the best individual models — the improvement is consistent with
resampling noise given the observed overlap.

What *can* be claimed, with backing:
- All 10 individual models are not equal — both ANOVA (F=142.60,
  p≈3.98e-30 on R²; F=150.72, p≈0 on MSE) and Kruskal-Wallis (H=42.01,
  p=0.000003 on R²; H=41.96, p=0.000003 on MSE) reject the null of equal
  means/distributions.
- The three linear models (LinearRegression, Ridge, BayesianRidge) are
  significantly worse than the top tree/boosting/MLP cluster — supported
  both by the primary (non-parametric, Holm-corrected) test on R² and by
  the assumption-clean ANOVA/Tukey on MSE.
- Within the top cluster (LightGBM, CatBoost, GradientBoosting, XGBoost,
  MLP, and now the two ensembles), no pairwise comparison in this analysis
  clears the bar for a defensible "significantly better" claim — neither
  Dunn/Holm on CV R², Tukey on CV MSE, nor the bootstrap CI overlap check on
  test-set MSE.

### Suggested README results-section paragraph

> Across 5-fold cross-validation, a one-way ANOVA found a statistically
> significant difference among the 10 candidate models (R²: F=142.60,
> p=1.63e-27; MSE: F=150.72, p=5.59e-28). Because Shapiro-Wilk flagged 2 of 10
> models' R² fold-distributions as non-normal (LinearRegression p=0.045,
> BayesianRidge p=0.048; Levene's test showed no variance-homogeneity
> violation, F=0.109, p=0.999), we treat the non-parametric Kruskal-Wallis
> test (H=42.01, p=3e-6) and Dunn's post-hoc test (Holm-corrected) as the
> primary evidence for R², rather than the parametric Tukey HSD. Under that
> test, CatBoost and MLP are significantly better (p_holm<0.02) than the
> three linear models (LinearRegression, Ridge, BayesianRidge); the CV MSE
> metric (which passed all assumption checks) additionally shows the full
> tree/boosting/MLP cluster and MLP beating KNN and SVM significantly. No
> pairwise difference within the top cluster (LightGBM, CatBoost,
> GradientBoosting, XGBoost, MLP) reached significance on either metric.
> On the held-out official test split, both ensembles (Stacking_Ridge:
> R²=0.6820, MSE=241.88; FixedWeighted_XGBoost70_MLP30: R²=0.6817,
> MSE=242.13) posted marginally better point estimates than the best
> individual model (XGBoost: R²=0.6776, MSE=245.19). A bootstrap (10,000
> resamples, seed=0) on row-level squared error (n=13,096) shows the
> ensembles' 95% CIs ([232.9, 251.0] and [233.3, 251.2]) overlap
> substantially with XGBoost's ([236.3, 254.4]) and MLP's ([236.3, 254.5]).
> We therefore **cannot claim the ensembles are statistically significantly
> better than the best individual models** — the apparent gain is within
> resampling noise. Note also that the CV folds used here are row-level
> (not grouped by engine), so these CV-based significance results should
> not be read as leakage-safe; see the leakage-red-team subagent's PGTS
> re-evaluation for that check.

---

## Files produced

- `results/tables/anova_tukey_r2.txt` — full Shapiro-Wilk/Levene/ANOVA/Tukey/Kruskal-Wallis/Dunn text report, R²
- `results/tables/anova_tukey_mse_cv.txt` — same, CV MSE
- `results/tables/tukey_hsd_r2_pairs.csv` — full 45-pair Tukey table, R²
- `results/tables/dunn_r2_pairs.csv` — full 45-pair Dunn table (Holm-corrected), R²
- `results/tables/tukey_hsd_mse_cv_pairs.csv` — full 45-pair Tukey table, CV MSE
- `results/tables/dunn_mse_cv_pairs.csv` — full 45-pair Dunn table (Holm-corrected), CV MSE
- `results/tables/bootstrap_ci_mse.csv` — point estimate + 95% CI, row-level test MSE, all 10 models + 2 ensembles
- `results/tables/STATS_SUMMARY.md` — this file
