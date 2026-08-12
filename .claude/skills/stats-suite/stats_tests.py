"""
Statistical validation suite for comparing model performance across multiple
models/folds. This is what turns "model A got a lower number than model B"
into "model A is *significantly* better than model B" — or, just as
importantly, tells you when it can't make that claim.

Pipeline:
  1. Shapiro-Wilk  — are per-fold scores normally distributed? (assumption
     check for ANOVA)
  2. Levene's test — is variance similar across models? (assumption check)
  3. Bootstrap CI  — resampled confidence interval on a metric (e.g. MSE)
  4. One-way ANOVA — is there a significant difference among model means?
  5. Tukey HSD     — which specific pairs of models differ significantly?

Do not skip straight to Tukey HSD without checking 1 and 2 — if normality or
homoscedasticity assumptions are badly violated, report that explicitly
rather than proceeding as if ANOVA's assumptions were satisfied. A
non-parametric alternative (Kruskal-Wallis + Dunn's test) is the honest
fallback when they are violated.
"""
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def check_normality(scores_by_model: dict, alpha=0.05):
    """Shapiro-Wilk test per model. Returns dict of {model: (W, p, normal_bool)}."""
    results = {}
    for name, scores in scores_by_model.items():
        scores = np.asarray(scores)
        if len(scores) < 3:
            raise ValueError(f"Shapiro-Wilk needs >= 3 samples, got {len(scores)} for '{name}'")
        W, p = stats.shapiro(scores)
        results[name] = (float(W), float(p), p > alpha)
    return results


def check_variance_homogeneity(scores_by_model: dict, alpha=0.05):
    """Levene's test across all models. Returns (F, p, homoscedastic_bool)."""
    groups = list(scores_by_model.values())
    F, p = stats.levene(*groups)
    return float(F), float(p), p > alpha


def bootstrap_ci(values, n_resamples=10_000, ci=0.95, seed=0):
    """Bootstrap confidence interval on the mean of `values` (e.g. fold-level MSE)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_resamples)
    ])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(values.mean()), float(lo), float(hi)


def one_way_anova(scores_by_model: dict):
    """One-way ANOVA across models. Returns (F, p)."""
    groups = list(scores_by_model.values())
    F, p = stats.f_oneway(*groups)
    return float(F), float(p)


def tukey_hsd(scores_by_model: dict, alpha=0.05):
    """
    Tukey HSD post-hoc test. Returns the statsmodels summary table object —
    print it directly, don't hand-parse it.
    """
    all_scores = np.concatenate(list(scores_by_model.values()))
    all_labels = np.concatenate([
        [name] * len(scores) for name, scores in scores_by_model.items()
    ])
    return pairwise_tukeyhsd(endog=all_scores, groups=all_labels, alpha=alpha)


def run_full_suite(scores_by_model: dict, metric_name="R2"):
    """Run everything and print a readable report. scores_by_model maps
    model_name -> array-like of per-fold scores (e.g. 5-fold CV R^2)."""
    print(f"=== Statistical validation suite ({metric_name}) ===\n")

    print("-- Shapiro-Wilk (normality per model) --")
    normality = check_normality(scores_by_model)
    for name, (W, p, is_normal) in normality.items():
        flag = "OK" if is_normal else "VIOLATED"
        print(f"  {name:25s} W={W:.3f}  p={p:.3f}  [{flag}]")

    print("\n-- Levene's test (variance homogeneity) --")
    F, p, homoscedastic = check_variance_homogeneity(scores_by_model)
    flag = "OK" if homoscedastic else "VIOLATED"
    print(f"  F={F:.3f}  p={p:.3f}  [{flag}]")

    any_violation = (not homoscedastic) or any(not n[2] for n in normality.values())
    if any_violation:
        print("\n  WARNING: ANOVA assumptions violated. Report this explicitly in your "
              "writeup, and consider Kruskal-Wallis + Dunn's test as a non-parametric "
              "fallback instead of trusting the ANOVA/Tukey results below at face value.")

    print("\n-- One-way ANOVA --")
    F, p = one_way_anova(scores_by_model)
    print(f"  F={F:.3f}  p={p:.4f}  {'(significant difference exists)' if p < 0.05 else '(no significant difference detected)'}")

    print("\n-- Tukey HSD post-hoc --")
    print(tukey_hsd(scores_by_model))

    return {
        "normality": normality,
        "levene": (F, p, homoscedastic),
        "anova": one_way_anova(scores_by_model),
        "assumptions_violated": any_violation,
    }


if __name__ == "__main__":
    # Synthetic sanity check — NOT real model results. Three synthetic "models"
    # with clearly different means, to confirm the pipeline runs end-to-end and
    # detects a difference that's engineered to be there.
    rng = np.random.default_rng(0)
    synthetic_scores = {
        "model_a_synthetic": rng.normal(loc=0.90, scale=0.02, size=5),
        "model_b_synthetic": rng.normal(loc=0.85, scale=0.02, size=5),
        "model_c_synthetic": rng.normal(loc=0.70, scale=0.02, size=5),
    }
    report = run_full_suite(synthetic_scores, metric_name="R2 (synthetic, not real results)")
    assert report["anova"][1] < 0.05, "synthetic groups were constructed to differ significantly"
    print("\nsanity checks passed (synthetic data only — replace with real fold scores)")
