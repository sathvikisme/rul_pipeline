"""
Paired-comparison statistics suite — the companion to `stats-suite` for when
observations are NOT independent samples.

Why this skill exists (Phase 2, Issue 6 in the audit): Phase 1's
`stats-suite` treats each model's 5 CV-fold scores as independent samples
(ANOVA/Tukey HSD) and bootstraps individual test rows as if independent.
Neither is true here:

  1. Every model in the nested-CV pipeline is evaluated on the SAME 5 outer
     GroupKFold folds (same engines held out each fold). Model A's fold-3
     score and Model B's fold-3 score are correlated (same held-out engines,
     same operating noise, same difficulty) — that's a *paired* design, not
     an independent-groups design. Treating them as independent (ANOVA)
     ignores this correlation and distorts the variance estimate (usually
     inflating it, making real differences look non-significant — but the
     direction of the distortion isn't guaranteed, so don't assume ANOVA is
     simply "conservative" here; use the paired test instead).
  2. The official-split bootstrap resampled individual TEST ROWS (~13,096 of
     them) as if independent, when they are really ~131 correlated cycles
     nested within each of 100 engines. Resampling rows breaks the
     within-engine correlation structure and understates the true
     resampling uncertainty (CIs come out too narrow). The fix is to
     resample the cluster (engine), not the row — `paired_engine_bootstrap`
     below.

This module does NOT replace `.claude/skills/stats-suite/stats_tests.py`.
That skill remains the documented Phase-1 procedure (kept as the negative
control / historical record per CLAUDE.md). This is the more correct
companion for paired per-fold and per-engine comparisons.

Pipeline for a single pair of models:
  1. `paired_permutation_test` — sign-flip permutation test on the paired
     per-fold (or per-engine) differences. Distribution-free, exact under
     the null of a symmetric-around-zero difference distribution, robust to
     n=5 folds where asymptotic tests have no business claiming validity.
  2. `wilcoxon_signed_rank` — a second, textbook non-parametric paired test,
     for when someone (reasonably) questions the permutation test's
     assumptions or wants a citable canonical procedure alongside it.
  3. `holm_bonferroni_correction` — correct across the full family of
     pairwise comparisons (e.g. 45 pairs for 10 models). Holm-Bonferroni,
     not Benjamini-Hochberg: Phase 1's Dunn's-test correction
     (`results/tables/dunn_*_pairs.csv`) already used Holm for the same
     45-pairwise-comparisons-among-CV-folds problem in this exact project,
     and Holm controls the family-wise error rate (not just the false
     discovery rate), which is the more conservative and defensible choice
     for a small, fixed, complete set of pairwise model comparisons where
     any single false positive ("model A beats model B") could get
     over-claimed in the README. Consistency with the established Phase 1
     convention is also a documented reason on its own.
  4. `practical_significance_flag` — a statistically significant p-value
     with a difference that doesn't matter in practice (e.g. p=0.01 but
     ΔMSE = 0.3%) should be reported as a "statistical tie" in practice, not
     as "model A wins." Threshold: 1.0% relative difference in MSE by
     default — chosen because it's roughly the same order of magnitude as
     the fold-to-fold noise visible in `nested_cv_metrics.csv` (MSE swings
     of several percent across the 5 outer folds for the same model), so a
     <1% mean difference is smaller than the noise floor and should not be
     read as a meaningful practical improvement. Document/override this
     threshold explicitly if a different operational tolerance applies.
  5. `paired_engine_bootstrap` — resample the SET OF ENGINES (the actual
     unit of exchangeability in this dataset), not rows, with replacement;
     include every row for every sampled engine; compute both models' MSE
     on the identical resampled engine set each time (paired, so the
     comparison isn't confounded by which engines happened to be drawn);
     report the point estimate and percentile 95% CI of ΔMSE = MSE_a - MSE_b.
"""
import numpy as np
import pandas as pd
from scipy import stats


def paired_permutation_test(scores_a, scores_b, n_permutations=10_000, seed=42):
    """
    Sign-flip permutation test on paired differences (scores_a - scores_b).

    Null hypothesis: the paired differences are drawn from a distribution
    symmetric about zero (i.e. no systematic difference between A and B).
    Under that null, flipping the sign of any individual paired difference
    is exchangeable, so we build the null distribution of the mean
    difference by randomly flipping signs `n_permutations` times.

    Returns dict: {
      'mean_delta': observed mean(a - b),
      'p_value': two-sided permutation p-value,
      'n_permutations': int,
      'n_pairs': int,
    }
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores_a and scores_b must be paired (same shape), got {a.shape} vs {b.shape}")
    n = len(a)
    if n < 2:
        raise ValueError(f"need >= 2 paired observations, got {n}")

    diffs = a - b
    observed_mean = float(diffs.mean())

    rng = np.random.default_rng(seed)
    # (n_permutations, n) matrix of +-1 sign flips, vectorized.
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted_means = (signs * diffs[None, :]).mean(axis=1)

    # two-sided: how often does the permuted null exceed the observed
    # magnitude? +1/+1 correction avoids a p-value of exactly 0.
    n_extreme = np.sum(np.abs(permuted_means) >= np.abs(observed_mean) - 1e-15)
    p_value = float((n_extreme + 1) / (n_permutations + 1))

    return {
        "mean_delta": observed_mean,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_pairs": n,
    }


def wilcoxon_signed_rank(scores_a, scores_b):
    """
    Wrapper around scipy.stats.wilcoxon for a second, canonical paired
    non-parametric test. Uses the exact distribution when feasible (small n,
    no ties/zeros beyond what scipy's default zero_method='wilcox' handles),
    falling back to scipy's own auto-selected approximation otherwise.

    Returns dict: {'statistic': W, 'p_value': p, 'n_pairs': int,
                    'n_effective': n after zero-difference pairs dropped}.

    Caveat: with n=5 folds (the nested-CV case in this project) Wilcoxon has
    very low power — report it as a second opinion alongside the permutation
    test, not as the sole arbiter.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores_a and scores_b must be paired (same shape), got {a.shape} vs {b.shape}")
    diffs = a - b
    n_effective = int(np.sum(diffs != 0))
    if n_effective == 0:
        # every pair is exactly tied -- wilcoxon is undefined; report a
        # trivially non-significant result rather than letting scipy raise.
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(a), "n_effective": 0}
    try:
        statistic, p_value = stats.wilcoxon(a, b, zero_method="wilcox", mode="auto")
    except ValueError as e:
        # scipy raises if all-zero or too few effective pairs for exact mode
        return {"statistic": float("nan"), "p_value": 1.0, "n_pairs": len(a),
                "n_effective": n_effective, "error": str(e)}
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_pairs": len(a),
        "n_effective": n_effective,
    }


def holm_bonferroni_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """
    Holm-Bonferroni step-down correction across a family of pairwise
    comparisons. See module docstring for why Holm (not BH) was chosen for
    this project.

    p_values: dict of {comparison_key: raw_p_value}
    Returns dict of {comparison_key: {'p_raw':, 'p_corrected':, 'reject':}}
    """
    keys = list(p_values.keys())
    raw = np.array([p_values[k] for k in keys], dtype=float)
    m = len(raw)
    order = np.argsort(raw)  # ascending

    corrected = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * raw[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)

    return {
        keys[i]: {
            "p_raw": float(raw[i]),
            "p_corrected": float(corrected[i]),
            "reject": bool(corrected[i] < alpha),
        }
        for i in range(m)
    }


def practical_significance_flag(delta_mse, baseline_mse, pct_threshold: float = 1.0):
    """
    Flags whether a difference in MSE (delta_mse = MSE_a - MSE_b) is large
    enough, relative to a baseline MSE, to be called practically meaningful
    -- independent of whether it's statistically significant.

    delta_mse: MSE_a - MSE_b (signed, same units as baseline_mse)
    baseline_mse: reference scale, e.g. the better (lower) of the two
        models' MSE, or the pooled mean MSE -- caller's choice, but be
        consistent across a table of comparisons.
    pct_threshold: minimum |delta_mse| / baseline_mse * 100 to count as a
        practically significant difference. Default 1.0% -- see module
        docstring for the rationale (below the observed fold-to-fold MSE
        noise floor in this project's nested-CV results).

    Returns dict: {'delta_pct': signed percent difference,
                    'practically_significant': bool}
    """
    if baseline_mse == 0:
        raise ValueError("baseline_mse must be nonzero")
    delta_pct = float(delta_mse) / float(baseline_mse) * 100.0
    return {
        "delta_pct": delta_pct,
        "practically_significant": bool(abs(delta_pct) >= pct_threshold),
    }


def paired_engine_bootstrap(
    preds_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    engine_col: str = "engine_id",
    y_true_col: str = "y_true",
    pred_col_template: str = "pred_{model}",
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
):
    """
    Cluster (engine-level) paired bootstrap on delta MSE = MSE_a - MSE_b.

    preds_df must contain columns [engine_col, y_true_col,
    pred_col_template.format(model=model_a), pred_col_template.format(model=model_b)],
    one row per (engine, cycle) prediction -- e.g.
    results/tables/nested_cv_oof_predictions.csv.

    Procedure: resample the set of UNIQUE engines with replacement (size =
    number of unique engines), include every row belonging to each sampled
    engine (with repeats when an engine is drawn more than once), and
    compute MSE_a and MSE_b on that exact same resampled engine set (so the
    comparison stays paired -- both models see identical resampled data on
    every iteration). Repeat n_resamples times; report the percentile 95% CI
    of the resulting delta-MSE distribution.

    This fixes the row-level bootstrap issue (Issue 6): resampling
    individual rows treats ~131 correlated cycles per engine as independent
    and understates true resampling uncertainty (CIs too narrow). Resampling
    the engine is the correct cluster-bootstrap unit here.

    Returns dict: {
      'point_estimate_delta_mse': MSE_a - MSE_b on the FULL (unresampled) data,
      'ci_lower':, 'ci_upper':, 'ci_level':,
      'n_engines':, 'n_resamples':,
      'ci_excludes_zero': bool,
    }
    """
    col_a = pred_col_template.format(model=model_a)
    col_b = pred_col_template.format(model=model_b)
    for col in (engine_col, y_true_col, col_a, col_b):
        if col not in preds_df.columns:
            raise ValueError(f"expected column '{col}' not found in preds_df")

    df = preds_df[[engine_col, y_true_col, col_a, col_b]].dropna()
    sq_err_a = (df[y_true_col].to_numpy(dtype=float) - df[col_a].to_numpy(dtype=float)) ** 2
    sq_err_b = (df[y_true_col].to_numpy(dtype=float) - df[col_b].to_numpy(dtype=float)) ** 2
    engines = df[engine_col].to_numpy()

    unique_engines = np.unique(engines)
    n_engines = len(unique_engines)

    # per-engine sum of squared error and row count, indexed 0..n_engines-1
    engine_index = np.searchsorted(unique_engines, engines)
    sse_a = np.bincount(engine_index, weights=sq_err_a, minlength=n_engines)
    sse_b = np.bincount(engine_index, weights=sq_err_b, minlength=n_engines)
    n_rows = np.bincount(engine_index, minlength=n_engines).astype(float)

    # point estimate on the real (unresampled) data
    point_delta = float(sse_a.sum() / n_rows.sum() - sse_b.sum() / n_rows.sum())

    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, n_engines, size=(n_resamples, n_engines))
    boot_sse_a = sse_a[sampled_idx].sum(axis=1)
    boot_sse_b = sse_b[sampled_idx].sum(axis=1)
    boot_n = n_rows[sampled_idx].sum(axis=1)
    boot_delta = boot_sse_a / boot_n - boot_sse_b / boot_n

    lo = float(np.percentile(boot_delta, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_delta, (1 + ci) / 2 * 100))

    return {
        "point_estimate_delta_mse": point_delta,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_level": ci,
        "n_engines": n_engines,
        "n_resamples": n_resamples,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Sanity check on SYNTHETIC data (not real project results).
    #
    # IMPORTANT documented finding from this sanity check itself: a
    # sign-flip permutation test on n=5 paired folds has an inherent
    # resolution floor. With n paired observations there are only 2^n
    # possible sign-flip patterns, so the minimum achievable two-sided
    # p-value is 2/2^n. For n=5 (this project's outer-fold count), that
    # floor is 2/32 = 0.0625 -- meaning NO effect size, however large, can
    # produce a sign-flip permutation p < 0.05 with only 5 paired folds
    # where every fold agrees in direction. Scenario 1 below demonstrates
    # this explicitly (huge injected effect, correct direction, but p
    # capped at ~0.0625) so this isn't silently discovered later as a
    # surprise when running the real per-fold model comparisons. Scenario 3
    # then demonstrates the test *does* have power to detect real effects
    # once n is large enough (n=30, e.g. representative of a per-engine
    # paired comparison), confirming the test implementation itself is
    # correct and not simply broken.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)

    print("=== Scenario 1: real injected paired difference, n=5 folds (this project's actual fold count) ===")
    n_folds = 5
    base = rng.normal(loc=250, scale=15, size=n_folds)  # shared fold difficulty
    scores_a = base + rng.normal(loc=0, scale=1.5, size=n_folds)       # model A
    scores_b = base + 25 + rng.normal(loc=0, scale=1.5, size=n_folds)  # model B, worse by ~25 MSE (huge effect)

    perm = paired_permutation_test(scores_a, scores_b, n_permutations=10_000, seed=42)
    wil = wilcoxon_signed_rank(scores_a, scores_b)
    print(f"  permutation: mean_delta={perm['mean_delta']:.3f}  p={perm['p_value']:.4f}")
    print(f"  wilcoxon:    statistic={wil['statistic']:.3f}  p={wil['p_value']:.4f}")
    print(f"  NOTE: p is floored near 2/2^5=0.0625 regardless of effect size at n=5 -- "
          f"this is a real mathematical property of the sign-flip/Wilcoxon test at this "
          f"sample size, not a bug. Direction (mean_delta < 0, i.e. A better) is still correctly recovered.")
    assert perm["mean_delta"] < 0, "A should score lower (better, lower MSE) than B by construction"
    assert perm["p_value"] < 0.10, "even capped at the n=5 resolution floor, this should still read as suggestive"

    print("\n=== Scenario 2: no injected difference, n=5 folds ===")
    scores_a2 = base + rng.normal(loc=0, scale=2, size=n_folds)
    scores_b2 = base + rng.normal(loc=0, scale=2, size=n_folds)
    perm2 = paired_permutation_test(scores_a2, scores_b2, n_permutations=10_000, seed=42)
    wil2 = wilcoxon_signed_rank(scores_a2, scores_b2)
    print(f"  permutation: mean_delta={perm2['mean_delta']:.3f}  p={perm2['p_value']:.4f}")
    print(f"  wilcoxon:    statistic={wil2['statistic']:.3f}  p={wil2['p_value']:.4f}")
    assert perm2["p_value"] > 0.05, "no injected difference should not be flagged significant"

    print("\n=== Scenario 3: real injected difference, n=30 paired observations (demonstrates test HAS power "
          "at larger n -- e.g. a per-engine paired comparison) ===")
    n_big = 30
    base_big = rng.normal(loc=250, scale=15, size=n_big)
    scores_a3 = base_big + rng.normal(loc=0, scale=5, size=n_big)
    scores_b3 = base_big + 10 + rng.normal(loc=0, scale=5, size=n_big)  # B worse by ~10
    perm3 = paired_permutation_test(scores_a3, scores_b3, n_permutations=10_000, seed=42)
    wil3 = wilcoxon_signed_rank(scores_a3, scores_b3)
    print(f"  permutation: mean_delta={perm3['mean_delta']:.3f}  p={perm3['p_value']:.4f}")
    print(f"  wilcoxon:    statistic={wil3['statistic']:.3f}  p={wil3['p_value']:.4f}")
    assert perm3["p_value"] < 0.05, "n=30 with a real, consistent effect should clear p<0.05"
    assert wil3["p_value"] < 0.05, "n=30 with a real, consistent effect should clear p<0.05 (Wilcoxon)"

    print("\n=== Scenario 4: no injected difference, n=30 ===")
    scores_a4 = base_big + rng.normal(loc=0, scale=5, size=n_big)
    scores_b4 = base_big + rng.normal(loc=0, scale=5, size=n_big)
    perm4 = paired_permutation_test(scores_a4, scores_b4, n_permutations=10_000, seed=42)
    print(f"  permutation: mean_delta={perm4['mean_delta']:.3f}  p={perm4['p_value']:.4f}")
    assert perm4["p_value"] > 0.05, "no injected difference should not be flagged significant at n=30 either"

    print("\n=== Holm-Bonferroni correction sanity check ===")
    raw_ps = {"pair1": 0.001, "pair2": 0.02, "pair3": 0.03, "pair4": 0.04, "pair5": 0.5}
    corrected = holm_bonferroni_correction(raw_ps)
    for k, v in corrected.items():
        print(f"  {k}: p_raw={v['p_raw']:.4f} p_corrected={v['p_corrected']:.4f} reject={v['reject']}")
    assert corrected["pair1"]["p_corrected"] < 0.05
    assert corrected["pair5"]["p_corrected"] > 0.05

    print("\n=== Practical-significance flag sanity check ===")
    tiny = practical_significance_flag(delta_mse=1.5, baseline_mse=250.0, pct_threshold=1.0)
    big = practical_significance_flag(delta_mse=15.0, baseline_mse=250.0, pct_threshold=1.0)
    print(f"  tiny delta (0.6%): {tiny}")
    print(f"  big delta (6.0%): {big}")
    assert tiny["practically_significant"] is False
    assert big["practically_significant"] is True

    print("\n=== Engine-level paired bootstrap sanity check ===")
    # Independent RNG stream, deliberately decoupled from the scenarios
    # above, so this block's outcome doesn't depend on how much of the
    # earlier stream was consumed.
    rng_boot = np.random.default_rng(7)
    n_engines = 100
    rows_per_engine = 20
    engine_ids = np.repeat(np.arange(n_engines), rows_per_engine)
    # engine-level noise shared across all rows of that engine (the
    # correlation structure the row-level bootstrap ignores)
    engine_noise = rng_boot.normal(0, 10, size=n_engines)
    y_true = rng_boot.normal(100, 20, size=n_engines * rows_per_engine)

    # scenario A: real difference -- model_bad has +8 systematic error
    pred_good = y_true + engine_noise[engine_ids] + rng_boot.normal(0, 3, size=len(y_true))
    pred_bad = y_true + engine_noise[engine_ids] + 8 + rng_boot.normal(0, 3, size=len(y_true))
    df_diff = pd.DataFrame({
        "engine_id": engine_ids, "y_true": y_true,
        "pred_good": pred_good, "pred_bad": pred_bad,
    })
    boot_diff = paired_engine_bootstrap(df_diff, "good", "bad", n_resamples=10_000, seed=42)
    print(f"  real-difference case: delta_mse={boot_diff['point_estimate_delta_mse']:.3f}  "
          f"95% CI=[{boot_diff['ci_lower']:.3f}, {boot_diff['ci_upper']:.3f}]  "
          f"excludes_zero={boot_diff['ci_excludes_zero']}")
    assert boot_diff["ci_excludes_zero"], "injected difference should produce a CI excluding zero"

    # scenario B: no difference -- both predictions built from the exact
    # same noise draw (truly identical model outputs), so the only source
    # of a nonzero point estimate is floating point, i.e. none.
    pred_x = y_true + engine_noise[engine_ids] + rng_boot.normal(0, 3, size=len(y_true))
    pred_y = pred_x.copy()
    df_same = pd.DataFrame({
        "engine_id": engine_ids, "y_true": y_true,
        "pred_x": pred_x, "pred_y": pred_y,
    })
    boot_same = paired_engine_bootstrap(df_same, "x", "y", n_resamples=10_000, seed=42)
    print(f"  no-difference case:   delta_mse={boot_same['point_estimate_delta_mse']:.3f}  "
          f"95% CI=[{boot_same['ci_lower']:.3f}, {boot_same['ci_upper']:.3f}]  "
          f"excludes_zero={boot_same['ci_excludes_zero']}")
    assert not boot_same["ci_excludes_zero"], "no injected difference should produce a CI including zero"

    print("\nAll sanity checks passed (synthetic data only -- replace with real fold/prediction data).")
