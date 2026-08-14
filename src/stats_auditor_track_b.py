"""
Phase 2 stats-auditor deliverable (Track B): paired, leakage-safe
significance testing on the nested-CV results produced by model-trainer.

This is the companion to Phase 1's `results/tables/STATS_SUMMARY.md`
(row-level KFold ANOVA/Tukey + row-level bootstrap), NOT a replacement for
it -- Phase 1's files are left untouched per CLAUDE.md / the Phase 2 plan.
The point of this script is to fix two specific methodological issues in
the Phase 1 procedure when applied to the new nested-CV artifacts:

  Issue: every model here is scored on the SAME 5 outer GroupKFold folds
  (same held-out engines per fold) -- that's a PAIRED design, not
  independent groups, so ANOVA/Tukey HSD is the wrong tool. Use
  `paired_permutation_test` / `wilcoxon_signed_rank` on the paired per-fold
  differences instead (`.claude/skills/paired-stats/paired_stats.py`).

  Issue: the official-split test set has ~131 correlated cycles per engine,
  not independent rows. Use `paired_engine_bootstrap` (resample engines, not
  rows) instead of the row-level bootstrap.

Inputs (already produced by model-trainer -- NOT regenerated here):
  results/tables/nested_cv_metrics.csv          -- 50 rows, 10 models x 5 outer folds
  results/tables/nested_cv_oof_predictions.csv  -- 20,631 rows, per-row OOF preds, all 10 models, tagged engine_id
  results/tables/nested_stacking_metrics.csv    -- 5 rows, stacking ensemble's nested-CV performance

Outputs (new files, Phase 2):
  results/tables/model_pairwise_tests.csv
  results/tables/engine_bootstrap_metrics.csv
  results/tables/ensemble_selection.csv  (draft -- cost/interpretability/
      physical-plausibility columns intentionally left TBD, per the Phase 2
      plan: those are human-reviewed synthesis judgment calls, not
      something to auto-fill here)
  results/tables/PAIRED_STATS_SUMMARY.md

Fixed seed 42 throughout, per CLAUDE.md rule 3.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".claude" / "skills" / "paired-stats"))

from paired_stats import (  # noqa: E402
    paired_permutation_test,
    wilcoxon_signed_rank,
    holm_bonferroni_correction,
    practical_significance_flag,
    paired_engine_bootstrap,
)

TABLES = REPO_ROOT / "results" / "tables"
SEED = 42
N_PERMUTATIONS = 10_000
N_BOOT_RESAMPLES = 10_000
PRACTICAL_PCT_THRESHOLD = 1.0  # see paired_stats.py module docstring for rationale
ALPHA = 0.05

STACKING_NAME = "StackingEnsemble"


def load_per_fold_matrices():
    """Returns (r2_df, mse_df): outer_fold-indexed, model-columned wide
    frames, including the stacking ensemble as an 11th column. Verifies the
    stacking ensemble was evaluated on the identical outer-fold split as the
    base models (same n_train_rows/n_test_rows per fold) before treating
    its per-fold scores as pairable with the base models'."""
    metrics = pd.read_csv(TABLES / "nested_cv_metrics.csv")
    stacking = pd.read_csv(TABLES / "nested_stacking_metrics.csv")

    # sanity check: same outer-fold row counts, fold-for-fold, before
    # treating stacking's per-fold scores as paired with the base models'.
    base_fold_shapes = (
        metrics[metrics.model == "LightGBM"]
        .sort_values("outer_fold")[["outer_fold", "n_train_rows", "n_test_rows"]]
        .reset_index(drop=True)
    )
    stacking_fold_shapes = stacking.sort_values("outer_fold")[
        ["outer_fold", "n_train_rows", "n_test_rows"]
    ].reset_index(drop=True)
    if not base_fold_shapes.equals(stacking_fold_shapes):
        raise RuntimeError(
            "Stacking ensemble's outer-fold row counts do not match the base "
            "models' -- cannot treat stacking's per-fold scores as paired "
            "with the base models' without this guarantee."
        )

    r2_wide = metrics.pivot(index="outer_fold", columns="model", values="R2")
    mse_wide = metrics.pivot(index="outer_fold", columns="model", values="MSE")
    stacking_indexed = stacking.set_index("outer_fold")
    r2_wide[STACKING_NAME] = stacking_indexed["R2"]
    mse_wide[STACKING_NAME] = stacking_indexed["MSE"]
    r2_wide = r2_wide.sort_index()
    mse_wide = mse_wide.sort_index()
    return r2_wide, mse_wide


def run_pairwise_tests(r2_wide: pd.DataFrame, mse_wide: pd.DataFrame) -> pd.DataFrame:
    """Runs paired_permutation_test + wilcoxon_signed_rank for every pair of
    the 11 entities (10 base models + stacking ensemble) on paired per-fold
    R2 and MSE, applies Holm-Bonferroni correction PER METRIC FAMILY (i.e.
    55 R2 comparisons corrected together, 55 MSE comparisons corrected
    together -- not mixed, since they're different hypotheses), and flags
    practical significance based on the MSE effect size for every row
    (including R2-metric rows -- see rationale in the docstring below)."""
    models = list(r2_wide.columns)
    pairs = list(itertools.combinations(models, 2))

    rows = []
    for metric_name, wide in (("R2", r2_wide), ("MSE", mse_wide)):
        raw_p_by_pair = {}
        row_cache = {}
        for model_a, model_b in pairs:
            a = wide[model_a].to_numpy()
            b = wide[model_b].to_numpy()
            perm = paired_permutation_test(a, b, n_permutations=N_PERMUTATIONS, seed=SEED)
            wil = wilcoxon_signed_rank(a, b)

            # Practical significance is evaluated on the underlying MSE
            # effect size regardless of which metric (R2 or MSE) this row
            # nominally reports on -- R2 is a unitless, variance-normalized
            # transform of MSE and doesn't have a natural "1% of what"
            # baseline, whereas MSE is directly in RUL-cycles^2 units and
            # maps to a real practical/operational cost. Reuse the MSE pair
            # for both metric rows so "practically_significant" always means
            # the same thing.
            mse_a_mean = mse_wide[model_a].mean()
            mse_b_mean = mse_wide[model_b].mean()
            delta_mse = float(mse_a_mean - mse_b_mean)
            baseline_mse = float(min(mse_a_mean, mse_b_mean))  # relative to the better model
            practical = practical_significance_flag(
                delta_mse=delta_mse, baseline_mse=baseline_mse,
                pct_threshold=PRACTICAL_PCT_THRESHOLD,
            )

            key = (model_a, model_b)
            raw_p_by_pair[key] = perm["p_value"]
            row_cache[key] = {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric_name,
                "mean_delta": perm["mean_delta"],
                "permutation_p": perm["p_value"],
                "wilcoxon_p": wil["p_value"],
                "practically_significant": practical["practically_significant"],
                "delta_mse_pct": practical["delta_pct"],
            }

        # Holm-Bonferroni correction across the full 55-pair family for this
        # metric, applied to the permutation p-value (chosen as primary --
        # it's a finer-grained Monte-Carlo estimate than the exact Wilcoxon
        # distribution's coarse resolution at n=5; Wilcoxon is reported
        # alongside as a second opinion, per paired-stats SKILL.md).
        corrected = holm_bonferroni_correction(
            {k: v for k, v in raw_p_by_pair.items()}, alpha=ALPHA
        )
        for key, row in row_cache.items():
            c = corrected[key]
            row["p_corrected"] = c["p_corrected"]
            row["statistically_significant"] = c["reject"]
            if row["statistically_significant"] and row["practically_significant"]:
                verdict = "significant difference (statistical AND practical)"
            elif row["statistically_significant"] and not row["practically_significant"]:
                verdict = "statistical tie (p<0.05 after correction, but <1% MSE effect -- not practically meaningful)"
            elif not row["statistically_significant"] and row["practically_significant"]:
                verdict = "not statistically confirmed despite >=1% MSE effect (likely underpowered at n=5 folds)"
            else:
                verdict = "no significant difference detected"
            row["verdict"] = verdict
            rows.append(row)

    out = pd.DataFrame(rows)[[
        "model_a", "model_b", "metric", "mean_delta", "permutation_p",
        "wilcoxon_p", "p_corrected", "statistically_significant",
        "practically_significant", "verdict",
    ]]
    return out


def run_engine_bootstrap() -> pd.DataFrame:
    """Runs paired_engine_bootstrap for every pair of the 10 base models
    using the per-row nested-CV OOF predictions (which carry engine_id).
    The stacking ensemble is EXCLUDED here -- no per-row OOF prediction file
    exists for it (only per-fold aggregate metrics in
    nested_stacking_metrics.csv; confirmed by inspecting
    nested_stacking_config.json and the results/tables/ directory listing,
    neither of which references a row-level stacking OOF file). This
    limitation is called out explicitly in PAIRED_STATS_SUMMARY.md."""
    preds = pd.read_csv(TABLES / "nested_cv_oof_predictions.csv")
    pred_cols = [c for c in preds.columns if c.startswith("pred_")]
    models = [c[len("pred_"):] for c in pred_cols]

    rows = []
    for model_a, model_b in itertools.combinations(models, 2):
        result = paired_engine_bootstrap(
            preds, model_a, model_b,
            engine_col="engine_id", y_true_col="y_true",
            n_resamples=N_BOOT_RESAMPLES, seed=SEED,
        )
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "point_estimate_delta_mse": result["point_estimate_delta_mse"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "ci_level": result["ci_level"],
            "n_engines": result["n_engines"],
            "n_resamples": result["n_resamples"],
            "ci_excludes_zero": result["ci_excludes_zero"],
        })
    return pd.DataFrame(rows)


def build_ensemble_selection(r2_wide, mse_wide, pairwise_df) -> pd.DataFrame:
    """Draft ensemble_selection.csv. Anchor ('best') = the entity (among the
    10 base models + stacking) with the lowest mean nested-CV MSE, which
    directly answers "does the stacking ensemble actually beat the best
    individual model" if the anchor turns out to be the ensemble.
    cost / interpretability / physical_plausibility columns are left TBD --
    those are human-reviewed synthesis judgment calls per the Phase 2 plan,
    not something to auto-fill from these numbers."""
    mean_r2 = r2_wide.mean()
    std_r2 = r2_wide.std()
    mean_mse = mse_wide.mean()
    std_mse = mse_wide.std()

    best_model = mean_mse.idxmin()

    mse_pairwise_only = pairwise_df[pairwise_df.metric == "MSE"]
    mse_pairwise_lookup = {
        frozenset({row["model_a"], row["model_b"]}): row
        for _, row in mse_pairwise_only.iterrows()
    }

    rows = []
    for model in r2_wide.columns:
        if model == best_model:
            stat_dist = False
            prac_diff = False
            verdict_vs_best = "is the reference (lowest mean nested-CV MSE)"
        else:
            key = frozenset({model, best_model})
            if key in mse_pairwise_lookup:
                match = mse_pairwise_lookup[key]
                stat_dist = bool(match["statistically_significant"])
                prac_diff = bool(match["practically_significant"])
                verdict_vs_best = match["verdict"]
            else:
                stat_dist, prac_diff, verdict_vs_best = None, None, "pair not found"

        rows.append({
            "model": model,
            "is_ensemble": model == STACKING_NAME,
            "nested_cv_r2_mean": mean_r2[model],
            "nested_cv_r2_std": std_r2[model],
            "nested_cv_mse_mean": mean_mse[model],
            "nested_cv_mse_std": std_mse[model],
            "fold_stability_mse_std": std_mse[model],
            "is_best_by_mean_mse": model == best_model,
            "statistically_distinguishable_from_best": stat_dist,
            "practically_different_from_best": prac_diff,
            "verdict_vs_best": verdict_vs_best,
            "cost_tbd": "TBD -- human-reviewed synthesis",
            "interpretability_tbd": "TBD -- human-reviewed synthesis",
            "physical_plausibility_tbd": "TBD -- human-reviewed synthesis",
        })

    out = pd.DataFrame(rows).sort_values("nested_cv_mse_mean").reset_index(drop=True)
    return out, best_model


def main():
    print("Loading nested-CV per-fold matrices (10 base models + stacking ensemble)...")
    r2_wide, mse_wide = load_per_fold_matrices()
    print(f"  R2 matrix: {r2_wide.shape}, models: {list(r2_wide.columns)}")

    print("\nRunning paired permutation test + Wilcoxon signed-rank on all "
          f"{len(list(itertools.combinations(r2_wide.columns, 2)))} pairs x 2 metrics...")
    pairwise_df = run_pairwise_tests(r2_wide, mse_wide)
    pairwise_path = TABLES / "model_pairwise_tests.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"  wrote {pairwise_path} ({len(pairwise_df)} rows)")

    n_base_model_pairs = len(list(itertools.combinations(
        [c for c in r2_wide.columns if c != STACKING_NAME], 2
    )))
    print(f"\nRunning engine-level paired bootstrap on all {n_base_model_pairs} "
          "base-model pairs (nested_cv_oof_predictions.csv)...")
    boot_df = run_engine_bootstrap()
    boot_path = TABLES / "engine_bootstrap_metrics.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"  wrote {boot_path} ({len(boot_df)} rows)")

    print("\nBuilding draft ensemble_selection.csv...")
    selection_df, best_model = build_ensemble_selection(r2_wide, mse_wide, pairwise_df)
    selection_path = TABLES / "ensemble_selection.csv"
    selection_df.to_csv(selection_path, index=False)
    print(f"  wrote {selection_path} ({len(selection_df)} rows); best by mean MSE = {best_model}")

    return r2_wide, mse_wide, pairwise_df, boot_df, selection_df, best_model


if __name__ == "__main__":
    main()
