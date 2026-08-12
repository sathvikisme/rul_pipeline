# RUL-Bench — Project Rules

Reproducing and stress-testing Özcan, H. "Interpretable ensemble remaining
useful life prediction enables dynamic maintenance scheduling for aircraft
engines." *Scientific Reports* 15, 39795 (2025).
https://doi.org/10.1038/s41598-025-23473-2

Full methodology, dataset details, and the exact build plan: see
`PROJECT_BRIEF.md` at the repo root. Read it before starting work in a fresh
session if you haven't already.

## Non-negotiable rules

1. **No fabricated numbers, anywhere — including drafts.** Every metric that
   appears in code comments, results files, or the README must come from an
   actual executed run. If something hasn't been run yet, say so explicitly
   ("not yet run") rather than writing a plausible-looking placeholder. This
   applies to intermediate work, not just the final report.
2. **Cite the source paper explicitly** in the README and anywhere results
   are compared against it. This is a reproduction-plus-extension project,
   not original research presented as if it emerged from nothing — that
   framing is more credible, not less.
3. **Fixed random seeds everywhere.** Every model, every train/test split,
   every bootstrap resample. Non-reproducible numbers are worse than no
   numbers for this project.
4. **Don't trust the official split's numbers until leakage-red-team has
   run the PGTS re-evaluation on them.** The source paper's own headline
   result reportedly collapses under leakage-safe validation — treat any
   suspiciously strong result as a thing to verify, not celebrate.
5. **Statistical claims need statistical backing.** "Model A is better than
   model B" requires the stats-suite output (bootstrap CI or ANOVA + Tukey
   HSD), not just a lower point estimate.

## Subagents — when to delegate

This project has five custom subagents in `.claude/agents/`. Delegate to them
rather than doing their scoped work in the main thread — it keeps verbose
training/test output out of the main context and keeps each concern isolated:

| Subagent | Use for |
|---|---|
| `data-engineer` | Data loading, preprocessing, RUL labeling, feature engineering |
| `model-trainer` | Training/tuning all individual models, building both ensemble variants |
| `stats-auditor` | Significance testing on model comparisons |
| `leakage-red-team` | PGTS leakage-safe re-evaluation, adversarially checking any strong result |
| `interpretability-analyst` | SHAP analysis and write-up |

Claude Code's built-in subagents (Explore, Plan, general-purpose) are still
available and useful for codebase search or multi-step planning — use them
for that, not as a substitute for the scoped subagents above when the work
matches one of their descriptions.

A rough execution order: data-engineer → model-trainer → (stats-auditor and
leakage-red-team can run in parallel once model-trainer has saved trained
models) → interpretability-analyst last, once there's a settled best model
to explain.

## Skills — load automatically, don't reimplement

Three project-specific skills live in `.claude/skills/`. They're small,
tested, deterministic scripts — use them instead of re-deriving the same
formula/procedure from memory each session, which is exactly the kind of
inconsistency that would undermine a reproduction study:

- `phm08-scoring` — the RUL scoring formula (verified against the original
  PHM08 sign convention, sanity-tested)
- `pgts-split` — the purged group time series split (leakage-tested)
- `stats-suite` — Shapiro-Wilk / Levene / bootstrap / ANOVA / Tukey HSD
  pipeline (tested on synthetic data with a known significant difference)

All three have a `python <script>.py` sanity check built in — run it after
any edit to the script itself, before trusting output from it again.

## Environment

- Python 3.x, dependencies pinned in `requirements.txt` (create/maintain this
  — don't let the environment drift undocumented).
- Core libs: pandas, numpy, scikit-learn, lightgbm, catboost, xgboost, shap,
  scipy, statsmodels.
- No GPU required — FD001 is ~20K rows, everything here trains on CPU in
  seconds to low minutes.

## Repo structure

```
data/            raw + processed C-MAPSS files
src/             preprocessing.py, features.py, models.py, ensembling.py,
                 evaluation.py, stats_tests.py, pgts_split.py, shap_analysis.py
notebooks/       exploratory work only — final logic belongs in src/
results/         actual run outputs: tables, plots, saved models
README.md        cites the source paper; states reproduction vs. extension;
                 real numbers only
```

## Definition of done

See `PROJECT_BRIEF.md` § Deliverables for the full list. Short version: real
metrics under the official split, the full stats-suite output, SHAP plots
with interpretation, the PGTS re-evaluation reproduced, and an honest README
that doesn't blur reproduction with original contribution.
