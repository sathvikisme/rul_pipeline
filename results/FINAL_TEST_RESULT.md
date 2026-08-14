# Final Frozen Test-Set Result (Phase 2, single-shot evaluation)

This is the terminal measurement of the Phase 2 'freeze then evaluate once' process. Model selection (which two candidates, which hyperparameters) was decided entirely from training-engine evidence in `results/FREEZE_DECISION.md` and `results/tables/ensemble_selection.csv` *before* `data/processed/test.csv` was read by this script. No further model, hyperparameter, or feature decisions may be made based on the numbers below -- this is a report of what happened, not a selection step.

`data/processed/test.csv` access log entry written: `{"timestamp": "2026-08-12T15:52:44.514681+00:00", "models_evaluated": ["LightGBM", "StackingEnsemble"], "script": "final_eval.py"}`

## Headline numbers (official held-out test set)

n_test_rows=13096, n_test_engines=100

| Model | R2 | MSE | MAE | PHM08 RUL Score |
|---|---|---|---|---|
| LightGBM | 0.6800 | 243.37 | 9.98 | 71101.9 |
| StackingEnsemble | 0.6793 | 243.95 | 10.02 | 74096.8 |

## Hyperparameters used

Each was the mode (most common) hyperparameter set chosen for that model across nested_cv.py's 5 outer GroupKFold folds (`results/tables/nested_cv_best_params.json`); ties broken by the mean inner-CV score among tied options.

- **LightGBM** (mode 3/5 folds): `{"num_leaves": 15, "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05}`
- **CatBoost** (mode 2/5 folds): `{"learning_rate": 0.05, "l2_leaf_reg": 7, "iterations": 150, "depth": 8}`
- **XGBoost** (mode 2/5 folds -- **TIE-BROKEN** by mean inner CV score): `{"subsample": 0.85, "n_estimators": 50, "max_depth": 4, "learning_rate": 0.15, "colsample_bytree": 0.7}`
- **GradientBoosting** (mode 5/5 folds): `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 200}`
- **MLP** (mode 5/5 folds): `{"alpha": 0.0001, "hidden_layer_sizes": [50, 50], "learning_rate_init": 0.01}`
- **Ridge meta-learner alpha**: 100 (GroupKFold(5) tuning on the pooled nested-OOF matrix, grid=[0.01, 0.1, 1.0, 10.0, 100.0])

**Note on the XGBoost tie-break** (the only base learner where the mode was ambiguous): two hyperparameter sets each won 2 of the 5 outer folds. The tied option was resolved to the one with the better mean inner-CV score: {"tied_options": [{"params": {"estimator__subsample": 0.85, "estimator__n_estimators": 50, "estimator__max_depth": 4, "estimator__learning_rate": 0.15, "estimator__colsample_bytree": 0.7}, "folds": [1, 4], "mean_inner_cv_score_neg_mse": -277.3177760407}, {"params": {"estimator__subsample": 0.85, "estimator__n_estimators": 100, "estimator__max_depth": 5, "estimator__learning_rate": 0.05, "estimator__colsample_bytree": 1.0}, "folds": [2, 3], "mean_inner_cv_score_neg_mse": -288.82464630475}], "winner_params": {"estimator__subsample": 0.85, "estimator__n_estimators": 50, "estimator__max_depth": 4, "estimator__learning_rate": 0.15, "estimator__colsample_bytree": 0.7}, "rule": "higher (less negative) mean inner_cv_best_score_neg_mse among tied-count options"}

## Comparison to Phase 1's official-split numbers

**Caveat up front: these are NOT the same methodology**, so a raw before/after delta partly reflects methodology changes, not just 'the leakage-safe process picked a worse/better model.' Differences include: Phase 1's LightGBM hyperparameters came from a single `GridSearchCV` under a plain (non-grouped, row-level) `KFold`; Phase 2's LightGBM hyperparameters are the mode across 5 genuinely engine-grouped nested-CV outer folds. Phase 1's `Stacking_Ridge` used `cross_val_predict(KFold(5, shuffle=True))` (row-level, non-grouped) OOF predictions for its meta-learner and its base learners' hyperparameters also came from plain-KFold `GridSearchCV`; Phase 2's `StackingEnsemble` uses genuinely engine-grouped nested-CV OOF predictions throughout, and its base learners are also mode-selected from the same nested-CV process, not independently tuned. Phase 1's ensemble selection (`FixedWeighted_XGBoost70_MLP30` and, implicitly, which 5 models fed `Stacking_Ridge`) was itself chosen by looking at official test-set R2 (`results/tables/ensembling_config.json`'s `"selection_metric": "official test-set R2"`) -- the exact test-set-contamination problem Phase 2 exists to fix. Phase 2's freeze decision was made with zero test-set influence.

### LightGBM: Phase 1 (official split) vs Phase 2 (frozen, official test set)

| | R2 | MSE | MAE | PHM08 |
|---|---|---|---|---|
| Phase 1 | 0.6754 | 246.92 | 10.02 | 76549.0 |
| Phase 2 (frozen) | 0.6800 | 243.37 | 9.98 | 71101.9 |

Delta (Phase 2 - Phase 1): R2 +0.0047, MSE -3.55. 

### Stacking: Phase 1 `Stacking_Ridge` vs Phase 2 `StackingEnsemble` (frozen, official test set)

Not the same base-learner-tuning or OOF-generation methodology (see caveat above) -- reported side by side for transparency, not as an apples-to-apples ablation.

| | R2 | MSE | MAE | PHM08 |
|---|---|---|---|---|
| Phase 1 `Stacking_Ridge` | 0.6820 | 241.88 | 9.98 | 76737.5 |
| Phase 2 `StackingEnsemble` (frozen) | 0.6793 | 243.95 | 10.02 | 74096.8 |

Delta (Phase 2 - Phase 1): R2 -0.0027, MSE +2.06. 

## Did the leakage-safe frozen selection process change the final numbers meaningfully from Phase 1's test-set-selected ones?

LightGBM R2 moved by +0.0047 and the stacking R2 moved by -0.0027 relative to Phase 1's numbers for the (non-identical) closest-analogue architectures. Both are within a small margin of Phase 1's numbers, which is the expected shape of this result: this dataset's official split places cycles from every engine's early life in train and only each engine's LAST cycles in test, so it is far less prone to the catastrophic collapse the PGTS/leakage-red-team re-evaluation demonstrates for genuinely leaky protocols (e.g. Track A's row-level, engine_id-as-feature reproduction of the paper's own headline number). The methodology changes here (genuinely grouped nested CV instead of plain KFold, hyperparameters not picked by looking at the test set) mainly change WHICH hyperparameters/ensemble members get used, not whether the official-split evaluation itself is leaky -- so a large jump was never expected here, unlike the PGTS re-evaluation elsewhere in this project. See `results/tables/pgts_comparison.csv` and `results/LEAKAGE_REPORT.md` for the split-level (not selection-level) leakage story.

## Source paper citation

Özcan, H. "Interpretable ensemble remaining useful life prediction enables dynamic maintenance scheduling for aircraft engines." *Scientific Reports* 15, 39795 (2025). https://doi.org/10.1038/s41598-025-23473-2 -- this project reproduces and stress-tests that paper's methodology; the numbers above are this project's own frozen, leakage-safe re-evaluation, not the paper's own reported numbers (see `results/REPRODUCTION_REPORT.md` / `README.md` for the direct paper comparison).
