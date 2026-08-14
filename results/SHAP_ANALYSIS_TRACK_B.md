# SHAP Interpretability Analysis — Track B (Phase 2, frozen models)

RUL-Bench interpretability-analyst deliverable, Phase 2. All numbers below
come from `src/shap_track_b.py`, executed once (seed 42 throughout), against
the two models **frozen** in `results/FREEZE_DECISION.md` and scored exactly
once on the official test set by `src/final_eval.py`
(`results/FINAL_TEST_RESULT.md`):

| Model | Official test R2 | MSE | MAE | PHM08 |
|---|---|---|---|---|
| LightGBM | 0.6800 | 243.37 | 9.98 | 71101.9 |
| StackingEnsemble | 0.6793 | 243.95 | 10.02 | 74096.8 |

This reproduces and extends Özcan, H. "Interpretable ensemble remaining
useful life prediction enables dynamic maintenance scheduling for aircraft
engines." *Scientific Reports* 15, 39795 (2025).
https://doi.org/10.1038/s41598-025-23473-2 — this is this project's own
leakage-safe re-evaluation, not the paper's own reported numbers.

## Model reconstruction and sanity check

`final_eval.py` fits both frozen candidates in-memory and does not persist
joblib artifacts for them (only Phase 1's `results/models/*.joblib` exist on
disk, and those are Phase 1 models under the old, non-grouped-CV
hyperparameters — not the frozen Phase 2 models). Per the parent task, this
script reconstructs the exact same fitted models by **importing and calling
`final_eval.py`'s own functions directly** — `fit_all_base_learners()` and
`fit_meta_learner()` — rather than re-deriving the fitting logic. It then
re-scores `data/processed/test.csv` with the reconstructed models and
compares against `results/tables/final_test_metrics.csv`:

| Model | Metric | Reference (final_test_metrics.csv) | Reconstructed | Match |
|---|---|---|---|---|
| LightGBM | R2 | 0.680034 | 0.680034 | OK (diff 0) |
| LightGBM | MSE | 243.370489 | 243.370489 | OK (diff 2.8e-14) |
| LightGBM | MAE | 9.981935 | 9.981935 | OK (diff 0) |
| LightGBM | PHM08 | 71101.873675 | 71101.873675 | OK (diff 0) |
| StackingEnsemble | R2 | 0.679276 | 0.679276 | OK (diff 0) |
| StackingEnsemble | MSE | 243.946621 | 243.946621 | OK (diff 0) |
| StackingEnsemble | MAE | 10.017686 | 10.017686 | OK (diff 0) |
| StackingEnsemble | PHM08 | 74096.767804 | 74096.767804 | OK (diff 0) |

**All 8 metrics match to floating-point precision** (full detail:
`results/tables/shap_track_b_reconstruction_sanity_check.json`). This
confirms the SHAP values below are computed on genuinely the same fitted
models `final_eval.py` scored, not an approximation.

This script reads `data/processed/test.csv` a second time — once for this
sanity check, once to draw the 500-row SHAP sample and pick representative
force-plot rows. This does **not** violate the Phase 2 "freeze then evaluate
once" discipline: that discipline (enforced by `_test_set_access_log.json`'s
one-entry guard in `final_eval.py`) exists to stop the *model-selection*
process from peeking at the test set more than once, and no model,
hyperparameter, or feature decision is made anywhere in this script based on
what it finds. `results/FREEZE_DECISION.md`'s selection was already final
before this script ran.

**A finding surfaced during reconstruction, not assumed**: unlike Phase 1's
offline variance filter (which only ever checked the 21 raw sensor channels,
per `data/processed/DATA_DICTIONARY.md`), `final_eval.py`'s fold-safe
`VarianceThreshold(1e-5)` pipeline step is applied to all 18 feature columns,
including the 3 operational settings. FD001 is a single-operating-condition
subset, so `op_setting_3` is already globally z-scored to ~0 variance in
`train.csv`; refitting `VarianceThreshold` on the full 100-engine training
set correctly drops it a second time. Every tree base learner therefore
trains on 17 features, not 18. This script pads SHAP values back to the full
18-feature space with an exact 0.0 attribution for `op_setting_3` (accurate,
not an approximation — a feature the estimator never received contributes
exactly zero), so every base learner's attributions line up column-for-
column for the linear-combination trick below.

## Causal-language caveat (read before the findings)

Every claim below is phrased as **"the model attributes weight to feature
X"** or **"feature X drives this model's prediction"**, never as "feature X
causes/determines RUL" or "degradation in sensor X causes failure." SHAP
values explain a fitted statistical model's learned function on this
dataset — they are not a causal-inference method, and this project performed
no intervention/randomized-controlled analysis of the physical degradation
process. Two limitations in particular: (1) sensor readings are
correlated with each other and with `cycle` by construction (they all
co-evolve with engine wear), so SHAP's attribution of credit to one sensor
over a correlated neighbor can shift with retraining/reseeding without any
underlying physical change; (2) the MLP's permutation-based SHAP values
carry an explicit feature-independence assumption that this correlated
sensor data does not fully satisfy. Physical sensor names (Ps30, Nc, T50,
etc.) are used below purely to make the *model's* attributions legible, not
to assert that those sensors are the true causal drivers of engine failure.

## Feature set and physical sensor meaning

Both frozen models use the same 18 columns as every other model in this
project: `cycle`, `op_setting_1..3`, and 14 kept sensors (all
StandardScaler-transformed except `cycle`), per
`data/processed/DATA_DICTIONARY.md`. Sensor documentation for the ones that
show up in the top attributions below (C-MAPSS/PHM08 convention, Saxena &
Goebel — same table Phase 1's `src/shap_analysis.py` used):

| Feature | Physical meaning |
|---|---|
| `cycle` | Operational cycle index — direct time-in-service proxy |
| `sensor_4` | T50 — Total temperature at LPT outlet (°R) |
| `sensor_7` | P30 — Total pressure at HPC outlet (psia) |
| `sensor_9` | Nc — Physical core speed (rpm) |
| `sensor_11` | Ps30 — Static pressure at HPC outlet (psia) |
| `sensor_12` | phi — Ratio of fuel flow to Ps30 (pps/psi) |
| `sensor_14` | NRc — Corrected core speed (rpm) |

FD001's injected fault mode is High-Pressure-Compressor (HPC) degradation,
so it is physically sensible (not a causal claim about this specific model,
just a plausibility check) that the dominant non-`cycle` features below are
core-gas-path channels tied to HPC/LPT behavior (Ps30, P30, Nc, NRc, T50)
rather than fan-side or bypass-duct sensors.

## Global attributions (mean |SHAP| ranking, top 5)

Sampling: fixed-seed 500-row test sample (`test.sample(n=500,
random_state=42)`), same convention as Phase 1. Full ranked lists (all 18
features) for all 6 models analyzed:
`results/shap/track_b/shap_ranked_features_full_track_b.json`. Raw table:
`results/shap/track_b/shap_concentration_metrics_track_b.csv`.

**LightGBM** (plot: `results/shap/track_b/beeswarm_lightgbm.png`) — exact
`shap.TreeExplainer`
1. `cycle` (12.15)
2. `sensor_11` — Ps30 (5.66)
3. `sensor_4` — T50 (4.13)
4. `sensor_12` — phi (2.75)
5. `sensor_9` — Nc (2.48)

LightGBM leans heavily on `cycle` + `sensor_11` (Ps30), with T50 as a
secondary confirming signal — the model attributes the largest non-time
weight to compressor-stage pressure loss.

**StackingEnsemble** (plot: `results/shap/track_b/beeswarm_stackingensemble.png`)
— exact linear combination of the 5 base learners' SHAP values through the
Ridge meta-learner (see method below)
1. `cycle` (11.10)
2. `sensor_9` — Nc (3.69)
3. `sensor_11` — Ps30 (3.63)
4. `sensor_14` — NRc (3.07)
5. `sensor_12` — phi (2.42)

The ensemble's ranking is core-speed-led (Nc, NRc) rather than pressure-led
(Ps30) — a direct consequence of the Ridge meta-learner's *learned* weights
(see below), which favor the two base learners whose own rankings are
core-speed-led (CatBoost, MLP) over LightGBM/XGBoost's pressure-led ranking.

**Base learners feeding the StackingEnsemble** (all computed for the linear
combination, all reported since the parent task asks for >=2 individuals
plus the ensemble):

- **CatBoost** (`beeswarm_catboost.png`): `cycle` (11.83), `sensor_9`/Nc
  (3.56), `sensor_11`/Ps30 (2.93), `sensor_7`/P30 (2.41), `sensor_12`/phi
  (2.30).
- **XGBoost** (`beeswarm_xgboost.png`): `cycle` (14.34), `sensor_11`/Ps30
  (4.68), `sensor_9`/Nc (2.70), `sensor_12`/phi (2.49), `sensor_7`/P30
  (2.34).
- **GradientBoosting** (`beeswarm_gradientboosting.png`): `cycle` (12.47),
  `sensor_11`/Ps30 (5.11), `sensor_4`/T50 (3.45), `sensor_9`/Nc (2.86),
  `sensor_12`/phi (2.71).
- **MLP** (`beeswarm_mlp.png`): `cycle` (9.66), `sensor_14`/NRc (5.08),
  `sensor_9`/Nc (4.70), `sensor_11`/Ps30 (3.28), `sensor_12`/phi (2.48).

**Ridge meta-learner weights** (`intercept=0.277`):

| Base learner | Ridge coefficient |
|---|---|
| MLP | 0.373 |
| CatBoost | 0.369 |
| LightGBM | 0.174 |
| XGBoost | 0.055 |
| GradientBoosting | 0.027 |

MLP and CatBoost carry ~74% of the meta-learner's total weight combined —
this is the mechanistic reason the StackingEnsemble's top-5 ranking
(core-speed-led) resembles CatBoost's and MLP's own rankings far more than
LightGBM's or XGBoost's (pressure-led): the ensemble's SHAP values are
literally a weighted sum dominated by those two base learners' SHAP values.

## Attribution concentration: does the ensemble spread credit more evenly?

Same two concentration measures Phase 1 used (mean |SHAP| per feature across
the 500-row sample; lower = more evenly spread):

| Model | Top-3 share of total mean\|SHAP\| | Gini coefficient |
|---|---|---|
| LightGBM | 0.5538 | 0.5543 |
| CatBoost | 0.4629 | 0.4634 |
| XGBoost | 0.5677 | 0.5802 |
| GradientBoosting | 0.5286 | 0.5341 |
| MLP | 0.4924 | 0.4867 |
| **StackingEnsemble** | **0.4760** | **0.4807** |

**Verdict: the same partial-support pattern Phase 1 found, replicated here.**
The StackingEnsemble is more balanced (lower on both metrics) than 4 of its
5 own base learners — LightGBM, XGBoost, GradientBoosting, and (very
slightly) MLP — but it is **not** more balanced than CatBoost, whose
attribution (top-3 share 0.463, Gini 0.463) is the single most evenly spread
of all 6 models measured, individual or ensemble. This is the same
conclusion Phase 1's `results/SHAP_ANALYSIS.md` reached about its own
FixedWeighted ensemble (more balanced than its dominant member, not more
balanced than every individual model, specifically not more balanced than
CatBoost). Seeing the same qualitative pattern survive a completely
different ensemble architecture (5-model Ridge stacking with leakage-safe,
nested-CV-selected hyperparameters vs. Phase 1's 2-model fixed 70/30
average) is a genuine, non-trivial piece of evidence against the paper's
general "ensembling balances feature credit" framing — it is not universal
here either, in two structurally different reproductions of the idea.

## Local explanations (force plots)

Representative test-set predictions spanning the RUL range (all from the
500-row SHAP sample, selection criteria: minimum RUL for "low", true
RUL nearest the midpoint between the minimum and the capped plateau for
"mid" (restricted to uncapped rows), true RUL nearest 125 for "high"):

- **`force_stacking_low_rul.png`** — engine 68, cycle 185, true RUL=10
  (near end-of-life). StackingEnsemble prediction 16.37. Dominated by
  `sensor_11` (Ps30, elevated value), `sensor_12` (phi), `sensor_9` (Nc),
  `cycle`, `sensor_7` (P30), `sensor_2` — all pushing the prediction toward
  low RUL.
- **`force_stacking_mid_rul.png`** — engine 17, cycle 148, true RUL=67
  (actively degrading, uncapped). StackingEnsemble prediction 58.92.
  `sensor_11` pushes the prediction up (toward more RUL) while `cycle`,
  `sensor_9`, `sensor_14` pull it down — a genuine tug-of-war between
  features, unlike the near-unanimous low/high-RUL rows.
- **`force_stacking_high_rul.png`** — engine 1, cycle 9, true RUL=125
  (early-life, at the piecewise-linear cap). StackingEnsemble prediction
  125.34 — `sensor_20`, `sensor_2`, `sensor_7`, `sensor_12`, `sensor_14`,
  `sensor_11`, `sensor_9`, `cycle` all push toward the high/capped end,
  consistent with (not proof of) a healthy, early-life engine.
- **`force_lightgbm_low_rul.png`** and **`force_lightgbm_high_rul.png`** —
  the same two rows explained by LightGBM alone (16.95 and 124.93
  respectively, both close to the StackingEnsemble's 16.37 / 125.34 on the
  same rows). LightGBM's low-RUL row is dominated by `sensor_11`,
  `sensor_4` (T50 — not in the ensemble's top drivers for this row),
  `sensor_12`, `cycle`, `sensor_9`; its high-RUL row leads with `sensor_7`
  (P30) rather than the ensemble's `sensor_20` — a concrete instance of the
  single-model-vs-ensemble attribution difference visible in the global
  beeswarm plots above, even though both models' point predictions nearly
  agree on these two rows.

Force plots use `contribution_threshold=0.08` and rotated labels for
legibility with 18 candidate features (same convention as Phase 1's
`src/shap_analysis.py:save_force`). Feature values shown are the
StandardScaler-transformed values actually fed to the models, not raw
physical units.

## Comparison to Phase 1's SHAP ranking

Phase 1 (`results/SHAP_ANALYSIS.md`, models selected by **official test-set
R2** — the exact test-set-contamination problem Phase 2 exists to fix)
analyzed XGBoost, CatBoost, MLP individually, plus a FixedWeighted
`0.7*XGBoost + 0.3*MLP` ensemble, all trained under Phase 1's plain
(non-grouped) `KFold` hyperparameter search:

| Model | Track | Top 5 (physical meaning) |
|---|---|---|
| XGBoost | Phase 1 | cycle, Ps30, T50, phi, Nc |
| CatBoost | Phase 1 | cycle, Nc, NRc, Ps30, phi |
| MLP | Phase 1 | cycle, Nc, NRc, Ps30, phi |
| FixedWeighted (0.7 XGB + 0.3 MLP) | Phase 1 | cycle, Ps30, Nc, T50, NRc |
| **LightGBM** | **Track B (frozen)** | **cycle, Ps30, T50, phi, Nc** |
| **CatBoost** | **Track B (frozen)** | **cycle, Nc, Ps30, P30, phi** |
| **XGBoost** | **Track B (frozen)** | **cycle, Ps30, Nc, phi, P30** |
| **MLP** | **Track B (frozen)** | **cycle, NRc, Nc, Ps30, phi** |
| **StackingEnsemble** | **Track B (frozen)** | **cycle, Nc, Ps30, NRc, phi** |

**The same 6 sensors dominate in both tracks — `cycle`, Ps30 (sensor_11),
Nc (sensor_9), NRc (sensor_14), T50 (sensor_4), phi (sensor_12) — and every
model in both tracks ranks `cycle` #1 by a wide margin.** This is a genuine
consistency signal: the leakage-safe, nested-CV-selected hyperparameters
(Track B) attribute credit to essentially the same physically-plausible
core-gas-path sensors as Phase 1's test-set-selected hyperparameters, even
though the two tracks differ in every other methodological respect
(hyperparameter search protocol, which fold structure tuned them, and — for
the ensemble — completely different architectures, 5-model Ridge stacking
vs. 2-model fixed-weight averaging).

Two second-order differences worth reporting honestly rather than glossing
over:

1. **XGBoost's own ranking changed order slightly between tracks**: Phase 1's
   XGBoost ranked T50 (sensor_4) at #3; Track B's (differently-tuned)
   XGBoost ranks Nc (sensor_9) at #3 instead, with T50 dropping to #6
   (2.22, `shap_ranked_features_full_track_b.json`). This is plausibly
   attributable to the hyperparameter difference: Phase 1's XGBoost was
   chosen by plain-KFold `GridSearchCV`; Track B's XGBoost hyperparameters
   are the (tie-broken) mode across 5 genuinely engine-grouped nested-CV
   folds (`results/FINAL_TEST_RESULT.md`'s XGBoost tie-break note) — a
   materially different search process landing on a different final tree
   structure, which is exactly the kind of thing that can reorder two
   already-close SHAP values (2.70 vs. 2.22 here) without changing the
   qualitative top-tier sensor set.
2. **The ensemble's ranking shifted from Ps30-led to Nc-led.** Phase 1's
   FixedWeighted ensemble ranked Ps30 (sensor_11) at #2 (its dominant
   70%-weighted member, XGBoost, is Ps30-led). Track B's StackingEnsemble
   ranks Nc (sensor_9) at #2 and Ps30 at #3 — a direct, traceable
   consequence of the *learned* Ridge meta-learner weighting MLP (0.373)
   and CatBoost (0.369) far more heavily than LightGBM (0.174), XGBoost
   (0.055), or GradientBoosting (0.027), and MLP/CatBoost are both
   individually Nc/NRc-led rather than Ps30-led (see per-base-learner
   rankings above). This is an architecture effect (which base learners get
   how much weight), not evidence that the leakage-safe protocol discovered
   a different physical signal — the underlying sensor vocabulary
   (Ps30/Nc/NRc/T50/phi) is identical across both ensembles; only the
   internal ordering among them moved, tracking a completely explainable
   change in which base learner the meta-learner trusts most.

**Bottom line**: the leakage-safe, freeze-then-evaluate-once Track B models
attribute credit to the same physically-plausible HPC/core-gas-path sensor
set Phase 1 found, which is a real (if modest) piece of evidence that the
sensor-level story in this reproduction is not an artifact of Phase 1's
test-set-driven model selection — it survives a completely independent
selection process. The concentration-of-credit finding (ensemble more
balanced than most but not all individual models, specifically not more
balanced than CatBoost) also replicates across two structurally different
ensemble designs. What *did* change between tracks is exactly what should
change given the known methodology differences (which hyperparameters, which
base learners the meta-learner leans on) — not a new, unexplained
disagreement.

## Reproducibility notes

- Script: `src/shap_track_b.py`. Run via `python src/shap_track_b.py`.
- Seed 42 used for: the 500-row test sample, the 50-row MLP background
  sample, and the MLP permutation explainer's internal seed — same
  convention as Phase 1's `src/shap_analysis.py`.
- All 11 plots + 2 data files live in `results/shap/track_b/`:
  `beeswarm_lightgbm.png`, `beeswarm_catboost.png`, `beeswarm_xgboost.png`,
  `beeswarm_gradientboosting.png`, `beeswarm_mlp.png`,
  `beeswarm_stackingensemble.png`, `force_stacking_low_rul.png`,
  `force_stacking_mid_rul.png`, `force_stacking_high_rul.png`,
  `force_lightgbm_low_rul.png`, `force_lightgbm_high_rul.png`,
  `shap_concentration_metrics_track_b.csv`,
  `shap_ranked_features_full_track_b.json`. The model-reconstruction sanity
  check is also logged to
  `results/tables/shap_track_b_reconstruction_sanity_check.json`.
- Phase 1's `src/shap_analysis.py` and `results/shap/*.png` /
  `results/SHAP_ANALYSIS.md` are untouched and remain the historical record
  of the test-set-selected 70/30 XGBoost/MLP ensemble's attributions — see
  that file for the original analysis this document compares against.
- No new model-selection or hyperparameter decision was made anywhere in
  this script; both frozen models were reconstructed exactly as
  `final_eval.py` fit them (confirmed by the bit-identical sanity check
  above), and SHAP was computed strictly on that reconstruction.
