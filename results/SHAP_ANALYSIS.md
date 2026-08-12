# SHAP Interpretability Analysis

RUL-Bench interpretability-analyst deliverable (PROJECT_BRIEF.md §7), reproducing
and stress-testing Özcan, H. (2025), *Scientific Reports* 15, 39795, on NASA
C-MAPSS FD001. All numbers below come from `src/shap_analysis.py`, executed
against the already-trained, saved model artifacts in `results/models/`
(nothing here was retrained; SHAP was run once, seed 42 throughout).

## Models analyzed and why

Per `results/tables/official_split_metrics.csv`, the top tier by official
test R² is XGBoost (0.6776), MLP (0.6775), CatBoost (0.6759), LightGBM
(0.6754). Two individuals were chosen from this tier plus one ensemble:

- **XGBoost** and **CatBoost** — the two individual models. Both are
  gradient-boosted tree ensembles, so `shap.TreeExplainer` gives *exact*
  (non-sampled) attributions, and both are top-tier by test R².
- **FixedWeighted ensemble** (`results/models/FixedWeightedEnsemble.joblib`
  = 0.7·XGBoost + 0.3·MLP, per `results/tables/ensembling_config.json`) —
  the ensemble analyzed. This was chosen over Stacking_Ridge because SHAP is
  linear under linear combination: since the FixedWeighted ensemble *is*
  a literal linear combination of two already-fitted models,
  `SHAP(ensemble) = 0.7·SHAP(XGBoost) + 0.3·SHAP(MLP)` is **exact**, not an
  approximation — with no extra explainer calls beyond what XGBoost and MLP
  already require. This exactness was verified numerically: reconstructing
  each row's prediction as `base_value + sum(SHAP)` matched the ensemble's
  actual weighted-average prediction to within 6.2e-05 RUL-cycles, max
  absolute error, across the 500-row sample.
- **MLP** (individual) — computed as a required ingredient of the ensemble
  above, and reported standalone as a bonus third individual model since it
  came "for free."

MLP is not a tree model, so its SHAP values use `shap.Explainer(model.predict,
background)`, which shap dispatches to its Permutation explainer for a
generic callable (background = `shap.sample(train, n=50, random_state=42)`).

**Sampling**: SHAP values were computed on a fixed-seed sample of
**500 test rows** (`test.sample(n=500, random_state=42)`) — large enough for
stable beeswarm/summary statistics, small enough to keep the MLP permutation
explainer's runtime sane (~20s for 500 rows with a 50-row background).

## Feature set and physical sensor meaning

All models use the same 18 features (`cycle`, `op_setting_1..3`, and 14
kept sensors, all StandardScaler-transformed except `cycle`). C-MAPSS/PHM08
sensor documentation for the sensors that showed up among the top
attributions below:

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
so it is physically sensible that the dominant non-`cycle` features are all
core-gas-path channels tied directly to HPC/LPT behavior (Ps30, P30, Nc,
NRc, T50) rather than fan-side or bypass-duct sensors.

## Global attributions (mean |SHAP| ranking, top 5 per model)

Full ranked list for all 18 features per model: `results/shap/shap_ranked_features_full.json`.
Raw table: `results/shap/shap_concentration_metrics.csv`.

**XGBoost** (plot: `results/shap/beeswarm_xgboost.png`)
1. `cycle` (12.98)
2. `sensor_11` — Ps30, HPC outlet static pressure (5.90)
3. `sensor_4` — T50, LPT outlet temperature (3.81)
4. `sensor_12` — phi, fuel-flow/Ps30 ratio (2.71)
5. `sensor_9` — Nc, physical core speed (2.55)

XGBoost leans heavily on `cycle` + `sensor_11` (Ps30) — consistent with
turbine/compressor-stage pressure loss being XGBoost's dominant late-life
signal, with T50 (LPT outlet temperature) as a secondary confirming signal.

**CatBoost** (plot: `results/shap/beeswarm_catboost.png`)
1. `cycle` (12.29)
2. `sensor_9` — Nc, physical core speed (3.41)
3. `sensor_14` — NRc, corrected core speed (2.95)
4. `sensor_11` — Ps30 (2.68)
5. `sensor_12` — phi (2.54)

CatBoost's non-`cycle` attribution is spread more evenly across the
core-speed pair (Nc / NRc) and Ps30/phi — no single sensor dominates nearly
as much as `sensor_11` does for XGBoost (5.90 vs. CatBoost's runner-up at
3.41). This shows up quantitatively in the concentration metrics below.

**MLP** (plot: `results/shap/beeswarm_mlp.png`, bonus/third individual)
1. `cycle` (9.81)
2. `sensor_9` — Nc (5.67)
3. `sensor_14` — NRc (5.20)
4. `sensor_11` — Ps30 (3.05)
5. `sensor_12` — phi (2.52)

MLP's top-5 set is nearly identical to CatBoost's (Nc, NRc, Ps30, phi) but
weighted more heavily toward the core-speed pair than `cycle` itself relative
to the tree models — `cycle`'s lead over `sensor_9` is much narrower here
(9.81 vs. 5.67) than for XGBoost (12.98 vs. 5.90).

**FixedWeighted Ensemble = 0.7·XGBoost + 0.3·MLP** (plot: `results/shap/beeswarm_ensemble_fixedweighted.png`)
1. `cycle` (11.93)
2. `sensor_11` — Ps30 (5.03)
3. `sensor_9` — Nc (3.46)
4. `sensor_4` — T50 (3.02)
5. `sensor_14` — NRc (2.69)

The ensemble's ranking is essentially a blend of its two components:
`sensor_11` (XGBoost's dominant secondary feature) stays #2, but `sensor_9`/
`sensor_14` (MLP's core-speed emphasis) move up relative to XGBoost alone,
and the gap between rank 2 and rank 3 narrows (5.03→3.46, a 1.45x gap) versus
XGBoost's rank 2→3 gap (5.90→3.81, a 1.55x gap, off a higher rank-1).

## Attribution concentration: does the ensemble spread credit more evenly?

The source paper claims its ensemble spreads feature credit more evenly
(is less concentrated on a few dominant sensors) than its individual base
models. Two concentration measures, computed on mean |SHAP| per feature
across the same 500-row sample (`results/shap/shap_concentration_metrics.csv`):

| Model | Top-3 share of total mean\|SHAP\| | Gini coefficient |
|---|---|---|
| XGBoost | 0.5681 | 0.5645 |
| CatBoost | 0.4693 | 0.4797 |
| MLP | 0.5263 | 0.5189 |
| **FixedWeighted Ensemble** | **0.5207** | **0.5360** |

(Lower = more evenly spread across features in both columns.)

**Verdict: the claim holds only partially here.**

- **Vs. XGBoost** (the ensemble's 70%-weighted dominant member): the
  ensemble *is* more balanced on both metrics (top-3 share 0.521 vs. 0.568;
  Gini 0.536 vs. 0.564). Averaging in MLP measurably dilutes XGBoost's
  concentration on `sensor_11`/`cycle`.
- **Vs. MLP**: mixed — the ensemble is very slightly *less* concentrated by
  top-3 share (0.521 vs. 0.526) but very slightly *more* concentrated by
  Gini (0.536 vs. 0.519). Essentially a wash; not a meaningful improvement
  over MLP alone.
- **Vs. CatBoost** (an individual model not part of this ensemble, but one
  of the two individuals analyzed here): the ensemble is **more**
  concentrated on both metrics (top-3 share 0.521 vs. 0.469; Gini 0.536 vs.
  0.480). CatBoost alone shows the most evenly spread attribution of all
  four models measured.

So the ensemble is more balanced than its own dominant constituent
(unsurprising — that's what averaging in a second model does), but it is
**not** more balanced than every individual model tested — CatBoost, which
wasn't even part of this ensemble, is noticeably more balanced than the
ensemble by both measures. The paper's general "ensembling balances feature
credit" framing does not hold universally in this reproduction; it holds
conditionally on which individual model you compare against.

## Local explanations (force plots)

Three representative test-set predictions from the FixedWeighted ensemble,
spanning the RUL range, plus the same near-failure row explained by each
individual model for comparison:

- `results/shap/force_ensemble_low_rul.png` — engine 68, cycle 185, true
  RUL=10 (near end-of-life). Ensemble prediction 13.51. Dominated by
  `sensor_11` (Ps30, high/elevated value pushing prediction down toward
  failure), `sensor_12` (phi), `sensor_4` (T50), `cycle`, `sensor_9` (Nc),
  `sensor_7` (P30) — all pushing the prediction toward low RUL.
- `results/shap/force_ensemble_mid_rul.png` — engine 17, cycle 148, true
  RUL=67 (actively degrading, uncapped). Ensemble prediction 56.78.
  `sensor_11` pushes the prediction up (toward more RUL) while `cycle`,
  `sensor_9`, `sensor_14` pull it down.
- `results/shap/force_ensemble_high_rul.png` — engine 1, cycle 9, true
  RUL=125 (early-life, at the piecewise-linear cap). Ensemble prediction
  124.68 — essentially all major features (`sensor_7`, `sensor_12`,
  `sensor_14`, `sensor_9`, `sensor_11`, `cycle`) push toward the high/capped
  end, correctly reflecting a healthy, early-life engine.
- `results/shap/force_xgboost_low_rul.png` and
  `results/shap/force_catboost_low_rul.png` — the same near-failure row
  (engine 68, cycle 185) explained individually. XGBoost (pred 14.79) and
  CatBoost (pred 15.14) agree closely with the ensemble (13.51) and with
  each other on direction, but CatBoost weights `sensor_2` and `sensor_8`
  into its top contributors where XGBoost and the ensemble do not — a
  concrete instance of the CatBoost/XGBoost attribution differences visible
  in the global beeswarm plots above.

Force plots use `contribution_threshold=0.08` (group features contributing
<8% of the total effect into the plot's implicit remainder) and rotated
labels for legibility with 18 candidate features — see
`src/shap_analysis.py:save_force`. Feature values shown are the
StandardScaler-transformed values actually fed to the models (train-fit
scaler, per `data/processed/DATA_DICTIONARY.md`), not raw physical units.

## Reproducibility notes

- Script: `src/shap_analysis.py`. Run via `python src/shap_analysis.py`.
- Seed 42 used for: the 500-row test sample, the 50-row MLP background
  sample, and the MLP permutation explainer's internal seed.
- All 9 plots + 2 data files live in `results/shap/`:
  `beeswarm_xgboost.png`, `beeswarm_catboost.png`, `beeswarm_mlp.png`,
  `beeswarm_ensemble_fixedweighted.png`, `force_ensemble_low_rul.png`,
  `force_ensemble_mid_rul.png`, `force_ensemble_high_rul.png`,
  `force_xgboost_low_rul.png`, `force_catboost_low_rul.png`,
  `shap_concentration_metrics.csv`, `shap_ranked_features_full.json`.
- No model was retrained or fine-tuned to produce this analysis — SHAP was
  computed strictly on the saved artifacts in `results/models/`, per the
  interpretability-analyst's scope restriction.
