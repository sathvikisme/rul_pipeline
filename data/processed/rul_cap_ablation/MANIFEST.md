# data/processed/rul_cap_ablation — MANIFEST

Three parallel RUL-labeled datasets built from the SAME raw NASA C-MAPSS FD001 files (`data/raw/train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) by `src/rul_cap_variants.py` (data-engineer subagent, Phase 2). **The only difference between variants A/B/C is the RUL target's cap treatment — every feature column (engine_id, cycle, op_setting_1-3, sensor_1-21) is identical in value across all three variants**, verified programmatically at build time (see checks below), not just claimed.

RUL cap value used where capping is applied: **125** cycles, reused directly from `preprocessing.RUL_CAP` (not re-hardcoded) — same value and same rationale as Phase 1 (Heimes 2008 plateau convention, also used by Özcan 2025; see `src/preprocessing.py` module docstring).

## Feature scope

Op settings (3) + ALL 21 raw sensor channels, UNSCALED, with NO variance-threshold feature selection applied (unlike Phase 1's data/processed/train.csv, which drops 7 near-constant sensors and z-score-scales the rest). Feature selection and scaling are applied fold-safely later, inside Track B's `nested_cv.py` / `src/fold_safe_pipeline.py`, not at data-prep time — see that module's docstring for why.

## Variants

| Variant | Train RUL | Test RUL | Notes |
|---|---|---|---|
| A | capped at 125 | **raw / uncapped** | tests against true remaining life |
| B | capped at 125 | capped at 125 | matches current Phase 1 `data/processed/{train,test}.csv` exactly |
| C | raw / uncapped | raw / uncapped | fully uncapped baseline |

## Actual row counts / RUL ranges from this run

| Variant | train shape | train RUL range | test shape | test RUL range |
|---|---|---|---|---|
| A | 20631 x 27 | [0, 125] | 13096 x 27 | [7, 340] |
| B | 20631 x 27 | [0, 125] | 13096 x 27 | [7, 125] |
| C | 20631 x 27 | [0, 361] | 13096 x 27 | [7, 340] |

## Feature-column identity checks (actually run, not assumed)

| Check | Result |
|---|---|
| train_feature_cols_A_vs_B | True |
| train_feature_cols_A_vs_C | True |
| test_feature_cols_A_vs_B | True |
| test_feature_cols_A_vs_C | True |
| train_RUL_A_equals_B (both capped) | True |
| test_RUL_A_equals_C (both uncapped) | True |
| train_RUL_A_equals_C (should be False -- A capped, C not) | False |
| test_RUL_A_equals_B (should be False -- A uncapped, B capped) | False |

## Files

```
data/processed/rul_cap_ablation/
  A/train.csv  A/test.csv
  B/train.csv  B/test.csv
  C/train.csv  C/test.csv
  MANIFEST.md  (this file)
```

Columns in every train.csv/test.csv: `engine_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1..sensor_21, RUL` (26 raw feature columns + RUL target, unscaled, unselected).

Caveat for downstream consumers (per Phase 2 plan §6): PHM08 score and R2/MSE are only meaningfully comparable **within** a cap regime (A vs A, B vs B, C vs C) — comparing raw metric values ACROSS variants conflates "model got better" with "target distribution changed," since capping directly changes the scale/skew of the target.
