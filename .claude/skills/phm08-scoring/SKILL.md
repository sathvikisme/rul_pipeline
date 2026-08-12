---
name: phm08-scoring
description: Compute the PHM08 asymmetric RUL scoring function. Use whenever a RUL Score needs to be calculated or reported anywhere in this project — do not re-derive the formula from memory or from a paper excerpt each time.
---

# PHM08 RUL Scoring

Different papers state the PHM08 score's sign/constant convention slightly
differently. To avoid silently reproducing a bug or a mismatched convention,
always import and use `score.py` in this directory rather than re-implementing
the formula inline.

## Usage

```python
from score import phm08_score

total, per_sample = phm08_score(y_true, y_pred)
```

`y_true` and `y_pred` are RUL values in cycles (array-like, same shape).
Returns the summed score (lower is better) and the per-sample breakdown for
diagnostics/plotting.

## Before trusting any reported score

1. Run `python score.py` — it runs two sanity checks (perfect prediction
   scores 0, a late error scores worse than an equal early error) and will
   fail loudly if the implementation is broken.
2. If comparing against the source paper's reported score, re-check that
   paper's exact sign convention for `d_i` — some define it as
   `true - predicted` instead of `predicted - true`. Getting this backwards
   silently flips which errors get penalized more.
3. Report the score alongside RMSE/MAE, never as a replacement for them —
   the RUL Score is not on a directly comparable scale across differently
   sized test sets, so use it for within-study comparison, not as a
   universal benchmark number.
