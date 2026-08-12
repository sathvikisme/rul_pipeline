"""
PHM08 RUL scoring function (Saxena, Goebel, Simon & Eklund, 2008 —
"Damage propagation modeling for aircraft engine run-to-failure simulation",
1st Int. Conf. on Prognostics and Health Management).

The score penalizes LATE predictions (d > 0, i.e. predicted RUL > true RUL,
meaning the model told you the engine had more life left than it did — the
dangerous direction) more heavily than EARLY predictions (d < 0, unnecessary
but safe maintenance).

IMPORTANT — verify the sign/constant convention before trusting any paper's
numbers, including ours. Some papers define d = true - predicted instead of
predicted - true, which flips which branch is "early" vs "late". This
implementation uses the original PHM08 convention:

    d_i = predicted_RUL_i - true_RUL_i

    s_i = exp(-d_i / a1) - 1   if d_i < 0   (early prediction, a1 = 13)
    s_i = exp( d_i / a2) - 1   if d_i >= 0  (late prediction,  a2 = 10)

    Score = sum(s_i)

Lower is better. This is NOT symmetric — a late prediction of the same
magnitude as an early one scores worse, by design, because underestimating
time-to-failure risk is operationally more dangerous than over-maintaining.
"""
import numpy as np


def phm08_score(y_true, y_pred):
    """
    Compute the PHM08 RUL score.

    Parameters
    ----------
    y_true : array-like of true RUL values
    y_pred : array-like of predicted RUL values

    Returns
    -------
    total_score : float, sum of per-sample scores (lower is better)
    per_sample : np.ndarray, individual sample scores (for diagnostics/plots)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    d = y_pred - y_true
    a1, a2 = 13.0, 10.0

    per_sample = np.where(
        d < 0,
        np.exp(-d / a1) - 1.0,
        np.exp(d / a2) - 1.0,
    )
    return float(per_sample.sum()), per_sample


if __name__ == "__main__":
    # Sanity checks against known properties of the function, not against a
    # specific paper's headline number — don't treat this as validating any
    # particular model's accuracy.
    y_true = np.array([50.0, 50.0, 50.0])
    y_pred_perfect = np.array([50.0, 50.0, 50.0])
    y_pred_early = np.array([40.0, 40.0, 40.0])   # predicted less than true -> early
    y_pred_late = np.array([60.0, 60.0, 60.0])    # predicted more than true -> late

    s_perfect, _ = phm08_score(y_true, y_pred_perfect)
    s_early, _ = phm08_score(y_true, y_pred_early)
    s_late, _ = phm08_score(y_true, y_pred_late)

    assert s_perfect == 0.0, "perfect predictions must score exactly 0"
    assert s_late > s_early, "a late error must score worse than an equal-magnitude early error"

    print(f"perfect: {s_perfect:.4f}")
    print(f"early (d=-10 each): {s_early:.4f}")
    print(f"late  (d=+10 each): {s_late:.4f}")
    print("sanity checks passed")
