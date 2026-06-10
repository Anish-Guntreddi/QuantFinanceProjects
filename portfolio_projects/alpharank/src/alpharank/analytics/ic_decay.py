"""IC decay analysis across multiple forward-return horizons.

For each horizon h in ``horizons``, this module computes the mean IC and
Newey-West HAC t-statistic of the given factor scores against h-month forward
returns.  A fast-decaying IC profile indicates a short-horizon signal; a
gradual decay indicates persistent factor alpha.
"""

from __future__ import annotations

import pandas as pd

from alpharank.analytics.ic import compute_ic_series, newey_west_ic_tstat
from alpharank.labels.forward_returns import make_forward_returns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ic_decay(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 6),
) -> pd.DataFrame:
    """Compute IC decay across multiple forward-return horizons.

    For each horizon h in ``horizons``:
    1. Build h-month forward returns from ``prices`` using
       ``make_forward_returns(prices, h)``.
    2. Compute the per-date Spearman rank-IC series against ``scores``
       (reusing the same scores for all horizons).
    3. Compute mean IC and Newey-West HAC t-statistic / p-value.

    Parameters
    ----------
    scores : pd.DataFrame
        Factor scores, shape (T, n_assets).  Same index/columns as ``prices``.
    prices : pd.DataFrame
        Monthly close prices, shape (T, n_assets).  DatetimeIndex of month-end
        dates.  Used to construct forward returns at each horizon.
    horizons : tuple[int, ...], default (1, 2, 3, 6)
        Horizons (in months) at which to evaluate IC.

    Returns
    -------
    pd.DataFrame
        Index = horizons, columns = [mean_ic, t_stat, p_value, n_obs].
        Each row summarises the IC distribution at that horizon.

    Notes
    -----
    The same factor ``scores`` are used for every horizon.  This is correct
    for a point-in-time factor: we ask "how predictive is this score at horizon
    h?" for each h.  Scores are NOT re-lagged per horizon.
    """
    rows = []
    for h in horizons:
        fwd_ret = make_forward_returns(prices, horizon=h)
        ic_series = compute_ic_series(scores, fwd_ret)

        n_obs = len(ic_series)
        if n_obs == 0:
            rows.append(
                {"horizon": h, "mean_ic": float("nan"), "t_stat": float("nan"),
                 "p_value": float("nan"), "n_obs": 0}
            )
            continue

        mean_ic, t_stat, p_value = newey_west_ic_tstat(ic_series)
        rows.append(
            {"horizon": h, "mean_ic": mean_ic, "t_stat": t_stat,
             "p_value": p_value, "n_obs": n_obs}
        )

    result = pd.DataFrame(rows).set_index("horizon")
    return result
