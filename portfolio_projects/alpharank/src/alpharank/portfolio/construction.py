"""Decile long-short weight construction.

Public API
----------
build_decile_weights(scores, n_deciles=10) -> dict[pd.Timestamp, dict[str, float]]

Design notes
------------
- Top-decile names receive equal positive weights summing to +1.
- Bottom-decile names receive equal negative weights summing to -1.
- NaN-scored symbols are excluded before decile cuts (handles delisted assets).
- Symbols held at the prior rebalance but absent from the new legs appear with
  weight 0.0 so PrecomputedWeightsStrategy can emit EXIT signals for them.
- Small universes (< n_deciles names after NaN drop): each leg gets at least 1
  name via top-N / bottom-N fallback where N = max(1, n // n_deciles).
"""

from __future__ import annotations

import pandas as pd


def build_decile_weights(
    scores: pd.DataFrame,
    n_deciles: int = 10,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Build equal-weight long-short decile portfolio weights from cross-sectional scores.

    Parameters
    ----------
    scores:
        DataFrame of shape (T_rebalances, N_symbols) with score values.
        Index: pd.DatetimeIndex of rebalance dates.
        Columns: symbol strings.
        NaN values are treated as missing / delisted.

    n_deciles:
        Number of quantile buckets.  Top decile = longs; bottom decile = shorts.

    Returns
    -------
    dict[pd.Timestamp, dict[str, float]]
        Keyed by rebalance date.  Each value maps symbol → weight.
        - Positive weights (top decile) sum to +1.
        - Negative weights (bottom decile) sum to -1.
        - Symbols that were in the prior period's legs but are absent from the
          current period's legs appear with weight 0.0 (EXIT signal trigger).
        - Middle-decile symbols are omitted entirely (absent from the dict,
          unless they need a 0.0 for EXIT).
    """
    results: dict[pd.Timestamp, dict[str, float]] = {}
    prev_active: set[str] = set()   # symbols in longs/shorts at prior rebalance

    for date, row in scores.iterrows():
        row_clean = row.dropna()

        n = len(row_clean)
        leg_size = max(1, n // n_deciles)

        # Rank ascending → top decile = highest rank
        ranked = row_clean.rank(ascending=True, method="first")
        sorted_ranked = ranked.sort_values(ascending=True)

        bottom_syms = list(sorted_ranked.index[:leg_size])
        top_syms    = list(sorted_ranked.index[-leg_size:])

        long_weight  =  1.0 / len(top_syms)
        short_weight = -1.0 / len(bottom_syms)

        w: dict[str, float] = {}

        # Symbols from prior period that now drop to neither leg → EXIT (weight 0.0)
        current_active = set(top_syms) | set(bottom_syms)
        for sym in prev_active:
            if sym not in current_active:
                w[sym] = 0.0

        # Assign new weights
        for sym in top_syms:
            w[sym] = long_weight
        for sym in bottom_syms:
            w[sym] = short_weight

        results[date] = w
        prev_active = current_active

    return results
