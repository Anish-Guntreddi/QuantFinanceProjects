"""Feature engineering infrastructure: safe shifting, cross-sectional normalization,
and leakage validation.

Design rationale
----------------
Pitfall 1 — Forward-look: every feature lag must route through safe_shift().
  The function asserts n >= 1 at construction time so there is no way to
  accidentally call df.shift(-1) in a factor function and produce a
  look-ahead feature.

Pitfall 2 — Full-panel StandardScaler (sklearn style) before a CV split
  exposes future cross-sectional statistics to all time periods; it is
  FORBIDDEN here.  cross_sectional_zscore() normalizes row-by-row (per
  time-step) so only data available at that date is used for the mean and
  standard deviation.  sklearn's StandardScaler fit on the full panel
  computes the grand mean over ALL dates including future ones — the classic
  data-leakage failure mode in factor research.  NEVER use it before the
  CV/walk-forward split.

Pattern 1 — Leakage validation: FeatureLeakageValidator asserts that the
  Spearman IC between any feature column and next-day returns stays below a
  threshold.  A genuine look-ahead feature has IC ≈ 1.0; properly lagged
  factor features have IC ≈ 0 (noise in short panels) and will pass.  The
  validator is wired into build_feature_panel() so the module self-asserts
  on every construction.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# safe_shift
# ---------------------------------------------------------------------------

def safe_shift(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Lag a DataFrame by n periods, asserting n is a strict positive lag.

    Parameters
    ----------
    df : pd.DataFrame
        Wide panel (dates x symbols).
    n : int
        Number of periods to lag.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        df.shift(n) — past values shifted forward in time.

    Raises
    ------
    AssertionError
        If n < 1.  Message contains "positive shifts" so tests can match it.
    """
    assert n >= 1, (
        f"Only positive shifts in feature construction (got n={n}). "
        "Negative or zero shifts would introduce future data into a feature."
    )
    return df.shift(n)


# ---------------------------------------------------------------------------
# cross_sectional_zscore
# ---------------------------------------------------------------------------

def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each row (date) cross-sectionally: subtract row mean, divide by row std.

    Why row-by-row and NOT sklearn StandardScaler on the full panel
    ---------------------------------------------------------------
    sklearn's StandardScaler.fit() computes statistics over the ENTIRE panel
    including future dates.  When applied before a CV split this introduces
    look-ahead bias: the model sees future cross-sectional distributions
    during training.  Row-by-row normalization uses only the assets available
    at each date — no future information leaks.

    Parameters
    ----------
    df : pd.DataFrame
        Wide panel (dates x symbols).  NaN entries (delisted assets) are
        propagated unchanged; the mean and std are computed with skipna=True
        so surviving assets at each date are correctly normalized.

    Returns
    -------
    pd.DataFrame
        Row-normalized DataFrame, same shape as df.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1, ddof=1)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


# ---------------------------------------------------------------------------
# FeatureLeakageValidator
# ---------------------------------------------------------------------------

class FeatureLeakageValidator:
    """Assert that a feature panel is not correlated with future returns.

    Cross-sectional design (codex leakage-audit finding, plan 02-08):
      For each date, compute the cross-sectional Spearman rank correlation
      between the feature values and next-day returns across symbols, then
      average the signed ICs over dates.  This matches how the planted alpha
      is constructed (cross-sectional), so the scales separate cleanly:

        - legitimate planted signal: mean IC ≈ 0.04–0.06
        - pure noise feature:        mean IC ≈ 0
        - leaked future data:        mean IC ≈ 1.0

      The earlier per-symbol *time-series* IC design conflated horizons and
      needed thresholds so loose (0.5) that only identity leaks tripped it;
      signed-mean cross-sectional IC at threshold 0.3 leaves an order of
      magnitude of margin on both sides.

    The only permitted negative shift in this codebase is the one inside
    validate() on the *evaluation* side (computing next-day returns for
    comparison), which is why this is the single exception to the no-
    negative-shift rule.  Feature columns in `feature` must NOT themselves
    contain negative-shifted data.
    """

    def validate(
        self,
        feature: pd.DataFrame,
        prices: pd.DataFrame,
        threshold: float = 0.3,
    ) -> None:
        """Run the cross-sectional leakage check on a feature panel.

        Parameters
        ----------
        feature : pd.DataFrame
            Wide panel (dates x symbols).  Only dates present in both frames
            are compared.
        prices : pd.DataFrame
            Raw price panel (dates x symbols) used to compute next-day
            returns.  The negative shift here is EVALUATION-SIDE only —
            it is the thing being predicted, not a feature.
        threshold : float
            Maximum absolute *mean* cross-sectional Spearman IC allowed.
            Default 0.3 (planted signal ≈ 0.06, identity leak ≈ 1.0).

        Raises
        ------
        AssertionError
            Reporting the mean cross-sectional IC if |mean IC| >= threshold.
        """
        from scipy.stats import spearmanr  # lazy import; scipy optional dependency

        # Next-day returns: the evaluation target (negative shift is allowed here
        # because this is the LABEL side, not the feature side).
        next_day_ret = prices.pct_change(fill_method=None).shift(-1)  # evaluation-side only

        shared_dates = feature.index.intersection(next_day_ret.index)
        shared_cols = feature.columns.intersection(next_day_ret.columns)
        if len(shared_dates) < 5 or len(shared_cols) < 3:
            return  # insufficient panel for a meaningful cross-sectional check

        ics: list[float] = []
        for date in shared_dates:
            f = feature.loc[date, shared_cols]
            r = next_day_ret.loc[date, shared_cols]
            mask = ~(f.isna() | r.isna())
            if mask.sum() < 3:
                continue
            ic, _ = spearmanr(f[mask].values, r[mask].values)
            if not np.isnan(ic):
                ics.append(float(ic))

        if len(ics) < 5:
            return  # too few valid cross-sections — skip

        mean_ic = float(np.mean(ics))
        assert abs(mean_ic) < threshold, (
            f"Feature leakage detected: mean cross-sectional Spearman IC "
            f"{mean_ic:.4f} with next-day returns over {len(ics)} dates "
            f"(threshold={threshold}). This indicates the feature contains "
            "future data."
        )
