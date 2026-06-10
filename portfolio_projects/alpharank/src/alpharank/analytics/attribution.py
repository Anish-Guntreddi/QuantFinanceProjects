"""Factor attribution via OLS regression.

Decomposes strategy returns into factor exposures (betas), alpha (intercept),
and residual (idiosyncratic) returns using statsmodels OLS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def factor_attribution(
    strategy_rets: pd.Series,
    factor_rets: pd.DataFrame,
) -> dict:
    """Regress strategy returns on factor returns via OLS.

    Fits the model::

        strategy_rets[t] = alpha + sum_k(beta_k * factor_rets[t, k]) + eps[t]

    using ``statsmodels.OLS`` with ``add_constant``.

    Parameters
    ----------
    strategy_rets : pd.Series
        Time series of strategy returns (monthly).  Index is a DatetimeIndex.
    factor_rets : pd.DataFrame
        Factor return matrix, shape (T', n_factors).  Columns are factor names.
        Will be reindexed to ``strategy_rets.index`` (inner join).

    Returns
    -------
    dict with keys:
        alpha : float
            OLS intercept (annualised only if the caller chooses).
        alpha_tstat : float
            t-statistic for the intercept.
        alpha_pvalue : float
            Two-sided p-value for the intercept.
        betas : dict[str, float]
            OLS slope coefficients keyed by factor name.
        r_squared : float
            In-sample R-squared of the regression.
        residual : pd.Series
            Residuals indexed like ``strategy_rets``.

    Notes
    -----
    Factor returns are reindexed to the strategy index using an inner join.
    Any rows with NaN in either ``strategy_rets`` or ``factor_rets`` are
    dropped before fitting.
    """
    # Align strategy and factor returns on shared dates
    aligned_factors = factor_rets.reindex(strategy_rets.index)

    # Build combined frame and drop any rows with NaN
    combined = pd.concat(
        [strategy_rets.rename("__y__"), aligned_factors], axis=1
    ).dropna()

    y = combined["__y__"].values
    X_raw = combined.drop(columns="__y__")
    factor_names = list(X_raw.columns)
    X = sm.add_constant(X_raw.values, prepend=True)

    model = sm.OLS(y, X).fit()

    alpha = float(model.params[0])
    alpha_tstat = float(model.tvalues[0])
    alpha_pvalue = float(model.pvalues[0])

    betas = {name: float(model.params[i + 1]) for i, name in enumerate(factor_names)}
    r_squared = float(model.rsquared)

    residual = pd.Series(
        model.resid,
        index=combined.index,
        name="residual",
    )

    return {
        "alpha": alpha,
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "betas": betas,
        "r_squared": r_squared,
        "residual": residual,
    }
