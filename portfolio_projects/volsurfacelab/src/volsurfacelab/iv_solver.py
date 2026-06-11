"""Robust implied-volatility solver for VolSurfaceLab.

Contract: NEVER raises. All failure modes return float('nan').

Primary path: LetsBeRational (vollib 1.0.12) — fast rational-approximation IV
inversion, accurate to ~1e-14 on valid inputs.

Fallback path: scipy brentq bisection on the Black-Scholes price function.
Activated when the primary path raises a non-economic exception (RuntimeError,
ValueError, etc.) or returns a non-positive / non-finite result.

Economic exceptions from py_lets_be_rational.exceptions:
  - BelowIntrinsicException  -> NaN immediately (economically meaningless quote)
  - AboveMaximumException    -> NaN immediately (economically meaningless quote)

Namespace note: vollib.* is used throughout (NOT py_vollib.*) to avoid the
DeprecationWarning emitted by the py_vollib legacy namespace (both are provided
by the py-vollib 1.0.12 package; vollib is the canonical name).
"""

import logging
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from vollib.black_scholes.implied_volatility import implied_volatility as _lbr_iv

try:
    from py_lets_be_rational.exceptions import (
        AboveMaximumException,
        BelowIntrinsicException,
    )
except ImportError:  # pragma: no cover — both always present in quant venv
    # Fallback: define placeholder exceptions so the except clauses still compile
    class BelowIntrinsicException(Exception):  # type: ignore[no-redef]
        pass

    class AboveMaximumException(Exception):  # type: ignore[no-redef]
        pass


from volsurfacelab.chain import ChainData

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["bs_price", "robust_iv", "solve_chain_iv"]

# ---------------------------------------------------------------------------
# Black-Scholes price (independent of vollib — used by brentq fallback
# and reusable by tests as a second implementation)
# ---------------------------------------------------------------------------

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def bs_price(sigma: float, S: float, K: float, T: float, r: float, flag: str) -> float:
    """Closed-form Black-Scholes option price.

    Parameters
    ----------
    sigma : float
        Implied volatility (annualized, >0).
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years (>0).
    r : float
        Continuously-compounded risk-free rate.
    flag : str
        'c' for call, 'p' for put.

    Returns
    -------
    float
        BS theoretical option price.
    """
    if sigma <= 0.0 or T <= 0.0 or S <= 0.0 or K <= 0.0:
        return float("nan")
    sq_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sq_T)
    d2 = d1 - sigma * sq_T
    if flag == "c":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Robust IV solver
# ---------------------------------------------------------------------------

_BRENTQ_LO = 1e-6
_BRENTQ_HI = 20.0


def robust_iv(
    price: float, S: float, K: float, T: float, r: float, flag: str
) -> float:
    """Implied volatility, or float('nan') on irrecoverable failure.

    NEVER raises an exception under any input.

    Parameters
    ----------
    price : float
        Market option price.
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously-compounded risk-free rate.
    flag : str
        'c' for call, 'p' for put.

    Returns
    -------
    float
        Recovered implied volatility, or NaN if not recoverable.
    """
    # Guard rails: non-positive inputs are economically meaningless
    if price <= 0.0 or T <= 0.0 or S <= 0.0 or K <= 0.0:
        return float("nan")

    # -- Primary: LetsBeRational ------------------------------------------------
    try:
        result = _lbr_iv(price, S, K, T, r, flag)
        if isinstance(result, float) and result > 0.0 and math.isfinite(result):
            return result
        # Non-positive or non-finite: fall through to brentq
    except (BelowIntrinsicException, AboveMaximumException):
        # Economically invalid quote — no vol exists
        return float("nan")
    except Exception:
        # Any other LBR failure (RuntimeError, ValueError, etc.) — try brentq
        pass

    # -- Fallback: brentq bisection on bs_price(sigma) - price -----------------
    # brentq requires a sign change on [lo, hi].
    # If there is no bracketing root (vanishing-vega deep OTM, or the price is
    # outside the no-arb range at these bounds), brentq raises ValueError and
    # we return NaN — this is the correct behavior (vol is not recoverable).
    try:
        f_lo = bs_price(_BRENTQ_LO, S, K, T, r, flag) - price
        f_hi = bs_price(_BRENTQ_HI, S, K, T, r, flag) - price
        if f_lo * f_hi > 0.0:
            # No sign change — no bracketing root; price outside solvable range
            return float("nan")
        iv = brentq(
            lambda sig: bs_price(sig, S, K, T, r, flag) - price,
            _BRENTQ_LO,
            _BRENTQ_HI,
            xtol=1e-10,
            maxiter=200,
        )
        if iv > 0.0 and math.isfinite(iv):
            return iv
        return float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Chain-wide solver
# ---------------------------------------------------------------------------


def solve_chain_iv(chain: ChainData) -> pd.DataFrame:
    """Solve implied volatility for every row in a ChainData options frame.

    Parameters
    ----------
    chain : ChainData
        Options chain from SyntheticChainGenerator (or equivalent).

    Returns
    -------
    pd.DataFrame
        Copy of chain.options with an added 'iv' column.  Rows that could
        not be solved hold NaN.  The number of failed rows (NaN) is logged
        at WARNING level.
    """
    df = chain.options.copy()
    ivs = [
        robust_iv(row.price, chain.spot, row.K, row.T, chain.risk_free, row.flag)
        for row in df.itertuples(index=False)
    ]
    df = df.assign(iv=ivs)
    n_failed = int(pd.isna(df["iv"]).sum())
    n_total = len(df)
    if n_failed > 0:
        logger.warning(
            "solve_chain_iv: %d / %d rows failed (NaN iv); "
            "check for below-intrinsic or above-maximum prices.",
            n_failed,
            n_total,
        )
    else:
        logger.debug(
            "solve_chain_iv: all %d rows solved successfully.", n_total
        )
    return df
