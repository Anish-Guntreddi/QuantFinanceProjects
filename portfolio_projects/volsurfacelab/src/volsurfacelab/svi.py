"""SVI (Stochastic Volatility Inspired) calibration and no-arbitrage validation.

Implements VSL-03:
- Per-maturity SVI slice fitting via multi-restart SLSQP with Gatheral-Jacquier
  butterfly constraint g(k) >= 0 discretized over a k-grid.
- Static no-arbitrage gate: butterfly convexity per slice + calendar monotonicity
  across slices, checked on the TRADED moneyness range only ([-1.5, 1.5]).
- Violated slices are logged with warnings.warn and excluded — never silently
  passed downstream.

Math reference:
  w(k)   = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
  w'(k)  = b*(rho + (k-m)/sqrt((k-m)^2 + sigma^2))
  w''(k) = b*sigma^2 / ((k-m)^2 + sigma^2)^1.5
  g(k)   = (1 - k*w'/(2*w))^2 - (w'^2/4)*(1/w + 0.25) + w''/2   [butterfly]

Source: Gatheral & Jacquier "Arbitrage-free SVI volatility surfaces" (2014).
Calendar check restricted to traded k-range: deep-wing SVI behavior can violate
calendar monotonicity even for sensible params; only the traded range is actionable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "svi_w",
    "svi_wp",
    "svi_wpp",
    "g_func",
    "SVISliceFit",
    "fit_svi_slice",
    "check_calendar_arb",
    "validate_surface",
    "calibrate_surface",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Discretized k-grid for butterfly constraint enforcement in SLSQP
K_CONSTRAINT: np.ndarray = np.linspace(-3.0, 3.0, 100)

# Parameter bounds for SLSQP:  (a, b, rho, m, sigma)
# a: can be slightly negative for short-dated surfaces (pitfall 7 in RESEARCH.md)
# b: wing steepness; strictly positive
# rho: skew; bounded away from +-1 for numerical stability
# m: ATM shift; reasonable range
# sigma: smile width / ATM curvature; strictly positive
SVI_BOUNDS = [
    (-0.5, 1.0),       # a
    (1e-4, 2.0),       # b
    (-0.999, 0.999),   # rho
    (-3.0, 3.0),       # m
    (1e-4, 2.0),       # sigma
]

# Multi-restart initial guesses covering negative/zero/positive rho and
# varying wing steepness — verified to converge in quant venv.
INITIAL_GUESSES = [
    (0.04, 0.3, -0.3, 0.0, 0.2),
    (0.02, 0.2, -0.1, -0.1, 0.3),
    (0.06, 0.4, -0.5, 0.1, 0.15),
    (0.01, 0.15, 0.0, 0.0, 0.4),
    (0.08, 0.5, -0.7, 0.0, 0.1),
]


# ---------------------------------------------------------------------------
# SVI analytic functions (vectorized numpy)
# ---------------------------------------------------------------------------

def svi_w(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """SVI total variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_wp(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """First derivative w'(k) = b*(rho + (k-m)/sqrt((k-m)^2 + sigma^2))."""
    return b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_wpp(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Second derivative w''(k) = b*sigma^2 / ((k-m)^2 + sigma^2)^1.5."""
    return b * sigma ** 2 / ((k - m) ** 2 + sigma ** 2) ** 1.5


def g_func(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Gatheral-Jacquier butterfly density g(k).

    g(k) = (1 - k*w'/(2*w))^2 - (w'^2/4)*(1/w + 0.25) + w''/2

    No-arbitrage condition: g(k) >= 0 for all k.

    Source: Gatheral & Jacquier (2014) eq. (2.1).
    """
    w = svi_w(k, a, b, rho, m, sigma)
    wp = svi_wp(k, a, b, rho, m, sigma)
    wpp = svi_wpp(k, a, b, rho, m, sigma)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp ** 2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


# ---------------------------------------------------------------------------
# SVISliceFit dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SVISliceFit:
    """Result of fitting a single SVI maturity slice.

    Attributes:
        T: Maturity (time to expiry in years).
        params: Fitted (a, b, rho, m, sigma) tuple, or None if all restarts failed.
        success: True iff at least one SLSQP restart converged.
        sse: Sum of squared errors at the best solution (inf if success=False).
        n_restarts_used: Number of restarts attempted.
    """
    T: float
    params: Optional[tuple]
    success: bool
    sse: float
    n_restarts_used: int


# ---------------------------------------------------------------------------
# fit_svi_slice — multi-restart SLSQP with butterfly constraint
# ---------------------------------------------------------------------------

def fit_svi_slice(
    k_obs: np.ndarray,
    w_obs: np.ndarray,
    k_constraint: Optional[np.ndarray] = None,
    initial_guesses: Optional[list] = None,
    T: float = float("nan"),
) -> SVISliceFit:
    """Fit a raw SVI model to one maturity slice.

    Uses multi-restart SLSQP with the Gatheral-Jacquier g(k) >= 0 butterfly
    constraint discretized over a k-grid. Only convergent (res.success=True)
    restarts are considered. Returns the best result by SSE, or a failure
    SVISliceFit if every restart fails or raises.

    Parameters
    ----------
    k_obs : array of log-moneyness observations for this slice.
    w_obs : array of total variance observations (true_iv^2 * T) for this slice.
    k_constraint : k-grid for butterfly constraint (default: K_CONSTRAINT = [-3,3] x 100).
    initial_guesses : list of (a, b, rho, m, sigma) starting points.
    T : maturity label stored in the returned SVISliceFit.

    Returns
    -------
    SVISliceFit with success=True and best params, or success=False if all restarts fail.
    """
    k_obs = np.asarray(k_obs, dtype=float)
    w_obs = np.asarray(w_obs, dtype=float)

    if k_constraint is None:
        k_constraint = K_CONSTRAINT
    if initial_guesses is None:
        initial_guesses = INITIAL_GUESSES

    def objective(p: np.ndarray) -> float:
        return float(np.sum((svi_w(k_obs, *p) - w_obs) ** 2))

    # SLSQP vectorized inequality constraint: g(k) >= 0 over constraint grid
    butterfly_constraint = {
        "type": "ineq",
        "fun": lambda p: g_func(k_constraint, *p),
    }
    # Also enforce w(k) > 0 over constraint grid (handles negative-a pitfall 7)
    positivity_constraint = {
        "type": "ineq",
        "fun": lambda p: svi_w(k_constraint, *p),
    }

    best_params: Optional[np.ndarray] = None
    best_fun: float = np.inf
    n_restarts = len(initial_guesses)

    for x0 in initial_guesses:
        try:
            res = minimize(
                objective,
                np.array(x0, dtype=float),
                method="SLSQP",
                bounds=SVI_BOUNDS,
                constraints=[butterfly_constraint, positivity_constraint],
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if res.success and res.fun < best_fun:
                best_params = res.x
                best_fun = res.fun
        except Exception:
            # SLSQP can raise on pathological inputs (e.g. ill-conditioned Jacobian)
            # A restart failure is skipped, never propagated — robustness requirement.
            continue

    if best_params is None:
        return SVISliceFit(
            T=T,
            params=None,
            success=False,
            sse=float("inf"),
            n_restarts_used=n_restarts,
        )

    return SVISliceFit(
        T=T,
        params=tuple(float(x) for x in best_params),
        success=True,
        sse=float(best_fun),
        n_restarts_used=n_restarts,
    )


# ---------------------------------------------------------------------------
# check_calendar_arb — monotonicity check over TRADED range only
# ---------------------------------------------------------------------------

def check_calendar_arb(
    params_by_maturity: dict,
    k_grid: Optional[np.ndarray] = None,
    tol: float = 1e-10,
) -> list:
    """Check calendar (term-structure) no-arbitrage over the TRADED moneyness range.

    Calendar no-arb requires w(k, T2) >= w(k, T1) for all k whenever T2 > T1.

    CRITICAL: The default k-grid is restricted to [-1.5, 1.5] (traded range only).
    Deep-wing SVI behavior at |k| > 1.5 can produce spurious violations even
    for well-calibrated surfaces — these are parameterization artifacts, not
    actionable arbitrage.  (See RESEARCH.md pitfall 2.)

    Parameters
    ----------
    params_by_maturity : dict mapping T -> (a, b, rho, m, sigma)
    k_grid : moneyness grid to check (default: np.linspace(-1.5, 1.5, 200))
    tol : tolerance for declaring a violation (violation if w2 < w1 - tol)

    Returns
    -------
    List of (T_short, T_long, n_violated_points) for each violating adjacent pair.
    Empty list if no violations.
    """
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 200)  # TRADED range only — see pitfall 2

    maturities = sorted(params_by_maturity)
    violations = []

    for T1, T2 in zip(maturities[:-1], maturities[1:]):
        w1 = svi_w(k_grid, *params_by_maturity[T1])
        w2 = svi_w(k_grid, *params_by_maturity[T2])
        n_violated = int(np.sum(w2 < w1 - tol))
        if n_violated > 0:
            violations.append((T1, T2, n_violated))

    return violations


# ---------------------------------------------------------------------------
# validate_surface — butterfly + calendar gate with warnings, no exceptions
# ---------------------------------------------------------------------------

def validate_surface(params_by_maturity: dict) -> dict:
    """Remove arbitrage-violating slices from a fitted SVI surface.

    Two-pass gate:
    1. Butterfly check per slice — g(k) >= -1e-8 over [-3, 3] x 100 points.
       Also verifies w(k) > 0 over the same grid.
       Violations: warn("butterfly violation ... excluded") + drop slice.
    2. Calendar check on surviving slices — w(k, T2) >= w(k, T1) over [-1.5, 1.5].
       Violations: warn("Calendar violation ...") + drop the LONGER maturity.

    Warnings use warnings.warn with UserWarning.  Gate behavior is never to raise
    — the rest of the surface proceeds regardless.

    Parameters
    ----------
    params_by_maturity : dict mapping T -> (a, b, rho, m, sigma)

    Returns
    -------
    Dict mapping T -> params for surviving slices only.
    """
    butterfly_k_grid = np.linspace(-3.0, 3.0, 100)

    # --- Pass 1: butterfly check per slice ---
    clean: dict = {}
    for T, params in sorted(params_by_maturity.items()):
        g_vals = g_func(butterfly_k_grid, *params)
        w_vals = svi_w(butterfly_k_grid, *params)

        butterfly_ok = np.all(g_vals >= -1e-8)
        positivity_ok = np.all(w_vals > 0)

        if not butterfly_ok:
            warnings.warn(
                f"Slice T={T:.4f}: butterfly violation "
                f"(min g={g_vals.min():.6f}), excluded",
                UserWarning,
                stacklevel=2,
            )
        elif not positivity_ok:
            warnings.warn(
                f"Slice T={T:.4f}: total variance w(k) <= 0 "
                f"(min w={w_vals.min():.6f}), excluded",
                UserWarning,
                stacklevel=2,
            )
        else:
            clean[T] = params

    # --- Pass 2: calendar check on survivors (traded range only) ---
    cal_violations = check_calendar_arb(clean)
    for T1, T2, n in cal_violations:
        warnings.warn(
            f"Calendar violation: T={T1:.4f} vs T={T2:.4f} "
            f"at {n} moneyness points, excluding T={T2:.4f}",
            UserWarning,
            stacklevel=2,
        )
        clean.pop(T2, None)

    return clean


# ---------------------------------------------------------------------------
# calibrate_surface — end-to-end: fit + validate, returns (fits, excluded)
# ---------------------------------------------------------------------------

def calibrate_surface(
    chain,
    calendar_k_range: tuple = (-1.5, 1.5),
) -> tuple:
    """Fit SVI to each maturity slice and run the no-arbitrage gate.

    Groups chain.options by T, builds w_obs = true_iv^2 * T from call rows
    (calls and puts share the same true_iv, so calls are sufficient), fits
    each slice via fit_svi_slice, validates the fitted surface, and returns
    only the validated slices together with an excluded list.

    Reads nothing from disk — config values are passed as arguments.

    Parameters
    ----------
    chain : ChainData with options DataFrame (columns: T, k, flag, true_iv, ...)
    calendar_k_range : (k_min, k_max) tuple for calendar arb check
                       (default (-1.5, 1.5) — traded range only)

    Returns
    -------
    (fits, excluded) where:
        fits     : dict mapping T -> SVISliceFit for validated slices
        excluded : list of (T, reason) tuples for excluded slices
    """
    options = chain.options
    excluded: list = []

    # Build per-maturity observations using call rows only
    # (calls and puts share true_iv; using calls avoids duplicate k observations)
    calls = options[options["flag"] == "c"].copy()
    maturities = sorted(calls["T"].unique())

    raw_fits: dict = {}
    for T in maturities:
        slice_df = calls[calls["T"] == T]
        k_obs = slice_df["k"].values
        w_obs = (slice_df["true_iv"].values ** 2) * T
        fit = fit_svi_slice(k_obs, w_obs, T=T)
        raw_fits[T] = fit
        if not fit.success:
            excluded.append((T, "fit_failed"))

    # Build params dict for successful fits only
    successful_params: dict = {}
    for T, fit in raw_fits.items():
        if fit.success:
            successful_params[T] = fit.params

    # Run validate_surface; collect warnings as exclusions
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        validated_params = validate_surface(successful_params)

    # Re-emit the caught warnings so callers see them
    for w in caught_warnings:
        warnings.warn(w.message, w.category, stacklevel=2)

    # Determine which slices were excluded by the gate
    excluded_by_gate = set(successful_params.keys()) - set(validated_params.keys())
    for T in excluded_by_gate:
        # Determine reason from the warning message
        reason_msg = "gate_excluded"
        for w in caught_warnings:
            msg = str(w.message)
            t_str = f"T={T:.4f}"
            if t_str in msg:
                if "butterfly" in msg:
                    reason_msg = "butterfly_violation"
                elif "Calendar" in msg or "calendar" in msg:
                    reason_msg = "calendar_violation"
                elif "w(k)" in msg or "total variance" in msg:
                    reason_msg = "positivity_violation"
                break
        excluded.append((T, reason_msg))

    # Build final fits dict — only validated slices
    validated_fits = {
        T: raw_fits[T] for T in validated_params.keys() if T in raw_fits
    }

    return validated_fits, excluded
