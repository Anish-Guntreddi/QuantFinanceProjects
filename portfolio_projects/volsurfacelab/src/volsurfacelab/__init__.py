"""VolSurfaceLab — Options Volatility Surface Research System.

Public API
----------

Chain (synthetic data + real-data loader)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SyntheticChainGenerator — deterministic SVI-based synthetic chain generator
ChainData               — frozen dataclass: options chain + metadata
generate_underlying_returns — seeded GARCH(1,1) daily return path
validate_chain_coverage     — chain coverage validator
SYNTHETIC_SVI_SURFACE       — ground-truth SVI parameter dict by maturity

IV Solver
~~~~~~~~~
robust_iv      — single-option IV: LetsBeRational + brentq fallback
solve_chain_iv — vectorised IV solve for a full ChainData
bs_price       — standalone Black-Scholes closed-form pricer

SVI Calibration
~~~~~~~~~~~~~~~
svi_w             — SVI total-variance formula: w(k)
g_func            — butterfly density g(k) for no-arb check
fit_svi_slice     — SLSQP single-maturity SVI fit (butterfly + positivity constraints)
check_calendar_arb — pairwise calendar-spread violation check on calibrated slices
validate_surface  — surface-level no-arb gate (warns + excludes; never raises)
calibrate_surface — full surface: fit all slices, gate, return validated fits
SVISliceFit       — dataclass: fitted SVI parameters for one maturity slice

Forecast
~~~~~~~~
realized_variance  — Rogers-Satchell realized variance from OHLCV returns
HARForecaster      — HAR-RV rolling forecaster (statsmodels OLS)
fit_garch_robust   — GARCH/EGARCH multi-restart fit (arch library)
qlike              — Patton (2011) QLIKE loss: rv/h - log(rv/h) - 1
diebold_mariano    — DM test for equal predictive accuracy
compare_forecasts  — compare HAR vs GARCH/EGARCH: QLIKE, MSE, DM
ForecastComparison — dataclass: forecast comparison results

Strategy
~~~~~~~~
OptionLeg             — immutable options position (dataclass)
StandalonePortfolio   — standalone delta-hedged portfolio accounting (no qbacktest dep)
daily_gamma_pnl       — daily P&L from gamma scalping
compute_leg_greeks    — delta/gamma/theta/vega for a single leg
portfolio_greeks_summary — aggregate greeks across all legs
run_vrp_strategy      — run VRP delta-hedged straddle strategy
VRPResult             — dataclass: full strategy results

Pipeline
~~~~~~~~
VolSurfacePipeline — end-to-end pipeline: chain → IV → SVI → forecast → strategy → report
PipelineResults    — frozen dataclass returned by VolSurfacePipeline.run()
load_config        — load YAML config (with defaults)

Report (lazy — pulled only on first access; avoids matplotlib.pyplot at package init)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ReportBuilder      — figures + markdown summary for the pipeline
load_yfinance_chain — optional real-data loader (lazy yfinance import)

Version
~~~~~~~
__version__        — package version string
"""

from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
from volsurfacelab.chain import (
    SyntheticChainGenerator,
    ChainData,
    generate_underlying_returns,
    validate_chain_coverage,
    SYNTHETIC_SVI_SURFACE,
)

# ---------------------------------------------------------------------------
# IV Solver
# ---------------------------------------------------------------------------
from volsurfacelab.iv_solver import (
    robust_iv,
    solve_chain_iv,
    bs_price,
)

# ---------------------------------------------------------------------------
# SVI Calibration
# ---------------------------------------------------------------------------
from volsurfacelab.svi import (
    svi_w,
    g_func,
    fit_svi_slice,
    check_calendar_arb,
    validate_surface,
    calibrate_surface,
    SVISliceFit,
)

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
from volsurfacelab.forecast import (
    realized_variance,
    HARForecaster,
    fit_garch_robust,
    qlike,
    diebold_mariano,
    compare_forecasts,
    ForecastComparison,
)

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
from volsurfacelab.strategy import (
    OptionLeg,
    StandalonePortfolio,
    daily_gamma_pnl,
    compute_leg_greeks,
    portfolio_greeks_summary,
    run_vrp_strategy,
    VRPResult,
)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
from volsurfacelab.pipeline import (
    VolSurfacePipeline,
    PipelineResults,
    load_config,
)


# ---------------------------------------------------------------------------
# Lazy exports — pulled only on first access.
#
# ReportBuilder: volsurfacelab/report.py calls matplotlib.use("Agg") and
# imports matplotlib.pyplot at module scope.  Deferring to __getattr__ keeps
# `import volsurfacelab` headless-safe (no pyplot side-effect).
# This mirrors the macroregime ReportBuilder pattern (Phase 3 locked).
#
# load_yfinance_chain: volsurfacelab/chain.py does NOT import yfinance at
# module scope (yfinance is inside the function body), so the chain module
# itself is safe to import eagerly.  However, we keep load_yfinance_chain
# lazy here so that the volsurfacelab package namespace never accidentally
# triggers a yfinance import via future refactors.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    if name == "ReportBuilder":
        from volsurfacelab.report import ReportBuilder
        return ReportBuilder
    if name == "load_yfinance_chain":
        from volsurfacelab.chain import load_yfinance_chain
        return load_yfinance_chain
    raise AttributeError(f"module 'volsurfacelab' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Public API surface (frozen in plan 04-08)
# ---------------------------------------------------------------------------

__all__ = [
    "__version__",
    # Chain
    "SyntheticChainGenerator",
    "ChainData",
    "generate_underlying_returns",
    "validate_chain_coverage",
    "SYNTHETIC_SVI_SURFACE",
    # IV Solver
    "robust_iv",
    "solve_chain_iv",
    "bs_price",
    # SVI Calibration
    "svi_w",
    "g_func",
    "fit_svi_slice",
    "check_calendar_arb",
    "validate_surface",
    "calibrate_surface",
    "SVISliceFit",
    # Forecast
    "realized_variance",
    "HARForecaster",
    "fit_garch_robust",
    "qlike",
    "diebold_mariano",
    "compare_forecasts",
    "ForecastComparison",
    # Strategy
    "OptionLeg",
    "StandalonePortfolio",
    "daily_gamma_pnl",
    "compute_leg_greeks",
    "portfolio_greeks_summary",
    "run_vrp_strategy",
    "VRPResult",
    # Pipeline
    "VolSurfacePipeline",
    "PipelineResults",
    "load_config",
    # Lazy (avoids pyplot / yfinance at package import)
    "ReportBuilder",
    "load_yfinance_chain",
]
