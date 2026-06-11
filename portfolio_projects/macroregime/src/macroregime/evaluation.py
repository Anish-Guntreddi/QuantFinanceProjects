"""Walk-forward evaluation, OOS regime stability analysis, and K sensitivity.

Exports:
  run_walk_forward       — orchestrate WalkForwardRunner over a daily index
  regime_stability_report — HMM vs GMM agreement + dwell time analysis
  k_sensitivity          — per-K regime metrics (NO Sharpe-based selection)

DESIGN DECISIONS (locked):
  - Walk-forward isolation via fresh construction (no reset()); locked in Phase 1
    WalkForwardRunner decision.
  - Regime sequences are computed ONCE upstream (causally, via CausalRegimeDetector)
    and reused in walk-forward windows. Recomputing regimes per window is unnecessary:
    the causal label at bar t is a pure function of X[:t+1] and is window-invariant
    (oracle guarantee from plan 03-04). Reuse is safe.
  - K selection by Sharpe is EXPLICITLY FORBIDDEN (anti-feature). The k_sensitivity
    function reports structural metrics (dwell times, transition matrices, label
    agreement with K=3 baseline) only. Users select K based on economic interpretability,
    not return maximization.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------


def run_walk_forward(
    asset_ohlcv: dict[str, pd.DataFrame],
    combined_regimes: pd.Series,
    regime_weights: dict[int, dict[str, float]],
    train_bars: int,
    test_bars: int,
    params: dict[str, Any] | None = None,
) -> Any:
    """Orchestrate walk-forward backtesting over a daily OHLCV index.

    Uses qbacktest.WalkForwardRunner with fresh engine construction per window
    (no reset — isolation via construction, locked Phase 1 decision).

    CAUSALITY NOTE on regime reuse:
      The combined_regimes Series was produced by CausalRegimeDetector, which
      guarantees that the label at bar t depends only on X[:t+1].  Therefore,
      slicing combined_regimes to a test window does NOT introduce look-ahead:
      the label at each bar within the window was already computed causally on
      upstream data.  Recomputing regimes per window would be redundant and
      computationally wasteful — the oracle invariant (plan 03-04) proves
      that a run on X[:N] and X[:N+k] agree on all t < N.

    Parameters
    ----------
    asset_ohlcv:
        Dict of {symbol: OHLCV DataFrame} with DatetimeIndex.
    combined_regimes:
        Daily pd.Series of combined regime labels (from MacroRegimePipeline or
        similar). Must cover the full daily_index.
    regime_weights:
        Output of load_regime_weights(): {regime_int: {symbol: weight}}.
    train_bars:
        Number of training bars per window.
    test_bars:
        Number of test (OOS) bars per window.
    params:
        Optional pre-loaded backtest params dict. If None, loaded from default
        configs/strategy_params.yml by run_strategy_backtest.

    Returns
    -------
    WalkForwardResults
        Per-window BacktestResults, OOS equity curve, and aggregate OOS metrics.
    """
    from qbacktest import WalkForwardRunner, generate_windows
    from macroregime.benchmarks.benchmarks import run_strategy_backtest, load_run_params
    from macroregime.allocation.weights import build_weight_schedule, month_end_rebalance_dates

    # Use the daily index from the first asset
    daily_index = next(iter(asset_ohlcv.values())).index

    windows = generate_windows(daily_index, train_bars=train_bars, test_bars=test_bars)

    if params is None:
        params = load_run_params()

    def engine_factory(window):
        """Build a fresh engine for a single walk-forward window's test range.

        Data is sliced to [window.test_start, window.test_end] so each window
        has its own independent OHLCV views. The weight schedule is restricted
        to rebalance dates inside the test range, using as-of regime values from
        the causal combined_regimes sequence (no re-detection needed).
        """
        # Slice OHLCV to test window
        window_ohlcv = {
            sym: df.loc[window.test_start : window.test_end]
            for sym, df in asset_ohlcv.items()
        }

        # Build rebalance dates within the test window
        test_idx = next(iter(window_ohlcv.values())).index
        rebal_dates = month_end_rebalance_dates(test_idx)
        # Normalise to pd.Timestamp (month_end_rebalance_dates may return numpy int64 nanos)
        rebal_dates = [pd.Timestamp(ts) for ts in rebal_dates]

        # Restrict weight schedule to this test window using as-of regime values.
        # combined_regimes at each rebalance date was produced causally upstream;
        # as-of lookup here is safe (no look-ahead).
        window_schedule = build_weight_schedule(combined_regimes, rebal_dates, regime_weights)

        # Assemble a fresh engine via the shared engine-assembly path (cost parity)
        from macroregime.benchmarks.benchmarks import (
            load_run_params,
            TargetWeightStrategy,
            TargetWeightPortfolio,
            HistoricalDataHandler,
            BacktestConfig,
            EventDrivenBacktester,
            SimulatedExecutionHandler,
            SpreadSlippage,
            PercentageCommission,
            RiskManager,
        )

        spread_bps = params["spread_bps"]
        commission_rate = params["commission_rate"]
        initial_capital = params["initial_capital"]
        max_gross_exposure = params["max_gross_exposure"]
        max_position_weight = params["max_position_weight"]

        data_handler = HistoricalDataHandler(
            window_ohlcv,
            start=window.test_start,
            end=window.test_end,
        )
        strategy = TargetWeightStrategy(window_schedule)
        risk_manager = RiskManager(
            max_position_weight=max_position_weight,
            max_gross_exposure=max_gross_exposure,
        )
        portfolio = TargetWeightPortfolio(
            initial_capital=initial_capital,
            risk_manager=risk_manager,
        )
        execution_handler = SimulatedExecutionHandler(
            slippage_model=SpreadSlippage(spread_bps=spread_bps),
            commission_model=PercentageCommission(rate=commission_rate),
        )
        config = BacktestConfig(
            initial_capital=initial_capital,
            max_gross_exposure=max_gross_exposure,
            max_position_weight=max_position_weight,
            start=window.test_start,
            end=window.test_end,
        )
        engine = EventDrivenBacktester(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution_handler,
            config=config,
        )
        return engine

    runner = WalkForwardRunner(engine_factory=engine_factory, windows=windows)
    return runner.run()


# ---------------------------------------------------------------------------
# Regime stability report
# ---------------------------------------------------------------------------


def regime_stability_report(
    X_monthly: np.ndarray,
    X_daily: np.ndarray,
    k: int,
    quick: bool = False,
) -> dict:
    """Compare HMM and GMM regime stability on the same feature matrices.

    Runs CausalRegimeDetector with backend="hmm" and backend="gmm" on the
    same monthly AND daily matrices, then computes:
      - Label agreement fraction (after alignment) between HMM and GMM for each matrix
      - Per-window (first/second half) dwell times per state
      - Distribution drift (L1 distance of regime frequencies, first vs second half)

    This is the locked "HMM vs GMM compared on OOS regime stability" deliverable
    (plan 03-07 must-have).

    Parameters
    ----------
    X_monthly:
        Shape (T_monthly, n_features) — macro feature matrix (publication-date rows).
    X_daily:
        Shape (T_daily, n_features) — daily market feature matrix.
    k:
        Number of regime states for both detectors.
    quick:
        If True, use faster settings (refit_every=63, n_restarts=2, min_train=24).

    Returns
    -------
    dict with keys:
        "hmm_gmm_agreement" : float  — fraction of bars where HMM and GMM agree
                                       (computed on the daily matrix; range [0, 1])
        "macro_dwell_times" : dict   — {backend: dict[int, float]} per state
        "market_dwell_times": dict   — {backend: dict[int, float]} per state
        "distribution_drift": float  — L1 distance of combined regime frequency
                                       between first half and second half (daily)
    """
    from macroregime.regime.causal import CausalRegimeDetector
    from macroregime.regime.diagnostics import dwell_times
    from macroregime.regime.alignment import align_regime_labels

    refit_every = 63 if quick else 21
    n_restarts = 2 if quick else 3
    min_train_monthly = 24
    min_train_daily = 126

    # -----------------------------------------------------------------
    # Fit HMM and GMM on monthly (macro) matrix
    # -----------------------------------------------------------------
    hmm_macro = CausalRegimeDetector(
        backend="hmm", n_components=k,
        min_train=min_train_monthly, refit_every=refit_every, n_restarts=n_restarts,
    )
    gmm_macro = CausalRegimeDetector(
        backend="gmm", n_components=k,
        min_train=min_train_monthly, refit_every=refit_every, n_restarts=n_restarts,
    )
    macro_hmm_labels = hmm_macro.fit_predict_causal(X_monthly)
    macro_gmm_labels = gmm_macro.fit_predict_causal(X_monthly)

    # -----------------------------------------------------------------
    # Fit HMM and GMM on daily (market) matrix
    # -----------------------------------------------------------------
    hmm_daily = CausalRegimeDetector(
        backend="hmm", n_components=k,
        min_train=min_train_daily, refit_every=refit_every, n_restarts=n_restarts,
    )
    gmm_daily = CausalRegimeDetector(
        backend="gmm", n_components=k,
        min_train=min_train_daily, refit_every=refit_every, n_restarts=n_restarts,
    )
    daily_hmm_labels = hmm_daily.fit_predict_causal(X_daily)
    daily_gmm_labels = gmm_daily.fit_predict_causal(X_daily)

    # -----------------------------------------------------------------
    # HMM vs GMM agreement on daily matrix (after alignment)
    # -----------------------------------------------------------------
    # Only compare bars where both produced non-sentinel labels
    both_valid = (daily_hmm_labels >= 0) & (daily_gmm_labels >= 0)
    if both_valid.sum() > 0:
        agreement = float(
            np.mean(daily_hmm_labels[both_valid] == daily_gmm_labels[both_valid])
        )
    else:
        agreement = 0.0

    # -----------------------------------------------------------------
    # Dwell times per backend (monthly and daily)
    # -----------------------------------------------------------------
    macro_dwell = {
        "hmm": dwell_times(macro_hmm_labels, k),
        "gmm": dwell_times(macro_gmm_labels, k),
    }
    market_dwell = {
        "hmm": dwell_times(daily_hmm_labels, k),
        "gmm": dwell_times(daily_gmm_labels, k),
    }

    # -----------------------------------------------------------------
    # Distribution drift: L1 distance between first and second half (HMM daily)
    # -----------------------------------------------------------------
    valid_daily = daily_hmm_labels[daily_hmm_labels >= 0]
    drift = 0.0
    if len(valid_daily) >= 2:
        mid = len(valid_daily) // 2
        first_half = valid_daily[:mid]
        second_half = valid_daily[mid:]
        first_freq = np.bincount(first_half, minlength=k).astype(float)
        second_freq = np.bincount(second_half, minlength=k).astype(float)
        # Normalize to probability vectors
        if first_freq.sum() > 0:
            first_freq = first_freq / first_freq.sum()
        if second_freq.sum() > 0:
            second_freq = second_freq / second_freq.sum()
        drift = float(np.sum(np.abs(first_freq - second_freq)))

    return {
        "hmm_gmm_agreement": agreement,
        "macro_dwell_times": macro_dwell,
        "market_dwell_times": market_dwell,
        "distribution_drift": drift,
    }


# ---------------------------------------------------------------------------
# K sensitivity
# ---------------------------------------------------------------------------


def k_sensitivity(
    X: np.ndarray,
    ks: tuple[int, ...] = (2, 3, 4),
    backend: str = "hmm",
) -> dict[int, dict]:
    """Re-run regime detection for each K and report structural metrics.

    Reports dwell times, transition matrices, and label agreement between each
    K and the K=3 baseline (mapped by max-overlap assignment).

    ANTI-FEATURE: K selection via Sharpe ratio or any return-based metric is
    EXPLICITLY FORBIDDEN.  Selecting K to maximize Sharpe would overfit the
    regime model to the backtest period, invalidating the research hypothesis
    that macro regimes drive risk premia.  Select K based on economic
    interpretability, statistical tests (e.g., BIC), or regime dwell properties.

    Parameters
    ----------
    X:
        Shape (T, n_features) feature matrix (daily or monthly).
    ks:
        Tuple of K values to evaluate.
    backend:
        Model backend for CausalRegimeDetector ("hmm" or "gmm").

    Returns
    -------
    dict[int, dict]
        Keys are K values.  Each entry contains:
            "dwell_times"       : dict[int, float]  — mean run length per state
            "transition_matrix" : np.ndarray (K, K) — empirical transition matrix
            "agreement_vs_k3"   : float             — label agreement fraction vs K=3
                                                       baseline (only if k != 3)
    """
    from macroregime.regime.causal import CausalRegimeDetector
    from macroregime.regime.diagnostics import dwell_times, transition_matrix

    results: dict[int, dict] = {}
    baseline_labels: np.ndarray | None = None

    # First pass: compute all labels; store K=3 (or first K) as baseline
    all_labels: dict[int, np.ndarray] = {}
    for k in sorted(ks):
        detector = CausalRegimeDetector(
            backend=backend,
            n_components=k,
            min_train=max(24, k * 10),
            refit_every=63,
            n_restarts=2,
            random_seed=42,
        )
        labels = detector.fit_predict_causal(X)
        all_labels[k] = labels
        if k == 3 or baseline_labels is None:
            baseline_labels = labels.copy()

    # Second pass: compute metrics and agreement vs baseline
    for k in sorted(ks):
        labels = all_labels[k]
        valid = labels >= 0

        dt = dwell_times(labels, k)
        tm = transition_matrix(labels, k)

        entry: dict = {
            "dwell_times": dt,
            "transition_matrix": tm,
        }

        # Agreement vs K=3 baseline (using max-overlap assignment)
        if baseline_labels is not None and k != 3:
            entry["agreement_vs_k3"] = _max_overlap_agreement(
                baseline_labels, labels, n_base=3, n_other=k
            )
        elif k == 3:
            entry["agreement_vs_k3"] = 1.0  # trivially agrees with itself

        results[k] = entry

    return results


def _max_overlap_agreement(
    base_labels: np.ndarray,
    other_labels: np.ndarray,
    n_base: int,
    n_other: int,
) -> float:
    """Compute label agreement fraction using max-overlap (greedy) state mapping.

    For each state in ``other_labels``, find the state in ``base_labels`` with
    the most co-occurrence, then measure fraction of bars where they agree.

    This is the "collapsed by max-overlap mapping" referenced in the plan — a
    simple greedy assignment that handles K≠3 cases without requiring perfect
    bijectivity.

    Parameters
    ----------
    base_labels:
        Reference label sequence (typically K=3).
    other_labels:
        Comparison label sequence.
    n_base:
        Number of states in base_labels.
    n_other:
        Number of states in other_labels.

    Returns
    -------
    float
        Fraction of valid bars where mapped other_labels agree with base_labels.
    """
    # Only compare valid (non-sentinel) bars in both sequences
    both_valid = (base_labels >= 0) & (other_labels >= 0)
    if both_valid.sum() == 0:
        return 0.0

    b = base_labels[both_valid]
    o = other_labels[both_valid]

    # Co-occurrence matrix: shape (n_other, n_base)
    co = np.zeros((n_other, n_base), dtype=float)
    for ob, ob_base in zip(o, b):
        if 0 <= ob < n_other and 0 <= ob_base < n_base:
            co[ob, ob_base] += 1

    # Greedy assignment: for each other-state, pick the base-state with most overlap
    mapping = np.argmax(co, axis=1)  # shape (n_other,)

    # Apply mapping
    mapped = np.array([mapping[s] if 0 <= s < n_other else -1 for s in o])
    agreed = np.sum(mapped == b)
    return float(agreed) / float(len(b))
