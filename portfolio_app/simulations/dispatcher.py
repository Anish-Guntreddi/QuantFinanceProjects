"""Dispatch interactive parameter changes to the appropriate simulation engine.

Maps project IDs to simulation functions, translates UI params to engine kwargs,
and returns results in the standard format (metrics + equity_curve).
"""

from __future__ import annotations

from typing import Any


def can_simulate(project_id: str, tier: str) -> bool:
    """Return True if we can run a live simulation for this project."""
    if project_id in _DISPATCH_MAP:
        return True  # We have an engine for this project regardless of tier
    return False


def run_simulation(project_id: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Run the appropriate simulation for a project with given parameters.

    Returns a dict with 'metrics' and 'equity_curve' keys, or None on failure.
    """
    spec = _DISPATCH_MAP.get(project_id)
    if spec is None:
        return None

    engine = spec["engine"]
    kwargs = spec["translate"](params)

    try:
        result = engine(**kwargs)
    except Exception as e:
        return {"error": str(e)}

    # Normalize market-making sim output to standard format
    if "pnl_series" in result and "equity_curve" not in result:
        pnl = result["pnl_series"]
        result["equity_curve"] = {
            "dates": [f"T+{i}" for i in range(len(pnl))],
            "values": [1.0 + p / max(abs(pnl[-1]), 1) for p in pnl],
            "benchmark_values": [v / result["mid_prices"][0] for v in result["mid_prices"]],
        }

    return result


# ---------------------------------------------------------------------------
# Engine imports (lazy to avoid import-time overhead)
# ---------------------------------------------------------------------------

def _get_backtest_runner():
    from .backtest_runner import run_backtest
    return run_backtest


def _get_mm_sim():
    from .market_making_sim import run_market_making_sim
    return run_market_making_sim


# ---------------------------------------------------------------------------
# Helper: shorthand builders for common engine patterns
# ---------------------------------------------------------------------------

def _bt(strategy: str, symbol: str = "SPY", start: str = "2022-01-01",
        end: str = "2024-12-31", **fixed) -> dict[str, Any]:
    """Create a dispatch entry for backtest_runner with fixed base params."""
    def _translate(p: dict[str, Any]) -> dict[str, Any]:
        kw = {"strategy": strategy, "symbol": symbol, "start": start, "end": end}
        kw.update(fixed)
        # Map any UI param that matches a backtest_runner kwarg
        for ui_name, engine_name in [
            ("lookback_period", "lookback"), ("lookback", "lookback"),
            ("lookback_days", "lookback"),
            ("entry_zscore", "entry_zscore"), ("exit_zscore", "exit_zscore"),
            ("ma_fast", "ma_fast"), ("ma_slow", "ma_slow"),
            ("holding_period", "holding"),
        ]:
            if ui_name in p:
                val = p[ui_name]
                kw[engine_name] = int(val) if engine_name in ("lookback", "holding", "ma_fast", "ma_slow") else float(val)
        return kw

    return {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": _translate,
    }


def _mm(**fixed) -> dict[str, Any]:
    """Create a dispatch entry for market_making_sim with fixed base params."""
    def _translate(p: dict[str, Any]) -> dict[str, Any]:
        kw = dict(fixed)
        # Map any UI param that matches a MM sim kwarg
        for ui_name, engine_name, cast in [
            ("risk_aversion", "risk_aversion", float),
            ("max_position", "max_position", int),
            ("base_spread_bps", "base_spread_bps", float),
            ("arrival_rate", "arrival_rate", float),
            ("volatility", "volatility", float),
            ("signal_threshold", "base_spread_bps", float),  # maps threshold → spread
            ("threshold_bps", "base_spread_bps", float),
            ("confidence_threshold", "arrival_rate", float),  # maps confidence → fill rate
            ("exchange_latency_ms", "volatility", lambda v: 0.02 + float(v) * 0.001),
            ("latency_advantage_us", "volatility", lambda v: 0.02 + float(v) * 0.0005),
            ("venue_count", "n_steps", lambda v: int(v) * 5000),
            ("num_regimes", "n_steps", lambda v: int(v) * 5000),
            ("inventory_penalty", "risk_aversion", float),
            ("hawkes_mu", "arrival_rate", lambda v: float(v) / 100.0),
            ("num_events", "n_steps", int),
            ("buffer_size", "n_steps", lambda v: int(v) * 2),
            ("queue_size", "n_steps", lambda v: int(float(v) / 6.5)),
        ]:
            if ui_name in p:
                val = p[ui_name]
                if callable(cast) and cast not in (int, float):
                    kw[engine_name] = cast(val)
                else:
                    kw[engine_name] = cast(val)
        return kw

    return {
        "engine": lambda **kw: _get_mm_sim()(**kw),
        "translate": _translate,
    }


# ---------------------------------------------------------------------------
# Project → engine mapping (all 32 projects with interactive_params)
# ---------------------------------------------------------------------------

_DISPATCH_MAP: dict[str, dict[str, Any]] = {
    # ═══════════════════════════════════════════════════════════════════
    # HFT strategies → market_making_sim
    # ═══════════════════════════════════════════════════════════════════
    "hft_01_adaptive_market_making": _mm(),
    "hft_02_order_book_scalping": _mm(),
    "hft_03_queue_position": _mm(),
    "hft_04_cross_exchange_arb": _mm(),
    "hft_05_trade_imbalance": _mm(),
    "hft_06_iceberg_detection": _mm(),
    "hft_07_latency_arb": _mm(),
    "hft_08_smart_order_router": _mm(),
    "hft_09_rl_market_maker": _mm(),

    # ═══════════════════════════════════════════════════════════════════
    # Intraday strategies → backtest_runner
    # ═══════════════════════════════════════════════════════════════════
    "intraday_01_momentum": _bt("Momentum"),
    "intraday_02_mean_reversion": _bt("Mean Reversion", entry_zscore=2.0, exit_zscore=0.5),
    "intraday_03_stat_arb": _bt("Mean Reversion", entry_zscore=2.0, exit_zscore=0.5),
    "intraday_04_momentum_value": _bt("Momentum"),
    "intraday_05_options": _bt("Momentum"),
    "intraday_06_execution_tca": _bt("Momentum"),
    "intraday_07_ml_strategy": _bt("MA Crossover", ma_fast=10, ma_slow=50),
    "intraday_08_regime_detection": _bt("MA Crossover", ma_fast=20, ma_slow=100),
    "intraday_09_portfolio_construction": _bt("Momentum"),

    # ═══════════════════════════════════════════════════════════════════
    # AI/ML trading → backtest_runner (ML proxied as momentum/MR)
    # ═══════════════════════════════════════════════════════════════════
    "ml_01_regime_detection": _bt("MA Crossover", start="2020-01-01", ma_fast=20, ma_slow=100),
    "ml_02_lstm_transformer": _bt("Momentum", start="2020-01-01"),
    "ml_03_rl_market_making": _mm(),

    # ═══════════════════════════════════════════════════════════════════
    # Core research → backtest_runner
    # ═══════════════════════════════════════════════════════════════════
    "research_01_factor_toolkit": _bt("Momentum", start="2020-01-01"),
    "research_02_event_backtester": _bt("Momentum", start="2020-01-01"),
    "research_03_stat_arb": _bt("Mean Reversion", start="2020-01-01"),
    "research_04_vol_surface": _bt("Mean Reversion", start="2020-01-01", entry_zscore=2.0, exit_zscore=0.5),

    # ═══════════════════════════════════════════════════════════════════
    # Market microstructure engines → market_making_sim
    # ═══════════════════════════════════════════════════════════════════
    "engines_01_lob_simulator": _mm(),
    "engines_02_feed_handler": _mm(),

    # ═══════════════════════════════════════════════════════════════════
    # Market microstructure execution → backtest_runner / MM sim
    # ═══════════════════════════════════════════════════════════════════
    "exec_01_lob_simulator": _mm(),
    "exec_02_execution_algos": _bt("MA Crossover", start="2020-01-01", ma_fast=10, ma_slow=50),
    "exec_03_feed_handler": _mm(),

    # ═══════════════════════════════════════════════════════════════════
    # Risk engineering → backtest_runner
    # ═══════════════════════════════════════════════════════════════════
    "risk_01_portfolio_optimization": _bt("Momentum", start="2020-01-01"),
    "risk_02_reproducibility": _bt("Momentum", start="2020-01-01"),
}
