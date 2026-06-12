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
            ("arrival_rate", "arrival_rate", lambda v: min(0.95, float(v) / 200.0)),  # normalize [10,200] → [0.05,0.95]
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
    # intraday_07/08: lookback_period → ma_slow; ma_fast = lookback // 5
    "intraday_07_ml_strategy": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "MA Crossover", "symbol": "SPY",
            "start": "2022-01-01", "end": "2024-12-31",
            "ma_fast": max(2, int(p.get("lookback_period", 60)) // 5),
            "ma_slow": int(p.get("lookback_period", 60)),
        },
    },
    "intraday_08_regime_detection": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "MA Crossover", "symbol": "SPY",
            "start": "2022-01-01", "end": "2024-12-31",
            "ma_fast": max(5, int(p.get("lookback_period", 252)) // 8),
            "ma_slow": int(p.get("lookback_period", 252)),
        },
    },
    "intraday_09_portfolio_construction": _bt("Momentum"),

    # ═══════════════════════════════════════════════════════════════════
    # AI/ML trading → backtest_runner (ML proxied as momentum/MR)
    # ═══════════════════════════════════════════════════════════════════
    # ml_01: lookback_days → ma_slow (regime window), ma_fast stays 20
    "ml_01_regime_detection": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "MA Crossover",
            "symbol": "SPY",
            "start": "2020-01-01",
            "end": "2024-12-31",
            "ma_fast": max(5, int(p.get("lookback_days", 252)) // 10),
            "ma_slow": int(p.get("lookback_days", 252)),
        },
    },
    "ml_02_lstm_transformer": _bt("Momentum", start="2020-01-01"),
    "ml_03_rl_market_making": _mm(),

    # ═══════════════════════════════════════════════════════════════════
    # Core research → backtest_runner
    # ═══════════════════════════════════════════════════════════════════
    "research_01_factor_toolkit": _bt("Momentum", start="2020-01-01"),
    # research_02: commission_bps → lookback (higher commission → longer holding period)
    "research_02_event_backtester": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "Momentum", "symbol": "SPY",
            "start": "2020-01-01", "end": "2024-12-31",
            "lookback": max(5, int(p.get("commission_bps", 10)) * 2),
        },
    },
    "research_03_stat_arb": _bt("Mean Reversion", start="2020-01-01"),
    # research_04: base_vol → entry_zscore (higher vol → wider entry threshold)
    "research_04_vol_surface": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "Mean Reversion", "symbol": "SPY",
            "start": "2020-01-01", "end": "2024-12-31",
            "lookback": 60,
            "entry_zscore": max(0.5, float(p.get("base_vol", 0.2)) * 10),
            "exit_zscore": max(0.1, float(p.get("base_vol", 0.2)) * 3),
        },
    },

    # ═══════════════════════════════════════════════════════════════════
    # Market microstructure engines → market_making_sim
    # ═══════════════════════════════════════════════════════════════════
    "engines_01_lob_simulator": _mm(),
    "engines_02_feed_handler": _mm(),

    # ═══════════════════════════════════════════════════════════════════
    # Market microstructure execution → backtest_runner / MM sim
    # ═══════════════════════════════════════════════════════════════════
    "exec_01_lob_simulator": _mm(),
    # exec_02: participation_rate → ma_fast (higher rate = faster execution = shorter window)
    "exec_02_execution_algos": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "MA Crossover",
            "symbol": "SPY",
            "start": "2020-01-01",
            "end": "2024-12-31",
            "ma_fast": max(2, int(p.get("participation_rate", 0.1) * 100)),
            "ma_slow": max(10, int(p.get("participation_rate", 0.1) * 500)),
        },
    },
    "exec_03_feed_handler": _mm(),

    # ═══════════════════════════════════════════════════════════════════
    # Risk engineering → backtest_runner
    # ═══════════════════════════════════════════════════════════════════
    # risk_01: max_weight → lookback (higher concentration → shorter momentum window)
    "risk_01_portfolio_optimization": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "Momentum", "symbol": "SPY",
            "start": "2020-01-01", "end": "2024-12-31",
            "lookback": max(5, int((1.0 - float(p.get("max_weight", 0.3))) * 120)),
        },
    },
    # risk_02: n_splits → lookback (more CV splits → longer evaluation window)
    "risk_02_reproducibility": {
        "engine": lambda **kw: _get_backtest_runner()(**kw),
        "translate": lambda p: {
            "strategy": "Momentum", "symbol": "SPY",
            "start": "2020-01-01", "end": "2024-12-31",
            "lookback": int(p.get("n_splits", 5)) * 10,
        },
    },
}
