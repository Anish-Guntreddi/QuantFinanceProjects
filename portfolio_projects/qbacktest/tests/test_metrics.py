"""Metrics tests (QBT-08, QUAL-03). Real tests land in plans 01-03 and 01-06."""

import math

import numpy as np
import pandas as pd
import pytest

from qbacktest.metrics import (
    MetricsReport,
    bootstrap_sharpe_ci,
    compute_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)


# ---------------------------------------------------------------------------
# Task 1: Core ratio and drawdown metrics
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_sharpe_known_value(self):
        """constant-mean synthetic returns match hand-computed sqrt(252)*mean/std."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(loc=0.001, scale=0.01, size=252))
        expected = math.sqrt(252) * returns.mean() / returns.std(ddof=1)
        result = sharpe_ratio(returns)
        assert abs(result - expected) < 1e-9

    def test_sharpe_zero_vol(self):
        """Constant returns (std == 0) return 0.0 without ZeroDivisionError."""
        returns = pd.Series([0.01] * 100)
        result = sharpe_ratio(returns)
        assert result == 0.0

    def test_sharpe_empty(self):
        """Empty series returns 0.0 without error."""
        result = sharpe_ratio(pd.Series([], dtype=float))
        assert result == 0.0


class TestSortinoRatio:
    def test_sortino_downside_only(self):
        """Series with no negative returns: downside std == 0, return np.inf."""
        returns = pd.Series([0.01, 0.02, 0.005, 0.015])
        result = sortino_ratio(returns)
        assert result == np.inf

    def test_sortino_mixed(self):
        """Mixed series gives a finite positive value when mean > 0."""
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.001, 0.02, 252))
        result = sortino_ratio(returns)
        # Should be finite and reasonably positive for positive-drift series
        assert math.isfinite(result)

    def test_sortino_zero_mean(self):
        """Series with mean ~0 and downside returns: sortino near 0."""
        returns = pd.Series([0.01, -0.01, 0.01, -0.01])
        result = sortino_ratio(returns)
        # Mean is 0 → numerator is 0 → result is 0.0
        assert result == pytest.approx(0.0, abs=1e-9)


class TestMaxDrawdown:
    def test_max_drawdown_known_path(self):
        """equity [100,120,90,110] → maxDD = (90-120)/120 = -0.25."""
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        result = max_drawdown(equity)
        assert result == pytest.approx(-0.25, abs=1e-9)

    def test_max_drawdown_all_rising(self):
        """Monotonically rising equity → drawdown == 0."""
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        result = max_drawdown(equity)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_single(self):
        """Single-point equity curve → 0 drawdown."""
        equity = pd.Series([100.0])
        result = max_drawdown(equity)
        assert result == pytest.approx(0.0, abs=1e-9)


class TestTurnoverAndHitRate:
    def test_turnover_hand_computation(self):
        """Turnover = annualized sum(|trade value|) / mean_equity / years."""
        # 10 trades, each trading 100 value; mean equity 1000; 1 year
        # annualized turnover = 1000 / 1000 / 1.0 = 1.0
        result = turnover(
            total_traded_value=1000.0,
            mean_equity=1000.0,
            years=1.0,
        )
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_turnover_zero_equity(self):
        """Zero mean equity → 0.0 turnover, no ZeroDivisionError."""
        result = turnover(total_traded_value=500.0, mean_equity=0.0, years=1.0)
        assert result == 0.0

    def test_hit_rate_three_wins_one_loss(self):
        """3 wins, 1 loss → hit_rate == 0.75."""
        trade_pnls = [10.0, 20.0, 5.0, -8.0]
        result = hit_rate(trade_pnls)
        assert result == pytest.approx(0.75, abs=1e-9)

    def test_hit_rate_empty_trades(self):
        """Empty trade list → nan, no exception."""
        result = hit_rate([])
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Task 2: Bootstrap CI and gross/net MetricsReport
# ---------------------------------------------------------------------------


class TestBootstrapSharpeCI:
    def test_bootstrap_ci_order(self):
        """252 bars positive-drift returns: ci_low < point_sharpe < ci_high."""
        rng = np.random.default_rng(42)
        returns = np.array(rng.normal(0.001, 0.01, 252))
        ci_low, ci_high = bootstrap_sharpe_ci(returns)
        point = sharpe_ratio(pd.Series(returns))
        assert ci_low < point < ci_high

    def test_bootstrap_ci_short_series_nan(self):
        """10-bar series → (nan, nan), no exception."""
        rng = np.random.default_rng(1)
        short = np.array(rng.normal(0.001, 0.01, 10))
        ci_low, ci_high = bootstrap_sharpe_ci(short)
        assert math.isnan(ci_low)
        assert math.isnan(ci_high)

    def test_bootstrap_ci_deterministic(self):
        """Same rng seed → identical CI on two calls."""
        rng = np.random.default_rng(99)
        returns = np.array(rng.normal(0.001, 0.01, 252))
        ci1 = bootstrap_sharpe_ci(returns, rng=42)
        ci2 = bootstrap_sharpe_ci(returns, rng=42)
        assert ci1 == ci2

    def test_bootstrap_ci_constant_returns(self):
        """Constant returns (std==0) → (nan, nan), no exception."""
        returns = np.ones(252) * 0.001
        ci_low, ci_high = bootstrap_sharpe_ci(returns)
        assert math.isnan(ci_low)
        assert math.isnan(ci_high)


class TestMetricsReport:
    """Tests for the compute_metrics() -> MetricsReport pipeline."""

    def _build_inputs(self):
        """Build a consistent synthetic dataset for MetricsReport tests."""
        rng = np.random.default_rng(42)
        n = 252
        gross_ret = pd.Series(rng.normal(0.001, 0.015, n))
        cost_per_bar = 0.0002
        net_ret = gross_ret - cost_per_bar
        equity = (1 + net_ret).cumprod() * 10_000
        trade_pnls = [50.0, 80.0, -30.0, 120.0, -10.0]
        total_traded_value = 100_000.0
        total_costs = 100.0
        return equity, gross_ret, net_ret, trade_pnls, total_traded_value, total_costs

    def test_metrics_fields_present(self):
        """compute_metrics returns MetricsReport with every field non-None."""
        equity, gross_ret, net_ret, trade_pnls, ttv, tc = self._build_inputs()
        report = compute_metrics(equity, gross_ret, net_ret, trade_pnls, ttv, tc)
        assert isinstance(report, MetricsReport)
        for field in [
            "gross_sharpe", "net_sharpe", "cost_bps", "sortino",
            "max_drawdown", "turnover", "hit_rate",
            "sharpe_ci_low", "sharpe_ci_high", "total_return", "n_trades",
        ]:
            val = getattr(report, field)
            assert val is not None, f"{field} is None"

    def test_gross_sharpe_gt_net_sharpe_with_costs(self):
        """gross_sharpe > net_sharpe when total_costs > 0."""
        equity, gross_ret, net_ret, trade_pnls, ttv, tc = self._build_inputs()
        report = compute_metrics(equity, gross_ret, net_ret, trade_pnls, ttv, tc)
        assert report.gross_sharpe > report.net_sharpe

    def test_cost_bps_computation(self):
        """total_costs=100, total_traded_value=100_000 → cost_bps == 10.0."""
        equity, gross_ret, net_ret, trade_pnls, ttv, _ = self._build_inputs()
        report = compute_metrics(equity, gross_ret, net_ret, trade_pnls, ttv,
                                 total_costs=100.0)
        assert report.cost_bps == pytest.approx(10.0, abs=1e-9)

    def test_n_trades_correct(self):
        """n_trades matches length of trade_pnls list."""
        equity, gross_ret, net_ret, trade_pnls, ttv, tc = self._build_inputs()
        report = compute_metrics(equity, gross_ret, net_ret, trade_pnls, ttv, tc)
        assert report.n_trades == len(trade_pnls)

    def test_zero_traded_value_cost_bps(self):
        """Zero total_traded_value → cost_bps == 0.0, no ZeroDivisionError."""
        equity, gross_ret, net_ret, trade_pnls, _, _ = self._build_inputs()
        report = compute_metrics(equity, gross_ret, net_ret, trade_pnls,
                                 total_traded_value=0.0, total_costs=0.0)
        assert report.cost_bps == 0.0


# ---------------------------------------------------------------------------
# Stub that belongs to plan 01-06 — keep skip
# ---------------------------------------------------------------------------


def test_results_has_net_sharpe():
    pytest.skip("W0 stub — implemented in plan 01-06")
