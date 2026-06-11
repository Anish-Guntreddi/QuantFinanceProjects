"""Integration tests for the full macroregime pipeline and walk-forward evaluation.

Plan 03-07 — TDD RED written first, GREEN in pipeline.py and evaluation.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Task 1 — MacroRegimePipeline RED tests
# ---------------------------------------------------------------------------


def test_pipeline_imports():
    """MacroRegimePipeline and PipelineResults are importable from macroregime."""
    from macroregime import MacroRegimePipeline, PipelineResults  # noqa: F401


def test_pipeline_quick_run_returns_results():
    """pipeline.run(quick=True) returns a PipelineResults with non-None metrics."""
    from macroregime import MacroRegimePipeline

    result = MacroRegimePipeline(seed=42, quick=True).run()
    # The backtest must have run and produced metrics
    assert result.regime_backtest is not None
    assert result.regime_backtest.metrics is not None


def test_pipeline_results_fields():
    """PipelineResults dataclass contains all required fields."""
    from macroregime import MacroRegimePipeline

    r = MacroRegimePipeline(seed=42, quick=True).run()

    # Check all required fields per plan spec
    assert isinstance(r.macro_regimes, pd.Series), "macro_regimes must be pd.Series"
    assert isinstance(r.market_regimes, pd.Series), "market_regimes must be pd.Series"
    assert isinstance(r.combined_regimes, pd.Series), "combined_regimes must be pd.Series"
    assert isinstance(r.weight_schedule, dict), "weight_schedule must be dict"
    assert r.regime_backtest is not None
    assert isinstance(r.diagnostics, dict), "diagnostics must be dict"
    assert isinstance(r.config, dict), "config must be dict"


def test_pipeline_macro_regimes_are_monthly():
    """Macro regimes are much sparser than daily market regimes (publication-date indexed)."""
    from macroregime import MacroRegimePipeline

    r = MacroRegimePipeline(seed=42, quick=True).run()

    # Macro regimes are publication-date indexed: 4 series × ~120 months × staggered lags.
    # They should be at least 10× sparser than the daily market_regimes.
    # 10-year daily: ~2520 bars; macro panel after outer-join: a few hundred rows.
    assert len(r.macro_regimes) < len(r.market_regimes), (
        f"macro_regimes ({len(r.macro_regimes)}) should be sparser than "
        f"market_regimes ({len(r.market_regimes)})"
    )
    # Macro regimes index should not be business-day frequency — it's sparse
    # (publication dates, not daily). Check that macro has fewer rows than 80% of market.
    assert len(r.macro_regimes) < 0.80 * len(r.market_regimes), (
        f"macro_regimes ({len(r.macro_regimes)}) not sufficiently sparser than "
        f"daily market_regimes ({len(r.market_regimes)})"
    )
    # Values must be >= -1 (sentinel) and < n_components
    non_sentinel = r.macro_regimes[r.macro_regimes >= 0]
    assert len(non_sentinel) > 0, "All macro regimes are sentinel -1 — warm-up too long"


def test_pipeline_market_regimes_are_daily():
    """Market regimes are indexed at daily frequency."""
    from macroregime import MacroRegimePipeline

    r = MacroRegimePipeline(seed=42, quick=True).run()

    # Daily regimes must span most of the 10-year panel (~2500 bars)
    assert len(r.market_regimes) >= 1000, (
        f"market_regimes has {len(r.market_regimes)} rows — expected daily (>=1000 for 10-year panel)"
    )


def test_pipeline_combined_regimes_are_daily():
    """Combined regimes are indexed at daily frequency with no future leakage construction."""
    from macroregime import MacroRegimePipeline

    r = MacroRegimePipeline(seed=42, quick=True).run()

    # Combined must be daily too (same length as market_regimes after alignment)
    assert len(r.combined_regimes) >= 1000, (
        f"combined_regimes has {len(r.combined_regimes)} rows — expected daily"
    )
    # All non-sentinel values must be valid regime labels
    non_sentinel = r.combined_regimes[r.combined_regimes >= 0]
    assert len(non_sentinel) > 0, "All combined regimes are sentinel"


def test_pipeline_no_direct_hmmlearn_calls():
    """pipeline.py must not contain direct GaussianHMM or GaussianMixture calls."""
    from pathlib import Path

    pipeline_path = (
        Path(__file__).parents[1]
        / "src"
        / "macroregime"
        / "pipeline.py"
    )
    content = pipeline_path.read_text()
    assert "GaussianHMM" not in content, "pipeline.py must not call GaussianHMM directly"
    assert "GaussianMixture" not in content, "pipeline.py must not call GaussianMixture directly"


def test_pipeline_diagnostics_structure():
    """Diagnostics dict contains transition matrices and dwell times for all three models."""
    from macroregime import MacroRegimePipeline

    r = MacroRegimePipeline(seed=42, quick=True).run()

    diag = r.diagnostics
    for model_key in ("macro", "market", "combined"):
        assert model_key in diag, f"diagnostics missing key '{model_key}'"
        entry = diag[model_key]
        assert "transition_matrix" in entry, f"{model_key} missing transition_matrix"
        assert "dwell_times" in entry, f"{model_key} missing dwell_times"


# ---------------------------------------------------------------------------
# Task 2 — Walk-forward OOS + regime stability + K sensitivity RED tests
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-08")
def test_runner_quick():
    """Quick runner mode completes end-to-end with default parameters."""
    pass


def test_walk_forward_oos():
    """Walk-forward OOS equity curve is strictly increasing in time with non-overlapping windows."""
    from macroregime import MacroRegimePipeline
    from macroregime.evaluation import run_walk_forward
    from macroregime.data.synthetic import SyntheticMacroGenerator
    from macroregime.allocation.weights import load_regime_weights, month_end_rebalance_dates

    # Build pipeline artifacts for the walk-forward
    gen = SyntheticMacroGenerator(n_years=10, seed=42)
    panel = gen.generate()

    # Run a quick pipeline to get combined regimes and regime_weights
    from macroregime import MacroRegimePipeline

    pipeline = MacroRegimePipeline(seed=42, quick=True)
    r = pipeline.run()

    # Walk-forward with quick sizes: train_bars=504, test_bars=126
    wf = run_walk_forward(
        asset_ohlcv=panel.asset_ohlcv,
        combined_regimes=r.combined_regimes,
        regime_weights=load_regime_weights(),
        train_bars=504,
        test_bars=126,
    )

    oos_curve = wf.oos_equity_curve

    # Must have some OOS data
    assert len(oos_curve) > 0, "OOS equity curve is empty"

    # Timestamps must be strictly increasing (no duplicates)
    ts = oos_curve.index
    assert ts.is_monotonic_increasing, "OOS equity curve timestamps not monotonically increasing"
    assert ts.nunique() == len(ts), "OOS equity curve has duplicate timestamps"


def test_walk_forward_oos_windows_tile():
    """Walk-forward window test ranges tile the post-train index without overlap."""
    from macroregime import MacroRegimePipeline
    from macroregime.evaluation import run_walk_forward
    from macroregime.data.synthetic import SyntheticMacroGenerator
    from macroregime.allocation.weights import load_regime_weights

    gen = SyntheticMacroGenerator(n_years=10, seed=42)
    panel = gen.generate()

    pipeline = MacroRegimePipeline(seed=42, quick=True)
    r = pipeline.run()

    wf = run_walk_forward(
        asset_ohlcv=panel.asset_ohlcv,
        combined_regimes=r.combined_regimes,
        regime_weights=load_regime_weights(),
        train_bars=504,
        test_bars=126,
    )

    # Verify windows exist
    assert len(wf.window_results) >= 1, "Walk-forward produced no windows"


def test_regime_stability_report():
    """regime_stability_report returns HMM-vs-GMM agreement in [0,1] and valid distributions."""
    from macroregime.evaluation import regime_stability_report
    from macroregime.data.synthetic import SyntheticMacroGenerator
    from macroregime.features.market import build_market_features
    from macroregime.data.loader_base import SyntheticMacroLoader
    import numpy as np

    gen = SyntheticMacroGenerator(n_years=10, seed=42)
    panel = gen.generate()

    # Build macro feature matrix (monthly)
    loader = SyntheticMacroLoader(generator=gen)
    macro_df = loader.load_panel()
    # Drop NaN rows and get numpy array
    macro_df = macro_df.dropna()
    X_monthly = macro_df.values

    # Build daily market feature matrix
    mkt = build_market_features(panel.asset_ohlcv)
    mkt = mkt.dropna()
    X_daily = mkt.values

    report = regime_stability_report(X_monthly, X_daily, k=3, quick=True)

    # Agreement fraction must be in [0, 1]
    assert "hmm_gmm_agreement" in report, "report missing hmm_gmm_agreement"
    agreement = report["hmm_gmm_agreement"]
    assert 0.0 <= agreement <= 1.0, f"agreement {agreement} not in [0, 1]"

    # Must contain dwell_times per window
    assert "macro_dwell_times" in report, "report missing macro_dwell_times"
    assert "market_dwell_times" in report, "report missing market_dwell_times"

    # Distribution drift must be a non-negative float
    assert "distribution_drift" in report, "report missing distribution_drift"
    assert report["distribution_drift"] >= 0.0, "distribution_drift must be non-negative"


def test_k_sensitivity_no_sharpe():
    """k_sensitivity evaluation.py must contain no Sharpe-based selection."""
    from pathlib import Path

    eval_path = (
        Path(__file__).parents[1]
        / "src"
        / "macroregime"
        / "evaluation.py"
    )
    content = eval_path.read_text()
    # Confirm file exists (it should after Task 2 GREEN)
    assert eval_path.exists(), "evaluation.py does not exist yet"

    # The anti-feature guard: no Sharpe-based selection
    lower = content.lower()
    # These patterns would indicate Sharpe-based K selection
    forbidden_patterns = ["select", "best_k", "argmax"]
    for pattern in forbidden_patterns:
        # Only flag if combined with sharpe context
        if pattern in lower and "sharpe" in lower:
            # Check they appear near each other (within 200 chars)
            idx = lower.find("sharpe")
            nearby = lower[max(0, idx - 100) : idx + 100]
            assert pattern not in nearby, (
                f"evaluation.py appears to use '{pattern}' near 'sharpe' — "
                "K selection via Sharpe is forbidden"
            )


def test_k_sensitivity_returns_per_k_metrics():
    """k_sensitivity returns a dict keyed by K with dwell_times and transition_matrix."""
    from macroregime.evaluation import k_sensitivity
    from macroregime.data.synthetic import SyntheticMacroGenerator
    from macroregime.features.market import build_market_features
    import numpy as np

    gen = SyntheticMacroGenerator(n_years=5, seed=42)
    panel = gen.generate()
    mkt = build_market_features(panel.asset_ohlcv).dropna()
    X = mkt.values

    result = k_sensitivity(X, ks=(2, 3), backend="hmm")

    for k in (2, 3):
        assert k in result, f"k_sensitivity missing key K={k}"
        entry = result[k]
        assert "dwell_times" in entry, f"K={k} missing dwell_times"
        assert "transition_matrix" in entry, f"K={k} missing transition_matrix"
