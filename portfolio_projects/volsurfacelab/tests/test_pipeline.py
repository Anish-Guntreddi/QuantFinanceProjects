"""Tests for VolSurfacePipeline end-to-end assembly (plan 04-06).

Tests cover:
- Pipeline runs end-to-end (quick=True) returning PipelineResults without exception
- PipelineResults is frozen (FrozenInstanceError on assignment)
- IVs solved from PRICES via solve_chain_iv (not from true_iv directly)
- No-arb gate wired: planted calendar violation yields excluded_slices non-empty
- Forecast and strategy results attached to PipelineResults
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from volsurfacelab.chain import make_calendar_violating_surface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quick_results():
    """Module-scope PipelineResults from a quick (seed=42, quick=True) run."""
    from volsurfacelab.pipeline import VolSurfacePipeline
    return VolSurfacePipeline(seed=42, quick=True).run()


# ---------------------------------------------------------------------------
# test_pipeline_runs_end_to_end
# ---------------------------------------------------------------------------

def test_pipeline_runs_end_to_end(quick_results):
    """Pipeline runs end-to-end without exception and returns PipelineResults."""
    from volsurfacelab.pipeline import PipelineResults
    assert isinstance(quick_results, PipelineResults)


# ---------------------------------------------------------------------------
# test_results_frozen
# ---------------------------------------------------------------------------

def test_results_frozen(quick_results):
    """PipelineResults must raise FrozenInstanceError on field assignment."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        quick_results.seed = 99


# ---------------------------------------------------------------------------
# test_pipeline_uses_solved_iv
# ---------------------------------------------------------------------------

def test_pipeline_uses_solved_iv(quick_results):
    """iv_frame must have an 'iv' column with finite values (solved from prices).

    The honest-path discipline: pipeline calls solve_chain_iv on PRICES.
    true_iv is a test oracle only. This test verifies the 'iv' column is
    present and that it does not equal NaN for the majority of rows.
    """
    iv_frame = quick_results.iv_frame
    assert "iv" in iv_frame.columns, "iv_frame must have 'iv' column from solve_chain_iv"
    n_valid = iv_frame["iv"].notna().sum()
    assert n_valid > 0, "solve_chain_iv must recover at least some IVs"
    # For the standard synthetic chain, all 78 rows should round-trip cleanly
    pct_valid = n_valid / len(iv_frame)
    assert pct_valid >= 0.9, (
        f"Expected >= 90% valid IVs, got {pct_valid:.1%} ({n_valid}/{len(iv_frame)}). "
        "Pipeline may be using true_iv instead of solving from prices."
    )


def test_pipeline_iv_monkeypatch(monkeypatch):
    """Breaking solve_chain_iv must cause pipeline to fail or yield NaN IVs.

    Verifies the pipeline actually calls solve_chain_iv (honest path), not
    that it reads true_iv directly from the chain.
    """
    import volsurfacelab.pipeline as pipeline_mod
    from volsurfacelab.chain import ChainData

    def broken_solve(chain):
        """Return all-NaN iv_frame to simulate solve failure."""
        df = chain.options.copy()
        df["iv"] = float("nan")
        return df

    monkeypatch.setattr(pipeline_mod, "solve_chain_iv", broken_solve)

    from volsurfacelab.pipeline import VolSurfacePipeline
    # With all-NaN IVs the pipeline should raise RuntimeError (no valid slices)
    with pytest.raises(RuntimeError):
        VolSurfacePipeline(seed=42, quick=True).run()


# ---------------------------------------------------------------------------
# test_no_arb_gate_wired
# ---------------------------------------------------------------------------

def test_no_arb_gate_wired():
    """Planted calendar violation yields excluded_slices non-empty.

    Uses make_calendar_violating_surface to inject a calendar-violating
    surface into the pipeline via the svi_surface parameter. The gate should
    detect and exclude the violating maturity, leaving excluded_slices non-empty.
    """
    from volsurfacelab.pipeline import VolSurfacePipeline

    violating_surface = make_calendar_violating_surface()
    pipeline = VolSurfacePipeline(
        seed=42,
        quick=True,
        svi_surface=violating_surface,
    )
    results = pipeline.run()

    # At least one slice must have been excluded
    assert len(results.excluded_slices) > 0, (
        "Expected at least one excluded slice from calendar-violating surface, "
        f"got excluded_slices={results.excluded_slices}"
    )

    # Excluded maturity must not appear in svi_fits
    excluded_maturities = {item[0] for item in results.excluded_slices}
    validated_maturities = set(results.svi_fits.keys())
    overlap = excluded_maturities & validated_maturities
    assert len(overlap) == 0, (
        f"Excluded maturities {excluded_maturities} must not appear "
        f"in svi_fits {validated_maturities}"
    )


# ---------------------------------------------------------------------------
# test_forecast_and_strategy_attached
# ---------------------------------------------------------------------------

def test_forecast_and_strategy_attached(quick_results):
    """forecast.table has 3 model rows; vrp.net_pnl is non-empty; config_used is set."""
    from volsurfacelab.pipeline import PipelineResults
    from volsurfacelab.forecast import ForecastComparison
    from volsurfacelab.strategy import VRPResult

    # Forecast check
    assert isinstance(quick_results.forecast, ForecastComparison)
    assert len(quick_results.forecast.table) == 3, (
        f"Expected 3 model rows in forecast.table, got {len(quick_results.forecast.table)}"
    )
    expected_models = {"HAR", "GARCH", "EGARCH"}
    actual_models = set(quick_results.forecast.table.index)
    assert actual_models == expected_models, (
        f"Expected models {expected_models}, got {actual_models}"
    )

    # Strategy check
    assert isinstance(quick_results.vrp, VRPResult)
    assert len(quick_results.vrp.net_pnl) > 0, "vrp.net_pnl must be non-empty Series"
    assert isinstance(quick_results.vrp.net_pnl, pd.Series)

    # Config check
    config = quick_results.config_used
    assert "seed" in config, "config_used must carry 'seed'"
    assert "cost_rate" in config, "config_used must carry 'cost_rate'"
    assert quick_results.seed == 42


# ---------------------------------------------------------------------------
# test_svi_fits_populated
# ---------------------------------------------------------------------------

def test_svi_fits_populated(quick_results):
    """svi_fits must be a non-empty dict of SVISliceFit objects."""
    from volsurfacelab.svi import SVISliceFit
    assert len(quick_results.svi_fits) > 0, "svi_fits must be non-empty after gate"
    for T, fit in quick_results.svi_fits.items():
        assert isinstance(fit, SVISliceFit), f"svi_fits[{T}] must be SVISliceFit"
        assert fit.success, f"svi_fits[{T}].success must be True"


# ---------------------------------------------------------------------------
# test_quick_mode_completes_fast  (timing guard — ~30 seconds)
# ---------------------------------------------------------------------------

def test_quick_mode_completes_fast():
    """quick=True pipeline must complete in under 30 seconds."""
    import time
    from volsurfacelab.pipeline import VolSurfacePipeline
    t0 = time.monotonic()
    VolSurfacePipeline(seed=42, quick=True).run()
    elapsed = time.monotonic() - t0
    assert elapsed < 30.0, (
        f"quick=True pipeline took {elapsed:.1f}s (>30s limit). "
        "Reduce n_restarts or n_days in quick mode."
    )
