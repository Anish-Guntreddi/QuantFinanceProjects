"""Tests for VSL-08: End-to-end pipeline integration.

Requirements:
- Pipeline runs end-to-end without exception on synthetic data (quick=True)
- Surface figures generated without exception (Agg backend)
- Report written to output directory
- All pipeline components produce non-empty results

Plan: 04-08 (Wave 5)
"""

import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-08")
def test_pipeline_end_to_end():
    """VolSurfacePipeline runs without exception on synthetic data with quick=True."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-08")
def test_report_figures():
    """ReportBuilder generates surface figures without exception (Agg backend)."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-08")
def test_pipeline_results_completeness():
    """PipelineResults contains non-empty calibration, forecast, and strategy results."""
    ...
