"""Tests for ReportBuilder — figures, summary.md, Agg backend.

Tests cover (per plan 04-06):
a) All expected PNGs exist and have size > 0
b) summary.md exists and contains "QLIKE", "Greeks", and "net" substrings
c) matplotlib backend is Agg after importing volsurfacelab.report
d) Excluded slices (if any) are named in summary.md
"""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Session-scope quick pipeline result fixture (shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quick_results():
    """Module-scope quick PipelineResults."""
    from volsurfacelab.pipeline import VolSurfacePipeline
    return VolSurfacePipeline(seed=42, quick=True).run()


@pytest.fixture(scope="module")
def report_paths(quick_results, tmp_path_factory):
    """Module-scope: build report into a tmp_path/figures dir, return paths dict."""
    from volsurfacelab.report import ReportBuilder
    figures_dir = tmp_path_factory.mktemp("figures")
    builder = ReportBuilder(quick_results, output_dir=figures_dir)
    return builder.build(), figures_dir


# ---------------------------------------------------------------------------
# test_agg_backend — must be first (backend is set at module import time)
# ---------------------------------------------------------------------------

def test_agg_backend_after_import():
    """matplotlib backend must be Agg after importing volsurfacelab.report."""
    import matplotlib
    import volsurfacelab.report  # noqa: F401 — import to trigger backend switch
    assert matplotlib.get_backend().lower() == "agg", (
        f"Expected Agg backend, got: {matplotlib.get_backend()}"
    )


# ---------------------------------------------------------------------------
# test_smile_figures_exist
# ---------------------------------------------------------------------------

def test_smile_figures_exist(quick_results, report_paths):
    """One smile_T{T}.png must exist and be non-empty per validated maturity."""
    paths, figures_dir = report_paths
    validated_maturities = sorted(quick_results.svi_fits.keys())

    assert len(validated_maturities) > 0, "Need at least one validated maturity"

    for T in validated_maturities:
        key = f"smile_T{T}"
        assert key in paths, f"Expected '{key}' in paths dict"
        fpath = paths[key]
        assert fpath.exists(), f"Smile figure {fpath.name} not found in {figures_dir}"
        assert fpath.stat().st_size > 0, f"Smile figure {fpath.name} is empty"


# ---------------------------------------------------------------------------
# test_surface_3d_exists
# ---------------------------------------------------------------------------

def test_surface_3d_exists(report_paths):
    """surface_3d.png must exist and be non-empty."""
    paths, figures_dir = report_paths
    assert "surface_3d" in paths, "Expected 'surface_3d' in paths dict"
    fpath = paths["surface_3d"]
    assert fpath.exists(), f"surface_3d.png not found in {figures_dir}"
    assert fpath.stat().st_size > 0, "surface_3d.png is empty"


# ---------------------------------------------------------------------------
# test_surface_heatmap_exists
# ---------------------------------------------------------------------------

def test_surface_heatmap_exists(report_paths):
    """surface_heatmap.png must exist and be non-empty."""
    paths, figures_dir = report_paths
    assert "surface_heatmap" in paths, "Expected 'surface_heatmap' in paths dict"
    fpath = paths["surface_heatmap"]
    assert fpath.exists(), f"surface_heatmap.png not found in {figures_dir}"
    assert fpath.stat().st_size > 0, "surface_heatmap.png is empty"


# ---------------------------------------------------------------------------
# test_vrp_pnl_exists
# ---------------------------------------------------------------------------

def test_vrp_pnl_exists(report_paths):
    """vrp_pnl.png must exist and be non-empty."""
    paths, figures_dir = report_paths
    assert "vrp_pnl" in paths, "Expected 'vrp_pnl' in paths dict"
    fpath = paths["vrp_pnl"]
    assert fpath.exists(), f"vrp_pnl.png not found in {figures_dir}"
    assert fpath.stat().st_size > 0, "vrp_pnl.png is empty"


# ---------------------------------------------------------------------------
# test_forecast_qlike_exists
# ---------------------------------------------------------------------------

def test_forecast_qlike_exists(report_paths):
    """forecast_qlike.png must exist and be non-empty."""
    paths, figures_dir = report_paths
    assert "forecast_qlike" in paths, "Expected 'forecast_qlike' in paths dict"
    fpath = paths["forecast_qlike"]
    assert fpath.exists(), f"forecast_qlike.png not found in {figures_dir}"
    assert fpath.stat().st_size > 0, "forecast_qlike.png is empty"


# ---------------------------------------------------------------------------
# test_summary_md_exists_with_required_content
# ---------------------------------------------------------------------------

def test_summary_md_exists_with_required_content(report_paths):
    """summary.md must exist beside figures dir and contain required substrings."""
    paths, figures_dir = report_paths
    assert "summary_md" in paths, "Expected 'summary_md' in paths dict"
    summary_path = paths["summary_md"]

    assert summary_path.exists(), (
        f"summary.md not found at {summary_path}. "
        "Must be written to output_dir.parent (beside figures/)."
    )
    assert summary_path.stat().st_size > 0, "summary.md is empty"

    # Verify location: must be in parent of figures dir
    assert summary_path.parent == figures_dir.parent, (
        f"summary.md at {summary_path.parent}, expected {figures_dir.parent}. "
        "Must be beside figures dir, not inside it."
    )

    content = summary_path.read_text(encoding="utf-8")
    for substr in ("QLIKE", "Greeks", "net"):
        assert substr in content, (
            f"Expected '{substr}' substring in summary.md, but not found.\n"
            f"summary.md content:\n{content[:500]}..."
        )


# ---------------------------------------------------------------------------
# test_excluded_slices_in_summary
# ---------------------------------------------------------------------------

def test_excluded_slices_in_summary():
    """Excluded slices (if any injected) must be named in summary.md."""
    from volsurfacelab.chain import make_calendar_violating_surface
    from volsurfacelab.pipeline import VolSurfacePipeline
    from volsurfacelab.report import ReportBuilder
    import tempfile

    violating_surface = make_calendar_violating_surface()
    pipeline = VolSurfacePipeline(seed=42, quick=True, svi_surface=violating_surface)
    results = pipeline.run()

    # Verify we have exclusions
    assert len(results.excluded_slices) > 0, (
        "Expected exclusions from calendar-violating surface"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        figures_dir = Path(tmpdir) / "figures"
        builder = ReportBuilder(results, output_dir=figures_dir)
        paths = builder.build()

        summary_path = paths["summary_md"]
        content = summary_path.read_text(encoding="utf-8")

        # Check each excluded maturity is mentioned in the summary
        for T, reason in results.excluded_slices:
            t_str = f"{T:.4f}"
            assert t_str in content, (
                f"Excluded maturity T={T} (reason: {reason}) not found in summary.md."
            )


# ---------------------------------------------------------------------------
# test_figures_all_non_empty (consolidated check)
# ---------------------------------------------------------------------------

def test_all_figures_non_empty(report_paths):
    """All paths returned by build() must point to non-empty files (comprehensive)."""
    paths, _ = report_paths
    for name, fpath in paths.items():
        if name == "summary_md":
            continue  # tested separately
        assert fpath.exists(), f"Figure '{name}' not found at {fpath}"
        assert fpath.stat().st_size > 0, f"Figure '{name}' is an empty file"


# ---------------------------------------------------------------------------
# test_no_open_figure_leaks
# ---------------------------------------------------------------------------

def test_no_open_figure_leaks(quick_results, tmp_path):
    """After build(), no open matplotlib figures should remain in memory."""
    import matplotlib.pyplot as plt
    from volsurfacelab.report import ReportBuilder

    figures_dir = tmp_path / "figures"
    builder = ReportBuilder(quick_results, output_dir=figures_dir)

    before = len(plt.get_fignums())
    builder.build()
    after = len(plt.get_fignums())

    # Each figure must be closed after savefig
    # Allow for pre-existing figures (before) — we only check no net increase
    assert after <= before, (
        f"Open figure leak: {after - before} figures still open after build() "
        f"(before={before}, after={after}). "
        "Call plt.close(fig) after each savefig."
    )
