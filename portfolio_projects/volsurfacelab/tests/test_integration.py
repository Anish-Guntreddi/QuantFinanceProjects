"""Integration tests for plan 04-07: one-command runner (VSL-08).

Tests call main() in-process (no subprocess) to keep CI fast and offline.
run_pipeline.py lives at portfolio_projects/volsurfacelab/run_pipeline.py
(project root, one level above tests/).  It is imported via importlib to
avoid sys.path hacks, following the locked macroregime test pattern.

Tests
-----
test_runner_quick
    main(["--quick", "--output-dir", str(figures_dir)]) returns 0; expected
    PNG files exist; summary.md mentions QLIKE and Greeks.

test_runner_seed_determinism
    Two runs with --seed 7 produce byte-identical summary.md net P&L lines.

test_runner_bad_args
    main(["--seed", "not-an-int"]) raises SystemExit with nonzero code (2).

test_no_network
    After a quick run, yfinance and fredapi are NOT in sys.modules
    (offline-safe guarantee).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared helper: import run_pipeline.main by file path (no sys.path hack)
# ---------------------------------------------------------------------------

def _load_main():
    """Load run_pipeline.main from the project root via importlib."""
    runner_path = Path(__file__).parents[1] / "run_pipeline.py"
    assert runner_path.exists(), (
        f"run_pipeline.py not found at {runner_path}. "
        "Implement it in the project root (portfolio_projects/volsurfacelab/)."
    )
    spec = importlib.util.spec_from_file_location("run_pipeline", runner_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


# ---------------------------------------------------------------------------
# test_runner_quick
# ---------------------------------------------------------------------------

def test_runner_quick(tmp_path):
    """main() returns 0; expected figures + summary.md are created."""
    main = _load_main()

    figures_dir = tmp_path / "figures"
    rc = main(["--quick", "--output-dir", str(figures_dir)])
    assert rc == 0, f"run_pipeline.main returned {rc}"

    # Required figures
    expected_pngs = [
        "surface_3d.png",
        "surface_heatmap.png",
        "vrp_pnl.png",
    ]
    for fname in expected_pngs:
        p = figures_dir / fname
        assert p.exists(), f"Expected figure not found: {p}"
        assert p.stat().st_size > 0, f"Figure is empty: {p}"

    # At least one smile plot
    smile_files = list(figures_dir.glob("smile_*.png"))
    assert len(smile_files) >= 1, "No smile_*.png figures found"

    # summary.md beside figures/ (i.e. in tmp_path)
    summary = tmp_path / "summary.md"
    assert summary.exists(), f"summary.md not found at {summary}"
    content = summary.read_text(encoding="utf-8")
    assert "QLIKE" in content, "summary.md does not mention QLIKE"
    assert "Greeks" in content, "summary.md does not mention Greeks"


# ---------------------------------------------------------------------------
# test_runner_seed_determinism
# ---------------------------------------------------------------------------

def test_runner_seed_determinism(tmp_path):
    """Two runs with the same seed produce identical summary.md net P&L lines."""
    main = _load_main()

    dir_a = tmp_path / "run_a" / "figures"
    dir_b = tmp_path / "run_b" / "figures"

    rc_a = main(["--quick", "--seed", "7", "--output-dir", str(dir_a)])
    rc_b = main(["--quick", "--seed", "7", "--output-dir", str(dir_b)])
    assert rc_a == 0 and rc_b == 0

    summary_a = (dir_a.parent / "summary.md").read_text(encoding="utf-8")
    summary_b = (dir_b.parent / "summary.md").read_text(encoding="utf-8")

    # Extract lines that mention Net P&L (the strategy dollar totals)
    def pnl_lines(text: str) -> list[str]:
        return [ln.strip() for ln in text.splitlines() if "Net total P&L" in ln or "Gross total P&L" in ln]

    lines_a = pnl_lines(summary_a)
    lines_b = pnl_lines(summary_b)

    assert lines_a, "No P&L lines found in summary_a"
    assert lines_a == lines_b, (
        f"P&L lines differ between runs with same seed:\n  run_a: {lines_a}\n  run_b: {lines_b}"
    )


# ---------------------------------------------------------------------------
# test_runner_bad_args
# ---------------------------------------------------------------------------

def test_runner_bad_args():
    """main(["--seed", "not-an-int"]) raises SystemExit with code != 0."""
    main = _load_main()

    with pytest.raises(SystemExit) as exc_info:
        main(["--seed", "not-an-int"])

    assert exc_info.value.code != 0, (
        f"Expected nonzero exit code for bad args, got {exc_info.value.code}"
    )


# ---------------------------------------------------------------------------
# test_no_network
# ---------------------------------------------------------------------------

def test_no_network(tmp_path):
    """Offline guarantee: yfinance and fredapi are not imported during a quick run."""
    # Remove from sys.modules if somehow already present from another test
    for mod in ("yfinance", "fredapi"):
        sys.modules.pop(mod, None)

    main = _load_main()

    figures_dir = tmp_path / "figures"
    rc = main(["--quick", "--output-dir", str(figures_dir)])
    assert rc == 0

    assert "yfinance" not in sys.modules, "yfinance was imported during quick run"
    assert "fredapi" not in sys.modules, "fredapi was imported during quick run"
