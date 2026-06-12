"""Integration tests for run_pipeline.py runner (plan 05-08).

Import pattern (locked Phase 3/4 convention): importlib.util.spec_from_file_location
No sys.path hacks. Runner lives at portfolio_projects/defiregimenet/run_pipeline.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Runner import via spec_from_file_location (locked pattern, no sys.path hacks)
# ---------------------------------------------------------------------------

_RUNNER_PATH = Path(__file__).parents[1] / "run_pipeline.py"


def _load_runner():
    """Load run_pipeline.py as a module without sys.path mutations."""
    spec = importlib.util.spec_from_file_location("defi_runner", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    """Return the runner module (loaded once per test session)."""
    return _load_runner()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_quick(tmp_path, runner):
    """Quick run exits 0, produces >= 6 PNGs, and writes summary.md."""
    figures_dir = tmp_path / "figures"
    rc = runner.main(["--quick", "--output-dir", str(figures_dir)])
    assert rc == 0, f"Runner returned non-zero exit code: {rc}"

    pngs = list(figures_dir.glob("*.png"))
    assert len(pngs) >= 6, (
        f"Expected >= 6 PNG figures in {figures_dir}, found {len(pngs)}: {[p.name for p in pngs]}"
    )

    summary = tmp_path / "summary.md"
    assert summary.exists(), f"summary.md not found at {summary}"
    text = summary.read_text(encoding="utf-8")
    assert len(text) > 100, "summary.md appears empty"


def test_runner_offline(tmp_path, runner):
    """Quick run completes without network access (synthetic-only path)."""
    import socket

    _orig_connect = socket.socket.connect

    def _block_connect(self, address):
        raise ConnectionRefusedError(f"Network access blocked during test: {address}")

    socket.socket.connect = _block_connect  # type: ignore[method-assign]
    try:
        figures_dir = tmp_path / "figures_offline"
        rc = runner.main(["--quick", "--output-dir", str(figures_dir)])
        assert rc == 0, "Runner failed without network access (should be synthetic-only)"
    finally:
        socket.socket.connect = _orig_connect  # type: ignore[method-assign]

    # ccxt should not be in sys.modules (not even attempted)
    assert "ccxt" not in sys.modules, "ccxt imported during offline quick run"


def test_runner_bad_args(runner):
    """Unknown flag exits with SystemExit code 2 (argparse behavior)."""
    with pytest.raises(SystemExit) as exc_info:
        runner.main(["--bogus-flag"])
    assert exc_info.value.code == 2, (
        f"Expected exit code 2 for bad args, got {exc_info.value.code}"
    )


def test_runner_deterministic_artifacts(tmp_path, runner):
    """Two quick runs with the same seed produce identical Results sections."""
    dir_a = tmp_path / "run_a" / "figures"
    dir_b = tmp_path / "run_b" / "figures"

    rc_a = runner.main(["--quick", "--seed", "42", "--output-dir", str(dir_a)])
    rc_b = runner.main(["--quick", "--seed", "42", "--output-dir", str(dir_b)])

    assert rc_a == 0 and rc_b == 0, "One or both quick runs failed"

    summary_a = (tmp_path / "run_a" / "summary.md").read_text(encoding="utf-8")
    summary_b = (tmp_path / "run_b" / "summary.md").read_text(encoding="utf-8")

    # Extract the ## Results section from each summary
    def _extract_results(text: str) -> str:
        lines = text.splitlines()
        in_results = False
        result_lines = []
        for line in lines:
            if line.startswith("## Results"):
                in_results = True
            elif line.startswith("## ") and in_results:
                break
            if in_results:
                result_lines.append(line)
        return "\n".join(result_lines)

    results_a = _extract_results(summary_a)
    results_b = _extract_results(summary_b)

    assert results_a == results_b, (
        "Results sections differ between two identical-seed quick runs.\n"
        f"Run A:\n{results_a}\n\nRun B:\n{results_b}"
    )
