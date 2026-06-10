"""Integration and end-to-end tests for the AlphaRank pipeline.

Implements:
  - test_end_to_end: in-process quick pipeline via run_pipeline.run()
  - test_runner_smoke: subprocess --quick run exits 0 and writes expected files
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).parent.parent


def _quick_config(tmp_output: Path) -> dict:
    """Build a minimal quick-mode config for in-process testing."""
    config_path = PROJECT_ROOT / "configs" / "alpharank_config.yml"
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Apply quick block
    quick = cfg.get("quick", {})
    cfg.setdefault("data", {})
    cfg["data"]["n_assets"] = quick.get("n_assets", 20)
    cfg["data"]["n_months"] = quick.get("n_months", 36)
    cfg["data"]["seed"] = 42
    cfg["_lgbm_n_estimators_override"] = 50
    # Reduced CV for quick mode (skfolio constraint: purge+embargo < fold_size)
    cfg["cv"] = {"n_folds": 5, "n_test_folds": 2, "purged_size": 1, "embargo_size": 0}
    cfg["_use_real_data"] = False

    return cfg


def test_end_to_end(tmp_path: Path) -> None:
    """In-process quick pipeline produces valid comparison table, backtests, attribution."""
    from run_pipeline import run

    cfg = _quick_config(tmp_path)
    results = run(cfg, output_dir=tmp_path)

    # Comparison table has 4 rows (one per baseline model)
    assert len(results.comparison_table) == 4, (
        f"Expected 4 model rows, got {len(results.comparison_table)}"
    )

    # Every model row has finite gross and net Sharpe
    for model_name, summary in results.backtest_summaries.items():
        gross = summary.get("gross_sharpe")
        net = summary.get("net_sharpe")
        assert gross is not None and gross == gross, (
            f"{model_name}: gross_sharpe is NaN or missing"
        )
        assert net is not None and net == net, (
            f"{model_name}: net_sharpe is NaN or missing"
        )
        # Net Sharpe <= Gross Sharpe (costs can only reduce performance)
        assert net <= gross + 1e-6, (
            f"{model_name}: net_sharpe ({net:.4f}) > gross_sharpe ({gross:.4f})"
        )

    # Attribution has r_squared in [0, 1]
    if results.attribution:
        r2 = results.attribution.get("r_squared", float("nan"))
        assert 0.0 <= r2 <= 1.0, f"r_squared={r2} not in [0, 1]"

    # Output files created: RESULTS.md and at least 4 figures
    assert (tmp_path / "RESULTS.md").exists(), "RESULTS.md not created"
    pngs = list((tmp_path / "figures").glob("*.png"))
    assert len(pngs) >= 4, f"Expected >= 4 PNG figures, found {len(pngs)}: {pngs}"


def test_runner_smoke() -> None:
    """Subprocess --quick run exits 0, writes >= 4 PNG figures and RESULTS.md."""
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--quick"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes ceiling; quick run is ~17-30s in practice
    )

    assert result.returncode == 0, (
        f"run_pipeline.py --quick exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )

    reports_figures = PROJECT_ROOT / "reports" / "figures"
    pngs = list(reports_figures.glob("*.png"))
    assert len(pngs) >= 4, (
        f"Expected >= 4 PNG files in reports/figures/, found {len(pngs)}: {pngs}"
    )

    results_md = PROJECT_ROOT / "reports" / "RESULTS.md"
    assert results_md.exists(), "reports/RESULTS.md not found after pipeline run"
