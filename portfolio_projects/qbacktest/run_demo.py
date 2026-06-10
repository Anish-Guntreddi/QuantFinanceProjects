#!/usr/bin/env python3
"""One-command end-to-end demo: synthetic data → MA crossover strategy → engine → tearsheet.

Usage:
    python3 run_demo.py

Outputs:
    reports/figures/demo_tearsheet.png   — 3-panel tearsheet PNG
    stdout                               — gross/net summary table

Exit code: 0 on success, non-zero on any error.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local src is importable if running outside installed mode
_repo_root = Path(__file__).parent
sys.path.insert(0, str(_repo_root / "src"))
# Add examples/ directory so demo_ma_strategy is importable
sys.path.insert(0, str(_repo_root / "examples"))

import yaml

from qbacktest import (
    BacktestConfig,
    EventDrivenBacktester,
    HistoricalDataHandler,
    SyntheticOHLCVGenerator,
    TearsheetRenderer,
)
from qbacktest.execution import FixedSlippage, PercentageCommission
from qbacktest.execution.handler import SimulatedExecutionHandler

from demo_ma_strategy import MovingAverageCrossStrategy


def main() -> int:
    """Run the demo backtest and produce tearsheet + summary table."""

    # ---- Load config -------------------------------------------------------
    config_path = _repo_root / "configs" / "backtest_config.yml"
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    initial_capital = float(cfg.get("initial_capital", 100_000.0))
    position_size = float(cfg.get("position_size", 0.1))

    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        position_size=position_size,
    )

    # ---- Generate synthetic data -------------------------------------------
    # 504 bars (~2 trading years) across 3 symbols so all 3 tearsheet panels
    # exercise properly (Pitfall 6: need >50 bars for SMA warmup, >1 month for
    # monthly returns panel).
    gen = SyntheticOHLCVGenerator(
        symbols=["AAPL", "MSFT", "GOOG"],
        n_bars=504,
        seed=42,
    )
    data = gen.generate()

    # ---- Build components --------------------------------------------------
    data_handler = HistoricalDataHandler(data)
    strategy = MovingAverageCrossStrategy(fast=20, slow=50)

    execution_handler = SimulatedExecutionHandler(
        slippage_model=FixedSlippage(bps=10),
        commission_model=PercentageCommission(rate=0.001),
    )

    # ---- Run backtest -------------------------------------------------------
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        execution_handler=execution_handler,
        config=backtest_config,
    )
    results = engine.run()

    # ---- Render tearsheet --------------------------------------------------
    output_dir = _repo_root / "reports" / "figures"
    renderer = TearsheetRenderer(output_dir=str(output_dir))

    out_path = renderer.render(
        results,
        title="QBacktest Demo: MA Crossover Strategy (Synthetic OHLCV, 504 bars)",
        filename="demo_tearsheet.png",
    )

    if out_path is not None:
        print(f"Tearsheet written to: {out_path}")
    else:
        print("WARNING: equity curve too short to render tearsheet", file=sys.stderr)

    # ---- Print summary table -----------------------------------------------
    print()
    print("=" * 52)
    print("Demo Backtest Results — MA Crossover (20/50) SMA")
    print("=" * 52)
    table = renderer.summary_table(results)
    print(table)
    print("=" * 52)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
