"""TearsheetRenderer — 3-panel matplotlib PNG tearsheet and summary table.

Design decisions (Plan 01-08, locked):
  - Exactly 3 panels: equity curve (top), drawdown (middle), monthly returns bar (bottom)
  - Figures live under reports/figures/ (QUAL-05)
  - matplotlib.use("Agg") called before pyplot import for headless safety
  - Degenerate curve guard: equity curve < 2 bars → render() returns None, no file written
  - pandas 2.2+ monthly resample uses 'ME' not deprecated 'M'
  - Figures closed after savefig to avoid open-figure warnings
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must be set BEFORE any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from qbacktest.engine import BacktestResults


class TearsheetRenderer:
    """Render BacktestResults to a 3-panel matplotlib PNG tearsheet plus summary table.

    Parameters
    ----------
    output_dir:
        Directory where the PNG will be saved.  Created if it does not exist.
        Defaults to ``reports/figures`` relative to current working directory.
    """

    def __init__(self, output_dir: str | Path = "reports/figures") -> None:
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public: render PNG
    # ------------------------------------------------------------------

    def render(
        self,
        results: BacktestResults,
        title: str,
        filename: str = "tearsheet.png",
    ) -> Path | None:
        """Render a 3-panel tearsheet PNG.

        Parameters
        ----------
        results:
            BacktestResults from EventDrivenBacktester.run().
        title:
            Figure suptitle (shown at top).
        filename:
            Output file name (relative to output_dir).

        Returns
        -------
        Path
            Path to the written PNG, or ``None`` when the equity curve has < 2 bars
            (graceful guard — no file is written).
        """
        equity = results.equity_curve

        # Guard: degenerate curve (Pitfall 6 from RESEARCH.md)
        if equity is None or len(equity) < 2:
            return None

        # Prepare panels data
        net_returns = results.net_returns

        # Drawdown series: vectorised expanding-max formula
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max

        # Monthly returns: pandas 2.2+ uses 'ME' (not deprecated 'M')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            monthly_returns = net_returns.resample("ME").apply(
                lambda r: (1 + r).prod() - 1
            )

        # Build figure
        fig, axes = plt.subplots(
            3, 1,
            figsize=(10, 12),
            constrained_layout=True,
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # ---- Panel 1: Equity Curve ----------------------------------------
        ax1 = axes[0]
        ax1.plot(equity.index, equity.values, color="#2563EB", linewidth=1.5)
        ax1.set_title("Equity Curve (Net)", fontsize=11)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}"
        ))
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=30)

        # ---- Panel 2: Drawdown -------------------------------------------
        ax2 = axes[1]
        ax2.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color="#EF4444",
            alpha=0.7,
        )
        ax2.set_title("Drawdown", fontsize=11)
        ax2.set_ylabel("Drawdown (%)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x * 100:.1f}%"
        ))
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=30)

        # ---- Panel 3: Monthly Returns Bar Chart --------------------------
        ax3 = axes[2]
        colors = [
            "#16A34A" if r >= 0 else "#EF4444"
            for r in monthly_returns.values
        ]
        ax3.bar(
            monthly_returns.index,
            monthly_returns.values * 100,  # convert to percent
            color=colors,
            width=20,  # ~20 days wide for monthly bars
        )
        ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax3.set_title("Monthly Returns", fontsize=11)
        ax3.set_ylabel("Return (%)")
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.1f}%"
        ))
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.tick_params(axis="x", rotation=30)

        # ---- Save and close -----------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)

        return out_path

    # ------------------------------------------------------------------
    # Public: summary table
    # ------------------------------------------------------------------

    def summary_table(self, results: BacktestResults) -> str:
        """Format a fixed-width text table of Gross | Net metrics.

        Rows: Total Return, Sharpe, Sharpe 95% CI, Sortino, Max Drawdown,
              Turnover, Hit Rate, Cost (bps), Trades.

        Parameters
        ----------
        results:
            BacktestResults from EventDrivenBacktester.run().

        Returns
        -------
        str
            Multi-line formatted table string.
        """
        m = results.metrics

        def _fmt_pct(v: float) -> str:
            """Format as percentage string; handle nan/inf."""
            if math.isnan(v) or math.isinf(v):
                return "n/a"
            return f"{v * 100:.2f}%"

        def _fmt_float(v: float, decimals: int = 4) -> str:
            """Format as float string; handle nan/inf."""
            if math.isnan(v) or math.isinf(v):
                return "n/a"
            return f"{v:.{decimals}f}"

        # CI string
        ci_low = _fmt_float(m.sharpe_ci_low, 2)
        ci_high = _fmt_float(m.sharpe_ci_high, 2)
        ci_str = f"[{ci_low}, {ci_high}]"

        # Hit rate
        hr_str = _fmt_pct(m.hit_rate) if not math.isnan(m.hit_rate) else "n/a"

        # Build rows: (label, gross_col, net_col)
        # gross column shows n/a for cost-only or net-only rows
        rows = [
            ("Metric", "Gross", "Net"),
            ("-" * 30, "-" * 12, "-" * 12),
            ("Total Return", _fmt_pct(m.total_return), _fmt_pct(m.total_return)),
            ("Sharpe", _fmt_float(m.gross_sharpe, 4), _fmt_float(m.net_sharpe, 4)),
            ("Sharpe 95% CI", "n/a", ci_str),
            ("Sortino", "n/a", _fmt_float(m.sortino, 4)),
            ("Max Drawdown", "n/a", _fmt_pct(m.max_drawdown)),
            ("Turnover (ann.)", "n/a", _fmt_float(m.turnover, 4)),
            ("Hit Rate", "n/a", hr_str),
            ("Cost (bps)", "n/a", _fmt_float(m.cost_bps, 2)),
            ("Trades", "n/a", str(m.n_trades)),
        ]

        col_width_label = 20
        col_width_val = 14

        lines: list[str] = []
        for label, gross_val, net_val in rows:
            lines.append(
                f"{label:<{col_width_label}}  {gross_val:<{col_width_val}}  {net_val}"
            )

        return "\n".join(lines)
