"""ReportBuilder: figure rendering and markdown results report for AlphaRank.

Design notes
------------
- matplotlib.use('Agg') called at module level BEFORE pyplot import — headless
  safety pattern locked from Phase 1 (qbacktest tearsheet convention).
- All figures saved to output_dir/figures/ with deterministic filenames (no
  timestamps); fixed figsize/dpi for reproducibility.
- write_markdown renders the full results report: IC comparison table, backtest
  table (gross vs net Sharpe with bootstrap CI — QUAL-03), attribution table,
  and honest planted-alpha disclosure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must be set BEFORE pyplot import (Phase 1 pattern)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ["ReportBuilder"]


class ReportBuilder:
    """Build figures and markdown report for an AlphaRank pipeline run.

    Parameters
    ----------
    output_dir : Path
        Root output directory (e.g. ``reports/``).  Figures are saved to
        ``output_dir/figures/`` which is created if it does not exist.
    """

    # Fixed figure params for deterministic output (no timestamps in filenames)
    _FIGSIZE = (10, 5)
    _FIGSIZE_TALL = (10, 8)
    _DPI = 100

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: IC comparison bar chart
    # ------------------------------------------------------------------

    def fig_ic_comparison(self, table_df: pd.DataFrame) -> Path:
        """Bar chart of mean IC per model (baseline order) with NW t-stat annotations.

        Parameters
        ----------
        table_df : pd.DataFrame
            Output of run_model_comparison() — columns: model, mean_ic, icir,
            nw_tstat, p_value, n_months.  Rows in BASELINE_ORDER.

        Returns
        -------
        Path
            Saved PNG path.
        """
        fig, ax = plt.subplots(figsize=self._FIGSIZE, dpi=self._DPI)

        models = table_df["model"].tolist()
        mean_ics = table_df["mean_ic"].tolist()
        nw_tstats = table_df["nw_tstat"].tolist()

        x = np.arange(len(models))
        colors = ["#2196F3" if ic >= 0 else "#F44336" for ic in mean_ics]
        bars = ax.bar(x, mean_ics, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

        # Annotate NW t-stat above each bar
        for i, (bar, tstat) in enumerate(zip(bars, nw_tstats)):
            y_offset = max(0.0, bar.get_height()) + 0.002
            if bar.get_height() < 0:
                y_offset = bar.get_height() - 0.008
            label = f"t={tstat:.2f}" if not np.isnan(tstat) else "t=n/a"
            ax.text(bar.get_x() + bar.get_width() / 2, y_offset, label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Horizontal reference at 0
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Mean OOS IC (Spearman)")
        ax.set_title("Model Comparison: Mean OOS Information Coefficient\n(Newey-West t-stats annotated)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = self.figures_dir / "ic_comparison.png"
        fig.savefig(path, dpi=self._DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Figure 2: IC decay
    # ------------------------------------------------------------------

    def fig_ic_decay(self, decay_by_model: dict[str, pd.DataFrame]) -> Path:
        """Line plot of IC decay across horizons for all four models.

        Parameters
        ----------
        decay_by_model : dict[str, pd.DataFrame]
            {model_name: ic_decay DataFrame} where each DataFrame has index=horizon,
            column mean_ic.

        Returns
        -------
        Path
        """
        fig, ax = plt.subplots(figsize=self._FIGSIZE, dpi=self._DPI)

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        markers = ["o", "s", "^", "D"]

        for (model_name, decay_df), color, marker in zip(
            decay_by_model.items(), colors, markers
        ):
            if "mean_ic" in decay_df.columns and len(decay_df) > 0:
                horizons = decay_df.index.tolist()
                mean_ics = decay_df["mean_ic"].tolist()
                ax.plot(horizons, mean_ics, marker=marker, color=color,
                        label=model_name, linewidth=2, markersize=7)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon (months)")
        ax.set_ylabel("Mean IC (Spearman)")
        ax.set_title("IC Decay Profile Across Forward-Return Horizons")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks([1, 2, 3, 6])
        fig.tight_layout()

        path = self.figures_dir / "ic_decay.png"
        fig.savefig(path, dpi=self._DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Figure 3: Monthly IC time series
    # ------------------------------------------------------------------

    def fig_monthly_ic(
        self,
        ic_series_by_model: dict[str, pd.Series],
    ) -> Path:
        """Monthly IC time series for all four models, with rolling 6m mean for LGBM.

        Parameters
        ----------
        ic_series_by_model : dict[str, pd.Series]
            {model_name: monthly IC series with DatetimeIndex}.

        Returns
        -------
        Path
        """
        fig, ax = plt.subplots(figsize=self._FIGSIZE_TALL, dpi=self._DPI)

        colors = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#E1BEE7"]  # light for thin lines
        highlight_color = "#9C27B0"  # purple for LGBM rolling mean

        lgbm_series = None

        for (model_name, ic_series), color in zip(ic_series_by_model.items(), colors):
            if len(ic_series) == 0:
                continue
            ax.plot(ic_series.index, ic_series.values, color=color, linewidth=0.8,
                    alpha=0.7, label=model_name)
            if "lgbm" in model_name.lower() or "lightgbm" in model_name.lower():
                lgbm_series = ic_series

        # Highlighted 6-month rolling mean for LGBM
        if lgbm_series is not None and len(lgbm_series) >= 6:
            rolling_mean = lgbm_series.rolling(6, min_periods=3).mean()
            ax.plot(rolling_mean.index, rolling_mean.values,
                    color=highlight_color, linewidth=2.5,
                    label="LGBM 6m Rolling Mean", zorder=5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly IC (Spearman)")
        ax.set_title("Monthly IC Time Series — All Models\n(LGBM 6-month rolling mean highlighted)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        path = self.figures_dir / "monthly_ic_series.png"
        fig.savefig(path, dpi=self._DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Figure 4: Equity curves
    # ------------------------------------------------------------------

    def fig_equity_curves(
        self,
        results_by_model: dict[str, object],  # dict[str, BacktestResults]
        initial: float = 1_000_000.0,
    ) -> Path:
        """Two-panel equity curve figure.

        Panel 1: Net equity for all four models (one line each).
        Panel 2: Gross vs net equity for the LGBM strategy only (cost drag).

        Parameters
        ----------
        results_by_model : dict[str, BacktestResults]
        initial : float
            Starting capital for display normalization (default 1M).

        Returns
        -------
        Path
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self._FIGSIZE_TALL, dpi=self._DPI)

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        lgbm_results = None
        lgbm_name = None

        # Panel 1: Net equity all models
        for (model_name, results), color in zip(results_by_model.items(), colors):
            eq = results.equity_curve
            if eq is None or len(eq) == 0:
                continue
            ax1.plot(eq.index, eq.values, color=color, linewidth=1.5, label=model_name)
            if "lgbm" in model_name.lower() or "lightgbm" in model_name.lower():
                lgbm_results = results
                lgbm_name = model_name

        ax1.axhline(initial, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title("Net Equity Curves — All Models")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
        )

        # Panel 2: Gross vs net for LGBM
        if lgbm_results is not None:
            net_eq = lgbm_results.equity_curve
            # Reconstruct gross equity from gross returns
            if hasattr(lgbm_results, "gross_returns") and lgbm_results.gross_returns is not None:
                gross_rets = lgbm_results.gross_returns
                gross_eq = (1 + gross_rets).cumprod() * initial
                ax2.plot(gross_eq.index, gross_eq.values, color="#2196F3",
                         linewidth=1.8, label="Gross", linestyle="--")
            ax2.plot(net_eq.index, net_eq.values, color="#9C27B0",
                     linewidth=1.8, label="Net")
            ax2.axhline(initial, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
            ax2.set_ylabel("Portfolio Value ($)")
            ax2.set_title(f"Gross vs Net Equity — {lgbm_name or 'LGBM'} (Cost Drag Visible)")
            ax2.legend(loc="upper left", fontsize=9)
            ax2.grid(alpha=0.3)
            ax2.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
            )
        else:
            ax2.text(0.5, 0.5, "LGBM results not available",
                     ha="center", va="center", transform=ax2.transAxes)

        fig.autofmt_xdate()
        fig.tight_layout()

        path = self.figures_dir / "equity_curves.png"
        fig.savefig(path, dpi=self._DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # write_markdown: results report
    # ------------------------------------------------------------------

    def write_markdown(self, path: Path, sections: dict) -> Path:
        """Render the results report to markdown.

        Parameters
        ----------
        path : Path
            Output path (e.g. reports/RESULTS.md).
        sections : dict
            Expected keys:
            - "comparison_table": pd.DataFrame with columns model/mean_ic/icir/nw_tstat/p_value/n_months
            - "backtest_table": dict[str, dict] keyed by model name, values from summarize_results()
            - "attribution": dict from factor_attribution()
            - "planted_alpha_note": str describing targets

        Returns
        -------
        Path
        """
        comparison_df: pd.DataFrame = sections.get("comparison_table", pd.DataFrame())
        backtest_table: dict = sections.get("backtest_table", {})
        attribution: dict = sections.get("attribution", {})
        planted_note: str = sections.get("planted_alpha_note", "")

        lines: list[str] = []

        lines.append("# AlphaRank Pipeline Results\n")
        lines.append(
            "> **Honesty notice:** This is a methodology showcase on **synthetic** data.\n"
            "> The alpha signal is deliberately planted — see Planted Alpha section below.\n"
            "> Results are NOT real-world investment performance.\n"
        )

        # ---- IC Comparison Table ----
        lines.append("\n## Model Comparison: OOS IC Statistics\n")
        if not comparison_df.empty:
            lines.append(
                "| Model | Mean IC | ICIR | NW t-stat | p-value | N months |"
            )
            lines.append(
                "|-------|---------|------|-----------|---------|----------|"
            )
            for _, row in comparison_df.iterrows():
                lines.append(
                    f"| {row['model']} "
                    f"| {_fmt(row.get('mean_ic')):.4f} "
                    f"| {_fmt(row.get('icir')):.3f} "
                    f"| {_fmt(row.get('nw_tstat')):.3f} "
                    f"| {_fmt(row.get('p_value')):.3f} "
                    f"| {int(row.get('n_months', 0))} |"
                )
        else:
            lines.append("_No comparison data available._")

        # ---- Backtest Table (QUAL-03: gross AND net side-by-side) ----
        lines.append("\n## Backtest Results: Gross vs Net Performance\n")
        if backtest_table:
            lines.append(
                "| Model | Gross Sharpe | Net Sharpe | Sharpe 95% CI | "
                "Cost bps | Turnover | Max DD | Trades |"
            )
            lines.append(
                "|-------|-------------|-----------|---------------|"
                "---------|----------|--------|--------|"
            )
            for model_name, stats in backtest_table.items():
                ci_low = _fmt(stats.get("sharpe_ci_low"))
                ci_high = _fmt(stats.get("sharpe_ci_high"))
                ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
                lines.append(
                    f"| {model_name} "
                    f"| {_fmt(stats.get('gross_sharpe')):.3f} "
                    f"| {_fmt(stats.get('net_sharpe')):.3f} "
                    f"| {ci_str} "
                    f"| {_fmt(stats.get('cost_bps')):.1f} "
                    f"| {_fmt(stats.get('turnover')):.3f} "
                    f"| {_fmt(stats.get('max_drawdown')):.3f} "
                    f"| {int(stats.get('n_trades', 0))} |"
                )
        else:
            lines.append("_No backtest data available._")

        # ---- Attribution Table (ALR-08) ----
        lines.append("\n## Factor Attribution (LGBM Strategy)\n")
        if attribution:
            alpha = _fmt(attribution.get("alpha"))
            alpha_tstat = _fmt(attribution.get("alpha_tstat"))
            r_sq = _fmt(attribution.get("r_squared"))
            betas = attribution.get("betas", {})

            lines.append(
                "| Metric | Value |\n"
                "|--------|-------|"
            )
            lines.append(f"| Alpha (monthly) | {alpha:.5f} |")
            lines.append(f"| Alpha t-stat | {alpha_tstat:.3f} |")
            lines.append(f"| R-squared | {r_sq:.3f} |")
            for fname, beta_val in betas.items():
                lines.append(f"| Beta ({fname}) | {_fmt(beta_val):.3f} |")
        else:
            lines.append("_No attribution data available._")

        # ---- Figures ----
        lines.append("\n## Figures\n")
        lines.append("![IC Comparison](figures/ic_comparison.png)")
        lines.append("\n![IC Decay](figures/ic_decay.png)")
        lines.append("\n![Monthly IC Series](figures/monthly_ic_series.png)")
        lines.append("\n![Equity Curves](figures/equity_curves.png)")
        lines.append("\n![LGBM Tearsheet](figures/lgbm_tearsheet.png)")

        # ---- Planted Alpha Disclosure ----
        lines.append("\n## Planted Alpha Disclosure\n")
        if planted_note:
            lines.append(planted_note)
        else:
            lines.append(
                "The alpha signal in this dataset is **deliberately planted** using the "
                "formula: `alpha = IC_target * sigma_noise / sqrt(1 - IC_target^2)`. "
                "Momentum IC target = 0.06, Value IC target = 0.04. "
                "Reversal, volatility, quality, and liquidity are negative controls "
                "(IC target ≈ 0). The models are *expected* to recover this planted signal."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fmt(val) -> float:
    """Return float(val) if not None/NaN, else 0.0."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return 0.0
