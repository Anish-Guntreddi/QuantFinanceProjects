"""VolSurfaceLab ReportBuilder — figures and markdown summary.

IMPORTANT: matplotlib.use("Agg") is called at module import BEFORE any pyplot
import.  This mirrors the qbacktest tearsheet and macroregime ReportBuilder
patterns and ensures the module is safe in headless CI environments.

Output layout:
    <output_dir>/               <- figures directory (figures/)
        smile_T{T}.png          <- one smile plot per validated maturity
        surface_3d.png          <- 3-D fitted SVI IV surface
        surface_heatmap.png     <- seaborn heatmap of same grid
        vrp_pnl.png             <- cumulative gross vs net P&L + VRP twin panel
        forecast_qlike.png      <- QLIKE bar chart per model
    <output_dir>.parent/        <- report root (e.g. reports/)
        summary.md              <- markdown summary alongside figures dir

Usage:
    from volsurfacelab.report import ReportBuilder
    paths = ReportBuilder(results, output_dir).build()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # MUST be before any pyplot import; headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

from volsurfacelab.pipeline import PipelineResults
from volsurfacelab.svi import svi_w

__all__ = ["ReportBuilder"]

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_K_DENSE = np.linspace(-1.5, 1.5, 200)  # dense moneyness grid for SVI curves
_T_HEATMAP_N_ROWS = 40                   # number of T points for 3d/heatmap grid
_FIGSIZE_SMILE = (8, 5)
_FIGSIZE_SURFACE = (10, 7)
_FIGSIZE_HEATMAP = (10, 6)
_FIGSIZE_PNL = (10, 7)
_FIGSIZE_FORECAST = (8, 5)


class ReportBuilder:
    """Build all report artifacts from a PipelineResults object.

    Parameters
    ----------
    results : PipelineResults
        Frozen output from VolSurfacePipeline.run().
    output_dir : Path
        Directory for figure files (e.g. reports/figures/).
        Will be created if it does not exist.
    """

    def __init__(self, results: PipelineResults, output_dir: Path) -> None:
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Path]:
        """Build all figures and summary.md.

        Returns
        -------
        dict mapping artifact name -> Path for every artifact produced.
        """
        paths: Dict[str, Path] = {}

        # Smile plots: one per validated maturity
        for T, fit in sorted(self.results.svi_fits.items()):
            key = f"smile_T{T}"
            paths[key] = self._build_smile(T, fit)

        # 3D surface
        paths["surface_3d"] = self._build_surface_3d()

        # Heatmap
        paths["surface_heatmap"] = self._build_surface_heatmap()

        # VRP P&L
        paths["vrp_pnl"] = self._build_vrp_pnl()

        # Forecast QLIKE bar chart
        paths["forecast_qlike"] = self._build_forecast_qlike()

        # summary.md (beside figures dir)
        paths["summary_md"] = self._build_summary_md()

        return paths

    # ------------------------------------------------------------------
    # Smile plots
    # ------------------------------------------------------------------

    def _build_smile(self, T: float, fit) -> Path:
        """Scatter of solved IVs vs k, overlaid with fitted SVI curve."""
        fig, ax = plt.subplots(figsize=_FIGSIZE_SMILE)

        # Observed IVs from solved iv_frame (calls only)
        iv_frame = self.results.iv_frame
        calls = iv_frame[(iv_frame["flag"] == "c") & (iv_frame["T"] == T)].dropna(subset=["iv"])
        if not calls.empty:
            ax.scatter(
                calls["k"].values,
                calls["iv"].values,
                marker="o",
                s=40,
                color="steelblue",
                alpha=0.8,
                label="Solved IV (prices)",
                zorder=3,
            )

        # Fitted SVI curve
        if fit.params is not None:
            w_fit = svi_w(_K_DENSE, *fit.params)
            # Total variance w = iv^2 * T  =>  iv = sqrt(w / T)
            iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
            ax.plot(
                _K_DENSE,
                iv_fit,
                color="orangered",
                linewidth=2,
                label="Fitted SVI",
            )

            # ATM IV annotation
            atm_w = svi_w(np.array([0.0]), *fit.params)[0]
            atm_iv = math.sqrt(max(atm_w, 0.0) / T)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.annotate(
                f"ATM IV ≈ {atm_iv:.1%}",
                xy=(0, atm_iv),
                xytext=(0.1, atm_iv + 0.01),
                fontsize=9,
                color="darkred",
            )

        T_label = f"{T:.2f}y"
        ax.set_title(f"IV Smile — Maturity T={T_label}")
        ax.set_xlabel("Log-moneyness k = log(K/F)")
        ax.set_ylabel("Implied Volatility")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = self.output_dir / f"smile_T{T:.2f}.png".replace(".", "_", 1)
        # Use canonical name with dot
        fname = self.output_dir / f"smile_T{T}.png"
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        return fname

    # ------------------------------------------------------------------
    # 3D surface
    # ------------------------------------------------------------------

    def _build_surface_3d(self) -> Path:
        """mpl_toolkits.mplot3d surface of fitted IV over (k, T) grid."""
        fitted = self.results.svi_fits
        if not fitted:
            # Edge case: nothing to plot (shouldn't happen after gate check)
            fig = plt.figure(figsize=_FIGSIZE_SURFACE)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No validated slices", ha="center", va="center")
            fname = self.output_dir / "surface_3d.png"
            fig.savefig(fname, dpi=100)
            plt.close(fig)
            return fname

        T_vals = np.array(sorted(fitted.keys()))
        k_vals = _K_DENSE

        # Build IV grid: rows = T, cols = k
        iv_grid = np.full((len(T_vals), len(k_vals)), np.nan)
        for i, T in enumerate(T_vals):
            fit = fitted[T]
            if fit.params is not None:
                w = svi_w(k_vals, *fit.params)
                iv_grid[i, :] = np.sqrt(np.maximum(w, 0.0) / T)

        K_mesh, T_mesh = np.meshgrid(k_vals, T_vals)

        fig = plt.figure(figsize=_FIGSIZE_SURFACE)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            K_mesh,
            T_mesh,
            iv_grid,
            cmap="viridis",
            alpha=0.85,
            linewidth=0,
            antialiased=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Implied Volatility")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("IV")
        ax.set_title("Fitted SVI Implied Volatility Surface")

        fname = self.output_dir / "surface_3d.png"
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        return fname

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _build_surface_heatmap(self) -> Path:
        """Seaborn heatmap of fitted IV surface (T rows, k columns)."""
        fitted = self.results.svi_fits
        if not fitted:
            fig, ax = plt.subplots(figsize=_FIGSIZE_HEATMAP)
            ax.text(0.5, 0.5, "No validated slices", ha="center", va="center")
            fname = self.output_dir / "surface_heatmap.png"
            fig.savefig(fname, dpi=100)
            plt.close(fig)
            return fname

        T_vals = np.array(sorted(fitted.keys()))
        k_vals = _K_DENSE

        iv_grid = np.full((len(T_vals), len(k_vals)), np.nan)
        for i, T in enumerate(T_vals):
            fit = fitted[T]
            if fit.params is not None:
                w = svi_w(k_vals, *fit.params)
                iv_grid[i, :] = np.sqrt(np.maximum(w, 0.0) / T)

        # Subsample k columns to keep the heatmap readable
        n_k_ticks = 13
        step = max(1, len(k_vals) // n_k_ticks)
        k_subsample = k_vals[::step]
        iv_subsample = iv_grid[:, ::step]

        df_heat = pd.DataFrame(
            iv_subsample,
            index=[f"T={T:.2f}" for T in T_vals],
            columns=[f"{k:.2f}" for k in k_subsample],
        )

        fig, ax = plt.subplots(figsize=_FIGSIZE_HEATMAP)
        sns.heatmap(
            df_heat,
            ax=ax,
            cmap="RdYlGn_r",
            fmt=".2f",
            annot=True,
            annot_kws={"size": 8},
            cbar_kws={"label": "Implied Volatility"},
            linewidths=0.3,
        )
        ax.set_title("Fitted SVI IV Surface — Heatmap (T rows, k columns)")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Maturity T")
        fig.tight_layout()

        fname = self.output_dir / "surface_heatmap.png"
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        return fname

    # ------------------------------------------------------------------
    # VRP P&L
    # ------------------------------------------------------------------

    def _build_vrp_pnl(self) -> Path:
        """Cumulative gross vs net P&L (top panel) + VRP series (bottom panel)."""
        vrp = self.results.vrp
        gross_cum = vrp.gross_pnl.cumsum()
        net_cum = vrp.net_pnl.cumsum()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=_FIGSIZE_PNL, sharex=True)

        # Top: cumulative P&L
        ax1.plot(gross_cum.index, gross_cum.values, label="Gross P&L", color="steelblue", linewidth=1.5)
        ax1.plot(net_cum.index, net_cum.values, label="Net P&L", color="orangered", linewidth=1.5, linestyle="--")
        ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax1.set_ylabel("Cumulative P&L ($)")
        ax1.set_title(
            f"VRP Strategy ({vrp.side.capitalize()} Straddle) — "
            f"Entry IV={vrp.entry_iv:.1%}  |  Total costs=${vrp.total_costs:,.0f}"
        )
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Bottom: VRP series
        ax2.fill_between(
            vrp.vrp_series.index,
            vrp.vrp_series.values,
            0,
            where=vrp.vrp_series.values > 0,
            alpha=0.4,
            color="green",
            label="VRP > 0 (IV > RV)",
        )
        ax2.fill_between(
            vrp.vrp_series.index,
            vrp.vrp_series.values,
            0,
            where=vrp.vrp_series.values <= 0,
            alpha=0.4,
            color="red",
            label="VRP ≤ 0 (IV < RV)",
        )
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("VRP = IV² - RV (ann.)")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = self.output_dir / "vrp_pnl.png"
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        return fname

    # ------------------------------------------------------------------
    # Forecast QLIKE bar chart
    # ------------------------------------------------------------------

    def _build_forecast_qlike(self) -> Path:
        """Bar chart of QLIKE per model."""
        table = self.results.forecast.table
        models = list(table.index)
        qlike_vals = [float(table.loc[m, "qlike"]) for m in models]

        fig, ax = plt.subplots(figsize=_FIGSIZE_FORECAST)
        colors = ["steelblue", "orangered", "seagreen"]
        bars = ax.bar(models, qlike_vals, color=colors[: len(models)], alpha=0.8, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, qlike_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1e-7,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title("RV Forecast Comparison — QLIKE Loss (lower is better)")
        ax.set_ylabel("QLIKE (Patton 2011)")
        ax.set_xlabel("Model")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        fname = self.output_dir / "forecast_qlike.png"
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        return fname

    # ------------------------------------------------------------------
    # summary.md
    # ------------------------------------------------------------------

    def _build_summary_md(self) -> Path:
        """Write summary.md to output_dir.parent (beside the figures dir)."""
        results = self.results

        lines: List[str] = []
        lines.append("# VolSurfaceLab Research Summary\n")

        # ------ Surface section ------
        lines.append("## Surface\n")
        n_validated = len(results.svi_fits)
        n_excluded = len(results.excluded_slices)
        lines.append(f"- **Validated slices:** {n_validated}")
        lines.append(f"- **Excluded slices:** {n_excluded}")

        if results.excluded_slices:
            lines.append("\n**Excluded slices:**\n")
            lines.append("| Maturity | Reason |")
            lines.append("|----------|--------|")
            for T, reason in sorted(results.excluded_slices):
                lines.append(f"| T={T:.4f} | {reason} |")

        lines.append("\n**Fitted SVI Parameters:**\n")
        lines.append("| Maturity T | a | b | rho | m | sigma | SSE |")
        lines.append("|-----------|---|---|-----|---|-------|-----|")
        for T, fit in sorted(results.svi_fits.items()):
            if fit.params is not None:
                a, b, rho, m, sigma = fit.params
                lines.append(
                    f"| {T:.2f} | {a:.6f} | {b:.6f} | {rho:.6f} | {m:.6f} | {sigma:.6f} | {fit.sse:.2e} |"
                )

        # ------ RV Forecast Comparison section ------
        lines.append("\n## RV Forecast Comparison\n")
        table = results.forecast.table
        lines.append("| Model | QLIKE | MSE | Converged |")
        lines.append("|-------|-------|-----|-----------|")
        for model in table.index:
            qlike_val = float(table.loc[model, "qlike"])
            mse_val = float(table.loc[model, "mse"])
            conv = results.forecast.convergence.get(model, True)
            lines.append(f"| {model} | {qlike_val:.6f} | {mse_val:.2e} | {conv} |")

        lines.append("\n**Diebold-Mariano p-values (H0: equal predictive accuracy):**\n")
        lines.append("| Pair | DM stat | p-value |")
        lines.append("|------|---------|---------|")
        for pair, dm in results.forecast.dm_pvalues.items():
            lines.append(f"| {pair} | {dm['dm_stat']:.4f} | {dm['p_value']:.4f} |")

        lines.append(
            "\n_Note: With N~250-500 OOS observations, DM test results are indicative only "
            "(small-sample caveat, Research Open Question 2)._\n"
        )

        # ------ Strategy section ------
        lines.append("## Strategy\n")
        vrp = results.vrp
        total_gross = float(vrp.gross_pnl.sum())
        total_net = float(vrp.net_pnl.sum())
        mean_vrp = float(vrp.vrp_series.mean())
        lines.append(f"- **Side:** {vrp.side}")
        lines.append(f"- **Entry IV:** {vrp.entry_iv:.1%}")
        lines.append(f"- **Gross total P&L:** ${total_gross:,.2f}")
        lines.append(f"- **Net total P&L (net of costs):** ${total_net:,.2f}")
        lines.append(f"- **Total costs:** ${vrp.total_costs:,.2f}")
        lines.append(f"- **Mean VRP (IV² - RV, ann.):** {mean_vrp:.6f}")
        lines.append(f"- **N trading days:** {len(vrp.net_pnl)}")

        # ------ Greeks Risk Summary section ------
        lines.append("\n## Greeks Risk Summary\n")
        greeks_df = vrp.greeks_summary
        if not greeks_df.empty:
            # Render as markdown table
            cols = list(greeks_df.columns)
            lines.append("| Position | " + " | ".join(cols) + " |")
            lines.append("|----------|" + "|".join(["------"] * len(cols)) + "|")
            for idx, row in greeks_df.iterrows():
                vals = " | ".join(f"{v:.6f}" for v in row.values)
                lines.append(f"| {idx} | {vals} |")
        else:
            lines.append("_No positions in Greeks summary._")

        lines.append("\n---")
        lines.append("*Generated by VolSurfaceLab ReportBuilder (matplotlib Agg backend)*")

        # Write to parent of figures dir
        summary_path = self.output_dir.parent / "summary.md"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary_path
