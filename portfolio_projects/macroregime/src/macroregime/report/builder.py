"""ReportBuilder — figures and markdown summary tables for the macroregime pipeline.

HEADLESS SAFETY (Phase 1 locked pattern):
    matplotlib.use("Agg") is called here at module import before any pyplot import.
    Never remove or move this call — it must precede all pyplot imports to prevent
    display errors in CI/offline environments.

LABEL-ALIGNMENT RULE (documented here, required by MCR-05):
    States are ordered ascending by the economic observable (observable_dim in
    CausalRegimeDetector). State 0 = lowest value of the observable (e.g. lowest
    GDP growth = recession/contraction), State K-1 = highest value (expansion).
    Alignment is implemented as:
        permutation = np.argsort(np.argsort(means[:, observable_dim]))
    The double argsort produces the rank (rank->index -> raw->rank) so that
    raw state index maps to its rank in the ordering. This is the inverse
    permutation of np.argsort which gives rank->raw.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent color palette for up to 5 regime states
_REGIME_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]

# Strategy display names for the comparison table
_STRATEGY_LABELS = {
    "regime": "Regime",
    "60_40": "60/40",
    "equal_weight": "EqualWeight",
    "risk_parity": "RiskParity",
}


class ReportBuilder:
    """Build figures and markdown summary tables for the macroregime research report.

    Parameters
    ----------
    output_dir:
        Root directory under which all figures and the summary.md will be written.
        PNGs go directly into this directory; summary.md is written to the parent.
    """

    def __init__(self, output_dir: str | Path = "reports/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # summary.md sits one level above the figures directory
        self._summary_path = self.output_dir.parent / "summary.md"
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure methods — each saves a PNG and returns its Path
    # ------------------------------------------------------------------

    def regime_timeline(
        self,
        combined_regimes: pd.Series,
        asset_ohlcv: dict[str, pd.DataFrame],
        filename: str = "regime_timeline.png",
    ) -> Path:
        """Equity close price with regime-colored background shading.

        Each regime state is shaded with a distinct color across the full
        EQUITY price series (log scale). A legend maps regime index to color.
        The label-alignment rule (state 0 = lowest observable) is noted.

        Parameters
        ----------
        combined_regimes:
            Daily pd.Series of combined regime labels (sentinel -1 = warm-up).
        asset_ohlcv:
            Dict of {symbol: OHLCV DataFrame}. EQUITY close is plotted.

        Returns
        -------
        Path
            Absolute path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot EQUITY close price
        equity_key = "EQUITY" if "EQUITY" in asset_ohlcv else (
            next(iter(asset_ohlcv)) if asset_ohlcv else None
        )
        if equity_key is not None:
            equity_close = asset_ohlcv[equity_key]["close"]
            ax.semilogy(
                equity_close.index, equity_close.values,
                color="black", linewidth=0.8, zorder=3
            )
            ax.set_ylabel("EQUITY Close (log scale)")

            # Shade background by regime
            k_unique = sorted([s for s in combined_regimes.unique() if s >= 0])
            regime_series = (
                combined_regimes
                .reindex(equity_close.index, method="ffill")
                .fillna(-1)
                .astype(int)
            )

            if len(regime_series) > 0:
                current_regime = regime_series.iloc[0]
                span_start = regime_series.index[0]
                for i in range(1, len(regime_series)):
                    r = int(regime_series.iloc[i])
                    if r != current_regime:
                        if current_regime >= 0:
                            color = _REGIME_COLORS[current_regime % len(_REGIME_COLORS)]
                            ax.axvspan(
                                span_start, regime_series.index[i - 1],
                                color=color, alpha=0.25, zorder=1,
                            )
                        current_regime = r
                        span_start = regime_series.index[i]
                # Final span
                if current_regime >= 0:
                    color = _REGIME_COLORS[current_regime % len(_REGIME_COLORS)]
                    ax.axvspan(
                        span_start, regime_series.index[-1],
                        color=color, alpha=0.25, zorder=1,
                    )

            # Legend
            patches = [
                mpatches.Patch(
                    color=_REGIME_COLORS[k % len(_REGIME_COLORS)],
                    alpha=0.5,
                    label=f"Regime {k}",
                )
                for k in k_unique
            ]
            ax.legend(handles=patches, loc="upper left", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No OHLCV data", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("Date")
        ax.set_title(
            "Regime Timeline (state 0 = lowest economic observable = contraction/stress)"
        )

        fig.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out_path)
        return out_path

    def transition_heatmap(
        self,
        diagnostics: dict,
        filename: str = "transition_heatmap.png",
    ) -> Path:
        """Heatmap of the combined-regime transition matrix with dwell-time annotations.

        Parameters
        ----------
        diagnostics:
            The diagnostics dict from PipelineResults (keys: macro, market, combined).

        Returns
        -------
        Path
        """
        try:
            import seaborn as sns
            _has_seaborn = True
        except ImportError:
            _has_seaborn = False

        combined_diag = diagnostics.get("combined", {})
        tm = combined_diag.get("transition_matrix", None)
        dwell = combined_diag.get("dwell_times", {})

        if tm is None or tm.size == 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.text(0.5, 0.5, "No transition data", ha="center", va="center",
                    transform=ax.transAxes)
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return out_path

        k = tm.shape[0]
        labels = [f"R{i}" for i in range(k)]
        dwell_str = ", ".join(f"R{i}: {dwell.get(i, 0):.1f}d" for i in range(k))

        fig, ax = plt.subplots(figsize=(6, 5))
        if _has_seaborn:
            sns.heatmap(
                tm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                vmin=0.0,
                vmax=1.0,
            )
        else:
            im = ax.imshow(tm, cmap="Blues", vmin=0, vmax=1)
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f"{tm[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(k))
            ax.set_xticklabels(labels)
            ax.set_yticks(range(k))
            ax.set_yticklabels(labels)
            plt.colorbar(im, ax=ax)

        ax.set_title(
            f"Combined-Regime Transition Matrix\n"
            f"Mean dwell times: {dwell_str}\n"
            f"(State 0 = lowest observable, state {k - 1} = highest)"
        )
        ax.set_xlabel("To Regime")
        ax.set_ylabel("From Regime")

        fig.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out_path)
        return out_path

    def dwell_time_chart(
        self,
        diagnostics: dict,
        filename: str = "dwell_time_chart.png",
    ) -> Path:
        """Bar chart of mean dwell times per regime for macro, market, and combined models.

        Parameters
        ----------
        diagnostics:
            The diagnostics dict from PipelineResults.

        Returns
        -------
        Path
        """
        models = ["macro", "market", "combined"]
        # Gather all regime indices
        all_regimes: set = set()
        for m in models:
            diag = diagnostics.get(m, {})
            dt = diag.get("dwell_times", {})
            all_regimes.update(kk for kk in dt if kk >= 0)

        regime_ids = sorted(all_regimes)
        if not regime_ids:
            regime_ids = [0]
        n_regimes = len(regime_ids)
        x = np.arange(n_regimes)
        width = 0.25
        n_models = len(models)

        fig, ax = plt.subplots(figsize=(8, 5))
        for mi, model_name in enumerate(models):
            diag = diagnostics.get(model_name, {})
            dt = diag.get("dwell_times", {})
            values = [dt.get(r, 0.0) for r in regime_ids]
            offset = (mi - n_models / 2.0 + 0.5) * width
            ax.bar(x + offset, values, width=width * 0.9, label=model_name.capitalize())

        ax.set_xlabel("Regime State (0 = lowest observable)")
        ax.set_ylabel("Mean Dwell Time (bars)")
        ax.set_title("Mean Dwell Time per Regime: Macro vs Market vs Combined")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{r}" for r in regime_ids])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out_path)
        return out_path

    def equity_comparison(
        self,
        results: dict[str, Any],
        filename: str = "equity_comparison.png",
    ) -> Path:
        """OOS equity curves for Regime, 60/40, EqualWeight, RiskParity (log scale).

        All strategies are plotted on identical periods starting from the same
        initial equity (re-based to 1.0 at the first available bar).

        Parameters
        ----------
        results:
            Dict keyed by strategy name -> BacktestResults (with .equity_curve attribute).

        Returns
        -------
        Path
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        strategy_colors = {
            "regime": "#e15759",
            "60_40": "#4e79a7",
            "equal_weight": "#59a14f",
            "risk_parity": "#f28e2b",
        }

        plotted = False
        for name, res in results.items():
            if res is None:
                continue
            eq = getattr(res, "equity_curve", None)
            if eq is None or len(eq) == 0:
                continue
            # Re-base to 1.0 at start
            first_val = float(eq.iloc[0])
            if first_val == 0.0:
                continue
            eq_rebased = eq / first_val
            label = _STRATEGY_LABELS.get(name, name)
            color = strategy_colors.get(name, "#aaaaaa")
            lw = 2.0 if name == "regime" else 1.2
            ax.semilogy(eq_rebased.index, eq_rebased.values,
                        label=label, color=color, linewidth=lw)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No equity data", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (re-based to 1.0, log scale)")
        ax.set_title("OOS Equity Comparison: Regime vs Static Benchmarks")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out_path)
        return out_path

    def summary_table(self, results: dict[str, Any]) -> str:
        """Build a markdown comparison table and write to summary.md.

        Rows: 4 strategies (Regime, 60/40, EqualWeight, RiskParity).
        Columns: Gross Sharpe, Net Sharpe, Net Sharpe 95% CI [low, high],
                 Sortino, MaxDD, Turnover.

        LABEL-ALIGNMENT RULE documented inline (MCR-05 requirement).

        Parameters
        ----------
        results:
            Dict keyed by strategy name -> BacktestResults (must have .metrics field).

        Returns
        -------
        str
            Markdown table string (also written as the body of summary.md).
        """
        strategy_order = ["regime", "60_40", "equal_weight", "risk_parity"]
        display_names = {
            "regime": "Regime",
            "60_40": "60/40",
            "equal_weight": "EqualWeight",
            "risk_parity": "RiskParity",
        }

        header = (
            "## Strategy Comparison\n\n"
            "> **Label-alignment rule**: Regime states are ordered ascending by the\n"
            "> economic observable (observable_dim). State 0 = lowest value\n"
            "> (contraction/recession/stress); state K-1 = highest (expansion/risk-on).\n"
            "> Implementation: `permutation = np.argsort(np.argsort(means[:, dim]))`\n"
            "> (double argsort maps raw->rank; single argsort gives rank->raw).\n\n"
            "Net Sharpe CIs are 95% bootstrap (1000 resamples, percentile method).\n\n"
        )

        col_header = (
            "| Strategy | Gross Sharpe | Net Sharpe | Net CI Low | Net CI High "
            "| Sortino | MaxDD | Turnover |\n"
            "|----------|-------------|-----------|-----------|------------|---------|-------|----------|\n"
        )

        def _f(v: Any, fmt: str = ".3f") -> str:
            if v is None:
                return "nan"
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return "nan"
            if np.isnan(fv):
                return "nan"
            if np.isinf(fv):
                return "inf"
            return format(fv, fmt)

        rows: list[str] = []
        for name in strategy_order:
            res = results.get(name)
            if res is None:
                rows.append(
                    f"| {display_names.get(name, name)} | — | — | — | — | — | — | — |"
                )
                continue

            m = getattr(res, "metrics", None)
            if m is None:
                rows.append(
                    f"| {display_names.get(name, name)} | — | — | — | — | — | — | — |"
                )
                continue

            rows.append(
                f"| {display_names.get(name, name)} "
                f"| {_f(m.gross_sharpe)} "
                f"| {_f(m.net_sharpe)} "
                f"| {_f(m.sharpe_ci_low)} "
                f"| {_f(m.sharpe_ci_high)} "
                f"| {_f(m.sortino)} "
                f"| {_f(m.max_drawdown)} "
                f"| {_f(m.turnover)} |"
            )

        table_str = header + col_header + "\n".join(rows) + "\n"

        # Write summary.md (overwrite with fresh header + table)
        self._write_to_summary(table_str, mode="w")
        return table_str

    def stability_table(self, stability: dict, ksens: dict) -> str:
        """Build HMM-vs-GMM agreement and K-sensitivity tables; append to summary.md.

        Parameters
        ----------
        stability:
            Output of evaluation.regime_stability_report().
        ksens:
            Output of evaluation.k_sensitivity().

        Returns
        -------
        str
            Markdown string of the stability section (also appended to summary.md).
        """
        lines: list[str] = ["## Regime Stability & K Sensitivity\n\n"]

        # --- HMM vs GMM agreement ---
        agreement = stability.get("hmm_gmm_agreement", float("nan"))
        drift = stability.get("distribution_drift", float("nan"))
        lines.append("### HMM vs GMM Stability\n\n")
        lines.append(
            f"- **HMM/GMM agreement (daily, aligned)**: "
            f"{agreement:.1%}\n"
        )
        lines.append(
            f"- **Distribution drift (L1, first vs second half)**: "
            f"{drift:.4f}\n\n"
        )

        # Market dwell times per backend
        mkt_dwell = stability.get("market_dwell_times", {})
        if mkt_dwell:
            lines.append("**Market dwell times (mean bars per regime)**\n\n")
            lines.append("| State | HMM | GMM |\n|-------|-----|-----|\n")
            hmm_dwell = mkt_dwell.get("hmm", {})
            gmm_dwell = mkt_dwell.get("gmm", {})
            all_states = sorted(
                set(list(hmm_dwell.keys()) + list(gmm_dwell.keys()))
            )
            for s in all_states:
                if s >= 0:
                    lines.append(
                        f"| R{s} | {hmm_dwell.get(s, 0):.1f} | {gmm_dwell.get(s, 0):.1f} |\n"
                    )
            lines.append("\n")

        # Macro dwell times per backend
        mac_dwell = stability.get("macro_dwell_times", {})
        if mac_dwell:
            lines.append("**Macro dwell times (mean observations per regime)**\n\n")
            lines.append("| State | HMM | GMM |\n|-------|-----|-----|\n")
            hmm_dwell = mac_dwell.get("hmm", {})
            gmm_dwell = mac_dwell.get("gmm", {})
            all_states = sorted(
                set(list(hmm_dwell.keys()) + list(gmm_dwell.keys()))
            )
            for s in all_states:
                if s >= 0:
                    lines.append(
                        f"| R{s} | {hmm_dwell.get(s, 0):.1f} | {gmm_dwell.get(s, 0):.1f} |\n"
                    )
            lines.append("\n")

        # --- K sensitivity ---
        lines.append("### K Sensitivity\n\n")
        lines.append(
            "> K was **not** selected by Sharpe (anti-feature: selecting K to maximize "
            "Sharpe overfits the regime model to the backtest period, invalidating the "
            "research hypothesis). K=3 is the default economic choice (contraction, "
            "neutral, expansion). Use BIC or dwell-time interpretability to select K.\n\n"
        )

        if ksens:
            lines.append(
                "| K | Mean Dwell R0 | Mean Dwell R1 | Mean Dwell R2+ "
                "| Agreement vs K=3 |\n"
            )
            lines.append(
                "|---|--------------|--------------|---------------|-----------------|\n"
            )
            for k_val in sorted(ksens.keys()):
                entry = ksens[k_val]
                dt = entry.get("dwell_times", {})
                d0 = dt.get(0, 0.0)
                d1 = dt.get(1, 0.0)
                d2 = dt.get(2, 0.0)
                agr = entry.get("agreement_vs_k3", float("nan"))
                agr_str = f"{agr:.1%}" if not np.isnan(agr) else "—"
                lines.append(
                    f"| {k_val} | {d0:.1f} | {d1:.1f} | {d2:.1f} | {agr_str} |\n"
                )

        stability_str = "".join(lines)
        self._write_to_summary(stability_str, mode="a")
        return stability_str

    def build_all(
        self,
        pipeline_results: Any,
        benchmark_results: dict[str, Any],
        wf_results: Any,
        stability: dict,
        ksens: dict,
        asset_ohlcv: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Path]:
        """Orchestrate all figure and table generation.

        Parameters
        ----------
        pipeline_results:
            PipelineResults from MacroRegimePipeline.run().
        benchmark_results:
            Dict of {"60_40": BacktestResults, "equal_weight": ..., "risk_parity": ...}.
        wf_results:
            WalkForwardResults from evaluation.run_walk_forward() (may be None).
        stability:
            Output of evaluation.regime_stability_report().
        ksens:
            Output of evaluation.k_sensitivity().
        asset_ohlcv:
            Optional dict of OHLCV DataFrames. If None, regime_timeline will render
            a placeholder.

        Returns
        -------
        dict[str, Path]
            Keys are artifact names; values are file paths.
        """
        artifacts: dict[str, Path] = {}

        # Combine results for equity comparison
        comparison_results: dict[str, Any] = {
            "regime": pipeline_results.regime_backtest,
            **benchmark_results,
        }

        # Override regime equity with OOS walk-forward curve if available
        if wf_results is not None:
            try:
                oos_eq = wf_results.oos_equity_curve
                if oos_eq is not None and len(oos_eq) > 0:
                    # Create a lightweight wrapper with equity_curve attribute
                    class _WFEquityWrapper:
                        def __init__(self, eq, bt):
                            self.equity_curve = eq
                            self.metrics = getattr(bt, "metrics", None)

                    comparison_results["regime"] = _WFEquityWrapper(
                        oos_eq, pipeline_results.regime_backtest
                    )
            except Exception as exc:
                logger.debug("Could not use WF equity curve for comparison: %s", exc)

        # 1. Regime timeline
        artifacts["regime_timeline"] = self.regime_timeline(
            pipeline_results.combined_regimes,
            asset_ohlcv or {},
        )

        # 2. Transition heatmap
        artifacts["transition_heatmap"] = self.transition_heatmap(pipeline_results.diagnostics)

        # 3. Dwell time chart
        artifacts["dwell_time_chart"] = self.dwell_time_chart(pipeline_results.diagnostics)

        # 4. Equity comparison
        artifacts["equity_comparison"] = self.equity_comparison(comparison_results)

        # 5. Summary table (written to summary.md)
        self.summary_table(comparison_results)
        artifacts["summary_md"] = self._summary_path

        # 6. Stability table (appended to summary.md)
        self.stability_table(stability, ksens)

        return artifacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_summary(self, content: str, mode: str = "a") -> None:
        """Write or append content to the summary.md file."""
        with open(self._summary_path, mode, encoding="utf-8") as fh:
            if mode == "w":
                fh.write("# MacroRegime Pipeline Summary\n\n")
            fh.write(content)
            fh.write("\n")
