"""MacroRegime one-command research pipeline.

Usage
-----
    python run_macroregime.py --quick               # 10-year synthetic panel, fast settings
    python run_macroregime.py --seed 123            # different random seed
    python run_macroregime.py --k 4                 # K=4 regime states
    python run_macroregime.py --backend gmm         # GMM instead of HMM
    python run_macroregime.py --output-dir out      # custom output directory

Design notes
------------
- Structured as `def main(argv=None) -> int` returning 0 on success so
  test_integration.py::test_runner_quick can call main() in-process without
  subprocess overhead (Phase 2 locked pattern).
- No sys.path hacks — macroregime is installed as an editable package into
  the quant/ venv via `pip install -e portfolio_projects/macroregime`.
- --quick is the fast research path: n_years=10, refit_every=63, n_restarts=2,
  reduced walk-forward window sizes, k_sensitivity with ks=(2,3) only.
  Every report section is still exercised (regime timeline, heatmap, dwell
  chart, equity comparison, stability table, K-sensitivity table).
- Output: PNGs to <output_dir>/, summary.md to parent(<output_dir>)/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("macroregime.runner")


def _banner(step: str, t0: float | None = None) -> float:
    """Print a timed step banner and return current time."""
    elapsed = f" (+{time.time() - t0:.1f}s)" if t0 is not None else ""
    msg = f"\n{'=' * 60}\nStep: {step}{elapsed}\n{'=' * 60}"
    logger.info(msg)
    return time.time()


def main(argv: list[str] | None = None) -> int:
    """Run the full macroregime research pipeline in-process.

    Parameters
    ----------
    argv:
        Argument list (defaults to sys.argv[1:] when None).

    Returns
    -------
    int
        0 on success, 1 on error.
    """
    args = _parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Run the pipeline (macro + market regimes, backtest)
    # ------------------------------------------------------------------
    t0 = _banner("1 / 6  MacroRegimePipeline")
    from macroregime import MacroRegimePipeline

    pipeline = MacroRegimePipeline(
        seed=args.seed,
        k=args.k,
        backend=args.backend,
        quick=args.quick,
    )
    results = pipeline.run()
    logger.info(
        "Pipeline done: %d combined regime bars, backtest gross_sharpe=%.3f",
        len(results.combined_regimes),
        results.regime_backtest.gross_sharpe,
    )

    # Recover asset_ohlcv for regime_timeline figure
    from macroregime.data.synthetic import SyntheticMacroGenerator
    gen = SyntheticMacroGenerator(n_years=pipeline.n_years, seed=args.seed)
    panel = gen.generate()
    asset_ohlcv = panel.asset_ohlcv

    # ------------------------------------------------------------------
    # Step 2: Benchmark backtests (60/40, equal weight, risk parity)
    # ------------------------------------------------------------------
    t0 = _banner("2 / 6  Benchmark backtests", t0)
    from macroregime.benchmarks.benchmarks import (
        build_60_40_weights,
        build_equal_weight_weights,
        build_risk_parity_weights,
        run_strategy_backtest,
    )
    from macroregime.allocation.weights import month_end_rebalance_dates
    import pandas as pd

    daily_index = next(iter(asset_ohlcv.values())).index
    rebal_dates_raw = month_end_rebalance_dates(daily_index)
    rebal_dates = [pd.Timestamp(ts) for ts in rebal_dates_raw]

    # Build returns for risk parity (needed for inverse-vol weighting)
    closes = pd.DataFrame(
        {sym: df["close"] for sym, df in asset_ohlcv.items()}
    )
    asset_returns = closes.pct_change(fill_method=None).dropna()

    sched_6040 = build_60_40_weights(rebal_dates)
    sched_ew = build_equal_weight_weights(rebal_dates)
    sched_rp = build_risk_parity_weights(asset_returns, rebal_dates)

    bt_6040 = run_strategy_backtest(asset_ohlcv=asset_ohlcv, weight_schedule=sched_6040)
    bt_ew = run_strategy_backtest(asset_ohlcv=asset_ohlcv, weight_schedule=sched_ew)
    bt_rp = run_strategy_backtest(asset_ohlcv=asset_ohlcv, weight_schedule=sched_rp)

    benchmark_results = {
        "60_40": bt_6040,
        "equal_weight": bt_ew,
        "risk_parity": bt_rp,
    }
    for name, bt in benchmark_results.items():
        logger.info(
            "  %s: gross_sharpe=%.3f, net_sharpe=%.3f",
            name, bt.gross_sharpe, bt.net_sharpe,
        )

    # ------------------------------------------------------------------
    # Step 3: Walk-forward OOS evaluation
    # ------------------------------------------------------------------
    t0 = _banner("3 / 6  Walk-forward OOS evaluation", t0)
    from macroregime.evaluation import run_walk_forward
    from macroregime.allocation.weights import load_regime_weights

    if args.quick:
        train_bars, test_bars = 504, 126
    else:
        train_bars, test_bars = 756, 252

    regime_weights = load_regime_weights()
    try:
        wf_results = run_walk_forward(
            asset_ohlcv=asset_ohlcv,
            combined_regimes=results.combined_regimes,
            regime_weights=regime_weights,
            train_bars=train_bars,
            test_bars=test_bars,
        )
        logger.info(
            "Walk-forward: %d windows, OOS net_sharpe=%.3f",
            len(wf_results.window_results),
            wf_results.oos_metrics.net_sharpe,
        )
    except Exception as exc:
        logger.warning("Walk-forward failed (non-fatal): %s", exc)
        wf_results = None

    # ------------------------------------------------------------------
    # Step 4: Regime stability report + K sensitivity
    # ------------------------------------------------------------------
    t0 = _banner("4 / 6  Stability report + K sensitivity", t0)
    from macroregime.evaluation import regime_stability_report, k_sensitivity
    from macroregime.data.loader_base import SyntheticMacroLoader
    from macroregime.features.market import build_market_features
    import numpy as np

    # Build raw feature matrices (same as pipeline uses)
    loader = SyntheticMacroLoader(generator=gen, seed=args.seed)
    macro_df = loader.load_panel().dropna()
    X_monthly = macro_df.values

    mkt_df = build_market_features(asset_ohlcv).dropna()
    X_daily = mkt_df.values

    stability = regime_stability_report(X_monthly, X_daily, k=args.k, quick=args.quick)
    logger.info(
        "Stability: HMM/GMM agreement=%.1f%%, drift=%.4f",
        stability["hmm_gmm_agreement"] * 100,
        stability["distribution_drift"],
    )

    # K sensitivity — use reduced ks in quick mode to keep runtime < 2 min
    if args.quick:
        ks = (2, 3)
    else:
        ks = (2, 3, 4)
    ksens = k_sensitivity(X_daily, ks=ks, backend=args.backend)
    for k_val, entry in ksens.items():
        dt = entry.get("dwell_times", {})
        logger.info("  K=%d dwell: %s", k_val, {k: f"{v:.1f}" for k, v in dt.items()})

    # ------------------------------------------------------------------
    # Step 5: Report figures + summary tables
    # ------------------------------------------------------------------
    t0 = _banner("5 / 6  Report figures + summary tables", t0)
    from macroregime.report.builder import ReportBuilder

    builder = ReportBuilder(output_dir=output_dir)
    artifacts = builder.build_all(
        pipeline_results=results,
        benchmark_results=benchmark_results,
        wf_results=wf_results,
        stability=stability,
        ksens=ksens,
        asset_ohlcv=asset_ohlcv,
    )
    for name, path in artifacts.items():
        logger.info("  %s -> %s", name, path)

    # ------------------------------------------------------------------
    # Step 6: Print summary table to stdout
    # ------------------------------------------------------------------
    t0 = _banner("6 / 6  Summary", t0)
    _print_summary(results, benchmark_results)

    total_time = time.time() - pipe_start
    logger.info("\nPipeline complete in %.1f seconds.", total_time)
    return 0


# ---------------------------------------------------------------------------
# Print summary table to stdout
# ---------------------------------------------------------------------------


def _print_summary(pipeline_results: object, benchmark_results: dict) -> None:
    """Print the strategy comparison table to stdout."""
    all_results = {
        "Regime": pipeline_results.regime_backtest,
        "60/40": benchmark_results.get("60_40"),
        "EqualWeight": benchmark_results.get("equal_weight"),
        "RiskParity": benchmark_results.get("risk_parity"),
    }

    print("\n" + "=" * 80)
    print("MACROREGIME PIPELINE SUMMARY")
    print("=" * 80)
    print(
        f"\n{'Strategy':<14} {'Gross Sharpe':>12} {'Net Sharpe':>10} "
        f"{'Net CI Low':>10} {'Net CI High':>11} {'Sortino':>8} "
        f"{'MaxDD':>7} {'Turnover':>9}"
    )
    print("-" * 80)

    def _f(v, fmt=".3f"):
        if v is None:
            return "  nan"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "  nan"
        import math
        if math.isnan(fv):
            return "  nan"
        if math.isinf(fv):
            return "  inf"
        return format(fv, fmt)

    for name, res in all_results.items():
        if res is None:
            print(f"{name:<14} {'—':>12} {'—':>10} {'—':>10} {'—':>11} {'—':>8} {'—':>7} {'—':>9}")
            continue
        m = getattr(res, "metrics", None)
        if m is None:
            print(f"{name:<14} {'—':>12} {'—':>10} {'—':>10} {'—':>11} {'—':>8} {'—':>7} {'—':>9}")
            continue
        print(
            f"{name:<14} {_f(m.gross_sharpe):>12} {_f(m.net_sharpe):>10} "
            f"{_f(m.sharpe_ci_low):>10} {_f(m.sharpe_ci_high):>11} "
            f"{_f(m.sortino):>8} {_f(m.max_drawdown):>7} {_f(m.turnover):>9}"
        )

    print("-" * 80)
    print(
        "\nNote: Results are based on a synthetic DGP (4-state Markov-switching).\n"
        "Sharpe/Sortino are annualised (252 trading days). CIs are 95% bootstrap.\n"
        "No parameter tuning was performed — illustrative allocation only.\n"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MacroRegime research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: n_years=10, fast detector settings",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation",
    )
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of regime states K",
    )
    p.add_argument(
        "--backend",
        choices=["hmm", "gmm"],
        default="hmm",
        help="Regime model backend",
    )
    p.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Output directory for PNG figures (summary.md written to parent)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
