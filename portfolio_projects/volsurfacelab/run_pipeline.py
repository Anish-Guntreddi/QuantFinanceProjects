"""VolSurfaceLab one-command research pipeline.

Usage
-----
    python run_pipeline.py --quick                    # fast path (~5s), all sections exercised
    python run_pipeline.py --seed 123                 # different random seed
    python run_pipeline.py --output-dir out/figures   # custom output directory
    python run_pipeline.py --config configs/custom.yaml

Design notes
------------
- Structured as `def main(argv=None) -> int` returning 0 on success so
  tests/test_integration.py can call main() in-process without subprocess
  overhead (Phase 2 locked pattern, mirrors run_macroregime.py).
- No sys.path hacks — volsurfacelab is installed as an editable package into
  the quant/ venv via `pip install -e portfolio_projects/volsurfacelab`.
- --quick reduces n_days=400 and n_restarts=2 but exercises every report
  section: smile plots, 3D surface, heatmap, VRP P&L, forecast QLIKE,
  Greeks summary, and summary.md.
- Output layout:
    <output-dir>/           (default: reports/figures/)
        smile_T*.png        one per validated maturity
        surface_3d.png
        surface_heatmap.png
        vrp_pnl.png
        forecast_qlike.png
    <output-dir>/../        (default: reports/)
        summary.md
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
logger = logging.getLogger("volsurfacelab.runner")


# ---------------------------------------------------------------------------
# Step banner helper
# ---------------------------------------------------------------------------

def _banner(step: str, t0: float | None = None) -> float:
    """Print a timed step banner and return current time."""
    elapsed = f" (+{time.time() - t0:.1f}s)" if t0 is not None else ""
    msg = f"\n{'=' * 60}\nStep: {step}{elapsed}\n{'=' * 60}"
    logger.info(msg)
    return time.time()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VolSurfaceLab research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: n_days=400, n_restarts=2, fast for integration testing",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic chain and GARCH returns path",
    )
    p.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Output directory for PNG figures (summary.md written to parent dir)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional path to a custom YAML config file",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: object) -> None:
    """Print a compact console summary of pipeline results."""
    from volsurfacelab.pipeline import PipelineResults

    r: PipelineResults = results  # type: ignore[assignment]

    print("\n" + "=" * 70)
    print("VOLSURFACELAB PIPELINE SUMMARY")
    print("=" * 70)

    # Surface
    n_val = len(r.svi_fits)
    n_excl = len(r.excluded_slices)
    print(f"\nSurface: {n_val} validated slices, {n_excl} excluded")
    for T_excl, reason in sorted(r.excluded_slices):
        print(f"  Excluded T={T_excl:.2f}: {reason}")

    # Forecast comparison
    print("\nRV Forecast Comparison (QLIKE / MSE):")
    table = r.forecast.table
    print(f"  {'Model':<12} {'QLIKE':>10} {'MSE':>12}")
    print("  " + "-" * 36)
    for model in table.index:
        qlike = float(table.loc[model, "qlike"])
        mse = float(table.loc[model, "mse"])
        print(f"  {model:<12} {qlike:>10.6f} {mse:>12.2e}")

    print("\nDiebold-Mariano p-values (H0: equal predictive accuracy):")
    for pair, dm in r.forecast.dm_pvalues.items():
        print(f"  {pair}: DM={dm['dm_stat']:.4f}, p={dm['p_value']:.4f}")

    # Strategy
    vrp = r.vrp
    gross = float(vrp.gross_pnl.sum())
    net = float(vrp.net_pnl.sum())
    mean_vrp = float(vrp.vrp_series.mean())
    print(f"\nStrategy ({vrp.side.capitalize()} delta-hedged straddle):")
    print(f"  Entry IV:   {vrp.entry_iv:.1%}")
    print(f"  Gross P&L:  ${gross:>10,.2f}")
    print(f"  Net P&L:    ${net:>10,.2f}  (after costs ${vrp.total_costs:,.2f})")
    print(f"  Mean VRP:   {mean_vrp:.6f}  (IV^2 - RV, annualised)")
    print(f"  N days:     {len(vrp.net_pnl)}")

    print("\n" + "-" * 70)
    print(
        "Note: Results use a synthetic DGP (seeded GARCH(1,1) underlying + SVI\n"
        "chain). VRP on synthetic data demonstrates the machinery, not a\n"
        "tradable anomaly. See README.md Limitations for full caveats.\n"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the full VolSurfaceLab research pipeline in-process.

    Parameters
    ----------
    argv:
        Argument list (defaults to sys.argv[1:] when None).

    Returns
    -------
    int
        0 on success, 1 on error.
    """
    try:
        args = _parse_args(argv)
    except SystemExit:
        raise  # propagate argparse SystemExit (bad args) directly

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe_start = time.time()

    try:
        # ------------------------------------------------------------------
        # Step 1: Run the pipeline
        # ------------------------------------------------------------------
        t0 = _banner("1 / 2  VolSurfacePipeline")
        from volsurfacelab.pipeline import VolSurfacePipeline, load_config

        config = None
        if args.config is not None:
            config = load_config(Path(args.config))

        pipeline = VolSurfacePipeline(
            config=config,
            seed=args.seed,
            quick=args.quick,
        )
        results = pipeline.run()
        logger.info(
            "Pipeline done: %d validated slices, %d excluded, entry_iv=%.1f%%",
            len(results.svi_fits),
            len(results.excluded_slices),
            results.vrp.entry_iv * 100,
        )

        # ------------------------------------------------------------------
        # Step 2: Build report figures + summary.md
        # ------------------------------------------------------------------
        t0 = _banner("2 / 2  ReportBuilder", t0)
        from volsurfacelab.report import ReportBuilder

        artifacts = ReportBuilder(results, output_dir).build()
        for name, path in artifacts.items():
            logger.info("  %s -> %s", name, path)

        # ------------------------------------------------------------------
        # Console summary
        # ------------------------------------------------------------------
        _print_summary(results)

        total_time = time.time() - pipe_start
        logger.info("\nPipeline complete in %.1f seconds.", total_time)
        return 0

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return 1


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())
