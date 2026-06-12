"""DeFiRegimeNet one-command research pipeline runner.

Usage
-----
    python run_pipeline.py --quick                    # fast path (~60s), all sections
    python run_pipeline.py --seed 123                 # different random seed
    python run_pipeline.py --output-dir out/figures   # custom output directory

Design notes
------------
- Structured as `def main(argv=None) -> int` returning 0 on success so tests can
  call main() in-process without subprocess overhead (locked Phase 3/4 pattern,
  mirrors run_macroregime.py and volsurfacelab/run_pipeline.py).
- No sys.path hacks — defiregimenet is installed as an editable package.
- --quick applies quick-mode overrides in the pipeline (n_years=2, 2 tokens,
  reduced restarts). Exercises every report section offline.
- Output layout:
      <output-dir>/                (default: reports/figures/)
          regime_timeline_*.png
          transition_heatmaps.png
          cross_token_v_heatmap.png
          model_comparison.png
          qlike_table.png
          k_sensitivity.png
      <output-dir>/../             (default: reports/)
          summary.md
- argparse SystemExit propagated directly (bad args exit code 2, tested via
  pytest.raises(SystemExit)).
- All other exceptions caught, logged to stderr, return 1.
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
logger = logging.getLogger("defiregimenet.runner")


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
        description="DeFiRegimeNet research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2 years, 2 tokens, reduced restarts (for integration testing)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for PNG figures (summary.md written to parent dir). "
            "Defaults to reports/figures/ relative to this script."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------


def _print_summary(results: object) -> None:
    """Print a compact console summary of pipeline results."""
    r = results

    print("\n" + "=" * 70)
    print("DEFIREGIMENET PIPELINE SUMMARY")
    print("=" * 70)

    print(f"\nTokens: {list(r.tokens)}")
    print(f"Bars per token: {r.n_bars}")
    print(f"Seed: {r.seed}")

    # Model comparison
    mc = r.model_comparison
    print("\nModel Comparison (accuracy / log-loss):")
    print(f"  {'Model':<12} {'Accuracy':>10} {'Log-Loss':>12}")
    print("  " + "-" * 36)
    for model in mc.index:
        acc = float(mc.loc[model, "accuracy"])
        ll = float(mc.loc[model, "log_loss"])
        print(f"  {model:<12} {acc:>10.4f} {ll:>12.4f}")

    # Cross-token V
    cv = r.cross_token_v
    import numpy as np
    n = cv.shape[0]
    off_diag = [float(cv.iloc[i, j]) for i in range(n) for j in range(n) if i != j]
    mean_v = float(np.mean(off_diag)) if off_diag else float("nan")
    print(f"\nCross-token Cramér's V (mean off-diagonal): {mean_v:.3f}")
    print(
        "  (Independently detected per-token sequences; > 0.3 indicates genuine\n"
        "   shared-factor recovery; 1.0 would be degenerate shared-sequence shortcut)"
    )

    print("\n" + "-" * 70)
    print(
        "Note: Results use a synthetic DGP (4-state Markov, Student-t df=4,\n"
        "market_factor_weight=0.70). Per-token independent regime detection is the\n"
        "honest experimental setting — see README.md Limitations for full caveats."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the full DeFiRegimeNet research pipeline in-process.

    Parameters
    ----------
    argv : list[str] | None
        Argument list. If None, uses sys.argv[1:] (standard argparse default).

    Returns
    -------
    int
        0 on success, 1 on runtime error.
        argparse SystemExit (code 2) propagated directly for bad arguments.
    """
    # Propagate argparse SystemExit directly (tests assert code == 2)
    try:
        args = _parse_args(argv)
    except SystemExit:
        raise

    # Resolve output_dir relative to this file (runner location)
    _here = Path(__file__).parent
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _here / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe_start = time.time()

    try:
        # ------------------------------------------------------------------
        # Step 1: Run the pipeline
        # ------------------------------------------------------------------
        t0 = _banner("1 / 2  run_pipeline")
        from defiregimenet.pipeline import load_config, run_pipeline

        cfg = load_config()
        results = run_pipeline(config=cfg, quick=args.quick, seed=args.seed)
        logger.info(
            "Pipeline done: %d tokens, %d bars, seed=%d",
            len(results.tokens),
            results.n_bars,
            results.seed,
        )
        t0 = _banner("1 / 2  run_pipeline", t0)

        # ------------------------------------------------------------------
        # Step 2: Build report figures + summary.md
        # ------------------------------------------------------------------
        t0 = _banner("2 / 2  ReportBuilder", t0)
        from defiregimenet.report.builder import ReportBuilder

        artifacts = ReportBuilder(results, output_dir).build_all()
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
