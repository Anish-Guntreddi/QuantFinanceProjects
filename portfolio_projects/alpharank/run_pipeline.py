"""AlphaRank one-command research pipeline.

Usage
-----
    python run_pipeline.py                    # full run (n_assets=50, n_months=60)
    python run_pipeline.py --quick            # quick run (~2 min; n_assets=20, n_months=36)
    python run_pipeline.py --seed 123         # different random seed
    python run_pipeline.py --output-dir out   # custom output directory

Design notes
------------
- --real-data is a hidden opt-in: imports load_real_universe INSIDE the branch
  so the default offline path never touches yfinance.
- --quick swaps the config quick block (n_assets=20, n_months=36) and reduces
  LGBM n_estimators to 50 via config_override at model instantiation.
- Seed flows from --seed into CrossSectionalGenerator only; all models use
  fixed random_state=42 internally.
- Steps: generator → features → labels → purged-CV comparison → IC decay →
  decile backtests → factor attribution → report figures + markdown.

Provides a run(config: dict) -> PipelineResults function so test_integration.py
can call the pipeline in-process without subprocess overhead.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# PipelineResults container (exposed for in-process tests)
# ---------------------------------------------------------------------------

@dataclass
class PipelineResults:
    """Return value from run()."""
    comparison_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_frames: dict = field(default_factory=dict)
    backtest_summaries: dict = field(default_factory=dict)
    attribution: dict = field(default_factory=dict)
    ic_decay_by_model: dict = field(default_factory=dict)
    ic_series_by_model: dict = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("reports"))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alpharank.pipeline")


def _banner(step: str, t0: float | None = None) -> float:
    """Print a timed step banner and return current time."""
    elapsed = ""
    if t0 is not None:
        elapsed = f" (+{time.time() - t0:.1f}s)"
    msg = f"\n{'='*60}\nStep: {step}{elapsed}\n{'='*60}"
    logger.info(msg)
    return time.time()


# ---------------------------------------------------------------------------
# run(): in-process pipeline function (called by tests and main())
# ---------------------------------------------------------------------------

def run(config: dict, output_dir: Path | None = None) -> PipelineResults:
    """Execute the full AlphaRank research pipeline in-process.

    Parameters
    ----------
    config : dict
        Loaded and merged YAML config (with quick-block applied if needed).
    output_dir : Path, optional
        Root output directory for figures/RESULTS.md.  Defaults to ``reports/``.

    Returns
    -------
    PipelineResults
    """
    if output_dir is None:
        output_dir = Path("reports")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe_start = time.time()
    results = PipelineResults(output_dir=output_dir)

    # ------------------------------------------------------------------
    # Step 1: Data generation
    # ------------------------------------------------------------------
    t0 = _banner("1 / 8  Data generation")
    data_cfg = config.get("data", {})
    use_real_data = config.get("_use_real_data", False)

    if use_real_data:
        # Isolated import — default path never touches yfinance
        from alpharank.data.loader import load_real_universe  # type: ignore
        panel = load_real_universe(**data_cfg)
    else:
        from alpharank.data.generator import CrossSectionalGenerator
        gen = CrossSectionalGenerator(
            n_assets=data_cfg.get("n_assets", 50),
            n_months=data_cfg.get("n_months", 60),
            seed=data_cfg.get("seed", 42),
            momentum_ic_target=data_cfg.get("momentum_ic_target", 0.06),
            value_ic_target=data_cfg.get("value_ic_target", 0.04),
            monthly_vol=data_cfg.get("monthly_vol", 0.04),
            delist_prob_annual=data_cfg.get("delist_prob_annual", 0.03),
            start=data_cfg.get("start", "2018-01-31"),
        )
        panel = gen.generate()
        logger.info(
            "Generated panel: %d assets × %d months",
            data_cfg.get("n_assets", 50),
            data_cfg.get("n_months", 60),
        )

    # ------------------------------------------------------------------
    # Step 2: Features + labels
    # ------------------------------------------------------------------
    t0 = _banner("2 / 8  Features + labels", t0)
    from alpharank.features.factors import build_feature_panel
    from alpharank.labels.forward_returns import make_labels

    features = build_feature_panel(panel)
    logger.info("Feature panel: %d rows", len(features))

    # Month-end closes (BME aligned — panel.monthly_returns.index)
    close_wide = pd.DataFrame(
        {sym: panel.ohlcv[sym]["close"] for sym in panel.ohlcv}
    ).sort_index()
    month_end_closes = close_wide.reindex(panel.monthly_returns.index, method="ffill")

    label_df = make_labels(month_end_closes, horizon=config.get("labels", {}).get("horizon", 1))

    # Join features and labels on shared (date, symbol) pairs
    label_stacked = label_df.stack(future_stack=True).dropna()
    label_stacked.name = "label"
    label_stacked.index.names = ["date", "symbol"]

    X = features
    y = label_stacked.reindex(X.index).dropna()
    X = X.loc[y.index]
    logger.info("X shape: %s, y length: %d", X.shape, len(y))

    # ------------------------------------------------------------------
    # Step 3: Purged-CV model comparison
    # ------------------------------------------------------------------
    t0 = _banner("3 / 8  Purged-CV model comparison", t0)
    from alpharank.validation.purged_cv import PurgedCVEvaluator
    from alpharank.models.comparison import run_model_comparison

    cv_cfg = config.get("cv", {})
    evaluator = PurgedCVEvaluator(
        n_folds=cv_cfg.get("n_folds", 6),
        n_test_folds=cv_cfg.get("n_test_folds", 2),
        purged_size=cv_cfg.get("purged_size", 1),
        embargo_size=cv_cfg.get("embargo_size", 1),
    )

    # LGBM n_estimators reduced in quick mode
    lgbm_override = config.get("_lgbm_n_estimators_override", None)
    if lgbm_override is not None:
        from alpharank.models.lgbm import LGBMRankModel
        from alpharank.models import BASELINE_ORDER, EqualWeightComposite, LinearRankModel, ElasticNetRankModel

        # Monkey-patch LGBM default n_estimators for this run via subclass
        class _QuickLGBM(LGBMRankModel):
            def __init__(self):
                super().__init__()
                self._model.set_params(n_estimators=lgbm_override)

        quick_order = [EqualWeightComposite, LinearRankModel, ElasticNetRankModel, _QuickLGBM]
        # Temporarily patch BASELINE_ORDER used by run_model_comparison
        import alpharank.models.comparison as _cmp_mod
        import alpharank.models as _models_mod
        _orig_order = _models_mod.BASELINE_ORDER
        _models_mod.BASELINE_ORDER = quick_order
        _cmp_mod_orig = None
        comparison_table, oos_frames = run_model_comparison(X, y, evaluator)
        _models_mod.BASELINE_ORDER = _orig_order
    else:
        comparison_table, oos_frames = run_model_comparison(X, y, evaluator)

    results.comparison_table = comparison_table
    results.oos_frames = oos_frames
    logger.info("Comparison done: %d models", len(comparison_table))

    # ------------------------------------------------------------------
    # Step 4: IC decay per model
    # ------------------------------------------------------------------
    t0 = _banner("4 / 8  IC decay analysis", t0)
    from alpharank.analytics.ic_decay import ic_decay

    ic_horizons = tuple(config.get("ic", {}).get("horizons", [1, 2, 3, 6]))
    ic_decay_by_model: dict[str, pd.DataFrame] = {}
    ic_series_by_model: dict[str, pd.Series] = {}

    for model_name, oos_wide in oos_frames.items():
        if oos_wide.empty:
            ic_decay_by_model[model_name] = pd.DataFrame()
            ic_series_by_model[model_name] = pd.Series(dtype=float)
            continue
        try:
            decay_df = ic_decay(oos_wide, month_end_closes, horizons=ic_horizons)
            ic_decay_by_model[model_name] = decay_df
        except Exception as exc:
            logger.warning("IC decay failed for %s: %s", model_name, exc)
            ic_decay_by_model[model_name] = pd.DataFrame()

        # Monthly IC series at horizon=1 for the time-series figure
        from alpharank.analytics.ic import compute_ic_series
        from alpharank.labels.forward_returns import make_forward_returns
        fwd1 = make_forward_returns(month_end_closes, horizon=1)
        ic_s = compute_ic_series(oos_wide, fwd1)
        ic_series_by_model[model_name] = ic_s

    results.ic_decay_by_model = ic_decay_by_model
    results.ic_series_by_model = ic_series_by_model

    # ------------------------------------------------------------------
    # Step 5: Decile backtests
    # ------------------------------------------------------------------
    t0 = _banner("5 / 8  Decile backtests", t0)
    from alpharank.portfolio.construction import build_decile_weights
    from alpharank.portfolio.backtest import run_decile_backtest, summarize_results
    from alpharank.models import BASELINE_ORDER

    backtest_results: dict[str, object] = {}
    backtest_summaries: dict[str, dict] = {}

    baseline_names = [cls().name for cls in BASELINE_ORDER]

    for model_name in baseline_names:
        oos_wide = oos_frames.get(model_name)
        if oos_wide is None or oos_wide.empty:
            logger.warning("No OOS scores for %s — skipping backtest", model_name)
            continue
        try:
            weights = build_decile_weights(oos_wide)
            bt_results = run_decile_backtest(panel.ohlcv, weights)
            backtest_results[model_name] = bt_results
            backtest_summaries[model_name] = summarize_results(bt_results)
            s = backtest_summaries[model_name]
            logger.info(
                "  %s: gross_sharpe=%.3f, net_sharpe=%.3f, cost_bps=%.1f, n_trades=%d",
                model_name, s.get("gross_sharpe", 0), s.get("net_sharpe", 0),
                s.get("cost_bps", 0), s.get("n_trades", 0),
            )
        except Exception as exc:
            logger.warning("Backtest failed for %s: %s", model_name, exc)

    results.backtest_summaries = backtest_summaries

    # ------------------------------------------------------------------
    # Step 6: Factor attribution (ALR-08)
    # ------------------------------------------------------------------
    t0 = _banner("6 / 8  Factor attribution (LGBM strategy)", t0)
    from alpharank.analytics.attribution import factor_attribution
    from alpharank.analytics.ic import compute_ic_series as _cis

    # Build momentum and value factor-mimicking portfolio returns (frictionless
    # long-short on each single factor's z-scores — documented as frictionless).
    attribution_result: dict = {}

    # Find LGBM model name
    lgbm_model_name = next(
        (n for n in baseline_names if "lgbm" in n.lower() or "lightgbm" in n.lower()),
        None,
    )

    if lgbm_model_name and lgbm_model_name in backtest_results:
        lgbm_bt = backtest_results[lgbm_model_name]
        # Monthly net returns from backtest equity curve
        eq = lgbm_bt.equity_curve
        if eq is not None and len(eq) > 1:
            monthly_net_eq = eq.resample("ME").last()
            monthly_net_rets = monthly_net_eq.pct_change(fill_method=None).dropna()

            # Factor-mimicking portfolios: equal-weight long-short on z-score
            # (top half vs bottom half, frictionless)
            factor_rets_dict: dict[str, pd.Series] = {}
            for factor_col, factor_tag in [("momentum", "momentum"), ("value", "value")]:
                factor_col_scores = features[factor_col].unstack("symbol") if factor_col in features.columns else None
                if factor_col_scores is not None:
                    # Monthly long-short return: top half vs bottom half by z-score
                    fwd1 = make_forward_returns(month_end_closes, horizon=1)
                    # Reindex both to shared dates
                    shared = factor_col_scores.index.intersection(fwd1.index)
                    scores_m = factor_col_scores.loc[shared]
                    ret_m = fwd1.loc[shared]

                    ls_rets: list[float] = []
                    ls_dates: list = []
                    for date in shared:
                        s_row = scores_m.loc[date].dropna()
                        r_row = ret_m.loc[date].dropna()
                        common = s_row.index.intersection(r_row.index)
                        if len(common) < 6:
                            continue
                        med = s_row[common].median()
                        longs = s_row[common][s_row[common] >= med].index
                        shorts = s_row[common][s_row[common] < med].index
                        if len(longs) == 0 or len(shorts) == 0:
                            continue
                        ret_long = r_row[longs].mean()
                        ret_short = r_row[shorts].mean()
                        ls_rets.append(ret_long - ret_short)
                        ls_dates.append(date)

                    factor_rets_dict[factor_tag] = pd.Series(
                        ls_rets, index=pd.DatetimeIndex(ls_dates), name=factor_tag
                    )

            if factor_rets_dict:
                factor_rets_df = pd.DataFrame(factor_rets_dict)
                try:
                    attribution_result = factor_attribution(monthly_net_rets, factor_rets_df)
                    logger.info(
                        "Attribution: alpha=%.5f (t=%.3f), R²=%.3f, betas=%s",
                        attribution_result.get("alpha", 0),
                        attribution_result.get("alpha_tstat", 0),
                        attribution_result.get("r_squared", 0),
                        {k: f"{v:.3f}" for k, v in attribution_result.get("betas", {}).items()},
                    )
                except Exception as exc:
                    logger.warning("Attribution failed: %s", exc)

    results.attribution = attribution_result

    # ------------------------------------------------------------------
    # Step 7: Report figures + markdown
    # ------------------------------------------------------------------
    t0 = _banner("7 / 8  Report figures + markdown", t0)
    from alpharank.report.builder import ReportBuilder

    builder = ReportBuilder(output_dir)

    builder.fig_ic_comparison(comparison_table)
    builder.fig_ic_decay(ic_decay_by_model)
    builder.fig_monthly_ic(ic_series_by_model)
    builder.fig_equity_curves(backtest_results)

    planted_note = (
        "The alpha signal is **deliberately planted** using the formula:\n\n"
        "```\nalpha = IC_target * sigma_noise / sqrt(1 - IC_target^2)\n```\n\n"
        "Planted IC targets:\n"
        "- **Momentum**: IC_target = 0.06\n"
        "- **Value**: IC_target = 0.04\n\n"
        "Unplanted (negative controls, IC_target = 0):\n"
        "- Reversal, Volatility, Quality, Liquidity\n\n"
        "Survivorship and delist handling: OHLCV frames truncated at delist month "
        "(no NaN rows). Delist probability: 3% per year.\n\n"
        "> The models are *expected* to recover this planted signal. "
        "This is a methodology showcase, not a real-alpha claim."
    )

    results_md = output_dir / "RESULTS.md"
    builder.write_markdown(
        path=results_md,
        sections={
            "comparison_table": comparison_table,
            "backtest_table": backtest_summaries,
            "attribution": attribution_result,
            "planted_alpha_note": planted_note,
        },
    )
    logger.info("Results report written to: %s", results_md)

    # ------------------------------------------------------------------
    # Step 8: qbacktest TearsheetRenderer for LGBM strategy
    # ------------------------------------------------------------------
    t0 = _banner("8 / 8  Tearsheet (LGBM)", t0)
    if lgbm_model_name and lgbm_model_name in backtest_results:
        try:
            from qbacktest import TearsheetRenderer
            renderer = TearsheetRenderer(output_dir=output_dir / "figures")
            renderer.render(
                results=backtest_results[lgbm_model_name],
                title=f"AlphaRank — {lgbm_model_name} Decile Long-Short",
                filename="lgbm_tearsheet.png",
            )
            logger.info("Tearsheet saved to: %s/figures/lgbm_tearsheet.png", output_dir)
        except Exception as exc:
            logger.warning("Tearsheet render failed: %s", exc)

    # Print final summary to stdout
    _print_summary(comparison_table, backtest_summaries)

    total_time = time.time() - pipe_start
    logger.info("\nPipeline complete in %.1f seconds.", total_time)

    return results


# ---------------------------------------------------------------------------
# Print summary table to stdout
# ---------------------------------------------------------------------------

def _print_summary(
    comparison_table: pd.DataFrame,
    backtest_summaries: dict[str, dict],
) -> None:
    """Print the final summary table to stdout."""
    print("\n" + "=" * 72)
    print("ALPHARANK PIPELINE SUMMARY")
    print("=" * 72)

    if not comparison_table.empty:
        print(f"\n{'Model':<28} {'Mean IC':>8} {'ICIR':>7} {'NW-t':>7} "
              f"{'p':>7} {'GrShp':>7} {'NetShp':>7} {'CostBps':>8}")
        print("-" * 72)
        for _, row in comparison_table.iterrows():
            model = row["model"]
            mean_ic = row.get("mean_ic", float("nan"))
            icir_val = row.get("icir", float("nan"))
            nw_t = row.get("nw_tstat", float("nan"))
            p_val = row.get("p_value", float("nan"))
            bt = backtest_summaries.get(model, {})
            gr_shp = bt.get("gross_sharpe", float("nan"))
            net_shp = bt.get("net_sharpe", float("nan"))
            cost = bt.get("cost_bps", float("nan"))

            def _f(v, fmt=".4f"):
                return format(v, fmt) if v == v else "  nan"

            print(f"{model:<28} {_f(mean_ic):>8} {_f(icir_val, '.3f'):>7} "
                  f"{_f(nw_t, '.3f'):>7} {_f(p_val, '.3f'):>7} "
                  f"{_f(gr_shp, '.3f'):>7} {_f(net_shp, '.3f'):>7} {_f(cost, '.1f'):>8}")
        print("-" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlphaRank research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: n_assets=20, n_months=36, LGBM n_estimators=50")
    p.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    p.add_argument("--config",
                   default="configs/alpharank_config.yml",
                   help="YAML config file path")
    p.add_argument("--output-dir", default="reports",
                   help="Root output directory for figures and reports")
    # Hidden real-data opt-in — only available if explicitly passed
    p.add_argument("--real-data", action="store_true",
                   help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load YAML config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        logger.warning("Config not found at %s — using defaults", config_path)
        cfg = {}

    # Apply --quick block
    if args.quick:
        quick = cfg.get("quick", {})
        cfg.setdefault("data", {})
        cfg["data"]["n_assets"] = quick.get("n_assets", 20)
        cfg["data"]["n_months"] = quick.get("n_months", 36)
        cfg["_lgbm_n_estimators_override"] = 50
        # Reduce CV params so purge+embargo fits in the smaller train folds.
        # skfolio requires n_test_folds >= 2. With n_months=36, n_folds=5,
        # n_test_folds=2: train_folds=3, fold_size≈7 months, purge=1 < 7, valid.
        cfg["cv"] = {"n_folds": 5, "n_test_folds": 2, "purged_size": 1, "embargo_size": 0}
        logger.info("Quick mode: n_assets=%d, n_months=%d, LGBM n_estimators=50",
                    cfg["data"]["n_assets"], cfg["data"]["n_months"])

    # Apply --seed
    cfg.setdefault("data", {})["seed"] = args.seed

    # Hidden real-data path
    cfg["_use_real_data"] = args.real_data

    output_dir = Path(args.output_dir)
    run(cfg, output_dir=output_dir)

    sys.exit(0)


if __name__ == "__main__":
    main()
