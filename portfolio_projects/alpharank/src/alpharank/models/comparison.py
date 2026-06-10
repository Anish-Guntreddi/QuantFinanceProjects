"""Model comparison harness for the four baseline ranking models.

Evaluates every model in BASELINE_ORDER through the IDENTICAL PurgedCVEvaluator
protocol — no per-model evaluation shortcuts, no model-specific branches.

This structural uniformity is the implementation of ALR-04's "same protocol"
requirement: the code path from model instantiation to IC table row is a single
loop with zero conditionals on model type.

Usage
-----
    from alpharank.models.comparison import run_model_comparison
    from alpharank.validation.purged_cv import PurgedCVEvaluator

    evaluator = PurgedCVEvaluator()
    table, oos_frames = run_model_comparison(X, y, evaluator)

    # table: 4-row DataFrame(model, mean_ic, icir, nw_tstat, p_value, n_months)
    # oos_frames: {model_name: (date x symbol) score DataFrame}
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpharank.analytics.ic import icir, newey_west_ic_tstat
from alpharank.models import BASELINE_ORDER
from alpharank.validation.purged_cv import PurgedCVEvaluator


__all__ = ["run_model_comparison"]

logger = logging.getLogger(__name__)


def run_model_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    evaluator: PurgedCVEvaluator,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate all four baseline models with the identical CV protocol.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with MultiIndex (date, symbol).
    y : pd.Series
        Label series aligned to X (percentile ranks, from make_labels).
    evaluator : PurgedCVEvaluator
        Configured purged-CV evaluator.  The SAME evaluator instance is used
        for every model — one code path, zero model-specific branches.

    Returns
    -------
    table : pd.DataFrame
        4-row comparison table in BASELINE_ORDER with columns:
        [model, mean_ic, icir, nw_tstat, p_value, n_months]
    oos_frames : dict[str, pd.DataFrame]
        Per-model OOS score frames keyed by model name.  Each frame is
        (date x symbol) — the unstacked oos_scores from evaluator.evaluate(),
        suitable for downstream decile portfolio construction.

    Notes
    -----
    Cross-model ordering of IC values is NOT asserted.  With linearly planted
    alpha, LinearRegression may legitimately outperform LGBM because the planted
    signal is a linear combination of factors, and gradient-boosted trees do not
    necessarily dominate linear models on purely linear signals.  The comparison
    table reports factual OOS IC statistics; interpretation is left to the
    researcher.
    """
    rows: list[dict] = []
    oos_frames: dict[str, pd.DataFrame] = {}

    for model_cls in BASELINE_ORDER:
        # Instantiate fresh — no state shared between models
        model = model_cls()
        model_name = model.name

        # ONE code path for all four models — no per-model branches
        result = evaluator.evaluate(model, X, y)
        ic_series: pd.Series = result["ic_series"]
        oos_scores: pd.Series = result["oos_scores"]

        # Compute IC statistics
        mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
        icir_val = icir(ic_series)
        if len(ic_series) >= 2:
            _, nw_tstat, p_value = newey_west_ic_tstat(ic_series)
        else:
            nw_tstat, p_value = np.nan, np.nan

        rows.append({
            "model": model_name,
            "mean_ic": mean_ic,
            "icir": icir_val,
            "nw_tstat": nw_tstat,
            "p_value": p_value,
            "n_months": len(ic_series),
        })

        # Unstack OOS scores to (date x symbol) frame for decile construction
        oos_scores_nonan = oos_scores.dropna()
        if len(oos_scores_nonan) > 0:
            oos_wide = oos_scores_nonan.unstack(level="symbol")
            oos_wide.index.name = "date"
        else:
            oos_wide = pd.DataFrame()
        oos_frames[model_name] = oos_wide

        logger.debug(
            "Model %s: mean_IC=%.4f, ICIR=%.3f, NW t=%.3f, p=%.3f, n=%d",
            model_name, mean_ic, icir_val, nw_tstat, p_value, len(ic_series),
        )

    table = pd.DataFrame(rows)

    # Log the comparison table (run_pipeline will reuse this)
    _log_comparison_table(table)

    return table, oos_frames


def _log_comparison_table(table: pd.DataFrame) -> None:
    """Print the comparison table to stdout and log at INFO level."""
    header = f"{'Model':<28} {'Mean IC':>8} {'ICIR':>8} {'NW t':>8} {'p':>8} {'N':>6}"
    separator = "-" * len(header)
    lines = [separator, header, separator]
    for _, row in table.iterrows():
        lines.append(
            f"{row['model']:<28} "
            f"{row['mean_ic']:>8.4f} "
            f"{row['icir']:>8.3f} "
            f"{row['nw_tstat']:>8.3f} "
            f"{row['p_value']:>8.3f} "
            f"{row['n_months']:>6.0f}"
        )
    lines.append(separator)
    table_str = "\n".join(lines)
    print(table_str)
    logger.info("Model comparison table:\n%s", table_str)
