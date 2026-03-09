"""Backtesting notebook template — for core_research_backtesting/ (4 projects)."""

from __future__ import annotations
import nbformat as nbf
from .common_cells import (
    title_cell, environment_setup_cell, config_cell,
    data_acquisition_yfinance, performance_viz_cell,
    metrics_cell, sensitivity_cell, export_cell, summary_cell,
    monthly_heatmap_cell, get_ticker_for_project,
)


def _factor_analysis_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Factor construction (adapt to specific project)
returns = close.pct_change() if isinstance(close, pd.Series) else close.iloc[:, 0].pct_change()
price = close if isinstance(close, pd.Series) else close.iloc[:, 0]

# Example: momentum factor
lookback = PARAMS.get("lookback", 20)
factor = price.pct_change(lookback)

# Information Coefficient (rank correlation with forward returns)
fwd_periods = [1, 5, 10, 21]
print("Factor IC by forward period:")
for fwd in fwd_periods:
    fwd_ret = returns.shift(-fwd)
    valid = pd.DataFrame({"factor": factor, "fwd_ret": fwd_ret}).dropna()
    ic, pval = spearmanr(valid["factor"], valid["fwd_ret"])
    print(f"  {fwd:>3}d forward: IC = {ic:+.4f}  (p = {pval:.4f})")

# IC time series (rolling)
rolling_ic = factor.rolling(63).corr(returns.shift(-5))
print(f"\\nAvg rolling IC (63d): {rolling_ic.mean():.4f}")
print(f"IC hit rate (>0): {(rolling_ic > 0).mean():.2%}")
""")


def _backtest_execution_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Event-driven backtest
returns = close.pct_change() if isinstance(close, pd.Series) else close.iloc[:, 0].pct_change()
price = close if isinstance(close, pd.Series) else close.iloc[:, 0]

# Signal generation
lookback = PARAMS.get("lookback", 20)
signal = price.pct_change(lookback).shift(1)

# Position sizing: z-score of signal
zscore_window = PARAMS.get("zscore_window", 126)
z = (signal - signal.rolling(zscore_window).mean()) / signal.rolling(zscore_window).std()
positions = z.clip(-2, 2) / 2  # normalize to [-1, 1]

# Apply transaction costs
tc_bps = PARAMS.get("transaction_cost_bps", 5) / 10000
turnover = positions.diff().abs()
tc = turnover * tc_bps

strategy_returns_raw = (positions.shift(1) * returns - tc).dropna()

# Build equity curve
equity_curve = (1 + strategy_returns_raw).cumprod()
benchmark_equity = (1 + returns.loc[equity_curve.index]).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"Avg daily turnover: {turnover.mean():.4f}")
print(f"Total transaction costs: {tc.sum():.4f}")
""")


def _walk_forward_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""# Walk-forward validation
import numpy as np

n = len(strategy_returns_raw)
print(f"Available backtest rows: {n}")

# Adaptive window: use at most 40% of data as training, step = 25% of window
window = min(126, max(21, int(n * 0.40)))
step   = max(10,  int(window * 0.25))
print(f"Window: {window} days, Step: {step} days")

walk_forward_sharpes = []
for start in range(0, n - window - step, step):
    train_end = start + window
    test_end  = min(train_end + step, n)
    test_rets = strategy_returns_raw.iloc[train_end:test_end]
    if len(test_rets) > 5:
        wf_sharpe = (
            test_rets.mean() / test_rets.std() * np.sqrt(252)
            if test_rets.std() > 0 else 0.0
        )
        walk_forward_sharpes.append(wf_sharpe)

if walk_forward_sharpes:
    print(f"Walk-forward Sharpe (mean): {np.mean(walk_forward_sharpes):.4f}")
    print(f"Walk-forward Sharpe (std):  {np.std(walk_forward_sharpes):.4f}")
    print(f"Positive periods: {sum(s > 0 for s in walk_forward_sharpes)}/{len(walk_forward_sharpes)}")
else:
    print(f"Not enough data for walk-forward (need >{window + step} rows, have {n}).")
""")


def build_backtesting_notebook(card: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    params = {p["name"]: p["default"] for p in card.get("interactive_params", [])}

    nb.cells = [
        title_cell(card["title"], "Core Research & Backtesting", card.get("long_description", card.get("short_description", "")), card["project_id"]),
        environment_setup_cell(requires_gpu=False),
        config_cell(params),
        data_acquisition_yfinance(get_ticker_for_project(card["project_id"])),
        nbf.v4.new_markdown_cell("## Factor Analysis / Signal Construction"),
        _factor_analysis_cell(),
        nbf.v4.new_markdown_cell("## Backtest Execution"),
        _backtest_execution_cell(),
        performance_viz_cell(),
        metrics_cell(),
        monthly_heatmap_cell(),
        nbf.v4.new_markdown_cell("## Walk-Forward Validation"),
        _walk_forward_cell(),
        sensitivity_cell(card.get("interactive_params", [{"name": "lookback"}])[0].get("name", "lookback")),
        export_cell(card["project_id"]),
        summary_cell(card["title"]),
    ]
    return nb
