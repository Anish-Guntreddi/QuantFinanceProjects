"""Shared notebook cell builders used by all domain templates.

Each function returns a list of nbformat cell dicts (code or markdown).
"""

from __future__ import annotations
import nbformat as nbf


# ---------------------------------------------------------------------------
# Per-project ticker map — gives each notebook distinct market data
# ---------------------------------------------------------------------------
_PROJECT_TICKER_MAP: dict[str, str | list] = {
    # Intraday — each uses a different asset class / sector
    "intraday_01_momentum":              "SPY",
    "intraday_02_mean_reversion":        "GLD",
    "intraday_03_stat_arb":              ["SPY", "QQQ"],
    "intraday_04_momentum_value":        "IWM",
    "intraday_05_options":               "AAPL",
    "intraday_06_execution_tca":         "QQQ",
    "intraday_07_ml_strategy":           "TSLA",
    "intraday_08_regime_detection":      "SPY",
    "intraday_09_portfolio_construction": ["SPY", "QQQ", "IWM", "GLD", "TLT"],
    # Research backtesting — factor/event/arb/vol flavors
    "research_01_factor_toolkit":        ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "research_02_event_backtester":      "QQQ",
    "research_03_stat_arb":              ["SPY", "QQQ"],
    "research_04_vol_surface":           "SPY",
    # Risk engineering — portfolio-level multi-asset
    "risk_01_portfolio_optimization":    ["SPY", "TLT", "GLD", "IWM", "QQQ"],
    "risk_02_reproducibility":           ["SPY", "TLT", "GLD"],
    # ML trading — different underlying per model type
    "ml_01_regime_detection":            "SPY",
    "ml_02_lstm_transformer":            "QQQ",
}


def get_ticker_for_project(project_id: str, fallback: str | list = "SPY") -> str | list:
    """Return the canonical ticker(s) for a given project_id."""
    return _PROJECT_TICKER_MAP.get(project_id, fallback)


# ---------------------------------------------------------------------------
# Ticker extraction helper
# ---------------------------------------------------------------------------
def _extract_tickers(data_source: str, default: str | list = "SPY") -> str | list:
    """Parse ticker(s) from a card's data_source field.

    Handles formats like:
      - "yfinance"          -> default
      - "synthetic"         -> default
      - "yfinance:SPY"      -> "SPY"
      - "yfinance:SPY,AGG"  -> ["SPY", "AGG"]
      - "yfinance(SPY,AGG)" -> ["SPY", "AGG"]
    """
    if not data_source:
        return default
    src = data_source.strip().lower()
    if src in ("synthetic", "yfinance", ""):
        return default
    if ":" in data_source:
        part = data_source.split(":", 1)[1].strip()
    elif "(" in data_source and ")" in data_source:
        part = data_source[data_source.index("(") + 1 : data_source.index(")")].strip()
    else:
        return default
    tickers = [t.strip().upper() for t in part.split(",") if t.strip()]
    if not tickers:
        return default
    return tickers if len(tickers) > 1 else tickers[0]


# ---------------------------------------------------------------------------
# Cell 1: Title & Overview
# ---------------------------------------------------------------------------
def title_cell(title: str, category: str, description: str, project_id: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        f"# {title}\n\n"
        f"**Category:** {category}  \n"
        f"**Project ID:** `{project_id}`  \n\n"
        f"{description}\n"
    )


# ---------------------------------------------------------------------------
# Cell 2: Environment Setup
# ---------------------------------------------------------------------------
def environment_setup_cell(requires_gpu: bool = False) -> nbf.NotebookNode:
    gpu_note = (
        'print("  [GPU recommended for this project — deep learning / RL]")'
        if requires_gpu else ""
    )
    code = f"""import platform, sys, warnings
warnings.filterwarnings("ignore")

# ── Environment info ────────────────────────────────────────────────────────
env_info = {{"os": platform.system(), "python": platform.python_version()}}

# Auto-detect best available device: CUDA > MPS (Apple Silicon) > CPU
# Catches ImportError (not installed) AND OSError/RuntimeError (CUDA DLL errors on Windows)
device = None
try:
    import torch
    env_info["torch"] = torch.__version__
    if torch.cuda.is_available():
        device = torch.device("cuda")
        env_info["device"] = f"CUDA ({{torch.cuda.get_device_name(0)}})"
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        env_info["device"] = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        env_info["device"] = "CPU"
except Exception as _e:
    env_info["device"] = f"CPU (torch unavailable: {{type(_e).__name__}})"
{gpu_note}
# Core scientific stack — import individually so a missing package doesn't block the rest
for _pkg in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
    try:
        _mod = __import__(_pkg)
        env_info[_pkg] = getattr(_mod, "__version__", "installed")
    except ImportError:
        env_info[_pkg] = "not installed"

for k, v in env_info.items():
    print(f"  {{k:>12}}: {{v}}")
"""
    return nbf.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Cell 3: Configuration
# ---------------------------------------------------------------------------
def config_cell(params: dict, start: str = "2022-01-01", end: str = "2024-12-31",
                seed: int = 42) -> nbf.NotebookNode:
    params_str = ", ".join(f'"{k}": {repr(v)}' for k, v in params.items())
    code = f"""import numpy as np

# Reproducibility
SEED = {seed}
np.random.seed(SEED)

# Strategy parameters
PARAMS = {{{params_str}}}

# Backtest period
START_DATE = "{start}"
END_DATE = "{end}"
BENCHMARK = "SPY"

print("Configuration loaded:")
for k, v in PARAMS.items():
    print(f"  {{k:>25}}: {{v}}")
"""
    return nbf.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Cell 4: Data Acquisition (yfinance or synthetic)
# ---------------------------------------------------------------------------
def data_acquisition_yfinance(tickers: list[str] | str = "SPY") -> nbf.NotebookNode:
    if isinstance(tickers, list):
        tickers_repr = repr(tickers)
    else:
        tickers_repr = repr(tickers)

    code = f"""import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = {tickers_repr}
data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

# Handle MultiIndex columns (yfinance >=0.2.x returns MultiIndex for multiple tickers)
if isinstance(data.columns, pd.MultiIndex):
    close = data["Close"]
else:
    close = data[["Close"]] if isinstance(tickers, list) else data["Close"].to_frame()

# Drop rows where all Close values are NaN
close = close.dropna(how="all")
data = data.loc[close.index]

if close.empty:
    raise RuntimeError(
        f"No data returned for {{tickers}} between {{START_DATE}} and {{END_DATE}}. "
        "Check ticker symbols and date range."
    )

print(f"Data shape: {{data.shape}}")
print(f"Date range: {{close.index[0].strftime('%Y-%m-%d')}} to {{close.index[-1].strftime('%Y-%m-%d')}}")
print(f"\\nSummary statistics:")
print(close.describe().round(4))

# Price chart
fig, ax = plt.subplots(figsize=(14, 5))
(close / close.iloc[0]).plot(ax=ax, linewidth=1.5)
ax.set_title("Normalized Prices", fontsize=14)
ax.set_ylabel("Normalized Price")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
"""
    return nbf.v4.new_code_cell(code)


def data_acquisition_synthetic() -> nbf.NotebookNode:
    code = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic market data
n_steps = 100_000
dt = 1.0  # tick-level
rng = np.random.default_rng(SEED)

# Mid-price: random walk with drift
returns = rng.normal(0, 0.0002, n_steps)
mid_prices = 100.0 * np.exp(np.cumsum(returns))

# Synthetic order flow
bid_sizes = rng.exponential(100, n_steps).astype(int)
ask_sizes = rng.exponential(100, n_steps).astype(int)
spread_bps = rng.exponential(3, n_steps) + 1.0

data = pd.DataFrame({
    "mid_price": mid_prices,
    "bid_size": bid_sizes,
    "ask_size": ask_sizes,
    "spread_bps": spread_bps,
    "imbalance": (bid_sizes - ask_sizes) / (bid_sizes + ask_sizes),
})

print(f"Synthetic data: {len(data):,} ticks")
print(data.describe().round(4))

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(data["mid_price"].values[:5000], linewidth=0.5)
axes[0].set_title("Mid Price (first 5K ticks)")
axes[1].bar(range(5000), data["imbalance"].values[:5000], width=1, alpha=0.5)
axes[1].set_title("Order Book Imbalance")
plt.tight_layout()
plt.show()
"""
    return nbf.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Cell 8: Performance Visualization
# ---------------------------------------------------------------------------
def performance_viz_cell() -> nbf.NotebookNode:
    code = """import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 2]})

# Equity curve
axes[0].plot(equity_curve.index, equity_curve.values, label="Strategy", linewidth=1.5, color="#f59e0b")
if benchmark_equity is not None:
    axes[0].plot(benchmark_equity.index, benchmark_equity.values, label="Benchmark",
                 linewidth=1, linestyle="--", color="#6b7280")
axes[0].set_title("Equity Curve", fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Drawdown
drawdown = equity_curve / equity_curve.cummax() - 1
axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="#ef4444")
axes[1].set_title("Drawdown", fontsize=12)
axes[1].grid(True, alpha=0.3)

# Rolling Sharpe (63-day)
rolling_ret = equity_curve.pct_change()
rolling_sharpe = (rolling_ret.rolling(63).mean() / rolling_ret.rolling(63).std()) * np.sqrt(252)
axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1, color="#3b82f6")
axes[2].axhline(0, color="#6b7280", linewidth=0.5)
axes[2].set_title("Rolling Sharpe (63-day)", fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""
    return nbf.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Cell 9: Risk & Performance Metrics
# ---------------------------------------------------------------------------
def metrics_cell() -> nbf.NotebookNode:
    code = """import numpy as np

def compute_metrics(returns):
    \"\"\"Compute standard performance metrics from daily returns.\"\"\"
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / max(len(returns[returns != 0]), 1)
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "annualized_vol": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": min(profit_factor, 99.99),
        "total_trades": len(returns[returns != 0]),
    }

strategy_returns = equity_curve.pct_change().dropna()
metrics = compute_metrics(strategy_returns)

print("=" * 50)
print("PERFORMANCE METRICS")
print("=" * 50)
for k, v in metrics.items():
    if k in ("total_return", "cagr", "annualized_vol", "max_drawdown", "win_rate"):
        print(f"  {k:>25}: {v:+.2%}")
    elif k == "total_trades":
        print(f"  {k:>25}: {int(v):,}")
    else:
        print(f"  {k:>25}: {v:.4f}")
"""
    return nbf.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Cell 10: Sensitivity Analysis
# ---------------------------------------------------------------------------
def sensitivity_cell(param_name: str, param_range: str | None = None) -> nbf.NotebookNode:
    # Pick a sensible sweep range based on the param type
    if param_range is None:
        if any(k in param_name for k in ("spread", "bps", "cost", "impact")):
            param_range = "[1, 3, 5, 8, 10, 15, 20]"
        elif any(k in param_name for k in ("stop", "loss", "drawdown")):
            param_range = "[0.005, 0.01, 0.02, 0.03, 0.05, 0.10]"
        else:
            param_range = "range(5, 65, 5)"  # lookback / window style

    code = f"""# Parameter sensitivity analysis — inline backtest sweep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

param_values = list({param_range})
sharpes, max_dds = [], []

# Resolve price / returns from whatever data this notebook loaded
# yfinance notebooks define `close`; HFT/synthetic notebooks define `data` with various price columns
if "close" in dir() and close is not None:
    _price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
elif "data" in dir() and hasattr(data, "columns"):
    _pcol = next((c for c in ("mid_price", "price_a", "fast_price", "Close") if c in data.columns), None)
    if _pcol is not None:
        _price = pd.Series(data[_pcol].values, dtype=float)
    else:
        _price = pd.Series(data.iloc[:, 0].values, dtype=float)
else:
    raise RuntimeError("No price series found. Expected 'close' or a DataFrame named 'data'.")
_price   = _price.ffill()
_returns = _price.pct_change()

for val in param_values:
    try:
        if "{param_name}" in ("spread_bps", "transaction_cost_bps", "impact_alpha"):
            # Cost sensitivity: fix lookback=20 momentum, vary cost
            _sig  = _price.pct_change(20).shift(1).clip(-1, 1)
            _tc   = _sig.diff().abs() * (float(val) / 10000)
            _rets = (_sig * _returns - _tc).dropna()
        elif "{param_name}" in ("stop_loss", "max_drawdown_limit"):
            # Stop-loss sensitivity: apply to already-computed returns
            _base = strategy_returns_raw if "strategy_returns_raw" in dir() else \
                    (_price.pct_change(20).shift(1).clip(-1, 1) * _returns).dropna()
            _cum = (1 + _base.fillna(0)).cumprod()
            _dd  = _cum / _cum.cummax() - 1
            _rets = _base.copy()
            _rets[_dd < -float(val)] = 0.0
        else:
            # Lookback / window sweep: momentum signal
            lb    = max(int(val), 2)
            _sig  = _price.pct_change(lb).shift(1).clip(-1, 1)
            _tc   = _sig.diff().abs() * PARAMS.get("transaction_cost_bps", 5) / 10000
            _rets = (_sig * _returns - _tc).dropna()

        if len(_rets) > 20 and _rets.std() > 0:
            sharpes.append(_rets.mean() / _rets.std() * np.sqrt(252))
            _cum = (1 + _rets).cumprod()
            max_dds.append((_cum / _cum.cummax() - 1).min())
        else:
            sharpes.append(np.nan)
            max_dds.append(np.nan)
    except Exception as e:
        sharpes.append(np.nan)
        max_dds.append(np.nan)

fig, ax1 = plt.subplots(figsize=(10, 5))
color_sharpe = "#f59e0b"
color_dd     = "#ef4444"

valid_mask = [not np.isnan(s) for s in sharpes]
ax1.plot(param_values, sharpes, "o-", color=color_sharpe, label="Sharpe Ratio")
ax1.set_xlabel("{param_name}")
ax1.set_ylabel("Sharpe Ratio", color=color_sharpe)
ax1.tick_params(axis="y", labelcolor=color_sharpe)
ax1.axhline(0, color="#6b7280", linewidth=0.5, linestyle=":")

ax2 = ax1.twinx()
ax2.plot(param_values, max_dds, "s--", color=color_dd, alpha=0.7, label="Max DD")
ax2.set_ylabel("Max Drawdown", color=color_dd)
ax2.tick_params(axis="y", labelcolor=color_dd)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
ax1.grid(True, alpha=0.2)

plt.title(f"Sensitivity Analysis: {param_name}")
fig.tight_layout()
plt.show()

# Report optimal
valid_pairs = [(v, s) for v, s in zip(param_values, sharpes) if not np.isnan(s)]
if valid_pairs:
    best_val, best_sharpe = max(valid_pairs, key=lambda x: x[1])
    print(f"Optimal {param_name}: {{best_val}}  (Sharpe: {{best_sharpe:+.4f}})")
    print(f"Current {param_name}: {{PARAMS.get('{param_name}', 'not set')}}")
"""
    return nbf.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Cell 11: Export Results
# ---------------------------------------------------------------------------
def export_cell(project_id: str) -> nbf.NotebookNode:
    code = f"""import json
from datetime import datetime

# Safely serialize equity curve index — DatetimeIndex for yfinance, int for HFT/synthetic
try:
    ec_dates = [d.strftime("%Y-%m-%d") for d in equity_curve.index]
except AttributeError:
    ec_dates = [str(i) for i in equity_curve.index]

# Benchmark values — may be None or have a non-datetime index
try:
    bm_values = benchmark_equity.values.tolist() if benchmark_equity is not None else []
except Exception:
    bm_values = []

# Monthly returns — only works with DatetimeIndex
try:
    monthly = strategy_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_dict = {{d.strftime("%Y-%m"): float(v) for d, v in monthly.items()}}
except Exception:
    monthly_dict = {{}}

# Export results for portfolio app
results_export = {{
    "project_id": "{project_id}",
    "timestamp": datetime.now().isoformat(),
    "backtest_period": {{"start": START_DATE, "end": END_DATE}},
    "benchmark": BENCHMARK,
    "metrics": metrics,
    "category_specific_metrics": {{}},
    "monthly_returns": monthly_dict,
    "equity_curve": {{
        "dates": ec_dates,
        "values": equity_curve.values.tolist(),
        "benchmark_values": bm_values,
    }},
    "parameter_sensitivity": [],
}}

with open("results.json", "w") as f:
    json.dump(results_export, f, indent=2, default=str)
print(f"Results exported to results.json")
"""
    return nbf.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Cell 12: Summary
# ---------------------------------------------------------------------------
def summary_cell(title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        f"## Summary\n\n"
        f"### {title}\n\n"
        f"**Key Findings:**\n"
        f"- *Add your analysis findings here after running the notebook*\n\n"
        f"**Limitations:**\n"
        f"- Backtest uses historical data which may not reflect future conditions\n"
        f"- Transaction costs and slippage are approximated\n"
        f"- No live market microstructure effects\n\n"
        f"**Production Considerations:**\n"
        f"- Real-time data feed integration required\n"
        f"- Position sizing and risk limits must be calibrated\n"
        f"- Monitoring and alerting infrastructure needed\n"
    )


# ---------------------------------------------------------------------------
# Monthly heatmap helper
# ---------------------------------------------------------------------------
def monthly_heatmap_cell() -> nbf.NotebookNode:
    code = """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Monthly returns heatmap — requires DatetimeIndex; skipped for synthetic/HFT data
try:
    monthly = strategy_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
except Exception:
    print("Monthly heatmap skipped: equity curve does not have a DatetimeIndex (synthetic data).")
    monthly = None

if monthly is not None:
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot = pivot.reindex(columns=range(1, 13))  # ensure all 12 months present
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly Returns (%)", fontsize=14)

    # Text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(val) < 0.03 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()
"""
    return nbf.v4.new_code_cell(code)
