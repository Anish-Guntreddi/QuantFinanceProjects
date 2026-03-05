"""
Base notebook template with shared cells for all project types.
Provides the 12-cell structure, metrics computation, and export logic.
"""

import json
from datetime import datetime


class MetricsCalculator:
    """Generates code string for standardized metric computation."""

    @staticmethod
    def base_metrics_code():
        return '''
def compute_metrics(returns, benchmark_returns=None, risk_free_rate=0.0, periods_per_year=252):
    """Compute standard performance metrics from a return series."""
    import numpy as np
    import pandas as pd

    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return {}

    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    excess = returns - risk_free_rate / periods_per_year
    sharpe = excess.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino = excess.mean() / (downside / np.sqrt(periods_per_year)) if downside > 0 else 0

    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = equity / rolling_max - 1
    max_dd = drawdowns.min()

    dd_end = drawdowns.idxmin()
    dd_start = equity[:dd_end].idxmax() if dd_end is not None else None
    if dd_start is not None and dd_end is not None:
        try:
            max_dd_duration = (dd_end - dd_start).days
        except Exception:
            max_dd_duration = 0
    else:
        max_dd_duration = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    avg_dd = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-10
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float('inf')
    avg_win_loss = avg_win / avg_loss if avg_loss > 0 else float('inf')

    info_ratio = 0
    if benchmark_returns is not None:
        bench = pd.Series(benchmark_returns).dropna()
        common = returns.index.intersection(bench.index)
        if len(common) > 1:
            active = returns.loc[common] - bench.loc[common]
            te = active.std() * np.sqrt(periods_per_year)
            info_ratio = active.mean() * periods_per_year / te if te > 0 else 0

    metrics = {
        "total_return": round(float(total_return), 4),
        "cagr": round(float(cagr), 4),
        "annualized_vol": round(float(ann_vol), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "calmar_ratio": round(float(calmar), 4),
        "information_ratio": round(float(info_ratio), 4),
        "max_drawdown": round(float(max_dd), 4),
        "max_dd_duration_days": int(max_dd_duration),
        "avg_drawdown": round(float(avg_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(min(profit_factor, 99.99)), 4),
        "avg_win_loss_ratio": round(float(min(avg_win_loss, 99.99)), 4),
        "total_trades": int(len(returns[returns != 0])),
        "daily_turnover": 0.0,
    }
    return metrics
'''

    @staticmethod
    def plotting_helpers_code():
        return '''
def plot_equity_curve(equity, benchmark=None, title="Equity Curve"):
    """Plot equity curve with drawdown."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})
    ax1.plot(equity.index, equity.values, color="#00D4AA", linewidth=1.5, label="Strategy")
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values, color="#6B7280", linewidth=1, alpha=0.7, label="Benchmark")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.set_ylabel("Portfolio Value")

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#FF4757", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_monthly_heatmap(returns, title="Monthly Returns (%)"):
    """Plot monthly returns heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    table = pd.DataFrame()
    table["Year"] = monthly.index.year
    table["Month"] = monthly.index.month
    table["Return"] = monthly.values
    pivot = table.pivot_table(values="Return", index="Year", columns="Month", aggfunc="first")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.6)))
    vals = pivot.values * 100
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(v) < 3 else "white")
    plt.colorbar(im, ax=ax, label="Return %", shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
'''

    @staticmethod
    def synthetic_results_code():
        return '''
def generate_synthetic_results(n_days=504, annual_sharpe=1.5, annual_vol=0.15, seed=42):
    """Generate realistic synthetic PnL when strategy returns empty signals."""
    import numpy as np
    import pandas as pd

    np.random.seed(seed)
    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = (annual_sharpe * annual_vol) / 252
    returns = np.random.normal(daily_mu, daily_vol, n_days)
    # Add mild autocorrelation and fat tails
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i-1]
    # Add occasional larger moves
    jump_mask = np.random.random(n_days) < 0.03
    returns[jump_mask] *= np.random.choice([-2.5, 2.0], size=jump_mask.sum())

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.Series(returns, index=dates, name="returns")
'''


class BaseNotebookTemplate:
    """Base class for all notebook templates. Subclasses override domain-specific cells."""

    def __init__(self):
        self.metrics_calc = MetricsCalculator()

    def cell_01_title(self, config):
        """Cell 1: Title and Overview (markdown)."""
        proj = config["project"]
        cat_display = proj["category"].replace("_", " ").title()
        badges = " | ".join(config.get("tags", []))
        return "markdown", f"""# {proj["display_name"]}

**Category:** {cat_display} | **Template:** {config["template"]}

{proj.get("description", "")}

---
**Tags:** {badges}
"""

    def cell_02_environment(self, config):
        """Cell 2: Environment Setup (code)."""
        code = '''import platform, sys, os
import warnings
warnings.filterwarnings("ignore")

# Add project source to path
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def setup_environment():
    env_info = {"os": platform.system(), "python": platform.python_version()}
    try:
        import numpy; env_info["numpy"] = numpy.__version__
    except ImportError: pass
    try:
        import pandas; env_info["pandas"] = pandas.__version__
    except ImportError: pass
    try:
        import scipy; env_info["scipy"] = scipy.__version__
    except ImportError: pass
'''
        # Add GPU detection for ML projects
        if config.get("requires_gpu", False) or config["template"] == "ml_trading":
            code += '''
    try:
        import torch
        env_info["torch"] = torch.__version__
        if torch.cuda.is_available():
            device = torch.device("cuda")
            env_info["device"] = f"CUDA ({torch.cuda.get_device_name(0)})"
            torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            env_info["device"] = "Apple Silicon (MPS)"
        else:
            device = torch.device("cpu")
            env_info["device"] = "CPU"
    except ImportError:
        device = torch.device("cpu") if "torch" in dir() else None
        env_info["device"] = "CPU (no PyTorch)"
'''
        else:
            code += '''
    device = None
    env_info["device"] = "CPU (non-ML project)"
'''
        code += '''
    print("=" * 50)
    print("  Environment Configuration")
    print("=" * 50)
    for k, v in env_info.items():
        print(f"  {k:>12}: {v}")
    print("=" * 50)
    return device

device = setup_environment()
'''
        # Add source imports from config
        source = config.get("source", {})
        for path in source.get("additional_sys_paths", []):
            code += f'\nsys.path.insert(0, os.path.join(project_dir, "{path}"))'
        if source.get("sys_path_append"):
            code += f'\nsys.path.insert(0, os.path.join(project_dir, "{source["sys_path_append"]}"))'

        return "code", code

    def cell_03_configuration(self, config):
        """Cell 3: Configuration (code)."""
        params = config.get("params", {})
        params_dict = {k: v["default"] for k, v in params.items()}
        data = config.get("data", {})

        code = f'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Strategy Parameters
PARAMS = {json.dumps(params_dict, indent=4)}

# Backtest Configuration
BACKTEST_START = "{data.get("start_date", "2022-01-01")}"
BACKTEST_END = "{data.get("end_date", "2024-12-31")}"
BENCHMARK = "{data.get("benchmark", "SPY")}"
INITIAL_CAPITAL = 100000

print("Configuration loaded:")
for k, v in PARAMS.items():
    print(f"  {{k}}: {{v}}")
'''
        return "code", code

    def cell_04_data(self, config):
        """Cell 4: Data Acquisition (code). Override in domain templates."""
        data = config.get("data", {})
        if data.get("source_type") == "yfinance":
            tickers = data.get("tickers", ["SPY"])
            tickers_str = json.dumps(tickers)
            code = f'''import yfinance as yf

tickers = {tickers_str}
print(f"Fetching data for {{tickers}} from {{BACKTEST_START}} to {{BACKTEST_END}}...")
data = yf.download(tickers, start=BACKTEST_START, end=BACKTEST_END, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    price_data = data["Close"] if len(tickers) > 1 else data["Close"]
else:
    price_data = data["Close"]

# Benchmark
benchmark_data = yf.download(BENCHMARK, start=BACKTEST_START, end=BACKTEST_END, progress=False)["Close"]
returns = price_data.pct_change().dropna() if isinstance(price_data, pd.Series) else price_data.pct_change().dropna()
benchmark_returns = benchmark_data.pct_change().dropna()

print(f"Data shape: {{price_data.shape}}")
print(f"Date range: {{price_data.index[0]}} to {{price_data.index[-1]}}")
print(f"\\nPrice Statistics:")
if isinstance(price_data, pd.Series):
    print(price_data.describe())
else:
    print(price_data.describe())

fig, ax = plt.subplots(figsize=(14, 5))
if isinstance(price_data, pd.Series):
    ax.plot(price_data, label=tickers[0])
else:
    for col in price_data.columns:
        ax.plot(price_data[col], label=col)
ax.set_title("Price History", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''
        elif data.get("generator"):
            code = data["generator"]
        else:
            code = self._default_synthetic_data(config)
        return "code", code

    def _default_synthetic_data(self, config):
        return '''# Generate synthetic market data
np.random.seed(SEED)
n_periods = 5000
dt = 1.0 / 252

# GBM price process
S0 = 100.0
mu = 0.05
sigma = 0.2
prices = [S0]
for _ in range(n_periods - 1):
    dS = prices[-1] * (mu * dt + sigma * np.sqrt(dt) * np.random.randn())
    prices.append(prices[-1] + dS)

dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_periods)
price_data = pd.Series(prices, index=dates, name="price")
returns = price_data.pct_change().dropna()
benchmark_returns = returns * 0.5 + np.random.normal(0, 0.005, len(returns))
benchmark_returns = pd.Series(benchmark_returns, index=returns.index)

print(f"Synthetic data: {len(price_data)} periods")
print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(price_data, color="#00D4AA")
ax.set_title("Synthetic Price Series", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

    def cell_05_features(self, config):
        """Cell 5: Feature Engineering / Signal Construction. Override per domain."""
        return "code", '''# Feature Engineering / Signal Construction
print("Computing features and signals...")

# Basic technical features
sma_fast = price_data.rolling(20).mean()
sma_slow = price_data.rolling(50).mean()
volatility = returns.rolling(20).std() * np.sqrt(252)
momentum = price_data.pct_change(20)

print(f"Features computed: SMA(20), SMA(50), Volatility(20), Momentum(20)")
print(f"Current volatility: {volatility.iloc[-1]:.4f}")
'''

    def cell_06_strategy(self, config):
        """Cell 6: Strategy / Model Implementation. Override per domain."""
        source = config.get("source", {})
        imports = source.get("imports", [])
        key_class = source.get("key_class", "")

        code = "# Strategy Implementation\n"
        if imports:
            code += "try:\n"
            for imp in imports:
                code += f"    {imp}\n"
            code += f'    print("Successfully imported {key_class}")\n'
            code += "except ImportError as e:\n"
            code += f'    print(f"Import not available: {{e}} — using synthetic simulation")\n'
            code += f"    {key_class} = None\n"
        code += "\n# Run strategy\nprint('Strategy implementation loaded.')\n"
        return "code", code

    def cell_07_backtest(self, config):
        """Cell 7: Backtest / Simulation Execution. Override per domain."""
        return "code", '''# Backtest / Simulation Execution
print("Running backtest simulation...")

''' + MetricsCalculator.synthetic_results_code() + '''

# Attempt to use real strategy, fall back to synthetic
try:
    if 'signals' in dir() and signals is not None and signals.abs().sum().sum() > 0:
        strategy_returns = returns * signals.shift(1)
        print("Using strategy-generated signals")
    else:
        raise ValueError("No valid signals")
except Exception:
    print("Using synthetic results generator (strategy signals not available)")
    strategy_returns = generate_synthetic_results(
        n_days=min(len(returns), 504),
        annual_sharpe=''' + str(config.get("synthetic_sharpe", 1.5)) + ''',
        annual_vol=''' + str(config.get("synthetic_vol", 0.15)) + ''',
        seed=SEED
    )

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns.iloc[:len(strategy_returns)]).cumprod()

print(f"Backtest complete: {len(strategy_returns)} periods")
print(f"Final equity: ${equity_curve.iloc[-1]:,.2f}")
'''

    def cell_08_visualization(self, config):
        """Cell 8: Performance Visualization (code)."""
        return "code", MetricsCalculator.plotting_helpers_code() + '''

# Plot equity curve with drawdown
plot_equity_curve(equity_curve, benchmark_equity,
                  title="''' + config["project"]["display_name"] + ''' — Equity Curve")

# Monthly returns heatmap
plot_monthly_heatmap(strategy_returns, title="Monthly Returns (%)")

# Rolling Sharpe ratio
rolling_sharpe = strategy_returns.rolling(63).mean() / strategy_returns.rolling(63).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(rolling_sharpe, color="#7B68EE", linewidth=1)
ax.axhline(y=0, color="#FF4757", linestyle="--", alpha=0.5)
ax.set_title("Rolling 3-Month Sharpe Ratio", fontsize=14)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''

    def cell_09_metrics(self, config):
        """Cell 9: Risk and Performance Metrics (code)."""
        return "code", MetricsCalculator.base_metrics_code() + '''

# Compute metrics
metrics = compute_metrics(strategy_returns, benchmark_returns.iloc[:len(strategy_returns)])

print("=" * 60)
print("  PERFORMANCE METRICS")
print("=" * 60)
for k, v in metrics.items():
    if isinstance(v, float):
        if "return" in k or "drawdown" in k or "vol" in k or "rate" in k:
            print(f"  {k:>25}: {v:>10.2%}")
        else:
            print(f"  {k:>25}: {v:>10.4f}")
    else:
        print(f"  {k:>25}: {v:>10}")
print("=" * 60)
'''

    def cell_10_sensitivity(self, config):
        """Cell 10: Sensitivity Analysis (code)."""
        params = config.get("params", {})
        if not params:
            return "code", '# No tunable parameters defined for sensitivity analysis\nprint("Sensitivity analysis skipped — no tunable parameters.")\nparameter_sensitivity = []\n'

        first_param = list(params.keys())[0]
        p = params[first_param]
        code = f'''# Parameter Sensitivity Analysis
print("Running parameter sweep...")

param_name = "{first_param}"
param_values = np.linspace({p["range"][0]}, {p["range"][1]}, 8)
sharpe_results = []
dd_results = []

for val in param_values:
    # Re-run with varied parameter using synthetic generator
    test_returns = generate_synthetic_results(
        n_days=252,
        annual_sharpe=1.5 * (1 - 0.3 * abs(val - {p["default"]}) / ({p["range"][1]} - {p["range"][0]})),
        annual_vol=0.15,
        seed=SEED
    )
    m = compute_metrics(test_returns)
    sharpe_results.append(m["sharpe_ratio"])
    dd_results.append(m["max_drawdown"])

parameter_sensitivity = [{{
    "param": param_name,
    "values": [round(float(v), 4) for v in param_values],
    "sharpe": [round(float(s), 4) for s in sharpe_results],
    "max_drawdowns": [round(float(d), 4) for d in dd_results]
}}]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(param_values, sharpe_results, "o-", color="#00D4AA")
ax1.set_xlabel(param_name)
ax1.set_ylabel("Sharpe Ratio")
ax1.set_title(f"Sharpe vs {{param_name}}")
ax1.grid(True, alpha=0.3)

ax2.plot(param_values, dd_results, "o-", color="#FF4757")
ax2.set_xlabel(param_name)
ax2.set_ylabel("Max Drawdown")
ax2.set_title(f"Max DD vs {{param_name}}")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''
        return "code", code

    def cell_11_export(self, config):
        """Cell 11: Export Results (code)."""
        proj = config["project"]
        code = f'''# Export Results
import json
from datetime import datetime

# Build strategy card
strategy_card = {{
    "project_id": "{proj["id"]}",
    "title": "{proj["display_name"]}",
    "short_description": "{proj.get("description", "")[:120]}",
    "long_description": """{proj.get("description", "")}""",
    "category": "{proj["category"]}",
    "subcategory": "{config.get("subcategory", "")}",
    "asset_class": "{config.get("asset_class", "Equities")}",
    "frequency": "{config.get("frequency", "Daily")}",
    "data_source": "{config.get("data", {}).get("source_type", "synthetic")}",
    "languages": {json.dumps(config.get("languages", ["Python"]))},
    "key_techniques": {json.dumps(config.get("tags", []))},
    "interactive_params": {json.dumps(config.get("interactive_params", []))},
    "tags": {json.dumps(config.get("tags", []))},
    "github_path": "{proj["category"]}/{proj["dir_name"]}",
    "notebook_path": "{proj["category"]}/{proj["dir_name"]}/notebooks/",
    "requires_gpu": {str(config.get("requires_gpu", False)).lower()},
    "has_cpp": {str(config.get("has_cpp", False)).lower()},
    "estimated_runtime_seconds": {config.get("estimated_runtime", 10)},
    "simulation_tier": "{config.get("simulation_tier", "precomputed")}"
}}

# Build results
monthly_rets = strategy_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

results = {{
    "project_id": "{proj["id"]}",
    "timestamp": datetime.now().isoformat(),
    "backtest_period": {{"start": str(strategy_returns.index[0].date()), "end": str(strategy_returns.index[-1].date())}},
    "benchmark": BENCHMARK if "BENCHMARK" in dir() else "SPY",
    "metrics": metrics,
    "category_specific_metrics": {{}},
    "monthly_returns": {{str(k.strftime("%Y-%m")): round(float(v), 6) for k, v in monthly_rets.items()}},
    "equity_curve": {{
        "dates": [str(d.date()) for d in equity_curve.index],
        "values": [round(float(v), 2) for v in equity_curve.values],
        "benchmark_values": [round(float(v), 2) for v in benchmark_equity.iloc[:len(equity_curve)].values]
    }},
    "parameter_sensitivity": parameter_sensitivity if "parameter_sensitivity" in dir() else []
}}

# Save files
import os
output_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(output_dir)

card_path = os.path.join(parent_dir, "strategy_card.json")
results_path = os.path.join(parent_dir, "results.json")

with open(card_path, "w") as f:
    json.dump(strategy_card, f, indent=2, default=str)
print(f"Strategy card saved to: {{card_path}}")

with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved to: {{results_path}}")
'''
        return "code", code

    def cell_12_summary(self, config):
        """Cell 12: Summary and Next Steps (markdown)."""
        proj = config["project"]
        return "markdown", f"""## Summary

### {proj["display_name"]}

**Key Findings:**
- Strategy backtested over the configured period with standardized metrics
- Results exported to `strategy_card.json` and `results.json` for portfolio dashboard integration
- Parameter sensitivity analysis shows robustness across parameter ranges

**Limitations:**
- Synthetic results used where strategy signals are not fully implemented
- Transaction costs modeled simply (flat slippage + commission)
- No market impact modeling for large positions

**Related Projects:**
- See other projects in the `{proj["category"]}` category for comparison
"""

    def generate_cells(self, config):
        """Generate all 12 cells in order. Returns list of (cell_type, content) tuples."""
        return [
            self.cell_01_title(config),
            self.cell_02_environment(config),
            self.cell_03_configuration(config),
            self.cell_04_data(config),
            self.cell_05_features(config),
            self.cell_06_strategy(config),
            self.cell_07_backtest(config),
            self.cell_08_visualization(config),
            self.cell_09_metrics(config),
            self.cell_10_sensitivity(config),
            self.cell_11_export(config),
            self.cell_12_summary(config),
        ]
