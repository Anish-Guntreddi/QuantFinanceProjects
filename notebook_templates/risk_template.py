"""Risk engineering notebook template — portfolio optimization, VaR, stress testing."""

from .common import BaseNotebookTemplate, MetricsCalculator


class RiskTemplate(BaseNotebookTemplate):

    def cell_04_data(self, config):
        data = config.get("data", {})
        if data.get("generator"):
            return "code", data["generator"]
        tickers = data.get("tickers", ["SPY", "AGG", "GLD", "EFA", "VNQ"])
        return "code", f'''import yfinance as yf

tickers = {tickers}
print(f"Fetching data for {{tickers}} from {{BACKTEST_START}} to {{BACKTEST_END}}...")
raw = yf.download(tickers, start=BACKTEST_START, end=BACKTEST_END, progress=False)
price_data = raw["Close"]
returns = price_data.pct_change().dropna()

benchmark_data = yf.download("SPY", start=BACKTEST_START, end=BACKTEST_END, progress=False)["Close"]
benchmark_returns = benchmark_data.pct_change().dropna()

print(f"Data loaded: {{len(returns)}} observations, {{len(tickers)}} assets")

fig, ax = plt.subplots(figsize=(14, 6))
normalized = price_data / price_data.iloc[0] * 100
for col in normalized.columns:
    ax.plot(normalized[col], label=col, linewidth=1)
ax.set_title("Normalized Price History (base=100)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

    def cell_05_features(self, config):
        subcategory = config.get("subcategory", "").lower()
        if "portfolio" in subcategory or "optimization" in subcategory:
            return "code", '''# Portfolio Analytics
print("Computing portfolio analytics...")

# Annualized stats
mu = returns.mean() * 252
sigma = returns.std() * np.sqrt(252)
cov = returns.cov() * 252
corr = returns.corr()

# Risk contributions (equal weight baseline)
n = len(returns.columns)
w_eq = np.ones(n) / n
port_vol = np.sqrt(w_eq @ cov.values @ w_eq)
marginal_risk = cov.values @ w_eq / port_vol
risk_contrib = w_eq * marginal_risk
risk_pct = risk_contrib / port_vol * 100

print("Asset Statistics:")
print(f"  {'Asset':>6} {'Return':>10} {'Vol':>10} {'Sharpe':>10} {'Risk%':>10}")
for i, col in enumerate(returns.columns):
    sr = mu[col] / sigma[col] if sigma[col] > 0 else 0
    print(f"  {col:>6} {mu[col]:>10.2%} {sigma[col]:>10.2%} {sr:>10.2f} {risk_pct[i]:>10.1f}%")

print(f"\\nEqual-weight portfolio vol: {port_vol:.2%}")
print(f"Diversification ratio: {sigma.mean() / port_vol:.2f}")
'''
        elif "reproducib" in subcategory:
            return "code", '''# Research Reproducibility Features
print("Setting up reproducibility tracking...")

# Experiment configuration
experiment_config = {
    "seed": SEED,
    "data_version": "v1.0",
    "model_type": "baseline",
    "features": ["momentum", "volatility", "mean_reversion"],
    "validation": "walk_forward_5_splits",
    "metric_target": "sharpe_ratio"
}

print("Experiment Configuration:")
for k, v in experiment_config.items():
    print(f"  {k}: {v}")

# Simulated walk-forward validation
n_splits = 5
split_results = []
total = len(returns) if isinstance(returns, pd.Series) else len(returns.iloc[:, 0])
split_size = total // (n_splits + 1)

for i in range(n_splits):
    train_end = (i + 1) * split_size
    test_end = min(train_end + split_size, total)
    ret_slice = returns.iloc[train_end:test_end] if isinstance(returns, pd.Series) else returns.iloc[train_end:test_end, 0]
    sharpe = ret_slice.mean() / ret_slice.std() * np.sqrt(252) if ret_slice.std() > 0 else 0
    split_results.append({"split": i+1, "train_end": train_end, "test_size": test_end - train_end, "sharpe": round(sharpe, 3)})

print(f"\\nWalk-Forward Results:")
for r in split_results:
    print(f"  Split {r['split']}: Sharpe={r['sharpe']:.3f} (test_size={r['test_size']})")
'''
        else:
            return "code", '''# Risk Features
print("Computing risk features...")

ret = returns if isinstance(returns, pd.Series) else returns.mean(axis=1)
rolling_vol = ret.rolling(21).std() * np.sqrt(252)
rolling_skew = ret.rolling(63).skew()
rolling_kurt = ret.rolling(63).apply(lambda x: x.kurtosis())

print(f"Current annualized vol: {rolling_vol.iloc[-1]:.2%}")
print(f"Current skewness: {rolling_skew.iloc[-1]:.3f}")
print(f"Current excess kurtosis: {rolling_kurt.iloc[-1]:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(rolling_vol * 100, color="#FF6B35")
axes[0].set_title("Rolling 21-day Annualized Volatility (%)")
axes[1].plot(rolling_skew, color="#7B68EE")
axes[1].axhline(0, color="#6B7280", linestyle="--")
axes[1].set_title("Rolling 63-day Skewness")
axes[2].plot(rolling_kurt, color="#1E90FF")
axes[2].axhline(0, color="#6B7280", linestyle="--")
axes[2].set_title("Rolling 63-day Excess Kurtosis")
for ax in axes:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''

    def cell_07_backtest(self, config):
        return "code", MetricsCalculator.synthetic_results_code() + f'''

# Risk Strategy Simulation
print("Running risk/portfolio simulation...")

try:
    if "w_eq" in dir():
        # Portfolio returns with equal weights
        portfolio_returns = (returns * w_eq).sum(axis=1)
        strategy_returns = portfolio_returns
        bench = benchmark_returns.reindex(strategy_returns.index)
        print("Using equal-weight portfolio")
    elif "signals" in dir() and signals is not None:
        strategy_returns = returns * signals.shift(1)
        bench = benchmark_returns.iloc[:len(strategy_returns)]
        print("Using signal-based strategy")
    else:
        raise ValueError("No portfolio defined")
except Exception:
    print("Using synthetic risk/portfolio results")
    strategy_returns = generate_synthetic_results(
        n_days=504,
        annual_sharpe={config.get("synthetic_sharpe", 0.9)},
        annual_vol={config.get("synthetic_vol", 0.10)},
        seed=SEED
    )
    bench = benchmark_returns.iloc[:len(strategy_returns)]

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + bench.iloc[:len(strategy_returns)]).cumprod()
print(f"Simulation complete: {{len(strategy_returns)}} periods, final: ${{equity_curve.iloc[-1]:,.2f}}")
'''

    def cell_09_metrics(self, config):
        return "code", MetricsCalculator.base_metrics_code() + '''

metrics = compute_metrics(strategy_returns, benchmark_returns.iloc[:len(strategy_returns)])

# Risk-specific metrics
from scipy import stats

ret_arr = strategy_returns.values
var_95 = np.percentile(ret_arr, 5)
cvar_95 = ret_arr[ret_arr <= var_95].mean() if (ret_arr <= var_95).any() else var_95

risk_metrics = {
    "var_95": round(float(var_95), 6),
    "cvar_95": round(float(cvar_95), 6),
    "skewness": round(float(stats.skew(ret_arr)), 4),
    "kurtosis": round(float(stats.kurtosis(ret_arr)), 4),
}

if "cov" in dir() and "w_eq" in dir():
    port_vol = np.sqrt(w_eq @ cov.values @ w_eq)
    avg_vol = (returns.std() * np.sqrt(252)).mean()
    risk_metrics["diversification_ratio"] = round(float(avg_vol / port_vol), 4)

metrics.update(risk_metrics)

print("=" * 60)
print("  RISK METRICS")
print("=" * 60)
for k, v in metrics.items():
    if isinstance(v, float):
        if "return" in k or "drawdown" in k or "vol" in k or "rate" in k or "var" in k:
            print(f"  {k:>25}: {v:>10.2%}")
        else:
            print(f"  {k:>25}: {v:>10.4f}")
    else:
        print(f"  {k:>25}: {v:>10}")
print("=" * 60)
'''
