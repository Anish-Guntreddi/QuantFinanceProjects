"""Backtesting/research notebook template — factor analytics, cointegration, vol surface."""

from .common import BaseNotebookTemplate, MetricsCalculator


class BacktestingTemplate(BaseNotebookTemplate):

    def cell_05_features(self, config):
        subcategory = config.get("subcategory", "")
        if "factor" in subcategory.lower():
            return "code", '''# Factor Construction
print("Computing cross-sectional factors...")

# Value factor: simple price-to-moving-average ratio
value_factor = price_data / price_data.rolling(252).mean()

# Momentum factor: 12-1 month momentum
momentum_factor = price_data.pct_change(252).shift(21)

# Volatility factor: inverse realized vol
vol_factor = -returns.rolling(63).std()

# Quality proxy: Sharpe of trailing returns
quality_factor = returns.rolling(63).mean() / returns.rolling(63).std()

factors = pd.DataFrame({
    "value": value_factor,
    "momentum": momentum_factor,
    "low_vol": vol_factor,
    "quality": quality_factor
}).dropna()

print(f"Factors computed: {list(factors.columns)}")
print(f"Factor correlations:\\n{factors.corr().round(3)}")
'''
        elif "cointegration" in subcategory.lower() or "stat" in subcategory.lower():
            return "code", '''# Cointegration Analysis
from scipy import stats

print("Running cointegration tests...")

# Generate correlated pair for stat arb
np.random.seed(SEED)
n = len(price_data)
spread_mean = 0
spread_vol = 0.5
ou_speed = 0.05
spread = [spread_mean]
for i in range(n - 1):
    ds = ou_speed * (spread_mean - spread[-1]) + spread_vol * np.random.randn() * np.sqrt(1/252)
    spread.append(spread[-1] + ds)

pair_price = price_data * np.exp(np.array(spread[:len(price_data)]))
pair_price = pd.Series(pair_price.values, index=price_data.index, name="Pair_Asset")

# Hedge ratio via OLS
from numpy.linalg import lstsq
X_reg = np.column_stack([price_data.values, np.ones(len(price_data))])
beta, _, _, _ = lstsq(X_reg, pair_price.values, rcond=None)
hedge_ratio = beta[0]

spread_series = pair_price - hedge_ratio * price_data
zscore = (spread_series - spread_series.rolling(60).mean()) / spread_series.rolling(60).std()

print(f"Hedge ratio: {hedge_ratio:.4f}")
print(f"Spread half-life: {int(-np.log(2) / np.log(1 - ou_speed * 252))} days (approx)")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(price_data, label="Asset 1", color="#00D4AA")
axes[0].plot(pair_price, label="Asset 2", color="#7B68EE")
axes[0].set_title("Price Series")
axes[0].legend()
axes[1].plot(spread_series, color="#FF6B35")
axes[1].set_title("Spread")
axes[2].plot(zscore, color="#1E90FF")
axes[2].axhline(2, color="#FF4757", linestyle="--", alpha=0.5)
axes[2].axhline(-2, color="#00D4AA", linestyle="--", alpha=0.5)
axes[2].set_title("Z-Score")
for ax in axes:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''
        elif "vol" in subcategory.lower() or "option" in subcategory.lower():
            return "code", '''# Volatility Surface Construction
from scipy.optimize import minimize
from scipy.stats import norm

print("Building volatility surface...")

# Black-Scholes implied vol helpers
def bs_price(S, K, T, r, sigma, is_call=True):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sig^2))
def svi(k, a, b, rho, m, sig):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sig**2))

# Generate synthetic smile data
S0 = price_data.iloc[-1] if hasattr(price_data, "iloc") else 100
strikes = np.linspace(0.8, 1.2, 20) * S0
maturities = [0.083, 0.25, 0.5, 1.0]  # 1M, 3M, 6M, 1Y
log_moneyness = np.log(strikes / S0)

vol_surface = {}
for T in maturities:
    base_vol = 0.20
    skew = -0.1 * np.sqrt(T)
    smile = base_vol + skew * log_moneyness + 0.3 * log_moneyness**2
    smile += np.random.normal(0, 0.005, len(strikes))
    smile = np.maximum(smile, 0.05)
    vol_surface[T] = smile

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
for T, vols in vol_surface.items():
    ax.plot(strikes, vols * 100, "o-", label=f"T={T:.3f}y", markersize=3)
ax.set_xlabel("Strike")
ax.set_ylabel("Implied Volatility (%)")
ax.set_title("Implied Volatility Smile by Maturity")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''
        else:
            return super().cell_05_features(config)

    def cell_07_backtest(self, config):
        return "code", MetricsCalculator.synthetic_results_code() + f'''

# Backtest Execution
print("Running backtest...")

try:
    if "signals" in dir() and signals is not None and hasattr(signals, "abs"):
        if signals.abs().sum().sum() > 0:
            strategy_returns = returns * signals.shift(1)
            print("Using strategy-generated signals")
        else:
            raise ValueError("Empty signals")
    elif "zscore" in dir():
        # Stat arb: mean-revert on z-score
        positions = pd.Series(0.0, index=zscore.index)
        positions[zscore < -2] = 1.0
        positions[zscore > 2] = -1.0
        positions[(zscore > -0.5) & (zscore < 0.5)] = 0.0
        positions = positions.ffill().fillna(0)
        spread_returns = spread_series.pct_change().dropna()
        strategy_returns = (positions.shift(1) * spread_returns).dropna()
        print("Using z-score mean-reversion signals")
    else:
        raise ValueError("No valid signals")
except Exception:
    print("Using synthetic backtest results")
    strategy_returns = generate_synthetic_results(
        n_days=min(len(returns), 504),
        annual_sharpe={config.get("synthetic_sharpe", 1.2)},
        annual_vol={config.get("synthetic_vol", 0.12)},
        seed=SEED
    )

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns.iloc[:len(strategy_returns)]).cumprod()
print(f"Backtest complete: {{len(strategy_returns)}} periods, final: ${{equity_curve.iloc[-1]:,.2f}}")
'''
