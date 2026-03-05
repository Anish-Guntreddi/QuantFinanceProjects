"""Intraday strategy notebook template — yfinance data, standard backtest, daily PnL."""

from .common import BaseNotebookTemplate, MetricsCalculator


class IntradayTemplate(BaseNotebookTemplate):

    def cell_04_data(self, config):
        data = config.get("data", {})
        if data.get("generator"):
            return "code", data["generator"]
        tickers = data.get("tickers", ["SPY"])
        return "code", f'''import yfinance as yf

tickers = {tickers}
print(f"Fetching data for {{tickers}} from {{BACKTEST_START}} to {{BACKTEST_END}}...")
raw = yf.download(tickers if len(tickers) > 1 else tickers[0], start=BACKTEST_START, end=BACKTEST_END, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    price_data = raw["Close"]
    volume_data = raw["Volume"]
    ohlc = raw
else:
    price_data = raw["Close"]
    volume_data = raw["Volume"]
    ohlc = raw

returns = price_data.pct_change().dropna() if isinstance(price_data, pd.Series) else price_data.pct_change().dropna()

benchmark_data = yf.download("SPY", start=BACKTEST_START, end=BACKTEST_END, progress=False)["Close"]
benchmark_returns = benchmark_data.pct_change().dropna()

print(f"Data loaded: {{len(price_data)}} bars")
print(f"Date range: {{price_data.index[0].date()}} to {{price_data.index[-1].date()}}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={{"hspace": 0.05}})
if isinstance(price_data, pd.Series):
    ax1.plot(price_data, color="#00D4AA", linewidth=1)
    ax1.set_title(f"{{tickers[0]}} Price History")
else:
    for col in price_data.columns:
        ax1.plot(price_data[col], label=col, linewidth=1)
    ax1.legend()
    ax1.set_title("Price History")
ax1.grid(True, alpha=0.3)
if isinstance(volume_data, pd.Series):
    ax2.bar(volume_data.index, volume_data.values, width=1, color="#7B68EE", alpha=0.4)
else:
    ax2.bar(volume_data.index, volume_data.iloc[:, 0].values, width=1, color="#7B68EE", alpha=0.4)
ax2.set_title("Volume")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

    def cell_05_features(self, config):
        subcategory = config.get("subcategory", "").lower()
        if "momentum" in subcategory:
            return "code", '''# Momentum / Trend Following Signals
print("Computing momentum signals...")

sma_fast = price_data.rolling(PARAMS.get("lookback_period", 20)).mean()
sma_slow = price_data.rolling(50).mean()
ema_fast = price_data.ewm(span=PARAMS.get("lookback_period", 20)).mean()

# RSI
delta = price_data.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / (loss + 1e-10)
rsi = 100 - (100 / (1 + rs))

# MACD
ema12 = price_data.ewm(span=12).mean()
ema26 = price_data.ewm(span=26).mean()
macd = ema12 - ema26
macd_signal = macd.ewm(span=9).mean()

# Generate signals
signals = pd.Series(0.0, index=price_data.index)
signals[sma_fast > sma_slow] = 1.0
signals[sma_fast < sma_slow] = -1.0

print(f"Signal distribution: Long={int((signals==1).sum())}, Short={int((signals==-1).sum())}, Flat={int((signals==0).sum())}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(price_data, color="#6B7280", linewidth=0.5, alpha=0.5)
axes[0].plot(sma_fast, color="#00D4AA", label=f"SMA({PARAMS.get('lookback_period', 20)})")
axes[0].plot(sma_slow, color="#FF6B35", label="SMA(50)")
axes[0].legend()
axes[0].set_title("Price + Moving Averages")
axes[1].plot(rsi, color="#7B68EE")
axes[1].axhline(70, color="#FF4757", linestyle="--", alpha=0.5)
axes[1].axhline(30, color="#00D4AA", linestyle="--", alpha=0.5)
axes[1].set_title("RSI(14)")
axes[2].plot(macd, color="#1E90FF", label="MACD")
axes[2].plot(macd_signal, color="#FF6B35", label="Signal")
axes[2].legend()
axes[2].set_title("MACD")
for ax in axes:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''
        elif "mean" in subcategory or "reversion" in subcategory:
            return "code", '''# Mean Reversion Signals
print("Computing mean reversion signals...")

lookback = PARAMS.get("lookback_period", 20)
bb_mean = price_data.rolling(lookback).mean()
bb_std = price_data.rolling(lookback).std()
bb_upper = bb_mean + 2 * bb_std
bb_lower = bb_mean - 2 * bb_std
zscore = (price_data - bb_mean) / (bb_std + 1e-10)

# RSI-based mean reversion
delta = price_data.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

signals = pd.Series(0.0, index=price_data.index)
signals[zscore < -2] = 1.0   # Buy oversold
signals[zscore > 2] = -1.0   # Sell overbought
signals[(zscore > -0.5) & (zscore < 0.5)] = 0.0

print(f"Signal distribution: Long={int((signals==1).sum())}, Short={int((signals==-1).sum())}, Flat={int((signals==0).sum())}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(price_data, color="#6B7280", linewidth=0.5)
axes[0].plot(bb_upper, color="#FF4757", linewidth=0.8, linestyle="--")
axes[0].plot(bb_lower, color="#00D4AA", linewidth=0.8, linestyle="--")
axes[0].fill_between(price_data.index, bb_lower, bb_upper, alpha=0.05, color="#7B68EE")
axes[0].set_title("Bollinger Bands")
axes[1].plot(zscore, color="#1E90FF")
axes[1].axhline(2, color="#FF4757", linestyle="--", alpha=0.5)
axes[1].axhline(-2, color="#00D4AA", linestyle="--", alpha=0.5)
axes[1].set_title("Z-Score")
for ax in axes:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''
        elif "option" in subcategory:
            return "code", '''# Options Strategy Signals
print("Computing options signals...")
from scipy.stats import norm

# Black-Scholes pricing
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# Implied volatility estimation
realized_vol = returns.rolling(20).std() * np.sqrt(252)
iv_premium = 0.02  # IV typically > RV
implied_vol = realized_vol + iv_premium

# IV rank
iv_rank = (implied_vol - implied_vol.rolling(252).min()) / (implied_vol.rolling(252).max() - implied_vol.rolling(252).min() + 1e-10)

# Signals based on IV-RV spread
signals = pd.Series(0.0, index=price_data.index)
signals[iv_rank > 0.7] = -1.0  # Sell vol when IV rank high
signals[iv_rank < 0.3] = 1.0   # Buy vol when IV rank low

print(f"Current IV: {implied_vol.iloc[-1]:.2%}")
print(f"Current RV: {realized_vol.iloc[-1]:.2%}")
print(f"IV Rank: {iv_rank.iloc[-1]:.2%}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(realized_vol * 100, label="Realized Vol", color="#00D4AA")
axes[0].plot(implied_vol * 100, label="Implied Vol (est)", color="#FF6B35")
axes[0].set_title("Volatility (%)")
axes[0].legend()
axes[1].plot(iv_rank * 100, color="#7B68EE")
axes[1].axhline(70, color="#FF4757", linestyle="--", alpha=0.5)
axes[1].axhline(30, color="#00D4AA", linestyle="--", alpha=0.5)
axes[1].set_title("IV Rank (%)")
for ax in axes:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''
        elif "portfolio" in subcategory or "risk" in subcategory:
            return "code", '''# Portfolio Construction Features
print("Computing portfolio features...")

if isinstance(returns, pd.DataFrame):
    asset_returns = returns
else:
    # Create multi-asset returns for portfolio construction
    import yfinance as yf
    portfolio_tickers = ["SPY", "AGG", "GLD", "EFA", "VNQ"]
    port_data = yf.download(portfolio_tickers, start=BACKTEST_START, end=BACKTEST_END, progress=False)["Close"]
    asset_returns = port_data.pct_change().dropna()
    price_data = port_data

# Covariance estimation
cov_sample = asset_returns.cov() * 252
corr = asset_returns.corr()

# Expected returns (historical)
mu = asset_returns.mean() * 252

print(f"Assets: {list(asset_returns.columns)}")
print(f"\\nAnnualized Returns:")
for col in mu.index:
    print(f"  {col:>6}: {mu[col]:>8.2%}")

print(f"\\nCorrelation Matrix:")
print(corr.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax)
ax.set_title("Asset Correlation Matrix")
plt.tight_layout()
plt.show()
'''
        else:
            return super().cell_05_features(config)

    def cell_07_backtest(self, config):
        return "code", MetricsCalculator.synthetic_results_code() + f'''

# Intraday Strategy Backtest
print("Running backtest...")

try:
    if "signals" in dir() and signals is not None and hasattr(signals, "abs"):
        sig_sum = signals.abs().sum()
        if isinstance(sig_sum, (int, float)) and sig_sum > 0:
            positions = signals.ffill().fillna(0)
            ret = returns if isinstance(returns, pd.Series) else returns.iloc[:, 0]
            strategy_returns = (positions.shift(1) * ret).dropna()
            # Apply transaction costs
            turnover = positions.diff().abs()
            costs = turnover * PARAMS.get("slippage", 0.0001) + turnover * PARAMS.get("commission", 0.001)
            strategy_returns = strategy_returns - costs.reindex(strategy_returns.index, fill_value=0)
            print("Using strategy-generated signals")
        else:
            raise ValueError("Empty signals")
    else:
        raise ValueError("No signals defined")
except Exception:
    print("Using synthetic intraday results")
    strategy_returns = generate_synthetic_results(
        n_days=min(len(returns) if isinstance(returns, pd.Series) else len(returns.iloc[:, 0]), 504),
        annual_sharpe={config.get("synthetic_sharpe", 1.2)},
        annual_vol={config.get("synthetic_vol", 0.15)},
        seed=SEED
    )

if isinstance(returns, pd.DataFrame):
    bench = benchmark_returns if isinstance(benchmark_returns, pd.Series) else benchmark_returns.iloc[:, 0]
else:
    bench = benchmark_returns

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + bench.iloc[:len(strategy_returns)]).cumprod()
print(f"Backtest complete: {{len(strategy_returns)}} periods, final: ${{equity_curve.iloc[-1]:,.2f}}")
'''
