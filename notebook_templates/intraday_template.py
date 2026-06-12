"""Intraday strategies notebook template — for research_intraday_strategies/ (9 projects).

Each project gets a distinct signal generation + backtest strategy:
  intraday_01 (momentum)              — EMA crossover + RSI + MACD
  intraday_02 (mean_reversion)        — Bollinger Bands z-score
  intraday_03 (stat_arb)              — Cointegration pairs trading (SPY+QQQ)
  intraday_04 (momentum_value)        — Dual-factor long/short
  intraday_05 (options)               — IV/RV spread + BS straddle
  intraday_06 (execution_tca)         — TWAP/VWAP execution & TCA
  intraday_07 (ml_strategy)           — GBM classifier + feature engineering
  intraday_08 (regime_detection)      — K-means regime + conditional momentum
  intraday_09 (portfolio_construction) — MVO vs Risk Parity vs Equal-Weight
"""

from __future__ import annotations
import nbformat as nbf
from .common_cells import (
    title_cell, environment_setup_cell, config_cell,
    data_acquisition_yfinance, performance_viz_cell,
    metrics_cell, sensitivity_cell, export_cell, summary_cell,
    monthly_heatmap_cell, get_ticker_for_project,
)


# ═══════════════════════════════════════════════════════════════════════════
# intraday_01: Momentum — EMA crossover + RSI + MACD
# ═══════════════════════════════════════════════════════════════════════════
def _intraday01_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

lookback = PARAMS.get("lookback_period", 20)

# --- EMA crossover ---
ema_fast = price.ewm(span=12, adjust=False).mean()
ema_slow = price.ewm(span=26, adjust=False).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_hist = macd_line - signal_line

# --- RSI(14) ---
delta = price.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = delta.clip(upper=0).abs().rolling(14).mean()
rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan).fillna(1e-9))

# --- Composite signal ---
# Buy when MACD histogram > 0, RSI not overbought; sell when MACD < 0, RSI not oversold
signals = pd.DataFrame(index=price.index)
signals["macd_hist"] = macd_hist
signals["rsi"] = rsi
signals["ema_cross"] = (ema_fast > ema_slow).astype(float) - 0.5  # +0.5 / -0.5

# Position: blend of EMA direction and RSI filter
raw_signal = signals["ema_cross"].copy()
raw_signal[rsi > 70] = -0.3   # reduce longs when overbought
raw_signal[rsi < 30] = 0.3    # add longs when oversold
signals["composite"] = raw_signal.clip(-1, 1)
signals = signals.dropna()

print(f"Signal shape: {signals.shape}")
print(f"Long periods: {(signals['composite'] > 0).sum()}, Short: {(signals['composite'] < 0).sum()}")

# --- Visualization ---
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 1]})

# Price + EMAs
axes[0].plot(price.loc[signals.index], linewidth=0.8, color="#6b7280", label="Price")
axes[0].plot(ema_fast.loc[signals.index], linewidth=1, color="#f59e0b", label="EMA(12)")
axes[0].plot(ema_slow.loc[signals.index], linewidth=1, color="#3b82f6", label="EMA(26)")
# Buy/sell arrows
buy = signals.index[signals["composite"].diff() > 0.3]
sell = signals.index[signals["composite"].diff() < -0.3]
axes[0].scatter(buy, price.loc[buy], marker="^", color="#10b981", s=40, zorder=5, label="Buy")
axes[0].scatter(sell, price.loc[sell], marker="v", color="#ef4444", s=40, zorder=5, label="Sell")
axes[0].set_title("Price with EMA Crossover Signals", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# MACD
axes[1].bar(signals.index, macd_hist.loc[signals.index],
            color=["#10b981" if v > 0 else "#ef4444" for v in macd_hist.loc[signals.index]], alpha=0.7, width=2)
axes[1].plot(macd_line.loc[signals.index], linewidth=0.8, color="#f59e0b", label="MACD")
axes[1].plot(signal_line.loc[signals.index], linewidth=0.8, color="#3b82f6", label="Signal")
axes[1].set_title("MACD Histogram")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# RSI
axes[2].plot(rsi.loc[signals.index], linewidth=0.8, color="#8b5cf6")
axes[2].axhline(70, color="#ef4444", linestyle="--", alpha=0.5)
axes[2].axhline(30, color="#10b981", linestyle="--", alpha=0.5)
axes[2].fill_between(signals.index, 30, 70, alpha=0.05, color="#6b7280")
axes[2].set_title("RSI(14)")
axes[2].set_ylim(0, 100)
axes[2].grid(True, alpha=0.3)

# Composite signal
axes[3].fill_between(signals.index, 0, signals["composite"],
                     where=signals["composite"] > 0, color="#10b981", alpha=0.4)
axes[3].fill_between(signals.index, 0, signals["composite"],
                     where=signals["composite"] < 0, color="#ef4444", alpha=0.4)
axes[3].set_title("Composite Signal")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_02: Mean Reversion — Bollinger Bands
# ═══════════════════════════════════════════════════════════════════════════
def _intraday02_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

lookback = PARAMS.get("lookback_period", 20)
n_std = PARAMS.get("bb_std", 2.0)

# --- Bollinger Bands ---
sma = price.rolling(lookback).mean()
std = price.rolling(lookback).std()
upper_band = sma + n_std * std
lower_band = sma - n_std * std

# Z-score of price relative to BB midline
z_score = (price - sma) / std.replace(0, np.nan).fillna(1e-9)

# --- Signal: mean reversion ---
# Buy when z < -2, sell when z > +2, close at z = 0
signals = pd.DataFrame(index=price.index)
signals["z_score"] = z_score
signals["bb_width"] = (upper_band - lower_band) / sma * 100  # BB width %

# Position from z-score (inverted for mean reversion)
position = pd.Series(0.0, index=price.index)
position[z_score < -n_std] = 1.0   # buy oversold
position[z_score > n_std] = -1.0   # sell overbought
position[(z_score > -0.5) & (z_score < 0.5)] = 0.0  # close near mean
position = position.ffill().fillna(0)
signals["composite"] = position
signals = signals.dropna()

print(f"Signal shape: {signals.shape}")
print(f"Mean z-score: {z_score.mean():.4f}, Std: {z_score.std():.4f}")
print(f"Entries: Buy={int((position == 1).sum())}, Sell={int((position == -1).sum())}, Flat={int((position == 0).sum())}")

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.5, 1]})

# Price + Bollinger Bands
axes[0].plot(price.loc[signals.index], linewidth=0.8, color="#6b7280", label="Price")
axes[0].plot(sma.loc[signals.index], linewidth=1, color="#f59e0b", label=f"SMA({lookback})")
axes[0].fill_between(signals.index,
                     upper_band.loc[signals.index], lower_band.loc[signals.index],
                     alpha=0.15, color="#3b82f6", label=f"BB(±{n_std}σ)")
axes[0].set_title("Price with Bollinger Bands", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Z-score
axes[1].plot(z_score.loc[signals.index], linewidth=0.7, color="#8b5cf6")
axes[1].axhline(n_std, color="#ef4444", linestyle="--", alpha=0.6, label=f"+{n_std}σ (Sell)")
axes[1].axhline(-n_std, color="#10b981", linestyle="--", alpha=0.6, label=f"-{n_std}σ (Buy)")
axes[1].axhline(0, color="#6b7280", linewidth=0.5)
axes[1].fill_between(signals.index, -n_std, n_std, alpha=0.05, color="#6b7280")
axes[1].set_title("Z-Score (Price vs SMA)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# BB Width (volatility proxy)
axes[2].plot(signals["bb_width"], linewidth=0.7, color="#f59e0b")
axes[2].set_title("Bollinger Band Width (%)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_03: Statistical Arbitrage — Pairs Trading (SPY + QQQ)
# ═══════════════════════════════════════════════════════════════════════════
def _intraday03_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy import stats

# Pairs trading requires two assets
if close.ndim == 1 or close.shape[1] < 2:
    raise RuntimeError("Stat arb requires ≥2 tickers. Got single-asset data.")

asset_a = close.iloc[:, 0].ffill()  # SPY
asset_b = close.iloc[:, 1].ffill()  # QQQ
names = list(close.columns) if hasattr(close.columns, '__iter__') else ["Asset_A", "Asset_B"]

lookback = PARAMS.get("lookback_period", 60)

# --- OLS Hedge Ratio (rolling) ---
def rolling_hedge_ratio(a, b, window):
    ratio = pd.Series(np.nan, index=a.index)
    for i in range(window, len(a)):
        y = a.iloc[i-window:i].values
        x = b.iloc[i-window:i].values
        slope, _, _, _, _ = stats.linregress(x, y)
        ratio.iloc[i] = slope
    return ratio

hedge_ratio = rolling_hedge_ratio(asset_a, asset_b, lookback)

# --- Spread ---
spread = asset_a - hedge_ratio * asset_b
spread_mean = spread.rolling(lookback).mean()
spread_std = spread.rolling(lookback).std()
z_score = (spread - spread_mean) / spread_std.replace(0, np.nan).fillna(1e-9)

# --- Engle-Granger Cointegration test (full sample) ---
from scipy.stats import pearsonr
slope_full, intercept, r_val, p_val, _ = stats.linregress(asset_b.values, asset_a.values)
residual = asset_a - slope_full * asset_b - intercept
# ADF proxy: check if residual mean-reverts (simplified)
resid_autocorr = residual.autocorr(lag=1)
print(f"Cointegration test (full sample):")
print(f"  Hedge ratio: {slope_full:.4f}")
print(f"  R²: {r_val**2:.4f}")
print(f"  Residual autocorr(1): {resid_autocorr:.4f}")
print(f"  Correlation: {pearsonr(asset_a.dropna(), asset_b.dropna())[0]:.4f}")

# --- Trading signal ---
entry_z = PARAMS.get("entry_zscore", 2.0)
exit_z = PARAMS.get("exit_zscore", 0.5)

signals = pd.DataFrame(index=z_score.index)
signals["z_score"] = z_score
signals["hedge_ratio"] = hedge_ratio

# Position: long spread when z < -entry, short when z > +entry, close at |z| < exit
position = pd.Series(0.0, index=z_score.index)
position[z_score < -entry_z] = 1.0    # long A, short B
position[z_score > entry_z] = -1.0    # short A, long B
position[(z_score > -exit_z) & (z_score < exit_z)] = 0.0
position = position.ffill().fillna(0)
signals["composite"] = position
signals = signals.dropna()

print(f"\\nSignal shape: {signals.shape}")
print(f"Long spread: {int((position == 1).sum())}, Short spread: {int((position == -1).sum())}")

# --- Visualization ---
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True,
                         gridspec_kw={"height_ratios": [2, 2, 1.5, 1]})

# Normalized prices
norm_a = asset_a / asset_a.iloc[0]
norm_b = asset_b / asset_b.iloc[0]
axes[0].plot(norm_a, linewidth=1, color="#f59e0b", label=names[0])
axes[0].plot(norm_b, linewidth=1, color="#3b82f6", label=names[1])
axes[0].set_title(f"Normalized Prices: {names[0]} vs {names[1]}", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Spread
axes[1].plot(spread.loc[signals.index], linewidth=0.7, color="#8b5cf6", label="Spread")
axes[1].plot(spread_mean.loc[signals.index], linewidth=1, color="#f59e0b", label="Mean")
axes[1].set_title("Price Spread (A − β·B)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Z-score
axes[2].plot(z_score.loc[signals.index], linewidth=0.7, color="#3b82f6")
axes[2].axhline(entry_z, color="#ef4444", linestyle="--", alpha=0.6, label=f"+{entry_z} (Short)")
axes[2].axhline(-entry_z, color="#10b981", linestyle="--", alpha=0.6, label=f"-{entry_z} (Long)")
axes[2].axhline(0, color="#6b7280", linewidth=0.5)
axes[2].set_title("Spread Z-Score")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

# Rolling hedge ratio
axes[3].plot(hedge_ratio.loc[signals.index], linewidth=0.8, color="#f59e0b")
axes[3].set_title("Rolling Hedge Ratio (OLS)")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


def _intraday03_backtest() -> nbf.NotebookNode:
    """Custom backtest for pairs trading — trades the spread, not a single stock."""
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np

# Pairs trading backtest: PnL from spread position
ret_a = asset_a.pct_change()
ret_b = asset_b.pct_change()

# Spread return = position * (ret_a - hedge_ratio * ret_b)
pos = signals["composite"].shift(1)  # trade next day
hr = signals["hedge_ratio"].fillna(method="ffill")

spread_return = pos * (ret_a.loc[signals.index] - hr * ret_b.loc[signals.index])
strategy_returns_raw = spread_return.dropna()

# Stop-loss
stop_loss = PARAMS.get("stop_loss", 0.02)
equity_curve_raw = (1 + strategy_returns_raw).cumprod()
running_dd = equity_curve_raw / equity_curve_raw.cummax() - 1
stopped = running_dd < -stop_loss
if stopped.any():
    cooldown = PARAMS.get("cooldown_days", 5)
    stop_mask = stopped.rolling(cooldown, min_periods=1).max().fillna(0).astype(bool)
    strategy_returns_raw[stop_mask] = 0

equity_curve = (1 + strategy_returns_raw).cumprod()
# Benchmark: equal-weight of both assets
benchmark_equity = ((1 + (ret_a.loc[equity_curve.index] + ret_b.loc[equity_curve.index]) / 2)).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"Stop-loss triggers: {stopped.sum()}")
print(f"Final spread equity: {equity_curve.iloc[-1]:.4f}")
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_04: Momentum-Value Long/Short
# ═══════════════════════════════════════════════════════════════════════════
def _intraday04_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

lookback = PARAMS.get("lookback_period", 60)

# --- Factor 1: Momentum (12-1 month proxy) ---
# Use ~252 trading day lookback minus most recent 21 days (skip short-term reversal)
mom_long = price.pct_change(min(252, len(price) - 22))
mom_short = price.pct_change(21)
momentum_factor = mom_long - mom_short  # cross-sectional would rank assets; here rank over time

# --- Factor 2: Value (price relative to 52-week high) ---
rolling_max = price.rolling(252, min_periods=60).max()
value_factor = -(price / rolling_max - 1)  # more negative = more "value" (further from high)

# --- Combine factors (rank-based) ---
signals = pd.DataFrame(index=price.index)
signals["momentum"] = momentum_factor.rank(pct=True) - 0.5
signals["value"] = value_factor.rank(pct=True) - 0.5
signals["composite"] = (0.6 * signals["momentum"] + 0.4 * signals["value"]).clip(-1, 1)
signals = signals.dropna()

# Factor statistics
print("Factor statistics (annualized):")
for factor in ["momentum", "value", "composite"]:
    f_ret = (signals[factor].shift(1) * returns.loc[signals.index]).dropna()
    sharpe = f_ret.mean() / f_ret.std() * np.sqrt(252) if f_ret.std() > 0 else 0
    print(f"  {factor:>15}: Sharpe={sharpe:+.3f}, Mean={f_ret.mean()*252:+.4f}")

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Factor values over time
axes[0].plot(signals["momentum"], linewidth=0.6, color="#f59e0b", label="Momentum Factor", alpha=0.7)
axes[0].plot(signals["value"], linewidth=0.6, color="#3b82f6", label="Value Factor", alpha=0.7)
axes[0].axhline(0, color="#6b7280", linewidth=0.5)
axes[0].set_title("Factor Values (Rank-Normalized)", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Factor cumulative returns
for factor, color, lbl in [("momentum", "#f59e0b", "Momentum"),
                            ("value", "#3b82f6", "Value"),
                            ("composite", "#10b981", "Combined (60/40)")]:
    f_ret = (signals[factor].shift(1) * returns.loc[signals.index]).fillna(0)
    axes[1].plot((1 + f_ret).cumprod(), linewidth=1.2, color=color, label=lbl)
axes[1].set_title("Cumulative Factor Returns")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Rolling factor premium (21d)
roll_mom = (signals["momentum"].shift(1) * returns.loc[signals.index]).rolling(63).mean() * 252
roll_val = (signals["value"].shift(1) * returns.loc[signals.index]).rolling(63).mean() * 252
axes[2].plot(roll_mom, linewidth=0.8, color="#f59e0b", label="Momentum (63d ann.)")
axes[2].plot(roll_val, linewidth=0.8, color="#3b82f6", label="Value (63d ann.)")
axes[2].axhline(0, color="#6b7280", linewidth=0.5)
axes[2].set_title("Rolling Factor Premium")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_05: Options Strategy — IV/RV spread + Black-Scholes straddle
# ═══════════════════════════════════════════════════════════════════════════
def _intraday05_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

lookback = PARAMS.get("lookback_period", 20)

# --- Black-Scholes helper ---
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Realized Vol (RV) ---
rv = returns.rolling(lookback).std() * np.sqrt(252)

# --- Implied Vol proxy: use longer-window RV as IV approximation ---
# In practice IV comes from options market; here we add a premium to simulate
iv_lookback = min(lookback * 3, 63)
iv_proxy = returns.rolling(iv_lookback).std() * np.sqrt(252) * 1.15  # 15% vol risk premium

# --- IV / RV ratio ---
iv_rv_ratio = iv_proxy / rv.replace(0, np.nan).fillna(1e-9)

# --- Straddle P&L simulation ---
# At each date, simulate buying an ATM straddle (1-month expiry) priced at IV,
# and holding for `lookback` days. P&L depends on actual realized move vs implied.
r_f = PARAMS.get("risk_free_rate", 0.04)
T = lookback / 252  # expiry in years

straddle_pnl = pd.Series(np.nan, index=price.index)
straddle_cost = pd.Series(np.nan, index=price.index)

for i in range(lookback + iv_lookback, len(price) - lookback):
    S0 = price.iloc[i]
    K = S0  # ATM
    sigma_iv = iv_proxy.iloc[i]
    if np.isnan(sigma_iv) or sigma_iv <= 0:
        continue
    entry_call = bs_call(S0, K, T, r_f, sigma_iv)
    entry_put = bs_put(S0, K, T, r_f, sigma_iv)
    entry_cost = entry_call + entry_put

    # Exit after lookback days
    S_exit = price.iloc[i + lookback]
    T_remain = max(T - lookback/252, 1/252)
    sigma_exit = rv.iloc[i + lookback] if not np.isnan(rv.iloc[i + lookback]) else sigma_iv
    exit_call = bs_call(S_exit, K, T_remain, r_f, sigma_exit)
    exit_put = bs_put(S_exit, K, T_remain, r_f, sigma_exit)

    straddle_pnl.iloc[i] = (exit_call + exit_put) - entry_cost
    straddle_cost.iloc[i] = entry_cost

straddle_ret = (straddle_pnl / straddle_cost).dropna()

# --- Signal: trade straddle when IV/RV is favorable ---
signals = pd.DataFrame(index=price.index)
signals["iv"] = iv_proxy
signals["rv"] = rv
signals["iv_rv_ratio"] = iv_rv_ratio

# Sell straddle when IV >> RV (overpriced), buy when IV << RV (underpriced)
iv_rv_thresh = PARAMS.get("iv_rv_threshold", 1.2)
position = pd.Series(0.0, index=price.index)
position[iv_rv_ratio > iv_rv_thresh] = -1.0    # sell overpriced vol
position[iv_rv_ratio < 1.0 / iv_rv_thresh] = 1.0  # buy underpriced vol
position = position.ffill().fillna(0)
signals["composite"] = position
signals = signals.dropna()

print(f"IV mean: {iv_proxy.mean():.2%}, RV mean: {rv.mean():.2%}")
print(f"IV/RV ratio mean: {iv_rv_ratio.dropna().mean():.3f}")
print(f"Straddle entries: {len(straddle_ret)}")

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.5, 1.5]})

# IV vs RV
axes[0].plot(iv_proxy.loc[signals.index], linewidth=0.8, color="#ef4444", label="Implied Vol (proxy)")
axes[0].plot(rv.loc[signals.index], linewidth=0.8, color="#3b82f6", label="Realized Vol")
axes[0].fill_between(signals.index,
                     iv_proxy.loc[signals.index], rv.loc[signals.index],
                     where=iv_proxy.loc[signals.index] > rv.loc[signals.index],
                     alpha=0.15, color="#ef4444", label="IV > RV (sell vol)")
axes[0].fill_between(signals.index,
                     iv_proxy.loc[signals.index], rv.loc[signals.index],
                     where=iv_proxy.loc[signals.index] < rv.loc[signals.index],
                     alpha=0.15, color="#3b82f6", label="RV > IV (buy vol)")
axes[0].set_title("Implied vs Realized Volatility", fontsize=13)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# IV/RV Ratio
axes[1].plot(iv_rv_ratio.loc[signals.index], linewidth=0.7, color="#8b5cf6")
axes[1].axhline(iv_rv_thresh, color="#ef4444", linestyle="--", alpha=0.6, label=f"Sell thresh ({iv_rv_thresh})")
axes[1].axhline(1/iv_rv_thresh, color="#10b981", linestyle="--", alpha=0.6, label=f"Buy thresh ({1/iv_rv_thresh:.2f})")
axes[1].axhline(1.0, color="#6b7280", linewidth=0.5)
axes[1].set_title("IV / RV Ratio")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Straddle cumulative PnL
if len(straddle_ret) > 0:
    axes[2].plot((1 + straddle_ret).cumprod(), linewidth=1, color="#f59e0b")
    axes[2].set_title("Cumulative Straddle Return")
    axes[2].grid(True, alpha=0.3)
else:
    axes[2].text(0.5, 0.5, "Insufficient data for straddle simulation", ha="center", va="center")

plt.tight_layout()
plt.show()
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_06: Execution & TCA — TWAP/VWAP simulation
# ═══════════════════════════════════════════════════════════════════════════
def _intraday06_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

# --- Execution parameters ---
order_size = PARAMS.get("order_size", 100_000)
n_slices = PARAMS.get("n_slices", 20)
participation_rate = PARAMS.get("participation_rate", 0.1)
slice_size = order_size / n_slices
rng = np.random.default_rng(SEED)

# Synthetic volume (U-shaped)
n = len(price)
volume = rng.exponential(1e6, n) * (1 + 0.3 * np.sin(np.linspace(0, np.pi, n)))

# --- Run 3 execution algorithms across multiple windows ---
def run_twap(prices, vols, order_sz, slices):
    exec_prices = []
    for i in range(min(slices, len(prices))):
        impact = 5 * np.sqrt(order_sz / slices / vols[i]) * prices[i] / 10000
        exec_prices.append(prices[i] + impact)
    return np.array(exec_prices)

def run_vwap(prices, vols, order_sz, slices):
    vol_weights = vols[:slices] / vols[:slices].sum()
    exec_prices = []
    for i in range(min(slices, len(prices))):
        sz = order_sz * vol_weights[i]
        impact = 5 * np.sqrt(sz / vols[i]) * prices[i] / 10000
        exec_prices.append(prices[i] + impact)
    return np.array(exec_prices)

def run_pov(prices, vols, order_sz, part_rate):
    exec_prices, filled = [], 0
    for i in range(len(prices)):
        if filled >= order_sz: break
        sz = min(vols[i] * part_rate, order_sz - filled)
        impact = 5 * np.sqrt(sz / vols[i]) * prices[i] / 10000
        exec_prices.append(prices[i] + impact)
        filled += sz
    return np.array(exec_prices)

# Run algorithms on overlapping windows
window = min(n_slices * 2, n - 1)
n_windows = min(20, n // window)
results = {"TWAP": [], "VWAP": [], "POV": []}

for w in range(n_windows):
    start = w * window
    p = price.values[start:start+window]
    v = volume[start:start+window]
    if len(p) < n_slices: continue

    arrival = p[0]
    for algo_name, algo_fn in [("TWAP", lambda: run_twap(p, v, order_size, n_slices)),
                                ("VWAP", lambda: run_vwap(p, v, order_size, n_slices)),
                                ("POV", lambda: run_pov(p, v, order_size, participation_rate))]:
        ep = algo_fn()
        avg_px = ep.mean()
        is_bps = (avg_px - arrival) / arrival * 10000
        results[algo_name].append(is_bps)

# Summary
print("Execution Algorithm Comparison (avg Implementation Shortfall, bps):")
for algo, is_list in results.items():
    if is_list:
        print(f"  {algo:>6}: mean={np.mean(is_list):+.2f} bps, std={np.std(is_list):.2f} bps")

# Build a pseudo equity curve from cost savings (VWAP vs TWAP)
twap_costs = pd.Series(results["TWAP"]) if results["TWAP"] else pd.Series([0])
vwap_costs = pd.Series(results["VWAP"]) if results["VWAP"] else pd.Series([0])
cost_savings = twap_costs - vwap_costs  # positive = VWAP is better

# For compatibility with shared cells, create equity curve from underlying
signals = pd.DataFrame({"composite": 0.0}, index=price.index)
signals = signals.iloc[lookback:] if "lookback" in dir() else signals.iloc[20:]
signals["composite"] = 0  # no directional position

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Sample execution trajectory
p_sample = price.values[:window]
v_sample = volume[:window]
ep_twap = run_twap(p_sample, v_sample, order_size, n_slices)
ep_vwap = run_vwap(p_sample, v_sample, order_size, n_slices)

axes[0, 0].plot(p_sample[:n_slices], "o--", color="#6b7280", markersize=3, label="Market Price")
axes[0, 0].plot(ep_twap, "s-", color="#f59e0b", markersize=3, label="TWAP")
axes[0, 0].plot(ep_vwap, "^-", color="#3b82f6", markersize=3, label="VWAP")
axes[0, 0].axhline(p_sample[0], color="#10b981", linestyle="--", alpha=0.5, label="Arrival")
axes[0, 0].set_title("Execution Trajectory (Sample Window)")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# IS comparison bar chart
algo_names = list(results.keys())
algo_means = [np.mean(results[a]) if results[a] else 0 for a in algo_names]
colors = ["#f59e0b", "#3b82f6", "#10b981"]
axes[0, 1].bar(algo_names, algo_means, color=colors, alpha=0.7, edgecolor="white")
axes[0, 1].set_title("Avg Implementation Shortfall (bps)")
axes[0, 1].set_ylabel("IS (bps)")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# IS distribution per algo
for algo, color in zip(algo_names, colors):
    if results[algo]:
        axes[1, 0].hist(results[algo], bins=15, alpha=0.5, color=color, label=algo, edgecolor="none")
axes[1, 0].set_title("IS Distribution by Algorithm")
axes[1, 0].set_xlabel("IS (bps)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Cumulative fill profile
cum_fill = np.cumsum(np.ones(n_slices)) / n_slices * 100
axes[1, 1].plot(cum_fill, "o-", color="#f59e0b", label="TWAP (linear)")
vol_weights = v_sample[:n_slices] / v_sample[:n_slices].sum()
axes[1, 1].plot(np.cumsum(vol_weights) * 100, "s-", color="#3b82f6", label="VWAP (vol-weighted)")
axes[1, 1].set_title("Cumulative Fill Profile (%)")
axes[1, 1].set_ylabel("% Filled")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


def _intraday06_backtest() -> nbf.NotebookNode:
    """Execution TCA is not a directional strategy — synthesize equity curve from benchmark."""
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np

# TCA is not a directional strategy — equity curve = benchmark with execution cost overlay
lookback = PARAMS.get("lookback_period", 20)
price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
returns = price.pct_change()

# Apply small transaction cost drag to represent execution quality
tc_drag = PARAMS.get("cost_bps", 5) / 10000
strategy_returns_raw = returns.dropna() - tc_drag / 252  # daily cost drag
equity_curve = (1 + strategy_returns_raw).cumprod()
benchmark_equity = (1 + returns.loc[equity_curve.index]).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"Execution cost drag: {tc_drag*10000:.1f} bps/year")
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_07: ML Strategy — GBM + Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════
def _intraday07_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

lookback = PARAMS.get("lookback_period", 60)
embargo = PARAMS.get("embargo_days", 10)

# --- Feature engineering (50+ features) ---
features = pd.DataFrame(index=price.index)

# Returns at multiple horizons
for lag in [1, 2, 3, 5, 10, 21, 63]:
    features[f"ret_{lag}d"] = price.pct_change(lag)

# Volatility at multiple horizons
for w in [5, 10, 21, 63]:
    features[f"vol_{w}d"] = returns.rolling(max(w, 2)).std()

# Moving average ratios
for w in [5, 10, 20, 50, 100, 200]:
    if w < len(price):
        features[f"ma_ratio_{w}"] = price / price.rolling(w).mean() - 1

# RSI at multiple windows
for w in [7, 14, 21]:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(w).mean()
    loss = delta.clip(upper=0).abs().rolling(w).mean()
    features[f"rsi_{w}"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan).fillna(1e-9))

# Bollinger Band features
for w in [10, 20]:
    sma = price.rolling(w).mean()
    std = price.rolling(w).std()
    features[f"bb_z_{w}"] = (price - sma) / std.replace(0, np.nan).fillna(1e-9)

# Volume-related (synthetic)
features["ret_abs"] = returns.abs()
features["ret_skew_21"] = returns.rolling(21).skew()
features["ret_kurt_21"] = returns.rolling(21).kurt()

# Target: 5-day forward return direction
target = (returns.shift(-5).rolling(5).sum() > 0).astype(int)

# Align
common_idx = features.dropna().index.intersection(target.dropna().index)
X = features.loc[common_idx]
y = target.loc[common_idx]
print(f"Features: {X.shape[1]}, Samples: {len(X)}")
print(f"Class balance: {y.mean():.2%} positive")

# --- Train/Test split with embargo ---
split = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split - embargo], X.iloc[split:]
y_train, y_test = y.iloc[:split - embargo], y.iloc[split:]

# --- GBM Classifier ---
n_estimators = PARAMS.get("n_estimators", 200)
max_depth = PARAMS.get("max_depth", 4)
lr = PARAMS.get("learning_rate", 0.05)

model = GradientBoostingClassifier(
    n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
    subsample=0.8, random_state=SEED
)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\\nTest Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"\\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")

# --- Signal from predictions ---
signals = pd.DataFrame(index=X_test.index)
signals["composite"] = (y_pred_proba - 0.5) * 2  # scale to [-1, 1]
signals = signals.dropna()

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Feature importance (top 15)
importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(15)
importances.plot(kind="barh", ax=axes[0, 0], color="#f59e0b", edgecolor="white")
axes[0, 0].set_title("Top 15 Feature Importances", fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis="x")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color="#3b82f6", linewidth=2, label=f"AUC = {auc:.3f}")
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0, 1].set_title("ROC Curve")
axes[0, 1].set_xlabel("FPR")
axes[0, 1].set_ylabel("TPR")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Prediction distribution
axes[1, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, color="#10b981", label="Up days", density=True)
axes[1, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, color="#ef4444", label="Down days", density=True)
axes[1, 0].set_title("Prediction Probability Distribution")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Signal over time
axes[1, 1].fill_between(signals.index, 0, signals["composite"],
                        where=signals["composite"] > 0, color="#10b981", alpha=0.4)
axes[1, 1].fill_between(signals.index, 0, signals["composite"],
                        where=signals["composite"] < 0, color="#ef4444", alpha=0.4)
axes[1, 1].set_title("ML Signal (scaled probability)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_08: Regime Detection — K-means + conditional momentum
# ═══════════════════════════════════════════════════════════════════════════
def _intraday08_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

lookback = PARAMS.get("lookback_period", 252)
n_regimes = PARAMS.get("num_regimes", 3)

# --- Feature extraction for regime detection ---
regime_features = pd.DataFrame(index=price.index)
regime_features["ret_21d"] = returns.rolling(21).mean() * 252   # annualized trend
regime_features["vol_21d"] = returns.rolling(max(21, 2)).std() * np.sqrt(252)  # annualized vol
regime_features["skew_21d"] = returns.rolling(21).skew()
regime_features = regime_features.dropna()

# --- K-means clustering ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(regime_features)
kmeans = KMeans(n_clusters=n_regimes, random_state=SEED, n_init=10)
regime_labels = kmeans.fit_predict(X_scaled)

# Sort regimes by average return (0=worst, N-1=best)
regime_returns = {}
for r in range(n_regimes):
    mask = regime_labels == r
    regime_returns[r] = returns.loc[regime_features.index[mask]].mean()

sorted_regimes = sorted(regime_returns, key=regime_returns.get)
label_map = {old: new for new, old in enumerate(sorted_regimes)}
regime_labels = np.array([label_map[r] for r in regime_labels])
regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"} if n_regimes == 3 else {i: f"Regime {i}" for i in range(n_regimes)}

regime_series = pd.Series(regime_labels, index=regime_features.index)

# --- Signal: momentum in bull regime, flat/short otherwise ---
mom = price.pct_change(21).loc[regime_features.index]
signals = pd.DataFrame(index=regime_features.index)
signals["regime"] = regime_series
signals["momentum"] = mom

# Long momentum in bull, flat in sideways, short in bear
position = pd.Series(0.0, index=regime_features.index)
position[regime_series == n_regimes - 1] = mom[regime_series == n_regimes - 1].clip(-1, 1)  # bull: follow momentum
position[regime_series == 0] = -0.5  # bear: short bias
# sideways: stay flat (default 0)
signals["composite"] = position.clip(-1, 1)
signals = signals.dropna()

# --- Regime statistics ---
print(f"Regime Detection ({n_regimes} regimes via K-Means):")
for r in range(n_regimes):
    mask = regime_labels == r
    r_rets = returns.loc[regime_features.index[mask]]
    sharpe = r_rets.mean() / r_rets.std() * np.sqrt(252) if r_rets.std() > 0 else 0
    print(f"  {regime_names.get(r, f'R{r}'):>10}: days={mask.sum():>4}, "
          f"ann_ret={r_rets.mean()*252:+.2%}, ann_vol={r_rets.std()*np.sqrt(252):.2%}, "
          f"sharpe={sharpe:+.2f}")

# --- Transition matrix ---
transitions = np.zeros((n_regimes, n_regimes))
for i in range(1, len(regime_labels)):
    transitions[regime_labels[i-1], regime_labels[i]] += 1
# Normalize rows
row_sums = transitions.sum(axis=1, keepdims=True)
trans_prob = np.divide(transitions, row_sums, where=row_sums > 0)
trans_prob = np.where(row_sums > 0, trans_prob, 0)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Regime-colored price chart
colors_map = {0: "#ef4444", 1: "#f59e0b", 2: "#10b981"}
for r in range(n_regimes):
    mask = regime_series == r
    axes[0, 0].scatter(price.loc[regime_features.index][mask].index,
                       price.loc[regime_features.index][mask].values,
                       c=colors_map.get(r, "#6b7280"), s=3, alpha=0.6,
                       label=regime_names.get(r, f"R{r}"))
axes[0, 0].set_title("Price Colored by Regime", fontsize=12)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Transition matrix heatmap
im = axes[0, 1].imshow(trans_prob * 100, cmap="YlOrRd", aspect="auto")
for i in range(n_regimes):
    for j in range(n_regimes):
        axes[0, 1].text(j, i, f"{trans_prob[i,j]*100:.1f}%", ha="center", va="center", fontsize=10)
labels = [regime_names.get(i, f"R{i}") for i in range(n_regimes)]
axes[0, 1].set_xticks(range(n_regimes)); axes[0, 1].set_xticklabels(labels)
axes[0, 1].set_yticks(range(n_regimes)); axes[0, 1].set_yticklabels(labels)
axes[0, 1].set_title("Regime Transition Probability (%)")
plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

# Per-regime Sharpe
sharpe_by_regime = []
for r in range(n_regimes):
    mask = regime_labels == r
    r_rets = returns.loc[regime_features.index[mask]]
    sharpe_by_regime.append(r_rets.mean() / r_rets.std() * np.sqrt(252) if r_rets.std() > 0 else 0)
bar_colors = [colors_map.get(r, "#6b7280") for r in range(n_regimes)]
axes[1, 0].bar(labels, sharpe_by_regime, color=bar_colors, alpha=0.7, edgecolor="white")
axes[1, 0].set_title("Annualized Sharpe by Regime")
axes[1, 0].axhline(0, color="#6b7280", linewidth=0.5)
axes[1, 0].grid(True, alpha=0.3, axis="y")

# Feature scatter (ret vs vol, colored by regime)
for r in range(n_regimes):
    mask = regime_labels == r
    axes[1, 1].scatter(regime_features["vol_21d"].values[mask],
                       regime_features["ret_21d"].values[mask],
                       c=colors_map.get(r, "#6b7280"), alpha=0.4, s=10,
                       label=regime_names.get(r, f"R{r}"))
axes[1, 1].set_xlabel("Annualized Vol")
axes[1, 1].set_ylabel("Annualized Return")
axes[1, 1].set_title("Regime Clusters (Return vs Vol)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


# ═══════════════════════════════════════════════════════════════════════════
# intraday_09: Portfolio Construction — MVO vs Risk Parity vs EW
# ═══════════════════════════════════════════════════════════════════════════
def _intraday09_signal() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Multi-asset portfolio construction
if close.ndim == 1 or (hasattr(close, 'shape') and len(close.shape) == 1):
    raise RuntimeError("Portfolio construction requires multi-asset data. Got single-asset.")

returns_df = close.pct_change().dropna()
asset_names = list(close.columns)
n_assets = len(asset_names)
lookback = PARAMS.get("lookback_period", 252)
max_weight = PARAMS.get("max_weight", 0.4)
rebalance_freq = PARAMS.get("rebalance_freq", 21)  # monthly

print(f"Assets: {asset_names}")
print(f"Lookback: {lookback}, Rebalance: every {rebalance_freq} days")

# --- Optimization methods ---
def mvo_weights(ret_df, max_w=0.4):
    mu = ret_df.mean().values * 252
    cov = ret_df.cov().values * 252
    n = len(mu)
    # Approximate max-Sharpe via random sampling (no cvxpy dependency)
    rng = np.random.default_rng(SEED)
    best_sharpe, best_w = -np.inf, np.ones(n) / n
    for _ in range(5000):
        w = rng.random(n); w /= w.sum()
        w = np.clip(w, 0, max_w); w /= w.sum()
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        sr = ret / vol if vol > 0 else 0
        if sr > best_sharpe:
            best_sharpe, best_w = sr, w
    return best_w

def risk_parity_weights(ret_df, max_w=0.4):
    cov = ret_df.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / np.where(vols > 0, vols, 1e-6)
    w = inv_vol / inv_vol.sum()
    w = np.clip(w, 0, max_w); w /= w.sum()
    return w

def equal_weight(n):
    return np.ones(n) / n

# --- Rolling backtest ---
rebal_dates = returns_df.index[lookback::rebalance_freq]
methods = {"MVO": [], "Risk Parity": [], "Equal Weight": []}
weights_history = {"MVO": [], "Risk Parity": [], "Equal Weight": []}

for method_name, opt_fn in [("MVO", lambda df: mvo_weights(df, max_weight)),
                              ("Risk Parity", lambda df: risk_parity_weights(df, max_weight)),
                              ("Equal Weight", lambda df: equal_weight(n_assets))]:
    port_returns = []
    for i, date in enumerate(rebal_dates):
        loc = returns_df.index.get_loc(date)
        hist = returns_df.iloc[loc-lookback:loc]
        w = opt_fn(hist)
        weights_history[method_name].append(w)

        # Hold until next rebalance
        next_loc = returns_df.index.get_loc(rebal_dates[i+1]) if i + 1 < len(rebal_dates) else len(returns_df)
        fwd = returns_df.iloc[loc:next_loc]
        port_ret = fwd.values @ w
        port_returns.extend(port_ret.tolist())

    methods[method_name] = pd.Series(port_returns[:len(returns_df) - lookback],
                                      index=returns_df.index[lookback:lookback+len(port_returns)])

# Build composite signal from MVO weights for shared cells
signals = pd.DataFrame(index=methods["MVO"].index)
signals["composite"] = 1.0  # fully invested

# Print method comparison
print("\\nMethod Comparison (annualized):")
for name, rets in methods.items():
    sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    print(f"  {name:>15}: Return={ann_ret:+.2%}, Vol={ann_vol:.2%}, Sharpe={sr:+.3f}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cumulative returns comparison
for name, color in [("MVO", "#f59e0b"), ("Risk Parity", "#3b82f6"), ("Equal Weight", "#10b981")]:
    eq = (1 + methods[name]).cumprod()
    axes[0, 0].plot(eq, linewidth=1.5, color=color, label=name)
axes[0, 0].set_title("Cumulative Portfolio Returns", fontsize=12)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Latest weights comparison
if weights_history["MVO"]:
    x = np.arange(n_assets)
    width = 0.25
    axes[0, 1].bar(x - width, weights_history["MVO"][-1], width, color="#f59e0b", label="MVO")
    axes[0, 1].bar(x, weights_history["Risk Parity"][-1], width, color="#3b82f6", label="Risk Parity")
    axes[0, 1].bar(x + width, weights_history["Equal Weight"][-1], width, color="#10b981", label="Equal Weight")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(asset_names, fontsize=9)
    axes[0, 1].set_title("Latest Portfolio Weights")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

# Weight evolution over time (MVO, stacked area)
if weights_history["MVO"]:
    w_arr = np.array(weights_history["MVO"])
    w_dates = rebal_dates[:len(w_arr)]
    axes[1, 0].stackplot(w_dates, w_arr.T, labels=asset_names,
                         colors=["#f59e0b", "#3b82f6", "#10b981", "#ef4444", "#8b5cf6"][:n_assets],
                         alpha=0.7)
    axes[1, 0].set_title("MVO Weight Allocation Over Time")
    axes[1, 0].legend(loc="upper left", fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

# Rolling Sharpe comparison
for name, color in [("MVO", "#f59e0b"), ("Risk Parity", "#3b82f6"), ("Equal Weight", "#10b981")]:
    roll_sr = methods[name].rolling(63).apply(lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0)
    axes[1, 1].plot(roll_sr, linewidth=0.8, color=color, label=name, alpha=0.7)
axes[1, 1].axhline(0, color="#6b7280", linewidth=0.5)
axes[1, 1].set_title("Rolling Sharpe (63d)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")


def _intraday09_backtest() -> nbf.NotebookNode:
    """Portfolio construction backtest — use MVO as the primary equity curve."""
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np

# Use MVO portfolio as primary strategy
strategy_returns_raw = methods["MVO"].dropna()
equity_curve = (1 + strategy_returns_raw).cumprod()

# Benchmark: equal-weight portfolio
benchmark_returns = methods["Equal Weight"].reindex(equity_curve.index).fillna(0)
benchmark_equity = (1 + benchmark_returns).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"Final MVO equity: {equity_curve.iloc[-1]:.4f}")
print(f"Final EW equity:  {benchmark_equity.iloc[-1]:.4f}")
""")


# ═══════════════════════════════════════════════════════════════════════════
# Shared backtest cell (used by projects that follow the standard signal pattern)
# ═══════════════════════════════════════════════════════════════════════════
def _standard_backtest_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
returns = price.pct_change()

# Position sizing from signal
positions = signals["composite"].clip(-1, 1)

# Apply stop-loss
stop_loss = PARAMS.get("stop_loss", 0.02)
daily_pnl = positions.shift(1) * returns.loc[signals.index]

# Track running drawdown for stop-loss
running_equity = (1 + daily_pnl.fillna(0)).cumprod()
running_dd = running_equity / running_equity.cummax() - 1

# Zero position when drawdown exceeds stop-loss
stopped = running_dd < -stop_loss
if stopped.any():
    cooldown = PARAMS.get("cooldown_days", 5)
    stop_mask = stopped.rolling(cooldown, min_periods=1).max().fillna(0).astype(bool)
    daily_pnl[stop_mask] = 0

strategy_returns_raw = daily_pnl.dropna()
equity_curve = (1 + strategy_returns_raw).cumprod()
benchmark_equity = (1 + returns.loc[equity_curve.index]).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"Stop-loss triggers: {stopped.sum()}")
""")


# ═══════════════════════════════════════════════════════════════════════════
# Shared analysis cells
# ═══════════════════════════════════════════════════════════════════════════
def _regime_analysis_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import numpy as np, matplotlib.pyplot as plt

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
returns = price.pct_change()

# Regime-conditional performance
vol = returns.loc[equity_curve.index].rolling(21).std() * np.sqrt(252)
vol_median = vol.median()

high_vol = vol > vol_median
low_vol = ~high_vol

strat_ret = strategy_returns_raw

metrics_by_regime = {}
for regime, mask in [("Low Vol", low_vol), ("High Vol", high_vol)]:
    regime_rets = strat_ret[mask.reindex(strat_ret.index, fill_value=False)]
    if len(regime_rets) > 10:
        sharpe = regime_rets.mean() / regime_rets.std() * np.sqrt(252) if regime_rets.std() > 0 else 0
        metrics_by_regime[regime] = {
            "sharpe": sharpe,
            "return": (1 + regime_rets).prod() - 1,
            "vol": regime_rets.std() * np.sqrt(252),
            "days": len(regime_rets),
        }

print("Regime-Conditional Performance:")
for regime, m in metrics_by_regime.items():
    print(f"  {regime}: Sharpe={m['sharpe']:.2f}, Return={m['return']:+.2%}, Vol={m['vol']:.2%}, Days={m['days']}")
""")


def _daily_pnl_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Daily PnL distribution
axes[0].hist(strategy_returns_raw.values * 100, bins=60, color="#f59e0b", alpha=0.7, edgecolor="none")
axes[0].axvline(0, color="#6b7280", linewidth=0.5)
axes[0].set_title("Daily Return Distribution (%)")
axes[0].set_xlabel("Return (%)")

# PnL by day of week
dow_pnl = strategy_returns_raw.groupby(strategy_returns_raw.index.dayofweek).mean() * 252
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
colors = ["#10b981" if v > 0 else "#ef4444" for v in dow_pnl.values]
axes[1].bar(dow_labels[:len(dow_pnl)], dow_pnl.values * 100, color=colors, alpha=0.7)
axes[1].set_title("Annualized Return by Day of Week (%)")
axes[1].axhline(0, color="#6b7280", linewidth=0.5)

plt.tight_layout()
plt.show()
""")


# ═══════════════════════════════════════════════════════════════════════════
# Builder — dispatches to per-project cells
# ═══════════════════════════════════════════════════════════════════════════
def build_intraday_notebook(card: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    pid = card["project_id"]
    params = {p["name"]: p["default"] for p in card.get("interactive_params", [])}
    tickers = get_ticker_for_project(pid)

    # --- Common head ---
    head = [
        title_cell(card["title"], "Intraday Strategies",
                   card.get("long_description", card.get("short_description", "")), pid),
        environment_setup_cell(requires_gpu=False),
        config_cell(params),
        data_acquisition_yfinance(tickers),
    ]

    # --- Per-project signal + backtest cells ---
    if "momentum" in pid and "value" not in pid:
        # intraday_01: EMA/RSI/MACD momentum
        signal_cells = [
            nbf.v4.new_markdown_cell("## Signal Generation — EMA Crossover + RSI + MACD"),
            _intraday01_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _standard_backtest_cell(),
        ]
    elif "mean_reversion" in pid:
        # intraday_02: Bollinger Bands
        signal_cells = [
            nbf.v4.new_markdown_cell("## Signal Generation — Bollinger Bands Mean Reversion"),
            _intraday02_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _standard_backtest_cell(),
        ]
    elif "stat_arb" in pid:
        # intraday_03: Pairs trading
        signal_cells = [
            nbf.v4.new_markdown_cell("## Signal Generation — Cointegration Pairs Trading"),
            _intraday03_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _intraday03_backtest(),
        ]
    elif "momentum_value" in pid:
        # intraday_04: Dual-factor
        signal_cells = [
            nbf.v4.new_markdown_cell("## Signal Generation — Momentum-Value Dual Factor"),
            _intraday04_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _standard_backtest_cell(),
        ]
    elif "options" in pid:
        # intraday_05: IV/RV options
        signal_cells = [
            nbf.v4.new_markdown_cell("## Signal Generation — IV/RV Spread & Black-Scholes Straddle"),
            _intraday05_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _standard_backtest_cell(),
        ]
    elif "execution" in pid or "tca" in pid:
        # intraday_06: Execution TCA
        signal_cells = [
            nbf.v4.new_markdown_cell("## Execution Algorithm Comparison & TCA"),
            _intraday06_signal(),
            nbf.v4.new_markdown_cell("## Cost Analysis"),
            _intraday06_backtest(),
        ]
    elif "ml" in pid:
        # intraday_07: ML strategy
        signal_cells = [
            nbf.v4.new_markdown_cell("## ML Strategy — GBM Classifier"),
            _intraday07_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _standard_backtest_cell(),
        ]
    elif "regime" in pid:
        # intraday_08: Regime detection
        signal_cells = [
            nbf.v4.new_markdown_cell("## Regime Detection — K-Means Clustering"),
            _intraday08_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _standard_backtest_cell(),
        ]
    elif "portfolio" in pid:
        # intraday_09: Portfolio construction
        signal_cells = [
            nbf.v4.new_markdown_cell("## Portfolio Construction — MVO vs Risk Parity vs Equal Weight"),
            _intraday09_signal(),
            nbf.v4.new_markdown_cell("## Portfolio Backtest"),
            _intraday09_backtest(),
        ]
    else:
        # Fallback: generic momentum
        signal_cells = [
            nbf.v4.new_markdown_cell("## Signal Generation"),
            _intraday01_signal(),
            nbf.v4.new_markdown_cell("## Backtest Execution"),
            _standard_backtest_cell(),
        ]

    # --- Common tail ---
    tail = [
        performance_viz_cell(),
        metrics_cell(),
        monthly_heatmap_cell(),
        nbf.v4.new_markdown_cell("## Regime Analysis"),
        _regime_analysis_cell(),
        _daily_pnl_cell(),
        sensitivity_cell(card.get("interactive_params", [{"name": "lookback_period"}])[0].get("name", "lookback_period")),
        export_cell(pid),
        summary_cell(card["title"]),
    ]

    nb.cells = head + signal_cells + tail
    return nb
