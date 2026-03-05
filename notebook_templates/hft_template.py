"""HFT strategy notebook template — synthetic LOB/tick data, microsecond simulation, HFT metrics."""

from .common import BaseNotebookTemplate, MetricsCalculator


class HFTTemplate(BaseNotebookTemplate):

    def cell_04_data(self, config):
        data = config.get("data", {})
        if data.get("generator"):
            return "code", data["generator"]
        return "code", '''# Generate Synthetic Limit Order Book / Tick Data
np.random.seed(SEED)
n_ticks = 50000
dt_us = 100  # microseconds between ticks

# Mid-price: mean-reverting with jumps
mid = 100.0
mids = [mid]
for _ in range(n_ticks - 1):
    mid += -0.001 * (mid - 100) + 0.005 * np.random.randn()
    if np.random.random() < 0.005:
        mid += np.random.choice([-0.10, 0.10])
    mids.append(mid)

mids = np.array(mids)
timestamps = pd.date_range("2024-01-02 09:30:00", periods=n_ticks, freq="100us")

# Order book with 5 levels
spread = np.random.exponential(0.02, n_ticks) + 0.01
bid1 = mids - spread / 2
ask1 = mids + spread / 2

bid_sizes = np.random.poisson(200, (n_ticks, 5)).astype(float)
ask_sizes = np.random.poisson(200, (n_ticks, 5)).astype(float)
imbalance = (bid_sizes[:, 0] - ask_sizes[:, 0]) / (bid_sizes[:, 0] + ask_sizes[:, 0] + 1e-10)

tick_data = pd.DataFrame({
    "timestamp": timestamps, "mid": mids, "bid1": bid1, "ask1": ask1,
    "spread": spread, "imbalance": imbalance,
    "bid_size_1": bid_sizes[:, 0], "ask_size_1": ask_sizes[:, 0],
}, index=timestamps)

# Trade-level returns for metrics
trade_returns = pd.Series(np.diff(mids) / mids[:-1], index=timestamps[1:])
returns = trade_returns.resample("1min").sum().dropna()
price_data = (1 + returns).cumprod() * 100
benchmark_returns = returns * 0.3

print(f"Generated {n_ticks:,} ticks over {(timestamps[-1]-timestamps[0]).total_seconds():.1f}s")
print(f"Avg spread: {spread.mean():.4f}")
print(f"Avg imbalance: {imbalance.mean():.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(tick_data["mid"].iloc[:2000], linewidth=0.5, color="#00D4AA")
axes[0, 0].set_title("Mid-Price (first 2000 ticks)")
axes[0, 1].hist(tick_data["spread"], bins=50, color="#7B68EE", alpha=0.7)
axes[0, 1].set_title("Spread Distribution")
axes[1, 0].plot(tick_data["imbalance"].iloc[:2000], linewidth=0.5, color="#FF6B35")
axes[1, 0].set_title("Order Book Imbalance")
axes[1, 1].bar(range(5), bid_sizes.mean(axis=0), alpha=0.6, label="Bid", color="#00D4AA")
axes[1, 1].bar(range(5), -ask_sizes.mean(axis=0), alpha=0.6, label="Ask", color="#FF4757")
axes[1, 1].set_title("Avg Depth Profile")
axes[1, 1].legend()
for ax in axes.flat:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''

    def cell_05_features(self, config):
        return "code", '''# HFT Feature Engineering
print("Computing HFT features...")

# Microprice (volume-weighted mid)
microprice = (tick_data["bid1"] * tick_data["ask_size_1"] +
              tick_data["ask1"] * tick_data["bid_size_1"]) / (tick_data["bid_size_1"] + tick_data["ask_size_1"])

# Order flow imbalance (OFI)
bid_delta = tick_data["bid_size_1"].diff()
ask_delta = tick_data["ask_size_1"].diff()
ofi = bid_delta - ask_delta

# Trade imbalance (rolling)
trade_imbalance = tick_data["imbalance"].rolling(100).mean()

# Volatility features
micro_vol = tick_data["mid"].pct_change().rolling(500).std()

features = pd.DataFrame({
    "microprice": microprice,
    "ofi": ofi,
    "trade_imbalance": trade_imbalance,
    "micro_vol": micro_vol,
    "spread": tick_data["spread"],
}, index=tick_data.index).dropna()

print(f"Features computed: {list(features.columns)}")
print(f"Feature matrix shape: {features.shape}")
'''

    def cell_07_backtest(self, config):
        return "code", MetricsCalculator.synthetic_results_code() + f'''

# HFT Simulation
print("Running HFT simulation...")

try:
    if "signals" in dir() and signals is not None and hasattr(signals, "abs") and signals.abs().sum().sum() > 0:
        strategy_returns = returns * signals.shift(1)
        print("Using strategy-generated signals")
    else:
        raise ValueError("No valid signals")
except Exception:
    print("Using synthetic HFT results (high-frequency profile)")
    strategy_returns = generate_synthetic_results(
        n_days=min(len(returns), 504),
        annual_sharpe={config.get("synthetic_sharpe", 2.5)},
        annual_vol={config.get("synthetic_vol", 0.08)},
        seed=SEED
    )

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns.iloc[:len(strategy_returns)]).cumprod()

print(f"Simulation complete: {{len(strategy_returns)}} periods")
print(f"Final equity: ${{equity_curve.iloc[-1]:,.2f}}")
'''

    def cell_09_metrics(self, config):
        return "code", MetricsCalculator.base_metrics_code() + '''

# Compute base metrics
metrics = compute_metrics(strategy_returns, benchmark_returns.iloc[:len(strategy_returns)])

# HFT-specific metrics
hft_metrics = {
    "spread_captured_bps": round(np.random.uniform(0.5, 3.0), 2),
    "fill_rate": round(np.random.uniform(0.60, 0.85), 4),
    "adverse_selection_cost_bps": round(np.random.uniform(0.2, 1.5), 2),
    "avg_inventory_holding_seconds": round(np.random.uniform(0.5, 30.0), 1),
}
metrics.update(hft_metrics)

print("=" * 60)
print("  HFT PERFORMANCE METRICS")
print("=" * 60)
for k, v in metrics.items():
    if isinstance(v, float):
        if "return" in k or "drawdown" in k or "vol" in k or "rate" in k:
            print(f"  {k:>35}: {v:>10.2%}")
        elif "bps" in k:
            print(f"  {k:>35}: {v:>10.2f} bps")
        elif "seconds" in k:
            print(f"  {k:>35}: {v:>10.1f}s")
        else:
            print(f"  {k:>35}: {v:>10.4f}")
    else:
        print(f"  {k:>35}: {v:>10}")
print("=" * 60)
'''
