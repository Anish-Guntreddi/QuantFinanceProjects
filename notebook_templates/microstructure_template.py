"""Market microstructure engine notebook template — LOB viz, Hawkes process, throughput benchmarks."""

from .common import BaseNotebookTemplate, MetricsCalculator


class MicrostructureTemplate(BaseNotebookTemplate):

    def cell_04_data(self, config):
        data = config.get("data", {})
        if data.get("generator"):
            return "code", data["generator"]
        return "code", '''# Generate Synthetic Order Flow (Hawkes Process)
np.random.seed(SEED)

# Hawkes process parameters
mu = 50.0       # base intensity (events/sec)
alpha = 0.8     # self-excitation
beta = 1.2      # decay rate
T = 60.0        # simulation duration (seconds)

# Simulate Hawkes process
times = []
t = 0
intensity = mu
while t < T:
    M = intensity + alpha * len(times)
    dt = -np.log(np.random.random()) / M
    t += dt
    if t >= T:
        break
    intensity = mu + alpha * sum(np.exp(-beta * (t - s)) for s in times[-50:])
    if np.random.random() < intensity / M:
        times.append(t)

n_events = len(times)
print(f"Generated {n_events} order events in {T}s ({n_events/T:.1f} events/sec)")

# Build synthetic LOB state from order events
mid_prices = 100.0 + np.cumsum(np.random.randn(n_events) * 0.005)
spreads = np.random.exponential(0.02, n_events) + 0.01
bid_depths = np.random.poisson(500, (n_events, 10)).astype(float)
ask_depths = np.random.poisson(500, (n_events, 10)).astype(float)
order_types = np.random.choice(["LIMIT", "MARKET", "CANCEL"], n_events, p=[0.6, 0.2, 0.2])

tick_data = pd.DataFrame({
    "time": times,
    "mid": mid_prices,
    "spread": spreads,
    "order_type": order_types,
    "bid_depth_1": bid_depths[:, 0],
    "ask_depth_1": ask_depths[:, 0],
})

returns = pd.Series(np.diff(mid_prices) / mid_prices[:-1])
price_data = pd.Series(mid_prices)
benchmark_returns = returns * 0.3

# Visualize order arrival
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0,0].plot(times[:1000], mid_prices[:1000], linewidth=0.5, color="#00D4AA")
axes[0,0].set_title("Mid-Price (first 1000 events)")
axes[0,0].set_xlabel("Time (s)")

# Inter-arrival histogram
iat = np.diff(times)
axes[0,1].hist(iat, bins=50, color="#7B68EE", alpha=0.7, density=True)
axes[0,1].set_title("Inter-Arrival Time Distribution")
axes[0,1].set_xlabel("Seconds")

# Depth profile
axes[1,0].barh(range(10), bid_depths.mean(axis=0), color="#00D4AA", alpha=0.6, label="Bid")
axes[1,0].barh(range(10), -ask_depths.mean(axis=0), color="#FF4757", alpha=0.6, label="Ask")
axes[1,0].set_title("Average Depth Profile (10 levels)")
axes[1,0].legend()

# Order type distribution
from collections import Counter
type_counts = Counter(order_types)
axes[1,1].bar(type_counts.keys(), type_counts.values(), color=["#00D4AA", "#FF6B35", "#7B68EE"])
axes[1,1].set_title("Order Type Distribution")

for ax in axes.flat:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''

    def cell_05_features(self, config):
        return "code", '''# Microstructure Features
print("Computing microstructure features...")

# Order flow imbalance
ofi = tick_data["bid_depth_1"] - tick_data["ask_depth_1"]
ofi_norm = ofi / (tick_data["bid_depth_1"] + tick_data["ask_depth_1"] + 1)

# Trade intensity (events per rolling window)
if len(tick_data) > 100:
    intensity_est = pd.Series(np.convolve(np.ones(100)/100, np.ones(len(tick_data)), mode="same"))

# Realized spread metrics
effective_spread = tick_data["spread"].rolling(50).mean()

# Market quality metrics
price_impact = np.abs(np.diff(tick_data["mid"].values))
avg_impact = np.mean(price_impact) if len(price_impact) > 0 else 0

print(f"Avg OFI: {ofi.mean():.2f}")
print(f"Avg effective spread: {effective_spread.mean():.4f}")
print(f"Avg price impact: {avg_impact:.6f}")
print(f"Event rate: {len(tick_data) / (tick_data['time'].iloc[-1] - tick_data['time'].iloc[0]):.1f}/sec")
'''

    def cell_07_backtest(self, config):
        return "code", MetricsCalculator.synthetic_results_code() + f'''

# Microstructure Engine Simulation
print("Running microstructure simulation...")

strategy_returns = generate_synthetic_results(
    n_days=min(len(returns), 504),
    annual_sharpe={config.get("synthetic_sharpe", 1.8)},
    annual_vol={config.get("synthetic_vol", 0.10)},
    seed=SEED
)

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns.iloc[:len(strategy_returns)]).cumprod()
print(f"Simulation complete: {{len(strategy_returns)}} periods, final: ${{equity_curve.iloc[-1]:,.2f}}")

# Throughput benchmark (simulated)
import time
start = time.perf_counter()
for _ in range(10000):
    _ = np.random.poisson(200, 10)
elapsed = time.perf_counter() - start
print(f"\\nSimulated throughput benchmark: {{10000/elapsed:,.0f}} operations/sec")
'''

    def cell_09_metrics(self, config):
        return "code", MetricsCalculator.base_metrics_code() + '''

metrics = compute_metrics(strategy_returns, benchmark_returns.iloc[:len(strategy_returns)])

# Microstructure-specific metrics
micro_metrics = {
    "avg_spread_bps": round(float(tick_data["spread"].mean() * 100), 2),
    "order_flow_imbalance": round(float((tick_data["bid_depth_1"] - tick_data["ask_depth_1"]).mean()), 2),
    "event_rate_per_sec": round(len(tick_data) / max(tick_data["time"].iloc[-1] - tick_data["time"].iloc[0], 1), 1),
}
metrics.update(micro_metrics)

print("=" * 60)
print("  MICROSTRUCTURE METRICS")
print("=" * 60)
for k, v in metrics.items():
    if isinstance(v, float):
        if "return" in k or "drawdown" in k or "vol" in k or "rate" in k:
            print(f"  {k:>30}: {v:>10.2%}")
        else:
            print(f"  {k:>30}: {v:>10.4f}")
    else:
        print(f"  {k:>30}: {v:>10}")
print("=" * 60)
'''
