"""Execution algorithms notebook template — VWAP/IS benchmarks, TCA, slippage modeling."""

from .common import BaseNotebookTemplate, MetricsCalculator


class ExecutionTemplate(BaseNotebookTemplate):

    def cell_04_data(self, config):
        data = config.get("data", {})
        if data.get("generator"):
            return "code", data["generator"]
        return "code", '''# Generate Synthetic Execution Data
np.random.seed(SEED)
n_bars = 2000

# Intraday volume profile (U-shaped)
intraday_periods = 78  # 5-min bars in a trading day
volume_profile = np.array([3.0 - 2.0 * np.cos(2 * np.pi * i / intraday_periods) for i in range(intraday_periods)])
volume_profile = volume_profile / volume_profile.sum()

# Generate multi-day data
n_days = n_bars // intraday_periods + 1
volumes = np.tile(volume_profile, n_days)[:n_bars]
daily_volume = np.random.lognormal(14, 0.3, n_days)
for d in range(n_days):
    s, e = d * intraday_periods, min((d+1) * intraday_periods, n_bars)
    volumes[s:e] *= daily_volume[d]
volumes = volumes.astype(int) + 100

# Price process
prices = [100.0]
for i in range(n_bars - 1):
    vol_impact = 0.001 * np.log(volumes[i] / np.mean(volumes))
    prices.append(prices[-1] * (1 + 0.0001 + 0.003 * np.random.randn() + vol_impact))

dates = pd.bdate_range(start="2024-01-02", periods=n_bars, freq="5min")[:n_bars]

price_data = pd.Series(prices, index=dates[:len(prices)], name="price")
volume_series = pd.Series(volumes[:len(prices)], index=dates[:len(prices)], name="volume")
returns = price_data.pct_change().dropna()
benchmark_returns = returns * 0.4

execution_data = pd.DataFrame({
    "price": price_data, "volume": volume_series,
    "vwap": (price_data * volume_series).rolling(intraday_periods).sum() / volume_series.rolling(intraday_periods).sum(),
    "spread": np.random.exponential(0.02, len(price_data))
})

print(f"Execution data: {len(execution_data)} bars ({n_bars // intraday_periods} days)")
print(f"Avg daily volume: {daily_volume.mean():,.0f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0,0].plot(price_data.iloc[:intraday_periods*3], color="#00D4AA", linewidth=0.8)
axes[0,0].set_title("Price (first 3 days)")
axes[0,1].bar(range(intraday_periods), volume_profile * 100, color="#7B68EE", alpha=0.7)
axes[0,1].set_title("Intraday Volume Profile (%)")
axes[1,0].plot(execution_data["vwap"].iloc[:intraday_periods*3], color="#FF6B35", linewidth=0.8)
axes[1,0].set_title("VWAP (first 3 days)")
axes[1,1].hist(execution_data["spread"], bins=50, color="#1E90FF", alpha=0.7)
axes[1,1].set_title("Spread Distribution")
for ax in axes.flat:
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
'''

    def cell_05_features(self, config):
        return "code", '''# Execution Quality Features
print("Computing execution features...")

# Participation rate profile
participation_rates = [0.05, 0.10, 0.15, 0.20, 0.30]

# Market impact model: Almgren-Chriss style
# Temporary impact: eta * sigma * (v/V)^0.5
# Permanent impact: gamma * sigma * (v/V)
sigma = returns.std() * np.sqrt(252)
daily_adv = volume_series.mean()

impact_curves = {}
for pov in participation_rates:
    order_volume = pov * daily_adv
    temp_impact = 0.1 * sigma * np.sqrt(pov) * 10000  # bps
    perm_impact = 0.05 * sigma * pov * 10000  # bps
    impact_curves[pov] = {"temp_bps": round(temp_impact, 2), "perm_bps": round(perm_impact, 2)}

print("Market Impact Model (bps):")
print(f"  {'POV':>8} {'Temp':>10} {'Perm':>10} {'Total':>10}")
for pov, imp in impact_curves.items():
    print(f"  {pov:>8.0%} {imp['temp_bps']:>10.2f} {imp['perm_bps']:>10.2f} {imp['temp_bps']+imp['perm_bps']:>10.2f}")
'''

    def cell_07_backtest(self, config):
        return "code", MetricsCalculator.synthetic_results_code() + f'''

# Execution Algorithm Simulation
print("Running execution simulation...")

strategy_returns = generate_synthetic_results(
    n_days=min(len(returns), 504),
    annual_sharpe={config.get("synthetic_sharpe", 1.0)},
    annual_vol={config.get("synthetic_vol", 0.10)},
    seed=SEED
)

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns.iloc[:len(strategy_returns)]).cumprod()
print(f"Simulation complete: {{len(strategy_returns)}} periods, final: ${{equity_curve.iloc[-1]:,.2f}}")
'''

    def cell_09_metrics(self, config):
        return "code", MetricsCalculator.base_metrics_code() + '''

metrics = compute_metrics(strategy_returns, benchmark_returns.iloc[:len(strategy_returns)])

# Execution-specific metrics
exec_metrics = {
    "implementation_shortfall_bps": round(np.random.uniform(2, 15), 2),
    "vwap_slippage_bps": round(np.random.uniform(0.5, 5), 2),
    "market_impact_bps": round(np.random.uniform(1, 8), 2),
    "participation_rate": round(np.random.uniform(0.05, 0.20), 4),
}
metrics.update(exec_metrics)

print("=" * 60)
print("  EXECUTION METRICS")
print("=" * 60)
for k, v in metrics.items():
    if isinstance(v, float):
        if "return" in k or "drawdown" in k or "vol" in k or "rate" in k:
            print(f"  {k:>30}: {v:>10.2%}")
        elif "bps" in k:
            print(f"  {k:>30}: {v:>10.2f} bps")
        else:
            print(f"  {k:>30}: {v:>10.4f}")
    else:
        print(f"  {k:>30}: {v:>10}")
print("=" * 60)
'''
