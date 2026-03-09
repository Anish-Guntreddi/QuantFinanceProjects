"""HFT notebook template — for HFT_strategy_projects/ (9 projects).

Each project gets a distinct synthetic data generator + strategy + visualization:
  hft_01 — Avellaneda-Stoikov adaptive market making
  hft_02 — Order book imbalance scalping
  hft_03 — Queue position modeling (Hawkes process)
  hft_04 — Cross-exchange arbitrage (dual-feed)
  hft_05 — Short-horizon trade imbalance (net order flow)
  hft_06 — Iceberg order detection
  hft_07 — Latency arbitrage simulator
  hft_08 — Smart order router (multi-venue)
  hft_09 — RL market maker (Q-learning)
"""

from __future__ import annotations
import nbformat as nbf
from .common_cells import (
    title_cell, environment_setup_cell, config_cell,
    data_acquisition_synthetic, performance_viz_cell,
    metrics_cell, sensitivity_cell, export_cell, summary_cell,
    monthly_heatmap_cell,
)


# ═══════════════════════════════════════════════════════════════════════════
# hft_01: Avellaneda-Stoikov Market Making
# ═══════════════════════════════════════════════════════════════════════════
def _hft01_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 50_000

# Mid-price: GBM with drift ~ 0
dt = 1.0 / 252 / 390  # ~1 second
sigma = PARAMS.get("volatility", 0.3)
mid = 100.0 * np.exp(np.cumsum(rng.normal(0, sigma * np.sqrt(dt), n_steps)))

# Poisson order arrivals
lam = PARAMS.get("arrival_rate", 50.0)  # orders/sec
arrivals = rng.poisson(lam, n_steps)

data = pd.DataFrame({"mid_price": mid, "arrivals": arrivals})
data["returns"] = pd.Series(mid).pct_change()
print(f"Synthetic data: {n_steps:,} ticks, σ={sigma:.1%}")
print(f"Avg arrivals/tick: {arrivals.mean():.1f}")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

# Avellaneda-Stoikov optimal market making
gamma = PARAMS.get("risk_aversion", 0.1)
max_pos = PARAMS.get("max_position", 100)
T = 1.0  # trading session = 1 unit
kappa = PARAMS.get("kappa", 500)  # order arrival intensity parameter
sigma = PARAMS.get("volatility", 0.3) * np.sqrt(1/252/390)  # per-tick volatility (return space)

n = len(data)
mid_prices = data["mid_price"].values
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)
reserv_prices = np.zeros(n)
opt_spreads = np.zeros(n)
prev_bid, prev_ask = 0.0, 1e12  # previous tick's posted quotes

for t in range(1, n):
    mid = mid_prices[t]
    tau = max((n - t) / n * T, 1e-6)  # time remaining

    # Check fills: did this tick's mid cross last tick's posted quotes?
    if mid <= prev_bid and position < max_pos:
        position += 1
        cash -= prev_bid
    elif mid >= prev_ask and position > -max_pos:
        position -= 1
        cash += prev_ask

    pnl[t] = cash + position * mid
    positions[t] = position

    # Reservation price: r = mid - q * gamma * sigma^2 * tau * mid
    reserv = mid - position * gamma * sigma**2 * tau * mid
    reserv_prices[t] = reserv

    # Optimal spread (fraction of mid): delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)
    opt_spread_frac = gamma * sigma**2 * tau + (2.0/gamma) * np.log(1 + gamma/kappa)
    opt_spreads[t] = opt_spread_frac

    # Post new quotes for next tick
    prev_bid = mid * (1 - opt_spread_frac / 2)
    prev_ask = mid * (1 + opt_spread_frac / 2)

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(mid_prices / mid_prices[0], index=range(n))

print(f"Total trades: {int((np.diff(positions) != 0).sum()):,}")
print(f"Final PnL: {pnl[-1]:,.2f}")
print(f"Max inventory: {int(np.max(np.abs(positions)))}")
print(f"Avg optimal spread: {np.mean(opt_spreads[1:])*10000:.2f} bps")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Mid-price + reservation price
window = min(5000, len(data))
axes[0, 0].plot(data["mid_price"].values[:window], linewidth=0.5, color="#6b7280", label="Mid Price", alpha=0.7)
axes[0, 0].plot(reserv_prices[:window], linewidth=0.5, color="#f59e0b", label="Reservation Price")
ax_inv = axes[0, 0].twinx()
ax_inv.plot(positions[:window], linewidth=0.5, color="#3b82f6", alpha=0.5)
ax_inv.set_ylabel("Inventory", color="#3b82f6")
axes[0, 0].set_title("Mid Price vs Reservation Price + Inventory")
axes[0, 0].legend(fontsize=8, loc="upper left")
axes[0, 0].grid(True, alpha=0.3)

# Optimal spread over time
axes[0, 1].plot(opt_spreads[1:] * 10000, linewidth=0.5, color="#10b981")
axes[0, 1].set_title("Optimal Spread (bps)")
axes[0, 1].set_ylabel("bps")
axes[0, 1].grid(True, alpha=0.3)

# PnL
axes[1, 0].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 0].set_title("Cumulative PnL")
axes[1, 0].grid(True, alpha=0.3)

# Spread surface: gamma vs sigma
gammas = np.linspace(0.01, 0.5, 20)
sigmas = np.linspace(0.1, 0.5, 20) * np.sqrt(1/252/390)
G, S = np.meshgrid(gammas, sigmas)
spread_surface = G * S**2 * T + (2.0/G) * np.log(1 + G/kappa)
im = axes[1, 1].contourf(gammas, sigmas * np.sqrt(252*390) * 100, spread_surface * 10000,
                          levels=20, cmap="YlOrRd")
plt.colorbar(im, ax=axes[1, 1], label="Spread (bps)")
axes[1, 1].set_xlabel("Risk Aversion (γ)")
axes[1, 1].set_ylabel("Annualized Vol (%)")
axes[1, 1].set_title("Optimal Spread Surface")

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_02: Order Book Imbalance Scalping
# ═══════════════════════════════════════════════════════════════════════════
def _hft02_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 50_000

# Mid-price random walk
mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n_steps))

# 10-level LOB snapshots
n_levels = 10
bid_sizes = rng.exponential(200, (n_steps, n_levels)).astype(int) + 10
ask_sizes = rng.exponential(200, (n_steps, n_levels)).astype(int) + 10

# Inject imbalance autocorrelation (trending order flow)
imb_trend = np.cumsum(rng.normal(0, 0.02, n_steps))
imb_trend = np.clip(imb_trend, -2, 2)
for t in range(n_steps):
    if imb_trend[t] > 0:
        bid_sizes[t] = (bid_sizes[t] * (1 + abs(imb_trend[t]) * 0.3)).astype(int)
    else:
        ask_sizes[t] = (ask_sizes[t] * (1 + abs(imb_trend[t]) * 0.3)).astype(int)

# Order book imbalance (top 3 levels)
top_n = 3
obi = (bid_sizes[:, :top_n].sum(axis=1) - ask_sizes[:, :top_n].sum(axis=1)) / \\
      (bid_sizes[:, :top_n].sum(axis=1) + ask_sizes[:, :top_n].sum(axis=1))

data = pd.DataFrame({
    "mid_price": mid,
    "obi": obi,
    "bid_total": bid_sizes.sum(axis=1),
    "ask_total": ask_sizes.sum(axis=1),
    "spread_bps": rng.exponential(2, n_steps) + 0.5,
})
print(f"Synthetic 10-level LOB: {n_steps:,} snapshots")
print(f"OBI range: [{obi.min():.3f}, {obi.max():.3f}], mean={obi.mean():.4f}")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

threshold = PARAMS.get("signal_threshold", 0.3)
spread_bps = PARAMS.get("spread_bps", 2) / 10000
max_pos = PARAMS.get("max_position", 50)

n = len(data)
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)
trade_obi = []  # OBI at trade time

for t in range(1, n):
    mid = data["mid_price"].iloc[t]
    obi = data["obi"].iloc[t]

    # Scalp: buy on strong bid imbalance, sell on strong ask imbalance
    if obi > threshold and position < max_pos:
        position += 1
        cash -= mid * (1 + spread_bps/2)
        trade_obi.append(obi)
    elif obi < -threshold and position > -max_pos:
        position -= 1
        cash += mid * (1 - spread_bps/2)
        trade_obi.append(obi)
    elif abs(obi) < 0.05 and position != 0:
        # Flatten near zero imbalance
        if position > 0:
            cash += mid * (1 - spread_bps/2)
            position -= 1
        else:
            cash -= mid * (1 + spread_bps/2)
            position += 1

    pnl[t] = cash + position * mid
    positions[t] = position

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["mid_price"].values / data["mid_price"].values[0], index=range(n))

print(f"Trades: {int((np.diff(positions) != 0).sum()):,}")
print(f"Final PnL: {pnl[-1]:,.2f}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# OBI histogram colored by direction
axes[0, 0].hist(data["obi"][data["obi"] > 0], bins=50, alpha=0.6, color="#10b981", label="Bid > Ask")
axes[0, 0].hist(data["obi"][data["obi"] < 0], bins=50, alpha=0.6, color="#ef4444", label="Ask > Bid")
axes[0, 0].axvline(threshold, color="#10b981", linestyle="--", label=f"+{threshold}")
axes[0, 0].axvline(-threshold, color="#ef4444", linestyle="--", label=f"-{threshold}")
axes[0, 0].set_title("Order Book Imbalance Distribution")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# PnL by OBI quintile
obi_vals = data["obi"].values[1:]
pnl_diff = np.diff(pnl)
quintiles = pd.qcut(obi_vals, 5, labels=["Q1(Ask)", "Q2", "Q3", "Q4", "Q5(Bid)"])
pnl_by_q = pd.Series(pnl_diff).groupby(np.asarray(quintiles)).mean()
colors = ["#ef4444", "#f59e0b", "#6b7280", "#3b82f6", "#10b981"]
axes[0, 1].bar(pnl_by_q.index, pnl_by_q.values, color=colors, alpha=0.7, edgecolor="white")
axes[0, 1].set_title("Avg PnL Change by OBI Quintile")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# OBI time series + position
window = 3000
axes[1, 0].plot(data["obi"].values[:window], linewidth=0.3, color="#8b5cf6", alpha=0.6)
ax2 = axes[1, 0].twinx()
ax2.plot(positions[:window], linewidth=0.5, color="#f59e0b", alpha=0.7)
ax2.set_ylabel("Position", color="#f59e0b")
axes[1, 0].set_title(f"OBI + Position (first {window} ticks)")
axes[1, 0].grid(True, alpha=0.3)

# Cumulative PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Cumulative PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_03: Queue Position Modeling (Hawkes Process)
# ═══════════════════════════════════════════════════════════════════════════
def _hft03_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 30_000

# Hawkes process: self-exciting order arrivals
mu = PARAMS.get("arrival_rate", 50.0)  # baseline intensity
alpha = 0.6  # excitation parameter
beta = 1.0   # decay parameter

# Simulate Hawkes intensity
intensity = np.zeros(n_steps)
events = np.zeros(n_steps)
intensity[0] = mu
for t in range(1, n_steps):
    # Decay previous excitation
    intensity[t] = mu + alpha * intensity[t-1] / (1 + beta)
    # Poisson sample from intensity
    events[t] = rng.poisson(min(intensity[t], 500))
    # Self-excitation from arrivals
    intensity[t] += alpha * events[t]

# Queue depth: cumulative arrivals minus fills
queue_depth = np.zeros(n_steps)
queue_ahead = np.zeros(n_steps)
fill_prob = np.zeros(n_steps)
mid = 100.0 + np.cumsum(rng.normal(0, 0.005, n_steps))

for t in range(1, n_steps):
    arrivals = int(events[t])
    queue_depth[t] = max(queue_depth[t-1] + arrivals - rng.poisson(mu * 0.8), 0)
    queue_ahead[t] = rng.uniform(0, max(queue_depth[t], 1))
    # Fill probability: P(fill) = 1 - exp(-lambda * (1 - ahead/total))
    total = max(queue_depth[t], 1)
    fill_prob[t] = 1 - np.exp(-mu * 0.01 * (1 - queue_ahead[t] / total))

data = pd.DataFrame({
    "mid_price": mid, "intensity": intensity, "events": events,
    "queue_depth": queue_depth, "queue_ahead": queue_ahead, "fill_prob": fill_prob,
})
print(f"Hawkes process: μ={mu}, α={alpha}, β={beta}")
print(f"Avg intensity: {intensity.mean():.1f}, Max: {intensity.max():.1f}")
print(f"Avg fill probability: {fill_prob.mean():.3f}")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

# Queue position strategy: join queue, hold if fill prob high, requeue if low
fill_threshold = PARAMS.get("fill_threshold", 0.3)
max_pos = PARAMS.get("max_position", 50)

n = len(data)
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)

for t in range(1, n):
    mid = data["mid_price"].iloc[t]
    fp = data["fill_prob"].iloc[t]

    # Enter when fill probability is high
    if fp > fill_threshold and position < max_pos:
        position += 1
        cash -= mid
    elif fp < fill_threshold * 0.3 and position > 0:
        # Exit when queue position degrades
        cash += mid
        position -= 1

    pnl[t] = cash + position * mid
    positions[t] = position

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["mid_price"].values / data["mid_price"].values[0], index=range(n))

print(f"Trades: {int((np.diff(positions) != 0).sum()):,}")
print(f"Final PnL: {pnl[-1]:,.2f}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Hawkes intensity
window = 5000
axes[0, 0].plot(data["intensity"].values[:window], linewidth=0.5, color="#f59e0b")
axes[0, 0].axhline(PARAMS.get("arrival_rate", 50), color="#6b7280", linestyle="--", label="Baseline μ")
axes[0, 0].set_title("Hawkes Process Intensity (self-exciting)")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Fill probability heatmap (queue depth vs intensity)
from matplotlib.colors import Normalize
fp = data["fill_prob"].values[1:]
qd = data["queue_depth"].values[1:]
axes[0, 1].scatter(qd[:5000], data["intensity"].values[1:5001], c=fp[:5000],
                   cmap="RdYlGn", s=2, alpha=0.5)
axes[0, 1].set_xlabel("Queue Depth")
axes[0, 1].set_ylabel("Arrival Intensity")
axes[0, 1].set_title("Fill Probability (color) by Queue State")
axes[0, 1].grid(True, alpha=0.3)

# Queue depth over time
axes[1, 0].plot(data["queue_depth"].values[:window], linewidth=0.5, color="#3b82f6")
axes[1, 0].set_title("Queue Depth Over Time")
axes[1, 0].grid(True, alpha=0.3)

# Cumulative PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Cumulative PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_04: Cross-Exchange Arbitrage
# ═══════════════════════════════════════════════════════════════════════════
def _hft04_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 50_000

# Two exchanges: correlated prices with latency gap
latency_ms = PARAMS.get("exchange_latency_ms", 5)
latency_ticks = max(1, int(latency_ms / 2))  # ~2ms per tick

# Exchange A: "fast" feed
returns_a = rng.normal(0, 0.0002, n_steps)
price_a = 100.0 * np.exp(np.cumsum(returns_a))

# Exchange B: "slow" feed — lagged version of A + noise
price_b = np.zeros(n_steps)
price_b[:latency_ticks] = price_a[:latency_ticks]
for t in range(latency_ticks, n_steps):
    price_b[t] = price_a[t - latency_ticks] + rng.normal(0, 0.001)

# Fee structure
fee_a = PARAMS.get("fee_bps_a", 2) / 10000
fee_b = PARAMS.get("fee_bps_b", 3) / 10000

data = pd.DataFrame({
    "price_a": price_a, "price_b": price_b,
    "divergence": price_a - price_b,
    "divergence_bps": (price_a - price_b) / price_a * 10000,
})
print(f"Dual-exchange: latency={latency_ms}ms ({latency_ticks} ticks)")
print(f"Mean divergence: {data['divergence_bps'].mean():.2f} bps")
print(f"Max divergence: {data['divergence_bps'].abs().max():.2f} bps")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

min_spread = PARAMS.get("min_spread_bps", 3) / 10000
max_pos = 20

n = len(data)
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)
arb_signals = np.zeros(n)

for t in range(1, n):
    pa = data["price_a"].iloc[t]
    pb = data["price_b"].iloc[t]
    spread = pa - pb

    arb_signals[t] = spread / pa

    # Arb: buy cheap, sell expensive (net of fees)
    net_spread = abs(spread) / pa - fee_a - fee_b

    if net_spread > min_spread:
        if spread > 0 and position < max_pos:
            # A expensive, B cheap: buy B sell A
            cash += pa * (1 - fee_a) - pb * (1 + fee_b)
            position += 1
        elif spread < 0 and position > -max_pos:
            # B expensive, A cheap: buy A sell B
            cash += pb * (1 - fee_b) - pa * (1 + fee_a)
            position -= 1

    pnl[t] = cash
    positions[t] = position

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["price_a"].values / data["price_a"].values[0], index=range(n))

arb_count = int((np.diff(positions) != 0).sum())
print(f"Arb trades: {arb_count:,}")
print(f"Final PnL: {pnl[-1]:,.2f}")
print(f"PnL per arb: {pnl[-1]/max(arb_count,1):.4f}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Dual-exchange price overlay
window = 3000
axes[0, 0].plot(data["price_a"].values[:window], linewidth=0.8, color="#f59e0b", label="Exchange A (fast)")
axes[0, 0].plot(data["price_b"].values[:window], linewidth=0.8, color="#3b82f6", label="Exchange B (slow)")
# Mark arb opportunities
arb_mask = np.abs(arb_signals[:window]) > min_spread
axes[0, 0].scatter(np.where(arb_mask)[0], data["price_a"].values[:window][arb_mask],
                   color="#ef4444", s=5, alpha=0.3, label="Arb Opportunity", zorder=5)
axes[0, 0].set_title("Dual-Exchange Prices")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Divergence
axes[0, 1].plot(data["divergence_bps"].values[:window], linewidth=0.3, color="#8b5cf6")
axes[0, 1].axhline(min_spread * 10000, color="#10b981", linestyle="--", alpha=0.5)
axes[0, 1].axhline(-min_spread * 10000, color="#ef4444", linestyle="--", alpha=0.5)
axes[0, 1].set_title("Price Divergence (bps)")
axes[0, 1].grid(True, alpha=0.3)

# Latency vs captured spread
captured = np.abs(np.diff(pnl))
axes[1, 0].hist(captured[captured > 0], bins=50, color="#f59e0b", alpha=0.7, edgecolor="none")
axes[1, 0].set_title("Captured Spread per Trade")
axes[1, 0].set_xlabel("Profit ($)")
axes[1, 0].grid(True, alpha=0.3)

# Cumulative PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Cumulative Arb PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_05: Short-Horizon Trade Imbalance (Net Order Flow)
# ═══════════════════════════════════════════════════════════════════════════
def _hft05_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 50_000

# Trade stream with signed trades (Lee-Ready classification)
mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n_steps))

# Signed volume: +1 buyer-initiated, -1 seller-initiated
# Add autocorrelation to simulate order flow persistence
signed_flow = np.zeros(n_steps)
signed_flow[0] = rng.choice([-1, 1])
for t in range(1, n_steps):
    # 60% chance of same direction (herding)
    if rng.random() < 0.6:
        signed_flow[t] = signed_flow[t-1]
    else:
        signed_flow[t] = rng.choice([-1, 1])

trade_sizes = rng.exponential(100, n_steps).astype(int) + 10
signed_volume = signed_flow * trade_sizes

data = pd.DataFrame({
    "mid_price": mid,
    "signed_volume": signed_volume,
    "trade_size": trade_sizes,
    "direction": signed_flow,
})
print(f"Trade stream: {n_steps:,} trades")
print(f"Buy ratio: {(signed_flow > 0).mean():.2%}")
print(f"Avg trade size: {trade_sizes.mean():.0f}")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

flow_window = PARAMS.get("flow_window", 50)
flow_threshold = PARAMS.get("flow_threshold", 0.3)  # net flow is normalized [-1, 1]
max_pos = PARAMS.get("max_position", 100)

n = len(data)
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)
net_flow = np.zeros(n)

# Rolling net order flow
for t in range(flow_window, n):
    window = data["signed_volume"].iloc[t-flow_window:t]
    net_flow[t] = window.sum() / window.abs().sum()  # normalized [-1, 1]

    mid = data["mid_price"].iloc[t]

    # Mean reversion: extreme imbalance tends to revert
    if net_flow[t] > flow_threshold and position > -max_pos:
        position -= 1  # sell into buying pressure (mean reversion)
        cash += mid
    elif net_flow[t] < -flow_threshold and position < max_pos:
        position += 1  # buy into selling pressure
        cash -= mid
    elif abs(net_flow[t]) < 0.1 and position != 0:
        # Flatten near equilibrium
        if position > 0: cash += mid; position -= 1
        else: cash -= mid; position += 1

    pnl[t] = cash + position * mid
    positions[t] = position

# Forward-fill the initial zero period so metrics aren't distorted
first_trade = np.argmax(pnl != 0)
if first_trade > 0:
    pnl[:first_trade] = pnl[first_trade]

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["mid_price"].values / data["mid_price"].values[0], index=range(n))

print(f"Trades: {int((np.diff(positions) != 0).sum()):,}")
print(f"Final PnL: {pnl[-1]:,.2f}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Net order flow time series
window = 5000
axes[0, 0].plot(net_flow[:window], linewidth=0.4, color="#3b82f6")
axes[0, 0].axhline(0, color="#6b7280", linewidth=0.5)
axes[0, 0].set_title(f"Net Order Flow (rolling {flow_window})")
axes[0, 0].grid(True, alpha=0.3)

# Autocorrelation of signed trades
from numpy import correlate
acf_lags = 50
signed = data["direction"].values
acf = np.array([np.corrcoef(signed[:-lag], signed[lag:])[0, 1] if lag > 0 else 1.0
                for lag in range(acf_lags)])
axes[0, 1].bar(range(acf_lags), acf, color="#f59e0b", alpha=0.7, edgecolor="none")
axes[0, 1].set_title("Autocorrelation of Signed Trade Flow")
axes[0, 1].set_xlabel("Lag")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Flow imbalance histogram
axes[1, 0].hist(net_flow[net_flow != 0], bins=60, color="#8b5cf6", alpha=0.7, edgecolor="none")
axes[1, 0].set_title("Net Flow Distribution")
axes[1, 0].grid(True, alpha=0.3)

# PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Cumulative PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_06: Iceberg Order Detection
# ═══════════════════════════════════════════════════════════════════════════
def _hft06_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 30_000

mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n_steps))

# Synthetic LOB with iceberg orders
# An iceberg refills at the same price level repeatedly
visible_size = rng.exponential(100, n_steps).astype(int) + 10
is_iceberg = np.zeros(n_steps, dtype=bool)
iceberg_total = np.zeros(n_steps)
refill_count = np.zeros(n_steps, dtype=int)

# Insert iceberg events (clusters of refills at same price)
iceberg_starts = rng.choice(range(100, n_steps - 50), size=n_steps // 200, replace=False)
for start in iceberg_starts:
    n_refills = rng.integers(3, 10)
    hidden_size = rng.integers(500, 5000)
    for j in range(min(n_refills, n_steps - start)):
        is_iceberg[start + j] = True
        iceberg_total[start + j] = hidden_size
        refill_count[start + j] = j + 1
        visible_size[start + j] = rng.integers(50, 150)  # small visible portion

data = pd.DataFrame({
    "mid_price": mid,
    "visible_size": visible_size,
    "is_iceberg": is_iceberg,
    "iceberg_total": iceberg_total,
    "refill_count": refill_count,
})

actual_icebergs = is_iceberg.sum()
print(f"Synthetic LOB: {n_steps:,} snapshots")
print(f"Actual iceberg events: {actual_icebergs:,} ({actual_icebergs/n_steps:.1%})")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

confidence_threshold = PARAMS.get("confidence_threshold", 0.7)

# Detection: look for consecutive small fills at same price level
n = len(data)
detected = np.zeros(n, dtype=bool)
detection_score = np.zeros(n)

price_levels = np.round(data["mid_price"].values, 1)  # quantize price

for t in range(3, n):
    # Count consecutive ticks at same price with small visible size
    same_price_count = 0
    for j in range(1, min(10, t)):
        if price_levels[t-j] == price_levels[t] and data["visible_size"].iloc[t-j] < 200:
            same_price_count += 1
        else:
            break

    # Detection score
    score = min(same_price_count / 5.0, 1.0)
    detection_score[t] = score
    detected[t] = score > confidence_threshold

# Evaluate detection accuracy
true_positives = (detected & data["is_iceberg"].values).sum()
false_positives = (detected & ~data["is_iceberg"].values).sum()
false_negatives = (~detected & data["is_iceberg"].values).sum()
true_negatives = (~detected & ~data["is_iceberg"].values).sum()

precision = true_positives / max(true_positives + false_positives, 1)
recall = true_positives / max(true_positives + false_negatives, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)

print(f"Detection Results (threshold={confidence_threshold}):")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1 Score: {f1:.3f}")
print(f"  TP={true_positives}, FP={false_positives}, FN={false_negatives}, TN={true_negatives}")

# Trading: trade ahead of detected icebergs (front-run hidden liquidity)
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)
max_pos = 20

for t in range(1, n):
    mid = data["mid_price"].iloc[t]
    if detected[t] and position < max_pos:
        position += 1
        cash -= mid
    elif not detected[t] and position > 0:
        cash += mid
        position -= 1
    pnl[t] = cash + position * mid
    positions[t] = position

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["mid_price"].values / data["mid_price"].values[0], index=range(n))
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Confusion matrix
cm = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])
im = axes[0, 0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0, 0].text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=14,
                        color="white" if cm[i,j] > cm.max()/2 else "black")
axes[0, 0].set_xticks([0, 1]); axes[0, 0].set_xticklabels(["No Iceberg", "Iceberg"])
axes[0, 0].set_yticks([0, 1]); axes[0, 0].set_yticklabels(["Not Detected", "Detected"])
axes[0, 0].set_title(f"Detection Confusion Matrix (F1={f1:.3f})")

# Refill pattern visualization
window = 2000
axes[0, 1].scatter(range(window), data["visible_size"].values[:window], s=2, alpha=0.3, color="#6b7280")
ice_mask = data["is_iceberg"].values[:window]
axes[0, 1].scatter(np.where(ice_mask)[0], data["visible_size"].values[:window][ice_mask],
                   s=8, color="#ef4444", alpha=0.6, label="Iceberg")
det_mask = detected[:window]
axes[0, 1].scatter(np.where(det_mask)[0], data["visible_size"].values[:window][det_mask],
                   s=15, facecolors="none", edgecolors="#10b981", linewidths=1, label="Detected")
axes[0, 1].set_title("Visible Size + Iceberg Events")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Detection score distribution
axes[1, 0].hist(detection_score[detection_score > 0], bins=50, color="#f59e0b", alpha=0.7, edgecolor="none")
axes[1, 0].axvline(confidence_threshold, color="#ef4444", linestyle="--", label=f"Threshold={confidence_threshold}")
axes[1, 0].set_title("Detection Score Distribution")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Cumulative PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_07: Latency Arbitrage
# ═══════════════════════════════════════════════════════════════════════════
def _hft07_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 50_000

latency_us = PARAMS.get("latency_advantage_us", 10)
latency_ticks = max(1, int(latency_us / 5))

# Fast feed (true price)
returns = rng.normal(0, 0.0002, n_steps)
fast_price = 100.0 * np.exp(np.cumsum(returns))

# Slow feed (delayed)
slow_price = np.zeros(n_steps)
slow_price[:latency_ticks] = fast_price[:latency_ticks]
for t in range(latency_ticks, n_steps):
    slow_price[t] = fast_price[t - latency_ticks]

# Stale quote detection
is_stale = np.abs(fast_price - slow_price) / fast_price > 0.0001  # >1 bps divergence
stale_duration = np.zeros(n_steps)
for t in range(1, n_steps):
    if is_stale[t]:
        stale_duration[t] = stale_duration[t-1] + 1
    else:
        stale_duration[t] = 0

data = pd.DataFrame({
    "fast_price": fast_price, "slow_price": slow_price,
    "is_stale": is_stale, "stale_duration": stale_duration,
    "divergence_bps": (fast_price - slow_price) / fast_price * 10000,
})
print(f"Latency arb: advantage={latency_us}μs ({latency_ticks} ticks)")
print(f"Stale quotes: {is_stale.sum():,} ({is_stale.mean():.1%})")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

min_edge = PARAMS.get("min_edge_bps", 1.0) / 10000
max_pos = 30

n = len(data)
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)

for t in range(1, n):
    fp = data["fast_price"].iloc[t]
    sp = data["slow_price"].iloc[t]
    edge = (fp - sp) / fp

    if abs(edge) > min_edge:
        if edge > 0 and position < max_pos:
            # Fast moved up, slow hasn't — buy on slow venue
            cash -= sp
            position += 1
        elif edge < 0 and position > -max_pos:
            # Fast moved down — sell on slow venue
            cash += sp
            position -= 1
    elif abs(edge) < min_edge * 0.3 and position != 0:
        # Close when prices converge
        if position > 0: cash += sp; position -= 1
        else: cash -= sp; position += 1

    pnl[t] = cash + position * fp
    positions[t] = position

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["fast_price"].values / data["fast_price"].values[0], index=range(n))

print(f"Trades: {int((np.diff(positions) != 0).sum()):,}")
print(f"Final PnL: {pnl[-1]:,.2f}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Stale quote duration histogram
durations = data["stale_duration"].values[data["stale_duration"].values > 0]
axes[0, 0].hist(durations, bins=50, color="#f59e0b", alpha=0.7, edgecolor="none")
axes[0, 0].set_title("Stale Quote Duration (ticks)")
axes[0, 0].set_xlabel("Duration")
axes[0, 0].grid(True, alpha=0.3)

# PnL per latency bucket
div = np.abs(data["divergence_bps"].values)
pnl_diff = np.diff(pnl)
buckets = pd.cut(div[1:], bins=5)
pnl_by_bucket = pd.Series(pnl_diff).groupby(np.asarray(buckets)).mean()
axes[0, 1].bar(range(len(pnl_by_bucket)), pnl_by_bucket.values, color="#3b82f6", alpha=0.7, edgecolor="white")
axes[0, 1].set_xticklabels([str(b)[:10] for b in pnl_by_bucket.index], rotation=45, fontsize=7)
axes[0, 1].set_title("Avg PnL by Divergence Bucket")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Fast vs slow price
window = 3000
axes[1, 0].plot(data["fast_price"].values[:window], linewidth=0.5, color="#f59e0b", label="Fast")
axes[1, 0].plot(data["slow_price"].values[:window], linewidth=0.5, color="#3b82f6", label="Slow")
axes[1, 0].set_title("Fast vs Slow Price Feed")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Cumulative PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Cumulative Arb PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_08: Smart Order Router (Multi-Venue)
# ═══════════════════════════════════════════════════════════════════════════
def _hft08_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 30_000
n_venues = PARAMS.get("venue_count", 3)

mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n_steps))

# Venue properties
venues = []
for v in range(n_venues):
    venues.append({
        "name": f"Venue_{chr(65+v)}",
        "maker_rebate": (2.5 - v * 0.8) / 10000,  # decreasing rebate
        "taker_fee": (3.0 + v * 0.5) / 10000,       # increasing fee
        "fill_prob": 0.7 - v * 0.1,                  # decreasing fill probability
        "avg_depth": 500 + v * 200,                   # increasing depth
    })

# Generate per-venue quotes
venue_data = {}
for v, venue in enumerate(venues):
    spread = rng.exponential(2, n_steps) + 1 + v * 0.5  # wider spreads at worse venues
    venue_data[f"spread_{v}"] = spread
    venue_data[f"depth_{v}"] = rng.exponential(venue["avg_depth"], n_steps).astype(int)

data = pd.DataFrame({"mid_price": mid, **venue_data})

print(f"Smart Order Router: {n_venues} venues")
for v in venues:
    print(f"  {v['name']}: maker_rebate={v['maker_rebate']*10000:.1f}bps, "
          f"taker_fee={v['taker_fee']*10000:.1f}bps, fill_prob={v['fill_prob']:.0%}")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

# Route to minimize expected cost = spread + fee - rebate + impact
max_pos = PARAMS.get("max_position", 50)

n = len(data)
position = 0
cash = 0.0
pnl = np.zeros(n)
positions = np.zeros(n)
venue_allocation = np.zeros((n, len(venues)))

for t in range(1, n):
    mid = data["mid_price"].iloc[t]

    # Compute expected cost per venue
    best_venue, best_cost = 0, np.inf
    for v_idx, venue in enumerate(venues):
        spread = data[f"spread_{v_idx}"].iloc[t] / 10000
        depth = data[f"depth_{v_idx}"].iloc[t]
        fill_prob = venue["fill_prob"]
        cost = spread + venue["taker_fee"] - venue["maker_rebate"] + 0.01 / max(depth, 1)
        expected_cost = cost / max(fill_prob, 0.01)
        if expected_cost < best_cost:
            best_cost = expected_cost
            best_venue = v_idx

    venue_allocation[t, best_venue] = 1

    # Simple momentum signal to generate trades
    if t > 20:
        mom = (mid - data["mid_price"].iloc[t-20]) / data["mid_price"].iloc[t-20]
        if mom > 0.001 and position < max_pos:
            position += 1
            cash -= mid * (1 + best_cost)
        elif mom < -0.001 and position > -max_pos:
            position -= 1
            cash += mid * (1 - best_cost)

    pnl[t] = cash + position * mid
    positions[t] = position

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["mid_price"].values / data["mid_price"].values[0], index=range(n))

print(f"Trades: {int((np.diff(positions) != 0).sum()):,}")
print(f"Final PnL: {pnl[-1]:,.2f}")
print(f"\\nVenue routing allocation:")
for v_idx, venue in enumerate(venues):
    alloc = venue_allocation[:, v_idx].sum() / max(venue_allocation.sum(), 1) * 100
    print(f"  {venue['name']}: {alloc:.1f}%")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Venue allocation pie chart
alloc_pcts = [venue_allocation[:, v].sum() for v in range(len(venues))]
if sum(alloc_pcts) > 0:
    labels = [v["name"] for v in venues]
    colors = ["#f59e0b", "#3b82f6", "#10b981", "#ef4444", "#8b5cf6"][:len(venues)]
    axes[0, 0].pie(alloc_pcts, labels=labels, colors=colors, autopct="%1.1f%%",
                   textprops={"fontsize": 10})
    axes[0, 0].set_title("Venue Routing Allocation")

# Cost breakdown per venue
venue_names = [v["name"] for v in venues]
spreads = [np.mean(data[f"spread_{v}"].values) for v in range(len(venues))]
fees = [venues[v]["taker_fee"] * 10000 for v in range(len(venues))]
rebates = [venues[v]["maker_rebate"] * 10000 for v in range(len(venues))]
x = np.arange(len(venues))
width = 0.25
axes[0, 1].bar(x - width, spreads, width, label="Spread (bps)", color="#f59e0b", alpha=0.7)
axes[0, 1].bar(x, fees, width, label="Taker Fee (bps)", color="#ef4444", alpha=0.7)
axes[0, 1].bar(x + width, rebates, width, label="Maker Rebate (bps)", color="#10b981", alpha=0.7)
axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(venue_names)
axes[0, 1].set_title("Cost Breakdown by Venue")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Rolling venue selection
window = 1000
for v_idx in range(len(venues)):
    rolling_alloc = pd.Series(venue_allocation[:, v_idx]).rolling(window).mean()
    axes[1, 0].plot(rolling_alloc.values, linewidth=1,
                   color=["#f59e0b", "#3b82f6", "#10b981"][v_idx % 3],
                   label=venues[v_idx]["name"])
axes[1, 0].set_title(f"Rolling Venue Selection (window={window})")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Cumulative PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# hft_09: RL Market Maker (Q-Learning)
# ═══════════════════════════════════════════════════════════════════════════
def _hft09_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_steps = 20_000

# Mid-price: GBM
sigma = 0.3
dt = 1.0 / 252 / 390
mid = 100.0 * np.exp(np.cumsum(rng.normal(0, sigma * np.sqrt(dt), n_steps)))

data = pd.DataFrame({"mid_price": mid})
data["returns"] = pd.Series(mid).pct_change()
data["vol"] = data["returns"].rolling(max(50, 2)).std()
print(f"RL Market Maker: {n_steps:,} ticks")
""")

    strategy = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

# Tabular Q-Learning market maker
n_inv_buckets = 5   # inventory states
n_vol_buckets = 3   # volatility states
n_actions = 3       # 0=tighten, 1=hold, 2=widen spread

learning_rate = PARAMS.get("learning_rate", 0.1)
gamma_rl = PARAMS.get("gamma", 0.95)
eps_start = PARAMS.get("eps_start", 1.0)
eps_decay = PARAMS.get("eps_decay", 0.995)
inv_penalty = PARAMS.get("inventory_penalty", 0.01)
n_episodes = PARAMS.get("n_episodes", 5)

# State discretization
inv_bins = np.linspace(-50, 50, n_inv_buckets + 1)
vol_quantiles = data["vol"].dropna().quantile(np.linspace(0, 1, n_vol_buckets + 1)).values

def discretize_state(inventory, vol_val):
    i_bucket = min(np.searchsorted(inv_bins[1:], inventory), n_inv_buckets - 1)
    v_bucket = min(np.searchsorted(vol_quantiles[1:], vol_val), n_vol_buckets - 1)
    return i_bucket * n_vol_buckets + v_bucket

n_states = n_inv_buckets * n_vol_buckets
Q = np.zeros((n_states, n_actions))
spread_levels = [0.0002, 0.0005, 0.001]  # tighten, hold, widen

episode_rewards = []
rng_rl = np.random.default_rng(SEED + 42)

for episode in range(n_episodes):
    epsilon = max(eps_start * (eps_decay ** episode), 0.05)
    position = 0
    cash = 0.0
    total_reward = 0

    for t in range(100, len(data) - 1):
        mid = data["mid_price"].iloc[t]
        vol = data["vol"].iloc[t]
        if np.isnan(vol): continue

        state = discretize_state(position, vol)

        # Epsilon-greedy action
        if rng_rl.random() < epsilon:
            action = rng_rl.integers(0, n_actions)
        else:
            action = np.argmax(Q[state])

        spread = spread_levels[action]
        bid = mid * (1 - spread)
        ask = mid * (1 + spread)

        # Simulate fills
        next_mid = data["mid_price"].iloc[t + 1]
        if next_mid <= bid and position < 50:
            position += 1; cash -= bid
        elif next_mid >= ask and position > -50:
            position -= 1; cash += ask

        # Reward: PnL change - inventory penalty
        new_pnl = cash + position * next_mid
        reward = (next_mid - mid) * position - inv_penalty * abs(position)
        total_reward += reward

        next_vol = data["vol"].iloc[t + 1]
        if np.isnan(next_vol): next_vol = vol
        next_state = discretize_state(position, next_vol)

        # Q-update
        Q[state, action] += learning_rate * (reward + gamma_rl * np.max(Q[next_state]) - Q[state, action])

    episode_rewards.append(total_reward)

# --- Final greedy rollout ---
position = 0
cash = 0.0
n = len(data)
pnl = np.zeros(n)
positions = np.zeros(n)
actions_taken = np.zeros(n, dtype=int)

for t in range(100, n - 1):
    mid = data["mid_price"].iloc[t]
    vol = data["vol"].iloc[t]
    if np.isnan(vol): continue

    state = discretize_state(position, vol)
    action = np.argmax(Q[state])
    actions_taken[t] = action

    spread = spread_levels[action]
    bid = mid * (1 - spread)
    ask = mid * (1 + spread)

    next_mid = data["mid_price"].iloc[t + 1]
    if next_mid <= bid and position < 50:
        position += 1; cash -= bid
    elif next_mid >= ask and position > -50:
        position -= 1; cash += ask

    pnl[t] = cash + position * next_mid
    positions[t] = position

# Forward-fill the initial zero period so metrics aren't distorted
first_trade = np.argmax(pnl != 0)
if first_trade > 0:
    pnl[:first_trade] = pnl[first_trade]

equity_curve = pd.Series(pnl, index=range(n))
equity_curve = equity_curve - equity_curve.min() + 1
benchmark_equity = pd.Series(data["mid_price"].values / data["mid_price"].values[0], index=range(n))

print(f"Q-Learning: {n_episodes} episodes, final ε={epsilon:.3f}")
print(f"Final PnL: {pnl[-1]:,.2f}")
print(f"Trades: {int((np.diff(positions) != 0).sum()):,}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Q-table heatmap
im = axes[0, 0].imshow(Q.T, cmap="RdYlGn", aspect="auto")
axes[0, 0].set_xlabel("State (inventory × vol)")
axes[0, 0].set_ylabel("Action")
axes[0, 0].set_yticks([0, 1, 2]); axes[0, 0].set_yticklabels(["Tighten", "Hold", "Widen"])
axes[0, 0].set_title("Q-Table Heatmap")
plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

# Learning curve
axes[0, 1].plot(episode_rewards, "o-", color="#f59e0b", markersize=4)
axes[0, 1].set_title("Episode Total Reward")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].grid(True, alpha=0.3)

# Policy visualization: action distribution by inventory bucket
for a, (label, color) in enumerate(zip(["Tighten", "Hold", "Widen"], ["#10b981", "#f59e0b", "#ef4444"])):
    action_counts = []
    for ib in range(n_inv_buckets):
        states = [ib * n_vol_buckets + vb for vb in range(n_vol_buckets)]
        preferred = sum(1 for s in states if np.argmax(Q[s]) == a)
        action_counts.append(preferred / n_vol_buckets)
    axes[1, 0].bar(np.arange(n_inv_buckets) + a * 0.25, action_counts, 0.25,
                   label=label, color=color, alpha=0.7)
inv_labels = [f"[{inv_bins[i]:.0f},{inv_bins[i+1]:.0f})" for i in range(n_inv_buckets)]
axes[1, 0].set_xticks(np.arange(n_inv_buckets) + 0.25)
axes[1, 0].set_xticklabels(inv_labels, fontsize=7)
axes[1, 0].set_title("Policy: Action Preference by Inventory")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3, axis="y")

# PnL
axes[1, 1].plot(pnl, linewidth=0.5, color="#10b981")
axes[1, 1].set_title("Greedy Rollout PnL")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, strategy, viz]


# ═══════════════════════════════════════════════════════════════════════════
# HFT-specific metrics (shared, adapts to whatever pnl/positions exist)
# ═══════════════════════════════════════════════════════════════════════════
def _hft_metrics_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import numpy as np

trades = np.diff(positions) != 0
n_trades = trades.sum()

pnl_per_trade = pnl[-1] / max(n_trades, 1)
trade_indices = np.where(trades)[0]
avg_holding = np.mean(np.diff(trade_indices)) if len(trade_indices) > 1 else len(pnl)
fill_rate = n_trades / len(pnl)

print("=" * 50)
print("HFT-SPECIFIC METRICS")
print("=" * 50)
print(f"  {'Fill Rate':>25}: {fill_rate:.4f}")
print(f"  {'PnL per Trade':>25}: {pnl_per_trade:.4f}")
print(f"  {'Avg Holding (ticks)':>25}: {avg_holding:.1f}")
print(f"  {'Max Inventory':>25}: {int(np.max(np.abs(positions)))}")
print(f"  {'Final Inventory':>25}: {int(positions[-1])}")
""")


# ═══════════════════════════════════════════════════════════════════════════
# Builder — dispatches to per-project cells
# ═══════════════════════════════════════════════════════════════════════════
_STRATEGY_MAP = {
    "hft_01_adaptive_market_making": _hft01_cells,
    "hft_02_order_book_scalping": _hft02_cells,
    "hft_03_queue_position": _hft03_cells,
    "hft_04_cross_exchange_arb": _hft04_cells,
    "hft_05_trade_imbalance": _hft05_cells,
    "hft_06_iceberg_detection": _hft06_cells,
    "hft_07_latency_arb": _hft07_cells,
    "hft_08_smart_order_router": _hft08_cells,
    "hft_09_rl_market_maker": _hft09_cells,
}


def build_hft_notebook(card: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    pid = card["project_id"]
    params = {p["name"]: p["default"] for p in card.get("interactive_params", [])}

    # Resolve strategy cells
    cell_fn = _STRATEGY_MAP.get(pid, _hft01_cells)
    domain_cells = cell_fn()

    nb.cells = [
        title_cell(card["title"], "HFT Strategies",
                   card.get("long_description", card.get("short_description", "")), pid),
        environment_setup_cell(requires_gpu=card.get("requires_gpu", False)),
        config_cell(params),
        nbf.v4.new_markdown_cell(f"## Data Generation — {card.get('title', 'HFT Strategy')}"),
        domain_cells[0],  # data
        nbf.v4.new_markdown_cell("## Strategy Implementation"),
        domain_cells[1],  # strategy
        nbf.v4.new_markdown_cell("## Visualization"),
        domain_cells[2],  # viz
        performance_viz_cell(),
        metrics_cell(),
        _hft_metrics_cell(),
        sensitivity_cell(card.get("interactive_params", [{}])[0].get("name", "spread_bps")),
        export_cell(pid),
        summary_cell(card["title"]),
    ]
    return nb
