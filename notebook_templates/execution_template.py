"""Execution notebook template — for market_microstructure_execution/ (3 projects).

Each project gets distinct analysis:
  exec_01 — LOB Simulator: Order book reconstruction, price-time priority matching
  exec_02 — Execution Algorithms: TWAP vs VWAP vs POV vs Almgren-Chriss comparison
  exec_03 — Realtime Feed Handler: Message rate, latency, gap detection
"""

from __future__ import annotations
import nbformat as nbf
from .common_cells import (
    title_cell, environment_setup_cell, config_cell,
    data_acquisition_yfinance, data_acquisition_synthetic,
    performance_viz_cell, metrics_cell, export_cell, summary_cell,
    get_ticker_for_project,
)


# ═══════════════════════════════════════════════════════════════════════════
# exec_01: LOB Simulator — Order book reconstruction + matching engine
# ═══════════════════════════════════════════════════════════════════════════
def _exec01_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd

rng = np.random.default_rng(SEED)
n_events = PARAMS.get("num_events", 50_000)

# Simulate order stream: {type, side, price, size, timestamp}
mid_price = 100.0
event_types = rng.choice(["new", "new", "new", "cancel", "modify"], n_events)
sides = rng.choice(["bid", "ask"], n_events)
prices = mid_price + rng.normal(0, 0.05, n_events)
sizes = rng.exponential(100, n_events).astype(int) + 10
timestamps = np.cumsum(rng.exponential(0.001, n_events))  # seconds

orders = pd.DataFrame({
    "timestamp": timestamps, "type": event_types, "side": sides,
    "price": np.round(prices, 2), "size": sizes,
})

print(f"Order stream: {n_events:,} events over {timestamps[-1]:.1f}s")
print(f"Event breakdown: {dict(orders['type'].value_counts())}")
""")

    engine = nbf.v4.new_code_cell("""import numpy as np, pandas as pd, time
from collections import defaultdict

# Price-time priority matching engine
class SimpleMatchingEngine:
    def __init__(self):
        self.bids = defaultdict(list)  # price -> [(time, size, id)]
        self.asks = defaultdict(list)
        self.trades = []
        self.order_id = 0
        self.depth_snapshots = []

    def add_order(self, side, price, size, timestamp):
        self.order_id += 1
        book = self.bids if side == "bid" else self.asks
        book[price].append((timestamp, size, self.order_id))
        self._try_match(timestamp)
        return self.order_id

    def cancel_order(self, side, price):
        book = self.bids if side == "bid" else self.asks
        if price in book and book[price]:
            book[price].pop(0)  # remove oldest (FIFO)
            if not book[price]: del book[price]  # clean up empty level

    def _try_match(self, timestamp):
        while self.bids and self.asks:
            best_bid = max(self.bids.keys()) if self.bids else 0
            best_ask = min(self.asks.keys()) if self.asks else float("inf")
            if best_bid >= best_ask:
                bid_order = self.bids[best_bid][0]
                ask_order = self.asks[best_ask][0]
                fill_size = min(bid_order[1], ask_order[1])
                fill_price = best_ask  # price-time priority: taker gets maker's price

                self.trades.append({
                    "timestamp": timestamp, "price": fill_price,
                    "size": fill_size, "bid_id": bid_order[2], "ask_id": ask_order[2]
                })

                # Update remaining sizes
                if bid_order[1] == fill_size:
                    self.bids[best_bid].pop(0)
                    if not self.bids[best_bid]: del self.bids[best_bid]
                else:
                    self.bids[best_bid][0] = (bid_order[0], bid_order[1] - fill_size, bid_order[2])

                if ask_order[1] == fill_size:
                    self.asks[best_ask].pop(0)
                    if not self.asks[best_ask]: del self.asks[best_ask]
                else:
                    self.asks[best_ask][0] = (ask_order[0], ask_order[1] - fill_size, ask_order[2])
            else:
                break

    def snapshot(self):
        bid_depth = sum(sum(s for _, s, _ in v) for v in self.bids.values())
        ask_depth = sum(sum(s for _, s, _ in v) for v in self.asks.values())
        best_bid = max(self.bids.keys()) if self.bids else 0
        best_ask = min(self.asks.keys()) if self.asks else 0
        return {"bid_depth": bid_depth, "ask_depth": ask_depth,
                "best_bid": best_bid, "best_ask": best_ask,
                "spread": best_ask - best_bid if best_bid > 0 and best_ask < 1e6 else 0,
                "n_levels_bid": len(self.bids), "n_levels_ask": len(self.asks)}

# Run matching engine
engine = SimpleMatchingEngine()
snapshot_interval = max(1, n_events // 500)

t0 = time.perf_counter()
for i in range(len(orders)):
    row = orders.iloc[i]
    if row["type"] == "new":
        engine.add_order(row["side"], row["price"], row["size"], row["timestamp"])
    elif row["type"] == "cancel":
        engine.cancel_order(row["side"], row["price"])
    elif row["type"] == "modify":
        engine.cancel_order(row["side"], row["price"])
        engine.add_order(row["side"], row["price"] + rng.normal(0, 0.01), row["size"], row["timestamp"])

    if i % snapshot_interval == 0:
        snap = engine.snapshot()
        snap["event_idx"] = i
        engine.depth_snapshots.append(snap)

elapsed = time.perf_counter() - t0
throughput = n_events / elapsed

trades_df = pd.DataFrame(engine.trades) if engine.trades else pd.DataFrame()
snapshots_df = pd.DataFrame(engine.depth_snapshots)

print(f"Matching engine: {elapsed:.3f}s, throughput={throughput:,.0f} events/sec")
print(f"Trades executed: {len(trades_df):,}")
print(f"Fill rate: {len(trades_df) / n_events:.2%}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Depth evolution over time
if len(snapshots_df) > 0:
    axes[0, 0].plot(snapshots_df["event_idx"], snapshots_df["bid_depth"],
                   linewidth=0.8, color="#10b981", label="Bid Depth")
    axes[0, 0].plot(snapshots_df["event_idx"], snapshots_df["ask_depth"],
                   linewidth=0.8, color="#ef4444", label="Ask Depth")
    axes[0, 0].set_title("Order Book Depth Over Time")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

# Spread evolution
if len(snapshots_df) > 0:
    spreads = snapshots_df["spread"]
    axes[0, 1].plot(snapshots_df["event_idx"], spreads * 10000 / 100,
                   linewidth=0.5, color="#f59e0b")
    axes[0, 1].set_title("Bid-Ask Spread (bps)")
    axes[0, 1].grid(True, alpha=0.3)

# Trade price distribution
if len(trades_df) > 0:
    axes[1, 0].hist(trades_df["price"], bins=50, color="#3b82f6", alpha=0.7, edgecolor="none")
    axes[1, 0].set_title("Trade Price Distribution")
    axes[1, 0].set_xlabel("Price")
    axes[1, 0].grid(True, alpha=0.3)

# Trade size distribution
if len(trades_df) > 0:
    axes[1, 1].hist(trades_df["size"], bins=50, color="#8b5cf6", alpha=0.7, edgecolor="none")
    axes[1, 1].set_title("Trade Size Distribution")
    axes[1, 1].set_xlabel("Size")
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [data, engine, viz]


# ═══════════════════════════════════════════════════════════════════════════
# exec_02: Execution Algorithms — TWAP vs VWAP vs POV vs Almgren-Chriss
# ═══════════════════════════════════════════════════════════════════════════
def _exec02_cells() -> list[nbf.NotebookNode]:
    algo_cell = nbf.v4.new_code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

rng = np.random.default_rng(SEED)
n = len(price)

# Synthetic volume (U-shaped)
volume = rng.exponential(1e6, n) * (1 + 0.3 * np.sin(np.linspace(0, np.pi, n)))

# Execution parameters
order_size = PARAMS.get("order_size", 100_000)
participation_rate = PARAMS.get("participation_rate", 0.1)
sigma = returns.std() * np.sqrt(252)
alpha_impact = PARAMS.get("impact_alpha", 0.5)

# --- Algorithm implementations ---
def run_algo(algo_name, prices, vols, order_sz, n_slices=20):
    exec_prices = []
    arrival = prices[0]

    if algo_name == "TWAP":
        sz_per_slice = order_sz / n_slices
        for i in range(min(n_slices, len(prices))):
            impact = alpha_impact * np.sqrt(sz_per_slice / max(vols[i], 1)) * sigma / np.sqrt(252) * prices[i]
            exec_prices.append(prices[i] + impact)

    elif algo_name == "VWAP":
        vol_weights = vols[:n_slices] / max(vols[:n_slices].sum(), 1)
        for i in range(min(n_slices, len(prices))):
            sz = order_sz * vol_weights[i]
            impact = alpha_impact * np.sqrt(sz / max(vols[i], 1)) * sigma / np.sqrt(252) * prices[i]
            exec_prices.append(prices[i] + impact)

    elif algo_name == "POV":
        filled = 0
        for i in range(len(prices)):
            if filled >= order_sz: break
            sz = min(vols[i] * participation_rate, order_sz - filled)
            impact = alpha_impact * np.sqrt(sz / max(vols[i], 1)) * sigma / np.sqrt(252) * prices[i]
            exec_prices.append(prices[i] + impact)
            filled += sz

    elif algo_name == "Almgren-Chriss":
        # Optimal execution: minimize IS + timing risk
        # Closed-form solution: exponential schedule
        lam = PARAMS.get("risk_aversion_ac", 1e-6)
        kappa_val = np.sqrt(lam * sigma**2 / (alpha_impact * sigma / np.sqrt(252)))
        remaining = order_sz
        for i in range(min(n_slices, len(prices))):
            tau = (n_slices - i) / n_slices
            trade_rate = remaining * kappa_val / max(np.sinh(kappa_val * tau), 1e-9)
            sz = min(trade_rate, remaining)
            impact = alpha_impact * np.sqrt(abs(sz) / max(vols[i], 1)) * sigma / np.sqrt(252) * prices[i]
            exec_prices.append(prices[i] + impact)
            remaining -= sz

    return np.array(exec_prices), arrival

# Run all algorithms on multiple windows
window_size = 30  # trading days per execution
n_windows = min(20, n // window_size)
algo_results = {algo: [] for algo in ["TWAP", "VWAP", "POV", "Almgren-Chriss"]}

for w in range(n_windows):
    start = w * window_size
    p = price.values[start:start+window_size]
    v = volume[start:start+window_size]
    if len(p) < 10: continue

    for algo in algo_results:
        ep, arrival = run_algo(algo, p, v, order_size)
        if len(ep) > 0:
            avg_px = np.mean(ep)
            is_bps = (avg_px - arrival) / arrival * 10000
            algo_results[algo].append(is_bps)

# Print comparison
print("Execution Algorithm Comparison (avg Implementation Shortfall, bps):")
print(f"  {'Algorithm':>15} {'Mean IS':>10} {'Std IS':>10} {'Max IS':>10}")
print("-" * 50)
for algo, is_list in algo_results.items():
    if is_list:
        print(f"  {algo:>15} {np.mean(is_list):>+10.2f} {np.std(is_list):>10.2f} {np.max(is_list):>+10.2f}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Sample execution trajectories (overlaid)
p_sample = price.values[:window_size]
v_sample = volume[:window_size]
colors = {"TWAP": "#f59e0b", "VWAP": "#3b82f6", "POV": "#10b981", "Almgren-Chriss": "#ef4444"}
markers = {"TWAP": "o", "VWAP": "s", "POV": "^", "Almgren-Chriss": "D"}

for algo in algo_results:
    ep, _ = run_algo(algo, p_sample, v_sample, order_size)
    axes[0, 0].plot(ep, f"{markers[algo]}-", color=colors[algo], markersize=3, label=algo, alpha=0.7)
axes[0, 0].axhline(p_sample[0], color="#6b7280", linestyle="--", alpha=0.5, label="Arrival")
axes[0, 0].set_title("Execution Trajectories (Sample Window)")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# IS comparison bar chart
algo_names = list(algo_results.keys())
algo_means = [np.mean(algo_results[a]) if algo_results[a] else 0 for a in algo_names]
algo_stds = [np.std(algo_results[a]) if algo_results[a] else 0 for a in algo_names]
bar_colors = [colors[a] for a in algo_names]
axes[0, 1].bar(algo_names, algo_means, yerr=algo_stds, color=bar_colors, alpha=0.7,
              edgecolor="white", capsize=5)
axes[0, 1].set_title("Avg Implementation Shortfall (bps)")
axes[0, 1].set_ylabel("IS (bps)")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# IS distribution per algo
for algo in algo_names:
    if algo_results[algo]:
        axes[1, 0].hist(algo_results[algo], bins=15, alpha=0.4, color=colors[algo],
                       label=algo, edgecolor="none")
axes[1, 0].set_title("IS Distribution by Algorithm")
axes[1, 0].set_xlabel("IS (bps)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Market impact curve
order_fracs = np.linspace(0.001, 0.05, 20)
impact_bps = alpha_impact * np.sqrt(order_fracs) * sigma * 10000
axes[1, 1].plot(order_fracs * 100, impact_bps, "o-", color="#f59e0b", linewidth=2)
axes[1, 1].fill_between(order_fracs * 100, 0, impact_bps, alpha=0.1, color="#f59e0b")
axes[1, 1].set_xlabel("Order Size (% of ADV)")
axes[1, 1].set_ylabel("Expected Impact (bps)")
axes[1, 1].set_title("Market Impact Curve (√-model)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

    backtest = nbf.v4.new_code_cell("""import pandas as pd, numpy as np

# Build equity curve from benchmark with execution cost drag
returns_s = price.pct_change().dropna()
tc_drag = np.mean(algo_results.get("VWAP", [5])) / 10000 / 252  # daily cost drag
strategy_returns_raw = returns_s - tc_drag
equity_curve = (1 + strategy_returns_raw).cumprod()
benchmark_equity = (1 + returns_s).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"VWAP cost drag: {tc_drag*252*10000:.2f} bps/year")
""")
    return [algo_cell, viz, backtest]


# ═══════════════════════════════════════════════════════════════════════════
# exec_03: Realtime Feed Handler — Message rate, latency, gap detection
# ═══════════════════════════════════════════════════════════════════════════
def _exec03_cells() -> list[nbf.NotebookNode]:
    data = nbf.v4.new_code_cell("""import numpy as np, pandas as pd, struct, time

rng = np.random.default_rng(SEED)
n_messages = PARAMS.get("n_messages", 100_000)
buffer_size = PARAMS.get("buffer_size", 4096)

# Simulate binary message stream (market data feed)
# Message format: [seq_num(4B), timestamp(8B), price(8B), size(4B), side(1B)] = 25 bytes
msg_format = "!IddiB"  # network byte order: seq(I), time(d), price(d), size(I), side(B)
msg_size = struct.calcsize(msg_format)

# Generate message sequence with occasional gaps
seq_nums = np.arange(1, n_messages + 1)
# Insert gaps (missed messages)
gap_positions = rng.choice(range(100, n_messages - 100), size=n_messages // 500, replace=False)
for pos in gap_positions:
    seq_nums[pos:] += rng.integers(1, 5)  # skip 1-4 sequence numbers

timestamps = np.cumsum(rng.exponential(0.0001, n_messages))  # ~10K msgs/sec
prices = 100.0 + np.cumsum(rng.normal(0, 0.001, n_messages))
sizes = rng.exponential(100, n_messages).astype(int) + 10
sides = rng.choice([0, 1], n_messages)  # 0=bid, 1=ask

# --- Encode/decode benchmark ---
encode_times = np.zeros(min(10000, n_messages))
decode_times = np.zeros(min(10000, n_messages))

for i in range(len(encode_times)):
    t0 = time.perf_counter_ns()
    msg = struct.pack(msg_format, int(seq_nums[i]), timestamps[i], prices[i], int(sizes[i]), int(sides[i]))
    encode_times[i] = (time.perf_counter_ns() - t0) / 1000  # μs

    t0 = time.perf_counter_ns()
    _ = struct.unpack(msg_format, msg)
    decode_times[i] = (time.perf_counter_ns() - t0) / 1000

# --- Gap detection ---
seq_diffs = np.diff(seq_nums)
gaps = np.where(seq_diffs > 1)[0]
gap_sizes = seq_diffs[gaps] - 1

print(f"Feed handler: {n_messages:,} messages, {msg_size} bytes/msg")
print(f"Message rate: {1 / np.mean(np.diff(timestamps)):,.0f} msgs/sec")
print(f"Gaps detected: {len(gaps)} (total {gap_sizes.sum()} missed messages)")
print(f"\\nEncode: p50={np.median(encode_times):.2f}μs, p99={np.percentile(encode_times, 99):.2f}μs")
print(f"Decode: p50={np.median(decode_times):.2f}μs, p99={np.percentile(decode_times, 99):.2f}μs")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Message rate over time
window = 1000
msg_rates = np.zeros(len(timestamps) - window)
for i in range(len(msg_rates)):
    dt = timestamps[i + window] - timestamps[i]
    msg_rates[i] = window / max(dt, 1e-9)
axes[0, 0].plot(msg_rates[:5000], linewidth=0.5, color="#f59e0b")
axes[0, 0].set_title("Message Rate (msgs/sec)")
axes[0, 0].grid(True, alpha=0.3)

# Latency CDF
sorted_decode = np.sort(decode_times)
cdf = np.arange(1, len(sorted_decode)+1) / len(sorted_decode) * 100
axes[0, 1].plot(sorted_decode, cdf, linewidth=1.5, color="#3b82f6", label="Decode")
sorted_encode = np.sort(encode_times)
cdf_enc = np.arange(1, len(sorted_encode)+1) / len(sorted_encode) * 100
axes[0, 1].plot(sorted_encode, cdf_enc, linewidth=1.5, color="#f59e0b", label="Encode")
axes[0, 1].set_xlabel("Latency (μs)")
axes[0, 1].set_ylabel("Percentile (%)")
axes[0, 1].set_title("Encode/Decode Latency CDF")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, np.percentile(decode_times, 99.5))

# Gap detection chart
if len(gaps) > 0:
    axes[1, 0].bar(gaps[:50], gap_sizes[:50], color="#ef4444", alpha=0.7, edgecolor="none")
    axes[1, 0].set_title(f"Sequence Gaps ({len(gaps)} detected)")
    axes[1, 0].set_xlabel("Message Index")
    axes[1, 0].set_ylabel("Gap Size (missed msgs)")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
else:
    axes[1, 0].text(0.5, 0.5, "No gaps detected", ha="center", va="center")

# Format comparison: struct vs JSON
import json
json_times = np.zeros(1000)
struct_times = np.zeros(1000)
for i in range(1000):
    msg_dict = {"seq": int(seq_nums[i]), "ts": float(timestamps[i]),
                "px": float(prices[i]), "sz": int(sizes[i]), "side": int(sides[i])}
    t0 = time.perf_counter_ns()
    _ = json.dumps(msg_dict).encode()
    json_times[i] = (time.perf_counter_ns() - t0) / 1000

    t0 = time.perf_counter_ns()
    _ = struct.pack(msg_format, int(seq_nums[i]), timestamps[i], prices[i], int(sizes[i]), int(sides[i]))
    struct_times[i] = (time.perf_counter_ns() - t0) / 1000

axes[1, 1].bar(["struct.pack", "JSON"], [np.median(struct_times), np.median(json_times)],
              color=["#10b981", "#f59e0b"], alpha=0.7, edgecolor="white")
axes[1, 1].set_title("Encoding p50 Latency (μs)")
axes[1, 1].set_ylabel("μs")
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()
""")
    return [data, viz]


# ═══════════════════════════════════════════════════════════════════════════
# Infrastructure export (no equity curve)
# ═══════════════════════════════════════════════════════════════════════════
def _infra_export_cell(project_id: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(f"""import json
from datetime import datetime

results_export = {{
    "project_id": "{project_id}",
    "timestamp": datetime.now().isoformat(),
    "type": "infrastructure_benchmark",
    "metrics": {{}},
}}

with open("results.json", "w") as f:
    json.dump(results_export, f, indent=2, default=str)
print("Results exported to results.json")
""")


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════
def build_execution_notebook(card: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    pid = card["project_id"]
    params = {p["name"]: p["default"] for p in card.get("interactive_params", [])}

    head = [
        title_cell(card["title"], "Microstructure Execution",
                   card.get("long_description", card.get("short_description", "")), pid),
        environment_setup_cell(requires_gpu=False),
        config_cell(params),
    ]

    if "lob" in pid or ("exec_01" in pid):
        # exec_01: LOB simulator with matching engine
        data_cell, engine_cell, viz_cell = _exec01_cells()
        nb.cells = head + [
            nbf.v4.new_markdown_cell("## Order Stream Generation"),
            data_cell,
            nbf.v4.new_markdown_cell("## Matching Engine — Price-Time Priority"),
            engine_cell,
            nbf.v4.new_markdown_cell("## Order Book Visualization"),
            viz_cell,
            _infra_export_cell(pid),
            summary_cell(card["title"]),
        ]

    elif "execution" in pid or "algo" in pid or "exec_02" in pid:
        # exec_02: 4 execution algorithms compared
        algo_cell, viz_cell, bt_cell = _exec02_cells()
        nb.cells = head + [
            data_acquisition_yfinance(get_ticker_for_project(pid, fallback="QQQ")),
            nbf.v4.new_markdown_cell("## Execution Algorithm Comparison — TWAP vs VWAP vs POV vs Almgren-Chriss"),
            algo_cell,
            nbf.v4.new_markdown_cell("## Execution Analysis"),
            viz_cell,
            nbf.v4.new_markdown_cell("## Cost Impact Backtest"),
            bt_cell,
            performance_viz_cell(),
            metrics_cell(),
            export_cell(pid),
            summary_cell(card["title"]),
        ]

    elif "feed" in pid or "handler" in pid or "exec_03" in pid:
        # exec_03: Feed handler / message processing
        data_cell, viz_cell = _exec03_cells()
        nb.cells = head + [
            nbf.v4.new_markdown_cell("## Feed Handler — Binary Message Processing & Gap Detection"),
            data_cell,
            nbf.v4.new_markdown_cell("## Throughput & Latency Analysis"),
            viz_cell,
            _infra_export_cell(pid),
            summary_cell(card["title"]),
        ]

    else:
        # Fallback
        algo_cell, viz_cell, bt_cell = _exec02_cells()
        nb.cells = head + [
            data_acquisition_yfinance("SPY"),
            algo_cell, viz_cell, bt_cell,
            performance_viz_cell(), metrics_cell(),
            export_cell(pid), summary_cell(card["title"]),
        ]

    return nb
