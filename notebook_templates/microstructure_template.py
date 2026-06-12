"""Microstructure notebook template — for market_microstructure_engines/ (2 projects).

Each project gets distinct analysis:
  engines_01 — LOB Simulator: Hawkes process arrivals, price-time priority matching, depth heatmap
  engines_02 — Feed Handler: Binary message encoding/decoding, throughput vs size, format comparison
"""

from __future__ import annotations
import nbformat as nbf
from .common_cells import (
    title_cell, environment_setup_cell, config_cell,
    data_acquisition_synthetic, export_cell, summary_cell,
)


# ═══════════════════════════════════════════════════════════════════════════
# engines_01: LOB Simulator — Hawkes arrivals + matching engine
# ═══════════════════════════════════════════════════════════════════════════
def _engines01_cells() -> list[nbf.NotebookNode]:
    sim = nbf.v4.new_code_cell("""import numpy as np, pandas as pd, time
from collections import defaultdict

rng = np.random.default_rng(SEED)
n_events = PARAMS.get("n_events", 100_000)

# --- Hawkes Process Order Arrivals ---
mu = PARAMS.get("hawkes_mu", 50.0)  # baseline intensity
alpha_h = 0.7  # excitation
beta_h = 1.2   # decay

intensity = np.zeros(n_events)
intensity[0] = mu
events_per_tick = np.zeros(n_events, dtype=int)

for t in range(1, n_events):
    intensity[t] = mu + alpha_h * intensity[t-1] / (1 + beta_h)
    events_per_tick[t] = rng.poisson(min(intensity[t], 300))
    intensity[t] += alpha_h * events_per_tick[t]

# Generate orders from Hawkes arrivals
orders = []
mid_price = 100.0
for t in range(n_events):
    n_orders = max(events_per_tick[t], 1)
    for _ in range(min(n_orders, 5)):
        side = rng.choice(["bid", "ask"])
        offset = rng.exponential(0.02)
        price = round(mid_price + (offset if side == "ask" else -offset), 2)
        size = int(rng.exponential(100) + 10)
        orders.append({"tick": t, "side": side, "price": price, "size": size,
                       "type": rng.choice(["new", "new", "new", "cancel"])})
    # Small mid-price drift
    mid_price += rng.normal(0, 0.002)

orders_df = pd.DataFrame(orders)
print(f"Hawkes LOB: μ={mu}, α={alpha_h}, β={beta_h}")
print(f"Generated {len(orders_df):,} orders from {n_events:,} ticks")
print(f"Avg intensity: {intensity.mean():.1f}, Peak: {intensity.max():.1f}")

# --- Price-Time Priority Matching Engine ---
bids = defaultdict(list)  # price -> [(tick, size)]
asks = defaultdict(list)
trades = []
depth_history = []

t0 = time.perf_counter()
for _, order in orders_df.iterrows():
    if order["type"] == "cancel":
        book = bids if order["side"] == "bid" else asks
        if order["price"] in book and book[order["price"]]:
            book[order["price"]].pop(0)
            if not book[order["price"]]: del book[order["price"]]
        continue

    book = bids if order["side"] == "bid" else asks
    book[order["price"]].append((order["tick"], order["size"]))

    # Try matching
    while bids and asks:
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        if best_bid >= best_ask:
            b = bids[best_bid][0]
            a = asks[best_ask][0]
            fill_sz = min(b[1], a[1])
            trades.append({"tick": order["tick"], "price": best_ask, "size": fill_sz})
            if b[1] == fill_sz:
                bids[best_bid].pop(0)
                if not bids[best_bid]: del bids[best_bid]
            else:
                bids[best_bid][0] = (b[0], b[1] - fill_sz)
            if a[1] == fill_sz:
                asks[best_ask].pop(0)
                if not asks[best_ask]: del asks[best_ask]
            else:
                asks[best_ask][0] = (a[0], a[1] - fill_sz)
        else:
            break

    # Snapshot every 500 orders
    if _ % 500 == 0:
        bid_depth = sum(sum(s for _, s in v) for v in bids.values())
        ask_depth = sum(sum(s for _, s in v) for v in asks.values())
        best_b = max(bids.keys()) if bids else 0
        best_a = min(asks.keys()) if asks else 999
        depth_history.append({"idx": _, "bid_depth": bid_depth, "ask_depth": ask_depth,
                              "best_bid": best_b, "best_ask": best_a,
                              "spread": (best_a - best_b) if best_b > 0 and best_a < 999 else 0,
                              "n_bid_levels": len(bids), "n_ask_levels": len(asks)})

elapsed = time.perf_counter() - t0
trades_df = pd.DataFrame(trades)
depth_df = pd.DataFrame(depth_history)

print(f"\\nMatching engine: {elapsed:.3f}s ({len(orders_df)/elapsed:,.0f} orders/sec)")
print(f"Trades executed: {len(trades_df):,}")
print(f"Fill rate: {len(trades_df) / len(orders_df):.2%}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Hawkes intensity
axes[0, 0].plot(intensity[:5000], linewidth=0.5, color="#f59e0b")
axes[0, 0].axhline(mu, color="#6b7280", linestyle="--", alpha=0.5, label=f"Baseline μ={mu}")
axes[0, 0].set_title("Hawkes Process Intensity (Self-Exciting)")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Depth evolution
if len(depth_df) > 0:
    axes[0, 1].fill_between(range(len(depth_df)), depth_df["bid_depth"],
                           alpha=0.4, color="#10b981", label="Bid Depth")
    axes[0, 1].fill_between(range(len(depth_df)), -depth_df["ask_depth"].values,
                           alpha=0.4, color="#ef4444", label="Ask Depth")
    axes[0, 1].set_title("Order Book Depth Over Time")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

# Spread histogram
if len(depth_df) > 0:
    spreads = depth_df["spread"][depth_df["spread"] > 0] * 10000 / 100
    axes[1, 0].hist(spreads, bins=50, color="#3b82f6", alpha=0.7, edgecolor="none")
    axes[1, 0].set_title("Bid-Ask Spread Distribution (bps)")
    axes[1, 0].set_xlabel("Spread (bps)")
    axes[1, 0].grid(True, alpha=0.3)

# Arrival intensity vs fill rate (scatter)
chunk = 100
n_chunks = min(len(orders_df) // chunk, 200)
chunk_intensity = [intensity[i*chunk:(i+1)*chunk].mean() for i in range(n_chunks)]
chunk_fills = []
for i in range(n_chunks):
    chunk_trades = trades_df[(trades_df["tick"] >= i*chunk) & (trades_df["tick"] < (i+1)*chunk)]
    chunk_fills.append(len(chunk_trades) / chunk)
axes[1, 1].scatter(chunk_intensity, chunk_fills, s=5, alpha=0.5, color="#8b5cf6")
axes[1, 1].set_xlabel("Avg Intensity")
axes[1, 1].set_ylabel("Fill Rate")
axes[1, 1].set_title("Arrival Intensity vs Fill Rate")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
    return [sim, viz]


# ═══════════════════════════════════════════════════════════════════════════
# engines_02: Feed Handler — Message encoding, throughput, format comparison
# ═══════════════════════════════════════════════════════════════════════════
def _engines02_cells() -> list[nbf.NotebookNode]:
    bench = nbf.v4.new_code_cell("""import numpy as np, pandas as pd, struct, json, time

rng = np.random.default_rng(SEED)
n_messages = PARAMS.get("n_messages", 50_000)
queue_size = PARAMS.get("queue_size", 65536)

# --- Binary message format ---
# Market data: [seq(4B), ts(8B), symbol(8B), price(8B), size(4B), side(1B)]
msg_format = "!Id8sdIb"
msg_size = struct.calcsize(msg_format)

symbols = [b"SPY\\x00\\x00\\x00\\x00\\x00", b"QQQ\\x00\\x00\\x00\\x00\\x00",
           b"AAPL\\x00\\x00\\x00\\x00", b"TSLA\\x00\\x00\\x00\\x00"]

# Generate test messages
seq_nums = np.arange(1, n_messages + 1)
timestamps = np.cumsum(rng.exponential(0.0001, n_messages))
prices = 100.0 + np.cumsum(rng.normal(0, 0.001, n_messages))
sizes = rng.exponential(200, n_messages).astype(int) + 10
sides = rng.choice([0, 1], n_messages)
sym_indices = rng.choice(len(symbols), n_messages)

# --- Benchmark 3 encoding formats ---
sample_size = min(10000, n_messages)
results = {}

# 1. struct.pack (binary)
latencies_struct = np.zeros(sample_size)
for i in range(sample_size):
    t0 = time.perf_counter_ns()
    msg = struct.pack(msg_format, int(seq_nums[i]), timestamps[i],
                      symbols[sym_indices[i]], prices[i], int(sizes[i]), int(sides[i]))
    _ = struct.unpack(msg_format, msg)
    latencies_struct[i] = (time.perf_counter_ns() - t0) / 1000
results["struct"] = {"latencies": latencies_struct, "msg_size": msg_size}

# 2. JSON
latencies_json = np.zeros(sample_size)
for i in range(sample_size):
    d = {"seq": int(seq_nums[i]), "ts": float(timestamps[i]),
         "sym": symbols[sym_indices[i]].decode().strip("\\x00"),
         "px": float(prices[i]), "sz": int(sizes[i]), "side": int(sides[i])}
    t0 = time.perf_counter_ns()
    encoded = json.dumps(d).encode()
    _ = json.loads(encoded)
    latencies_json[i] = (time.perf_counter_ns() - t0) / 1000
results["JSON"] = {"latencies": latencies_json, "msg_size": len(encoded)}

# 3. CSV-like (string)
latencies_csv = np.zeros(sample_size)
for i in range(sample_size):
    t0 = time.perf_counter_ns()
    encoded = f"{seq_nums[i]},{timestamps[i]},{symbols[sym_indices[i]].decode().strip(chr(0))},{prices[i]},{sizes[i]},{sides[i]}".encode()
    parts = encoded.decode().split(",")
    latencies_csv[i] = (time.perf_counter_ns() - t0) / 1000
results["CSV"] = {"latencies": latencies_csv, "msg_size": len(encoded)}

# Print comparison
print(f"Message Encoding Benchmark ({sample_size:,} roundtrips):")
print(f"  {'Format':>8} {'p50 (μs)':>10} {'p99 (μs)':>10} {'Msg Size':>10} {'Throughput':>12}")
print("-" * 55)
for fmt, data in results.items():
    lats = data["latencies"]
    tp = sample_size / (lats.sum() / 1e6)  # msgs/sec
    print(f"  {fmt:>8} {np.median(lats):>10.2f} {np.percentile(lats, 99):>10.2f} "
          f"{data['msg_size']:>8}B {tp:>11,.0f}/s")

# --- Queue simulation ---
print(f"\\nQueue Simulation (buffer={queue_size:,}):")
queue = []
queue_depths = np.zeros(n_messages)
dropped = 0

for i in range(n_messages):
    # Arrivals: Poisson batch
    n_arrivals = rng.poisson(3)
    for _ in range(n_arrivals):
        if len(queue) < queue_size:
            queue.append(i)
        else:
            dropped += 1

    # Processing: drain some messages
    n_process = rng.poisson(3)
    for _ in range(min(n_process, len(queue))):
        queue.pop(0)

    queue_depths[i] = len(queue)

print(f"  Peak queue depth: {int(queue_depths.max()):,}")
print(f"  Avg queue depth: {queue_depths.mean():.1f}")
print(f"  Messages dropped: {dropped:,} ({dropped/n_messages:.2%})")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Throughput comparison bar chart
fmt_names = list(results.keys())
throughputs = [sample_size / (results[f]["latencies"].sum() / 1e6) for f in fmt_names]
msg_sizes = [results[f]["msg_size"] for f in fmt_names]
colors = ["#10b981", "#f59e0b", "#ef4444"]

axes[0, 0].bar(fmt_names, throughputs, color=colors, alpha=0.7, edgecolor="white")
axes[0, 0].set_title("Encoding Throughput (roundtrips/sec)")
axes[0, 0].set_ylabel("msgs/sec")
axes[0, 0].grid(True, alpha=0.3, axis="y")

# Latency CDF per format
for fmt, color in zip(fmt_names, colors):
    lats = np.sort(results[fmt]["latencies"])
    cdf = np.arange(1, len(lats)+1) / len(lats) * 100
    axes[0, 1].plot(lats, cdf, linewidth=1.5, color=color, label=fmt)
axes[0, 1].set_xlabel("Latency (μs)")
axes[0, 1].set_ylabel("Percentile (%)")
axes[0, 1].set_title("Encoding Latency CDF")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, max(np.percentile(results[f]["latencies"], 99) for f in fmt_names) * 1.5)

# Queue depth over time
axes[1, 0].plot(queue_depths[:10000], linewidth=0.5, color="#3b82f6")
axes[1, 0].axhline(queue_size, color="#ef4444", linestyle="--", alpha=0.5, label=f"Buffer limit={queue_size:,}")
axes[1, 0].set_title("Queue Depth Over Time")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Message size comparison
axes[1, 1].bar(fmt_names, msg_sizes, color=colors, alpha=0.7, edgecolor="white")
axes[1, 1].set_title("Message Size (bytes)")
axes[1, 1].set_ylabel("bytes")
axes[1, 1].grid(True, alpha=0.3, axis="y")
# Annotate with size
for i, (name, sz) in enumerate(zip(fmt_names, msg_sizes)):
    axes[1, 1].text(i, sz + 2, f"{sz}B", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
""")
    return [bench, viz]


# ═══════════════════════════════════════════════════════════════════════════
# Infrastructure export
# ═══════════════════════════════════════════════════════════════════════════
def _infra_export_cell(project_id: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(f"""import json
from datetime import datetime

results_export = {{
    "project_id": "{project_id}",
    "timestamp": datetime.now().isoformat(),
    "type": "engine_benchmark",
    "metrics": {{}},
}}

with open("results.json", "w") as f:
    json.dump(results_export, f, indent=2, default=str)
print("Results exported to results.json")
""")


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════
def build_microstructure_notebook(card: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    pid = card["project_id"]
    params = {p["name"]: p["default"] for p in card.get("interactive_params", [])}

    head = [
        title_cell(card["title"], "Microstructure Engines",
                   card.get("long_description", card.get("short_description", "")), pid),
        environment_setup_cell(requires_gpu=False),
        config_cell(params),
    ]

    if "lob" in pid or "engines_01" in pid:
        # engines_01: LOB with Hawkes process + matching engine
        sim_cell, viz_cell = _engines01_cells()
        nb.cells = head + [
            nbf.v4.new_markdown_cell("## LOB Simulator — Hawkes Process Arrivals + Price-Time Matching"),
            sim_cell,
            nbf.v4.new_markdown_cell("## Visualization"),
            viz_cell,
            _infra_export_cell(pid),
            summary_cell(card["title"]),
        ]

    elif "feed" in pid or "handler" in pid or "engines_02" in pid:
        # engines_02: Feed handler encoding/throughput
        bench_cell, viz_cell = _engines02_cells()
        nb.cells = head + [
            nbf.v4.new_markdown_cell("## Feed Handler — Binary Message Encoding & Queue Simulation"),
            bench_cell,
            nbf.v4.new_markdown_cell("## Benchmark Visualization"),
            viz_cell,
            _infra_export_cell(pid),
            summary_cell(card["title"]),
        ]

    else:
        # Fallback
        sim_cell, viz_cell = _engines01_cells()
        nb.cells = head + [sim_cell, viz_cell,
                           _infra_export_cell(pid), summary_cell(card["title"])]

    return nb
