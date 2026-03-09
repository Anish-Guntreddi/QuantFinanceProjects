"""Risk engineering notebook template — for risk_engineering/ (4 projects).

Each project gets distinct analysis:
  risk_01 — Portfolio optimization: MVO vs Risk Parity vs Black-Litterman
  risk_02 — Research reproducibility: Walk-forward CV, bootstrap Sharpe significance
  risk_03 — Time-series storage: CSV vs Parquet vs HDF5 benchmarks (infrastructure, no equity curve)
  risk_04 — C++ latency utilities: Data structure benchmarks (infrastructure, no equity curve)
"""

from __future__ import annotations
import nbformat as nbf
from .common_cells import (
    title_cell, environment_setup_cell, config_cell,
    data_acquisition_yfinance, data_acquisition_synthetic,
    performance_viz_cell, metrics_cell, export_cell, summary_cell,
    monthly_heatmap_cell, get_ticker_for_project,
)


# ═══════════════════════════════════════════════════════════════════════════
# risk_01: Portfolio Optimization — MVO vs Risk Parity vs Black-Litterman
# ═══════════════════════════════════════════════════════════════════════════
def _risk01_optimization_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import numpy as np, pandas as pd, matplotlib.pyplot as plt

returns_df = close.pct_change().dropna()
asset_names = list(close.columns) if close.ndim > 1 else ["Asset"]
n_assets = len(asset_names)

mu = returns_df.mean().values * 252
cov = returns_df.cov().values * 252
rf = PARAMS.get("risk_free_rate", 0.04)
max_w = PARAMS.get("max_weight", 0.3)

rng = np.random.default_rng(SEED)

# --- Method 1: Mean-Variance Optimization (random search) ---
n_portfolios = 10000
results = np.zeros((n_portfolios, 3))
weights_all = []
for i in range(n_portfolios):
    w = rng.random(n_assets); w /= w.sum()
    w = np.clip(w, 0, max_w); w /= w.sum()
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sr = (ret - rf) / vol if vol > 0 else 0
    results[i] = [vol, ret, sr]
    weights_all.append(w)

mvo_idx = results[:, 2].argmax()
mvo_weights = weights_all[mvo_idx]

# --- Method 2: Risk Parity (inverse-vol) ---
vols = np.sqrt(np.diag(cov))
rp_weights = (1.0 / np.where(vols > 0, vols, 1e-6))
rp_weights = np.clip(rp_weights / rp_weights.sum(), 0, max_w)
rp_weights /= rp_weights.sum()

# --- Method 3: Black-Litterman (simplified) ---
# Market-cap proxy: equal weight as prior, blend with MVO view
tau = 0.05
pi = cov @ (np.ones(n_assets) / n_assets)  # equilibrium returns
P = np.eye(n_assets)  # each asset has a view
Q = mu  # views = historical means
omega = np.diag(np.diag(tau * cov))  # uncertainty

# BL posterior
inv_tau_cov = np.linalg.inv(tau * cov)
inv_omega = np.linalg.inv(omega)
bl_mu = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P) @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)
bl_cov = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
# Optimal BL weights
bl_weights = np.linalg.solve(cov, bl_mu)
bl_weights = np.clip(bl_weights / bl_weights.sum(), 0, max_w)
bl_weights /= bl_weights.sum()

# Store for later use
portfolio_weights = {"MVO": mvo_weights, "Risk Parity": rp_weights, "Black-Litterman": bl_weights}

print("Portfolio Weights:")
for method, w in portfolio_weights.items():
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sr = (ret - rf) / vol if vol > 0 else 0
    print(f"\\n  {method}: Return={ret:.2%}, Vol={vol:.2%}, Sharpe={sr:.3f}")
    for name, wi in zip(asset_names, w):
        print(f"    {name}: {wi:.2%}")

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Efficient frontier with 3 portfolios
scatter = axes[0].scatter(results[:, 0]*100, results[:, 1]*100, c=results[:, 2],
                          cmap="viridis", alpha=0.15, s=3)
plt.colorbar(scatter, ax=axes[0], label="Sharpe Ratio")

markers = {"MVO": ("*", "#f59e0b", 200), "Risk Parity": ("D", "#3b82f6", 100),
           "Black-Litterman": ("P", "#10b981", 120)}
for method, (marker, color, sz) in markers.items():
    w = portfolio_weights[method]
    ret, vol = w @ mu, np.sqrt(w @ cov @ w)
    axes[0].scatter(vol*100, ret*100, marker=marker, s=sz, c=color,
                   edgecolors="white", zorder=5, label=method)
axes[0].set_xlabel("Annualized Volatility (%)")
axes[0].set_ylabel("Annualized Return (%)")
axes[0].set_title("Efficient Frontier — 3 Optimization Methods")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Weight comparison bar chart
x = np.arange(n_assets)
width = 0.25
for i, (method, color) in enumerate([("MVO", "#f59e0b"), ("Risk Parity", "#3b82f6"), ("Black-Litterman", "#10b981")]):
    axes[1].bar(x + i*width, portfolio_weights[method]*100, width, label=method, color=color, alpha=0.7)
axes[1].set_xticks(x + width); axes[1].set_xticklabels(asset_names, fontsize=9)
axes[1].set_ylabel("Weight (%)")
axes[1].set_title("Portfolio Weight Comparison")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()
""")


def _risk01_backtest_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd, numpy as np

returns_df = close.pct_change().dropna()

# Backtest: use MVO weights (static) for equity curve
w = portfolio_weights["MVO"]
strategy_returns_raw = (returns_df.values @ w)
strategy_returns_raw = pd.Series(strategy_returns_raw, index=returns_df.index)
equity_curve = (1 + strategy_returns_raw).cumprod()

# Benchmark: equal-weight
ew_ret = returns_df.mean(axis=1)
benchmark_equity = (1 + ew_ret).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"MVO final equity: {equity_curve.iloc[-1]:.4f}")
print(f"EW final equity:  {benchmark_equity.iloc[-1]:.4f}")
""")


# ═══════════════════════════════════════════════════════════════════════════
# risk_02: Research Reproducibility — Walk-Forward CV + Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════
def _risk02_cells() -> list[nbf.NotebookNode]:
    cv_cell = nbf.v4.new_code_cell("""import numpy as np, pandas as pd, matplotlib.pyplot as plt

returns_df = close.pct_change().dropna()
if returns_df.ndim > 1:
    returns_s = returns_df.mean(axis=1)
else:
    returns_s = returns_df

n = len(returns_s)
n_splits = PARAMS.get("n_splits", 5)
embargo = PARAMS.get("embargo_days", 10)

# --- Walk-Forward Cross-Validation with Purging + Embargo ---
# Simple momentum strategy for testing
lookback = 20
signal = returns_s.rolling(lookback).mean().shift(1)
strat_ret = signal.apply(np.sign) * returns_s
strat_ret = strat_ret.dropna()

fold_size = len(strat_ret) // n_splits
fold_sharpes = []
fold_returns = []

print(f"Walk-Forward CV: {n_splits} folds, embargo={embargo} days")
print(f"Fold size: {fold_size} days\\n")

for fold in range(n_splits):
    test_start = fold * fold_size
    test_end = min(test_start + fold_size, len(strat_ret))
    # Train: everything except test + embargo buffer
    train_mask = np.ones(len(strat_ret), dtype=bool)
    purge_start = max(test_start - embargo, 0)
    purge_end = min(test_end + embargo, len(strat_ret))
    train_mask[purge_start:purge_end] = False

    test_ret = strat_ret.iloc[test_start:test_end]
    if len(test_ret) < 10:
        continue

    sr = test_ret.mean() / test_ret.std() * np.sqrt(252) if test_ret.std() > 0 else 0
    fold_sharpes.append(sr)
    fold_returns.append(test_ret.mean() * 252)
    print(f"  Fold {fold+1}: Sharpe={sr:+.3f}, Ann. Return={test_ret.mean()*252:+.2%}, Days={len(test_ret)}")

# --- Bootstrap Sharpe Significance Test ---
n_bootstrap = 5000
rng = np.random.default_rng(SEED)
full_sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0

bootstrap_sharpes = []
for _ in range(n_bootstrap):
    sample = rng.choice(strat_ret.values, size=len(strat_ret), replace=True)
    bs_sr = np.mean(sample) / np.std(sample) * np.sqrt(252) if np.std(sample) > 0 else 0
    bootstrap_sharpes.append(bs_sr)

bootstrap_sharpes = np.array(bootstrap_sharpes)
p_value = (bootstrap_sharpes <= 0).mean()
ci_lower = np.percentile(bootstrap_sharpes, 2.5)
ci_upper = np.percentile(bootstrap_sharpes, 97.5)

# --- Deflated Sharpe Ratio (simplified) ---
# Adjusts for multiple testing: DSR = Sharpe * sqrt(n) / sqrt(1 + skew*sharpe/6 + (kurt-3)*sharpe^2/24)
from scipy.stats import norm
skew = float(pd.Series(strat_ret).skew())
kurt = float(pd.Series(strat_ret).kurtosis()) + 3  # excess -> raw
dsr_denom = np.sqrt(1 + skew * full_sharpe / 6 + (kurt - 3) * full_sharpe**2 / 24)
dsr = full_sharpe / max(dsr_denom, 1e-9)

print(f"\\n--- Statistical Significance ---")
print(f"Full-sample Sharpe: {full_sharpe:+.3f}")
print(f"Bootstrap p-value (Sharpe > 0): {1 - p_value:.4f}")
print(f"95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]")
print(f"Deflated Sharpe Ratio: {dsr:+.3f}")
print(f"Skewness: {skew:.3f}, Kurtosis: {kurt:.3f}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Walk-forward Sharpe by fold
colors_bar = ["#10b981" if s > 0 else "#ef4444" for s in fold_sharpes]
axes[0, 0].bar(range(1, len(fold_sharpes)+1), fold_sharpes, color=colors_bar, alpha=0.7, edgecolor="white")
axes[0, 0].axhline(np.mean(fold_sharpes), color="#f59e0b", linestyle="--", label=f"Mean={np.mean(fold_sharpes):.3f}")
axes[0, 0].set_title("Walk-Forward Sharpe by Fold")
axes[0, 0].set_xlabel("Fold")
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3, axis="y")

# Bootstrap Sharpe distribution
axes[0, 1].hist(bootstrap_sharpes, bins=60, color="#3b82f6", alpha=0.6, edgecolor="none", density=True)
axes[0, 1].axvline(full_sharpe, color="#f59e0b", linewidth=2, label=f"Observed: {full_sharpe:+.3f}")
axes[0, 1].axvline(0, color="#ef4444", linewidth=1, linestyle="--", label="Null: 0")
axes[0, 1].axvline(ci_lower, color="#6b7280", linestyle=":", label=f"95% CI: [{ci_lower:+.2f}, {ci_upper:+.2f}]")
axes[0, 1].axvline(ci_upper, color="#6b7280", linestyle=":")
axes[0, 1].set_title(f"Bootstrap Sharpe Distribution (p={1-p_value:.3f})")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Cumulative strategy return per fold
offset = 0
for fold in range(min(n_splits, len(fold_sharpes))):
    test_start = fold * fold_size
    test_end = min(test_start + fold_size, len(strat_ret))
    fold_eq = (1 + strat_ret.iloc[test_start:test_end]).cumprod()
    color = "#10b981" if fold_sharpes[fold] > 0 else "#ef4444"
    axes[1, 0].plot(range(offset, offset + len(fold_eq)), fold_eq.values, linewidth=1, color=color)
    offset += len(fold_eq)
axes[1, 0].set_title("Cumulative Returns by Fold (colored by Sharpe sign)")
axes[1, 0].grid(True, alpha=0.3)

# Summary table
summary_data = [
    ["Full-sample Sharpe", f"{full_sharpe:+.3f}"],
    ["Mean CV Sharpe", f"{np.mean(fold_sharpes):+.3f}"],
    ["Std CV Sharpe", f"{np.std(fold_sharpes):.3f}"],
    ["Bootstrap p-value", f"{1-p_value:.4f}"],
    ["Deflated Sharpe", f"{dsr:+.3f}"],
    ["Significant?", "Yes" if (1-p_value) < 0.05 else "No"],
]
axes[1, 1].axis("off")
table = axes[1, 1].table(cellText=summary_data, colLabels=["Metric", "Value"],
                          loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(11)
table.scale(1, 1.5)
axes[1, 1].set_title("Statistical Summary", fontsize=12, pad=20)

plt.tight_layout()
plt.show()
""")

    backtest = nbf.v4.new_code_cell("""import pandas as pd, numpy as np

# Use the momentum strategy returns as the equity curve
strategy_returns_raw = strat_ret
equity_curve = (1 + strategy_returns_raw).cumprod()

returns_s_aligned = returns_s.loc[equity_curve.index] if hasattr(returns_s, 'loc') else returns_s
benchmark_equity = (1 + returns_s_aligned).cumprod()

print(f"Backtest: {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}")
""")

    return [cv_cell, backtest]


# ═══════════════════════════════════════════════════════════════════════════
# risk_03: Time-Series Storage — Format Benchmarks (infrastructure)
# ═══════════════════════════════════════════════════════════════════════════
def _risk03_cells() -> list[nbf.NotebookNode]:
    benchmark = nbf.v4.new_code_cell("""import numpy as np, pandas as pd, time, os, tempfile, matplotlib.pyplot as plt

rng = np.random.default_rng(SEED)

# Generate large synthetic financial time series
n_rows = 500_000
n_cols = 20
dates = pd.date_range("2000-01-01", periods=n_rows, freq="s")
data_df = pd.DataFrame(
    rng.standard_normal((n_rows, n_cols)),
    index=dates,
    columns=[f"feature_{i}" for i in range(n_cols)]
)
data_df["price"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.0001, n_rows)))
data_df["volume"] = rng.exponential(1e6, n_rows).astype(int)

print(f"Dataset: {n_rows:,} rows × {data_df.shape[1]} columns")
print(f"Memory: {data_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# --- Benchmark: CSV vs Parquet vs HDF5 ---
tmpdir = tempfile.mkdtemp()
formats = {}

# CSV
path_csv = os.path.join(tmpdir, "data.csv")
t0 = time.perf_counter()
data_df.to_csv(path_csv)
write_csv = time.perf_counter() - t0
size_csv = os.path.getsize(path_csv)
t0 = time.perf_counter()
_ = pd.read_csv(path_csv, index_col=0, parse_dates=True)
read_csv = time.perf_counter() - t0
formats["CSV"] = {"write": write_csv, "read": read_csv, "size": size_csv}

# Parquet
path_pq = os.path.join(tmpdir, "data.parquet")
try:
    t0 = time.perf_counter()
    data_df.to_parquet(path_pq, compression="snappy")
    write_pq = time.perf_counter() - t0
    size_pq = os.path.getsize(path_pq)
    t0 = time.perf_counter()
    _ = pd.read_parquet(path_pq)
    read_pq = time.perf_counter() - t0
    formats["Parquet"] = {"write": write_pq, "read": read_pq, "size": size_pq}
except Exception as e:
    print(f"Parquet not available: {e}")

# HDF5
path_h5 = os.path.join(tmpdir, "data.h5")
try:
    t0 = time.perf_counter()
    data_df.to_hdf(path_h5, key="data", mode="w", complevel=4, complib="blosc")
    write_h5 = time.perf_counter() - t0
    size_h5 = os.path.getsize(path_h5)
    t0 = time.perf_counter()
    _ = pd.read_hdf(path_h5, key="data")
    read_h5 = time.perf_counter() - t0
    formats["HDF5"] = {"write": write_h5, "read": read_h5, "size": size_h5}
except Exception as e:
    print(f"HDF5 not available: {e}")

# Feather
path_ft = os.path.join(tmpdir, "data.feather")
try:
    data_reset = data_df.reset_index()
    t0 = time.perf_counter()
    data_reset.to_feather(path_ft)
    write_ft = time.perf_counter() - t0
    size_ft = os.path.getsize(path_ft)
    t0 = time.perf_counter()
    _ = pd.read_feather(path_ft)
    read_ft = time.perf_counter() - t0
    formats["Feather"] = {"write": write_ft, "read": read_ft, "size": size_ft}
except Exception as e:
    print(f"Feather not available: {e}")

# Print results
print(f"\\n{'Format':>10} {'Write (s)':>10} {'Read (s)':>10} {'Size (MB)':>10} {'Compression':>12}")
print("-" * 55)
raw_size = data_df.memory_usage(deep=True).sum()
for fmt, m in formats.items():
    ratio = m['size'] / raw_size
    print(f"{fmt:>10} {m['write']:>10.3f} {m['read']:>10.3f} {m['size']/1e6:>10.1f} {ratio:>11.1%}")

# Cleanup
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

fmt_names = list(formats.keys())
colors = ["#ef4444", "#10b981", "#3b82f6", "#f59e0b"][:len(fmt_names)]

# Write time
axes[0].bar(fmt_names, [formats[f]["write"] for f in fmt_names], color=colors, alpha=0.7, edgecolor="white")
axes[0].set_title("Write Time (seconds)")
axes[0].grid(True, alpha=0.3, axis="y")

# Read time
axes[1].bar(fmt_names, [formats[f]["read"] for f in fmt_names], color=colors, alpha=0.7, edgecolor="white")
axes[1].set_title("Read Time (seconds)")
axes[1].grid(True, alpha=0.3, axis="y")

# File size
axes[2].bar(fmt_names, [formats[f]["size"]/1e6 for f in fmt_names], color=colors, alpha=0.7, edgecolor="white")
axes[2].set_title("File Size (MB)")
axes[2].grid(True, alpha=0.3, axis="y")

plt.suptitle(f"Storage Format Comparison ({n_rows:,} rows × {data_df.shape[1]} cols)", fontsize=13)
plt.tight_layout()
plt.show()
""")

    return [benchmark, viz]


# ═══════════════════════════════════════════════════════════════════════════
# risk_04: C++ Latency Utilities — Python Data Structure Benchmarks
# ═══════════════════════════════════════════════════════════════════════════
def _risk04_cells() -> list[nbf.NotebookNode]:
    benchmark = nbf.v4.new_code_cell("""import numpy as np, time, collections, matplotlib.pyplot as plt

n_ops = 100_000

# --- Benchmark: list vs deque vs numpy array ---
results = {}

# List append/pop
t0 = time.perf_counter()
lst = []
for i in range(n_ops):
    lst.append(i * 1.0)
for _ in range(n_ops):
    lst.pop()
results["list"] = time.perf_counter() - t0

# Deque append/popleft (FIFO)
t0 = time.perf_counter()
dq = collections.deque(maxlen=n_ops)
for i in range(n_ops):
    dq.append(i * 1.0)
for _ in range(min(n_ops, len(dq))):
    dq.popleft()
results["deque"] = time.perf_counter() - t0

# Numpy array operations
t0 = time.perf_counter()
arr = np.empty(n_ops)
for i in range(n_ops):
    arr[i] = i * 1.0
_ = arr.sum()
_ = arr.mean()
_ = arr.std()
results["numpy"] = time.perf_counter() - t0

# Dict operations (hash map)
t0 = time.perf_counter()
d = {}
for i in range(n_ops):
    d[i] = i * 1.0
for i in range(n_ops):
    _ = d[i]
results["dict"] = time.perf_counter() - t0

print(f"Data Structure Benchmarks ({n_ops:,} operations):")
for name, elapsed in results.items():
    throughput = n_ops / elapsed
    print(f"  {name:>10}: {elapsed:.4f}s ({throughput:,.0f} ops/sec)")

# --- Per-operation latency measurement ---
latency_data = {}
for name, fn in [("list_append", lambda: [].append(1.0)),
                  ("deque_append", lambda: collections.deque().append(1.0)),
                  ("np_random", lambda: np.random.random()),
                  ("dict_lookup", lambda: {0: 1.0}.get(0))]:
    latencies = np.zeros(10000)
    for i in range(10000):
        t0 = time.perf_counter_ns()
        fn()
        latencies[i] = (time.perf_counter_ns() - t0) / 1000  # μs
    latency_data[name] = latencies

# Print percentiles
print(f"\\nLatency Percentiles (μs):")
print(f"  {'Operation':>15} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'p99.9':>8}")
for name, lats in latency_data.items():
    pcts = [np.percentile(lats, p) for p in [50, 90, 95, 99, 99.9]]
    print(f"  {name:>15} {pcts[0]:>8.2f} {pcts[1]:>8.2f} {pcts[2]:>8.2f} {pcts[3]:>8.2f} {pcts[4]:>8.2f}")
""")

    viz = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Throughput comparison
names = list(results.keys())
throughputs = [n_ops / results[n] for n in names]
colors = ["#f59e0b", "#3b82f6", "#10b981", "#ef4444"][:len(names)]
axes[0, 0].bar(names, throughputs, color=colors, alpha=0.7, edgecolor="white")
axes[0, 0].set_title(f"Throughput ({n_ops:,} ops)")
axes[0, 0].set_ylabel("ops/sec")
axes[0, 0].grid(True, alpha=0.3, axis="y")

# Latency CDF per operation
for (name, lats), color in zip(latency_data.items(), ["#f59e0b", "#3b82f6", "#10b981", "#ef4444"]):
    sorted_lats = np.sort(lats)
    cdf = np.arange(1, len(sorted_lats)+1) / len(sorted_lats) * 100
    axes[0, 1].plot(sorted_lats, cdf, linewidth=1.5, color=color, label=name)
axes[0, 1].set_xlabel("Latency (μs)")
axes[0, 1].set_ylabel("Percentile (%)")
axes[0, 1].set_title("Latency CDF")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, np.percentile(list(latency_data.values())[0], 99.5))

# Latency distribution for list_append
lats = latency_data["list_append"]
axes[1, 0].hist(lats[lats < np.percentile(lats, 99)], bins=80, color="#f59e0b", alpha=0.7, edgecolor="none")
axes[1, 0].axvline(np.percentile(lats, 50), color="#10b981", linestyle="--", label="p50")
axes[1, 0].axvline(np.percentile(lats, 99), color="#ef4444", linestyle="--", label="p99")
axes[1, 0].set_title("List Append Latency Distribution")
axes[1, 0].set_xlabel("Latency (μs)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Memory allocation pattern (numpy)
alloc_sizes = [100, 1000, 10000, 100000, 1000000]
alloc_times = []
for sz in alloc_sizes:
    t0 = time.perf_counter_ns()
    _ = np.empty(sz)
    alloc_times.append((time.perf_counter_ns() - t0) / 1000)
axes[1, 1].plot(alloc_sizes, alloc_times, "o-", color="#3b82f6", linewidth=2)
axes[1, 1].set_xscale("log")
axes[1, 1].set_xlabel("Array Size (elements)")
axes[1, 1].set_ylabel("Allocation Time (μs)")
axes[1, 1].set_title("NumPy Allocation vs Size")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

    return [benchmark, viz]


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

# Collect benchmark results if available
if "formats" in dir():
    results_export["metrics"]["storage_benchmarks"] = {{
        fmt: {{"write_s": m["write"], "read_s": m["read"], "size_mb": m["size"]/1e6}}
        for fmt, m in formats.items()
    }}
elif "results" in dir():
    results_export["metrics"]["throughput_benchmarks"] = results

with open("results.json", "w") as f:
    json.dump(results_export, f, indent=2, default=str)
print("Results exported to results.json")
""")


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════
def build_risk_notebook(card: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    pid = card["project_id"]
    params = {p["name"]: p["default"] for p in card.get("interactive_params", [])}
    tickers = get_ticker_for_project(pid, fallback=["SPY", "TLT", "GLD", "IWM", "QQQ"])

    head = [
        title_cell(card["title"], "Risk Engineering",
                   card.get("long_description", card.get("short_description", "")), pid),
        environment_setup_cell(requires_gpu=False),
        config_cell(params),
    ]

    if "portfolio" in pid or "optimization" in pid:
        # risk_01: MVO + RP + BL
        nb.cells = head + [
            data_acquisition_yfinance(tickers),
            nbf.v4.new_markdown_cell("## Portfolio Optimization — MVO vs Risk Parity vs Black-Litterman"),
            _risk01_optimization_cell(),
            nbf.v4.new_markdown_cell("## Portfolio Backtest"),
            _risk01_backtest_cell(),
            performance_viz_cell(),
            metrics_cell(),
            monthly_heatmap_cell(),
            export_cell(pid),
            summary_cell(card["title"]),
        ]

    elif "reproducibility" in pid:
        # risk_02: Walk-forward CV + statistical tests
        cv_cell, bt_cell = _risk02_cells()
        nb.cells = head + [
            data_acquisition_yfinance(tickers),
            nbf.v4.new_markdown_cell("## Walk-Forward Cross-Validation & Statistical Significance"),
            cv_cell,
            nbf.v4.new_markdown_cell("## Strategy Backtest"),
            bt_cell,
            performance_viz_cell(),
            metrics_cell(),
            monthly_heatmap_cell(),
            export_cell(pid),
            summary_cell(card["title"]),
        ]

    elif "timeseries" in pid or "storage" in pid:
        # risk_03: Storage format benchmarks (infrastructure)
        bench_cell, viz_cell = _risk03_cells()
        nb.cells = head + [
            nbf.v4.new_markdown_cell("## Storage Format Benchmarks — CSV vs Parquet vs HDF5 vs Feather"),
            bench_cell,
            nbf.v4.new_markdown_cell("## Benchmark Visualization"),
            viz_cell,
            _infra_export_cell(pid),
            summary_cell(card["title"]),
        ]

    elif "cpp" in pid or "latency" in pid:
        # risk_04: Data structure benchmarks (infrastructure)
        bench_cell, viz_cell = _risk04_cells()
        nb.cells = head + [
            nbf.v4.new_markdown_cell("## Data Structure & Latency Benchmarks"),
            bench_cell,
            nbf.v4.new_markdown_cell("## Benchmark Visualization"),
            viz_cell,
            _infra_export_cell(pid),
            summary_cell(card["title"]),
        ]

    else:
        # Fallback: risk_01 style
        nb.cells = head + [
            data_acquisition_yfinance(tickers),
            nbf.v4.new_markdown_cell("## Portfolio Analysis"),
            _risk01_optimization_cell(),
            _risk01_backtest_cell(),
            performance_viz_cell(),
            metrics_cell(),
            export_cell(pid),
            summary_cell(card["title"]),
        ]

    return nb
