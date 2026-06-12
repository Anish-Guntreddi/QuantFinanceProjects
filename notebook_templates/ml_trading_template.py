"""ML Trading notebook template — for ai_ml_trading/ (3 projects).

Each project gets a distinct model approach:
  ml_01 (regime_detection)    — Hidden Markov / GBM regime classifier
  ml_02 (lstm_transformer)    — Sequence forecasting with MLP as LSTM proxy
  ml_03 (rl_market_making)    — Tabular Q-learning market maker
"""

from __future__ import annotations
import nbformat as nbf
from .common_cells import (
    title_cell, environment_setup_cell, config_cell,
    data_acquisition_yfinance, performance_viz_cell,
    metrics_cell, sensitivity_cell, export_cell, summary_cell,
    monthly_heatmap_cell, _extract_tickers, get_ticker_for_project,
)


# ---------------------------------------------------------------------------
# Shared feature engineering cell (all 3 projects)
# ---------------------------------------------------------------------------
def _ml_feature_engineering_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Feature engineering for ML model
price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
price = price.ffill()
returns = price.pct_change()

features = pd.DataFrame(index=price.index)

# Technical features
for lag in [1, 5, 10, 21, 63]:
    features[f"ret_{lag}d"] = price.pct_change(lag)
    features[f"vol_{lag}d"] = returns.rolling(max(lag, 2)).std()  # std needs >=2 obs

# Moving average features
for window in [10, 20, 50]:
    features[f"ma_ratio_{window}"] = price / price.rolling(window).mean() - 1

# RSI-14
gain = returns.clip(lower=0).rolling(14).mean()
loss = returns.clip(upper=0).abs().rolling(14).mean()
features["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan).fillna(1e-9))

features = features.dropna()
print(f"Features shape: {features.shape}")
print(f"\\nFeature correlations with 5d forward return:")
fwd_ret = returns.shift(-5).reindex(features.index)
corrs = features.corrwith(fwd_ret).sort_values(ascending=False)
print(corrs.head(10))
""")


# ---------------------------------------------------------------------------
# ml_01: Regime Detection — GBM classifier + HMM-style regime labels
# ---------------------------------------------------------------------------
def _regime_model_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Target: 5-day forward return direction (up/down regime)
target = (returns.shift(-5) > 0).astype(int).reindex(features.index).dropna()
features_aligned = features.loc[target.index]

split_idx = int(len(features_aligned) * PARAMS.get("train_ratio", 0.7))
embargo   = PARAMS.get("embargo_days", 10)

X_train = features_aligned.iloc[:split_idx - embargo]
y_train = target.iloc[:split_idx - embargo]
X_test  = features_aligned.iloc[split_idx:]
y_test  = target.iloc[split_idx:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Embargo: {embargo} days")

model = GradientBoostingClassifier(
    n_estimators=PARAMS.get("n_estimators", 200),
    max_depth=PARAMS.get("max_depth", 4),
    learning_rate=PARAMS.get("learning_rate", 0.05),
    subsample=PARAMS.get("subsample", 0.8),
    random_state=SEED,
)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
print(f"\\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")
print(classification_report(y_test, model.predict(X_test)))

importances = (
    pd.Series(model.feature_importances_, index=features.columns)
    .sort_values(ascending=False)
)
print("Top 10 features:")
print(importances.head(10))
""")


def _regime_backtest_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

predictions = model.predict_proba(X_test)[:, 1]
signal = pd.Series(predictions, index=X_test.index)
positions = (signal - 0.5).clip(-1, 1)

test_returns = returns.reindex(X_test.index)
strategy_returns_raw = (positions.shift(1) * test_returns).dropna()

equity_curve     = (1 + strategy_returns_raw).cumprod()
benchmark_equity = (1 + test_returns.loc[equity_curve.index]).cumprod()

print(f"Backtest: {equity_curve.index[0].date()} to {equity_curve.index[-1].date()}")
print(f"Strategy final: {equity_curve.iloc[-1]:.4f}  |  Benchmark: {benchmark_equity.iloc[-1]:.4f}")
""")


def _regime_specific_metrics_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""from scipy.stats import spearmanr
import pandas as pd

aligned = pd.DataFrame({
    "prediction": signal,
    "actual": returns.shift(-5).reindex(signal.index),
}).dropna()

ic, _ = spearmanr(aligned["prediction"], aligned["actual"])
dir_acc = ((aligned["prediction"] > 0.5) == (aligned["actual"] > 0)).mean()

print("=" * 50)
print("REGIME DETECTION METRICS")
print("=" * 50)
print(f"  {'Information Coefficient':>25}: {ic:.4f}")
print(f"  {'Directional Accuracy':>25}: {dir_acc:.4f}")
print(f"  {'Test Accuracy (GBM)':>25}: {test_acc:.4f}")
""")


# ---------------------------------------------------------------------------
# ml_02: LSTM / Transformer Forecasting — sequence model via sliding window
# ---------------------------------------------------------------------------
def _lstm_feature_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Build sequence dataset: predict 5-day forward return (regression)
LOOKBACK = PARAMS.get("lookback", 20)

price = close if isinstance(close, pd.Series) else close.iloc[:, 0]
returns = price.pct_change()
feat_cols = [c for c in features.columns]

X_seq, y_seq, idx_seq = [], [], []
for i in range(LOOKBACK, len(features) - 5):
    window = features.iloc[i - LOOKBACK : i][feat_cols].values  # (T, F)
    fwd    = returns.iloc[i + 5]
    if not np.isnan(fwd):
        X_seq.append(window.flatten())  # flatten for MLP proxy
        y_seq.append(fwd)
        idx_seq.append(features.index[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

split = int(len(X_seq) * 0.7)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
idx_test = idx_seq[split:]

print(f"Sequence shape: {X_seq.shape}  (samples × lookback*features)")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
""")


def _lstm_model_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Standardize inputs (critical for MLP)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# MLP as LSTM proxy (deep, with tanh activations to mimic recurrent dynamics)
model = MLPRegressor(
    hidden_layer_sizes=tuple(PARAMS.get("hidden_layers", [256, 128, 64])),
    activation=PARAMS.get("activation", "tanh"),
    max_iter=PARAMS.get("max_iter", 300),
    learning_rate_init=PARAMS.get("learning_rate", 0.001),
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
model.fit(X_train_s, y_train)

pred_train = model.predict(X_train_s)
pred_test  = model.predict(X_test_s)

train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
test_rmse  = np.sqrt(mean_squared_error(y_test,  pred_test))
train_dir  = ((pred_train > 0) == (y_train > 0)).mean()
test_dir   = ((pred_test  > 0) == (y_test  > 0)).mean()

print(f"Train RMSE: {train_rmse:.6f}  |  Dir accuracy: {train_dir:.4f}")
print(f"Test  RMSE: {test_rmse:.6f}  |  Dir accuracy: {test_dir:.4f}")
print(f"\\nModel converged after {model.n_iter_} iterations")
""")


def _lstm_backtest_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

signal = pd.Series(pred_test, index=idx_test)
positions = signal.clip(-0.02, 0.02) / 0.02  # scale to [-1, 1]

test_returns_s = returns.reindex(idx_test)
strategy_returns_raw = (positions.shift(1) * test_returns_s).dropna()

equity_curve     = (1 + strategy_returns_raw).cumprod()
benchmark_equity = (1 + test_returns_s.loc[equity_curve.index]).cumprod()

print(f"Backtest: {equity_curve.index[0].date()} to {equity_curve.index[-1].date()}")
print(f"Strategy final: {equity_curve.iloc[-1]:.4f}  |  Benchmark: {benchmark_equity.iloc[-1]:.4f}")
""")


def _lstm_specific_metrics_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""from scipy.stats import spearmanr
import numpy as np

ic, _ = spearmanr(pred_test, y_test)
hit_rate = ((pred_test > 0) == (y_test > 0)).mean()

print("=" * 50)
print("SEQUENCE MODEL METRICS")
print("=" * 50)
print(f"  {'Information Coefficient':>28}: {ic:.4f}")
print(f"  {'Directional Hit Rate':>28}: {hit_rate:.4f}")
print(f"  {'Test RMSE':>28}: {test_rmse:.6f}")
print(f"  {'Test Directional Accuracy':>28}: {test_dir:.4f}")
print(f"  {'MLP Layers':>28}: {model.hidden_layer_sizes}")
""")


# ---------------------------------------------------------------------------
# ml_03: RL Market Making — tabular Q-learning
# ---------------------------------------------------------------------------
def _rl_env_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import numpy as np
import pandas as pd

# Simplified RL market-making environment
# State: (inventory_bucket, spread_bucket, imbalance_bucket)
# Action: 0=tighten spread, 1=hold, 2=widen spread

price = (close if isinstance(close, pd.Series) else close.iloc[:, 0]).ffill()
returns = price.pct_change().dropna()
price = price.loc[returns.index]

MAX_INV  = PARAMS.get("max_inventory", 5)
N_SPREAD = 3   # spread actions: tighten / hold / widen
BASE_SPREAD_BPS = PARAMS.get("spread_bps", 10) / 10000

def discretize_state(inv, vol_z):
    inv_b = int(np.clip(inv + MAX_INV, 0, 2 * MAX_INV))
    vol_b = int(np.clip((vol_z + 2) / 4 * 4, 0, 3))
    return (inv_b, vol_b)

n_inv_states = 2 * MAX_INV + 1
n_vol_states = 4
Q = np.zeros((n_inv_states, n_vol_states, N_SPREAD))

print(f"State space: inv({n_inv_states}) × vol({n_vol_states}) = {n_inv_states * n_vol_states} states")
print(f"Action space: {N_SPREAD} (tighten/hold/widen spread)")
print(f"Q-table shape: {Q.shape}")
""")


def _rl_training_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import numpy as np

# Q-learning hyperparameters
alpha     = PARAMS.get("learning_rate", 0.05)
gamma     = PARAMS.get("gamma", 0.95)
epsilon   = 1.0
eps_min   = PARAMS.get("eps_min", 0.05)
eps_decay = PARAMS.get("eps_decay", 0.995)

vol_20 = returns.rolling(20).std()
vol_z  = ((vol_20 - vol_20.mean()) / vol_20.std()).fillna(0)

spread_mult = [0.5, 1.0, 2.0]  # multiplier on BASE_SPREAD_BPS
inventory   = 0
pnl_history = []
rng         = np.random.default_rng(SEED)

n_episodes  = PARAMS.get("n_episodes", 5)
episode_len = min(PARAMS.get("episode_length", 500), len(returns))

for ep in range(n_episodes):
    inventory = 0
    cash      = 0.0
    ep_pnl    = []

    for t in range(1, episode_len):
        ret  = returns.iloc[t]
        mid  = price.iloc[t]
        vz   = float(vol_z.iloc[t])
        state = discretize_state(inventory, vz)

        # Epsilon-greedy action
        if rng.random() < epsilon:
            action = rng.integers(N_SPREAD)
        else:
            action = int(np.argmax(Q[state]))

        spread = BASE_SPREAD_BPS * spread_mult[action]

        # Simulate fills: wider spread = lower fill probability
        fill_prob = max(0.05, 0.4 - spread * 50)
        if rng.random() < fill_prob and inventory < MAX_INV:
            inventory += 1
            cash -= mid * (1 - spread / 2)
        if rng.random() < fill_prob and inventory > -MAX_INV:
            inventory -= 1
            cash += mid * (1 + spread / 2)

        mark_to_mkt = cash + inventory * mid
        reward = mark_to_mkt - (0 if t == 1 else ep_pnl[-1])
        # Inventory penalty
        reward -= abs(inventory) * 0.001

        ep_pnl.append(mark_to_mkt)

        # Q-update
        next_state = discretize_state(inventory, float(vol_z.iloc[min(t + 1, episode_len - 1)]))
        best_next  = np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

    epsilon = max(eps_min, epsilon * eps_decay)
    pnl_history.append(ep_pnl[-1])
    print(f"Episode {ep + 1}/{n_episodes} | Final PnL: {ep_pnl[-1]:+,.2f} | ε: {epsilon:.3f}")
""")


def _rl_backtest_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Greedy rollout (epsilon=0)
inventory = 0
cash      = 0.0
pnl_ts    = [0.0]
vol_z_arr = vol_z.values

for t in range(1, len(returns)):
    ret  = returns.iloc[t]
    mid  = price.iloc[t]
    vz   = float(vol_z_arr[t])
    state  = discretize_state(inventory, vz)
    action = int(np.argmax(Q[state]))
    spread = BASE_SPREAD_BPS * spread_mult[action]

    fill_prob = max(0.05, 0.4 - spread * 50)
    rng2 = np.random.default_rng(SEED + t)
    if rng2.random() < fill_prob and inventory < MAX_INV:
        inventory += 1; cash -= mid * (1 - spread / 2)
    if rng2.random() < fill_prob and inventory > -MAX_INV:
        inventory -= 1; cash += mid * (1 + spread / 2)

    pnl_ts.append(cash + inventory * mid)

pnl_series = pd.Series(pnl_ts, index=price.index)
pnl_series = pnl_series - pnl_series.min() + 1  # normalize to start at 1

strategy_returns_raw = pnl_series.pct_change().dropna()
equity_curve     = (1 + strategy_returns_raw).cumprod()
benchmark_equity = (1 + returns.reindex(equity_curve.index)).cumprod()

print(f"Greedy rollout: {equity_curve.index[0].date()} to {equity_curve.index[-1].date()}")
print(f"Final PnL (normalized): {equity_curve.iloc[-1]:.4f}  |  Benchmark: {benchmark_equity.iloc[-1]:.4f}")
""")


def _rl_specific_metrics_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("""import numpy as np

# Q-table analysis
best_actions = np.argmax(Q, axis=2)
action_names = ["Tighten", "Hold", "Widen"]

print("=" * 50)
print("RL MARKET MAKING METRICS")
print("=" * 50)
print(f"  {'Final epsilon':>25}: {epsilon:.4f}")
print(f"  {'Episodes trained':>25}: {n_episodes}")
print(f"  {'Final episode PnL':>25}: {pnl_history[-1]:+,.2f}")
print(f"  {'Avg episode PnL':>25}: {sum(pnl_history)/len(pnl_history):+,.2f}")
print(f"\\nLearned policy (most common action by inventory):")
for inv in range(-MAX_INV, MAX_INV + 1):
    inv_b = int(np.clip(inv + MAX_INV, 0, 2 * MAX_INV))
    actions_for_inv = best_actions[inv_b, :]
    modal_action = int(np.bincount(actions_for_inv).argmax())
    print(f"  inv={inv:+d} -> {action_names[modal_action]}")
""")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_ml_notebook(card: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    pid = card["project_id"]
    params = {p["name"]: p["default"] for p in card.get("interactive_params", [])}
    tickers = get_ticker_for_project(pid, fallback=_extract_tickers(card.get("data_source", ""), default="SPY"))

    common_head = [
        title_cell(card["title"], "AI/ML Trading",
                   card.get("long_description", card.get("short_description", "")), pid),
        environment_setup_cell(requires_gpu=True),
        config_cell(params),
        data_acquisition_yfinance(tickers),
        nbf.v4.new_markdown_cell("## Feature Engineering"),
        _ml_feature_engineering_cell(),
    ]

    if "lstm" in pid or "transformer" in pid:
        # ml_02: sequence forecasting
        domain_cells = [
            nbf.v4.new_markdown_cell("## Sequence Dataset Construction"),
            _lstm_feature_cell(),
            nbf.v4.new_markdown_cell("## Sequence Model Training (MLP proxy for LSTM)"),
            _lstm_model_cell(),
            nbf.v4.new_markdown_cell("## Backtest"),
            _lstm_backtest_cell(),
            performance_viz_cell(),
            metrics_cell(),
            _lstm_specific_metrics_cell(),
            monthly_heatmap_cell(),
        ]
    elif "rl" in pid:
        # ml_03: reinforcement learning
        domain_cells = [
            nbf.v4.new_markdown_cell("## RL Environment Setup"),
            _rl_env_cell(),
            nbf.v4.new_markdown_cell("## Q-Learning Training"),
            _rl_training_cell(),
            nbf.v4.new_markdown_cell("## Greedy Backtest"),
            _rl_backtest_cell(),
            performance_viz_cell(),
            metrics_cell(),
            _rl_specific_metrics_cell(),
            monthly_heatmap_cell(),
        ]
    else:
        # ml_01: regime detection (GBM classifier)
        domain_cells = [
            nbf.v4.new_markdown_cell("## Regime Classifier Training"),
            _regime_model_cell(),
            nbf.v4.new_markdown_cell("## Backtest"),
            _regime_backtest_cell(),
            performance_viz_cell(),
            metrics_cell(),
            _regime_specific_metrics_cell(),
            monthly_heatmap_cell(),
        ]

    nb.cells = common_head + domain_cells + [
        export_cell(pid),
        summary_cell(card["title"]),
    ]
    return nb
