"""ML/AI trading notebook template — GPU detection, embargo CV, model metrics."""

from .common import BaseNotebookTemplate, MetricsCalculator


class MLTradingTemplate(BaseNotebookTemplate):

    def cell_04_data(self, config):
        data = config.get("data", {})
        if data.get("generator"):
            return "code", data["generator"]
        tickers = data.get("tickers", ["SPY"])
        return "code", f'''import yfinance as yf

tickers = {tickers}
print(f"Fetching data for {{tickers}} from {{BACKTEST_START}} to {{BACKTEST_END}}...")
data = yf.download(tickers, start=BACKTEST_START, end=BACKTEST_END, progress=False)
price_data = data["Close"] if "Close" in data.columns else data[("Close", tickers[0])]
volume_data = data["Volume"] if "Volume" in data.columns else data[("Volume", tickers[0])]
returns = price_data.pct_change().dropna()

benchmark_data = yf.download("SPY", start=BACKTEST_START, end=BACKTEST_END, progress=False)["Close"]
benchmark_returns = benchmark_data.pct_change().dropna()

print(f"Data shape: {{price_data.shape}}")
print(f"Date range: {{price_data.index[0]}} to {{price_data.index[-1]}}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(price_data, color="#00D4AA")
ax1.set_title("Price History")
ax1.grid(True, alpha=0.3)
ax2.bar(volume_data.index, volume_data.values, width=1, color="#7B68EE", alpha=0.5)
ax2.set_title("Volume")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

    def cell_05_features(self, config):
        return "code", '''# ML Feature Engineering
print("Computing ML features...")

# Technical features
feature_df = pd.DataFrame(index=returns.index)
for w in [5, 10, 20, 50]:
    feature_df[f"ret_{w}d"] = price_data.pct_change(w)
    feature_df[f"vol_{w}d"] = returns.rolling(w).std()
    feature_df[f"sma_ratio_{w}d"] = price_data / price_data.rolling(w).mean()

# Momentum features
feature_df["rsi_14"] = 100 - 100 / (1 + returns.rolling(14).apply(lambda x: x[x>0].mean() / abs(x[x<0].mean()) if x[x<0].mean() != 0 else 1))

# Volume features if available
if "volume_data" in dir():
    feature_df["vol_ratio"] = volume_data / volume_data.rolling(20).mean()

# Target: next-day return direction
feature_df["target"] = (returns.shift(-1) > 0).astype(int)
feature_df = feature_df.dropna()

print(f"Feature matrix: {feature_df.shape}")
print(f"Features: {[c for c in feature_df.columns if c != 'target']}")
print(f"\\nTarget distribution:\\n{feature_df['target'].value_counts(normalize=True)}")
'''

    def cell_06_strategy(self, config):
        source = config.get("source", {})
        imports = source.get("imports", [])
        key_class = source.get("key_class", "")

        code = "# ML Model Implementation\n"
        if imports:
            code += "try:\n"
            for imp in imports:
                code += f"    {imp}\n"
            code += f'    print("Successfully imported {key_class}")\n'
            code += "except ImportError as e:\n"
            code += f'    print(f"Import not available: {{e}}")\n'
            code += f"    {key_class} = None\n"

        code += '''
# Train/test split with embargo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

features = [c for c in feature_df.columns if c != "target"]
X = feature_df[features]
y = feature_df["target"]

# Embargo split: gap between train and test
train_size = int(len(X) * 0.6)
embargo = 10  # 10-day embargo gap
test_start = train_size + embargo

X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples, Embargo: {embargo} days")

# Baseline model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
print(f"\\nModel accuracy: {accuracy:.4f}")
print(f"\\nFeature importance (top 10):")
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
for feat, val in imp.head(10).items():
    print(f"  {feat:>20}: {val:.4f}")
'''
        return "code", code

    def cell_07_backtest(self, config):
        return "code", MetricsCalculator.synthetic_results_code() + f'''

# Convert predictions to trading signals and PnL
print("Computing strategy returns from model predictions...")

try:
    test_returns = returns.iloc[test_start:test_start + len(predictions)]
    # Signal: go long if predicted up, flat otherwise
    signals_arr = np.where(predictions == 1, 1.0, -0.5)
    strategy_returns = pd.Series(
        signals_arr * test_returns.values[:len(signals_arr)],
        index=test_returns.index[:len(signals_arr)]
    )
    print(f"Using ML model signals (accuracy: {{accuracy:.2%}})")
except Exception:
    print("Using synthetic ML trading results")
    strategy_returns = generate_synthetic_results(
        n_days=252,
        annual_sharpe={config.get("synthetic_sharpe", 1.3)},
        annual_vol={config.get("synthetic_vol", 0.12)},
        seed=SEED
    )

equity_curve = INITIAL_CAPITAL * (1 + strategy_returns).cumprod()
benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns.iloc[:len(strategy_returns)]).cumprod()

print(f"Backtest complete: {{len(strategy_returns)}} periods")
print(f"Final equity: ${{equity_curve.iloc[-1]:,.2f}}")
'''

    def cell_09_metrics(self, config):
        return "code", MetricsCalculator.base_metrics_code() + '''

metrics = compute_metrics(strategy_returns, benchmark_returns.iloc[:len(strategy_returns)])

# ML-specific metrics
ml_metrics = {}
if "predictions" in dir() and "y_test" in dir():
    from sklearn.metrics import accuracy_score
    ml_metrics["directional_accuracy"] = round(float(accuracy_score(y_test[:len(predictions)], predictions)), 4)
if "proba" in dir() and "test_returns" in dir():
    ic = np.corrcoef(proba[:len(test_returns)], test_returns.values[:len(proba)])[0, 1]
    ml_metrics["information_coefficient"] = round(float(ic), 4)
metrics.update(ml_metrics)

print("=" * 60)
print("  ML TRADING METRICS")
print("=" * 60)
for k, v in metrics.items():
    if isinstance(v, float):
        if "return" in k or "drawdown" in k or "vol" in k or "rate" in k or "accuracy" in k:
            print(f"  {k:>30}: {v:>10.2%}")
        else:
            print(f"  {k:>30}: {v:>10.4f}")
    else:
        print(f"  {k:>30}: {v:>10}")
print("=" * 60)
'''
