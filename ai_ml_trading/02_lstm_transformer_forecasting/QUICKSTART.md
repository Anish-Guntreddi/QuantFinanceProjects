# LSTM/Transformer Forecasting - Quick Start Guide

## 📦 Installation

```bash
# Navigate to project directory
cd ai_ml_trading/02_lstm_transformer_forecasting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Demo

### 1. Run Complete Pipeline
```bash
# Full demonstration with all features
python example_pipeline.py
```

This will:
- Generate synthetic financial data
- Train LSTM and Transformer models
- Perform cross-validation with embargo
- Calibrate predictions
- Calculate optimal position sizes
- Run backtesting with comprehensive metrics

## 💻 Basic Usage

### Simple LSTM Forecasting
```python
import torch
from models.lstm_model import AdvancedLSTM, LSTMConfig
from data.data_loader import TimeSeriesDataLoader
import numpy as np

# Generate sample data
data = np.random.randn(1000, 10)  # 1000 timesteps, 10 features
loader = TimeSeriesDataLoader(data, sequence_length=60, prediction_horizon=5)

# Create model
config = LSTMConfig(
    input_dim=10,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3,
    attention=True
)
model = AdvancedLSTM(config)

# Train (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for batch in loader:
        predictions = model(batch['features'])
        loss = torch.nn.MSELoss()(predictions['output'], batch['targets'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Transformer Forecasting
```python
from models.transformer_model import TimeSeriesTransformer, TransformerConfig

# Create transformer model
config = TransformerConfig(
    input_dim=10,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    sequence_length=60,
    prediction_horizon=5
)
transformer = TimeSeriesTransformer(config)

# Get predictions with attention weights
output = transformer(batch_data)
predictions = output['predictions']
attention_weights = output['attention_weights']
```

### Cross-Validation with Embargo
```python
from cv.time_series_cv import TimeSeriesCV, CVConfig
from cv.embargo import EmbargoCV

# Configure CV
cv_config = CVConfig(
    n_splits=5,
    test_size=252,  # 1 trading year
    gap=10,  # 10-day embargo
    expanding=False  # Use sliding window
)

# Create CV splitter
cv = TimeSeriesCV(cv_config)

# Generate splits
for train_idx, test_idx in cv.split(data):
    train_data = data[train_idx]
    test_data = data[test_idx]
    # Train and evaluate model
```

### Probability Calibration
```python
from calibration.isotonic import IsotonicCalibrator
from calibration.temperature_scaling import TemperatureScaling

# Calibrate predictions
calibrator = IsotonicCalibrator()
calibrator.fit(predicted_probs, true_labels)
calibrated_probs = calibrator.transform(new_predictions)

# Plot calibration curve
fig = calibrator.plot_calibration_curve()
```

### Position Sizing with Kelly Criterion
```python
from sizing.kelly_sizing import KellySizing

# Calculate optimal position size
kelly = KellySizing(max_leverage=1.0, kelly_fraction=0.25)
position_size = kelly.calculate_kelly_fraction(
    win_prob=0.65,
    win_return=0.02,
    loss_return=-0.01
)

# Multi-asset Kelly
weights = kelly.multi_asset_kelly(expected_returns, covariance_matrix)
```

## 📊 Key Components

### Models
- **AdvancedLSTM**: LSTM with attention, residual connections, layer normalization
- **TimeSeriesTransformer**: Transformer with temporal encoding and multi-head attention
- **HybridLSTMTransformer**: Combines strengths of both architectures
- **ConvLSTM**: Convolutional LSTM for spatiotemporal data

### Cross-Validation
- **TimeSeriesCV**: Sliding/expanding window validation
- **PurgedKFold**: Purged K-fold with embargo (López de Prado)
- **CombinatorialPurgedCV**: Combinatorial purged cross-validation
- **EmbargoCV**: Various embargo strategies

### Calibration
- **IsotonicCalibrator**: Non-parametric calibration
- **PlattScaling**: Sigmoid calibration
- **TemperatureScaling**: Neural network calibration
- **EnsembleCalibration**: Combine multiple calibrators

### Feature Engineering
- **TechnicalFeatures**: RSI, MACD, Bollinger Bands, etc.
- **StatisticalFeatures**: Moments, autocorrelation, entropy
- **TimeFeatures**: Seasonality, cyclical encoding
- **Feature importance**: SHAP, permutation, integrated gradients

### Evaluation
- **ForecastMetrics**: MAE, RMSE, MAPE, directional accuracy
- **BacktestMetrics**: Sharpe, Sortino, Calmar, max drawdown
- **DeflatedSharpe**: Corrected for multiple testing
- **CalibrationMetrics**: ECE, MCE, Brier score

## 🎯 Common Use Cases

### 1. Multi-Step Forecasting
```python
from models.model_factory import ModelFactory

# Create model for multi-step prediction
model = ModelFactory.create_model(
    model_type='transformer',
    input_dim=features.shape[1],
    prediction_horizon=10  # Predict 10 steps ahead
)
```

### 2. Model Ensemble
```python
from models.model_factory import ModelFactory

# Create ensemble
ensemble = ModelFactory.create_ensemble(
    model_configs=[
        {'type': 'lstm', 'hidden_dim': 256},
        {'type': 'transformer', 'd_model': 512},
        {'type': 'hybrid', 'lstm_dim': 128}
    ],
    aggregation='weighted_average'
)
```

### 3. Feature Importance Analysis
```python
from features.importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model)
importance_scores = analyzer.calculate_importance(
    data,
    method='integrated_gradients'
)
analyzer.plot_importance(importance_scores)
```

### 4. Backtesting with Position Sizing
```python
from evaluation.metrics import BacktestMetrics
from sizing.probability_sizing import ProbabilityAwareSizing

# Generate signals
predictions = model.predict(test_data)
probabilities = torch.sigmoid(predictions)

# Size positions
sizer = ProbabilityAwareSizing(min_probability=0.6)
positions = sizer.calculate_position_sizes(probabilities)

# Evaluate performance
metrics = BacktestMetrics.calculate_all_metrics(returns, positions)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## 📈 Outputs

The system provides:
1. **Predictions**: Point forecasts and prediction intervals
2. **Probabilities**: Calibrated probability estimates
3. **Attention Maps**: Temporal importance visualization
4. **Position Sizes**: Optimal allocation based on Kelly criterion
5. **Performance Metrics**: Comprehensive backtest statistics
6. **Feature Importance**: Which features drive predictions

## 🔧 Configuration

### Model Hyperparameters
```python
config = LSTMConfig(
    input_dim=20,
    hidden_dim=512,
    num_layers=4,
    dropout=0.4,
    attention=True,
    bidirectional=True,
    layer_norm=True
)
```

### Training Configuration
```python
train_config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping': True,
    'patience': 10,
    'scheduler': 'cosine'
}
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or model size
   config.batch_size = 16
   config.hidden_dim = 128
   ```

2. **Overfitting**
   ```python
   # Increase dropout and regularization
   config.dropout = 0.5
   config.weight_decay = 0.001
   ```

3. **Poor Calibration**
   ```python
   # Try different calibration methods
   calibrator = PlattScaling()  # Instead of isotonic
   ```

4. **Unstable Training**
   ```python
   # Use gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

## 📚 Advanced Features

### Custom Attention Mechanisms
```python
from models.attention_mechanisms import CustomAttention

attention = CustomAttention(
    embed_dim=256,
    num_heads=8,
    attention_type='multiplicative'
)
```

### Walk-Forward Optimization
```python
from cv.time_series_cv import WalkForwardCV

wf_cv = WalkForwardCV(
    train_period=252,
    test_period=63,
    retrain_frequency=21
)
```

### Risk-Adjusted Sizing
```python
from sizing.kelly_sizing import RiskAdjustedKelly

kelly = RiskAdjustedKelly(
    var_threshold=0.02,
    cvar_threshold=0.03,
    max_drawdown_limit=0.10
)
```

## 🎓 References

- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Vaswani et al. (2017). "Attention Is All You Need"
- Kelly, J. L. (1956). "A New Interpretation of Information Rate"

## 🤝 Next Steps

1. Review `example_pipeline.py` for complete workflow
2. Explore individual model files for customization
3. Check evaluation metrics for performance analysis
4. Experiment with different architectures and hyperparameters