# Regime Detection & Allocation - Quick Start Guide

## 📦 Installation

```bash
# Navigate to project directory
cd ai_ml_trading/01_regime_detection_allocation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Demo

### 1. Test Implementation
```bash
# Verify all components work
python test_implementation.py
```

### 2. Run Complete Pipeline
```bash
# Full demonstration with visualization
python example_complete_pipeline.py
```

## 💻 Basic Usage

### Simple Regime Detection
```python
from ml.models.hmm_regime import HMMRegimeDetector
from features.technical_features import TechnicalFeatureExtractor
import yfinance as yf
import pandas as pd

# Load market data
spy = yf.download('SPY', start='2020-01-01', end='2023-12-31')

# Extract features
extractor = TechnicalFeatureExtractor()
features = extractor.extract_features(spy)

# Detect regimes
detector = HMMRegimeDetector(n_regimes=4)
detector.fit(features.iloc[:, :10])  # Use first 10 features
regimes = detector.predict(features.iloc[:, :10])
```

### Using Ensemble Methods
```python
from ml.models.ensemble_regime import create_default_ensemble

# Create ensemble with multiple models
ensemble = create_default_ensemble(n_regimes=4)
ensemble.fit(features)

# Get regime predictions with confidence
predictions = ensemble.predict(features)
probabilities = ensemble.predict_proba(features)
confidence = ensemble.get_prediction_confidence(features)
```

### Feature Extraction
```python
from features.macro_features import MacroFeatureExtractor
from features.technical_features import TechnicalFeatureExtractor

# Technical features
tech_extractor = TechnicalFeatureExtractor()
tech_features = tech_extractor.extract_features(spy)

# Macro features (requires FRED API key)
macro_extractor = MacroFeatureExtractor(fred_api_key='YOUR_KEY')
macro_features = macro_extractor.extract_features('2020-01-01', '2023-12-31')
```

## 📊 Key Components

### Regime Detection Models
- **HMM**: Hidden Markov Models for probabilistic regime detection
- **Markov-Switching**: Regime-switching AR/VAR/ARCH models
- **Clustering**: HDBSCAN/KMeans for unsupervised regime identification
- **Ensemble**: Advanced combination of multiple models

### Feature Types
- **Technical**: 80+ indicators including trend, momentum, volatility
- **Macro**: 30+ economic indicators from FRED and market data
- **Microstructure**: Spread proxies, efficiency ratios, pattern features

### Data Utilities
- **DataLoader**: Multi-source data loading with caching
- **DataPreprocessor**: Cleaning, normalization, outlier handling
- **DataValidator**: Comprehensive data quality checks

## 📈 Outputs

The system provides:
1. **Regime Classifications**: Bull/Bear × Quiet/Volatile
2. **Transition Probabilities**: Likelihood of regime changes
3. **Confidence Scores**: Certainty of predictions
4. **Performance Metrics**: Model accuracy and stability
5. **Visualizations**: Regime evolution and characteristics

## 🔧 Configuration

### Custom Regime Types
```python
from ml.regimes import RegimeType

# Define custom regimes
class CustomRegimeType(RegimeType):
    TRENDING = "Trending"
    RANGEBOUND = "Rangebound"
    CRISIS = "Crisis"
```

### Model Parameters
```python
# HMM with custom settings
detector = HMMRegimeDetector(
    n_regimes=5,
    covariance_type='full',
    n_iter=200,
    tol=1e-4
)

# Clustering with specific method
cluster_detector = ClusteringRegimeDetector(
    method='hdbscan',
    min_cluster_size=50,
    use_pca=True,
    n_components=10
)
```

## 📊 Visualization

```python
# Built-in plotting
detector.plot_regimes(market_data, save_path='regimes.png')
ensemble.plot_transition_matrix(save_path='transitions.png')
```

## 🎯 Common Use Cases

### 1. Market Analysis
```python
# Identify current market regime
current_regime = detector.get_current_regime(features)
print(f"Market is in {current_regime.name} regime")
```

### 2. Risk Management
```python
# Adjust position sizing based on regime
if current_regime == RegimeType.BEAR_VOLATILE:
    position_size *= 0.5  # Reduce risk in volatile bear markets
```

### 3. Strategy Selection
```python
# Choose strategy based on regime
regime_strategies = {
    RegimeType.BULL_QUIET: 'trend_following',
    RegimeType.BEAR_VOLATILE: 'volatility_arbitrage',
    RegimeType.RANGEBOUND: 'mean_reversion'
}
active_strategy = regime_strategies.get(current_regime)
```

## 🔍 Troubleshooting

### Common Issues

1. **ImportError**: Install missing dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **TA-Lib Installation**: If TA-Lib fails, system uses fallback implementations

3. **FRED API**: Get free API key at https://fred.stlouisfed.org/docs/api/api_key.html

4. **Memory Issues**: Reduce data size or use sampling
   ```python
   features = features.sample(frac=0.5)  # Use 50% of data
   ```

## 📚 Next Steps

1. Review `IMPLEMENTATION_SUMMARY.md` for detailed documentation
2. Explore `example_complete_pipeline.py` for advanced usage
3. Check individual model files for specific parameters
4. Run tests to verify your setup

## 🤝 Support

For issues or questions:
- Check the README.md for detailed specifications
- Review test files for usage examples
- Examine docstrings in source code