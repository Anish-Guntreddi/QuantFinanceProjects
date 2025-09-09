# Short-Horizon Trade-Imbalance Momentum

## Overview
Predict next N ticks using recent trade signs, OFI (order-flow imbalance), and microprice. Start in Python for feature study, port to C++ for production speed.

## Core Architecture

### 1. Python Feature Research (`hft/ofi_features.py`)

```python
# hft/ofi_features.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple
import numba

class TradeImbalanceFeatures:
    """Feature engineering for short-horizon prediction"""
    
    @staticmethod
    @numba.jit(nopython=True)
    def calculate_ofi(bid_sizes: np.ndarray, ask_sizes: np.ndarray,
                     prev_bid_sizes: np.ndarray, prev_ask_sizes: np.ndarray) -> float:
        """Order Flow Imbalance calculation"""
        bid_increase = np.maximum(bid_sizes - prev_bid_sizes, 0)
        bid_decrease = np.maximum(prev_bid_sizes - bid_sizes, 0)
        ask_increase = np.maximum(ask_sizes - prev_ask_sizes, 0)
        ask_decrease = np.maximum(prev_ask_sizes - ask_sizes, 0)
        
        return bid_increase.sum() - bid_decrease.sum() - ask_increase.sum() + ask_decrease.sum()
    
    def compute_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute all predictive features"""
        features = pd.DataFrame(index=market_data.index)
        
        # Trade imbalance
        features['trade_imbalance'] = self.calculate_trade_imbalance(market_data)
        
        # OFI at multiple levels
        for level in [1, 3, 5]:
            features[f'ofi_l{level}'] = self.calculate_ofi_series(market_data, level)
        
        # Microprice and changes
        features['microprice'] = self.calculate_microprice(market_data)
        features['microprice_change'] = features['microprice'].diff()
        
        # Trade intensity
        features['buy_intensity'] = market_data['buy_volume'].rolling(10).sum()
        features['sell_intensity'] = market_data['sell_volume'].rolling(10).sum()
        features['net_intensity'] = features['buy_intensity'] - features['sell_intensity']
        
        # Volatility features
        features['realized_vol'] = market_data['midprice'].pct_change().rolling(20).std()
        
        return features
    
    def calculate_trade_imbalance(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """Recent trade sign imbalance"""
        buy_trades = data['trade_side'] == 'BUY'
        sell_trades = data['trade_side'] == 'SELL'
        
        buy_volume = data.loc[buy_trades, 'trade_volume'].rolling(window).sum()
        sell_volume = data.loc[sell_trades, 'trade_volume'].rolling(window).sum()
        
        return (buy_volume - sell_volume) / (buy_volume + sell_volume + 1)

class ShortHorizonPredictor:
    """ML model for next-tick prediction"""
    
    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=100,
            random_state=42
        )
        
    def prepare_training_data(self, features: pd.DataFrame, 
                            prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        
        # Forward returns as target
        forward_returns = prices.shift(-self.horizon) / prices - 1
        
        # Remove NaN rows
        valid_idx = ~(features.isna().any(axis=1) | forward_returns.isna())
        
        X = features[valid_idx].values
        y = forward_returns[valid_idx].values
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the prediction model"""
        self.model.fit(X, y)
        
        # Feature importance analysis
        importances = pd.DataFrame({
            'feature': range(X.shape[1]),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances
```

### 2. C++ Production Implementation (`hft/nowcast_sim.cpp`)

```cpp
// hft/nowcast_sim.cpp
#include <vector>
#include <deque>
#include <cmath>
#include <immintrin.h>  // For SIMD

namespace hft {

class NowcastEngine {
private:
    struct TickData {
        double price;
        uint64_t volume;
        int8_t side;  // 1 for buy, -1 for sell
        uint64_t timestamp_ns;
    };
    
    struct Features {
        double trade_imbalance;
        double ofi_l1;
        double ofi_l3;
        double microprice_change;
        double buy_intensity;
        double sell_intensity;
        double realized_vol;
    };
    
    std::deque<TickData> tick_buffer_;
    size_t buffer_size_ = 100;
    
    // Pre-computed model weights (from Python training)
    std::vector<double> model_weights_;
    double model_intercept_;
    
public:
    double predict_next_move(const std::vector<TickData>& recent_ticks) {
        Features features = calculate_features_fast(recent_ticks);
        return apply_model(features);
    }
    
    Features calculate_features_fast(const std::vector<TickData>& ticks) {
        Features feat;
        
        // Vectorized trade imbalance calculation
        __m256d buy_vol = _mm256_setzero_pd();
        __m256d sell_vol = _mm256_setzero_pd();
        
        for (size_t i = 0; i < ticks.size(); i += 4) {
            __m256d volumes = _mm256_loadu_pd(reinterpret_cast<const double*>(&ticks[i].volume));
            __m256i sides = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ticks[i].side));
            
            // Mask for buy/sell
            __m256d buy_mask = _mm256_cmp_pd(
                _mm256_cvtepi32_pd(_mm256_extracti128_si256(sides, 0)),
                _mm256_set1_pd(0.0), _CMP_GT_OQ
            );
            __m256d sell_mask = _mm256_cmp_pd(
                _mm256_cvtepi32_pd(_mm256_extracti128_si256(sides, 0)),
                _mm256_set1_pd(0.0), _CMP_LT_OQ
            );
            
            buy_vol = _mm256_add_pd(buy_vol, _mm256_and_pd(volumes, buy_mask));
            sell_vol = _mm256_add_pd(sell_vol, _mm256_and_pd(volumes, sell_mask));
        }
        
        // Horizontal sum
        double buy_total = horizontal_sum(buy_vol);
        double sell_total = horizontal_sum(sell_vol);
        
        feat.trade_imbalance = (buy_total - sell_total) / (buy_total + sell_total + 1.0);
        
        // Calculate other features...
        feat.ofi_l1 = calculate_ofi_fast(ticks, 1);
        feat.microprice_change = calculate_microprice_change(ticks);
        
        return feat;
    }
    
    double apply_model(const Features& features) {
        // Linear model for ultra-fast prediction
        double prediction = model_intercept_;
        prediction += model_weights_[0] * features.trade_imbalance;
        prediction += model_weights_[1] * features.ofi_l1;
        prediction += model_weights_[2] * features.ofi_l3;
        prediction += model_weights_[3] * features.microprice_change;
        prediction += model_weights_[4] * features.buy_intensity;
        prediction += model_weights_[5] * features.sell_intensity;
        prediction += model_weights_[6] * features.realized_vol;
        
        return prediction;
    }
    
private:
    double horizontal_sum(__m256d v) {
        __m128d vlow  = _mm256_castpd256_pd128(v);
        __m128d vhigh = _mm256_extractf128_pd(v, 1);
        vlow = _mm_add_pd(vlow, vhigh);
        __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
        return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
    }
};

class TradingSimulator {
private:
    NowcastEngine predictor_;
    
    struct Position {
        int64_t size;
        double entry_price;
        double pnl;
    };
    
    Position position_{0, 0.0, 0.0};
    
public:
    void on_tick(const TickData& tick) {
        // Get prediction
        double predicted_move = predictor_.predict_next_move(get_recent_ticks());
        
        // Trading logic
        const double THRESHOLD = 0.0001;  // 1 bps
        
        if (predicted_move > THRESHOLD && position_.size <= 0) {
            // Buy signal
            enter_position(1, tick.price);
        } else if (predicted_move < -THRESHOLD && position_.size >= 0) {
            // Sell signal
            enter_position(-1, tick.price);
        } else if (std::abs(predicted_move) < THRESHOLD / 2) {
            // Close position
            close_position(tick.price);
        }
    }
    
    void enter_position(int64_t size, double price) {
        if (position_.size != 0) {
            close_position(price);
        }
        position_.size = size * 100;  // 100 shares
        position_.entry_price = price;
    }
    
    void close_position(double price) {
        if (position_.size != 0) {
            double pnl = (price - position_.entry_price) * position_.size;
            position_.pnl += pnl;
            position_.size = 0;
        }
    }
};
}
```

### 3. Backtesting Framework

```python
# backtest/trade_imbalance_backtest.py
class TradeImbalanceBacktest:
    """Backtesting framework for trade imbalance strategy"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.feature_eng = TradeImbalanceFeatures()
        self.predictor = ShortHorizonPredictor()
        
    def run_backtest(self, train_size: float = 0.7) -> Dict:
        """Walk-forward backtesting"""
        
        # Split data
        split_idx = int(len(self.data) * train_size)
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        # Compute features
        train_features = self.feature_eng.compute_features(train_data)
        test_features = self.feature_eng.compute_features(test_data)
        
        # Train model
        X_train, y_train = self.predictor.prepare_training_data(
            train_features, train_data['midprice']
        )
        self.predictor.train(X_train, y_train)
        
        # Generate predictions
        X_test = test_features.dropna().values
        predictions = self.predictor.model.predict(X_test)
        
        # Simulate trading
        results = self.simulate_trading(predictions, test_data)
        
        return self.calculate_metrics(results)
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Performance metrics"""
        return {
            'total_return': results['cumulative_pnl'].iloc[-1],
            'sharpe_ratio': results['returns'].mean() / results['returns'].std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(results['cumulative_pnl']),
            'win_rate': (results['trade_pnl'] > 0).mean(),
            'avg_trade_duration': results['holding_time'].mean(),
            'prediction_accuracy': self.calculate_directional_accuracy(results)
        }
```

## Implementation Checklist

### Phase 1: Research
- [ ] Collect high-frequency tick data
- [ ] Engineer OFI and trade imbalance features
- [ ] Train prediction models
- [ ] Validate feature importance

### Phase 2: C++ Implementation
- [ ] Port feature calculations to C++
- [ ] Implement SIMD optimizations
- [ ] Build prediction engine
- [ ] Create trading simulator

### Phase 3: Integration
- [ ] Python-C++ model transfer
- [ ] Real-time feature updates
- [ ] Latency optimization
- [ ] Performance monitoring

## Expected Performance

- Prediction Latency: < 5 microseconds
- Feature Calculation: < 10 microseconds
- Directional Accuracy: > 52%
- Sharpe Ratio: > 2.5
- Average Holding Period: 5-30 seconds