# Order Book Imbalance Scalper (Sub-second)

## Overview
Hybrid system with Python research notebooks for signal discovery and C++ implementation for ultra-low latency execution. Trades micro-alpha signals from order book imbalance and queue depth analysis when (bid_vol - ask_vol)/(bid_vol + ask_vol) crosses calibrated thresholds.

## Core Architecture

### 1. C++ High-Frequency Signal Engine (`hft/`)

```cpp
// hft/imbalance_signal.hpp
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>

namespace hft {

template<size_t LEVELS = 10>
class ImbalanceSignal {
private:
    struct BookLevel {
        double price;
        uint64_t volume;
        uint32_t order_count;
    };
    
    struct ImbalanceMetrics {
        double volume_imbalance;
        double weighted_imbalance;
        double order_count_imbalance;
        double queue_position_score;
        double micro_price;
        std::chrono::nanoseconds timestamp;
    };
    
    // Lock-free circular buffer for metrics history
    alignas(64) std::array<ImbalanceMetrics, 1024> metrics_history_;
    alignas(64) std::atomic<size_t> write_idx_{0};
    alignas(64) std::atomic<size_t> read_idx_{0};
    
public:
    struct Signal {
        enum Direction { NONE, BUY, SELL, STRONG_BUY, STRONG_SELL };
        Direction direction;
        double strength;
        double expected_edge;
        uint32_t confidence;
        std::chrono::nanoseconds latency;
    };
    
    Signal calculate_signal(const BookLevel* bids, 
                           const BookLevel* asks,
                           size_t depth);
    
    double calculate_volume_imbalance(const BookLevel* bids,
                                     const BookLevel* asks,
                                     size_t depth);
    
    double calculate_weighted_imbalance(const BookLevel* bids,
                                       const BookLevel* asks,
                                       size_t depth);
    
    double estimate_queue_position(uint64_t ahead_volume,
                                  uint64_t level_volume,
                                  uint32_t order_count);
};

// hft/imbalance_signal.cpp
template<size_t LEVELS>
double ImbalanceSignal<LEVELS>::calculate_volume_imbalance(
    const BookLevel* bids,
    const BookLevel* asks,
    size_t depth) {
    
    uint64_t bid_vol = 0, ask_vol = 0;
    
    // Vectorized summation for first N levels
    for (size_t i = 0; i < depth; ++i) {
        bid_vol += bids[i].volume;
        ask_vol += asks[i].volume;
    }
    
    // Avoid division by zero
    uint64_t total_vol = bid_vol + ask_vol;
    if (total_vol == 0) return 0.0;
    
    // Return normalized imbalance [-1, 1]
    return static_cast<double>(bid_vol - ask_vol) / total_vol;
}

template<size_t LEVELS>
double ImbalanceSignal<LEVELS>::calculate_weighted_imbalance(
    const BookLevel* bids,
    const BookLevel* asks,
    size_t depth) {
    
    double mid_price = (bids[0].price + asks[0].price) / 2.0;
    double weighted_bid = 0.0, weighted_ask = 0.0;
    
    // Weight by inverse distance from mid
    for (size_t i = 0; i < depth; ++i) {
        double bid_weight = 1.0 / (1.0 + std::abs(bids[i].price - mid_price));
        double ask_weight = 1.0 / (1.0 + std::abs(asks[i].price - mid_price));
        
        weighted_bid += bids[i].volume * bid_weight;
        weighted_ask += asks[i].volume * ask_weight;
    }
    
    double total_weighted = weighted_bid + weighted_ask;
    if (total_weighted < 1e-9) return 0.0;
    
    return (weighted_bid - weighted_ask) / total_weighted;
}

template<size_t LEVELS>
typename ImbalanceSignal<LEVELS>::Signal 
ImbalanceSignal<LEVELS>::calculate_signal(
    const BookLevel* bids,
    const BookLevel* asks,
    size_t depth) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Calculate various imbalance metrics
    double vol_imb = calculate_volume_imbalance(bids, asks, depth);
    double weighted_imb = calculate_weighted_imbalance(bids, asks, depth);
    
    // Order count imbalance (indicative of order flow)
    uint32_t bid_orders = 0, ask_orders = 0;
    for (size_t i = 0; i < depth; ++i) {
        bid_orders += bids[i].order_count;
        ask_orders += asks[i].order_count;
    }
    double order_imb = static_cast<double>(bid_orders - ask_orders) / 
                      (bid_orders + ask_orders + 1);
    
    // Micro-price calculation
    double bid_size = bids[0].volume;
    double ask_size = asks[0].volume;
    double micro_price = (asks[0].price * bid_size + bids[0].price * ask_size) / 
                        (bid_size + ask_size);
    
    // Composite signal
    double signal_strength = 0.7 * vol_imb + 0.2 * weighted_imb + 0.1 * order_imb;
    
    Signal signal;
    signal.latency = std::chrono::high_resolution_clock::now() - start_time;
    
    // Signal thresholds (calibrated from research)
    const double WEAK_THRESHOLD = 0.15;
    const double STRONG_THRESHOLD = 0.35;
    
    if (signal_strength > STRONG_THRESHOLD) {
        signal.direction = Signal::STRONG_BUY;
        signal.strength = signal_strength;
        signal.confidence = 90;
    } else if (signal_strength > WEAK_THRESHOLD) {
        signal.direction = Signal::BUY;
        signal.strength = signal_strength;
        signal.confidence = 70;
    } else if (signal_strength < -STRONG_THRESHOLD) {
        signal.direction = Signal::STRONG_SELL;
        signal.strength = -signal_strength;
        signal.confidence = 90;
    } else if (signal_strength < -WEAK_THRESHOLD) {
        signal.direction = Signal::SELL;
        signal.strength = -signal_strength;
        signal.confidence = 70;
    } else {
        signal.direction = Signal::NONE;
        signal.strength = 0.0;
        signal.confidence = 0;
    }
    
    // Expected edge calculation (bps)
    signal.expected_edge = std::abs(signal_strength) * 2.5; // Calibrated multiplier
    
    // Store metrics for analysis
    size_t idx = write_idx_.fetch_add(1) % metrics_history_.size();
    metrics_history_[idx] = {
        vol_imb, weighted_imb, order_imb, 0.0, micro_price, signal.latency
    };
    
    return signal;
}
}
```

### 2. Python Research & Calibration (`hft/py_research.ipynb`)

```python
# hft/py_research.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
import seaborn as sns

class ImbalanceResearch:
    """Research framework for imbalance signal calibration"""
    
    def __init__(self, tick_data: pd.DataFrame):
        self.data = tick_data
        self.signals = None
        self.thresholds = {}
        
    def calculate_imbalance_features(self, levels: int = 5) -> pd.DataFrame:
        """Calculate various imbalance metrics"""
        features = pd.DataFrame(index=self.data.index)
        
        for level in range(1, levels + 1):
            # Volume imbalance at each level
            bid_vol = self.data[f'bid_vol_{level}']
            ask_vol = self.data[f'ask_vol_{level}']
            features[f'vol_imb_l{level}'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1)
            
            # Weighted by level
            weight = 1.0 / level
            features[f'weighted_imb_l{level}'] = features[f'vol_imb_l{level}'] * weight
        
        # Composite imbalances
        features['total_vol_imb'] = features[[f'vol_imb_l{i}' for i in range(1, levels + 1)]].mean(axis=1)
        features['weighted_total_imb'] = features[[f'weighted_imb_l{i}' for i in range(1, levels + 1)]].sum(axis=1)
        
        # Order count imbalance
        features['order_count_imb'] = (self.data['bid_orders'] - self.data['ask_orders']) / \
                                      (self.data['bid_orders'] + self.data['ask_orders'] + 1)
        
        # Queue position metrics
        features['queue_position'] = self.calculate_queue_position()
        
        # Micro-price
        features['micro_price'] = self.calculate_micro_price()
        features['micro_price_change'] = features['micro_price'].diff()
        
        return features
    
    def calculate_micro_price(self) -> pd.Series:
        """Calculate micro-price from order book"""
        bid_price = self.data['bid_price_1']
        ask_price = self.data['ask_price_1']
        bid_size = self.data['bid_vol_1']
        ask_size = self.data['ask_vol_1']
        
        return (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)
    
    def calculate_queue_position(self) -> pd.Series:
        """Estimate queue position probability"""
        # Simplified queue model
        ahead_volume = self.data['cum_volume_ahead']
        level_volume = self.data['level_volume']
        arrival_rate = self.data['order_arrival_rate']
        
        # Poisson approximation for queue position
        prob_fill = np.exp(-ahead_volume / (level_volume * arrival_rate))
        return prob_fill
    
    def backtest_signal(self, features: pd.DataFrame, 
                       forward_window: int = 10) -> pd.DataFrame:
        """Backtest imbalance signals"""
        results = []
        
        # Calculate forward returns
        mid_price = (self.data['bid_price_1'] + self.data['ask_price_1']) / 2
        forward_returns = mid_price.shift(-forward_window) / mid_price - 1
        
        # Test different thresholds
        thresholds = np.arange(0.05, 0.5, 0.05)
        
        for thresh in thresholds:
            # Generate signals
            buy_signal = features['total_vol_imb'] > thresh
            sell_signal = features['total_vol_imb'] < -thresh
            
            # Calculate PnL
            pnl = pd.Series(0.0, index=features.index)
            pnl[buy_signal] = forward_returns[buy_signal]
            pnl[sell_signal] = -forward_returns[sell_signal]
            
            # Transaction costs (conservative estimate)
            costs = 0.0001  # 1 bps
            pnl[buy_signal | sell_signal] -= costs
            
            # Performance metrics
            sharpe = np.sqrt(252 * 390 * 60) * pnl.mean() / pnl.std() if pnl.std() > 0 else 0
            hit_rate = (pnl > 0).mean()
            avg_pnl = pnl[pnl != 0].mean() * 10000  # in bps
            
            results.append({
                'threshold': thresh,
                'sharpe': sharpe,
                'hit_rate': hit_rate,
                'avg_pnl_bps': avg_pnl,
                'num_trades': (buy_signal | sell_signal).sum()
            })
        
        return pd.DataFrame(results)
    
    def optimize_thresholds(self, features: pd.DataFrame) -> Dict[str, float]:
        """Optimize signal thresholds using ML"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        
        # Prepare labels (profitable trades)
        mid_price = (self.data['bid_price_1'] + self.data['ask_price_1']) / 2
        forward_returns = mid_price.shift(-10) / mid_price - 1
        
        # Binary classification: profitable vs not
        y = (forward_returns > 0.0002).astype(int)  # 2 bps threshold
        
        # Feature matrix
        feature_cols = ['total_vol_imb', 'weighted_total_imb', 'order_count_imb', 
                       'queue_position', 'micro_price_change']
        X = features[feature_cols].fillna(0)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            
            # Predict probabilities
            probs = rf.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold
            best_thresh = 0.5
            best_score = 0
            
            for thresh in np.arange(0.3, 0.8, 0.05):
                preds = (probs > thresh).astype(int)
                score = ((preds == y_test) & (y_test == 1)).sum()  # True positives
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
            
            scores.append(best_thresh)
        
        return {
            'weak_threshold': np.percentile(scores, 25),
            'strong_threshold': np.percentile(scores, 75)
        }
```

### 3. Execution Engine with Risk Management

```cpp
// hft/execution_engine.hpp
#pragma once
#include "imbalance_signal.hpp"
#include <memory>
#include <queue>

namespace hft {

class ScalperExecutionEngine {
private:
    struct Position {
        int64_t quantity;
        double avg_price;
        std::chrono::nanoseconds entry_time;
        double unrealized_pnl;
        double max_adverse_excursion;
    };
    
    struct RiskLimits {
        int64_t max_position = 1000;
        double max_loss = -100.0;
        double position_timeout_ms = 5000;
        double max_spread = 0.005;
        int max_trades_per_minute = 100;
    };
    
    ImbalanceSignal<10> signal_engine_;
    Position current_position_;
    RiskLimits risk_limits_;
    std::atomic<int> trades_this_minute_{0};
    
public:
    struct ExecutionDecision {
        enum Action { NONE, BUY, SELL, CLOSE_POSITION };
        Action action;
        int64_t quantity;
        double limit_price;
        bool is_aggressive;  // IOC vs passive
    };
    
    ExecutionDecision make_decision(
        const ImbalanceSignal<10>::Signal& signal,
        const MarketData& market);
    
    bool check_risk_limits();
    void update_position(int64_t qty, double price);
    double calculate_position_pnl(double current_price);
};

// hft/execution_engine.cpp
ExecutionDecision ScalperExecutionEngine::make_decision(
    const ImbalanceSignal<10>::Signal& signal,
    const MarketData& market) {
    
    ExecutionDecision decision;
    decision.action = ExecutionDecision::NONE;
    
    // Check risk limits first
    if (!check_risk_limits()) {
        // Force close if in violation
        if (current_position_.quantity != 0) {
            decision.action = ExecutionDecision::CLOSE_POSITION;
            decision.quantity = -current_position_.quantity;
            decision.is_aggressive = true;  // Use IOC to close immediately
            decision.limit_price = current_position_.quantity > 0 ? 
                                  market.best_bid : market.best_ask;
        }
        return decision;
    }
    
    // Position timeout check
    auto now = std::chrono::high_resolution_clock::now();
    auto position_age = now - current_position_.entry_time;
    if (position_age > std::chrono::milliseconds(
            static_cast<int>(risk_limits_.position_timeout_ms))) {
        if (current_position_.quantity != 0) {
            decision.action = ExecutionDecision::CLOSE_POSITION;
            decision.quantity = -current_position_.quantity;
            decision.is_aggressive = false;  // Passive close
            decision.limit_price = current_position_.quantity > 0 ?
                                  market.best_bid + market.tick_size :
                                  market.best_ask - market.tick_size;
            return decision;
        }
    }
    
    // Signal-based execution
    switch (signal.direction) {
        case ImbalanceSignal<10>::Signal::STRONG_BUY:
            if (current_position_.quantity < risk_limits_.max_position) {
                decision.action = ExecutionDecision::BUY;
                decision.quantity = std::min(
                    100L, 
                    risk_limits_.max_position - current_position_.quantity
                );
                decision.is_aggressive = true;  // Aggressive for strong signal
                decision.limit_price = market.best_ask;
            }
            break;
            
        case ImbalanceSignal<10>::Signal::BUY:
            if (current_position_.quantity < risk_limits_.max_position / 2) {
                decision.action = ExecutionDecision::BUY;
                decision.quantity = 100;
                decision.is_aggressive = false;  // Passive for weak signal
                decision.limit_price = market.best_bid;
            }
            break;
            
        case ImbalanceSignal<10>::Signal::STRONG_SELL:
            if (current_position_.quantity > -risk_limits_.max_position) {
                decision.action = ExecutionDecision::SELL;
                decision.quantity = std::min(
                    100L,
                    risk_limits_.max_position + current_position_.quantity
                );
                decision.is_aggressive = true;
                decision.limit_price = market.best_bid;
            }
            break;
            
        case ImbalanceSignal<10>::Signal::SELL:
            if (current_position_.quantity > -risk_limits_.max_position / 2) {
                decision.action = ExecutionDecision::SELL;
                decision.quantity = 100;
                decision.is_aggressive = false;
                decision.limit_price = market.best_ask;
            }
            break;
            
        default:
            // No signal - consider closing position if profitable
            if (current_position_.quantity != 0) {
                double pnl = calculate_position_pnl(
                    current_position_.quantity > 0 ? market.best_bid : market.best_ask
                );
                if (pnl > 0.0002 * std::abs(current_position_.quantity * current_position_.avg_price)) {
                    decision.action = ExecutionDecision::CLOSE_POSITION;
                    decision.quantity = -current_position_.quantity;
                    decision.is_aggressive = false;
                    decision.limit_price = current_position_.quantity > 0 ?
                                          market.best_bid : market.best_ask;
                }
            }
            break;
    }
    
    return decision;
}

bool ScalperExecutionEngine::check_risk_limits() {
    // Position limit
    if (std::abs(current_position_.quantity) >= risk_limits_.max_position) {
        return false;
    }
    
    // Loss limit
    if (current_position_.unrealized_pnl < risk_limits_.max_loss) {
        return false;
    }
    
    // Trade frequency limit
    if (trades_this_minute_.load() >= risk_limits_.max_trades_per_minute) {
        return false;
    }
    
    return true;
}
```

### 4. Performance Analysis & Metrics

```python
# analysis/scalper_metrics.py
class ScalperPerformance:
    """Performance analysis for imbalance scalper"""
    
    def analyze_trades(self, trades_df: pd.DataFrame) -> Dict:
        """Comprehensive trade analysis"""
        
        metrics = {
            # PnL metrics
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl_per_trade': trades_df['pnl'].mean(),
            'win_rate': (trades_df['pnl'] > 0).mean(),
            'profit_factor': trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                           abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()),
            
            # Risk metrics
            'sharpe_ratio': self.calculate_sharpe(trades_df),
            'sortino_ratio': self.calculate_sortino(trades_df),
            'max_drawdown': self.calculate_max_drawdown(trades_df),
            'var_95': trades_df['pnl'].quantile(0.05),
            
            # Execution metrics
            'avg_holding_time_ms': trades_df['holding_time'].mean(),
            'avg_slippage_bps': trades_df['slippage'].mean() * 10000,
            'fill_rate': trades_df['filled'].mean(),
            
            # Signal metrics
            'signal_accuracy': self.calculate_signal_accuracy(trades_df),
            'avg_signal_strength': trades_df['signal_strength'].mean(),
            'false_positive_rate': self.calculate_false_positives(trades_df)
        }
        
        return metrics
    
    def calculate_signal_accuracy(self, trades_df: pd.DataFrame) -> float:
        """Measure how often signals predict correct direction"""
        correct = ((trades_df['signal_direction'] == 'BUY') & (trades_df['pnl'] > 0)) | \
                 ((trades_df['signal_direction'] == 'SELL') & (trades_df['pnl'] > 0))
        return correct.mean()
    
    def plot_performance(self, trades_df: pd.DataFrame):
        """Visualize scalper performance"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Cumulative PnL
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
        axes[0, 0].plot(trades_df.index, trades_df['cum_pnl'])
        axes[0, 0].set_title('Cumulative PnL')
        
        # PnL distribution
        axes[0, 1].hist(trades_df['pnl'], bins=50, alpha=0.7)
        axes[0, 1].set_title('PnL Distribution')
        
        # Holding time vs PnL
        axes[0, 2].scatter(trades_df['holding_time'], trades_df['pnl'], alpha=0.5)
        axes[0, 2].set_title('Holding Time vs PnL')
        
        # Signal strength vs PnL
        axes[1, 0].scatter(trades_df['signal_strength'], trades_df['pnl'], alpha=0.5)
        axes[1, 0].set_title('Signal Strength vs PnL')
        
        # Hourly performance
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        hourly_pnl = trades_df.groupby('hour')['pnl'].mean()
        axes[1, 1].bar(hourly_pnl.index, hourly_pnl.values)
        axes[1, 1].set_title('Average PnL by Hour')
        
        # Drawdown
        drawdown = self.calculate_drawdown_series(trades_df)
        axes[1, 2].fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red')
        axes[1, 2].set_title('Drawdown')
        
        plt.tight_layout()
        return fig
```

## Implementation Checklist

### Phase 1: Research & Calibration
- [ ] Collect high-frequency tick data
- [ ] Calculate imbalance features
- [ ] Backtest various thresholds
- [ ] Optimize signal parameters

### Phase 2: C++ Implementation
- [ ] Implement imbalance signal calculator
- [ ] Build execution engine with risk management
- [ ] Create order management system
- [ ] Add latency monitoring

### Phase 3: Integration & Testing
- [ ] Python-C++ bindings for research
- [ ] Unit tests for signal generation
- [ ] Integration tests with mock exchange
- [ ] Latency benchmarking

### Phase 4: Production Deployment
- [ ] Real-time monitoring dashboard
- [ ] Performance tracking system
- [ ] Risk alerts and circuit breakers
- [ ] A/B testing framework

## Configuration

```yaml
# config/scalper_config.yaml
signal:
  depth_levels: 5
  weak_threshold: 0.15
  strong_threshold: 0.35
  update_frequency_us: 100

execution:
  default_size: 100
  max_position: 1000
  position_timeout_ms: 5000
  
risk:
  max_loss: -100.0
  max_trades_per_minute: 100
  max_spread: 0.005
  
latency:
  target_signal_latency_us: 10
  max_order_latency_us: 100
```

## Expected Performance

Target metrics on live trading:
- Signal Latency: < 10 microseconds
- Sharpe Ratio: > 3.0
- Win Rate: > 55%
- Average Holding Time: < 5 seconds
- Daily Trades: 500-2000
- Average PnL per Trade: 0.5-2 bps