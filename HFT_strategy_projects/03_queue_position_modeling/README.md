# Queue Position Modeling & Requote Policy

## Overview
C++ implementation for estimating fill probability based on queue position in limit order book. Implements dynamic cancel/replace strategies to maintain top-of-book priority under latency constraints.

## Core Architecture

### 1. Queue Position Model (`hft/queue_model.hpp`)

```cpp
// hft/queue_model.hpp
#pragma once
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

namespace hft {

class QueuePositionModel {
private:
    struct QueueState {
        uint64_t ahead_volume;      // Volume ahead in queue
        uint64_t behind_volume;     // Volume behind in queue
        uint64_t total_volume;      // Total volume at price level
        uint32_t order_count;       // Number of orders at level
        double arrival_rate;        // Order arrival rate (orders/sec)
        double cancellation_rate;   // Cancellation rate
        std::chrono::nanoseconds timestamp;
    };
    
    struct FillProbability {
        double immediate;    // Probability of immediate fill
        double short_term;   // Probability within 100ms
        double medium_term;  // Probability within 1s
        double long_term;    // Probability within 10s
        double expected_time_ms;  // Expected time to fill
    };
    
    // Model parameters
    struct ModelParams {
        double base_arrival_rate = 10.0;    // Orders per second
        double base_cancel_rate = 0.3;      // Cancellation probability
        double size_decay_factor = 0.001;   // Impact of queue size
        double time_decay_factor = 0.995;   // Time decay of fill probability
        double aggressive_order_prob = 0.1;  // Probability of aggressive order
    } params_;
    
public:
    FillProbability estimate_fill_probability(const QueueState& state);
    double calculate_queue_dynamics(const QueueState& state, double horizon_ms);
    bool should_requote(const QueueState& current, const FillProbability& prob);
    
private:
    double poisson_fill_probability(double ahead_volume, double arrival_rate, double time_ms);
    double empirical_fill_curve(double queue_position, double total_volume);
};

// hft/queue_model.cpp
FillProbability QueuePositionModel::estimate_fill_probability(const QueueState& state) {
    FillProbability prob;
    
    // Normalized queue position [0, 1]
    double queue_position = static_cast<double>(state.ahead_volume) / 
                          (state.total_volume + 1.0);
    
    // Immediate fill probability (for market/aggressive orders)
    prob.immediate = params_.aggressive_order_prob * 
                    std::exp(-queue_position * 3.0);  // Exponential decay
    
    // Short-term (100ms) - Poisson model
    prob.short_term = poisson_fill_probability(
        state.ahead_volume, 
        state.arrival_rate * params_.aggressive_order_prob,
        100.0
    );
    
    // Medium-term (1s) - Include cancellations
    double effective_arrival = state.arrival_rate * 
                              (params_.aggressive_order_prob + 
                               params_.base_cancel_rate * 0.5);
    prob.medium_term = poisson_fill_probability(
        state.ahead_volume,
        effective_arrival,
        1000.0
    );
    
    // Long-term (10s) - Empirical model
    prob.long_term = empirical_fill_curve(queue_position, state.total_volume);
    
    // Expected time to fill (milliseconds)
    if (state.arrival_rate > 0) {
        double lambda = state.arrival_rate * params_.aggressive_order_prob;
        prob.expected_time_ms = (state.ahead_volume / lambda) * 1000.0;
    } else {
        prob.expected_time_ms = 1e6;  // Very large number
    }
    
    return prob;
}

double QueuePositionModel::poisson_fill_probability(
    double ahead_volume, 
    double arrival_rate, 
    double time_ms) {
    
    // Poisson process: P(fill) = 1 - P(less than ahead_volume aggressive orders arrive)
    double lambda = arrival_rate * time_ms / 1000.0;
    
    // Use incomplete gamma function for CDF
    double prob = 0.0;
    double term = std::exp(-lambda);
    double sum = term;
    
    for (int k = 0; k < ahead_volume && k < 100; ++k) {
        term *= lambda / (k + 1);
        sum += term;
    }
    
    prob = 1.0 - sum;
    return std::max(0.0, std::min(1.0, prob));
}

double QueuePositionModel::empirical_fill_curve(
    double queue_position, 
    double total_volume) {
    
    // Empirically calibrated fill curve
    // Based on historical fill rates at different queue positions
    
    double base_prob = 0.9 * std::exp(-3.0 * queue_position);
    
    // Adjust for total volume (larger queues = slower fills)
    double volume_factor = 1.0 / (1.0 + params_.size_decay_factor * total_volume);
    
    return base_prob * volume_factor;
}

bool QueuePositionModel::should_requote(
    const QueueState& current, 
    const FillProbability& prob) {
    
    // Requote decision based on fill probability and queue dynamics
    
    // If very unlikely to fill soon, consider requoting
    if (prob.short_term < 0.05 && prob.medium_term < 0.15) {
        return true;
    }
    
    // If queue is growing rapidly behind us, stay put
    double queue_growth_rate = current.behind_volume / 
                              (current.total_volume + 1.0);
    if (queue_growth_rate > 0.3) {
        return false;
    }
    
    // If expected time to fill is too long
    if (prob.expected_time_ms > 5000) {  // 5 seconds
        return true;
    }
    
    return false;
}
```

### 2. Requote Policy Engine (`hft/requote_policy.hpp`)

```cpp
// hft/requote_policy.hpp
#pragma once
#include "queue_model.hpp"
#include <memory>
#include <unordered_map>

namespace hft {

class RequotePolicy {
private:
    struct OrderInfo {
        uint64_t order_id;
        double price;
        uint64_t size;
        std::chrono::nanoseconds entry_time;
        uint64_t initial_ahead_volume;
        int requote_count;
        double cumulative_queue_time_ms;
    };
    
    struct RequoteDecision {
        enum Action { STAY, CANCEL_REPLACE, CANCEL, MODIFY_SIZE };
        Action action;
        double new_price;
        uint64_t new_size;
        double urgency;  // 0-1 scale
        std::string reason;
    };
    
    QueuePositionModel queue_model_;
    std::unordered_map<uint64_t, OrderInfo> active_orders_;
    
    // Latency-aware parameters
    struct LatencyParams {
        double cancel_latency_us = 50.0;
        double new_order_latency_us = 75.0;
        double modify_latency_us = 60.0;
        double market_data_latency_us = 10.0;
    } latency_;
    
public:
    RequoteDecision evaluate_order(
        uint64_t order_id,
        const QueueState& current_state,
        const MarketConditions& market);
    
    void execute_requote(const RequoteDecision& decision, uint64_t order_id);
    
private:
    double calculate_requote_cost(const OrderInfo& order, 
                                 const RequoteDecision& decision);
    double estimate_opportunity_cost(const QueueState& state, double time_ms);
};

// hft/requote_policy.cpp
RequoteDecision RequotePolicy::evaluate_order(
    uint64_t order_id,
    const QueueState& current_state,
    const MarketConditions& market) {
    
    RequoteDecision decision;
    decision.action = RequoteDecision::STAY;
    
    auto it = active_orders_.find(order_id);
    if (it == active_orders_.end()) {
        return decision;
    }
    
    OrderInfo& order = it->second;
    
    // Calculate fill probability
    auto fill_prob = queue_model_.estimate_fill_probability(current_state);
    
    // Time in queue
    auto now = std::chrono::high_resolution_clock::now();
    auto time_in_queue = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - order.entry_time
    ).count();
    
    // Decision logic based on multiple factors
    
    // 1. Queue position deterioration
    if (current_state.ahead_volume > order.initial_ahead_volume * 1.5) {
        if (fill_prob.medium_term < 0.1) {
            decision.action = RequoteDecision::CANCEL_REPLACE;
            decision.new_price = order.price + market.tick_size;  // More aggressive
            decision.new_size = order.size;
            decision.urgency = 0.8;
            decision.reason = "Queue position deteriorated significantly";
            return decision;
        }
    }
    
    // 2. Time-based requote
    if (time_in_queue > 3000 && fill_prob.immediate < 0.01) {  // 3 seconds
        decision.action = RequoteDecision::CANCEL_REPLACE;
        decision.new_price = order.price;  // Same price, back of queue
        decision.new_size = order.size;
        decision.urgency = 0.5;
        decision.reason = "Timeout - refresh queue position";
        return decision;
    }
    
    // 3. Market conditions changed
    if (std::abs(market.fair_value - order.price) > 2 * market.tick_size) {
        decision.action = RequoteDecision::CANCEL;
        decision.urgency = 1.0;
        decision.reason = "Market moved away from order price";
        return decision;
    }
    
    // 4. High fill probability - increase size
    if (fill_prob.immediate > 0.7 && order.requote_count < 2) {
        decision.action = RequoteDecision::MODIFY_SIZE;
        decision.new_size = order.size * 2;  // Double size
        decision.urgency = 0.6;
        decision.reason = "High fill probability - increase size";
        return decision;
    }
    
    // 5. Latency-aware decision
    double requote_latency = latency_.cancel_latency_us + latency_.new_order_latency_us;
    double expected_fill_time = fill_prob.expected_time_ms * 1000;  // Convert to microseconds
    
    if (requote_latency < expected_fill_time * 0.01) {  // Requote is fast relative to fill time
        if (queue_model_.should_requote(current_state, fill_prob)) {
            decision.action = RequoteDecision::CANCEL_REPLACE;
            decision.new_price = order.price;
            decision.new_size = order.size;
            decision.urgency = 0.4;
            decision.reason = "Opportunistic requote - low latency cost";
        }
    }
    
    return decision;
}

double RequotePolicy::calculate_requote_cost(
    const OrderInfo& order,
    const RequoteDecision& decision) {
    
    double cost = 0.0;
    
    // Latency cost (opportunity cost during requote)
    double latency_ms = 0.0;
    switch (decision.action) {
        case RequoteDecision::CANCEL_REPLACE:
            latency_ms = (latency_.cancel_latency_us + latency_.new_order_latency_us) / 1000.0;
            break;
        case RequoteDecision::MODIFY_SIZE:
            latency_ms = latency_.modify_latency_us / 1000.0;
            break;
        case RequoteDecision::CANCEL:
            latency_ms = latency_.cancel_latency_us / 1000.0;
            break;
        default:
            break;
    }
    
    // Opportunity cost during latency
    cost += estimate_opportunity_cost(current_state, latency_ms);
    
    // Queue position loss cost
    if (decision.action == RequoteDecision::CANCEL_REPLACE) {
        cost += 0.0001;  // Fixed cost for losing queue position (1 bps)
    }
    
    return cost;
}
```

### 3. Testing Framework (`tests/test_queue_fill.cpp`)

```cpp
// tests/test_queue_fill.cpp
#include <gtest/gtest.h>
#include "../hft/queue_model.hpp"
#include "../hft/requote_policy.hpp"
#include <random>

class QueueModelTest : public ::testing::Test {
protected:
    hft::QueuePositionModel model;
    hft::RequotePolicy policy;
    
    hft::QueueState generate_random_state() {
        static std::mt19937 gen(42);
        std::uniform_int_distribution<> vol_dist(100, 10000);
        std::uniform_real_distribution<> rate_dist(1.0, 50.0);
        
        hft::QueueState state;
        state.total_volume = vol_dist(gen);
        state.ahead_volume = vol_dist(gen) % state.total_volume;
        state.behind_volume = state.total_volume - state.ahead_volume;
        state.arrival_rate = rate_dist(gen);
        state.cancellation_rate = 0.3;
        
        return state;
    }
};

TEST_F(QueueModelTest, FillProbabilityMonotonic) {
    // Test that fill probability decreases with queue position
    hft::QueueState state;
    state.total_volume = 1000;
    state.arrival_rate = 10.0;
    
    double prev_prob = 1.0;
    for (uint64_t ahead = 0; ahead <= 1000; ahead += 100) {
        state.ahead_volume = ahead;
        state.behind_volume = 1000 - ahead;
        
        auto prob = model.estimate_fill_probability(state);
        EXPECT_LE(prob.immediate, prev_prob);
        prev_prob = prob.immediate;
    }
}

TEST_F(QueueModelTest, RequoteDecisionLogic) {
    // Test requote decision under various scenarios
    
    // Scenario 1: Good queue position
    hft::QueueState good_state;
    good_state.ahead_volume = 100;
    good_state.total_volume = 5000;
    good_state.arrival_rate = 20.0;
    
    auto fill_prob = model.estimate_fill_probability(good_state);
    EXPECT_FALSE(model.should_requote(good_state, fill_prob));
    
    // Scenario 2: Bad queue position
    hft::QueueState bad_state;
    bad_state.ahead_volume = 4900;
    bad_state.total_volume = 5000;
    bad_state.arrival_rate = 2.0;
    
    fill_prob = model.estimate_fill_probability(bad_state);
    EXPECT_TRUE(model.should_requote(bad_state, fill_prob));
}

TEST_F(QueueModelTest, LatencyAwareRequote) {
    // Test that requote decisions account for latency
    
    hft::MarketConditions market;
    market.fair_value = 100.0;
    market.tick_size = 0.01;
    
    // Create order with poor queue position
    uint64_t order_id = 12345;
    hft::QueueState state;
    state.ahead_volume = 8000;
    state.total_volume = 10000;
    state.arrival_rate = 5.0;
    
    auto decision = policy.evaluate_order(order_id, state, market);
    
    // Should recommend requote due to poor position
    EXPECT_NE(decision.action, hft::RequoteDecision::STAY);
}

TEST_F(QueueModelTest, StressTest) {
    // Stress test with many random states
    const int num_tests = 10000;
    
    for (int i = 0; i < num_tests; ++i) {
        auto state = generate_random_state();
        auto prob = model.estimate_fill_probability(state);
        
        // Sanity checks
        EXPECT_GE(prob.immediate, 0.0);
        EXPECT_LE(prob.immediate, 1.0);
        EXPECT_LE(prob.immediate, prob.short_term);
        EXPECT_LE(prob.short_term, prob.medium_term);
        EXPECT_LE(prob.medium_term, prob.long_term);
    }
}
```

### 4. Performance Monitoring

```cpp
// hft/queue_monitor.hpp
#pragma once
#include <atomic>
#include <vector>
#include <chrono>

namespace hft {

class QueuePerformanceMonitor {
private:
    struct FillStatistics {
        std::atomic<uint64_t> total_orders{0};
        std::atomic<uint64_t> filled_orders{0};
        std::atomic<uint64_t> cancelled_orders{0};
        std::atomic<uint64_t> requoted_orders{0};
        std::atomic<double> total_queue_time_ms{0.0};
        std::atomic<double> total_fill_time_ms{0.0};
    };
    
    struct QueueMetrics {
        double avg_fill_rate;
        double avg_queue_time_ms;
        double avg_queue_position;
        double requote_effectiveness;
        std::vector<double> fill_probability_accuracy;
    };
    
    FillStatistics stats_;
    std::vector<std::pair<double, double>> predicted_vs_actual_;
    
public:
    void record_order_placed(uint64_t order_id, const QueueState& initial_state);
    void record_order_filled(uint64_t order_id, double queue_time_ms);
    void record_order_cancelled(uint64_t order_id, double queue_time_ms);
    void record_requote(uint64_t old_id, uint64_t new_id);
    
    QueueMetrics calculate_metrics() const;
    double calculate_model_accuracy() const;
    void generate_report(const std::string& filename) const;
};

// hft/queue_monitor.cpp
QueueMetrics QueuePerformanceMonitor::calculate_metrics() const {
    QueueMetrics metrics;
    
    uint64_t total = stats_.total_orders.load();
    if (total == 0) return metrics;
    
    metrics.avg_fill_rate = static_cast<double>(stats_.filled_orders.load()) / total;
    
    uint64_t filled = stats_.filled_orders.load();
    if (filled > 0) {
        metrics.avg_queue_time_ms = stats_.total_queue_time_ms.load() / filled;
    }
    
    // Calculate requote effectiveness
    uint64_t requoted = stats_.requoted_orders.load();
    if (requoted > 0) {
        // Measure if requoting improved fill rates
        metrics.requote_effectiveness = calculate_requote_improvement();
    }
    
    // Model accuracy
    metrics.fill_probability_accuracy.clear();
    for (const auto& [predicted, actual] : predicted_vs_actual_) {
        double accuracy = 1.0 - std::abs(predicted - actual);
        metrics.fill_probability_accuracy.push_back(accuracy);
    }
    
    return metrics;
}

double QueuePerformanceMonitor::calculate_model_accuracy() const {
    if (predicted_vs_actual_.empty()) return 0.0;
    
    double mse = 0.0;
    for (const auto& [predicted, actual] : predicted_vs_actual_) {
        double error = predicted - actual;
        mse += error * error;
    }
    
    mse /= predicted_vs_actual_.size();
    return 1.0 - std::sqrt(mse);  // Convert MSE to accuracy score
}
}
```

## Implementation Checklist

### Phase 1: Core Modeling
- [ ] Implement queue position estimation
- [ ] Build Poisson fill probability model
- [ ] Create empirical fill curves from data
- [ ] Implement queue dynamics calculation

### Phase 2: Requote Logic
- [ ] Design requote decision engine
- [ ] Implement latency-aware cost model
- [ ] Build order tracking system
- [ ] Create cancellation strategies

### Phase 3: Testing & Calibration
- [ ] Unit tests for queue model
- [ ] Integration tests with mock exchange
- [ ] Calibrate model parameters from historical data
- [ ] Validate fill probability predictions

### Phase 4: Production Features
- [ ] Real-time performance monitoring
- [ ] Model accuracy tracking
- [ ] A/B testing framework
- [ ] Dynamic parameter adjustment

## Configuration

```yaml
# config/queue_config.yaml
model:
  base_arrival_rate: 10.0
  base_cancel_rate: 0.3
  size_decay_factor: 0.001
  time_decay_factor: 0.995
  aggressive_order_prob: 0.1

requote_policy:
  max_requotes_per_order: 3
  min_time_between_requotes_ms: 500
  queue_deterioration_threshold: 1.5
  timeout_threshold_ms: 3000

latency:
  cancel_latency_us: 50
  new_order_latency_us: 75
  modify_latency_us: 60
  market_data_latency_us: 10

monitoring:
  track_predictions: true
  report_frequency_seconds: 60
  accuracy_window_size: 1000
```

## Expected Performance

Target metrics:
- Fill Rate: > 70%
- Average Queue Time: < 2 seconds
- Model Accuracy: > 80%
- Requote Effectiveness: > 60%
- Latency per Decision: < 5 microseconds