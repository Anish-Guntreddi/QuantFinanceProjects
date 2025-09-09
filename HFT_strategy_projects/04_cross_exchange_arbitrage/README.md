# Cross-Exchange Arbitrage (Latency-aware)

## Overview
C++ implementation with optional Python orchestrator for cross-exchange arbitrage. Monitors synthetic multi-venue feeds, triggers when net-of-fees spread exceeds threshold, accounting for cancel risk, venue fees/rebates, and link latency.

## Core Architecture

### 1. Multi-Venue Feed Aggregator (`hft/xex_arb/`)

```cpp
// hft/xex_arb/venue_aggregator.hpp
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <unordered_map>

namespace xex {

struct VenueConfig {
    std::string name;
    double maker_fee;
    double taker_fee;
    double maker_rebate;
    double latency_us;
    double cancel_success_rate;
    bool supports_hidden;
};

struct ArbitrageOpportunity {
    std::string buy_venue;
    std::string sell_venue;
    double buy_price;
    double sell_price;
    uint64_t available_size;
    double gross_spread;
    double net_profit;
    double execution_risk;
    std::chrono::nanoseconds timestamp;
};

template<size_t NUM_VENUES>
class CrossExchangeArbitrage {
private:
    std::array<VenueConfig, NUM_VENUES> venues_;
    
    struct VenueBook {
        double best_bid;
        double best_ask;
        uint64_t bid_size;
        uint64_t ask_size;
        std::chrono::nanoseconds last_update;
        bool is_stale;
    };
    
    std::unordered_map<std::string, VenueBook> books_;
    std::atomic<uint64_t> opportunities_found_{0};
    std::atomic<uint64_t> opportunities_executed_{0};
    
public:
    std::vector<ArbitrageOpportunity> scan_opportunities();
    double calculate_net_profit(const ArbitrageOpportunity& opp);
    bool should_execute(const ArbitrageOpportunity& opp);
    void execute_arbitrage(const ArbitrageOpportunity& opp);
};

// hft/xex_arb/arbitrage_engine.cpp
template<size_t NUM_VENUES>
std::vector<ArbitrageOpportunity> 
CrossExchangeArbitrage<NUM_VENUES>::scan_opportunities() {
    
    std::vector<ArbitrageOpportunity> opportunities;
    auto now = std::chrono::high_resolution_clock::now();
    
    // Check all venue pairs
    for (size_t i = 0; i < NUM_VENUES; ++i) {
        for (size_t j = i + 1; j < NUM_VENUES; ++j) {
            const auto& venue_a = venues_[i];
            const auto& venue_b = venues_[j];
            
            const auto& book_a = books_[venue_a.name];
            const auto& book_b = books_[venue_b.name];
            
            // Skip if data is stale
            auto age_a = now - book_a.last_update;
            auto age_b = now - book_b.last_update;
            
            if (age_a > std::chrono::milliseconds(100) || 
                age_b > std::chrono::milliseconds(100)) {
                continue;
            }
            
            // Check both directions
            // Buy on A, Sell on B
            if (book_a.best_ask < book_b.best_bid) {
                ArbitrageOpportunity opp;
                opp.buy_venue = venue_a.name;
                opp.sell_venue = venue_b.name;
                opp.buy_price = book_a.best_ask;
                opp.sell_price = book_b.best_bid;
                opp.available_size = std::min(book_a.ask_size, book_b.bid_size);
                opp.gross_spread = (book_b.best_bid - book_a.best_ask) / book_a.best_ask;
                opp.timestamp = now;
                
                // Calculate net profit including fees
                opp.net_profit = calculate_net_profit(opp);
                
                if (opp.net_profit > 0) {
                    opportunities.push_back(opp);
                }
            }
            
            // Buy on B, Sell on A
            if (book_b.best_ask < book_a.best_bid) {
                ArbitrageOpportunity opp;
                opp.buy_venue = venue_b.name;
                opp.sell_venue = venue_a.name;
                opp.buy_price = book_b.best_ask;
                opp.sell_price = book_a.best_bid;
                opp.available_size = std::min(book_b.ask_size, book_a.bid_size);
                opp.gross_spread = (book_a.best_bid - book_b.best_ask) / book_b.best_ask;
                opp.timestamp = now;
                
                opp.net_profit = calculate_net_profit(opp);
                
                if (opp.net_profit > 0) {
                    opportunities.push_back(opp);
                }
            }
        }
    }
    
    // Sort by net profit
    std::sort(opportunities.begin(), opportunities.end(),
              [](const auto& a, const auto& b) {
                  return a.net_profit > b.net_profit;
              });
    
    return opportunities;
}

template<size_t NUM_VENUES>
double CrossExchangeArbitrage<NUM_VENUES>::calculate_net_profit(
    const ArbitrageOpportunity& opp) {
    
    // Find venue configurations
    const VenueConfig* buy_venue = nullptr;
    const VenueConfig* sell_venue = nullptr;
    
    for (const auto& venue : venues_) {
        if (venue.name == opp.buy_venue) buy_venue = &venue;
        if (venue.name == opp.sell_venue) sell_venue = &venue;
    }
    
    if (!buy_venue || !sell_venue) return -1.0;
    
    // Calculate fees and rebates
    double buy_cost = opp.buy_price * opp.available_size;
    double sell_revenue = opp.sell_price * opp.available_size;
    
    // Taker fees (assuming aggressive execution)
    double buy_fee = buy_cost * buy_venue->taker_fee;
    double sell_fee = sell_revenue * sell_venue->taker_fee;
    
    // Some venues offer rebates
    double buy_rebate = buy_cost * buy_venue->maker_rebate;
    double sell_rebate = sell_revenue * sell_venue->maker_rebate;
    
    // Net profit
    double gross_profit = sell_revenue - buy_cost;
    double net_profit = gross_profit - buy_fee - sell_fee + buy_rebate + sell_rebate;
    
    // Adjust for execution risk
    double latency_risk = 1.0 - (buy_venue->latency_us + sell_venue->latency_us) / 1000000.0;
    double cancel_risk = buy_venue->cancel_success_rate * sell_venue->cancel_success_rate;
    
    return net_profit * latency_risk * cancel_risk;
}
}
```

### 2. Latency-Aware Execution (`hft/xex_arb/latency_executor.cpp`)

```cpp
// hft/xex_arb/latency_executor.hpp
#pragma once
#include <thread>
#include <future>

namespace xex {

class LatencyAwareExecutor {
private:
    struct ExecutionPlan {
        std::vector<Order> buy_orders;
        std::vector<Order> sell_orders;
        std::chrono::nanoseconds max_latency;
        bool use_synchronized_execution;
    };
    
    struct ExecutionResult {
        bool success;
        double filled_quantity;
        double avg_buy_price;
        double avg_sell_price;
        double realized_pnl;
        std::chrono::nanoseconds total_latency;
    };
    
public:
    ExecutionResult execute_synchronized(
        const ArbitrageOpportunity& opp,
        const std::map<std::string, double>& venue_latencies);
    
    ExecutionResult execute_sequential(
        const ArbitrageOpportunity& opp);
        
private:
    void send_order_with_timing(
        const Order& order,
        std::chrono::nanoseconds target_time);
};

ExecutionResult LatencyAwareExecutor::execute_synchronized(
    const ArbitrageOpportunity& opp,
    const std::map<std::string, double>& venue_latencies) {
    
    ExecutionResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Calculate synchronized send times to arrive simultaneously
    double max_latency = std::max(
        venue_latencies.at(opp.buy_venue),
        venue_latencies.at(opp.sell_venue)
    );
    
    auto buy_delay = std::chrono::microseconds(
        static_cast<int64_t>(max_latency - venue_latencies.at(opp.buy_venue))
    );
    auto sell_delay = std::chrono::microseconds(
        static_cast<int64_t>(max_latency - venue_latencies.at(opp.sell_venue))
    );
    
    // Launch parallel execution threads
    auto buy_future = std::async(std::launch::async, [&]() {
        std::this_thread::sleep_for(buy_delay);
        return send_aggressive_order(opp.buy_venue, "BUY", 
                                    opp.buy_price, opp.available_size);
    });
    
    auto sell_future = std::async(std::launch::async, [&]() {
        std::this_thread::sleep_for(sell_delay);
        return send_aggressive_order(opp.sell_venue, "SELL",
                                    opp.sell_price, opp.available_size);
    });
    
    // Wait for both executions
    auto buy_result = buy_future.get();
    auto sell_result = sell_future.get();
    
    // Calculate results
    result.success = buy_result.filled && sell_result.filled;
    result.filled_quantity = std::min(buy_result.quantity, sell_result.quantity);
    result.avg_buy_price = buy_result.price;
    result.avg_sell_price = sell_result.price;
    result.realized_pnl = (result.avg_sell_price - result.avg_buy_price) * 
                          result.filled_quantity;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_latency = end_time - start_time;
    
    return result;
}
}
```

### 3. Performance Benchmarking (`perf/latency_bench.cpp`)

```cpp
// perf/latency_bench.cpp
#include <benchmark/benchmark.h>
#include "../hft/xex_arb/venue_aggregator.hpp"

static void BM_ArbitrageScan(benchmark::State& state) {
    xex::CrossExchangeArbitrage<5> arb_engine;
    
    // Setup mock data
    for (auto _ : state) {
        auto opportunities = arb_engine.scan_opportunities();
        benchmark::DoNotOptimize(opportunities);
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_NetProfitCalculation(benchmark::State& state) {
    xex::CrossExchangeArbitrage<5> arb_engine;
    xex::ArbitrageOpportunity opp;
    opp.buy_price = 100.0;
    opp.sell_price = 100.05;
    opp.available_size = 1000;
    
    for (auto _ : state) {
        double profit = arb_engine.calculate_net_profit(opp);
        benchmark::DoNotOptimize(profit);
    }
}

static void BM_SynchronizedExecution(benchmark::State& state) {
    xex::LatencyAwareExecutor executor;
    xex::ArbitrageOpportunity opp;
    std::map<std::string, double> latencies = {
        {"VENUE_A", 50.0},
        {"VENUE_B", 75.0}
    };
    
    for (auto _ : state) {
        auto result = executor.execute_synchronized(opp, latencies);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_ArbitrageScan)->Range(8, 8<<10);
BENCHMARK(BM_NetProfitCalculation)->Range(8, 8<<10);
BENCHMARK(BM_SynchronizedExecution)->Threads(2)->Threads(4);

BENCHMARK_MAIN();
```

### 4. Python Orchestrator (Optional)

```python
# py_orchestrator/arbitrage_manager.py
import asyncio
import numpy as np
from typing import Dict, List
import pandas as pd

class ArbitrageOrchestrator:
    """High-level arbitrage strategy orchestration"""
    
    def __init__(self, venues: List[str], config: Dict):
        self.venues = venues
        self.config = config
        self.positions = {venue: 0 for venue in venues}
        self.pnl_tracker = []
        
    async def monitor_opportunities(self):
        """Continuous monitoring of arbitrage opportunities"""
        while True:
            opportunities = await self.scan_all_venues()
            
            for opp in opportunities:
                if self.should_execute(opp):
                    asyncio.create_task(self.execute_opportunity(opp))
            
            await asyncio.sleep(0.001)  # 1ms scan frequency
    
    def calculate_optimal_size(self, opportunity: Dict) -> float:
        """Kelly criterion for position sizing"""
        win_prob = opportunity['success_probability']
        avg_win = opportunity['expected_profit']
        avg_loss = opportunity['max_loss']
        
        if avg_loss == 0:
            return 0
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * abs(avg_loss)) / avg_win
        
        # Apply Kelly fraction with safety factor
        safety_factor = 0.25  # Use 25% of Kelly
        position_size = kelly_fraction * safety_factor * self.config['capital']
        
        return min(position_size, opportunity['available_size'])
    
    def analyze_performance(self) -> pd.DataFrame:
        """Analyze arbitrage performance"""
        df = pd.DataFrame(self.pnl_tracker)
        
        metrics = {
            'total_opportunities': len(df),
            'success_rate': (df['success'] == True).mean(),
            'total_pnl': df['pnl'].sum(),
            'sharpe_ratio': df['pnl'].mean() / df['pnl'].std() * np.sqrt(252 * 24 * 60 * 60),
            'avg_spread_captured': df['spread_captured'].mean(),
            'avg_execution_time_us': df['execution_time'].mean()
        }
        
        return metrics
```

## Implementation Checklist

### Phase 1: Core Arbitrage Engine
- [ ] Multi-venue feed aggregation
- [ ] Opportunity detection algorithm
- [ ] Net profit calculation with fees
- [ ] Risk assessment framework

### Phase 2: Execution Layer
- [ ] Synchronized order execution
- [ ] Latency compensation
- [ ] Cancel/replace logic
- [ ] Position tracking

### Phase 3: Risk Management
- [ ] Position limits per venue
- [ ] Correlation risk monitoring
- [ ] Inventory imbalance management
- [ ] Circuit breakers

### Phase 4: Performance Optimization
- [ ] Lock-free data structures
- [ ] SIMD optimization for calculations
- [ ] Network optimization
- [ ] Co-location setup

## Configuration

```yaml
# config/xex_arb_config.yaml
venues:
  - name: VENUE_A
    maker_fee: 0.0002
    taker_fee: 0.0005
    maker_rebate: 0.0001
    latency_us: 50
    cancel_success_rate: 0.98
    
  - name: VENUE_B
    maker_fee: 0.0001
    taker_fee: 0.0004
    maker_rebate: 0.0002
    latency_us: 75
    cancel_success_rate: 0.95

execution:
  min_profit_threshold: 0.0001  # 1 bps
  max_position_per_venue: 100000
  use_synchronized_execution: true
  timeout_ms: 100

risk:
  max_total_exposure: 1000000
  max_spread_threshold: 0.01
  min_liquidity: 10000
```

## Expected Performance

Target metrics:
- Opportunity Detection Latency: < 1 microsecond
- Execution Latency: < 100 microseconds
- Success Rate: > 95%
- Average Profit per Trade: 2-5 bps
- Sharpe Ratio: > 5.0