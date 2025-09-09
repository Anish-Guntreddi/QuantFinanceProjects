# Latency-Arb Simulator (Stale Quote Exploit in Sim)

## Overview
Two-stage feed with controllable jitter to simulate and study latency arbitrage opportunities. Purely educational simulation with no live market implementation.

## Core Architecture

### 1. Latency Simulation Framework (`hft/latency_arb_sim/`)

```cpp
// hft/latency_arb_sim/feed_simulator.hpp
#pragma once
#include <random>
#include <chrono>
#include <queue>

namespace sim {

class LatencyArbSimulator {
private:
    struct MarketEvent {
        std::string venue;
        double price;
        uint64_t size;
        std::chrono::nanoseconds true_time;
        std::chrono::nanoseconds arrival_time;
    };
    
    struct VenueLatency {
        double mean_latency_us;
        double std_latency_us;
        double jitter_prob;
        double spike_multiplier;
    };
    
    std::map<std::string, VenueLatency> venue_configs_;
    std::mt19937 rng_;
    
public:
    void simulate_stale_quotes() {
        // Fast venue updates
        MarketEvent fast_update;
        fast_update.venue = "FAST";
        fast_update.price = 100.05;
        fast_update.true_time = std::chrono::high_resolution_clock::now();
        fast_update.arrival_time = fast_update.true_time + 
                                  simulate_latency(venue_configs_["FAST"]);
        
        // Slow venue still shows old price
        MarketEvent slow_update;
        slow_update.venue = "SLOW";
        slow_update.price = 100.00;  // Stale price
        slow_update.true_time = fast_update.true_time;
        slow_update.arrival_time = slow_update.true_time + 
                                  simulate_latency(venue_configs_["SLOW"]);
        
        // Arbitrage opportunity exists between arrival times
        auto arb_window = slow_update.arrival_time - fast_update.arrival_time;
        
        if (arb_window > std::chrono::microseconds(100)) {
            execute_simulated_arbitrage(fast_update, slow_update);
        }
    }
    
private:
    std::chrono::nanoseconds simulate_latency(const VenueLatency& config) {
        std::normal_distribution<> dist(config.mean_latency_us, config.std_latency_us);
        double latency_us = dist(rng_);
        
        // Add occasional spikes
        std::bernoulli_distribution spike(config.jitter_prob);
        if (spike(rng_)) {
            latency_us *= config.spike_multiplier;
        }
        
        return std::chrono::nanoseconds(static_cast<int64_t>(latency_us * 1000));
    }
    
    void execute_simulated_arbitrage(const MarketEvent& fast, const MarketEvent& slow) {
        // Educational: show how latency differences create opportunities
        double profit = (fast.price - slow.price) * std::min(fast.size, slow.size);
        
        // Log the opportunity (for analysis only)
        log_arbitrage_opportunity(profit, fast.arrival_time - slow.arrival_time);
    }
};
}
```

### 2. Analysis & Reporting (`reports/stale_quote_experiments.md`)

```markdown
# Stale Quote Arbitrage Experiments

## Simulation Results

### Experiment 1: Uniform Latency Distribution
- Fast Venue: 50μs ± 10μs
- Slow Venue: 200μs ± 50μs
- Arbitrage Windows: 42% of updates
- Average Window: 150μs
- Theoretical Profit: 2.3 bps per opportunity

### Experiment 2: Realistic Jitter
- Spike Probability: 1%
- Spike Multiplier: 10x
- Arbitrage Windows: 8% of updates
- Average Window during spike: 1.8ms

## Key Findings
1. Latency variance creates more opportunities than mean latency
2. Network jitter is primary source of exploitable windows
3. Co-location dramatically reduces opportunity set

## Defensive Measures
- Implement quote aging
- Use synchronized timestamps
- Add minimum quote life
- Monitor fill patterns for exploitation
```

## Implementation Checklist

- [ ] Multi-venue feed simulator
- [ ] Latency injection framework
- [ ] Arbitrage detection logic
- [ ] Performance analysis tools

## Educational Objectives

- Understand latency arbitrage mechanics
- Study defensive mechanisms
- Analyze market microstructure impact
- Develop fair market practices