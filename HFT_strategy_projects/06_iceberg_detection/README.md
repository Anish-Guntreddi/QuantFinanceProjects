# Iceberg Detection & Liquidity-Seeking "Sniper"

## Overview
Hybrid system to detect hidden liquidity via partial fills & refill patterns, routing IOC/pegged orders with minimal signaling.

## Core Architecture

### 1. Iceberg Detection Engine (`hft/iceberg_detector.cpp`)

```cpp
// hft/iceberg_detector.cpp
#include <unordered_map>
#include <deque>
#include <algorithm>

namespace hft {

class IcebergDetector {
private:
    struct OrderPattern {
        double price_level;
        std::deque<uint64_t> fill_sizes;
        std::deque<uint64_t> refill_times_ns;
        uint64_t total_volume_detected;
        double confidence_score;
    };
    
    std::unordered_map<double, OrderPattern> suspected_icebergs_;
    
public:
    struct IcebergSignal {
        double price;
        uint64_t estimated_hidden_size;
        double detection_confidence;
        uint64_t refresh_rate_ns;
    };
    
    IcebergSignal detect_iceberg(const OrderBookUpdate& update) {
        IcebergSignal signal;
        
        // Pattern 1: Consistent refills at same price
        if (detect_refill_pattern(update)) {
            signal = analyze_refill_behavior(update.price);
        }
        
        // Pattern 2: Partial fill followed by immediate replenishment
        if (detect_partial_fill_pattern(update)) {
            signal = analyze_partial_fills(update.price);
        }
        
        return signal;
    }
    
private:
    bool detect_refill_pattern(const OrderBookUpdate& update) {
        auto& pattern = suspected_icebergs_[update.price];
        
        // Check if volume increased after decrease
        if (update.volume_delta > 0 && pattern.fill_sizes.size() > 0) {
            pattern.refill_times_ns.push_back(update.timestamp_ns);
            
            // Calculate refill frequency
            if (pattern.refill_times_ns.size() > 3) {
                double avg_refill_time = calculate_avg_refill_time(pattern);
                if (avg_refill_time < 1000000000) {  // Less than 1 second
                    return true;
                }
            }
        }
        
        return false;
    }
};
}
```

### 2. Liquidity Seeking Router (`router/liquidity_seek.cpp`)

```cpp
// router/liquidity_seek.cpp
namespace router {

class LiquiditySeeker {
private:
    struct SnipeOrder {
        enum Type { IOC, PEGGED, HIDDEN, MIDPOINT };
        Type type;
        double price;
        uint64_t size;
        uint64_t max_show_size;
    };
    
public:
    std::vector<SnipeOrder> route_to_hidden_liquidity(
        const IcebergSignal& signal,
        uint64_t desired_size) {
        
        std::vector<SnipeOrder> orders;
        
        // Strategy 1: IOC probe orders
        if (signal.detection_confidence > 0.7) {
            SnipeOrder ioc;
            ioc.type = SnipeOrder::IOC;
            ioc.price = signal.price;
            ioc.size = std::min(desired_size, signal.estimated_hidden_size / 10);
            orders.push_back(ioc);
        }
        
        // Strategy 2: Pegged orders to track the iceberg
        if (signal.refresh_rate_ns < 100000000) {  // Fast refresh
            SnipeOrder pegged;
            pegged.type = SnipeOrder::PEGGED;
            pegged.price = signal.price;
            pegged.size = desired_size;
            pegged.max_show_size = 100;  // Minimal signaling
            orders.push_back(pegged);
        }
        
        return orders;
    }
};
}
```

## Implementation Checklist

- [ ] Pattern recognition algorithms
- [ ] Statistical iceberg detection
- [ ] Liquidity routing strategies
- [ ] Minimal signaling execution

## Expected Performance

- Detection Accuracy: > 75%
- False Positive Rate: < 10%
- Execution with minimal market impact
- Hidden liquidity capture rate: > 60%