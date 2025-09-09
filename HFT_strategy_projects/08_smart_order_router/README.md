# Smart Order Router (SOR) with Rebate-Aware Cost Model

## Overview
C++ implementation of Smart Order Router that minimizes expected cost including impact, fees, rebates, and fill probability. Supports dark and visible venues in simulation.

## Core Architecture

### 1. Cost Model Implementation (`hft/sor/cost_model.hpp`)

```cpp
// hft/sor/cost_model.hpp
#pragma once
#include <vector>
#include <algorithm>

namespace sor {

class RebateAwareCostModel {
private:
    struct VenueCost {
        std::string venue;
        double expected_price;
        double maker_fee;
        double taker_fee;
        double maker_rebate;
        double taker_rebate;
        double market_impact;
        double fill_probability;
        double opportunity_cost;
    };
    
public:
    struct RoutingDecision {
        std::string venue;
        uint64_t size;
        bool is_aggressive;
        double expected_cost;
    };
    
    std::vector<RoutingDecision> optimize_routing(
        uint64_t total_size,
        const std::vector<VenueSnapshot>& venues) {
        
        std::vector<RoutingDecision> routing_plan;
        
        // Calculate cost for each venue
        std::vector<VenueCost> venue_costs;
        for (const auto& venue : venues) {
            VenueCost cost = calculate_venue_cost(venue, total_size);
            venue_costs.push_back(cost);
        }
        
        // Sort by expected cost
        std::sort(venue_costs.begin(), venue_costs.end(),
                 [](const auto& a, const auto& b) {
                     return a.expected_cost < b.expected_cost;
                 });
        
        // Allocate orders to minimize total cost
        uint64_t remaining = total_size;
        for (const auto& venue : venue_costs) {
            if (remaining == 0) break;
            
            uint64_t venue_size = calculate_optimal_size(venue, remaining);
            if (venue_size > 0) {
                RoutingDecision decision;
                decision.venue = venue.venue;
                decision.size = venue_size;
                decision.is_aggressive = should_be_aggressive(venue);
                decision.expected_cost = venue.expected_cost * venue_size;
                routing_plan.push_back(decision);
                
                remaining -= venue_size;
            }
        }
        
        return routing_plan;
    }
    
private:
    double calculate_expected_cost(const VenueCost& venue, uint64_t size) {
        double price_cost = venue.expected_price * size;
        
        // Fee/rebate calculation
        double fee_cost = venue.is_aggressive ? 
                         venue.taker_fee * price_cost :
                         venue.maker_fee * price_cost;
        
        double rebate = venue.is_aggressive ?
                       venue.taker_rebate * price_cost :
                       venue.maker_rebate * price_cost;
        
        // Market impact
        double impact_cost = venue.market_impact * std::sqrt(size);
        
        // Opportunity cost (unfilled probability)
        double opp_cost = (1 - venue.fill_probability) * venue.opportunity_cost;
        
        return price_cost + fee_cost - rebate + impact_cost + opp_cost;
    }
};

// Dark pool routing
class DarkPoolRouter {
private:
    struct DarkVenue {
        std::string name;
        double min_size;
        double participation_rate;
        double information_leakage;
    };
    
public:
    std::vector<RoutingDecision> route_to_dark(
        uint64_t size,
        const std::vector<DarkVenue>& dark_venues) {
        
        std::vector<RoutingDecision> dark_orders;
        
        for (const auto& venue : dark_venues) {
            if (size >= venue.min_size) {
                RoutingDecision decision;
                decision.venue = venue.name;
                decision.size = size * venue.participation_rate;
                decision.is_aggressive = false;  // Dark pools are passive
                
                // Adjust for information leakage risk
                decision.expected_cost = calculate_leakage_cost(
                    decision.size, venue.information_leakage
                );
                
                dark_orders.push_back(decision);
            }
        }
        
        return dark_orders;
    }
};
}
```

### 2. Testing Framework (`tests/test_cost_model.cpp`)

```cpp
// tests/test_cost_model.cpp
#include <gtest/gtest.h>
#include "../hft/sor/cost_model.hpp"

TEST(CostModelTest, RebateOptimization) {
    sor::RebateAwareCostModel model;
    
    std::vector<VenueSnapshot> venues = {
        {"VENUE_A", 100.00, 1000, 0.0002, -0.0001},  // Maker rebate
        {"VENUE_B", 100.01, 2000, 0.0003, 0.0},      // No rebate
        {"VENUE_C", 99.99, 500, 0.0001, -0.0002}     // High rebate
    };
    
    auto routing = model.optimize_routing(1500, venues);
    
    // Should prioritize VENUE_C for rebate despite worse price
    EXPECT_EQ(routing[0].venue, "VENUE_C");
}

TEST(DarkPoolTest, MinimumSizeRequirement) {
    sor::DarkPoolRouter dark_router;
    
    std::vector<DarkVenue> dark_venues = {
        {"DARK_A", 1000, 0.3, 0.001},
        {"DARK_B", 500, 0.5, 0.002}
    };
    
    // Test with size below minimum
    auto routing = dark_router.route_to_dark(400, dark_venues);
    EXPECT_EQ(routing.size(), 0);  // No routing due to min size
    
    // Test with sufficient size
    routing = dark_router.route_to_dark(2000, dark_venues);
    EXPECT_GT(routing.size(), 0);
}
```

## Implementation Checklist

### Phase 1: Core Cost Model
- [ ] Fee/rebate calculation
- [ ] Market impact estimation
- [ ] Fill probability modeling
- [ ] Multi-venue optimization

### Phase 2: Dark Pool Integration
- [ ] Dark venue discovery
- [ ] Information leakage model
- [ ] Participation rate optimization
- [ ] Minimum size handling

### Phase 3: Advanced Features
- [ ] Time-weighted routing
- [ ] Adaptive learning
- [ ] Venue performance tracking
- [ ] Real-time cost updates

## Configuration

```yaml
# config/sor_config.yaml
venues:
  visible:
    - name: NYSE
      maker_fee: 0.0002
      taker_fee: 0.0003
      maker_rebate: -0.0001
      taker_rebate: 0
      
  dark:
    - name: SIGMA_X
      min_size: 1000
      participation_rate: 0.3
      leakage_factor: 0.001

routing:
  max_venues_per_order: 5
  min_fill_probability: 0.5
  max_market_impact: 0.01
```

## Expected Performance

- Routing Decision Latency: < 10 microseconds
- Cost Reduction vs Naive: > 15%
- Fill Rate: > 95%
- Dark Pool Participation: 20-40%