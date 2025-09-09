# Limit Order Book Simulator

## Project Overview
A high-performance discrete-event limit order book (LOB) engine implemented in C++ with Python bindings, supporting realistic order matching, multiple order types, and configurable market dynamics including Poisson/Hawkes arrival processes.

## Implementation Guide

### Phase 1: Project Setup & Architecture

#### 1.1 Environment Setup
```bash
# Create build environment
mkdir -p cpp/lob python/lob bench tests
mkdir -p build/{debug,release}

# Install dependencies
# Ubuntu/Debian
sudo apt-get install cmake g++ python3-dev pybind11-dev libbenchmark-dev

# macOS
brew install cmake pybind11 google-benchmark

# Python dependencies
pip install pybind11 numpy pandas matplotlib pytest pytest-benchmark
```

#### 1.2 Project Structure
```
01_limit_order_book_simulator/
├── cpp/
│   └── lob/
│       ├── include/
│       │   ├── order.hpp              # Order structure
│       │   ├── order_book.hpp         # Core LOB engine
│       │   ├── price_level.hpp        # Price level management
│       │   ├── matching_engine.hpp    # Matching logic
│       │   ├── event_queue.hpp        # Event management
│       │   └── types.hpp              # Type definitions
│       ├── src/
│       │   ├── order.cpp
│       │   ├── order_book.cpp
│       │   ├── price_level.cpp
│       │   ├── matching_engine.cpp
│       │   └── event_queue.cpp
│       ├── arrival_models/
│       │   ├── poisson.hpp            # Poisson arrivals
│       │   ├── hawkes.hpp             # Hawkes process
│       │   └── empirical.hpp          # Empirical distribution
│       └── utils/
│           ├── memory_pool.hpp        # Memory management
│           ├── ring_buffer.hpp        # Lock-free structures
│           └── timestamp.hpp          # High-res timing
├── python/
│   └── lob/
│       ├── __init__.py
│       ├── bindings.cpp               # Pybind11 bindings
│       ├── simulator.py               # Python wrapper
│       ├── visualization.py           # LOB visualization
│       └── analysis.py                # Market metrics
├── bench/
│   ├── benchmark_matching.cpp         # Matching performance
│   ├── benchmark_insertion.cpp        # Order insertion
│   ├── benchmark_cancellation.cpp     # Cancellation speed
│   └── benchmark_snapshot.cpp         # Snapshot generation
├── tests/
│   ├── cpp/
│   │   ├── test_order_book.cpp
│   │   ├── test_matching.cpp
│   │   └── test_price_level.cpp
│   └── python/
│       ├── test_bindings.py
│       └── test_simulator.py
├── CMakeLists.txt
├── setup.py                           # Python package setup
├── requirements.txt
└── README.md
```

### Phase 2: Core C++ Implementation

#### 2.1 Type Definitions (cpp/lob/include/types.hpp)
```cpp
#pragma once

#include <cstdint>
#include <chrono>
#include <functional>

namespace lob {

using OrderId = uint64_t;
using Price = int64_t;      // Price in ticks (fixed point)
using Quantity = uint64_t;
using Timestamp = std::chrono::nanoseconds::rep;

enum class Side {
    BUY,
    SELL
};

enum class OrderType {
    LIMIT,
    MARKET,
    STOP,
    STOP_LIMIT,
    ICEBERG,
    PEGGED
};

enum class TimeInForce {
    DAY,
    GTC,    // Good Till Cancelled
    IOC,    // Immediate or Cancel
    FOK,    // Fill or Kill
    GTD     // Good Till Date
};

enum class OrderStatus {
    NEW,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED,
    EXPIRED
};

enum class EventType {
    ORDER_ACCEPTED,
    ORDER_REJECTED,
    ORDER_FILLED,
    ORDER_PARTIALLY_FILLED,
    ORDER_CANCELLED,
    ORDER_EXPIRED,
    TRADE,
    BOOK_UPDATE
};

// Configuration
struct LOBConfig {
    Price tick_size = 1;
    Quantity min_order_size = 1;
    Quantity max_order_size = 1000000;
    bool allow_iceberg = true;
    bool allow_hidden = false;
    uint32_t max_price_levels = 10000;
    uint32_t latency_ns = 0;  // Simulated latency
};

} // namespace lob
```

#### 2.2 Order Structure (cpp/lob/include/order.hpp)
```cpp
#pragma once

#include "types.hpp"
#include <memory>
#include <string>

namespace lob {

class Order {
public:
    Order(OrderId id, Side side, Price price, Quantity quantity,
          OrderType type = OrderType::LIMIT,
          TimeInForce tif = TimeInForce::GTC)
        : id_(id), side_(side), price_(price), quantity_(quantity),
          remaining_quantity_(quantity), type_(type), tif_(tif),
          timestamp_(get_timestamp()), status_(OrderStatus::NEW) {}

    // Getters
    OrderId id() const { return id_; }
    Side side() const { return side_; }
    Price price() const { return price_; }
    Quantity quantity() const { return quantity_; }
    Quantity remaining_quantity() const { return remaining_quantity_; }
    Quantity filled_quantity() const { return quantity_ - remaining_quantity_; }
    OrderType type() const { return type_; }
    TimeInForce tif() const { return tif_; }
    Timestamp timestamp() const { return timestamp_; }
    OrderStatus status() const { return status_; }

    // Modifiers
    void fill(Quantity qty) {
        remaining_quantity_ -= std::min(qty, remaining_quantity_);
        if (remaining_quantity_ == 0) {
            status_ = OrderStatus::FILLED;
        } else {
            status_ = OrderStatus::PARTIALLY_FILLED;
        }
    }

    void cancel() {
        status_ = OrderStatus::CANCELLED;
    }

    bool is_marketable(Price best_opposite_price) const {
        if (type_ == OrderType::MARKET) return true;
        
        if (side_ == Side::BUY) {
            return price_ >= best_opposite_price;
        } else {
            return price_ <= best_opposite_price;
        }
    }

    // For iceberg orders
    void set_display_quantity(Quantity qty) {
        display_quantity_ = qty;
    }

    Quantity display_quantity() const {
        return (type_ == OrderType::ICEBERG) ? display_quantity_ : remaining_quantity_;
    }

private:
    static Timestamp get_timestamp() {
        return std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

    OrderId id_;
    Side side_;
    Price price_;
    Quantity quantity_;
    Quantity remaining_quantity_;
    Quantity display_quantity_ = 0;
    OrderType type_;
    TimeInForce tif_;
    Timestamp timestamp_;
    OrderStatus status_;
    
    // Additional fields for advanced features
    Price stop_price_ = 0;
    OrderId parent_order_id_ = 0;
    std::string trader_id_;
};

using OrderPtr = std::shared_ptr<Order>;

} // namespace lob
```

#### 2.3 Price Level Management (cpp/lob/include/price_level.hpp)
```cpp
#pragma once

#include "order.hpp"
#include <deque>
#include <unordered_map>

namespace lob {

class PriceLevel {
public:
    explicit PriceLevel(Price price) : price_(price), total_quantity_(0) {}

    Price price() const { return price_; }
    Quantity total_quantity() const { return total_quantity_; }
    size_t order_count() const { return orders_.size(); }
    bool empty() const { return orders_.empty(); }

    // Add order (price-time priority)
    void add_order(OrderPtr order) {
        orders_.push_back(order);
        order_map_[order->id()] = order;
        total_quantity_ += order->display_quantity();
    }

    // Remove order
    bool remove_order(OrderId order_id) {
        auto it = order_map_.find(order_id);
        if (it == order_map_.end()) return false;

        auto order = it->second;
        order_map_.erase(it);

        // Find and remove from deque
        auto order_it = std::find_if(orders_.begin(), orders_.end(),
            [order_id](const OrderPtr& o) { return o->id() == order_id; });
        
        if (order_it != orders_.end()) {
            total_quantity_ -= (*order_it)->display_quantity();
            orders_.erase(order_it);
            return true;
        }
        return false;
    }

    // Match orders (returns filled quantity)
    Quantity match(Quantity quantity, std::vector<std::pair<OrderPtr, Quantity>>& fills) {
        Quantity remaining = quantity;
        
        auto it = orders_.begin();
        while (it != orders_.end() && remaining > 0) {
            auto& order = *it;
            Quantity fill_qty = std::min(remaining, order->display_quantity());
            
            if (fill_qty > 0) {
                fills.push_back({order, fill_qty});
                order->fill(fill_qty);
                remaining -= fill_qty;
                total_quantity_ -= fill_qty;
                
                if (order->remaining_quantity() == 0) {
                    order_map_.erase(order->id());
                    it = orders_.erase(it);
                } else {
                    // Handle iceberg order reload
                    if (order->type() == OrderType::ICEBERG) {
                        reload_iceberg(order);
                    }
                    ++it;
                }
            } else {
                ++it;
            }
        }
        
        return quantity - remaining;
    }

    // Get top order
    OrderPtr top() const {
        return orders_.empty() ? nullptr : orders_.front();
    }

private:
    void reload_iceberg(OrderPtr order) {
        // Reload display quantity for iceberg order
        Quantity reload = std::min(order->display_quantity(), 
                                   order->remaining_quantity());
        order->set_display_quantity(reload);
    }

    Price price_;
    Quantity total_quantity_;
    std::deque<OrderPtr> orders_;  // FIFO queue for time priority
    std::unordered_map<OrderId, OrderPtr> order_map_;  // Fast lookup
};

} // namespace lob
```

#### 2.4 Order Book Core (cpp/lob/include/order_book.hpp)
```cpp
#pragma once

#include "price_level.hpp"
#include "types.hpp"
#include <map>
#include <functional>
#include <mutex>
#include <atomic>

namespace lob {

struct Trade {
    OrderId buy_order_id;
    OrderId sell_order_id;
    Price price;
    Quantity quantity;
    Timestamp timestamp;
};

struct BookSnapshot {
    struct Level {
        Price price;
        Quantity quantity;
        size_t order_count;
    };
    
    std::vector<Level> bids;
    std::vector<Level> asks;
    Timestamp timestamp;
};

class OrderBook {
public:
    using TradeCallback = std::function<void(const Trade&)>;
    using EventCallback = std::function<void(EventType, const Order&)>;

    explicit OrderBook(const LOBConfig& config = LOBConfig())
        : config_(config), next_order_id_(1) {}

    // Order management
    OrderId add_order(Side side, Price price, Quantity quantity,
                      OrderType type = OrderType::LIMIT,
                      TimeInForce tif = TimeInForce::GTC);
    
    bool cancel_order(OrderId order_id);
    bool modify_order(OrderId order_id, Price new_price, Quantity new_quantity);
    
    // Market data
    Price best_bid() const;
    Price best_ask() const;
    Price mid_price() const;
    Quantity bid_volume() const;
    Quantity ask_volume() const;
    
    BookSnapshot get_snapshot(size_t depth = 10) const;
    
    // Callbacks
    void set_trade_callback(TradeCallback callback) {
        trade_callback_ = callback;
    }
    
    void set_event_callback(EventCallback callback) {
        event_callback_ = callback;
    }

    // Statistics
    struct Stats {
        uint64_t total_orders = 0;
        uint64_t total_cancellations = 0;
        uint64_t total_trades = 0;
        uint64_t total_volume = 0;
        double avg_spread = 0;
        double fill_rate = 0;
    };
    
    Stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats(); }

private:
    void process_limit_order(OrderPtr order);
    void process_market_order(OrderPtr order);
    void match_order(OrderPtr order);
    void execute_trade(OrderPtr buy_order, OrderPtr sell_order, 
                       Price price, Quantity quantity);
    
    void add_to_book(OrderPtr order);
    void remove_from_book(OrderId order_id);
    
    Price apply_tick_size(Price price) const {
        return (price / config_.tick_size) * config_.tick_size;
    }

    // Latency simulation
    void simulate_latency() const {
        if (config_.latency_ns > 0) {
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(config_.latency_ns));
        }
    }

private:
    LOBConfig config_;
    std::atomic<OrderId> next_order_id_;
    
    // Price levels - using map for ordered traversal
    std::map<Price, PriceLevel, std::greater<Price>> bid_levels_;  // Descending
    std::map<Price, PriceLevel> ask_levels_;  // Ascending
    
    // Order lookup
    std::unordered_map<OrderId, OrderPtr> orders_;
    std::unordered_map<OrderId, Side> order_sides_;
    
    // Callbacks
    TradeCallback trade_callback_;
    EventCallback event_callback_;
    
    // Statistics
    mutable Stats stats_;
    
    // Thread safety (optional)
    mutable std::mutex book_mutex_;
};

// Implementation of key methods
inline Price OrderBook::best_bid() const {
    std::lock_guard<std::mutex> lock(book_mutex_);
    return bid_levels_.empty() ? 0 : bid_levels_.begin()->first;
}

inline Price OrderBook::best_ask() const {
    std::lock_guard<std::mutex> lock(book_mutex_);
    return ask_levels_.empty() ? 0 : ask_levels_.begin()->first;
}

inline Price OrderBook::mid_price() const {
    Price bid = best_bid();
    Price ask = best_ask();
    return (bid + ask) / 2;
}

} // namespace lob
```

#### 2.5 Order Book Implementation (cpp/lob/src/order_book.cpp)
```cpp
#include "../include/order_book.hpp"
#include <algorithm>
#include <chrono>

namespace lob {

OrderId OrderBook::add_order(Side side, Price price, Quantity quantity,
                             OrderType type, TimeInForce tif) {
    simulate_latency();
    
    std::lock_guard<std::mutex> lock(book_mutex_);
    
    // Validate order
    if (quantity < config_.min_order_size || quantity > config_.max_order_size) {
        if (event_callback_) {
            Order rejected_order(0, side, price, quantity, type, tif);
            event_callback_(EventType::ORDER_REJECTED, rejected_order);
        }
        return 0;
    }
    
    // Apply tick size
    price = apply_tick_size(price);
    
    // Create order
    OrderId order_id = next_order_id_++;
    auto order = std::make_shared<Order>(order_id, side, price, quantity, type, tif);
    
    // Store order
    orders_[order_id] = order;
    order_sides_[order_id] = side;
    
    // Update statistics
    stats_.total_orders++;
    
    // Process based on order type
    switch (type) {
        case OrderType::MARKET:
            process_market_order(order);
            break;
        case OrderType::LIMIT:
            process_limit_order(order);
            break;
        default:
            // Handle other order types
            break;
    }
    
    if (event_callback_) {
        event_callback_(EventType::ORDER_ACCEPTED, *order);
    }
    
    return order_id;
}

void OrderBook::process_limit_order(OrderPtr order) {
    // Try to match immediately
    match_order(order);
    
    // Add remaining quantity to book
    if (order->remaining_quantity() > 0 && 
        order->tif() != TimeInForce::IOC &&
        order->tif() != TimeInForce::FOK) {
        add_to_book(order);
    } else if (order->remaining_quantity() > 0) {
        // Cancel unfilled IOC/FOK orders
        order->cancel();
        if (event_callback_) {
            event_callback_(EventType::ORDER_CANCELLED, *order);
        }
    }
}

void OrderBook::process_market_order(OrderPtr order) {
    match_order(order);
    
    // Cancel any remaining quantity (market orders don't rest in book)
    if (order->remaining_quantity() > 0) {
        order->cancel();
        if (event_callback_) {
            event_callback_(EventType::ORDER_CANCELLED, *order);
        }
    }
}

void OrderBook::match_order(OrderPtr order) {
    std::vector<std::pair<OrderPtr, Quantity>> fills;
    
    if (order->side() == Side::BUY) {
        // Match against asks
        auto it = ask_levels_.begin();
        while (it != ask_levels_.end() && order->remaining_quantity() > 0) {
            if (order->is_marketable(it->first)) {
                it->second.match(order->remaining_quantity(), fills);
                
                // Execute trades
                for (const auto& [counter_order, fill_qty] : fills) {
                    execute_trade(order, counter_order, it->first, fill_qty);
                }
                fills.clear();
                
                // Remove empty level
                if (it->second.empty()) {
                    it = ask_levels_.erase(it);
                } else {
                    ++it;
                }
            } else {
                break;
            }
        }
    } else {
        // Match against bids
        auto it = bid_levels_.begin();
        while (it != bid_levels_.end() && order->remaining_quantity() > 0) {
            if (order->is_marketable(it->first)) {
                it->second.match(order->remaining_quantity(), fills);
                
                // Execute trades
                for (const auto& [counter_order, fill_qty] : fills) {
                    execute_trade(counter_order, order, it->first, fill_qty);
                }
                fills.clear();
                
                // Remove empty level
                if (it->second.empty()) {
                    it = bid_levels_.erase(it);
                } else {
                    ++it;
                }
            } else {
                break;
            }
        }
    }
}

void OrderBook::execute_trade(OrderPtr buy_order, OrderPtr sell_order,
                              Price price, Quantity quantity) {
    // Fill orders
    buy_order->fill(quantity);
    sell_order->fill(quantity);
    
    // Create trade
    Trade trade{
        buy_order->id(),
        sell_order->id(),
        price,
        quantity,
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    };
    
    // Update statistics
    stats_.total_trades++;
    stats_.total_volume += quantity;
    
    // Notify callbacks
    if (trade_callback_) {
        trade_callback_(trade);
    }
    
    if (event_callback_) {
        event_callback_(EventType::TRADE, *buy_order);
        event_callback_(EventType::TRADE, *sell_order);
    }
}

void OrderBook::add_to_book(OrderPtr order) {
    if (order->side() == Side::BUY) {
        auto& level = bid_levels_[order->price()];
        if (level.price() == 0) {
            level = PriceLevel(order->price());
        }
        level.add_order(order);
    } else {
        auto& level = ask_levels_[order->price()];
        if (level.price() == 0) {
            level = PriceLevel(order->price());
        }
        level.add_order(order);
    }
    
    if (event_callback_) {
        event_callback_(EventType::BOOK_UPDATE, *order);
    }
}

bool OrderBook::cancel_order(OrderId order_id) {
    simulate_latency();
    
    std::lock_guard<std::mutex> lock(book_mutex_);
    
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    auto order = it->second;
    auto side_it = order_sides_.find(order_id);
    if (side_it == order_sides_.end()) {
        return false;
    }
    
    Side side = side_it->second;
    
    // Remove from appropriate book side
    if (side == Side::BUY) {
        auto level_it = bid_levels_.find(order->price());
        if (level_it != bid_levels_.end()) {
            level_it->second.remove_order(order_id);
            if (level_it->second.empty()) {
                bid_levels_.erase(level_it);
            }
        }
    } else {
        auto level_it = ask_levels_.find(order->price());
        if (level_it != ask_levels_.end()) {
            level_it->second.remove_order(order_id);
            if (level_it->second.empty()) {
                ask_levels_.erase(level_it);
            }
        }
    }
    
    // Update order status
    order->cancel();
    
    // Update statistics
    stats_.total_cancellations++;
    
    // Notify callbacks
    if (event_callback_) {
        event_callback_(EventType::ORDER_CANCELLED, *order);
    }
    
    // Clean up
    orders_.erase(it);
    order_sides_.erase(side_it);
    
    return true;
}

BookSnapshot OrderBook::get_snapshot(size_t depth) const {
    std::lock_guard<std::mutex> lock(book_mutex_);
    
    BookSnapshot snapshot;
    snapshot.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Collect bid levels
    size_t count = 0;
    for (const auto& [price, level] : bid_levels_) {
        if (count >= depth) break;
        snapshot.bids.push_back({price, level.total_quantity(), level.order_count()});
        count++;
    }
    
    // Collect ask levels
    count = 0;
    for (const auto& [price, level] : ask_levels_) {
        if (count >= depth) break;
        snapshot.asks.push_back({price, level.total_quantity(), level.order_count()});
        count++;
    }
    
    return snapshot;
}

} // namespace lob
```

### Phase 3: Arrival Models

#### 3.1 Poisson Process (cpp/lob/arrival_models/poisson.hpp)
```cpp
#pragma once

#include <random>
#include <chrono>
#include "../include/types.hpp"

namespace lob {

class PoissonArrivalModel {
public:
    PoissonArrivalModel(double lambda_buy, double lambda_sell,
                       double lambda_cancel = 0.5)
        : lambda_buy_(lambda_buy), lambda_sell_(lambda_sell),
          lambda_cancel_(lambda_cancel),
          gen_(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    
    struct ArrivalEvent {
        enum Type { BUY_ORDER, SELL_ORDER, CANCEL_ORDER };
        Type type;
        double time;
        Price price;
        Quantity quantity;
    };
    
    ArrivalEvent next_arrival() {
        // Total rate
        double total_lambda = lambda_buy_ + lambda_sell_ + lambda_cancel_;
        
        // Time to next event (exponential distribution)
        std::exponential_distribution<> time_dist(total_lambda);
        double time = time_dist(gen_);
        
        // Determine event type
        std::uniform_real_distribution<> type_dist(0, total_lambda);
        double u = type_dist(gen_);
        
        ArrivalEvent event;
        event.time = time;
        
        if (u < lambda_buy_) {
            event.type = ArrivalEvent::BUY_ORDER;
            event.price = generate_price(Side::BUY);
            event.quantity = generate_quantity();
        } else if (u < lambda_buy_ + lambda_sell_) {
            event.type = ArrivalEvent::SELL_ORDER;
            event.price = generate_price(Side::SELL);
            event.quantity = generate_quantity();
        } else {
            event.type = ArrivalEvent::CANCEL_ORDER;
            event.price = 0;
            event.quantity = 0;
        }
        
        return event;
    }
    
    void set_reference_price(Price price) {
        reference_price_ = price;
    }

private:
    Price generate_price(Side side) {
        // Generate price around reference with power-law distribution
        std::uniform_real_distribution<> u_dist(0, 1);
        double u = u_dist(gen_);
        
        // Power law for distance from mid
        double alpha = 1.5;  // Power law exponent
        double max_distance = 100;  // Max ticks from mid
        double distance = max_distance * std::pow(u, 1.0 / alpha);
        
        if (side == Side::BUY) {
            return reference_price_ - static_cast<Price>(distance);
        } else {
            return reference_price_ + static_cast<Price>(distance);
        }
    }
    
    Quantity generate_quantity() {
        // Log-normal distribution for order sizes
        std::lognormal_distribution<> qty_dist(4.0, 1.5);
        return static_cast<Quantity>(qty_dist(gen_));
    }

private:
    double lambda_buy_;
    double lambda_sell_;
    double lambda_cancel_;
    Price reference_price_ = 10000;  // Default reference price
    std::mt19937 gen_;
};

} // namespace lob
```

#### 3.2 Hawkes Process (cpp/lob/arrival_models/hawkes.hpp)
```cpp
#pragma once

#include "poisson.hpp"
#include <deque>

namespace lob {

class HawkesArrivalModel {
public:
    HawkesArrivalModel(double base_intensity, double alpha, double beta)
        : base_intensity_(base_intensity), alpha_(alpha), beta_(beta),
          gen_(std::chrono::steady_clock::now().time_since_epoch().count()),
          current_time_(0) {}
    
    struct Event {
        double time;
        Side side;
        Price price;
        Quantity quantity;
        bool is_aggressive;  // Marketable order
    };
    
    Event next_arrival() {
        // Calculate current intensity
        double intensity = calculate_intensity(current_time_);
        
        // Generate next arrival time
        double u = std::uniform_real_distribution<>(0, 1)(gen_);
        double tau = -std::log(u) / intensity;
        
        current_time_ += tau;
        
        // Store event for self-excitation
        events_.push_back(current_time_);
        
        // Clean old events (beyond influence window)
        while (!events_.empty() && events_.front() < current_time_ - 10.0 / beta_) {
            events_.pop_front();
        }
        
        // Generate event details
        Event event;
        event.time = current_time_;
        
        // Aggressive orders more likely after recent activity
        double aggression_prob = 0.3 + 0.4 * (intensity / base_intensity_ - 1.0);
        aggression_prob = std::min(0.8, std::max(0.1, aggression_prob));
        
        std::bernoulli_distribution aggressive_dist(aggression_prob);
        event.is_aggressive = aggressive_dist(gen_);
        
        // Side and price generation
        std::bernoulli_distribution side_dist(0.5);
        event.side = side_dist(gen_) ? Side::BUY : Side::SELL;
        
        if (event.is_aggressive) {
            // Market or aggressive limit order
            event.price = generate_aggressive_price(event.side);
        } else {
            // Passive limit order
            event.price = generate_passive_price(event.side);
        }
        
        event.quantity = generate_quantity(event.is_aggressive);
        
        return event;
    }

private:
    double calculate_intensity(double t) {
        double intensity = base_intensity_;
        
        // Add self-excitation from past events
        for (double event_time : events_) {
            intensity += alpha_ * std::exp(-beta_ * (t - event_time));
        }
        
        return intensity;
    }
    
    Price generate_aggressive_price(Side side) {
        // Price at or through the spread
        std::normal_distribution<> offset_dist(0, 2);
        Price offset = static_cast<Price>(std::abs(offset_dist(gen_)));
        
        if (side == Side::BUY) {
            return reference_price_ + offset;  // At or above ask
        } else {
            return reference_price_ - offset;  // At or below bid
        }
    }
    
    Price generate_passive_price(Side side) {
        // Price in the book
        std::exponential_distribution<> depth_dist(0.1);
        Price depth = static_cast<Price>(depth_dist(gen_) * 10);
        
        if (side == Side::BUY) {
            return reference_price_ - depth - 5;  // Below bid
        } else {
            return reference_price_ + depth + 5;  // Above ask
        }
    }
    
    Quantity generate_quantity(bool is_aggressive) {
        if (is_aggressive) {
            // Smaller sizes for aggressive orders
            std::lognormal_distribution<> qty_dist(3.5, 1.2);
            return static_cast<Quantity>(qty_dist(gen_));
        } else {
            // Larger sizes for passive orders
            std::lognormal_distribution<> qty_dist(4.5, 1.5);
            return static_cast<Quantity>(qty_dist(gen_));
        }
    }

private:
    double base_intensity_;  // μ: baseline intensity
    double alpha_;           // α: self-excitation magnitude
    double beta_;            // β: decay rate
    double current_time_;
    Price reference_price_ = 10000;
    std::deque<double> events_;  // Past event times
    std::mt19937 gen_;
};

} // namespace lob
```

### Phase 4: Python Bindings

#### 4.1 Pybind11 Bindings (python/lob/bindings.cpp)
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "../../cpp/lob/include/order_book.hpp"
#include "../../cpp/lob/arrival_models/poisson.hpp"
#include "../../cpp/lob/arrival_models/hawkes.hpp"

namespace py = pybind11;
using namespace lob;

PYBIND11_MODULE(pylob, m) {
    m.doc() = "High-performance limit order book simulator";
    
    // Enums
    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL);
    
    py::enum_<OrderType>(m, "OrderType")
        .value("LIMIT", OrderType::LIMIT)
        .value("MARKET", OrderType::MARKET)
        .value("STOP", OrderType::STOP)
        .value("ICEBERG", OrderType::ICEBERG);
    
    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("DAY", TimeInForce::DAY)
        .value("GTC", TimeInForce::GTC)
        .value("IOC", TimeInForce::IOC)
        .value("FOK", TimeInForce::FOK);
    
    py::enum_<OrderStatus>(m, "OrderStatus")
        .value("NEW", OrderStatus::NEW)
        .value("PARTIALLY_FILLED", OrderStatus::PARTIALLY_FILLED)
        .value("FILLED", OrderStatus::FILLED)
        .value("CANCELLED", OrderStatus::CANCELLED);
    
    // Configuration
    py::class_<LOBConfig>(m, "LOBConfig")
        .def(py::init<>())
        .def_readwrite("tick_size", &LOBConfig::tick_size)
        .def_readwrite("min_order_size", &LOBConfig::min_order_size)
        .def_readwrite("max_order_size", &LOBConfig::max_order_size)
        .def_readwrite("latency_ns", &LOBConfig::latency_ns);
    
    // Order
    py::class_<Order, std::shared_ptr<Order>>(m, "Order")
        .def("id", &Order::id)
        .def("side", &Order::side)
        .def("price", &Order::price)
        .def("quantity", &Order::quantity)
        .def("remaining_quantity", &Order::remaining_quantity)
        .def("filled_quantity", &Order::filled_quantity)
        .def("status", &Order::status)
        .def("timestamp", &Order::timestamp);
    
    // Trade
    py::class_<Trade>(m, "Trade")
        .def_readonly("buy_order_id", &Trade::buy_order_id)
        .def_readonly("sell_order_id", &Trade::sell_order_id)
        .def_readonly("price", &Trade::price)
        .def_readonly("quantity", &Trade::quantity)
        .def_readonly("timestamp", &Trade::timestamp);
    
    // Book Snapshot
    py::class_<BookSnapshot::Level>(m, "Level")
        .def_readonly("price", &BookSnapshot::Level::price)
        .def_readonly("quantity", &BookSnapshot::Level::quantity)
        .def_readonly("order_count", &BookSnapshot::Level::order_count);
    
    py::class_<BookSnapshot>(m, "BookSnapshot")
        .def_readonly("bids", &BookSnapshot::bids)
        .def_readonly("asks", &BookSnapshot::asks)
        .def_readonly("timestamp", &BookSnapshot::timestamp);
    
    // Order Book
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<const LOBConfig&>(), py::arg("config") = LOBConfig())
        .def("add_order", &OrderBook::add_order,
             py::arg("side"), py::arg("price"), py::arg("quantity"),
             py::arg("type") = OrderType::LIMIT,
             py::arg("tif") = TimeInForce::GTC)
        .def("cancel_order", &OrderBook::cancel_order)
        .def("modify_order", &OrderBook::modify_order)
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask)
        .def("mid_price", &OrderBook::mid_price)
        .def("get_snapshot", &OrderBook::get_snapshot, py::arg("depth") = 10)
        .def("set_trade_callback", &OrderBook::set_trade_callback)
        .def("get_stats", &OrderBook::get_stats);
    
    // Statistics
    py::class_<OrderBook::Stats>(m, "Stats")
        .def_readonly("total_orders", &OrderBook::Stats::total_orders)
        .def_readonly("total_cancellations", &OrderBook::Stats::total_cancellations)
        .def_readonly("total_trades", &OrderBook::Stats::total_trades)
        .def_readonly("total_volume", &OrderBook::Stats::total_volume);
    
    // Poisson Arrival Model
    py::class_<PoissonArrivalModel>(m, "PoissonArrivalModel")
        .def(py::init<double, double, double>(),
             py::arg("lambda_buy"), py::arg("lambda_sell"),
             py::arg("lambda_cancel") = 0.5)
        .def("next_arrival", &PoissonArrivalModel::next_arrival)
        .def("set_reference_price", &PoissonArrivalModel::set_reference_price);
    
    py::class_<PoissonArrivalModel::ArrivalEvent>(m, "PoissonArrivalEvent")
        .def_readonly("type", &PoissonArrivalModel::ArrivalEvent::type)
        .def_readonly("time", &PoissonArrivalModel::ArrivalEvent::time)
        .def_readonly("price", &PoissonArrivalModel::ArrivalEvent::price)
        .def_readonly("quantity", &PoissonArrivalModel::ArrivalEvent::quantity);
    
    // Hawkes Arrival Model
    py::class_<HawkesArrivalModel>(m, "HawkesArrivalModel")
        .def(py::init<double, double, double>(),
             py::arg("base_intensity"), py::arg("alpha"), py::arg("beta"))
        .def("next_arrival", &HawkesArrivalModel::next_arrival);
    
    py::class_<HawkesArrivalModel::Event>(m, "HawkesEvent")
        .def_readonly("time", &HawkesArrivalModel::Event::time)
        .def_readonly("side", &HawkesArrivalModel::Event::side)
        .def_readonly("price", &HawkesArrivalModel::Event::price)
        .def_readonly("quantity", &HawkesArrivalModel::Event::quantity)
        .def_readonly("is_aggressive", &HawkesArrivalModel::Event::is_aggressive);
}
```

#### 4.2 Python Wrapper (python/lob/simulator.py)
```python
import numpy as np
import pandas as pd
from typing import Optional, Callable, List, Dict, Tuple
from dataclasses import dataclass
import pylob

@dataclass
class SimulationConfig:
    """Configuration for LOB simulation"""
    duration: float = 3600  # seconds
    tick_size: int = 1
    initial_price: int = 10000
    arrival_model: str = 'poisson'  # 'poisson' or 'hawkes'
    lambda_buy: float = 1.0
    lambda_sell: float = 1.0
    lambda_cancel: float = 0.5
    latency_ns: int = 0
    seed: Optional[int] = None

class LOBSimulator:
    """High-level Python interface for LOB simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Initialize order book
        lob_config = pylob.LOBConfig()
        lob_config.tick_size = config.tick_size
        lob_config.latency_ns = config.latency_ns
        
        self.book = pylob.OrderBook(lob_config)
        
        # Initialize arrival model
        if config.arrival_model == 'poisson':
            self.arrival_model = pylob.PoissonArrivalModel(
                config.lambda_buy,
                config.lambda_sell,
                config.lambda_cancel
            )
            self.arrival_model.set_reference_price(config.initial_price)
        elif config.arrival_model == 'hawkes':
            self.arrival_model = pylob.HawkesArrivalModel(
                config.lambda_buy,  # base intensity
                0.3,  # alpha
                1.0   # beta
            )
        else:
            raise ValueError(f"Unknown arrival model: {config.arrival_model}")
        
        # Storage for events
        self.trades = []
        self.snapshots = []
        self.orders = {}
        
        # Set callbacks
        self.book.set_trade_callback(self._on_trade)
        
        # Metrics
        self.metrics = {
            'total_orders': 0,
            'total_trades': 0,
            'total_volume': 0,
            'fill_rate': 0,
            'avg_spread': 0,
            'queue_position_errors': []
        }
    
    def _on_trade(self, trade):
        """Trade callback"""
        self.trades.append({
            'timestamp': trade.timestamp,
            'price': trade.price,
            'quantity': trade.quantity,
            'buy_order_id': trade.buy_order_id,
            'sell_order_id': trade.sell_order_id
        })
    
    def run_simulation(self, snapshot_interval: float = 1.0) -> pd.DataFrame:
        """Run LOB simulation"""
        
        current_time = 0
        next_snapshot_time = snapshot_interval
        
        # Initialize book with some orders
        self._initialize_book()
        
        while current_time < self.config.duration:
            # Get next arrival
            if self.config.arrival_model == 'poisson':
                event = self.arrival_model.next_arrival()
                current_time += event.time
                
                if event.type == pylob.PoissonArrivalEvent.BUY_ORDER:
                    order_id = self.book.add_order(
                        pylob.Side.BUY,
                        event.price,
                        event.quantity
                    )
                    if order_id > 0:
                        self.orders[order_id] = current_time
                        
                elif event.type == pylob.PoissonArrivalEvent.SELL_ORDER:
                    order_id = self.book.add_order(
                        pylob.Side.SELL,
                        event.price,
                        event.quantity
                    )
                    if order_id > 0:
                        self.orders[order_id] = current_time
                        
                elif event.type == pylob.PoissonArrivalEvent.CANCEL_ORDER:
                    if self.orders:
                        # Cancel random order
                        order_id = np.random.choice(list(self.orders.keys()))
                        if self.book.cancel_order(order_id):
                            del self.orders[order_id]
            
            else:  # Hawkes
                event = self.arrival_model.next_arrival()
                current_time = event.time
                
                # Determine order type based on aggressiveness
                order_type = pylob.OrderType.MARKET if event.is_aggressive else pylob.OrderType.LIMIT
                
                order_id = self.book.add_order(
                    event.side,
                    event.price,
                    event.quantity,
                    order_type
                )
                if order_id > 0:
                    self.orders[order_id] = current_time
            
            # Take snapshot
            if current_time >= next_snapshot_time:
                snapshot = self.book.get_snapshot(10)
                self._store_snapshot(snapshot, current_time)
                next_snapshot_time += snapshot_interval
            
            self.metrics['total_orders'] += 1
        
        # Calculate final metrics
        self._calculate_metrics()
        
        # Return trades DataFrame
        return pd.DataFrame(self.trades)
    
    def _initialize_book(self):
        """Initialize book with some orders"""
        mid = self.config.initial_price
        
        # Add some initial liquidity
        for i in range(10):
            # Bids
            self.book.add_order(
                pylob.Side.BUY,
                mid - (i + 1) * self.config.tick_size,
                np.random.randint(10, 100)
            )
            
            # Asks
            self.book.add_order(
                pylob.Side.SELL,
                mid + (i + 1) * self.config.tick_size,
                np.random.randint(10, 100)
            )
    
    def _store_snapshot(self, snapshot, timestamp):
        """Store book snapshot"""
        snap_data = {
            'timestamp': timestamp,
            'bid_prices': [level.price for level in snapshot.bids],
            'bid_quantities': [level.quantity for level in snapshot.bids],
            'ask_prices': [level.price for level in snapshot.asks],
            'ask_quantities': [level.quantity for level in snapshot.asks]
        }
        
        # Calculate spread
        if snapshot.bids and snapshot.asks:
            spread = snapshot.asks[0].price - snapshot.bids[0].price
            snap_data['spread'] = spread
            snap_data['mid_price'] = (snapshot.asks[0].price + snapshot.bids[0].price) / 2
        
        self.snapshots.append(snap_data)
    
    def _calculate_metrics(self):
        """Calculate simulation metrics"""
        stats = self.book.get_stats()
        
        self.metrics['total_trades'] = stats.total_trades
        self.metrics['total_volume'] = stats.total_volume
        
        if self.metrics['total_orders'] > 0:
            self.metrics['fill_rate'] = stats.total_trades / self.metrics['total_orders']
        
        if self.snapshots:
            spreads = [s.get('spread', 0) for s in self.snapshots if 'spread' in s]
            if spreads:
                self.metrics['avg_spread'] = np.mean(spreads)
    
    def get_snapshots_df(self) -> pd.DataFrame:
        """Get snapshots as DataFrame"""
        return pd.DataFrame(self.snapshots)
    
    def analyze_queue_position(self) -> Dict:
        """Analyze queue position accuracy"""
        # This would compare predicted vs actual fill times
        # Placeholder for now
        return {
            'mean_position_error': 0,
            'std_position_error': 0
        }
```

### Phase 5: Benchmarking

#### 5.1 Matching Engine Benchmark (bench/benchmark_matching.cpp)
```cpp
#include <benchmark/benchmark.h>
#include "../cpp/lob/include/order_book.hpp"
#include <random>

using namespace lob;

static void BM_OrderMatching(benchmark::State& state) {
    // Setup
    LOBConfig config;
    config.tick_size = 1;
    config.latency_ns = 0;
    
    OrderBook book(config);
    
    // Pre-populate book
    for (int i = 1; i <= 100; ++i) {
        book.add_order(Side::BUY, 10000 - i, 100);
        book.add_order(Side::SELL, 10000 + i, 100);
    }
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<> price_dist(9900, 10100);
    std::uniform_int_distribution<> qty_dist(1, 50);
    
    // Benchmark
    for (auto _ : state) {
        Price price = price_dist(gen);
        Quantity qty = qty_dist(gen);
        Side side = (gen() % 2) ? Side::BUY : Side::SELL;
        
        auto order_id = book.add_order(side, price, qty);
        benchmark::DoNotOptimize(order_id);
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_OrderCancellation(benchmark::State& state) {
    // Setup
    LOBConfig config;
    OrderBook book(config);
    
    std::vector<OrderId> order_ids;
    
    // Pre-populate
    for (int i = 0; i < 10000; ++i) {
        auto id = book.add_order(
            (i % 2) ? Side::BUY : Side::SELL,
            10000 + (i % 100),
            100
        );
        order_ids.push_back(id);
    }
    
    size_t index = 0;
    
    // Benchmark
    for (auto _ : state) {
        bool result = book.cancel_order(order_ids[index % order_ids.size()]);
        benchmark::DoNotOptimize(result);
        index++;
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_SnapshotGeneration(benchmark::State& state) {
    // Setup
    LOBConfig config;
    OrderBook book(config);
    
    // Create realistic book
    for (int i = 1; i <= 1000; ++i) {
        book.add_order(Side::BUY, 10000 - i, i * 10);
        book.add_order(Side::SELL, 10000 + i, i * 10);
    }
    
    size_t depth = state.range(0);
    
    // Benchmark
    for (auto _ : state) {
        auto snapshot = book.get_snapshot(depth);
        benchmark::DoNotOptimize(snapshot);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_OrderMatching)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_OrderCancellation)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SnapshotGeneration)
    ->Arg(1)->Arg(5)->Arg(10)->Arg(20)->Arg(50)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
```

### Phase 6: Testing

#### 6.1 C++ Tests (tests/cpp/test_order_book.cpp)
```cpp
#include <gtest/gtest.h>
#include "../../cpp/lob/include/order_book.hpp"

using namespace lob;

class OrderBookTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.tick_size = 1;
        config_.latency_ns = 0;
        book_ = std::make_unique<OrderBook>(config_);
    }
    
    LOBConfig config_;
    std::unique_ptr<OrderBook> book_;
};

TEST_F(OrderBookTest, AddLimitOrder) {
    auto order_id = book_->add_order(Side::BUY, 100, 10);
    EXPECT_GT(order_id, 0);
    EXPECT_EQ(book_->best_bid(), 100);
}

TEST_F(OrderBookTest, MatchLimitOrders) {
    std::vector<Trade> trades;
    book_->set_trade_callback([&trades](const Trade& t) {
        trades.push_back(t);
    });
    
    // Add buy order
    book_->add_order(Side::BUY, 100, 10);
    EXPECT_EQ(trades.size(), 0);
    
    // Add matching sell order
    book_->add_order(Side::SELL, 100, 5);
    EXPECT_EQ(trades.size(), 1);
    EXPECT_EQ(trades[0].price, 100);
    EXPECT_EQ(trades[0].quantity, 5);
}

TEST_F(OrderBookTest, PriceTimePriority) {
    // Add orders at same price
    auto id1 = book_->add_order(Side::BUY, 100, 10);
    auto id2 = book_->add_order(Side::BUY, 100, 10);
    
    std::vector<Trade> trades;
    book_->set_trade_callback([&trades](const Trade& t) {
        trades.push_back(t);
    });
    
    // Match against them
    book_->add_order(Side::SELL, 100, 15);
    
    // First order should be filled completely
    EXPECT_EQ(trades.size(), 2);
    EXPECT_EQ(trades[0].buy_order_id, id1);
    EXPECT_EQ(trades[0].quantity, 10);
    EXPECT_EQ(trades[1].buy_order_id, id2);
    EXPECT_EQ(trades[1].quantity, 5);
}

TEST_F(OrderBookTest, MarketOrder) {
    // Setup book
    book_->add_order(Side::BUY, 99, 10);
    book_->add_order(Side::BUY, 98, 20);
    book_->add_order(Side::SELL, 101, 10);
    book_->add_order(Side::SELL, 102, 20);
    
    std::vector<Trade> trades;
    book_->set_trade_callback([&trades](const Trade& t) {
        trades.push_back(t);
    });
    
    // Market buy order
    book_->add_order(Side::BUY, 0, 25, OrderType::MARKET);
    
    // Should match at 101 and 102
    EXPECT_EQ(trades.size(), 2);
    EXPECT_EQ(trades[0].price, 101);
    EXPECT_EQ(trades[0].quantity, 10);
    EXPECT_EQ(trades[1].price, 102);
    EXPECT_EQ(trades[1].quantity, 15);
}

TEST_F(OrderBookTest, CancelOrder) {
    auto order_id = book_->add_order(Side::BUY, 100, 10);
    EXPECT_EQ(book_->best_bid(), 100);
    
    bool cancelled = book_->cancel_order(order_id);
    EXPECT_TRUE(cancelled);
    EXPECT_EQ(book_->best_bid(), 0);
    
    // Try to cancel again
    cancelled = book_->cancel_order(order_id);
    EXPECT_FALSE(cancelled);
}

TEST_F(OrderBookTest, BookSnapshot) {
    // Build book
    for (int i = 1; i <= 10; ++i) {
        book_->add_order(Side::BUY, 100 - i, i * 10);
        book_->add_order(Side::SELL, 100 + i, i * 10);
    }
    
    auto snapshot = book_->get_snapshot(5);
    
    EXPECT_EQ(snapshot.bids.size(), 5);
    EXPECT_EQ(snapshot.asks.size(), 5);
    
    // Check best bid/ask
    EXPECT_EQ(snapshot.bids[0].price, 99);
    EXPECT_EQ(snapshot.asks[0].price, 101);
    
    // Check quantities
    EXPECT_EQ(snapshot.bids[0].quantity, 10);
    EXPECT_EQ(snapshot.asks[0].quantity, 10);
}
```

#### 6.2 Python Tests (tests/python/test_simulator.py)
```python
import pytest
import numpy as np
import pandas as pd
from python.lob.simulator import LOBSimulator, SimulationConfig

def test_basic_simulation():
    """Test basic simulation functionality"""
    config = SimulationConfig(
        duration=10.0,
        tick_size=1,
        initial_price=10000,
        arrival_model='poisson',
        lambda_buy=2.0,
        lambda_sell=2.0,
        lambda_cancel=0.5
    )
    
    sim = LOBSimulator(config)
    trades = sim.run_simulation(snapshot_interval=1.0)
    
    assert len(trades) >= 0
    assert sim.metrics['total_orders'] > 0
    assert sim.metrics['avg_spread'] > 0

def test_hawkes_simulation():
    """Test Hawkes process simulation"""
    config = SimulationConfig(
        duration=10.0,
        arrival_model='hawkes',
        lambda_buy=1.0
    )
    
    sim = LOBSimulator(config)
    trades = sim.run_simulation()
    
    # Check for clustering (characteristic of Hawkes)
    if len(trades) > 1:
        inter_trade_times = np.diff(trades['timestamp'])
        cv = np.std(inter_trade_times) / np.mean(inter_trade_times)
        # Hawkes should have higher CV than Poisson (CV=1)
        assert cv > 0.5

def test_order_matching():
    """Test order matching logic"""
    config = SimulationConfig(duration=1.0)
    sim = LOBSimulator(config)
    
    # Add specific orders
    buy_id = sim.book.add_order(pylob.Side.BUY, 100, 10)
    sell_id = sim.book.add_order(pylob.Side.SELL, 100, 5)
    
    # Check trade occurred
    assert len(sim.trades) == 1
    assert sim.trades[0]['price'] == 100
    assert sim.trades[0]['quantity'] == 5

def test_snapshot_accuracy():
    """Test snapshot generation"""
    config = SimulationConfig(duration=5.0)
    sim = LOBSimulator(config)
    
    # Initialize book
    sim._initialize_book()
    
    # Get snapshot
    snapshot = sim.book.get_snapshot(5)
    
    assert len(snapshot.bids) <= 5
    assert len(snapshot.asks) <= 5
    
    # Check price ordering
    if len(snapshot.bids) > 1:
        bid_prices = [level.price for level in snapshot.bids]
        assert bid_prices == sorted(bid_prices, reverse=True)
    
    if len(snapshot.asks) > 1:
        ask_prices = [level.price for level in snapshot.asks]
        assert ask_prices == sorted(ask_prices)

def test_metrics_calculation():
    """Test metrics calculation"""
    config = SimulationConfig(duration=10.0)
    sim = LOBSimulator(config)
    
    trades = sim.run_simulation()
    
    assert 'fill_rate' in sim.metrics
    assert 0 <= sim.metrics['fill_rate'] <= 1
    
    assert 'avg_spread' in sim.metrics
    assert sim.metrics['avg_spread'] >= 0
    
    assert sim.metrics['total_volume'] >= 0
```

### Phase 7: Build Configuration

#### 7.1 CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.14)
project(LOBSimulator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Find packages
find_package(Threads REQUIRED)

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
endif()

if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
endif()

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
endif()

# Library
add_library(lob STATIC
    cpp/lob/src/order.cpp
    cpp/lob/src/order_book.cpp
    cpp/lob/src/price_level.cpp
    cpp/lob/src/matching_engine.cpp
    cpp/lob/src/event_queue.cpp
)

target_include_directories(lob PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lob/include
)

target_link_libraries(lob PUBLIC Threads::Threads)

# Compile options
target_compile_options(lob PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -march=native
    -ffast-math
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_lob
        tests/cpp/test_order_book.cpp
        tests/cpp/test_matching.cpp
        tests/cpp/test_price_level.cpp
    )
    
    target_link_libraries(test_lob
        lob
        GTest::GTest
        GTest::Main
    )
    
    add_test(NAME test_lob COMMAND test_lob)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_executable(bench_matching bench/benchmark_matching.cpp)
    target_link_libraries(bench_matching lob benchmark::benchmark)
    
    add_executable(bench_insertion bench/benchmark_insertion.cpp)
    target_link_libraries(bench_insertion lob benchmark::benchmark)
    
    add_executable(bench_cancellation bench/benchmark_cancellation.cpp)
    target_link_libraries(bench_cancellation lob benchmark::benchmark)
    
    add_executable(bench_snapshot bench/benchmark_snapshot.cpp)
    target_link_libraries(bench_snapshot lob benchmark::benchmark)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(pylob python/lob/bindings.cpp)
    target_link_libraries(pylob PRIVATE lob)
    
    # Install to Python package directory
    install(TARGETS pylob DESTINATION python/lob)
endif()
```

#### 7.2 Setup.py
```python
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pylob",
        ["python/lob/bindings.cpp"],
        include_dirs=["cpp/lob/include"],
        libraries=["lob"],
        library_dirs=["build/release"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="lob-simulator",
    version="1.0.0",
    author="Your Name",
    description="High-performance limit order book simulator",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "pandas>=1.2",
        "matplotlib>=3.3",
        "pytest>=6.0",
    ],
)
```

### Phase 8: Usage Examples

#### 8.1 Performance Analysis Script
```python
#!/usr/bin/env python3
"""
Analyze LOB simulator performance
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python.lob.simulator import LOBSimulator, SimulationConfig

def benchmark_throughput():
    """Benchmark order processing throughput"""
    
    configs = [
        ('Poisson', SimulationConfig(
            duration=60,
            arrival_model='poisson',
            lambda_buy=100,
            lambda_sell=100,
            lambda_cancel=50
        )),
        ('Hawkes', SimulationConfig(
            duration=60,
            arrival_model='hawkes',
            lambda_buy=50
        ))
    ]
    
    results = []
    
    for name, config in configs:
        sim = LOBSimulator(config)
        
        start_time = time.time()
        trades = sim.run_simulation(snapshot_interval=10)
        elapsed = time.time() - start_time
        
        throughput = sim.metrics['total_orders'] / elapsed
        
        results.append({
            'Model': name,
            'Orders': sim.metrics['total_orders'],
            'Trades': len(trades),
            'Time (s)': elapsed,
            'Throughput (orders/sec)': throughput,
            'Fill Rate': sim.metrics['fill_rate'],
            'Avg Spread': sim.metrics['avg_spread']
        })
    
    return pd.DataFrame(results)

def analyze_market_quality():
    """Analyze market quality metrics"""
    
    config = SimulationConfig(
        duration=300,
        tick_size=1,
        initial_price=10000,
        arrival_model='hawkes'
    )
    
    sim = LOBSimulator(config)
    trades = sim.run_simulation(snapshot_interval=0.1)
    snapshots = sim.get_snapshots_df()
    
    # Calculate metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Spread over time
    axes[0, 0].plot(snapshots['timestamp'], snapshots['spread'])
    axes[0, 0].set_title('Spread Over Time')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Spread (ticks)')
    
    # Mid price evolution
    axes[0, 1].plot(snapshots['timestamp'], snapshots['mid_price'])
    axes[0, 1].set_title('Mid Price Evolution')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Price')
    
    # Trade size distribution
    if len(trades) > 0:
        axes[1, 0].hist(trades['quantity'], bins=30, edgecolor='black')
        axes[1, 0].set_title('Trade Size Distribution')
        axes[1, 0].set_xlabel('Size')
        axes[1, 0].set_ylabel('Frequency')
    
    # Volume profile
    axes[1, 1].plot(trades['timestamp'], trades['quantity'].cumsum())
    axes[1, 1].set_title('Cumulative Volume')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Volume')
    
    plt.tight_layout()
    plt.savefig('market_quality.png')
    
    return snapshots, trades

def test_latency_impact():
    """Test impact of latency on execution"""
    
    latencies = [0, 100, 1000, 10000, 100000]  # nanoseconds
    results = []
    
    for latency in latencies:
        config = SimulationConfig(
            duration=60,
            latency_ns=latency,
            arrival_model='poisson',
            lambda_buy=10,
            lambda_sell=10
        )
        
        sim = LOBSimulator(config)
        trades = sim.run_simulation()
        
        results.append({
            'Latency (ns)': latency,
            'Latency (μs)': latency / 1000,
            'Fill Rate': sim.metrics['fill_rate'],
            'Avg Spread': sim.metrics['avg_spread'],
            'Total Trades': len(trades)
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Benchmarking throughput...")
    throughput_df = benchmark_throughput()
    print(throughput_df.to_string())
    
    print("\nAnalyzing market quality...")
    snapshots, trades = analyze_market_quality()
    
    print("\nTesting latency impact...")
    latency_df = test_latency_impact()
    print(latency_df.to_string())
    
    # Save results
    throughput_df.to_csv('throughput_benchmark.csv')
    latency_df.to_csv('latency_impact.csv')
    
    print("\nResults saved to CSV files and plots saved to PNG")
```

## Performance Metrics & Targets

### Throughput Benchmarks
- **Order Processing**: > 1,000,000 orders/second
- **Order Matching**: > 500,000 matches/second
- **Cancellation**: > 2,000,000 cancels/second
- **Snapshot Generation**: < 10 μs for 10-level snapshot

### Latency Benchmarks
- **Order Insertion**: < 100 ns (p50), < 500 ns (p99)
- **Order Matching**: < 200 ns (p50), < 1 μs (p99)
- **Order Cancellation**: < 50 ns (p50), < 200 ns (p99)

### Memory Efficiency
- **Per Order**: < 128 bytes
- **Per Price Level**: < 256 bytes + orders
- **Total Memory**: < 100 MB for 1M orders

### Accuracy Metrics
- **Fill Rate**: Accurate to actual market conditions
- **Queue Position**: < 5% average position error
- **Price Discovery**: Convergence to theoretical equilibrium

## Testing & Validation Checklist

- [ ] Price-time priority is strictly enforced
- [ ] No lost orders or phantom fills
- [ ] Correct handling of all order types
- [ ] Proper partial fill handling
- [ ] Thread-safe operations (if enabled)
- [ ] Memory leak free over long simulations
- [ ] Accurate Poisson/Hawkes arrival generation
- [ ] Snapshot consistency with book state
- [ ] Python bindings match C++ behavior
- [ ] Benchmark results meet targets

## Next Steps

1. Implement advanced order types (stop-loss, hidden, pegged)
2. Add market maker models
3. Implement regulatory flags (self-trade prevention)
4. Add FIX protocol support
5. Build distributed simulation capability
6. Create market replay functionality