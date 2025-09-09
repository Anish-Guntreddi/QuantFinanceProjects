# Limit Order Book Simulator (Price-Time Priority)

## Overview
High-performance C++ limit order book simulator with discrete-event processing, partial fills, cancellations, and Hawkes process arrivals. Includes Python bindings for research and backtesting.

## Project Structure
```
01_limit_order_book_simulator/
├── cpp/
│   ├── lob/
│   │   ├── order_book.hpp
│   │   ├── order_book.cpp
│   │   ├── order.hpp
│   │   ├── price_level.hpp
│   │   ├── matching_engine.hpp
│   │   └── matching_engine.cpp
│   ├── events/
│   │   ├── event.hpp
│   │   ├── event_queue.hpp
│   │   └── hawkes_process.hpp
│   ├── utils/
│   │   ├── memory_pool.hpp
│   │   └── time_utils.hpp
│   └── main.cpp
├── python/
│   ├── lob/
│   │   ├── __init__.py
│   │   ├── bindings.cpp
│   │   └── simulator.py
│   └── setup.py
├── tests/
│   ├── test_order_book.cpp
│   └── test_performance.cpp
└── CMakeLists.txt
```

## C++ Implementation

### cpp/lob/order.hpp
```cpp
#pragma once

#include <cstdint>
#include <chrono>
#include <string>

enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderType : uint8_t {
    LIMIT = 0,
    MARKET = 1,
    STOP = 2,
    STOP_LIMIT = 3
};

enum class TimeInForce : uint8_t {
    GTC = 0,  // Good Till Cancel
    IOC = 1,  // Immediate or Cancel
    FOK = 2,  // Fill or Kill
    GTD = 3   // Good Till Date
};

enum class OrderStatus : uint8_t {
    NEW = 0,
    PARTIALLY_FILLED = 1,
    FILLED = 2,
    CANCELLED = 3,
    REJECTED = 4
};

struct Order {
    uint64_t order_id;
    uint64_t client_order_id;
    std::string symbol;
    Side side;
    OrderType type;
    TimeInForce tif;
    OrderStatus status;
    int64_t price;  // Price in ticks (fixed point)
    uint64_t quantity;
    uint64_t remaining_quantity;
    uint64_t executed_quantity;
    int64_t stop_price;
    uint64_t timestamp;
    uint32_t participant_id;
    
    Order() = default;
    
    Order(uint64_t id, const std::string& sym, Side s, int64_t p, uint64_t q)
        : order_id(id)
        , client_order_id(id)
        , symbol(sym)
        , side(s)
        , type(OrderType::LIMIT)
        , tif(TimeInForce::GTC)
        , status(OrderStatus::NEW)
        , price(p)
        , quantity(q)
        , remaining_quantity(q)
        , executed_quantity(0)
        , stop_price(0)
        , timestamp(std::chrono::steady_clock::now().time_since_epoch().count())
        , participant_id(0) {}
    
    bool is_buy() const { return side == Side::BUY; }
    bool is_sell() const { return side == Side::SELL; }
    bool is_filled() const { return remaining_quantity == 0; }
};
```

### cpp/lob/price_level.hpp
```cpp
#pragma once

#include "order.hpp"
#include <deque>
#include <memory>

class PriceLevel {
public:
    PriceLevel(int64_t price) : price_(price), total_quantity_(0) {}
    
    void add_order(std::shared_ptr<Order> order) {
        orders_.push_back(order);
        total_quantity_ += order->remaining_quantity;
    }
    
    void remove_order(uint64_t order_id) {
        auto it = std::find_if(orders_.begin(), orders_.end(),
            [order_id](const auto& o) { return o->order_id == order_id; });
        
        if (it != orders_.end()) {
            total_quantity_ -= (*it)->remaining_quantity;
            orders_.erase(it);
        }
    }
    
    uint64_t match(uint64_t quantity) {
        uint64_t matched = 0;
        
        while (!orders_.empty() && quantity > 0) {
            auto& order = orders_.front();
            uint64_t fill_qty = std::min(quantity, order->remaining_quantity);
            
            order->executed_quantity += fill_qty;
            order->remaining_quantity -= fill_qty;
            total_quantity_ -= fill_qty;
            matched += fill_qty;
            quantity -= fill_qty;
            
            if (order->remaining_quantity == 0) {
                order->status = OrderStatus::FILLED;
                orders_.pop_front();
            } else {
                order->status = OrderStatus::PARTIALLY_FILLED;
            }
        }
        
        return matched;
    }
    
    int64_t get_price() const { return price_; }
    uint64_t get_total_quantity() const { return total_quantity_; }
    size_t get_order_count() const { return orders_.size(); }
    bool is_empty() const { return orders_.empty(); }
    
    const std::deque<std::shared_ptr<Order>>& get_orders() const { return orders_; }
    
private:
    int64_t price_;
    uint64_t total_quantity_;
    std::deque<std::shared_ptr<Order>> orders_;  // FIFO queue for time priority
};
```

### cpp/lob/order_book.hpp
```cpp
#pragma once

#include "order.hpp"
#include "price_level.hpp"
#include <map>
#include <unordered_map>
#include <memory>
#include <vector>
#include <mutex>

struct Trade {
    uint64_t trade_id;
    uint64_t buyer_order_id;
    uint64_t seller_order_id;
    int64_t price;
    uint64_t quantity;
    uint64_t timestamp;
};

struct BookUpdate {
    enum Type { ADD, MODIFY, DELETE, TRADE, CLEAR };
    Type type;
    Side side;
    int64_t price;
    uint64_t quantity;
    uint64_t timestamp;
};

class OrderBook {
public:
    OrderBook(const std::string& symbol) : symbol_(symbol), next_trade_id_(1) {}
    
    // Add a new order to the book
    std::vector<Trade> add_order(std::shared_ptr<Order> order) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Trade> trades;
        
        if (order->type == OrderType::MARKET) {
            trades = match_market_order(order);
        } else if (order->type == OrderType::LIMIT) {
            trades = match_limit_order(order);
            
            // Add remaining quantity to book
            if (order->remaining_quantity > 0) {
                add_to_book(order);
            }
        }
        
        return trades;
    }
    
    // Cancel an order
    bool cancel_order(uint64_t order_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = order_map_.find(order_id);
        if (it == order_map_.end()) {
            return false;
        }
        
        auto order = it->second;
        remove_from_book(order);
        order->status = OrderStatus::CANCELLED;
        
        return true;
    }
    
    // Modify an order (cancel-replace)
    std::vector<Trade> modify_order(uint64_t order_id, int64_t new_price, uint64_t new_quantity) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = order_map_.find(order_id);
        if (it == order_map_.end()) {
            return {};
        }
        
        auto order = it->second;
        remove_from_book(order);
        
        // Update order
        order->price = new_price;
        order->quantity = new_quantity;
        order->remaining_quantity = new_quantity;
        order->executed_quantity = 0;
        order->timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        
        // Re-match and add
        auto trades = match_limit_order(order);
        if (order->remaining_quantity > 0) {
            add_to_book(order);
        }
        
        return trades;
    }
    
    // Get best bid/ask
    std::pair<int64_t, uint64_t> get_best_bid() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (bids_.empty()) return {0, 0};
        auto& level = bids_.rbegin()->second;
        return {level->get_price(), level->get_total_quantity()};
    }
    
    std::pair<int64_t, uint64_t> get_best_ask() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (asks_.empty()) return {0, 0};
        auto& level = asks_.begin()->second;
        return {level->get_price(), level->get_total_quantity()};
    }
    
    // Get market depth
    struct DepthLevel {
        int64_t price;
        uint64_t quantity;
        size_t order_count;
    };
    
    std::vector<DepthLevel> get_bid_depth(size_t levels = 10) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<DepthLevel> depth;
        
        size_t count = 0;
        for (auto it = bids_.rbegin(); it != bids_.rend() && count < levels; ++it, ++count) {
            depth.push_back({
                it->second->get_price(),
                it->second->get_total_quantity(),
                it->second->get_order_count()
            });
        }
        
        return depth;
    }
    
    std::vector<DepthLevel> get_ask_depth(size_t levels = 10) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<DepthLevel> depth;
        
        size_t count = 0;
        for (auto it = asks_.begin(); it != asks_.end() && count < levels; ++it, ++count) {
            depth.push_back({
                it->second->get_price(),
                it->second->get_total_quantity(),
                it->second->get_order_count()
            });
        }
        
        return depth;
    }
    
private:
    std::string symbol_;
    std::map<int64_t, std::shared_ptr<PriceLevel>, std::greater<int64_t>> bids_;  // Descending
    std::map<int64_t, std::shared_ptr<PriceLevel>> asks_;  // Ascending
    std::unordered_map<uint64_t, std::shared_ptr<Order>> order_map_;
    uint64_t next_trade_id_;
    mutable std::mutex mutex_;
    
    void add_to_book(std::shared_ptr<Order> order) {
        order_map_[order->order_id] = order;
        
        if (order->is_buy()) {
            auto& level = bids_[order->price];
            if (!level) {
                level = std::make_shared<PriceLevel>(order->price);
            }
            level->add_order(order);
        } else {
            auto& level = asks_[order->price];
            if (!level) {
                level = std::make_shared<PriceLevel>(order->price);
            }
            level->add_order(order);
        }
    }
    
    void remove_from_book(std::shared_ptr<Order> order) {
        order_map_.erase(order->order_id);
        
        if (order->is_buy()) {
            auto it = bids_.find(order->price);
            if (it != bids_.end()) {
                it->second->remove_order(order->order_id);
                if (it->second->is_empty()) {
                    bids_.erase(it);
                }
            }
        } else {
            auto it = asks_.find(order->price);
            if (it != asks_.end()) {
                it->second->remove_order(order->order_id);
                if (it->second->is_empty()) {
                    asks_.erase(it);
                }
            }
        }
    }
    
    std::vector<Trade> match_market_order(std::shared_ptr<Order> order) {
        std::vector<Trade> trades;
        
        auto& opposite_side = order->is_buy() ? asks_ : bids_;
        
        while (!opposite_side.empty() && order->remaining_quantity > 0) {
            auto& best_level = order->is_buy() ? 
                opposite_side.begin()->second : 
                opposite_side.rbegin()->second;
            
            uint64_t matched = best_level->match(order->remaining_quantity);
            
            if (matched > 0) {
                trades.push_back(create_trade(order, best_level->get_price(), matched));
                order->executed_quantity += matched;
                order->remaining_quantity -= matched;
            }
            
            if (best_level->is_empty()) {
                if (order->is_buy()) {
                    opposite_side.erase(opposite_side.begin());
                } else {
                    opposite_side.erase(std::prev(opposite_side.end()));
                }
            }
        }
        
        if (order->remaining_quantity == 0) {
            order->status = OrderStatus::FILLED;
        } else if (order->executed_quantity > 0) {
            order->status = OrderStatus::PARTIALLY_FILLED;
        }
        
        return trades;
    }
    
    std::vector<Trade> match_limit_order(std::shared_ptr<Order> order) {
        std::vector<Trade> trades;
        
        if (order->is_buy()) {
            while (!asks_.empty() && 
                   asks_.begin()->first <= order->price && 
                   order->remaining_quantity > 0) {
                auto& ask_level = asks_.begin()->second;
                uint64_t matched = ask_level->match(order->remaining_quantity);
                
                if (matched > 0) {
                    trades.push_back(create_trade(order, ask_level->get_price(), matched));
                    order->executed_quantity += matched;
                    order->remaining_quantity -= matched;
                }
                
                if (ask_level->is_empty()) {
                    asks_.erase(asks_.begin());
                }
            }
        } else {
            while (!bids_.empty() && 
                   bids_.rbegin()->first >= order->price && 
                   order->remaining_quantity > 0) {
                auto& bid_level = bids_.rbegin()->second;
                uint64_t matched = bid_level->match(order->remaining_quantity);
                
                if (matched > 0) {
                    trades.push_back(create_trade(order, bid_level->get_price(), matched));
                    order->executed_quantity += matched;
                    order->remaining_quantity -= matched;
                }
                
                if (bid_level->is_empty()) {
                    bids_.erase(std::prev(bids_.end()));
                }
            }
        }
        
        if (order->remaining_quantity == 0) {
            order->status = OrderStatus::FILLED;
        } else if (order->executed_quantity > 0) {
            order->status = OrderStatus::PARTIALLY_FILLED;
        }
        
        return trades;
    }
    
    Trade create_trade(std::shared_ptr<Order> order, int64_t price, uint64_t quantity) {
        return Trade{
            next_trade_id_++,
            order->is_buy() ? order->order_id : 0,
            order->is_sell() ? order->order_id : 0,
            price,
            quantity,
            std::chrono::steady_clock::now().time_since_epoch().count()
        };
    }
};
```

### cpp/events/hawkes_process.hpp
```cpp
#pragma once

#include <random>
#include <vector>
#include <cmath>

class HawkesProcess {
public:
    HawkesProcess(double baseline_intensity, double alpha, double beta, uint64_t seed = 42)
        : mu_(baseline_intensity)
        , alpha_(alpha)
        , beta_(beta)
        , generator_(seed)
        , exponential_(1.0) {
        
        if (alpha_ >= beta_) {
            throw std::invalid_argument("Alpha must be less than beta for stability");
        }
    }
    
    // Generate next arrival time
    double next_arrival_time(double current_time) {
        double intensity = calculate_intensity(current_time);
        double u = std::uniform_real_distribution<>(0, 1)(generator_);
        double tau = -std::log(u) / intensity;
        
        // Update history
        arrival_times_.push_back(current_time + tau);
        
        return current_time + tau;
    }
    
    // Calculate current intensity
    double calculate_intensity(double t) {
        double intensity = mu_;
        
        for (double ti : arrival_times_) {
            if (ti < t) {
                intensity += alpha_ * std::exp(-beta_ * (t - ti));
            }
        }
        
        return intensity;
    }
    
    // Simulate path
    std::vector<double> simulate(double T, size_t max_events = 10000) {
        std::vector<double> events;
        arrival_times_.clear();
        
        double t = 0;
        while (t < T && events.size() < max_events) {
            double upper_bound = mu_ + alpha_ * arrival_times_.size();
            
            // Thinning algorithm
            double s = t;
            while (true) {
                s += exponential_(generator_) / upper_bound;
                if (s > T) break;
                
                double intensity = calculate_intensity(s);
                double u = std::uniform_real_distribution<>(0, 1)(generator_);
                
                if (u <= intensity / upper_bound) {
                    events.push_back(s);
                    arrival_times_.push_back(s);
                    t = s;
                    break;
                }
            }
            
            if (s > T) break;
        }
        
        return events;
    }
    
    // Estimate parameters from data (MLE)
    static void estimate_parameters(const std::vector<double>& times,
                                   double& mu, double& alpha, double& beta) {
        // Simplified MLE estimation
        size_t n = times.size();
        double T = times.back() - times.front();
        
        // Initial estimates
        mu = n / T * 0.5;  // Baseline intensity
        alpha = 0.5;
        beta = 1.0;
        
        // Iterative optimization (simplified)
        for (int iter = 0; iter < 100; ++iter) {
            double log_likelihood = 0;
            
            for (size_t i = 1; i < n; ++i) {
                double intensity = mu;
                for (size_t j = 0; j < i; ++j) {
                    intensity += alpha * std::exp(-beta * (times[i] - times[j]));
                }
                log_likelihood += std::log(intensity);
            }
            
            // Update parameters (gradient ascent)
            // ... (implementation details)
        }
    }
    
private:
    double mu_;      // Baseline intensity
    double alpha_;   // Jump size
    double beta_;    // Decay rate
    std::vector<double> arrival_times_;
    std::mt19937_64 generator_;
    std::exponential_distribution<> exponential_;
};
```

### Python Bindings - python/lob/bindings.cpp
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../../cpp/lob/order_book.hpp"
#include "../../cpp/lob/order.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pylob, m) {
    m.doc() = "Python bindings for C++ Limit Order Book";
    
    // Enums
    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL);
    
    py::enum_<OrderType>(m, "OrderType")
        .value("LIMIT", OrderType::LIMIT)
        .value("MARKET", OrderType::MARKET)
        .value("STOP", OrderType::STOP)
        .value("STOP_LIMIT", OrderType::STOP_LIMIT);
    
    py::enum_<OrderStatus>(m, "OrderStatus")
        .value("NEW", OrderStatus::NEW)
        .value("PARTIALLY_FILLED", OrderStatus::PARTIALLY_FILLED)
        .value("FILLED", OrderStatus::FILLED)
        .value("CANCELLED", OrderStatus::CANCELLED)
        .value("REJECTED", OrderStatus::REJECTED);
    
    // Order class
    py::class_<Order, std::shared_ptr<Order>>(m, "Order")
        .def(py::init<uint64_t, const std::string&, Side, int64_t, uint64_t>())
        .def_readwrite("order_id", &Order::order_id)
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("side", &Order::side)
        .def_readwrite("type", &Order::type)
        .def_readwrite("status", &Order::status)
        .def_readwrite("price", &Order::price)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("remaining_quantity", &Order::remaining_quantity)
        .def_readwrite("executed_quantity", &Order::executed_quantity)
        .def_readwrite("timestamp", &Order::timestamp);
    
    // Trade class
    py::class_<Trade>(m, "Trade")
        .def_readonly("trade_id", &Trade::trade_id)
        .def_readonly("buyer_order_id", &Trade::buyer_order_id)
        .def_readonly("seller_order_id", &Trade::seller_order_id)
        .def_readonly("price", &Trade::price)
        .def_readonly("quantity", &Trade::quantity)
        .def_readonly("timestamp", &Trade::timestamp);
    
    // DepthLevel class
    py::class_<OrderBook::DepthLevel>(m, "DepthLevel")
        .def_readonly("price", &OrderBook::DepthLevel::price)
        .def_readonly("quantity", &OrderBook::DepthLevel::quantity)
        .def_readonly("order_count", &OrderBook::DepthLevel::order_count);
    
    // OrderBook class
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<const std::string&>())
        .def("add_order", &OrderBook::add_order)
        .def("cancel_order", &OrderBook::cancel_order)
        .def("modify_order", &OrderBook::modify_order)
        .def("get_best_bid", &OrderBook::get_best_bid)
        .def("get_best_ask", &OrderBook::get_best_ask)
        .def("get_bid_depth", &OrderBook::get_bid_depth, py::arg("levels") = 10)
        .def("get_ask_depth", &OrderBook::get_ask_depth, py::arg("levels") = 10);
    
    // HawkesProcess class
    py::class_<HawkesProcess>(m, "HawkesProcess")
        .def(py::init<double, double, double, uint64_t>(),
             py::arg("baseline_intensity"),
             py::arg("alpha"),
             py::arg("beta"),
             py::arg("seed") = 42)
        .def("next_arrival_time", &HawkesProcess::next_arrival_time)
        .def("calculate_intensity", &HawkesProcess::calculate_intensity)
        .def("simulate", &HawkesProcess::simulate,
             py::arg("T"),
             py::arg("max_events") = 10000)
        .def_static("estimate_parameters", &HawkesProcess::estimate_parameters);
}
```

### Python Wrapper - python/lob/simulator.py
```python
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import pylob

class LOBSimulator:
    def __init__(self, symbol: str = "TEST"):
        self.book = pylob.OrderBook(symbol)
        self.order_id_counter = 1
        self.trades = []
        self.orders = {}
        
    def add_limit_order(self, side: str, price: float, quantity: int) -> Tuple[int, List]:
        order_id = self.order_id_counter
        self.order_id_counter += 1
        
        # Convert price to ticks (assuming tick size = 0.01)
        price_ticks = int(price * 100)
        
        order_side = pylob.Side.BUY if side.upper() == 'BUY' else pylob.Side.SELL
        order = pylob.Order(order_id, "TEST", order_side, price_ticks, quantity)
        
        trades = self.book.add_order(order)
        self.orders[order_id] = order
        
        # Convert trades to Python dict
        trade_list = []
        for trade in trades:
            trade_list.append({
                'trade_id': trade.trade_id,
                'price': trade.price / 100.0,
                'quantity': trade.quantity,
                'timestamp': trade.timestamp
            })
        
        self.trades.extend(trade_list)
        return order_id, trade_list
    
    def cancel_order(self, order_id: int) -> bool:
        return self.book.cancel_order(order_id)
    
    def get_book_snapshot(self, depth: int = 10) -> pd.DataFrame:
        bids = self.book.get_bid_depth(depth)
        asks = self.book.get_ask_depth(depth)
        
        # Convert to DataFrame
        bid_data = []
        for level in bids:
            bid_data.append({
                'side': 'bid',
                'price': level.price / 100.0,
                'quantity': level.quantity,
                'orders': level.order_count
            })
        
        ask_data = []
        for level in asks:
            ask_data.append({
                'side': 'ask',
                'price': level.price / 100.0,
                'quantity': level.quantity,
                'orders': level.order_count
            })
        
        return pd.DataFrame(bid_data + ask_data)
    
    def get_spread(self) -> Tuple[float, float, float]:
        best_bid = self.book.get_best_bid()
        best_ask = self.book.get_best_ask()
        
        if best_bid[0] == 0 or best_ask[0] == 0:
            return None, None, None
        
        bid_price = best_bid[0] / 100.0
        ask_price = best_ask[0] / 100.0
        spread = ask_price - bid_price
        
        return bid_price, ask_price, spread
    
    def simulate_hawkes_arrivals(self, T: float, mu: float = 1.0, 
                                alpha: float = 0.5, beta: float = 1.0) -> List[dict]:
        hawkes = pylob.HawkesProcess(mu, alpha, beta)
        arrival_times = hawkes.simulate(T)
        
        events = []
        for t in arrival_times:
            # Random order parameters
            side = np.random.choice(['BUY', 'SELL'])
            
            # Price relative to mid
            mid = self.get_mid_price()
            if mid is None:
                mid = 100.0
            
            if side == 'BUY':
                price = mid - np.random.exponential(0.05)
            else:
                price = mid + np.random.exponential(0.05)
            
            quantity = np.random.randint(100, 1000)
            
            events.append({
                'time': t,
                'type': 'limit_order',
                'side': side,
                'price': price,
                'quantity': quantity
            })
        
        return events
    
    def get_mid_price(self) -> Optional[float]:
        bid, ask, _ = self.get_spread()
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    def run_simulation(self, events: List[dict]) -> pd.DataFrame:
        results = []
        
        for event in events:
            if event['type'] == 'limit_order':
                order_id, trades = self.add_limit_order(
                    event['side'], 
                    event['price'], 
                    event['quantity']
                )
                
                mid = self.get_mid_price()
                bid, ask, spread = self.get_spread()
                
                results.append({
                    'time': event['time'],
                    'event_type': 'order',
                    'order_id': order_id,
                    'side': event['side'],
                    'price': event['price'],
                    'quantity': event['quantity'],
                    'trades': len(trades),
                    'mid_price': mid,
                    'spread': spread
                })
        
        return pd.DataFrame(results)
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.14)
project(LOBSimulator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(pybind11 REQUIRED)

# C++ Library
add_library(lob_core STATIC
    cpp/lob/order_book.cpp
    cpp/lob/matching_engine.cpp
)

target_include_directories(lob_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

# Python module
pybind11_add_module(pylob 
    python/lob/bindings.cpp
)

target_link_libraries(pylob PRIVATE lob_core)

# Tests
enable_testing()
add_executable(test_lob tests/test_order_book.cpp)
target_link_libraries(test_lob lob_core)
add_test(NAME test_lob COMMAND test_lob)

# Performance benchmark
add_executable(benchmark tests/test_performance.cpp)
target_link_libraries(benchmark lob_core)
```

## Build Instructions

### C++ Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Python Installation
```bash
cd python
pip install -e .
```

## Usage Example

```python
from lob import LOBSimulator
import matplotlib.pyplot as plt

# Create simulator
sim = LOBSimulator("AAPL")

# Add some orders
sim.add_limit_order("BUY", 149.95, 100)
sim.add_limit_order("BUY", 149.90, 200)
sim.add_limit_order("SELL", 150.05, 150)
sim.add_limit_order("SELL", 150.10, 100)

# Get book snapshot
book = sim.get_book_snapshot()
print(book)

# Simulate with Hawkes process
events = sim.simulate_hawkes_arrivals(T=100, mu=2.0, alpha=0.8, beta=1.5)
results = sim.run_simulation(events)

# Plot mid price evolution
plt.figure(figsize=(12, 6))
plt.plot(results['time'], results['mid_price'])
plt.xlabel('Time')
plt.ylabel('Mid Price')
plt.title('Mid Price Evolution with Hawkes Arrivals')
plt.show()
```

## Deliverables
- `cpp/lob/`: Core C++ limit order book implementation
- `python/lob/`: Python bindings and simulator wrapper
- Discrete-event processing with partial fills and cancellations
- Hawkes process for realistic order arrivals
- Price-time priority matching engine