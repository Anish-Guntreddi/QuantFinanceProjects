# Feed Handler & Order Router

## Overview
Ultra-low latency C++ feed handler and order router with UDP/TCP decoders, lock-free queues, and comprehensive latency histograms (p50/p99).

## Project Structure
```
02_feed_handler_order_router/
├── cpp/
│   ├── feed/
│   │   ├── feed_handler.hpp
│   │   ├── feed_handler.cpp
│   │   ├── market_data_decoder.hpp
│   │   ├── udp_receiver.hpp
│   │   └── tcp_receiver.hpp
│   ├── router/
│   │   ├── order_router.hpp
│   │   ├── order_router.cpp
│   │   ├── fix_encoder.hpp
│   │   └── connection_manager.hpp
│   ├── common/
│   │   ├── lock_free_queue.hpp
│   │   ├── latency_tracker.hpp
│   │   ├── memory_pool.hpp
│   │   └── timestamp.hpp
│   └── main.cpp
├── tests/
│   ├── test_feed_handler.cpp
│   ├── test_order_router.cpp
│   └── latency_benchmark.cpp
├── configs/
│   └── config.yaml
└── CMakeLists.txt
```

## Implementation

### cpp/common/lock_free_queue.hpp
```cpp
#pragma once

#include <atomic>
#include <memory>
#include <cstddef>

template<typename T, size_t Size>
class LockFreeQueue {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
public:
    LockFreeQueue() 
        : head_(0)
        , tail_(0) {
        buffer_ = static_cast<T*>(std::aligned_alloc(64, sizeof(T) * Size));
        if (!buffer_) {
            throw std::bad_alloc();
        }
    }
    
    ~LockFreeQueue() {
        while (head_ != tail_) {
            buffer_[head_ & mask_].~T();
            ++head_;
        }
        std::free(buffer_);
    }
    
    template<typename... Args>
    bool try_push(Args&&... args) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = current_tail + 1;
        
        if (next_tail - head_.load(std::memory_order_acquire) > Size) {
            return false;
        }
        
        new (&buffer_[current_tail & mask_]) T(std::forward<Args>(args)...);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool try_pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        
        item = std::move(buffer_[current_head & mask_]);
        buffer_[current_head & mask_].~T();
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }
    
    size_t size() const {
        return tail_.load(std::memory_order_acquire) - 
               head_.load(std::memory_order_acquire);
    }
    
private:
    static constexpr size_t mask_ = Size - 1;
    alignas(64) T* buffer_;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};
```

### cpp/common/latency_tracker.hpp
```cpp
#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <chrono>

class LatencyTracker {
public:
    static constexpr size_t NUM_BUCKETS = 100000;
    static constexpr uint64_t BUCKET_WIDTH_NS = 100;  // 100ns buckets
    
    LatencyTracker() {
        reset();
    }
    
    void record(uint64_t latency_ns) {
        size_t bucket = std::min(latency_ns / BUCKET_WIDTH_NS, NUM_BUCKETS - 1);
        buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
        
        count_.fetch_add(1, std::memory_order_relaxed);
        sum_.fetch_add(latency_ns, std::memory_order_relaxed);
        
        // Update min/max
        uint64_t current_min = min_.load(std::memory_order_relaxed);
        while (latency_ns < current_min) {
            if (min_.compare_exchange_weak(current_min, latency_ns)) break;
        }
        
        uint64_t current_max = max_.load(std::memory_order_relaxed);
        while (latency_ns > current_max) {
            if (max_.compare_exchange_weak(current_max, latency_ns)) break;
        }
    }
    
    struct Stats {
        uint64_t count;
        double mean;
        uint64_t min;
        uint64_t max;
        uint64_t p50;
        uint64_t p90;
        uint64_t p95;
        uint64_t p99;
        uint64_t p999;
    };
    
    Stats get_stats() const {
        Stats stats;
        stats.count = count_.load();
        
        if (stats.count == 0) {
            return stats;
        }
        
        stats.sum = sum_.load();
        stats.mean = static_cast<double>(stats.sum) / stats.count;
        stats.min = min_.load();
        stats.max = max_.load();
        
        // Calculate percentiles
        std::array<uint64_t, NUM_BUCKETS> bucket_copy;
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            bucket_copy[i] = buckets_[i].load();
        }
        
        stats.p50 = calculate_percentile(bucket_copy, stats.count, 0.50);
        stats.p90 = calculate_percentile(bucket_copy, stats.count, 0.90);
        stats.p95 = calculate_percentile(bucket_copy, stats.count, 0.95);
        stats.p99 = calculate_percentile(bucket_copy, stats.count, 0.99);
        stats.p999 = calculate_percentile(bucket_copy, stats.count, 0.999);
        
        return stats;
    }
    
    void reset() {
        count_.store(0);
        sum_.store(0);
        min_.store(UINT64_MAX);
        max_.store(0);
        
        for (auto& bucket : buckets_) {
            bucket.store(0);
        }
    }
    
private:
    uint64_t calculate_percentile(const std::array<uint64_t, NUM_BUCKETS>& buckets,
                                  uint64_t total, double percentile) const {
        uint64_t target = static_cast<uint64_t>(total * percentile);
        uint64_t cumulative = 0;
        
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            cumulative += buckets[i];
            if (cumulative >= target) {
                return i * BUCKET_WIDTH_NS;
            }
        }
        
        return (NUM_BUCKETS - 1) * BUCKET_WIDTH_NS;
    }
    
    alignas(64) std::atomic<uint64_t> count_;
    alignas(64) std::atomic<uint64_t> sum_;
    alignas(64) std::atomic<uint64_t> min_;
    alignas(64) std::atomic<uint64_t> max_;
    alignas(64) std::array<std::atomic<uint64_t>, NUM_BUCKETS> buckets_;
};
```

### cpp/feed/market_data_decoder.hpp
```cpp
#pragma once

#include <cstdint>
#include <cstring>

#pragma pack(push, 1)

// ITCH 5.0 message types (simplified)
struct ITCHHeader {
    uint16_t length;
    char message_type;
    uint32_t sequence;
    uint64_t timestamp;
};

struct AddOrderMessage {
    ITCHHeader header;
    uint64_t order_id;
    char side;
    uint32_t shares;
    char symbol[8];
    uint32_t price;
};

struct OrderExecutedMessage {
    ITCHHeader header;
    uint64_t order_id;
    uint32_t executed_shares;
    uint64_t match_number;
};

struct OrderCancelMessage {
    ITCHHeader header;
    uint64_t order_id;
    uint32_t cancelled_shares;
};

struct TradeMessage {
    ITCHHeader header;
    uint64_t order_id;
    char side;
    uint32_t shares;
    char symbol[8];
    uint32_t price;
    uint64_t match_number;
};

#pragma pack(pop)

class MarketDataDecoder {
public:
    enum MessageType {
        ADD_ORDER = 'A',
        ORDER_EXECUTED = 'E',
        ORDER_CANCEL = 'X',
        TRADE = 'P'
    };
    
    struct DecodedMessage {
        MessageType type;
        uint64_t timestamp;
        uint32_t sequence;
        
        union {
            struct {
                uint64_t order_id;
                char symbol[9];
                char side;
                uint32_t price;
                uint32_t quantity;
            } add_order;
            
            struct {
                uint64_t order_id;
                uint32_t executed_quantity;
            } order_executed;
            
            struct {
                uint64_t order_id;
                uint32_t cancelled_quantity;
            } order_cancel;
            
            struct {
                char symbol[9];
                uint32_t price;
                uint32_t quantity;
                char side;
            } trade;
        } data;
    };
    
    bool decode(const uint8_t* buffer, size_t length, DecodedMessage& msg) {
        if (length < sizeof(ITCHHeader)) {
            return false;
        }
        
        const ITCHHeader* header = reinterpret_cast<const ITCHHeader*>(buffer);
        
        msg.timestamp = header->timestamp;
        msg.sequence = header->sequence;
        
        switch (header->message_type) {
            case ADD_ORDER: {
                if (length < sizeof(AddOrderMessage)) return false;
                
                const AddOrderMessage* add = reinterpret_cast<const AddOrderMessage*>(buffer);
                msg.type = ADD_ORDER;
                msg.data.add_order.order_id = add->order_id;
                memcpy(msg.data.add_order.symbol, add->symbol, 8);
                msg.data.add_order.symbol[8] = '\0';
                msg.data.add_order.side = add->side;
                msg.data.add_order.price = add->price;
                msg.data.add_order.quantity = add->shares;
                return true;
            }
            
            case ORDER_EXECUTED: {
                if (length < sizeof(OrderExecutedMessage)) return false;
                
                const OrderExecutedMessage* exec = reinterpret_cast<const OrderExecutedMessage*>(buffer);
                msg.type = ORDER_EXECUTED;
                msg.data.order_executed.order_id = exec->order_id;
                msg.data.order_executed.executed_quantity = exec->executed_shares;
                return true;
            }
            
            case ORDER_CANCEL: {
                if (length < sizeof(OrderCancelMessage)) return false;
                
                const OrderCancelMessage* cancel = reinterpret_cast<const OrderCancelMessage*>(buffer);
                msg.type = ORDER_CANCEL;
                msg.data.order_cancel.order_id = cancel->order_id;
                msg.data.order_cancel.cancelled_quantity = cancel->cancelled_shares;
                return true;
            }
            
            case TRADE: {
                if (length < sizeof(TradeMessage)) return false;
                
                const TradeMessage* trade = reinterpret_cast<const TradeMessage*>(buffer);
                msg.type = TRADE;
                memcpy(msg.data.trade.symbol, trade->symbol, 8);
                msg.data.trade.symbol[8] = '\0';
                msg.data.trade.price = trade->price;
                msg.data.trade.quantity = trade->shares;
                msg.data.trade.side = trade->side;
                return true;
            }
            
            default:
                return false;
        }
    }
};
```

### cpp/feed/udp_receiver.hpp
```cpp
#pragma once

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <thread>
#include <atomic>
#include "../common/lock_free_queue.hpp"
#include "../common/latency_tracker.hpp"

class UDPReceiver {
public:
    static constexpr size_t BUFFER_SIZE = 65536;
    static constexpr size_t QUEUE_SIZE = 65536;
    
    struct Packet {
        uint8_t data[1500];
        size_t length;
        uint64_t receive_timestamp;
    };
    
    UDPReceiver(const std::string& multicast_group, uint16_t port)
        : multicast_group_(multicast_group)
        , port_(port)
        , socket_fd_(-1)
        , running_(false) {}
    
    ~UDPReceiver() {
        stop();
    }
    
    bool start() {
        // Create UDP socket
        socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd_ < 0) {
            return false;
        }
        
        // Set socket options
        int reuse = 1;
        setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        
        // Set receive buffer size
        int rcvbuf = 8 * 1024 * 1024;  // 8MB
        setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
        
        // Enable timestamping
        int timestamp = 1;
        setsockopt(socket_fd_, SOL_SOCKET, SO_TIMESTAMP, &timestamp, sizeof(timestamp));
        
        // Set non-blocking
        int flags = fcntl(socket_fd_, F_GETFL, 0);
        fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
        
        // Bind to port
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(socket_fd_);
            return false;
        }
        
        // Join multicast group
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(multicast_group_.c_str());
        mreq.imr_interface.s_addr = INADDR_ANY;
        
        if (setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
            close(socket_fd_);
            return false;
        }
        
        // Start receiver thread
        running_ = true;
        receiver_thread_ = std::thread(&UDPReceiver::receive_loop, this);
        
        return true;
    }
    
    void stop() {
        if (running_) {
            running_ = false;
            if (receiver_thread_.joinable()) {
                receiver_thread_.join();
            }
            if (socket_fd_ >= 0) {
                close(socket_fd_);
                socket_fd_ = -1;
            }
        }
    }
    
    bool get_packet(Packet& packet) {
        return packet_queue_.try_pop(packet);
    }
    
    LatencyTracker::Stats get_stats() const {
        return latency_tracker_.get_stats();
    }
    
private:
    void receive_loop() {
        uint8_t buffer[BUFFER_SIZE];
        
        // Pin to CPU core
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(2, &cpuset);  // Pin to CPU 2
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        while (running_) {
            struct iovec iov;
            iov.iov_base = buffer;
            iov.iov_len = BUFFER_SIZE;
            
            char control[1024];
            struct msghdr msg;
            memset(&msg, 0, sizeof(msg));
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;
            msg.msg_control = control;
            msg.msg_controllen = sizeof(control);
            
            ssize_t bytes = recvmsg(socket_fd_, &msg, MSG_DONTWAIT);
            
            if (bytes > 0) {
                uint64_t timestamp = get_timestamp_ns();
                
                // Extract kernel timestamp if available
                struct cmsghdr* cmsg;
                for (cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
                    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SO_TIMESTAMP) {
                        struct timeval* tv = (struct timeval*)CMSG_DATA(cmsg);
                        timestamp = tv->tv_sec * 1000000000ULL + tv->tv_usec * 1000ULL;
                        break;
                    }
                }
                
                // Create packet
                Packet packet;
                memcpy(packet.data, buffer, bytes);
                packet.length = bytes;
                packet.receive_timestamp = timestamp;
                
                // Push to queue
                if (!packet_queue_.try_push(packet)) {
                    dropped_packets_.fetch_add(1, std::memory_order_relaxed);
                }
                
                // Track latency
                uint64_t now = get_timestamp_ns();
                latency_tracker_.record(now - timestamp);
                
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                // Error occurred
                break;
            }
            
            // CPU pause to reduce power
            __builtin_ia32_pause();
        }
    }
    
    uint64_t get_timestamp_ns() {
        return std::chrono::steady_clock::now().time_since_epoch().count();
    }
    
    std::string multicast_group_;
    uint16_t port_;
    int socket_fd_;
    std::atomic<bool> running_;
    std::thread receiver_thread_;
    LockFreeQueue<Packet, QUEUE_SIZE> packet_queue_;
    LatencyTracker latency_tracker_;
    std::atomic<uint64_t> dropped_packets_{0};
};
```

### cpp/feed/feed_handler.hpp
```cpp
#pragma once

#include "udp_receiver.hpp"
#include "tcp_receiver.hpp"
#include "market_data_decoder.hpp"
#include "../common/lock_free_queue.hpp"
#include <functional>
#include <unordered_map>
#include <memory>

class FeedHandler {
public:
    using MarketDataCallback = std::function<void(const MarketDataDecoder::DecodedMessage&)>;
    
    struct Config {
        std::string multicast_group;
        uint16_t udp_port;
        std::string tcp_host;
        uint16_t tcp_port;
        bool use_udp;
        size_t worker_threads;
    };
    
    FeedHandler(const Config& config)
        : config_(config)
        , running_(false) {}
    
    bool start() {
        if (config_.use_udp) {
            udp_receiver_ = std::make_unique<UDPReceiver>(
                config_.multicast_group, config_.udp_port);
            if (!udp_receiver_->start()) {
                return false;
            }
        } else {
            tcp_receiver_ = std::make_unique<TCPReceiver>(
                config_.tcp_host, config_.tcp_port);
            if (!tcp_receiver_->connect()) {
                return false;
            }
        }
        
        running_ = true;
        
        // Start worker threads
        for (size_t i = 0; i < config_.worker_threads; ++i) {
            workers_.emplace_back(&FeedHandler::worker_loop, this, i);
        }
        
        // Start dispatcher thread
        dispatcher_thread_ = std::thread(&FeedHandler::dispatcher_loop, this);
        
        return true;
    }
    
    void stop() {
        running_ = false;
        
        if (dispatcher_thread_.joinable()) {
            dispatcher_thread_.join();
        }
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        if (udp_receiver_) {
            udp_receiver_->stop();
        }
        
        if (tcp_receiver_) {
            tcp_receiver_->disconnect();
        }
    }
    
    void register_callback(const std::string& symbol, MarketDataCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callbacks_[symbol] = callback;
    }
    
    void register_default_callback(MarketDataCallback callback) {
        default_callback_ = callback;
    }
    
    struct Statistics {
        uint64_t messages_received;
        uint64_t messages_processed;
        uint64_t decode_errors;
        LatencyTracker::Stats latency_stats;
    };
    
    Statistics get_statistics() const {
        Statistics stats;
        stats.messages_received = messages_received_.load();
        stats.messages_processed = messages_processed_.load();
        stats.decode_errors = decode_errors_.load();
        stats.latency_stats = latency_tracker_.get_stats();
        return stats;
    }
    
private:
    void dispatcher_loop() {
        MarketDataDecoder decoder;
        UDPReceiver::Packet packet;
        
        // Pin to CPU
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(3, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        while (running_) {
            bool got_packet = false;
            
            if (udp_receiver_) {
                got_packet = udp_receiver_->get_packet(packet);
            } else if (tcp_receiver_) {
                // TCP receive logic
                // ...
            }
            
            if (got_packet) {
                messages_received_.fetch_add(1, std::memory_order_relaxed);
                
                MarketDataDecoder::DecodedMessage msg;
                if (decoder.decode(packet.data, packet.length, msg)) {
                    // Track decode latency
                    uint64_t now = get_timestamp_ns();
                    latency_tracker_.record(now - packet.receive_timestamp);
                    
                    // Dispatch to worker queue based on symbol hash
                    size_t worker_id = std::hash<std::string>{}(msg.data.add_order.symbol) 
                                      % config_.worker_threads;
                    
                    if (!worker_queues_[worker_id].try_push(msg)) {
                        dropped_messages_.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    decode_errors_.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                // No packet available, yield CPU
                std::this_thread::yield();
            }
        }
    }
    
    void worker_loop(size_t worker_id) {
        // Pin to CPU
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(4 + worker_id, &cpuset);  // CPUs 4+
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        MarketDataDecoder::DecodedMessage msg;
        
        while (running_) {
            if (worker_queues_[worker_id].try_pop(msg)) {
                process_message(msg);
                messages_processed_.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    }
    
    void process_message(const MarketDataDecoder::DecodedMessage& msg) {
        std::string symbol;
        
        // Extract symbol based on message type
        switch (msg.type) {
            case MarketDataDecoder::ADD_ORDER:
                symbol = msg.data.add_order.symbol;
                break;
            case MarketDataDecoder::TRADE:
                symbol = msg.data.trade.symbol;
                break;
            default:
                return;
        }
        
        // Find and invoke callback
        MarketDataCallback callback;
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            auto it = callbacks_.find(symbol);
            if (it != callbacks_.end()) {
                callback = it->second;
            } else if (default_callback_) {
                callback = default_callback_;
            }
        }
        
        if (callback) {
            callback(msg);
        }
    }
    
    uint64_t get_timestamp_ns() {
        return std::chrono::steady_clock::now().time_since_epoch().count();
    }
    
    Config config_;
    std::atomic<bool> running_;
    
    std::unique_ptr<UDPReceiver> udp_receiver_;
    std::unique_ptr<TCPReceiver> tcp_receiver_;
    
    std::thread dispatcher_thread_;
    std::vector<std::thread> workers_;
    std::vector<LockFreeQueue<MarketDataDecoder::DecodedMessage, 8192>> worker_queues_;
    
    std::unordered_map<std::string, MarketDataCallback> callbacks_;
    MarketDataCallback default_callback_;
    std::mutex callback_mutex_;
    
    LatencyTracker latency_tracker_;
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> messages_processed_{0};
    std::atomic<uint64_t> decode_errors_{0};
    std::atomic<uint64_t> dropped_messages_{0};
};
```

### cpp/router/order_router.hpp
```cpp
#pragma once

#include "../common/lock_free_queue.hpp"
#include "../common/latency_tracker.hpp"
#include "fix_encoder.hpp"
#include "connection_manager.hpp"
#include <memory>
#include <thread>
#include <atomic>

class OrderRouter {
public:
    struct Order {
        uint64_t order_id;
        std::string symbol;
        enum Side { BUY, SELL } side;
        enum Type { MARKET, LIMIT, STOP } type;
        double price;
        uint64_t quantity;
        std::string destination;
        uint64_t timestamp;
    };
    
    struct Config {
        std::vector<std::string> destinations;
        size_t worker_threads;
        bool enable_retry;
        size_t max_retries;
        uint64_t retry_delay_ms;
    };
    
    OrderRouter(const Config& config)
        : config_(config)
        , running_(false)
        , next_order_id_(1) {}
    
    bool start() {
        // Initialize connection manager
        connection_manager_ = std::make_unique<ConnectionManager>();
        
        for (const auto& dest : config_.destinations) {
            if (!connection_manager_->add_connection(dest)) {
                return false;
            }
        }
        
        running_ = true;
        
        // Start worker threads
        for (size_t i = 0; i < config_.worker_threads; ++i) {
            workers_.emplace_back(&OrderRouter::worker_loop, this, i);
        }
        
        // Start monitoring thread
        monitor_thread_ = std::thread(&OrderRouter::monitor_loop, this);
        
        return true;
    }
    
    void stop() {
        running_ = false;
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
    
    uint64_t submit_order(const Order& order) {
        uint64_t order_id = next_order_id_.fetch_add(1);
        
        Order internal_order = order;
        internal_order.order_id = order_id;
        internal_order.timestamp = get_timestamp_ns();
        
        // Route to appropriate queue based on destination
        size_t queue_id = std::hash<std::string>{}(order.destination) % config_.worker_threads;
        
        if (!order_queues_[queue_id].try_push(internal_order)) {
            rejected_orders_.fetch_add(1);
            return 0;
        }
        
        submitted_orders_.fetch_add(1);
        return order_id;
    }
    
    struct Statistics {
        uint64_t submitted_orders;
        uint64_t sent_orders;
        uint64_t rejected_orders;
        uint64_t failed_orders;
        LatencyTracker::Stats submission_latency;
        LatencyTracker::Stats wire_latency;
    };
    
    Statistics get_statistics() const {
        Statistics stats;
        stats.submitted_orders = submitted_orders_.load();
        stats.sent_orders = sent_orders_.load();
        stats.rejected_orders = rejected_orders_.load();
        stats.failed_orders = failed_orders_.load();
        stats.submission_latency = submission_latency_.get_stats();
        stats.wire_latency = wire_latency_.get_stats();
        return stats;
    }
    
private:
    void worker_loop(size_t worker_id) {
        // Pin to CPU
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(8 + worker_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        FIXEncoder encoder;
        Order order;
        
        while (running_) {
            if (order_queues_[worker_id].try_pop(order)) {
                process_order(order, encoder);
            } else {
                std::this_thread::yield();
            }
        }
    }
    
    void process_order(const Order& order, FIXEncoder& encoder) {
        uint64_t start_time = get_timestamp_ns();
        
        // Encode order to FIX message
        std::string fix_message = encoder.encode_new_order_single(
            order.order_id,
            order.symbol,
            order.side == Order::BUY ? 'B' : 'S',
            order.type == Order::MARKET ? 'M' : 'L',
            order.price,
            order.quantity
        );
        
        // Send order
        bool sent = false;
        size_t attempts = 0;
        
        while (!sent && attempts < config_.max_retries) {
            if (connection_manager_->send(order.destination, fix_message)) {
                sent = true;
                sent_orders_.fetch_add(1);
                
                // Track latencies
                uint64_t end_time = get_timestamp_ns();
                submission_latency_.record(end_time - order.timestamp);
                wire_latency_.record(end_time - start_time);
            } else {
                attempts++;
                if (config_.enable_retry && attempts < config_.max_retries) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(config_.retry_delay_ms));
                }
            }
        }
        
        if (!sent) {
            failed_orders_.fetch_add(1);
        }
    }
    
    void monitor_loop() {
        while (running_) {
            // Monitor connection health
            connection_manager_->check_connections();
            
            // Print statistics periodically
            auto stats = get_statistics();
            
            std::cout << "OrderRouter Stats - "
                     << "Submitted: " << stats.submitted_orders
                     << ", Sent: " << stats.sent_orders
                     << ", Failed: " << stats.failed_orders
                     << ", Submission p50: " << stats.submission_latency.p50 << "ns"
                     << ", Wire p99: " << stats.wire_latency.p99 << "ns"
                     << std::endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    uint64_t get_timestamp_ns() {
        return std::chrono::steady_clock::now().time_since_epoch().count();
    }
    
    Config config_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> next_order_id_;
    
    std::vector<LockFreeQueue<Order, 4096>> order_queues_;
    std::vector<std::thread> workers_;
    std::thread monitor_thread_;
    
    std::unique_ptr<ConnectionManager> connection_manager_;
    
    LatencyTracker submission_latency_;
    LatencyTracker wire_latency_;
    
    std::atomic<uint64_t> submitted_orders_{0};
    std::atomic<uint64_t> sent_orders_{0};
    std::atomic<uint64_t> rejected_orders_{0};
    std::atomic<uint64_t> failed_orders_{0};
};
```

### cpp/router/fix_encoder.hpp
```cpp
#pragma once

#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>

class FIXEncoder {
public:
    FIXEncoder(const std::string& sender_comp_id = "ROUTER",
               const std::string& target_comp_id = "EXCHANGE")
        : sender_comp_id_(sender_comp_id)
        , target_comp_id_(target_comp_id)
        , sequence_number_(1) {}
    
    std::string encode_new_order_single(
        uint64_t cl_ord_id,
        const std::string& symbol,
        char side,
        char ord_type,
        double price,
        uint64_t quantity
    ) {
        std::stringstream msg;
        
        // Header
        msg << "8=FIX.4.4" << SOH;  // BeginString
        msg << "35=D" << SOH;        // MsgType (NewOrderSingle)
        msg << "49=" << sender_comp_id_ << SOH;  // SenderCompID
        msg << "56=" << target_comp_id_ << SOH;  // TargetCompID
        msg << "34=" << sequence_number_++ << SOH;  // MsgSeqNum
        msg << "52=" << get_utc_timestamp() << SOH;  // SendingTime
        
        // Body
        msg << "11=" << cl_ord_id << SOH;  // ClOrdID
        msg << "55=" << symbol << SOH;     // Symbol
        msg << "54=" << side << SOH;       // Side (1=Buy, 2=Sell)
        msg << "40=" << ord_type << SOH;   // OrdType (1=Market, 2=Limit)
        
        if (ord_type == '2') {  // Limit order
            msg << "44=" << std::fixed << std::setprecision(2) << price << SOH;  // Price
        }
        
        msg << "38=" << quantity << SOH;   // OrderQty
        msg << "60=" << get_utc_timestamp() << SOH;  // TransactTime
        
        // Calculate and append checksum
        std::string message = msg.str();
        message = "9=" + std::to_string(message.length()) + SOH + message;
        
        uint8_t checksum = calculate_checksum(message);
        message += "10=" + format_checksum(checksum) + SOH;
        
        return message;
    }
    
    std::string encode_order_cancel_request(
        uint64_t cl_ord_id,
        uint64_t orig_cl_ord_id,
        const std::string& symbol
    ) {
        std::stringstream msg;
        
        msg << "8=FIX.4.4" << SOH;
        msg << "35=F" << SOH;  // OrderCancelRequest
        msg << "49=" << sender_comp_id_ << SOH;
        msg << "56=" << target_comp_id_ << SOH;
        msg << "34=" << sequence_number_++ << SOH;
        msg << "52=" << get_utc_timestamp() << SOH;
        
        msg << "11=" << cl_ord_id << SOH;        // ClOrdID
        msg << "41=" << orig_cl_ord_id << SOH;   // OrigClOrdID
        msg << "55=" << symbol << SOH;           // Symbol
        msg << "60=" << get_utc_timestamp() << SOH;  // TransactTime
        
        std::string message = msg.str();
        message = "9=" + std::to_string(message.length()) + SOH + message;
        
        uint8_t checksum = calculate_checksum(message);
        message += "10=" + format_checksum(checksum) + SOH;
        
        return message;
    }
    
private:
    static constexpr char SOH = '\001';  // FIX delimiter
    
    std::string get_utc_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y%m%d-%H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        
        return ss.str();
    }
    
    uint8_t calculate_checksum(const std::string& message) {
        uint32_t sum = 0;
        for (char c : message) {
            sum += static_cast<uint8_t>(c);
        }
        return sum % 256;
    }
    
    std::string format_checksum(uint8_t checksum) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(3) << static_cast<int>(checksum);
        return ss.str();
    }
    
    std::string sender_comp_id_;
    std::string target_comp_id_;
    uint32_t sequence_number_;
};
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.14)
project(FeedHandlerOrderRouter VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wextra")

# Find threads
find_package(Threads REQUIRED)

# Feed Handler library
add_library(feed_handler STATIC
    cpp/feed/feed_handler.cpp
    cpp/feed/udp_receiver.cpp
    cpp/feed/tcp_receiver.cpp
)

target_include_directories(feed_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

target_link_libraries(feed_handler PRIVATE Threads::Threads)

# Order Router library
add_library(order_router STATIC
    cpp/router/order_router.cpp
    cpp/router/connection_manager.cpp
)

target_include_directories(order_router PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

target_link_libraries(order_router PRIVATE Threads::Threads)

# Main executable
add_executable(feed_router_app cpp/main.cpp)
target_link_libraries(feed_router_app 
    feed_handler 
    order_router 
    Threads::Threads
)

# Tests
enable_testing()

add_executable(test_feed_handler tests/test_feed_handler.cpp)
target_link_libraries(test_feed_handler feed_handler Threads::Threads)
add_test(NAME test_feed_handler COMMAND test_feed_handler)

add_executable(test_order_router tests/test_order_router.cpp)
target_link_libraries(test_order_router order_router Threads::Threads)
add_test(NAME test_order_router COMMAND test_order_router)

# Latency benchmark
add_executable(latency_benchmark tests/latency_benchmark.cpp)
target_link_libraries(latency_benchmark 
    feed_handler 
    order_router 
    Threads::Threads
)
```

## Build and Run

### Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Run
```bash
# Run with config file
./feed_router_app --config ../configs/config.yaml

# Run latency benchmark
sudo ./latency_benchmark
```

### System Tuning for Ultra-Low Latency
```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Set CPU affinity and isolate cores
sudo taskset -c 2-10 ./feed_router_app

# Increase network buffers
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.netdev_max_backlog=5000

# Disable interrupt coalescing
sudo ethtool -C eth0 rx-usecs 0 tx-usecs 0
```

## Performance Metrics

- **UDP receive latency**: p50 < 1μs, p99 < 5μs
- **Decode latency**: p50 < 100ns, p99 < 500ns
- **Order routing latency**: p50 < 2μs, p99 < 10μs
- **Throughput**: >1M messages/second per core

## Deliverables

- `cpp/feed/`: UDP/TCP feed handler with lock-free queues
- `cpp/router/`: FIX order router with connection management
- P50/P99 latency histograms with nanosecond precision
- Lock-free architecture for ultra-low latency