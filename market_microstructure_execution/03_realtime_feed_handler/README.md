# Real-Time Feed Handler & Order Router

## Project Overview
A high-performance, low-latency market data feed handler and order router implemented in C++ with lock-free data structures, NUMA optimization, and microsecond-level latency monitoring for handling UDP/TCP market data feeds and order routing.

## Implementation Guide

### Phase 1: Project Setup & Architecture

#### 1.1 Environment Setup
```bash
# Create project structure
mkdir -p cpp/{feed,router,common,utils}
mkdir -p perf tests configs scripts
mkdir -p build/{debug,release}

# Install dependencies
# Ubuntu/Debian
sudo apt-get install cmake g++ libboost-all-dev libtbb-dev
sudo apt-get install linux-tools-common linux-tools-generic
sudo apt-get install libnuma-dev libpcap-dev
sudo apt-get install google-perftools libgoogle-perftools-dev

# macOS
brew install cmake boost tbb google-perftools
brew install libpcap numa

# Build tools
pip install conan  # C++ package manager
```

#### 1.2 Project Structure
```
03_realtime_feed_handler/
├── cpp/
│   ├── feed/
│   │   ├── include/
│   │   │   ├── feed_handler.hpp        # Main feed handler
│   │   │   ├── udp_receiver.hpp        # UDP packet receiver
│   │   │   ├── tcp_receiver.hpp        # TCP stream receiver
│   │   │   ├── decoder.hpp             # Protocol decoder
│   │   │   ├── parser.hpp              # Message parser
│   │   │   └── normalizer.hpp          # Data normalizer
│   │   └── src/
│   │       ├── feed_handler.cpp
│   │       ├── udp_receiver.cpp
│   │       ├── tcp_receiver.cpp
│   │       ├── decoder.cpp
│   │       └── parser.cpp
│   ├── router/
│   │   ├── include/
│   │   │   ├── order_router.hpp        # Order router
│   │   │   ├── smart_router.hpp        # Smart order routing
│   │   │   ├── venue_connector.hpp     # Venue connections
│   │   │   ├── order_manager.hpp       # Order state management
│   │   │   └── risk_checker.hpp        # Pre-trade risk
│   │   └── src/
│   │       ├── order_router.cpp
│   │       ├── smart_router.cpp
│   │       ├── venue_connector.cpp
│   │       └── order_manager.cpp
│   ├── common/
│   │   ├── include/
│   │   │   ├── types.hpp               # Common types
│   │   │   ├── ring_buffer.hpp         # Lock-free ring buffer
│   │   │   ├── spsc_queue.hpp         # Single producer/consumer queue
│   │   │   ├── memory_pool.hpp        # Memory pool
│   │   │   ├── timestamp.hpp          # High-res timestamps
│   │   │   └── protocol.hpp           # Protocol definitions
│   │   └── src/
│   │       ├── memory_pool.cpp
│   │       └── timestamp.cpp
│   ├── utils/
│   │   ├── include/
│   │   │   ├── cpu_affinity.hpp       # CPU pinning
│   │   │   ├── numa_allocator.hpp     # NUMA-aware allocation
│   │   │   ├── statistics.hpp         # Performance stats
│   │   │   └── logger.hpp             # Low-latency logging
│   │   └── src/
│   │       ├── cpu_affinity.cpp
│   │       ├── numa_allocator.cpp
│   │       └── statistics.cpp
│   └── main.cpp                       # Main application
├── perf/
│   ├── latency_histogram.cpp          # Latency measurement
│   ├── throughput_test.cpp            # Throughput testing
│   ├── burst_test.cpp                 # Burst handling test
│   └── benchmark.cpp                  # Performance benchmarks
├── tests/
│   ├── test_ring_buffer.cpp
│   ├── test_decoder.cpp
│   ├── test_router.cpp
│   └── test_integration.cpp
├── configs/
│   ├── feed_config.yml                # Feed configuration
│   ├── routing_config.yml             # Routing rules
│   └── performance.yml                # Performance tuning
├── scripts/
│   ├── setup_hugepages.sh             # HugePages setup
│   ├── tune_network.sh                # Network tuning
│   └── monitor.py                     # Real-time monitoring
├── CMakeLists.txt
└── README.md
```

### Phase 2: Core Data Structures

#### 2.1 Common Types (cpp/common/include/types.hpp)
```cpp
#pragma once

#include <cstdint>
#include <chrono>
#include <array>
#include <string_view>

namespace feed {

// Timestamp with nanosecond precision
using Timestamp = std::chrono::nanoseconds::rep;

// Market data types
using Symbol = uint32_t;
using Price = int64_t;      // Fixed point price (6 decimals)
using Quantity = uint64_t;
using OrderId = uint64_t;
using SequenceNumber = uint64_t;

enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class MessageType : uint8_t {
    ADD_ORDER = 'A',
    DELETE_ORDER = 'D',
    MODIFY_ORDER = 'M',
    EXECUTE_ORDER = 'E',
    TRADE = 'T',
    QUOTE = 'Q',
    IMBALANCE = 'I',
    STATUS = 'S'
};

// Cacheline-aligned structures for performance
static constexpr size_t CACHE_LINE_SIZE = 64;

struct alignas(CACHE_LINE_SIZE) MarketDataMessage {
    MessageType type;
    Symbol symbol;
    Timestamp exchange_timestamp;
    Timestamp receive_timestamp;
    SequenceNumber sequence;
    
    union {
        struct {
            OrderId order_id;
            Side side;
            Price price;
            Quantity quantity;
        } order;
        
        struct {
            Price bid_price;
            Price ask_price;
            Quantity bid_size;
            Quantity ask_size;
        } quote;
        
        struct {
            Price price;
            Quantity quantity;
            OrderId buy_order_id;
            OrderId sell_order_id;
        } trade;
    };
    
    // Padding to cache line
    char padding[CACHE_LINE_SIZE - 48];
};

struct alignas(CACHE_LINE_SIZE) OrderMessage {
    OrderId id;
    Symbol symbol;
    Side side;
    Price price;
    Quantity quantity;
    Timestamp timestamp;
    uint8_t venue_id;
    uint8_t order_type;
    
    char padding[CACHE_LINE_SIZE - 40];
};

// Performance statistics
struct LatencyStats {
    uint64_t count = 0;
    uint64_t sum_ns = 0;
    uint64_t min_ns = UINT64_MAX;
    uint64_t max_ns = 0;
    uint64_t p50_ns = 0;
    uint64_t p99_ns = 0;
    uint64_t p999_ns = 0;
    
    void update(uint64_t latency_ns) {
        count++;
        sum_ns += latency_ns;
        min_ns = std::min(min_ns, latency_ns);
        max_ns = std::max(max_ns, latency_ns);
    }
    
    double avg_ns() const {
        return count > 0 ? static_cast<double>(sum_ns) / count : 0;
    }
};

} // namespace feed
```

#### 2.2 Lock-Free Ring Buffer (cpp/common/include/ring_buffer.hpp)
```cpp
#pragma once

#include <atomic>
#include <vector>
#include <cstring>
#include <optional>
#include "types.hpp"

namespace feed {

template<typename T>
class alignas(CACHE_LINE_SIZE) RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(next_power_of_2(capacity))
        , mask_(capacity_ - 1)
        , buffer_(capacity_)
        , head_(0)
        , tail_(0) {
        
        // Pre-fault pages
        std::memset(buffer_.data(), 0, capacity_ * sizeof(T));
    }
    
    bool try_push(const T& item) {
        const auto current_head = head_.load(std::memory_order_relaxed);
        const auto next_head = (current_head + 1) & mask_;
        
        // Check if full
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        
        buffer_[current_head] = item;
        
        // Memory barrier to ensure write completes before updating head
        std::atomic_thread_fence(std::memory_order_release);
        head_.store(next_head, std::memory_order_release);
        
        return true;
    }
    
    std::optional<T> try_pop() {
        const auto current_tail = tail_.load(std::memory_order_relaxed);
        
        // Check if empty
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        
        T item = buffer_[current_tail];
        
        // Memory barrier before updating tail
        std::atomic_thread_fence(std::memory_order_release);
        tail_.store((current_tail + 1) & mask_, std::memory_order_release);
        
        return item;
    }
    
    size_t size() const {
        const auto head = head_.load(std::memory_order_acquire);
        const auto tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & mask_;
    }
    
    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
    
private:
    static size_t next_power_of_2(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        n++;
        return n;
    }
    
    const size_t capacity_;
    const size_t mask_;
    std::vector<T> buffer_;
    
    // Separate cache lines for head and tail to avoid false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
};

} // namespace feed
```

#### 2.3 SPSC Queue (cpp/common/include/spsc_queue.hpp)
```cpp
#pragma once

#include <atomic>
#include <memory>
#include <new>
#include "types.hpp"

namespace feed {

template<typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(size_t capacity)
        : capacity_(capacity)
        , buffer_(static_cast<T*>(std::aligned_alloc(CACHE_LINE_SIZE, 
                                                     capacity * sizeof(T))))
        , head_(0)
        , cached_head_(0)
        , tail_(0)
        , cached_tail_(0) {
        
        if (!buffer_) {
            throw std::bad_alloc();
        }
        
        // Initialize buffer
        for (size_t i = 0; i < capacity_; ++i) {
            new (&buffer_[i]) T();
        }
    }
    
    ~SPSCQueue() {
        // Destroy remaining elements
        while (tail_ != head_) {
            buffer_[tail_].~T();
            tail_ = (tail_ + 1) % capacity_;
        }
        std::free(buffer_);
    }
    
    // Producer interface
    bool push(const T& item) {
        const size_t next_head = (head_ + 1) % capacity_;
        
        // Check cached tail first (avoid cache miss)
        if (next_head == cached_tail_) {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            if (next_head == cached_tail_) {
                return false;  // Queue full
            }
        }
        
        buffer_[head_] = item;
        head_.store(next_head, std::memory_order_release);
        
        return true;
    }
    
    // Consumer interface
    bool pop(T& item) {
        // Check cached head first
        if (tail_ == cached_head_) {
            cached_head_ = head_.load(std::memory_order_acquire);
            if (tail_ == cached_head_) {
                return false;  // Queue empty
            }
        }
        
        item = buffer_[tail_];
        tail_.store((tail_ + 1) % capacity_, std::memory_order_release);
        
        return true;
    }
    
    // Batch operations for efficiency
    size_t push_batch(const T* items, size_t count) {
        size_t pushed = 0;
        
        while (pushed < count && push(items[pushed])) {
            pushed++;
        }
        
        return pushed;
    }
    
    size_t pop_batch(T* items, size_t max_count) {
        size_t popped = 0;
        
        while (popped < max_count && pop(items[popped])) {
            popped++;
        }
        
        return popped;
    }
    
private:
    const size_t capacity_;
    T* buffer_;
    
    // Producer variables (separate cache line)
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    size_t cached_tail_;
    
    // Consumer variables (separate cache line)
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
    size_t cached_head_;
};

} // namespace feed
```

### Phase 3: Feed Handler Implementation

#### 3.1 UDP Receiver (cpp/feed/include/udp_receiver.hpp)
```cpp
#pragma once

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <vector>
#include <thread>
#include <atomic>
#include "ring_buffer.hpp"
#include "types.hpp"

namespace feed {

class UDPReceiver {
public:
    struct Config {
        std::string multicast_group;
        uint16_t port;
        std::string interface;
        size_t recv_buffer_size = 8 * 1024 * 1024;  // 8MB
        size_t ring_buffer_size = 65536;
        int cpu_affinity = -1;  // CPU to pin thread to
        bool use_busy_poll = true;
        uint32_t busy_poll_usecs = 50;
    };
    
    explicit UDPReceiver(const Config& config);
    ~UDPReceiver();
    
    void start();
    void stop();
    
    // Get received packets
    RingBuffer<std::vector<uint8_t>>& get_packet_queue() {
        return packet_queue_;
    }
    
    LatencyStats get_stats() const {
        return stats_;
    }
    
private:
    void receive_loop();
    void setup_socket();
    void join_multicast_group();
    void set_socket_options();
    
    Config config_;
    int socket_fd_;
    std::thread receiver_thread_;
    std::atomic<bool> running_;
    
    RingBuffer<std::vector<uint8_t>> packet_queue_;
    
    // Statistics
    mutable LatencyStats stats_;
    uint64_t packets_received_ = 0;
    uint64_t packets_dropped_ = 0;
    uint64_t bytes_received_ = 0;
    
    // Receive buffer
    static constexpr size_t MAX_PACKET_SIZE = 9000;  // Jumbo frames
    alignas(CACHE_LINE_SIZE) uint8_t recv_buffer_[MAX_PACKET_SIZE];
};

// Implementation
UDPReceiver::UDPReceiver(const Config& config)
    : config_(config)
    , socket_fd_(-1)
    , running_(false)
    , packet_queue_(config.ring_buffer_size) {
}

void UDPReceiver::setup_socket() {
    // Create UDP socket
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }
    
    // Set socket options
    set_socket_options();
    
    // Bind to port
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.port);
    addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        throw std::runtime_error("Failed to bind socket");
    }
    
    // Join multicast group
    join_multicast_group();
}

void UDPReceiver::set_socket_options() {
    int yes = 1;
    
    // Allow address reuse
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    
    // Set receive buffer size
    int rcvbuf = config_.recv_buffer_size;
    setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    
    // Enable timestamping
    int timestamp_flags = SOF_TIMESTAMPING_RX_HARDWARE | 
                          SOF_TIMESTAMPING_RX_SOFTWARE |
                          SOF_TIMESTAMPING_SOFTWARE;
    setsockopt(socket_fd_, SOL_SOCKET, SO_TIMESTAMPING, 
               &timestamp_flags, sizeof(timestamp_flags));
    
    // Set busy polling if enabled
    if (config_.use_busy_poll) {
        setsockopt(socket_fd_, SOL_SOCKET, SO_BUSY_POLL,
                  &config_.busy_poll_usecs, sizeof(config_.busy_poll_usecs));
    }
    
    // Disable Nagle's algorithm
    int no_delay = 1;
    setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
}

void UDPReceiver::join_multicast_group() {
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(config_.multicast_group.c_str());
    
    if (!config_.interface.empty()) {
        mreq.imr_interface.s_addr = inet_addr(config_.interface.c_str());
    } else {
        mreq.imr_interface.s_addr = INADDR_ANY;
    }
    
    if (setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, 
                   &mreq, sizeof(mreq)) < 0) {
        throw std::runtime_error("Failed to join multicast group");
    }
}

void UDPReceiver::receive_loop() {
    // Set CPU affinity if configured
    if (config_.cpu_affinity >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(config_.cpu_affinity, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
    
    // Set thread priority
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    struct sockaddr_in sender_addr;
    socklen_t sender_len = sizeof(sender_addr);
    
    // Message control for timestamps
    struct msghdr msg;
    struct iovec iov;
    char control[256];
    
    iov.iov_base = recv_buffer_;
    iov.iov_len = MAX_PACKET_SIZE;
    
    msg.msg_name = &sender_addr;
    msg.msg_namelen = sender_len;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    
    while (running_.load(std::memory_order_acquire)) {
        // Receive packet
        ssize_t bytes = recvmsg(socket_fd_, &msg, 0);
        
        if (bytes < 0) {
            if (errno == EINTR) continue;
            packets_dropped_++;
            continue;
        }
        
        auto receive_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        // Extract hardware timestamp if available
        struct cmsghdr* cmsg;
        for (cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && 
                cmsg->cmsg_type == SO_TIMESTAMPING) {
                struct timespec* ts = (struct timespec*)CMSG_DATA(cmsg);
                // Use hardware timestamp if available
                if (ts[2].tv_sec || ts[2].tv_nsec) {
                    receive_time = ts[2].tv_sec * 1000000000LL + ts[2].tv_nsec;
                }
            }
        }
        
        // Copy to packet buffer
        std::vector<uint8_t> packet(recv_buffer_, recv_buffer_ + bytes);
        
        // Try to push to ring buffer
        if (!packet_queue_.try_push(std::move(packet))) {
            packets_dropped_++;
        } else {
            packets_received_++;
            bytes_received_ += bytes;
        }
        
        // Update statistics
        // (In production, would batch statistics updates)
    }
}

} // namespace feed
```

#### 3.2 Protocol Decoder (cpp/feed/include/decoder.hpp)
```cpp
#pragma once

#include <cstring>
#include <optional>
#include "types.hpp"

namespace feed {

// Mock protocol similar to ITCH/OUCH
#pragma pack(push, 1)
struct ProtocolHeader {
    uint16_t length;
    uint8_t message_type;
    uint32_t sequence;
    uint64_t timestamp;  // Exchange timestamp in nanoseconds
};

struct AddOrderMessage {
    ProtocolHeader header;
    uint64_t order_id;
    uint32_t symbol;
    uint8_t side;
    uint32_t quantity;
    int32_t price;  // Price in cents
};

struct ExecuteOrderMessage {
    ProtocolHeader header;
    uint64_t order_id;
    uint32_t executed_quantity;
    uint64_t match_id;
};

struct TradeMessage {
    ProtocolHeader header;
    uint32_t symbol;
    int32_t price;
    uint32_t quantity;
    uint64_t match_id;
    uint8_t trade_flags;
};
#pragma pack(pop)

class Decoder {
public:
    Decoder() = default;
    
    std::optional<MarketDataMessage> decode(const uint8_t* buffer, size_t length) {
        if (length < sizeof(ProtocolHeader)) {
            return std::nullopt;
        }
        
        const auto* header = reinterpret_cast<const ProtocolHeader*>(buffer);
        
        // Validate message
        if (header->length != length) {
            return std::nullopt;
        }
        
        MarketDataMessage msg;
        msg.type = static_cast<MessageType>(header->message_type);
        msg.sequence = header->sequence;
        msg.exchange_timestamp = header->timestamp;
        msg.receive_timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        switch (msg.type) {
            case MessageType::ADD_ORDER:
                return decode_add_order(buffer, length);
                
            case MessageType::EXECUTE_ORDER:
                return decode_execute_order(buffer, length);
                
            case MessageType::TRADE:
                return decode_trade(buffer, length);
                
            default:
                return std::nullopt;
        }
    }
    
private:
    std::optional<MarketDataMessage> decode_add_order(const uint8_t* buffer, size_t length) {
        if (length < sizeof(AddOrderMessage)) {
            return std::nullopt;
        }
        
        const auto* add = reinterpret_cast<const AddOrderMessage*>(buffer);
        
        MarketDataMessage msg;
        msg.type = MessageType::ADD_ORDER;
        msg.symbol = add->symbol;
        msg.sequence = add->header.sequence;
        msg.exchange_timestamp = add->header.timestamp;
        msg.receive_timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        msg.order.order_id = add->order_id;
        msg.order.side = static_cast<Side>(add->side);
        msg.order.price = add->price;
        msg.order.quantity = add->quantity;
        
        return msg;
    }
    
    std::optional<MarketDataMessage> decode_trade(const uint8_t* buffer, size_t length) {
        if (length < sizeof(TradeMessage)) {
            return std::nullopt;
        }
        
        const auto* trade = reinterpret_cast<const TradeMessage*>(buffer);
        
        MarketDataMessage msg;
        msg.type = MessageType::TRADE;
        msg.symbol = trade->symbol;
        msg.sequence = trade->header.sequence;
        msg.exchange_timestamp = trade->header.timestamp;
        msg.receive_timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        msg.trade.price = trade->price;
        msg.trade.quantity = trade->quantity;
        
        return msg;
    }
    
    std::optional<MarketDataMessage> decode_execute_order(const uint8_t* buffer, size_t length) {
        // Similar implementation
        return std::nullopt;
    }
};

} // namespace feed
```

#### 3.3 Feed Handler (cpp/feed/include/feed_handler.hpp)
```cpp
#pragma once

#include <memory>
#include <vector>
#include <thread>
#include <functional>
#include "udp_receiver.hpp"
#include "decoder.hpp"
#include "ring_buffer.hpp"
#include "spsc_queue.hpp"

namespace feed {

class FeedHandler {
public:
    using MessageCallback = std::function<void(const MarketDataMessage&)>;
    
    struct Config {
        UDPReceiver::Config udp_config;
        size_t decoder_threads = 2;
        size_t output_queue_size = 65536;
        bool enable_conflation = false;
        uint32_t conflation_interval_us = 100;
    };
    
    explicit FeedHandler(const Config& config);
    ~FeedHandler();
    
    void start();
    void stop();
    
    void set_callback(MessageCallback callback) {
        callback_ = callback;
    }
    
    // Subscribe to specific symbols
    void subscribe(Symbol symbol);
    void unsubscribe(Symbol symbol);
    
    // Get statistics
    struct Stats {
        uint64_t packets_received;
        uint64_t messages_decoded;
        uint64_t messages_processed;
        uint64_t decode_errors;
        LatencyStats wire_to_decode;
        LatencyStats decode_to_callback;
        LatencyStats total_latency;
    };
    
    Stats get_stats() const;
    
private:
    void decoder_thread(int thread_id);
    void processor_thread();
    void handle_message(const MarketDataMessage& msg);
    
    Config config_;
    std::unique_ptr<UDPReceiver> udp_receiver_;
    std::vector<std::thread> decoder_threads_;
    std::thread processor_thread_;
    
    Decoder decoder_;
    MessageCallback callback_;
    
    // Message queues
    SPSCQueue<MarketDataMessage> decoded_queue_;
    
    // Subscription management
    std::unordered_set<Symbol> subscriptions_;
    std::mutex subscription_mutex_;
    
    // Statistics
    mutable Stats stats_;
    std::atomic<bool> running_;
};

void FeedHandler::decoder_thread(int thread_id) {
    // Pin to CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id + 2, &cpuset);  // Skip cores 0-1 for system
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    
    auto& packet_queue = udp_receiver_->get_packet_queue();
    
    while (running_.load(std::memory_order_acquire)) {
        auto packet_opt = packet_queue.try_pop();
        
        if (!packet_opt) {
            // Busy wait or yield
            if (config_.udp_config.use_busy_poll) {
                __builtin_ia32_pause();  // CPU pause instruction
            } else {
                std::this_thread::yield();
            }
            continue;
        }
        
        const auto& packet = *packet_opt;
        auto decode_start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        // Decode message
        auto msg_opt = decoder_.decode(packet.data(), packet.size());
        
        if (!msg_opt) {
            stats_.decode_errors++;
            continue;
        }
        
        auto& msg = *msg_opt;
        
        // Calculate wire-to-decode latency
        auto wire_latency = decode_start - msg.exchange_timestamp;
        stats_.wire_to_decode.update(wire_latency);
        
        // Push to decoded queue
        if (!decoded_queue_.push(msg)) {
            // Queue full - handle overflow
            continue;
        }
        
        stats_.messages_decoded++;
    }
}

void FeedHandler::processor_thread() {
    // Pin to dedicated CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    
    // Batch processing for efficiency
    constexpr size_t BATCH_SIZE = 64;
    MarketDataMessage messages[BATCH_SIZE];
    
    while (running_.load(std::memory_order_acquire)) {
        size_t count = decoded_queue_.pop_batch(messages, BATCH_SIZE);
        
        if (count == 0) {
            if (config_.udp_config.use_busy_poll) {
                __builtin_ia32_pause();
            } else {
                std::this_thread::yield();
            }
            continue;
        }
        
        for (size_t i = 0; i < count; ++i) {
            handle_message(messages[i]);
        }
    }
}

void FeedHandler::handle_message(const MarketDataMessage& msg) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Check subscription
    {
        std::lock_guard<std::mutex> lock(subscription_mutex_);
        if (!subscriptions_.empty() && 
            subscriptions_.find(msg.symbol) == subscriptions_.end()) {
            return;
        }
    }
    
    // Calculate latencies
    auto decode_latency = now - msg.receive_timestamp;
    auto total_latency = now - msg.exchange_timestamp;
    
    stats_.decode_to_callback.update(decode_latency);
    stats_.total_latency.update(total_latency);
    
    // Call user callback
    if (callback_) {
        callback_(msg);
    }
    
    stats_.messages_processed++;
}

} // namespace feed
```

### Phase 4: Order Router Implementation

#### 4.1 Order Router (cpp/router/include/order_router.hpp)
```cpp
#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "types.hpp"
#include "spsc_queue.hpp"
#include "venue_connector.hpp"

namespace router {

using namespace feed;

class OrderRouter {
public:
    struct Config {
        size_t order_queue_size = 8192;
        size_t max_orders_per_second = 10000;
        bool enable_risk_checks = true;
        bool enable_throttling = true;
        int cpu_affinity = -1;
    };
    
    struct Route {
        uint8_t venue_id;
        double participation;  // Percentage to route to this venue
        int priority;
        bool is_active;
    };
    
    explicit OrderRouter(const Config& config);
    ~OrderRouter();
    
    void start();
    void stop();
    
    // Add venue connection
    void add_venue(uint8_t venue_id, std::unique_ptr<VenueConnector> connector);
    
    // Configure routing rules
    void set_routing_rules(Symbol symbol, const std::vector<Route>& routes);
    
    // Send order
    bool send_order(const OrderMessage& order);
    
    // Cancel order
    bool cancel_order(OrderId order_id);
    
    // Get order status
    enum class OrderStatus {
        PENDING,
        SENT,
        ACKNOWLEDGED,
        FILLED,
        PARTIALLY_FILLED,
        CANCELLED,
        REJECTED
    };
    
    OrderStatus get_order_status(OrderId order_id) const;
    
    // Statistics
    struct Stats {
        uint64_t orders_sent;
        uint64_t orders_acknowledged;
        uint64_t orders_rejected;
        uint64_t orders_filled;
        LatencyStats send_latency;
        LatencyStats ack_latency;
        LatencyStats fill_latency;
    };
    
    Stats get_stats() const;
    
private:
    void router_thread();
    void process_order(const OrderMessage& order);
    bool check_risk_limits(const OrderMessage& order);
    std::vector<std::pair<uint8_t, OrderMessage>> split_order(const OrderMessage& order);
    
    Config config_;
    std::thread router_thread_;
    std::atomic<bool> running_;
    
    // Order queue
    SPSCQueue<OrderMessage> order_queue_;
    
    // Venue connections
    std::unordered_map<uint8_t, std::unique_ptr<VenueConnector>> venues_;
    
    // Routing rules
    std::unordered_map<Symbol, std::vector<Route>> routing_rules_;
    std::mutex routing_mutex_;
    
    // Order tracking
    struct OrderState {
        OrderStatus status;
        uint8_t venue_id;
        Timestamp sent_time;
        Timestamp ack_time;
        Timestamp fill_time;
        Quantity filled_quantity;
    };
    
    std::unordered_map<OrderId, OrderState> order_states_;
    mutable std::mutex order_state_mutex_;
    
    // Risk limits
    struct RiskLimits {
        Quantity max_order_size = 100000;
        uint32_t max_orders_per_symbol = 100;
        uint64_t max_notional_per_day = 10000000;
    };
    
    RiskLimits risk_limits_;
    
    // Throttling
    std::atomic<uint64_t> orders_this_second_;
    std::chrono::steady_clock::time_point current_second_;
    
    // Statistics
    mutable Stats stats_;
};

void OrderRouter::router_thread() {
    // Set CPU affinity if configured
    if (config_.cpu_affinity >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(config_.cpu_affinity, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
    
    // Set real-time priority
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    OrderMessage order;
    
    while (running_.load(std::memory_order_acquire)) {
        if (!order_queue_.pop(order)) {
            __builtin_ia32_pause();
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Risk checks
        if (config_.enable_risk_checks && !check_risk_limits(order)) {
            // Reject order
            std::lock_guard<std::mutex> lock(order_state_mutex_);
            order_states_[order.id] = {
                OrderStatus::REJECTED,
                0,
                start_time.time_since_epoch().count(),
                0,
                0,
                0
            };
            stats_.orders_rejected++;
            continue;
        }
        
        // Throttling
        if (config_.enable_throttling) {
            auto now = std::chrono::steady_clock::now();
            if (now - current_second_ >= std::chrono::seconds(1)) {
                orders_this_second_ = 0;
                current_second_ = now;
            }
            
            if (orders_this_second_.fetch_add(1) >= config_.max_orders_per_second) {
                // Throttle - reject or queue
                continue;
            }
        }
        
        // Process order
        process_order(order);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto latency = (end_time - start_time).count();
        stats_.send_latency.update(latency);
    }
}

void OrderRouter::process_order(const OrderMessage& order) {
    // Get routing rules for symbol
    std::vector<Route> routes;
    {
        std::lock_guard<std::mutex> lock(routing_mutex_);
        auto it = routing_rules_.find(order.symbol);
        if (it != routing_rules_.end()) {
            routes = it->second;
        } else {
            // Use default routing
            routes = {{1, 1.0, 0, true}};  // Route all to venue 1
        }
    }
    
    // Split order based on routing rules
    auto split_orders = split_order(order);
    
    // Send to venues
    for (const auto& [venue_id, venue_order] : split_orders) {
        auto venue_it = venues_.find(venue_id);
        if (venue_it == venues_.end()) {
            continue;
        }
        
        // Track order state
        {
            std::lock_guard<std::mutex> lock(order_state_mutex_);
            order_states_[venue_order.id] = {
                OrderStatus::PENDING,
                venue_id,
                std::chrono::high_resolution_clock::now().time_since_epoch().count(),
                0,
                0,
                0
            };
        }
        
        // Send to venue
        if (venue_it->second->send_order(venue_order)) {
            stats_.orders_sent++;
            
            // Update state
            std::lock_guard<std::mutex> lock(order_state_mutex_);
            order_states_[venue_order.id].status = OrderStatus::SENT;
        }
    }
}

} // namespace router
```

#### 4.2 Venue Connector (cpp/router/include/venue_connector.hpp)
```cpp
#pragma once

#include <memory>
#include <functional>
#include "types.hpp"

namespace router {

class VenueConnector {
public:
    using FillCallback = std::function<void(OrderId, Price, Quantity)>;
    using AckCallback = std::function<void(OrderId, bool)>;
    
    virtual ~VenueConnector() = default;
    
    // Connect to venue
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;
    
    // Order operations
    virtual bool send_order(const feed::OrderMessage& order) = 0;
    virtual bool cancel_order(feed::OrderId order_id) = 0;
    virtual bool modify_order(feed::OrderId order_id, feed::Price new_price, feed::Quantity new_quantity) = 0;
    
    // Callbacks
    virtual void set_fill_callback(FillCallback callback) = 0;
    virtual void set_ack_callback(AckCallback callback) = 0;
    
    // Statistics
    virtual feed::LatencyStats get_latency_stats() const = 0;
};

// TCP venue connector implementation
class TCPVenueConnector : public VenueConnector {
public:
    struct Config {
        std::string host;
        uint16_t port;
        size_t send_buffer_size = 1024 * 1024;
        size_t recv_buffer_size = 1024 * 1024;
        bool tcp_nodelay = true;
        int cpu_affinity = -1;
    };
    
    explicit TCPVenueConnector(const Config& config);
    ~TCPVenueConnector();
    
    bool connect() override;
    void disconnect() override;
    bool is_connected() const override;
    
    bool send_order(const feed::OrderMessage& order) override;
    bool cancel_order(feed::OrderId order_id) override;
    bool modify_order(feed::OrderId order_id, feed::Price new_price, feed::Quantity new_quantity) override;
    
    void set_fill_callback(FillCallback callback) override {
        fill_callback_ = callback;
    }
    
    void set_ack_callback(AckCallback callback) override {
        ack_callback_ = callback;
    }
    
    feed::LatencyStats get_latency_stats() const override {
        return stats_;
    }
    
private:
    void send_thread();
    void receive_thread();
    bool send_message(const void* data, size_t length);
    
    Config config_;
    int socket_fd_;
    std::atomic<bool> connected_;
    
    std::thread send_thread_;
    std::thread receive_thread_;
    
    // Callbacks
    FillCallback fill_callback_;
    AckCallback ack_callback_;
    
    // Send queue
    feed::SPSCQueue<std::vector<uint8_t>> send_queue_;
    
    // Statistics
    mutable feed::LatencyStats stats_;
};

} // namespace router
```

### Phase 5: Performance Monitoring

#### 5.1 Latency Histogram (perf/latency_histogram.cpp)
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include "../cpp/feed/include/feed_handler.hpp"
#include "../cpp/router/include/order_router.hpp"

using namespace feed;
using namespace router;

class LatencyHistogram {
public:
    LatencyHistogram(uint64_t max_latency_ns = 1000000, size_t num_buckets = 1000)
        : max_latency_(max_latency_ns)
        , bucket_width_(max_latency_ns / num_buckets)
        , buckets_(num_buckets, 0)
        , total_count_(0)
        , total_sum_(0) {}
    
    void add(uint64_t latency_ns) {
        total_count_++;
        total_sum_ += latency_ns;
        
        size_t bucket = std::min(latency_ns / bucket_width_, buckets_.size() - 1);
        buckets_[bucket]++;
    }
    
    void print_histogram() const {
        if (total_count_ == 0) {
            std::cout << "No data\n";
            return;
        }
        
        // Calculate percentiles
        std::vector<uint64_t> sorted_latencies;
        sorted_latencies.reserve(total_count_);
        
        for (size_t i = 0; i < buckets_.size(); ++i) {
            uint64_t latency = i * bucket_width_ + bucket_width_ / 2;
            for (uint64_t j = 0; j < buckets_[i]; ++j) {
                sorted_latencies.push_back(latency);
            }
        }
        
        std::sort(sorted_latencies.begin(), sorted_latencies.end());
        
        auto percentile = [&](double p) {
            size_t idx = static_cast<size_t>(sorted_latencies.size() * p / 100.0);
            return sorted_latencies[std::min(idx, sorted_latencies.size() - 1)];
        };
        
        std::cout << "\n=== Latency Histogram ===\n";
        std::cout << "Count: " << total_count_ << "\n";
        std::cout << "Mean: " << (total_sum_ / total_count_) << " ns\n";
        std::cout << "Min: " << sorted_latencies.front() << " ns\n";
        std::cout << "Max: " << sorted_latencies.back() << " ns\n";
        std::cout << "P50: " << percentile(50) << " ns\n";
        std::cout << "P90: " << percentile(90) << " ns\n";
        std::cout << "P99: " << percentile(99) << " ns\n";
        std::cout << "P99.9: " << percentile(99.9) << " ns\n";
        std::cout << "P99.99: " << percentile(99.99) << " ns\n";
        
        // Print ASCII histogram
        uint64_t max_count = *std::max_element(buckets_.begin(), buckets_.end());
        const int bar_width = 50;
        
        std::cout << "\nDistribution:\n";
        for (size_t i = 0; i < buckets_.size(); i += buckets_.size() / 20) {
            uint64_t latency = i * bucket_width_;
            uint64_t count = buckets_[i];
            
            std::cout << std::setw(8) << latency << " ns: ";
            
            int bar_len = (count * bar_width) / max_count;
            for (int j = 0; j < bar_len; ++j) {
                std::cout << "#";
            }
            
            std::cout << " " << count << "\n";
        }
    }
    
    double get_median_absolute_jitter() const {
        if (total_count_ < 2) return 0;
        
        double mean = static_cast<double>(total_sum_) / total_count_;
        std::vector<double> deviations;
        
        for (size_t i = 0; i < buckets_.size(); ++i) {
            double latency = i * bucket_width_ + bucket_width_ / 2;
            for (uint64_t j = 0; j < buckets_[i]; ++j) {
                deviations.push_back(std::abs(latency - mean));
            }
        }
        
        std::sort(deviations.begin(), deviations.end());
        return deviations[deviations.size() / 2];
    }
    
private:
    uint64_t max_latency_;
    uint64_t bucket_width_;
    std::vector<uint64_t> buckets_;
    uint64_t total_count_;
    uint64_t total_sum_;
};

void run_latency_test() {
    // Setup feed handler
    FeedHandler::Config feed_config;
    feed_config.udp_config.multicast_group = "239.1.1.1";
    feed_config.udp_config.port = 12345;
    feed_config.udp_config.cpu_affinity = 2;
    feed_config.udp_config.use_busy_poll = true;
    
    FeedHandler feed_handler(feed_config);
    
    // Setup order router
    OrderRouter::Config router_config;
    router_config.cpu_affinity = 3;
    router_config.enable_risk_checks = false;  // Disable for latency test
    
    OrderRouter order_router(router_config);
    
    // Latency tracking
    LatencyHistogram feed_latency;
    LatencyHistogram order_latency;
    LatencyHistogram round_trip_latency;
    
    // Setup callbacks
    std::atomic<uint64_t> message_count(0);
    
    feed_handler.set_callback([&](const MarketDataMessage& msg) {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        uint64_t latency = now - msg.exchange_timestamp;
        
        feed_latency.add(latency);
        message_count++;
        
        // Generate order on every 10th message
        if (message_count % 10 == 0) {
            OrderMessage order;
            order.id = message_count;
            order.symbol = msg.symbol;
            order.side = Side::BUY;
            order.price = msg.quote.bid_price;
            order.quantity = 100;
            order.timestamp = now;
            
            auto send_start = std::chrono::high_resolution_clock::now();
            order_router.send_order(order);
            auto send_end = std::chrono::high_resolution_clock::now();
            
            order_latency.add((send_end - send_start).count());
        }
    });
    
    // Start components
    feed_handler.start();
    order_router.start();
    
    // Run for 10 seconds
    std::this_thread::sleep_for(std::chrono::seconds(10));
    
    // Stop components
    feed_handler.stop();
    order_router.stop();
    
    // Print results
    std::cout << "\n=== Feed Handler Latency ===\n";
    feed_latency.print_histogram();
    std::cout << "Median Absolute Jitter: " << feed_latency.get_median_absolute_jitter() << " ns\n";
    
    std::cout << "\n=== Order Router Latency ===\n";
    order_latency.print_histogram();
    
    // Print component statistics
    auto feed_stats = feed_handler.get_stats();
    std::cout << "\n=== Feed Handler Stats ===\n";
    std::cout << "Packets received: " << feed_stats.packets_received << "\n";
    std::cout << "Messages decoded: " << feed_stats.messages_decoded << "\n";
    std::cout << "Messages processed: " << feed_stats.messages_processed << "\n";
    std::cout << "Decode errors: " << feed_stats.decode_errors << "\n";
    
    auto router_stats = order_router.get_stats();
    std::cout << "\n=== Order Router Stats ===\n";
    std::cout << "Orders sent: " << router_stats.orders_sent << "\n";
    std::cout << "Orders acknowledged: " << router_stats.orders_acknowledged << "\n";
    std::cout << "Orders rejected: " << router_stats.orders_rejected << "\n";
}

int main() {
    // Set process priority
    if (nice(-20) == -1) {
        std::cerr << "Warning: Failed to set process priority\n";
    }
    
    // Disable CPU frequency scaling
    system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor");
    
    // Run test
    run_latency_test();
    
    return 0;
}
```

### Phase 6: Build Configuration

#### 6.1 CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.14)
project(RealTimeFeedHandler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

# Add warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Find packages
find_package(Threads REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)

# Optional packages
find_package(TBB)
find_package(numa)

# Library
add_library(feed_handler STATIC
    cpp/feed/src/feed_handler.cpp
    cpp/feed/src/udp_receiver.cpp
    cpp/feed/src/tcp_receiver.cpp
    cpp/feed/src/decoder.cpp
    cpp/feed/src/parser.cpp
    cpp/common/src/memory_pool.cpp
    cpp/common/src/timestamp.cpp
    cpp/utils/src/cpu_affinity.cpp
    cpp/utils/src/statistics.cpp
)

target_include_directories(feed_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/feed/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/utils/include
)

target_link_libraries(feed_handler PUBLIC
    Threads::Threads
    Boost::system
    Boost::thread
    ${CMAKE_DL_LIBS}
)

if(TBB_FOUND)
    target_link_libraries(feed_handler PUBLIC TBB::tbb)
    target_compile_definitions(feed_handler PUBLIC USE_TBB)
endif()

if(numa_FOUND)
    target_link_libraries(feed_handler PUBLIC numa)
    target_compile_definitions(feed_handler PUBLIC USE_NUMA)
endif()

# Order router library
add_library(order_router STATIC
    cpp/router/src/order_router.cpp
    cpp/router/src/smart_router.cpp
    cpp/router/src/venue_connector.cpp
    cpp/router/src/order_manager.cpp
)

target_include_directories(order_router PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/router/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/include
)

target_link_libraries(order_router PUBLIC
    feed_handler
)

# Main executable
add_executable(feed_router cpp/main.cpp)
target_link_libraries(feed_router
    feed_handler
    order_router
)

# Performance tests
add_executable(latency_histogram perf/latency_histogram.cpp)
target_link_libraries(latency_histogram
    feed_handler
    order_router
)

add_executable(throughput_test perf/throughput_test.cpp)
target_link_libraries(throughput_test
    feed_handler
)

add_executable(burst_test perf/burst_test.cpp)
target_link_libraries(burst_test
    feed_handler
)

# Unit tests
enable_testing()

add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer feed_handler)
add_test(NAME test_ring_buffer COMMAND test_ring_buffer)

add_executable(test_decoder tests/test_decoder.cpp)
target_link_libraries(test_decoder feed_handler)
add_test(NAME test_decoder COMMAND test_decoder)

# Installation
install(TARGETS feed_handler order_router
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY cpp/feed/include/ DESTINATION include/feed)
install(DIRECTORY cpp/router/include/ DESTINATION include/router)
install(DIRECTORY cpp/common/include/ DESTINATION include/common)
```

### Phase 7: System Tuning Scripts

#### 7.1 Network Tuning Script (scripts/tune_network.sh)
```bash
#!/bin/bash

# Network stack tuning for low latency
# Run with sudo

echo "Tuning network stack for low latency..."

# Increase network buffers
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.rmem_default = 25165824' >> /etc/sysctl.conf
echo 'net.core.wmem_default = 25165824' >> /etc/sysctl.conf
echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf

# TCP tuning
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_low_latency = 1' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_no_metrics_save = 1' >> /etc/sysctl.conf

# Disable TCP timestamps
echo 'net.ipv4.tcp_timestamps = 0' >> /etc/sysctl.conf

# Enable busy polling
echo 'net.core.busy_poll = 50' >> /etc/sysctl.conf
echo 'net.core.busy_read = 50' >> /etc/sysctl.conf

# Apply settings
sysctl -p

# Set interrupt affinity
echo "Setting interrupt affinity..."
for irq in $(grep eth0 /proc/interrupts | cut -d: -f1); do
    echo 2 > /proc/irq/$irq/smp_affinity
done

# Disable interrupt coalescing
ethtool -C eth0 rx-usecs 0 rx-frames 1
ethtool -C eth0 tx-usecs 0 tx-frames 1

# Enable hardware timestamping
ethtool -T eth0

echo "Network tuning complete"
```

#### 7.2 HugePages Setup (scripts/setup_hugepages.sh)
```bash
#!/bin/bash

# Setup HugePages for reduced TLB misses
# Run with sudo

echo "Setting up HugePages..."

# Reserve 1GB of HugePages (512 pages of 2MB each)
echo 512 > /proc/sys/vm/nr_hugepages

# Make persistent
echo 'vm.nr_hugepages = 512' >> /etc/sysctl.conf

# Mount hugetlbfs
mkdir -p /mnt/huge
mount -t hugetlbfs nodev /mnt/huge
echo 'nodev /mnt/huge hugetlbfs defaults 0 0' >> /etc/fstab

# Set permissions
chmod 777 /mnt/huge

echo "HugePages setup complete"
echo "Current HugePages:"
cat /proc/meminfo | grep Huge
```

## Performance Metrics & Targets

### Latency Targets
- **Wire-to-Handler**: < 1 μs (p50), < 5 μs (p99)
- **Decode Latency**: < 100 ns per message
- **Order Send Latency**: < 500 ns (p50), < 2 μs (p99)
- **Round-trip (Market Data → Order)**: < 5 μs (p50), < 10 μs (p99)

### Throughput Targets
- **Market Data Processing**: > 10 million messages/second
- **Order Routing**: > 1 million orders/second
- **Burst Handling**: 100x normal rate for 100ms

### Jitter & Consistency
- **Median Absolute Jitter**: < 500 ns
- **Packet Loss**: < 0.001% under normal load
- **Packet Loss during burst**: < 0.01%

## Testing & Validation Checklist

- [ ] Zero packet loss under normal load
- [ ] Correct message decoding for all message types
- [ ] Lock-free data structures are thread-safe
- [ ] CPU affinity properly set
- [ ] NUMA optimization working
- [ ] Latency meets targets at p50/p99/p99.9
- [ ] Burst handling without degradation
- [ ] Memory usage stable over time
- [ ] Network buffers sized appropriately
- [ ] Hardware timestamping enabled

## Next Steps

1. Implement FIX/FAST protocol support
2. Add kernel bypass (DPDK/Solarflare)
3. Implement FPGA acceleration
4. Add redundancy and failover
5. Build monitoring dashboard
6. Add replay capability for testing