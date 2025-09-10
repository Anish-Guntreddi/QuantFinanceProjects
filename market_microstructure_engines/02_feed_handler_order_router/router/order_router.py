"""
Ultra-low latency order router
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from ..common.queue import LockFreeQueue, LatencyTracker

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

@dataclass
class Order:
    """Order to route"""
    order_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    destination: str
    timestamp: int  # Nanoseconds

@dataclass
class RouterConfig:
    """Router configuration"""
    destinations: List[str]
    worker_threads: int = 2
    max_orders_per_second: int = 10000
    enable_retry: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 1

class FIXEncoder:
    """FIX protocol encoder"""
    
    SOH = chr(1)
    
    def __init__(self, sender_comp_id: str = "ROUTER", target_comp_id: str = "EXCHANGE"):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.sequence = 1
        
    def encode_new_order(self, order: Order) -> str:
        """Encode new order to FIX"""
        fields = [
            f"8=FIX.4.4",
            f"35=D",  # New Order Single
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"34={self.sequence}",
            f"52={self._get_timestamp()}",
            f"11={order.order_id}",
            f"55={order.symbol}",
            f"54={'1' if order.side == OrderSide.BUY else '2'}",
            f"40={'1' if order.order_type == OrderType.MARKET else '2'}",
            f"38={order.quantity}"
        ]
        
        if order.order_type == OrderType.LIMIT:
            fields.append(f"44={order.price}")
            
        self.sequence += 1
        
        # Calculate body length
        body = self.SOH.join(fields[1:])
        length = len(body)
        
        # Add length
        message = f"8=FIX.4.4{self.SOH}9={length}{self.SOH}{body}"
        
        # Calculate checksum
        checksum = sum(ord(c) for c in message) % 256
        message += f"{self.SOH}10={checksum:03d}{self.SOH}"
        
        return message
    
    def _get_timestamp(self) -> str:
        """Get FIX timestamp"""
        import datetime
        now = datetime.datetime.utcnow()
        return now.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]

class OrderRouter:
    """High-performance order router"""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.running = False
        
        # Order tracking
        self.next_order_id = 1
        self.orders: Dict[int, Order] = {}
        
        # Queues
        self.order_queue = LockFreeQueue(65536)
        
        # Encoder
        self.encoder = FIXEncoder()
        
        # Statistics
        self.submission_latency = LatencyTracker()
        self.wire_latency = LatencyTracker()
        self.submitted_orders = 0
        self.sent_orders = 0
        self.rejected_orders = 0
        self.failed_orders = 0
        
        # Threads
        self.worker_threads = []
        
    def start(self):
        """Start router"""
        self.running = True
        
        # Start worker threads
        for i in range(self.config.worker_threads):
            thread = threading.Thread(target=self._worker_loop, args=(i,))
            thread.start()
            self.worker_threads.append(thread)
            
    def stop(self):
        """Stop router"""
        self.running = False
        
        for thread in self.worker_threads:
            thread.join()
            
    def submit_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    price: float, quantity: int, destination: str = "DEFAULT") -> int:
        """Submit order for routing"""
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            destination=destination,
            timestamp=time.time_ns()
        )
        
        # Track order
        self.orders[order_id] = order
        
        # Queue order
        if self.order_queue.put_nowait(order):
            self.submitted_orders += 1
            return order_id
        else:
            self.rejected_orders += 1
            return 0
            
    def _worker_loop(self, worker_id: int):
        """Process orders"""
        
        while self.running:
            order = self.order_queue.get_nowait()
            
            if order:
                self._process_order(order)
            else:
                time.sleep(0.000001)
                
    def _process_order(self, order: Order):
        """Process and route order"""
        
        start_time = time.time_ns()
        
        # Track submission latency
        self.submission_latency.record(start_time - order.timestamp)
        
        # Encode order
        fix_message = self.encoder.encode_new_order(order)
        
        # Simulate sending (would actually send via TCP/UDP)
        success = self._send_order(order.destination, fix_message)
        
        if success:
            self.sent_orders += 1
            
            # Track wire latency
            wire_time = time.time_ns() - start_time
            self.wire_latency.record(wire_time)
        else:
            # Retry logic
            if self.config.enable_retry:
                for attempt in range(self.config.max_retries):
                    time.sleep(self.config.retry_delay_ms / 1000)
                    
                    if self._send_order(order.destination, fix_message):
                        self.sent_orders += 1
                        break
                else:
                    self.failed_orders += 1
            else:
                self.failed_orders += 1
                
    def _send_order(self, destination: str, message: str) -> bool:
        """Send order to destination (simulated)"""
        # In real implementation, would send via TCP/UDP connection
        # Simulate network delay
        time.sleep(0.000010)  # 10 microseconds
        
        # Simulate occasional failure
        import random
        return random.random() > 0.001  # 99.9% success rate
        
    def get_statistics(self) -> dict:
        """Get router statistics"""
        
        submission_stats = self.submission_latency.get_stats()
        wire_stats = self.wire_latency.get_stats()
        
        return {
            'submitted_orders': self.submitted_orders,
            'sent_orders': self.sent_orders,
            'rejected_orders': self.rejected_orders,
            'failed_orders': self.failed_orders,
            'queue_size': self.order_queue.qsize(),
            'submission_latency': submission_stats,
            'wire_latency': wire_stats
        }
