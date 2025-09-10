"""
Lock-free queue implementation for high-performance message passing
"""

from collections import deque
from threading import Lock
from typing import Any, Optional
import time

class LockFreeQueue:
    """
    Simplified lock-free queue using deque with minimal locking.
    For true lock-free, would need C extension or use multiprocessing.Queue
    """
    
    def __init__(self, maxsize: int = 65536):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
        self.lock = Lock()
        self.dropped = 0
        
    def put_nowait(self, item: Any) -> bool:
        """Put item without blocking"""
        with self.lock:
            if len(self.queue) >= self.maxsize:
                self.dropped += 1
                return False
            self.queue.append(item)
            return True
            
    def get_nowait(self) -> Optional[Any]:
        """Get item without blocking"""
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None
            
    def qsize(self) -> int:
        """Get queue size"""
        with self.lock:
            return len(self.queue)
            
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return len(self.queue) == 0

class LatencyTracker:
    """Track latency statistics"""
    
    def __init__(self, num_buckets: int = 10000):
        self.num_buckets = num_buckets
        self.bucket_width_ns = 1000  # 1 microsecond buckets
        self.buckets = [0] * num_buckets
        self.count = 0
        self.sum_ns = 0
        self.min_ns = float('inf')
        self.max_ns = 0
        
    def record(self, latency_ns: int):
        """Record latency sample"""
        bucket = min(int(latency_ns / self.bucket_width_ns), self.num_buckets - 1)
        self.buckets[bucket] += 1
        
        self.count += 1
        self.sum_ns += latency_ns
        self.min_ns = min(self.min_ns, latency_ns)
        self.max_ns = max(self.max_ns, latency_ns)
        
    def get_percentile(self, percentile: float) -> int:
        """Get percentile latency"""
        if self.count == 0:
            return 0
            
        target = int(self.count * percentile)
        cumulative = 0
        
        for i, count in enumerate(self.buckets):
            cumulative += count
            if cumulative >= target:
                return i * self.bucket_width_ns
                
        return self.max_ns
    
    def get_stats(self) -> dict:
        """Get latency statistics"""
        if self.count == 0:
            return {
                'count': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'p50': 0,
                'p90': 0,
                'p95': 0,
                'p99': 0,
                'p999': 0
            }
            
        return {
            'count': self.count,
            'mean': self.sum_ns / self.count,
            'min': self.min_ns,
            'max': self.max_ns,
            'p50': self.get_percentile(0.50),
            'p90': self.get_percentile(0.90),
            'p95': self.get_percentile(0.95),
            'p99': self.get_percentile(0.99),
            'p999': self.get_percentile(0.999)
        }
    
    def reset(self):
        """Reset statistics"""
        self.buckets = [0] * self.num_buckets
        self.count = 0
        self.sum_ns = 0
        self.min_ns = float('inf')
        self.max_ns = 0
