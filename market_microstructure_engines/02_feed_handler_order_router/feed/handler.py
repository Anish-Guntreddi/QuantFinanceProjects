"""
Ultra-low latency feed handler
"""

import asyncio
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import threading
from ..common.queue import LockFreeQueue, LatencyTracker
from .decoder import ITCHDecoder, FIXDecoder, MarketDataMessage

@dataclass
class FeedConfig:
    """Feed handler configuration"""
    protocol: str = "ITCH"  # ITCH, FIX, etc.
    host: str = "localhost"
    port: int = 12345
    use_multicast: bool = False
    multicast_group: Optional[str] = None
    buffer_size: int = 65536
    worker_threads: int = 2
    cpu_affinity: Optional[List[int]] = None

class FeedHandler:
    """High-performance feed handler"""
    
    def __init__(self, config: FeedConfig):
        self.config = config
        self.running = False
        
        # Decoders
        if config.protocol == "ITCH":
            self.decoder = ITCHDecoder()
        elif config.protocol == "FIX":
            self.decoder = FIXDecoder()
        else:
            raise ValueError(f"Unknown protocol: {config.protocol}")
            
        # Queues
        self.raw_queue = LockFreeQueue(config.buffer_size)
        self.decoded_queue = LockFreeQueue(config.buffer_size)
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {}
        self.default_callback: Optional[Callable] = None
        
        # Statistics
        self.latency_tracker = LatencyTracker()
        self.messages_received = 0
        self.messages_decoded = 0
        self.messages_processed = 0
        self.decode_errors = 0
        
        # Threads
        self.receiver_thread = None
        self.decoder_threads = []
        self.processor_thread = None
        
    async def start(self):
        """Start feed handler"""
        self.running = True
        
        # Start receiver
        self.receiver_thread = threading.Thread(target=self._receiver_loop)
        self.receiver_thread.start()
        
        # Start decoder threads
        for i in range(self.config.worker_threads):
            thread = threading.Thread(target=self._decoder_loop, args=(i,))
            thread.start()
            self.decoder_threads.append(thread)
            
        # Start processor
        self.processor_thread = threading.Thread(target=self._processor_loop)
        self.processor_thread.start()
        
    def stop(self):
        """Stop feed handler"""
        self.running = False
        
        # Wait for threads
        if self.receiver_thread:
            self.receiver_thread.join()
            
        for thread in self.decoder_threads:
            thread.join()
            
        if self.processor_thread:
            self.processor_thread.join()
            
    def register_callback(self, symbol: str, callback: Callable):
        """Register callback for symbol"""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
        self.callbacks[symbol].append(callback)
        
    def register_default_callback(self, callback: Callable):
        """Register default callback"""
        self.default_callback = callback
        
    def _receiver_loop(self):
        """Receive raw packets (simulated)"""
        import random
        
        while self.running:
            # Simulate receiving packet
            time.sleep(0.00001)  # 10 microseconds
            
            # Generate fake packet
            packet = {
                'data': b'\x00\x00A',  # Fake ITCH add order
                'timestamp': time.time_ns()
            }
            
            if self.raw_queue.put_nowait(packet):
                self.messages_received += 1
                
            # Occasionally generate other message types
            if random.random() < 0.1:
                packet = {
                    'data': b'\x00\x00P',  # Fake trade
                    'timestamp': time.time_ns()
                }
                self.raw_queue.put_nowait(packet)
                
    def _decoder_loop(self, worker_id: int):
        """Decode raw packets"""
        
        # Set CPU affinity if configured
        if self.config.cpu_affinity and worker_id < len(self.config.cpu_affinity):
            try:
                import os
                os.sched_setaffinity(0, {self.config.cpu_affinity[worker_id]})
            except:
                pass
                
        while self.running:
            packet = self.raw_queue.get_nowait()
            
            if packet:
                # Decode message
                try:
                    msg = self.decoder.decode(packet['data'])
                    
                    if msg:
                        # Track decode latency
                        decode_time = time.time_ns()
                        latency = decode_time - packet['timestamp']
                        self.latency_tracker.record(latency)
                        
                        # Add to decoded queue
                        self.decoded_queue.put_nowait({
                            'message': msg,
                            'decode_time': decode_time
                        })
                        self.messages_decoded += 1
                    else:
                        self.decode_errors += 1
                except Exception as e:
                    self.decode_errors += 1
            else:
                time.sleep(0.000001)  # 1 microsecond
                
    def _processor_loop(self):
        """Process decoded messages"""
        
        while self.running:
            item = self.decoded_queue.get_nowait()
            
            if item:
                msg = item['message']
                
                # Find callbacks
                callbacks = self.callbacks.get(msg.symbol, [])
                
                if not callbacks and self.default_callback:
                    callbacks = [self.default_callback]
                    
                # Invoke callbacks
                for callback in callbacks:
                    try:
                        callback(msg)
                    except Exception as e:
                        pass
                        
                self.messages_processed += 1
            else:
                time.sleep(0.000001)
                
    def get_statistics(self) -> dict:
        """Get feed handler statistics"""
        latency_stats = self.latency_tracker.get_stats()
        
        return {
            'messages_received': self.messages_received,
            'messages_decoded': self.messages_decoded,
            'messages_processed': self.messages_processed,
            'decode_errors': self.decode_errors,
            'raw_queue_size': self.raw_queue.qsize(),
            'decoded_queue_size': self.decoded_queue.qsize(),
            'dropped_packets': self.raw_queue.dropped,
            'latency': latency_stats
        }
