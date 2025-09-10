"""
High-performance feed handler implementation in Python
"""

import asyncio
import time
import struct
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import numpy as np
from enum import Enum

class MessageType(Enum):
    ADD_ORDER = 'A'
    DELETE_ORDER = 'D'
    MODIFY_ORDER = 'M'
    EXECUTE_ORDER = 'E'
    TRADE = 'T'
    QUOTE = 'Q'
    STATUS = 'S'

@dataclass
class MarketDataMessage:
    """Market data message"""
    msg_type: MessageType
    symbol: str
    timestamp: float
    sequence: int
    data: Dict[str, Any]

class RingBuffer:
    """Lock-free ring buffer for message passing"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.dropped = 0
        
    def push(self, item: Any) -> bool:
        """Push item to buffer"""
        try:
            self.buffer.append(item)
            return True
        except:
            self.dropped += 1
            return False
    
    def pop(self) -> Optional[Any]:
        """Pop item from buffer"""
        try:
            return self.buffer.popleft()
        except IndexError:
            return None
    
    def size(self) -> int:
        return len(self.buffer)

class UDPReceiver:
    """UDP multicast receiver"""
    
    def __init__(self, multicast_group: str, port: int, buffer_size: int = 65536):
        self.multicast_group = multicast_group
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.running = False
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'bytes_received': 0
        }
        
    def setup_socket(self):
        """Setup UDP socket for multicast"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to port
        self.socket.bind(('', self.port))
        
        # Join multicast group
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        # Set non-blocking
        self.socket.setblocking(False)
        
    async def receive_loop(self, packet_queue: RingBuffer):
        """Receive packets and push to queue"""
        self.running = True
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(self.buffer_size)
                timestamp = time.time_ns()
                
                packet = {
                    'data': data,
                    'timestamp': timestamp,
                    'source': addr
                }
                
                if packet_queue.push(packet):
                    self.stats['packets_received'] += 1
                    self.stats['bytes_received'] += len(data)
                else:
                    self.stats['packets_dropped'] += 1
                    
            except BlockingIOError:
                await asyncio.sleep(0.00001)  # 10 microseconds
            except Exception as e:
                print(f"Receive error: {e}")
                
    def stop(self):
        """Stop receiver"""
        self.running = False
        if self.socket:
            self.socket.close()

class MessageDecoder:
    """Decode binary messages to MarketDataMessage"""
    
    def __init__(self):
        self.sequence = 0
        
    def decode(self, data: bytes) -> Optional[MarketDataMessage]:
        """Decode binary message"""
        if len(data) < 16:  # Minimum message size
            return None
            
        try:
            # Simple binary format (customize as needed)
            # [msg_type(1), symbol(8), timestamp(8), ...data]
            msg_type = chr(data[0])
            symbol = data[1:9].decode('ascii').strip()
            timestamp = struct.unpack('>Q', data[9:17])[0]
            
            self.sequence += 1
            
            # Decode based on message type
            if msg_type == 'Q':  # Quote
                if len(data) >= 33:
                    bid_price = struct.unpack('>d', data[17:25])[0]
                    ask_price = struct.unpack('>d', data[25:33])[0]
                    
                    return MarketDataMessage(
                        msg_type=MessageType.QUOTE,
                        symbol=symbol,
                        timestamp=timestamp,
                        sequence=self.sequence,
                        data={'bid': bid_price, 'ask': ask_price}
                    )
                    
            elif msg_type == 'T':  # Trade
                if len(data) >= 33:
                    price = struct.unpack('>d', data[17:25])[0]
                    quantity = struct.unpack('>Q', data[25:33])[0]
                    
                    return MarketDataMessage(
                        msg_type=MessageType.TRADE,
                        symbol=symbol,
                        timestamp=timestamp,
                        sequence=self.sequence,
                        data={'price': price, 'quantity': quantity}
                    )
                    
        except Exception as e:
            print(f"Decode error: {e}")
            
        return None

class FeedHandler:
    """Main feed handler coordinating reception and processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.packet_queue = RingBuffer(config.get('queue_size', 65536))
        self.message_queue = RingBuffer(config.get('queue_size', 65536))
        self.receiver = None
        self.decoder = MessageDecoder()
        self.callbacks: Dict[MessageType, List[Callable]] = {}
        self.running = False
        self.stats = {
            'messages_decoded': 0,
            'decode_errors': 0,
            'messages_processed': 0,
            'total_latency_ns': 0
        }
        
    def subscribe(self, msg_type: MessageType, callback: Callable):
        """Subscribe to message type"""
        if msg_type not in self.callbacks:
            self.callbacks[msg_type] = []
        self.callbacks[msg_type].append(callback)
        
    async def decoder_task(self):
        """Decode packets to messages"""
        while self.running:
            packet = self.packet_queue.pop()
            
            if packet:
                msg = self.decoder.decode(packet['data'])
                
                if msg:
                    msg.receive_timestamp = packet['timestamp']
                    self.message_queue.push(msg)
                    self.stats['messages_decoded'] += 1
                else:
                    self.stats['decode_errors'] += 1
            else:
                await asyncio.sleep(0.00001)
                
    async def processor_task(self):
        """Process messages and call callbacks"""
        while self.running:
            msg = self.message_queue.pop()
            
            if msg:
                # Calculate latency
                now = time.time_ns()
                latency = now - msg.timestamp
                self.stats['total_latency_ns'] += latency
                self.stats['messages_processed'] += 1
                
                # Call callbacks
                if msg.msg_type in self.callbacks:
                    for callback in self.callbacks[msg.msg_type]:
                        try:
                            callback(msg)
                        except Exception as e:
                            print(f"Callback error: {e}")
            else:
                await asyncio.sleep(0.00001)
                
    async def start(self):
        """Start feed handler"""
        self.running = True
        
        # Setup receiver
        self.receiver = UDPReceiver(
            self.config['multicast_group'],
            self.config['port']
        )
        self.receiver.setup_socket()
        
        # Start tasks
        tasks = [
            asyncio.create_task(self.receiver.receive_loop(self.packet_queue)),
            asyncio.create_task(self.decoder_task()),
            asyncio.create_task(self.processor_task())
        ]
        
        await asyncio.gather(*tasks)
        
    def stop(self):
        """Stop feed handler"""
        self.running = False
        if self.receiver:
            self.receiver.stop()
            
    def get_stats(self) -> Dict:
        """Get statistics"""
        stats = dict(self.stats)
        
        if self.receiver:
            stats.update(self.receiver.stats)
            
        if stats['messages_processed'] > 0:
            stats['avg_latency_ns'] = stats['total_latency_ns'] / stats['messages_processed']
            stats['avg_latency_us'] = stats['avg_latency_ns'] / 1000
            
        return stats

class OrderRouter:
    """Simple order router"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.order_queue = RingBuffer(config.get('queue_size', 8192))
        self.venues = {}
        self.stats = {
            'orders_sent': 0,
            'orders_rejected': 0,
            'total_latency_ns': 0
        }
        
    def add_venue(self, venue_id: str, connector):
        """Add venue connector"""
        self.venues[venue_id] = connector
        
    async def send_order(self, order: Dict) -> bool:
        """Send order to venue"""
        start_time = time.time_ns()
        
        # Risk checks (simplified)
        if not self._check_risk(order):
            self.stats['orders_rejected'] += 1
            return False
            
        # Route to venue
        venue_id = order.get('venue', 'default')
        if venue_id in self.venues:
            success = await self.venues[venue_id].send(order)
            if success:
                self.stats['orders_sent'] += 1
                self.stats['total_latency_ns'] += time.time_ns() - start_time
                return True
                
        return False
        
    def _check_risk(self, order: Dict) -> bool:
        """Simple risk checks"""
        # Check order size
        if order.get('quantity', 0) > self.config.get('max_order_size', 100000):
            return False
            
        # Check price reasonableness
        price = order.get('price', 0)
        if price <= 0 or price > self.config.get('max_price', 10000):
            return False
            
        return True
        
    def get_stats(self) -> Dict:
        """Get router statistics"""
        stats = dict(self.stats)
        
        if stats['orders_sent'] > 0:
            stats['avg_latency_ns'] = stats['total_latency_ns'] / stats['orders_sent']
            stats['avg_latency_us'] = stats['avg_latency_ns'] / 1000
            
        return stats

# Example usage
async def main():
    """Example usage of feed handler"""
    
    # Configure feed handler
    feed_config = {
        'multicast_group': '239.1.1.1',
        'port': 12345,
        'queue_size': 65536
    }
    
    feed = FeedHandler(feed_config)
    
    # Configure order router
    router_config = {
        'max_order_size': 10000,
        'max_price': 1000
    }
    
    router = OrderRouter(router_config)
    
    # Subscribe to quotes
    def on_quote(msg: MarketDataMessage):
        print(f"Quote: {msg.symbol} Bid={msg.data['bid']:.2f} Ask={msg.data['ask']:.2f}")
        
    feed.subscribe(MessageType.QUOTE, on_quote)
    
    # Subscribe to trades
    def on_trade(msg: MarketDataMessage):
        print(f"Trade: {msg.symbol} Price={msg.data['price']:.2f} Qty={msg.data['quantity']}")
        
    feed.subscribe(MessageType.TRADE, on_trade)
    
    try:
        # Start feed handler
        await feed.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        feed.stop()
        
        # Print statistics
        print("\nFeed Handler Statistics:")
        for key, value in feed.get_stats().items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Use uvloop for better performance
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
        
    asyncio.run(main())
