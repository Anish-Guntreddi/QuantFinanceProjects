import pytest
import asyncio
import time
import struct
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feed.handler import (
    RingBuffer, MessageDecoder, FeedHandler, OrderRouter,
    MarketDataMessage, MessageType
)

def test_ring_buffer():
    """Test ring buffer"""
    buffer = RingBuffer(capacity=10)
    
    # Test push/pop
    assert buffer.push("item1")
    assert buffer.push("item2")
    assert buffer.size() == 2
    
    assert buffer.pop() == "item1"
    assert buffer.pop() == "item2"
    assert buffer.pop() is None

def test_message_decoder():
    """Test message decoder"""
    decoder = MessageDecoder()
    
    # Create test quote message
    msg_type = b'Q'
    symbol = b'AAPL    '  # 8 bytes
    timestamp = struct.pack('>Q', time.time_ns())
    bid_price = struct.pack('>d', 150.25)
    ask_price = struct.pack('>d', 150.30)
    
    data = msg_type + symbol + timestamp + bid_price + ask_price
    
    msg = decoder.decode(data)
    assert msg is not None
    assert msg.msg_type == MessageType.QUOTE
    assert msg.symbol == 'AAPL'
    assert msg.data['bid'] == 150.25
    assert msg.data['ask'] == 150.30

@pytest.mark.asyncio
async def test_feed_handler():
    """Test feed handler"""
    config = {
        'multicast_group': '239.1.1.1',
        'port': 12345,
        'queue_size': 100
    }
    
    feed = FeedHandler(config)
    
    # Test subscription
    messages_received = []
    
    def on_message(msg):
        messages_received.append(msg)
    
    feed.subscribe(MessageType.QUOTE, on_message)
    
    # Add test message to queue
    test_msg = MarketDataMessage(
        msg_type=MessageType.QUOTE,
        symbol='TEST',
        timestamp=time.time_ns(),
        sequence=1,
        data={'bid': 100, 'ask': 101}
    )
    
    feed.message_queue.push(test_msg)
    
    # Process message
    feed.running = True
    await feed.processor_task()
    
    # Check stats
    stats = feed.get_stats()
    assert stats['messages_processed'] > 0

def test_order_router():
    """Test order router"""
    config = {
        'max_order_size': 10000,
        'max_price': 1000
    }
    
    router = OrderRouter(config)
    
    # Test risk checks
    order = {
        'symbol': 'TEST',
        'side': 'buy',
        'quantity': 100,
        'price': 100
    }
    
    assert router._check_risk(order)
    
    # Test rejection for large order
    large_order = {
        'symbol': 'TEST',
        'side': 'buy',
        'quantity': 100000,
        'price': 100
    }
    
    assert not router._check_risk(large_order)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
