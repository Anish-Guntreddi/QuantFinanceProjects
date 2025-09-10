import pytest
import asyncio
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from common.queue import LockFreeQueue, LatencyTracker
from feed.decoder import ITCHDecoder, FIXDecoder, MessageType
from feed.handler import FeedHandler, FeedConfig
from router.order_router import OrderRouter, RouterConfig, OrderSide, OrderType

def test_lock_free_queue():
    """Test lock-free queue"""
    queue = LockFreeQueue(maxsize=10)
    
    # Test put and get
    assert queue.put_nowait("item1")
    assert queue.put_nowait("item2")
    assert queue.qsize() == 2
    
    assert queue.get_nowait() == "item1"
    assert queue.get_nowait() == "item2"
    assert queue.get_nowait() is None

def test_latency_tracker():
    """Test latency tracker"""
    tracker = LatencyTracker()
    
    # Record some samples
    for i in range(100):
        tracker.record(i * 1000)  # 0-99 microseconds
        
    stats = tracker.get_stats()
    assert stats['count'] == 100
    assert stats['min'] == 0
    assert stats['max'] == 99000
    assert stats['p50'] > 0

def test_itch_decoder():
    """Test ITCH decoder"""
    decoder = ITCHDecoder()
    
    # Test add order
    msg = decoder.decode(b'\x00\x00A')
    assert msg is not None
    assert msg.message_type == MessageType.ADD_ORDER
    assert msg.symbol == "TEST"

def test_fix_encoder():
    """Test FIX encoder"""
    from router.order_router import FIXEncoder
    
    encoder = FIXEncoder()
    order = type('Order', (), {
        'order_id': 123,
        'symbol': 'TEST',
        'side': OrderSide.BUY,
        'order_type': OrderType.LIMIT,
        'price': 100.50,
        'quantity': 1000
    })()
    
    message = encoder.encode_new_order(order)
    assert '35=D' in message  # New Order Single
    assert '55=TEST' in message
    assert '38=1000' in message

@pytest.mark.asyncio
async def test_feed_handler():
    """Test feed handler"""
    config = FeedConfig(
        protocol="ITCH",
        worker_threads=1
    )
    
    handler = FeedHandler(config)
    
    # Track messages
    messages = []
    
    def callback(msg):
        messages.append(msg)
    
    handler.register_default_callback(callback)
    
    await handler.start()
    await asyncio.sleep(0.1)
    handler.stop()
    
    stats = handler.get_statistics()
    assert stats['messages_received'] > 0

def test_order_router():
    """Test order router"""
    config = RouterConfig(
        destinations=["TEST"],
        worker_threads=1
    )
    
    router = OrderRouter(config)
    router.start()
    
    # Submit order
    order_id = router.submit_order(
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=100
    )
    
    assert order_id > 0
    
    time.sleep(0.1)
    router.stop()
    
    stats = router.get_statistics()
    assert stats['submitted_orders'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
