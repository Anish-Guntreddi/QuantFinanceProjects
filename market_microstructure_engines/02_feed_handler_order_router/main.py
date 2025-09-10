"""
Main application combining feed handler and order router
"""

import asyncio
import time
import threading
from feed.handler import FeedHandler, FeedConfig
from feed.decoder import MessageType
from router.order_router import OrderRouter, RouterConfig, OrderSide, OrderType

class TradingSystem:
    """Complete trading system with feed handler and order router"""
    
    def __init__(self):
        # Configure feed handler
        feed_config = FeedConfig(
            protocol="ITCH",
            host="localhost",
            port=12345,
            buffer_size=65536,
            worker_threads=2
        )
        
        self.feed_handler = FeedHandler(feed_config)
        
        # Configure order router
        router_config = RouterConfig(
            destinations=["EXCHANGE1", "EXCHANGE2"],
            worker_threads=2,
            max_orders_per_second=10000
        )
        
        self.order_router = OrderRouter(router_config)
        
        # Strategy state
        self.position = 0
        self.max_position = 1000
        
    async def start(self):
        """Start trading system"""
        
        # Register feed callbacks
        self.feed_handler.register_default_callback(self.on_market_data)
        
        # Start components
        await self.feed_handler.start()
        self.order_router.start()
        
        print("Trading system started")
        
    def stop(self):
        """Stop trading system"""
        self.feed_handler.stop()
        self.order_router.stop()
        print("Trading system stopped")
        
    def on_market_data(self, msg):
        """Handle market data"""
        
        # Simple strategy: buy on dips, sell on rallies
        if msg.message_type == MessageType.TRADE:
            if msg.trade_price and msg.trade_price < 99.5 and self.position < self.max_position:
                # Buy signal
                order_id = self.order_router.submit_order(
                    symbol=msg.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=msg.trade_price + 0.01,
                    quantity=100
                )
                
                if order_id:
                    self.position += 100
                    
            elif msg.trade_price and msg.trade_price > 100.5 and self.position > -self.max_position:
                # Sell signal
                order_id = self.order_router.submit_order(
                    symbol=msg.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=msg.trade_price - 0.01,
                    quantity=100
                )
                
                if order_id:
                    self.position -= 100
                    
    def print_statistics(self):
        """Print system statistics"""
        
        # Feed statistics
        feed_stats = self.feed_handler.get_statistics()
        print("\nFeed Handler Statistics:")
        print(f"  Messages received: {feed_stats['messages_received']}")
        print(f"  Messages decoded: {feed_stats['messages_decoded']}")
        print(f"  Messages processed: {feed_stats['messages_processed']}")
        print(f"  Decode errors: {feed_stats['decode_errors']}")
        
        if feed_stats['latency']['count'] > 0:
            print(f"  Decode latency p50: {feed_stats['latency']['p50']/1000:.1f} μs")
            print(f"  Decode latency p99: {feed_stats['latency']['p99']/1000:.1f} μs")
            
        # Router statistics
        router_stats = self.order_router.get_statistics()
        print("\nOrder Router Statistics:")
        print(f"  Orders submitted: {router_stats['submitted_orders']}")
        print(f"  Orders sent: {router_stats['sent_orders']}")
        print(f"  Orders failed: {router_stats['failed_orders']}")
        
        if router_stats['submission_latency']['count'] > 0:
            print(f"  Submission latency p50: {router_stats['submission_latency']['p50']/1000:.1f} μs")
            print(f"  Submission latency p99: {router_stats['submission_latency']['p99']/1000:.1f} μs")
            
        if router_stats['wire_latency']['count'] > 0:
            print(f"  Wire latency p50: {router_stats['wire_latency']['p50']/1000:.1f} μs")
            print(f"  Wire latency p99: {router_stats['wire_latency']['p99']/1000:.1f} μs")
            
        print(f"\nCurrent position: {self.position}")

async def main():
    """Main entry point"""
    
    system = TradingSystem()
    
    try:
        await system.start()
        
        # Run for a while and print statistics
        for i in range(10):
            await asyncio.sleep(1)
            system.print_statistics()
            
    finally:
        system.stop()

if __name__ == "__main__":
    asyncio.run(main())
