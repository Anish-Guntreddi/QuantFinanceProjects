"""
Test suite for execution handling and slippage models.

Tests order execution, slippage calculation, commission models, and fill generation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from execution import (
    SlippageModel, LinearSlippageModel, SquareRootSlippageModel,
    CommissionModel, FixedCommissionModel, PercentageCommissionModel, TieredCommissionModel,
    SimulatedExecutionHandler, RealisticExecutionHandler
)
from events import OrderEvent, FillEvent, MarketEvent


class TestSlippageModels:
    """Test slippage model implementations."""
    
    def create_test_order(self, quantity=100, direction="BUY"):
        """Create test order event."""
        return OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="MARKET",
            quantity=quantity,
            direction=direction
        )
    
    def create_test_market_data(self):
        """Create test market data."""
        return MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=149.95,
            ask=150.05,
            last=150.00,
            volume=1000000,
            close=150.00
        )
    
    def test_linear_slippage_model(self):
        """Test LinearSlippageModel calculation."""
        model = LinearSlippageModel(
            base_slippage_bps=10.0,  # 10 bps base
            impact_coefficient=0.1
        )
        
        order = self.create_test_order(quantity=1000)  # Larger order
        market_data = self.create_test_market_data()
        
        slippage = model.calculate_slippage(order, market_data)
        
        # Should be positive for buy orders (unfavorable)
        assert slippage > 0
        
        # Should include base slippage and size impact
        expected_base = 150.0 * (10.0 / 10000)  # 0.15
        assert slippage >= expected_base
    
    def test_linear_slippage_direction(self):
        """Test slippage direction for buy vs sell orders."""
        model = LinearSlippageModel(base_slippage_bps=10.0)
        market_data = self.create_test_market_data()
        
        buy_order = self.create_test_order(direction="BUY")
        sell_order = self.create_test_order(direction="SELL")
        
        buy_slippage = model.calculate_slippage(buy_order, market_data)
        sell_slippage = model.calculate_slippage(sell_order, market_data)
        
        # Buy slippage should be positive (price goes up)
        assert buy_slippage > 0
        # Sell slippage should be negative (price goes down)  
        assert sell_slippage < 0
        # Magnitudes should be similar
        assert abs(buy_slippage) == abs(sell_slippage)
    
    def test_square_root_slippage_model(self):
        """Test SquareRootSlippageModel calculation."""
        model = SquareRootSlippageModel(
            temporary_impact_coef=0.1,
            permanent_impact_coef=0.05,
            volatility_factor=1.0
        )
        
        order = self.create_test_order(quantity=10000)  # Large order
        market_data = self.create_test_market_data()
        
        slippage = model.calculate_slippage(order, market_data, volatility=0.02)
        
        assert isinstance(slippage, float)
        assert slippage != 0  # Should have some impact
    
    def test_slippage_scales_with_size(self):
        """Test that slippage increases with order size."""
        model = LinearSlippageModel()
        market_data = self.create_test_market_data()
        
        small_order = self.create_test_order(quantity=100)
        large_order = self.create_test_order(quantity=10000)
        
        small_slippage = abs(model.calculate_slippage(small_order, market_data))
        large_slippage = abs(model.calculate_slippage(large_order, market_data))
        
        # Large order should have more slippage
        assert large_slippage > small_slippage
    
    def test_slippage_with_no_volume(self):
        """Test slippage calculation when no volume data."""
        model = LinearSlippageModel()
        
        order = self.create_test_order()
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            close=150.0,
            volume=None  # No volume data
        )
        
        slippage = model.calculate_slippage(order, market_data)
        
        # Should still calculate some slippage (default minimal impact)
        assert isinstance(slippage, float)
        assert slippage != 0


class TestCommissionModels:
    """Test commission model implementations."""
    
    def create_test_order(self, quantity=100):
        """Create test order."""
        return OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="MARKET",
            quantity=quantity,
            direction="BUY"
        )
    
    def test_fixed_commission_model(self):
        """Test FixedCommissionModel calculation."""
        model = FixedCommissionModel(
            commission_per_share=0.01,
            min_commission=5.0
        )
        
        order = self.create_test_order(quantity=100)
        commission = model.calculate_commission(order, 150.0)
        
        # Should be max(100 * 0.01, 5.0) = 5.0
        assert commission == 5.0
        
        # Test with large order
        large_order = self.create_test_order(quantity=1000)
        large_commission = model.calculate_commission(large_order, 150.0)
        
        # Should be 1000 * 0.01 = 10.0
        assert large_commission == 10.0
    
    def test_percentage_commission_model(self):
        """Test PercentageCommissionModel calculation."""
        model = PercentageCommissionModel(
            commission_rate=0.001,  # 0.1%
            min_commission=2.0
        )
        
        order = self.create_test_order(quantity=100)
        commission = model.calculate_commission(order, 150.0)
        
        # Notional = 100 * 150 = 15,000
        # Commission = 15,000 * 0.001 = 15.0
        assert commission == 15.0
        
        # Test minimum commission
        small_order = self.create_test_order(quantity=1)
        small_commission = model.calculate_commission(small_order, 150.0)
        
        # Should be minimum commission
        assert small_commission == 2.0
    
    def test_tiered_commission_model(self):
        """Test TieredCommissionModel calculation."""
        tiers = {
            0: 0.001,      # 0.1% for first tier
            100000: 0.0005, # 0.05% for $100K+
            500000: 0.0003  # 0.03% for $500K+
        }
        
        model = TieredCommissionModel(
            tiers=tiers,
            tier_type='value',
            min_commission=1.0
        )
        
        # Small order (first tier)
        small_order = self.create_test_order(quantity=100)
        small_commission = model.calculate_commission(small_order, 100.0)
        # Notional = 10,000, rate = 0.001, commission = 10.0
        assert small_commission == 10.0
        
        # Simulate reaching higher tier
        model.cumulative_value = 200000  # Already traded $200K
        
        medium_order = self.create_test_order(quantity=1000)
        medium_commission = model.calculate_commission(medium_order, 100.0)
        # Now at second tier (0.0005), commission = 100,000 * 0.0005 = 50.0
        assert medium_commission == 50.0
    
    def test_tiered_commission_volume_based(self):
        """Test volume-based tiered commissions."""
        tiers = {
            0: 0.01,     # $0.01 per share first tier
            10000: 0.005  # $0.005 per share for 10K+ shares
        }
        
        model = TieredCommissionModel(
            tiers=tiers,
            tier_type='volume',
            min_commission=1.0
        )
        
        # First order (first tier)
        order1 = self.create_test_order(quantity=1000)
        commission1 = model.calculate_commission(order1, 100.0)
        assert commission1 == 10.0  # 1000 * 0.01
        
        # Simulate volume accumulation
        model.cumulative_volume = 15000  # Already traded 15K shares
        
        # Next order should use second tier
        order2 = self.create_test_order(quantity=1000)
        commission2 = model.calculate_commission(order2, 100.0)
        assert commission2 == 5.0  # 1000 * 0.005


class TestSimulatedExecutionHandler:
    """Test SimulatedExecutionHandler functionality."""
    
    def create_test_components(self):
        """Create test execution components."""
        slippage_model = LinearSlippageModel(base_slippage_bps=5.0)
        commission_model = FixedCommissionModel(commission_per_share=0.01)
        
        handler = SimulatedExecutionHandler(
            slippage_model=slippage_model,
            commission_model=commission_model,
            fill_probability=1.0,  # Always fill for testing
            partial_fill_probability=0.0  # No partial fills
        )
        
        return handler
    
    def test_market_order_execution(self):
        """Test market order execution."""
        handler = self.create_test_components()
        
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=149.95,
            ask=150.05,
            last=150.00,
            volume=1000000
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is not None
        assert isinstance(fill, FillEvent)
        assert fill.symbol == "AAPL"
        assert fill.quantity == 100  # Positive for buy
        assert fill.fill_price >= 150.05  # Should fill at ask + slippage for buy
        assert fill.commission > 0
        assert fill.order_id == order.order_id
    
    def test_sell_order_execution(self):
        """Test sell order execution."""
        handler = self.create_test_components()
        
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="MARKET",
            quantity=100,
            direction="SELL"
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=149.95,
            ask=150.05,
            last=150.00,
            volume=1000000
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is not None
        assert fill.quantity == -100  # Negative for sell
        assert fill.fill_price <= 149.95  # Should fill at bid - slippage for sell
    
    def test_limit_order_execution_favorable(self):
        """Test limit order execution when price is favorable."""
        handler = self.create_test_components()
        
        # Buy limit order with favorable market price
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="LIMIT",
            quantity=100,
            direction="BUY",
            price=150.00  # Limit price
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            last=149.00  # Market price below limit
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is not None
        assert fill.fill_price <= 150.00  # Should fill at or below limit
    
    def test_limit_order_no_execution_unfavorable(self):
        """Test limit order not executing when price is unfavorable."""
        handler = self.create_test_components()
        
        # Buy limit order with unfavorable market price
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="LIMIT",
            quantity=100,
            direction="BUY",
            price=150.00  # Limit price
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            last=155.00  # Market price above limit
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is None  # Should not fill
    
    def test_stop_order_triggering(self):
        """Test stop order triggering."""
        handler = self.create_test_components()
        
        # Stop loss order (sell when price falls)
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="STOP",
            quantity=100,
            direction="SELL",
            stop_price=145.00  # Stop price
        )
        
        # Market price triggers stop
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            last=144.00,  # Below stop price
            bid=143.95,
            ask=144.05
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is not None
        assert fill.quantity == -100  # Sell order
    
    def test_stop_order_not_triggered(self):
        """Test stop order not triggering."""
        handler = self.create_test_components()
        
        # Stop loss order
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="STOP",
            quantity=100,
            direction="SELL",
            stop_price=145.00
        )
        
        # Market price above stop
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            last=150.00  # Above stop price
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is None  # Should not trigger
    
    def test_partial_fill_execution(self):
        """Test partial fill execution."""
        handler = SimulatedExecutionHandler(
            slippage_model=LinearSlippageModel(),
            commission_model=FixedCommissionModel(),
            fill_probability=1.0,
            partial_fill_probability=1.0  # Always partial fill
        )
        
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="MARKET",
            quantity=1000,
            direction="BUY"
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            ask=150.00,
            volume=1000000
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is not None
        # Should be partially filled (between 30-90% of original quantity)
        assert 300 <= fill.quantity <= 900
        assert fill.quantity < order.quantity
    
    def test_order_rejection(self):
        """Test order rejection."""
        handler = SimulatedExecutionHandler(
            slippage_model=LinearSlippageModel(),
            commission_model=FixedCommissionModel(),
            fill_probability=0.0  # Never fill
        )
        
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            ask=150.00
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is None  # Should be rejected
    
    def test_execution_statistics(self):
        """Test execution statistics tracking."""
        handler = self.create_test_components()
        
        # Execute some orders
        for i in range(5):
            order = OrderEvent(
                symbol="AAPL",
                timestamp=datetime.now(),
                order_type="MARKET",
                quantity=100,
                direction="BUY"
            )
            
            market_data = MarketEvent(
                symbol="AAPL",
                timestamp=datetime.now(),
                ask=150.00
            )
            
            handler.execute_order(order, market_data)
        
        stats = handler.get_execution_stats()
        
        assert stats['total_orders'] == 5
        assert stats['filled_orders'] == 5  # All should fill with probability 1.0
        assert stats['fill_rate'] == 1.0
        assert stats['rejected_orders'] == 0
    
    def test_execution_delay(self):
        """Test execution delay implementation."""
        delay = timedelta(milliseconds=100)
        handler = SimulatedExecutionHandler(
            slippage_model=LinearSlippageModel(),
            commission_model=FixedCommissionModel(),
            execution_delay=delay
        )
        
        order_time = datetime.now()
        order = OrderEvent(
            symbol="AAPL",
            timestamp=order_time,
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=order_time,
            ask=150.00
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is not None
        # Fill timestamp should be after order timestamp + delay
        assert fill.timestamp > order_time
        time_diff = fill.timestamp - order_time
        assert time_diff >= delay


class TestRealisticExecutionHandler:
    """Test RealisticExecutionHandler advanced features."""
    
    def test_latency_simulation(self):
        """Test latency simulation in realistic handler."""
        handler = RealisticExecutionHandler(
            slippage_model=LinearSlippageModel(),
            commission_model=FixedCommissionModel(),
            latency_mean=0.001,  # 1ms
            latency_std=0.0005   # 0.5ms std
        )
        
        order_time = datetime.now()
        order = OrderEvent(
            symbol="AAPL",
            timestamp=order_time,
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=order_time,
            ask=150.00,
            volume=1000000
        )
        
        fill = handler.execute_order(order, market_data)
        
        assert fill is not None
        # Should have some latency
        assert fill.timestamp > order_time
        
        # Check latency metadata
        assert 'latency_ms' in fill.metadata
        assert fill.metadata['latency_ms'] >= 0
    
    def test_market_impact_history(self):
        """Test market impact history tracking."""
        handler = RealisticExecutionHandler(
            slippage_model=SquareRootSlippageModel(),
            commission_model=FixedCommissionModel()
        )
        
        # Execute multiple orders to build history
        for i in range(3):
            order = OrderEvent(
                symbol="AAPL",
                timestamp=datetime.now(),
                order_type="MARKET",
                quantity=1000 * (i + 1),  # Increasing size
                direction="BUY"
            )
            
            market_data = MarketEvent(
                symbol="AAPL",
                timestamp=datetime.now(),
                ask=150.00,
                volume=1000000
            )
            
            fill = handler.execute_order(order, market_data)
            if fill:
                assert 'historical_impact' in fill.metadata
    
    def test_order_book_depth_simulation(self):
        """Test order book depth simulation."""
        handler = RealisticExecutionHandler(
            slippage_model=SquareRootSlippageModel(),
            commission_model=FixedCommissionModel()
        )
        
        order = OrderEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_type="MARKET",
            quantity=10000,  # Large order
            direction="BUY"
        )
        
        market_data = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            ask=150.00,
            volume=50000  # Limited volume
        )
        
        fill = handler.execute_order(order, market_data)
        
        if fill:
            assert 'available_liquidity' in fill.metadata
            assert 'order_book_depth' in fill.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])