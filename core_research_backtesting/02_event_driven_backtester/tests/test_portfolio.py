"""
Test suite for portfolio management functionality.

Tests position tracking, P&L calculation, risk management, and order generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio import Portfolio, Position, Trade, RiskManager
from events import MarketEvent, SignalEvent, FillEvent


class TestPosition:
    """Test Position class functionality."""
    
    def test_position_creation(self):
        """Test position creation and basic properties."""
        position = Position(symbol="AAPL")
        
        assert position.symbol == "AAPL"
        assert position.quantity == 0.0
        assert position.avg_price == 0.0
        assert position.market_value == 0.0
        assert position.cost_basis == 0.0
        assert position.total_pnl == 0.0
    
    def test_position_market_value_calculation(self):
        """Test market value calculation."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_price=155.0
        )
        
        assert position.market_value == 15500.0  # 100 * 155
        assert position.cost_basis == 15000.0    # 100 * 150
    
    def test_position_update_market_price(self):
        """Test market price update and unrealized P&L."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0
        )
        
        timestamp = datetime.now()
        position.update_market_price(155.0, timestamp)
        
        assert position.market_price == 155.0
        assert position.last_update == timestamp
        assert position.unrealized_pnl == 500.0  # (155 - 150) * 100
    
    def test_position_pnl_calculation(self):
        """Test P&L calculation."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_price=155.0,
            realized_pnl=200.0
        )
        
        position.update_market_price(155.0, datetime.now())
        
        expected_unrealized = 500.0  # (155 - 150) * 100
        expected_total = 200.0 + 500.0  # realized + unrealized
        
        assert position.unrealized_pnl == expected_unrealized
        assert position.total_pnl == expected_total


class TestTrade:
    """Test Trade class functionality."""
    
    def test_trade_creation(self):
        """Test trade creation and P&L calculation."""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=2)
        
        trade = Trade(
            symbol="AAPL",
            entry_time=entry_time,
            exit_time=exit_time,
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            commission=2.0,
            slippage=1.0
        )
        
        assert trade.duration == timedelta(hours=2)
        assert trade.pnl == 497.0  # (155-150)*100 - 2 - 1
        assert trade.return_pct == 0.033133  # 497 / (100*150)
        assert trade.duration_hours == 2.0
    
    def test_trade_negative_pnl(self):
        """Test trade with negative P&L."""
        trade = Trade(
            symbol="AAPL",
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=1),
            quantity=100,
            entry_price=150.0,
            exit_price=145.0,  # Loss
            commission=2.0
        )
        
        expected_pnl = (145.0 - 150.0) * 100 - 2.0  # -502
        assert trade.pnl == expected_pnl
        assert trade.return_pct < 0


class TestRiskManager:
    """Test RiskManager functionality."""
    
    def test_position_limit_check(self):
        """Test position limit checking."""
        risk_manager = RiskManager(max_position_size=0.1)  # 10% max
        
        portfolio_value = 100000
        
        # Valid position (5% of portfolio)
        assert risk_manager.check_position_limit("AAPL", 50, portfolio_value) == True
        
        # Invalid position (15% of portfolio)
        assert risk_manager.check_position_limit("AAPL", 150, portfolio_value) == False
    
    def test_leverage_limit_check(self):
        """Test leverage limit checking."""
        risk_manager = RiskManager(max_leverage=2.0)
        
        portfolio_value = 100000
        
        # Valid leverage (1.5x)
        assert risk_manager.check_leverage_limit(150000, portfolio_value) == True
        
        # Invalid leverage (2.5x)
        assert risk_manager.check_leverage_limit(250000, portfolio_value) == False
    
    def test_drawdown_limit_check(self):
        """Test drawdown limit checking."""
        risk_manager = RiskManager(max_drawdown=0.2)  # 20% max
        
        # Set peak value
        assert risk_manager.check_drawdown_limit(100000) == True
        assert risk_manager.peak_portfolio_value == 100000
        
        # Test acceptable drawdown (10%)
        assert risk_manager.check_drawdown_limit(90000) == True
        assert risk_manager.current_drawdown == 0.1
        
        # Test excessive drawdown (25%)
        assert risk_manager.check_drawdown_limit(75000) == False
        assert risk_manager.current_drawdown == 0.25
    
    def test_symbol_specific_limits(self):
        """Test symbol-specific position limits."""
        position_limits = {'AAPL': 0.05, 'GOOGL': 0.08}  # 5% and 8% limits
        risk_manager = RiskManager(
            max_position_size=0.1,
            position_limits=position_limits
        )
        
        portfolio_value = 100000
        
        # AAPL within symbol limit (3%)
        assert risk_manager.check_position_limit("AAPL", 30, portfolio_value) == True
        
        # AAPL exceeds symbol limit (7% > 5%)
        assert risk_manager.check_position_limit("AAPL", 70, portfolio_value) == False
        
        # GOOGL within symbol limit (6% < 8%)
        assert risk_manager.check_position_limit("GOOGL", 60, portfolio_value) == True


class TestPortfolio:
    """Test Portfolio class functionality."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_capital=100000)
        
        assert portfolio.initial_capital == 100000
        assert portfolio.current_cash == 100000
        assert portfolio.total_portfolio_value == 100000
        assert portfolio.total_pnl == 0.0
        assert len(portfolio.positions) == 0
    
    def test_get_position(self):
        """Test position retrieval."""
        portfolio = Portfolio()
        
        # Getting non-existent position should create it
        position = portfolio.get_position("AAPL")
        assert position.symbol == "AAPL"
        assert position.quantity == 0.0
        
        # Should return same position object
        same_position = portfolio.get_position("AAPL")
        assert same_position is position
    
    def test_market_value_update(self):
        """Test market value updates."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create market event
        market_event = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            close=150.0,
            last=150.0
        )
        
        # Set position first
        portfolio.get_position("AAPL").quantity = 100
        portfolio.get_position("AAPL").avg_price = 145.0
        
        # Update market value
        portfolio.update_market_value(market_event)
        
        position = portfolio.get_position("AAPL")
        assert position.market_price == 150.0
        assert position.unrealized_pnl == 500.0  # (150-145)*100
    
    def test_fill_processing_new_position(self):
        """Test fill processing for new position."""
        portfolio = Portfolio(initial_capital=100000)
        
        fill_event = FillEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_id="test-123",
            quantity=100,  # Buy 100 shares
            fill_price=150.0,
            commission=1.0
        )
        
        initial_cash = portfolio.current_cash
        portfolio.update_fill(fill_event)
        
        position = portfolio.get_position("AAPL")
        assert position.quantity == 100
        assert position.avg_price == 150.0
        
        # Cash should decrease by fill value + commission
        expected_cash = initial_cash - (100 * 150.0) - 1.0
        assert portfolio.current_cash == expected_cash
    
    def test_fill_processing_add_to_position(self):
        """Test fill processing when adding to existing position."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Initial position
        initial_fill = FillEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_id="test-1",
            quantity=100,
            fill_price=150.0,
            commission=1.0
        )
        portfolio.update_fill(initial_fill)
        
        # Add to position
        additional_fill = FillEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_id="test-2",
            quantity=50,  # Buy 50 more
            fill_price=155.0,
            commission=1.0
        )
        portfolio.update_fill(additional_fill)
        
        position = portfolio.get_position("AAPL")
        assert position.quantity == 150
        
        # Check weighted average price: (100*150 + 50*155) / 150 = 151.67
        expected_avg_price = (100 * 150.0 + 50 * 155.0) / 150
        assert abs(position.avg_price - expected_avg_price) < 0.01
    
    def test_fill_processing_close_position(self):
        """Test fill processing when closing position."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open position
        open_fill = FillEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_id="test-1",
            quantity=100,
            fill_price=150.0,
            commission=1.0
        )
        portfolio.update_fill(open_fill)
        
        # Close position
        close_fill = FillEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_id="test-2",
            quantity=-100,  # Sell 100 shares
            fill_price=155.0,
            commission=1.0
        )
        portfolio.update_fill(close_fill)
        
        position = portfolio.get_position("AAPL")
        assert position.quantity == 0
        assert position.avg_price == 0.0
        
        # Check realized P&L: (155-150)*100 = 500, minus commissions
        expected_realized_pnl = 500.0
        assert abs(position.realized_pnl - expected_realized_pnl) < 0.01
        
        # Should have created a completed trade
        assert len(portfolio.closed_trades) == 1
        trade = portfolio.closed_trades[0]
        assert trade.symbol == "AAPL"
        assert trade.quantity == 100
        assert trade.entry_price == 150.0
        assert trade.exit_price == 155.0
    
    def test_risk_limit_checking(self):
        """Test risk limit checking for signals."""
        risk_manager = RiskManager(max_position_size=0.1)
        portfolio = Portfolio(initial_capital=100000, risk_manager=risk_manager)
        
        # Valid signal
        valid_signal = SignalEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type="LONG",
            strength=0.8,
            target_position=50  # Small position
        )
        
        risk_ok, reasons = portfolio.check_risk_limits(valid_signal)
        assert risk_ok == True
        assert len(reasons) == 0
        
        # Set market price for position
        portfolio.get_position("AAPL").market_price = 200.0
        
        # Invalid signal (too large position)
        invalid_signal = SignalEvent(
            symbol="AAPL", 
            timestamp=datetime.now(),
            signal_type="LONG",
            strength=0.8,
            target_position=600  # Large position: 600 * 200 = 120,000 > 10% of 100,000
        )
        
        risk_ok, reasons = portfolio.check_risk_limits(invalid_signal)
        assert risk_ok == False
        assert len(reasons) > 0
    
    def test_order_generation(self):
        """Test order generation from signals."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Set market price
        portfolio.get_position("AAPL").market_price = 150.0
        
        signal = SignalEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type="LONG",
            strength=0.8,
            target_position=100
        )
        
        orders = portfolio.generate_orders(signal)
        
        assert len(orders) == 1
        order = orders[0]
        assert order.symbol == "AAPL"
        assert order.direction == "BUY"
        assert order.quantity == 100
        assert order.order_type == "MARKET"
    
    def test_order_generation_position_change(self):
        """Test order generation when changing existing position."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Set existing position
        position = portfolio.get_position("AAPL")
        position.quantity = 50
        position.market_price = 150.0
        
        # Signal to increase position to 150
        signal = SignalEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type="LONG",
            strength=0.8,
            target_position=150
        )
        
        orders = portfolio.generate_orders(signal)
        
        assert len(orders) == 1
        order = orders[0]
        assert order.direction == "BUY"
        assert order.quantity == 100  # 150 - 50 = 100
    
    def test_order_generation_insufficient_cash(self):
        """Test order generation with insufficient cash."""
        portfolio = Portfolio(initial_capital=1000)  # Small capital
        
        portfolio.get_position("AAPL").market_price = 150.0
        
        # Signal for large position that exceeds cash
        signal = SignalEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type="LONG",
            strength=0.8,
            target_position=100  # Would cost 15,000 but only have 1,000 cash
        )
        
        orders = portfolio.generate_orders(signal)
        
        if orders:  # If any order generated, should be reduced size
            order = orders[0]
            estimated_cost = order.quantity * 150.0
            assert estimated_cost <= 1000 * 0.95  # Should respect cash buffer
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Add position
        fill = FillEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            order_id="test-1",
            quantity=100,
            fill_price=150.0,
            commission=1.0
        )
        portfolio.update_fill(fill)
        
        # Update market price
        market_event = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            close=155.0
        )
        portfolio.update_market_value(market_event)
        
        summary = portfolio.get_portfolio_summary()
        
        assert 'timestamp' in summary
        assert 'cash' in summary
        assert 'total_value' in summary
        assert 'total_pnl' in summary
        assert 'positions' in summary
        assert len(summary['positions']) == 1
        
        # Check position in summary
        pos_summary = summary['positions'][0]
        assert pos_summary['symbol'] == "AAPL"
        assert pos_summary['quantity'] == 100
        assert pos_summary['market_value'] == 15500  # 100 * 155


if __name__ == "__main__":
    pytest.main([__file__, "-v"])