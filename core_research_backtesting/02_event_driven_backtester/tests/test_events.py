"""
Test suite for the events module.

Tests event creation, validation, queue operations, and event handlers.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from events import (
    EventType, Event, MarketEvent, SignalEvent, OrderEvent, FillEvent, RiskEvent,
    EventQueue, EventHandler, EventDispatcher
)


class TestEventCreation:
    """Test basic event creation and validation."""
    
    def test_market_event_creation(self):
        """Test MarketEvent creation and properties."""
        timestamp = datetime.now()
        
        market_event = MarketEvent(
            symbol="AAPL",
            timestamp=timestamp,
            bid=150.0,
            ask=150.05,
            last=150.02,
            volume=1000000,
            open=149.5,
            high=151.0,
            low=149.0,
            close=150.02
        )
        
        assert market_event.event_type == EventType.MARKET
        assert market_event.symbol == "AAPL"
        assert market_event.timestamp == timestamp
        assert market_event.bid == 150.0
        assert market_event.ask == 150.05
        assert market_event.mid_price == 150.025
        assert market_event.spread == 0.05
    
    def test_signal_event_validation(self):
        """Test SignalEvent validation."""
        timestamp = datetime.now()
        
        # Valid signal
        signal = SignalEvent(
            symbol="AAPL",
            timestamp=timestamp,
            signal_type="LONG",
            strength=0.8,
            confidence=0.9
        )
        
        assert signal.event_type == EventType.SIGNAL
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        
        # Invalid strength should raise error
        with pytest.raises(ValueError):
            SignalEvent(
                symbol="AAPL",
                timestamp=timestamp,
                signal_type="LONG",
                strength=1.5  # Invalid: > 1
            )
        
        # Invalid confidence should raise error
        with pytest.raises(ValueError):
            SignalEvent(
                symbol="AAPL",
                timestamp=timestamp,
                signal_type="LONG",
                strength=0.5,
                confidence=1.5  # Invalid: > 1
            )
    
    def test_order_event_validation(self):
        """Test OrderEvent validation."""
        timestamp = datetime.now()
        
        # Valid market order
        order = OrderEvent(
            symbol="AAPL",
            timestamp=timestamp,
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        
        assert order.event_type == EventType.ORDER
        assert order.order_id is not None
        assert order.notional_value is None  # No price for market order
        
        # Valid limit order
        limit_order = OrderEvent(
            symbol="AAPL",
            timestamp=timestamp,
            order_type="LIMIT",
            quantity=100,
            direction="BUY",
            price=150.0
        )
        
        assert limit_order.price == 150.0
        assert limit_order.notional_value == 15000.0
        
        # Invalid: limit order without price
        with pytest.raises(ValueError):
            OrderEvent(
                symbol="AAPL",
                timestamp=timestamp,
                order_type="LIMIT",
                quantity=100,
                direction="BUY"
                # Missing price
            )
        
        # Invalid: negative quantity
        with pytest.raises(ValueError):
            OrderEvent(
                symbol="AAPL",
                timestamp=timestamp,
                order_type="MARKET",
                quantity=-100,  # Invalid: negative
                direction="BUY"
            )
    
    def test_fill_event_properties(self):
        """Test FillEvent properties and calculations."""
        timestamp = datetime.now()
        
        fill = FillEvent(
            symbol="AAPL",
            timestamp=timestamp,
            order_id="test-123",
            quantity=100,  # Positive for buy
            fill_price=150.0,
            commission=1.0,
            slippage=0.5,
            market_impact=0.2
        )
        
        assert fill.event_type == EventType.FILL
        assert fill.notional_value == 15000.0
        assert fill.total_cost == 1.7  # commission + slippage + market_impact
        assert abs(fill.cost_bps - 1.133) < 0.01  # (1.7/15000)*10000
    
    def test_risk_event_creation(self):
        """Test RiskEvent creation."""
        timestamp = datetime.now()
        
        risk_event = RiskEvent(
            timestamp=timestamp,
            risk_type="POSITION_LIMIT",
            severity="WARNING",
            message="Position size exceeds limit",
            current_value=0.15,
            limit_value=0.10,
            action_required=True
        )
        
        assert risk_event.event_type == EventType.RISK
        assert risk_event.risk_type == "POSITION_LIMIT"
        assert risk_event.severity == "WARNING"
        assert risk_event.action_required == True


class TestEventQueue:
    """Test EventQueue functionality."""
    
    def test_queue_operations(self):
        """Test basic queue operations."""
        queue = EventQueue()
        
        assert queue.empty()
        assert queue.size() == 0
        
        # Add events
        event1 = MarketEvent("AAPL", datetime.now())
        event2 = MarketEvent("GOOGL", datetime.now())
        
        queue.put(event1)
        queue.put(event2)
        
        assert not queue.empty()
        assert queue.size() == 2
        
        # Get events (should be FIFO for same priority)
        retrieved1 = queue.get()
        retrieved2 = queue.get()
        
        assert retrieved1 == event1
        assert retrieved2 == event2
        assert queue.empty()
    
    def test_priority_ordering(self):
        """Test priority-based ordering."""
        queue = EventQueue()
        
        time_base = datetime.now()
        
        # Add events with different priorities
        low_priority = MarketEvent("AAPL", time_base)
        high_priority = SignalEvent("AAPL", time_base, "LONG", 0.8)
        medium_priority = OrderEvent("AAPL", time_base, "MARKET", 100, "BUY")
        
        # Add in random order
        queue.put(low_priority, priority=3)
        queue.put(high_priority, priority=1)  # Highest priority (lowest number)
        queue.put(medium_priority, priority=2)
        
        # Should come out in priority order
        first = queue.get()
        second = queue.get() 
        third = queue.get()
        
        assert first == high_priority
        assert second == medium_priority
        assert third == low_priority
    
    def test_time_ordering(self):
        """Test time-based ordering for same priority."""
        queue = EventQueue()
        
        time_base = datetime.now()
        
        # Events with same priority but different times
        early_event = MarketEvent("AAPL", time_base)
        late_event = MarketEvent("AAPL", time_base + timedelta(seconds=1))
        
        # Add in reverse chronological order
        queue.put(late_event)
        queue.put(early_event)
        
        # Should come out in chronological order
        first = queue.get()
        second = queue.get()
        
        assert first == early_event
        assert second == late_event
    
    def test_queue_clear(self):
        """Test queue clearing."""
        queue = EventQueue()
        
        # Add some events
        for i in range(5):
            event = MarketEvent(f"STOCK_{i}", datetime.now())
            queue.put(event)
        
        assert queue.size() == 5
        
        queue.clear()
        
        assert queue.empty()
        assert queue.size() == 0
    
    def test_peek_functionality(self):
        """Test peek without removing."""
        queue = EventQueue()
        
        event = MarketEvent("AAPL", datetime.now())
        queue.put(event)
        
        # Peek should return event without removing
        peeked = queue.peek()
        assert peeked == event
        assert queue.size() == 1
        
        # Get should return same event and remove it
        retrieved = queue.get()
        assert retrieved == event
        assert queue.empty()


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self):
        self.handled_events = []
        self.can_handle_types = [EventType.MARKET]
    
    def handle_event(self, event: Event):
        self.handled_events.append(event)
    
    def can_handle(self, event_type: EventType) -> bool:
        return event_type in self.can_handle_types


class TestEventDispatcher:
    """Test EventDispatcher functionality."""
    
    def test_handler_registration(self):
        """Test handler registration and dispatch."""
        dispatcher = EventDispatcher()
        handler = MockEventHandler()
        
        # Register handler
        dispatcher.register_handler(EventType.MARKET, handler)
        
        # Create and dispatch event
        event = MarketEvent("AAPL", datetime.now())
        dispatcher.dispatch(event)
        
        # Check handler received event
        assert len(handler.handled_events) == 1
        assert handler.handled_events[0] == event
    
    def test_multiple_handlers(self):
        """Test multiple handlers for same event type."""
        dispatcher = EventDispatcher()
        handler1 = MockEventHandler()
        handler2 = MockEventHandler()
        
        # Register both handlers
        dispatcher.register_handler(EventType.MARKET, handler1)
        dispatcher.register_handler(EventType.MARKET, handler2)
        
        # Dispatch event
        event = MarketEvent("AAPL", datetime.now())
        dispatcher.dispatch(event)
        
        # Both handlers should receive event
        assert len(handler1.handled_events) == 1
        assert len(handler2.handled_events) == 1
    
    def test_handler_removal(self):
        """Test handler removal."""
        dispatcher = EventDispatcher()
        handler = MockEventHandler()
        
        # Register and remove handler
        dispatcher.register_handler(EventType.MARKET, handler)
        dispatcher.remove_handler(EventType.MARKET, handler)
        
        # Dispatch event
        event = MarketEvent("AAPL", datetime.now())
        dispatcher.dispatch(event)
        
        # Handler should not receive event
        assert len(handler.handled_events) == 0
    
    def test_event_type_filtering(self):
        """Test that handlers only receive appropriate event types."""
        dispatcher = EventDispatcher()
        handler = MockEventHandler()
        handler.can_handle_types = [EventType.MARKET]  # Only market events
        
        dispatcher.register_handler(EventType.MARKET, handler)
        dispatcher.register_handler(EventType.SIGNAL, handler)  # Also register for signals
        
        # Dispatch market event (should be handled)
        market_event = MarketEvent("AAPL", datetime.now())
        dispatcher.dispatch(market_event)
        
        # Dispatch signal event (should not be handled due to can_handle check)
        signal_event = SignalEvent("AAPL", datetime.now(), "LONG", 0.8)
        dispatcher.dispatch(signal_event)
        
        # Only market event should be handled
        assert len(handler.handled_events) == 1
        assert handler.handled_events[0] == market_event


class TestEventComparison:
    """Test event comparison and ordering."""
    
    def test_event_lt_comparison(self):
        """Test less-than comparison for event ordering."""
        time_base = datetime.now()
        
        early_event = MarketEvent("AAPL", time_base, priority=1)
        late_event = MarketEvent("AAPL", time_base + timedelta(seconds=1), priority=1)
        high_priority = MarketEvent("AAPL", time_base, priority=0)
        
        # Time-based comparison (same priority)
        assert early_event < late_event
        assert not (late_event < early_event)
        
        # Priority-based comparison (same time)
        assert high_priority < early_event
        assert not (early_event < high_priority)
    
    def test_event_sorting(self):
        """Test sorting of events."""
        time_base = datetime.now()
        
        events = [
            MarketEvent("C", time_base + timedelta(seconds=2), priority=2),
            MarketEvent("A", time_base, priority=1),
            MarketEvent("B", time_base + timedelta(seconds=1), priority=1),
            MarketEvent("D", time_base, priority=0)  # Highest priority
        ]
        
        sorted_events = sorted(events)
        
        # Should be ordered by: priority first, then time
        assert sorted_events[0].symbol == "D"  # Highest priority
        assert sorted_events[1].symbol == "A"  # Same priority, earliest time
        assert sorted_events[2].symbol == "B"  # Same priority, later time
        assert sorted_events[3].symbol == "C"  # Lowest priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])