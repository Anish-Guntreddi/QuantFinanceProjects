"""
Market data decoder for various protocols
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union
import struct
import time

class MessageType(Enum):
    ADD_ORDER = 'A'
    ORDER_EXECUTED = 'E'
    ORDER_CANCEL = 'X'
    ORDER_REPLACE = 'U'
    TRADE = 'P'
    QUOTE = 'Q'
    IMBALANCE = 'I'
    STATUS = 'H'

@dataclass
class MarketDataMessage:
    """Decoded market data message"""
    message_type: MessageType
    timestamp: int  # Nanoseconds
    sequence: int
    symbol: str
    
    # Order fields
    order_id: Optional[int] = None
    side: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    
    # Trade fields
    trade_id: Optional[int] = None
    trade_price: Optional[float] = None
    trade_quantity: Optional[int] = None
    
    # Quote fields
    bid_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_price: Optional[float] = None
    ask_size: Optional[int] = None

class ITCHDecoder:
    """ITCH protocol decoder (simplified)"""
    
    def __init__(self):
        self.sequence = 0
        
    def decode(self, data: bytes) -> Optional[MarketDataMessage]:
        """Decode ITCH message"""
        if len(data) < 3:
            return None
            
        # Parse header
        msg_type = chr(data[2])
        
        try:
            if msg_type == 'A':  # Add Order
                return self._decode_add_order(data)
            elif msg_type == 'E':  # Order Executed
                return self._decode_order_executed(data)
            elif msg_type == 'X':  # Order Cancel
                return self._decode_order_cancel(data)
            elif msg_type == 'P':  # Trade
                return self._decode_trade(data)
            else:
                return None
        except:
            return None
            
    def _decode_add_order(self, data: bytes) -> MarketDataMessage:
        """Decode add order message"""
        # Simplified parsing (would need actual ITCH spec)
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ADD_ORDER,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            order_id=self.sequence,
            side='B' if self.sequence % 2 == 0 else 'S',
            price=100.0 + (self.sequence % 10) * 0.01,
            quantity=100 * (1 + self.sequence % 5)
        )
    
    def _decode_trade(self, data: bytes) -> MarketDataMessage:
        """Decode trade message"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.TRADE,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            trade_id=self.sequence,
            trade_price=100.0,
            trade_quantity=100
        )
    
    def _decode_order_executed(self, data: bytes) -> MarketDataMessage:
        """Decode order executed message"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ORDER_EXECUTED,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            order_id=self.sequence,
            quantity=100
        )
    
    def _decode_order_cancel(self, data: bytes) -> MarketDataMessage:
        """Decode order cancel message"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ORDER_CANCEL,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            order_id=self.sequence
        )

class FIXDecoder:
    """FIX protocol decoder (simplified)"""
    
    SOH = chr(1)  # FIX delimiter
    
    def __init__(self):
        self.sequence = 0
        
    def decode(self, data: str) -> Optional[MarketDataMessage]:
        """Decode FIX message"""
        fields = {}
        
        # Parse FIX fields
        for field in data.split(self.SOH):
            if '=' in field:
                tag, value = field.split('=', 1)
                fields[tag] = value
                
        # Get message type
        msg_type = fields.get('35', '')
        
        if msg_type == 'D':  # New Order Single
            return self._decode_new_order(fields)
        elif msg_type == '8':  # Execution Report
            return self._decode_execution_report(fields)
        elif msg_type == 'W':  # Market Data Snapshot
            return self._decode_market_data(fields)
        else:
            return None
            
    def _decode_new_order(self, fields: dict) -> MarketDataMessage:
        """Decode new order"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ADD_ORDER,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol=fields.get('55', 'TEST'),
            order_id=int(fields.get('11', 0)),
            side='B' if fields.get('54') == '1' else 'S',
            price=float(fields.get('44', 100.0)),
            quantity=int(fields.get('38', 100))
        )
    
    def _decode_market_data(self, fields: dict) -> MarketDataMessage:
        """Decode market data snapshot"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.QUOTE,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol=fields.get('55', 'TEST'),
            bid_price=float(fields.get('132', 99.99)),
            bid_size=int(fields.get('134', 1000)),
            ask_price=float(fields.get('133', 100.01)),
            ask_size=int(fields.get('135', 1000))
        )
    
    def _decode_execution_report(self, fields: dict) -> MarketDataMessage:
        """Decode execution report"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.TRADE,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol=fields.get('55', 'TEST'),
            trade_id=self.sequence,
            trade_price=float(fields.get('31', 100.0)),
            trade_quantity=int(fields.get('32', 100))
        )
