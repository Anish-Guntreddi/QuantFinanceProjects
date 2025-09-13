"""
Portfolio management system for event-driven backtesting.

This module handles position tracking, P&L calculation, risk metrics,
order sizing, and portfolio-level risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from events import MarketEvent, SignalEvent, OrderEvent, FillEvent, RiskEvent

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a single instrument."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_price: float = 0.0
    last_update: Optional[datetime] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.market_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return abs(self.quantity) * self.avg_price
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def update_market_price(self, price: float, timestamp: datetime) -> None:
        """Update market price and unrealized P&L."""
        self.market_price = price
        self.last_update = timestamp
        
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade (round trip)."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    entry_price: float
    exit_price: float
    commission: float = 0.0
    slippage: float = 0.0
    duration: Optional[timedelta] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.duration = self.exit_time - self.entry_time
        
    @property
    def pnl(self) -> float:
        """Net P&L of the trade."""
        gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        return gross_pnl - self.commission - abs(self.slippage)
    
    @property
    def return_pct(self) -> float:
        """Percentage return of the trade."""
        if self.entry_price == 0:
            return 0.0
        return self.pnl / (abs(self.quantity) * self.entry_price)
    
    @property
    def duration_hours(self) -> float:
        """Duration of trade in hours."""
        return self.duration.total_seconds() / 3600 if self.duration else 0.0


class RiskManager:
    """Handles portfolio-level risk controls."""
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # 10% of portfolio
        max_leverage: float = 1.0,
        max_drawdown: float = 0.2,  # 20%
        var_limit: float = 0.05,  # 5% VaR
        position_limits: Optional[Dict[str, float]] = None
    ):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        self.position_limits = position_limits or {}
        
        # Risk metrics tracking
        self.daily_pnl_history: List[float] = []
        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0
        
    def check_position_limit(self, symbol: str, quantity: float, portfolio_value: float) -> bool:
        """Check if position size is within limits."""
        position_value = abs(quantity) * 100  # Approximate position value
        position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # General position size limit
        if position_weight > self.max_position_size:
            logger.warning(f"Position size limit exceeded for {symbol}: {position_weight:.2%} > {self.max_position_size:.2%}")
            return False
            
        # Symbol-specific limits
        symbol_limit = self.position_limits.get(symbol)
        if symbol_limit and position_weight > symbol_limit:
            logger.warning(f"Symbol-specific limit exceeded for {symbol}: {position_weight:.2%} > {symbol_limit:.2%}")
            return False
            
        return True
    
    def check_leverage_limit(self, total_exposure: float, portfolio_value: float) -> bool:
        """Check if leverage is within limits."""
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if leverage > self.max_leverage:
            logger.warning(f"Leverage limit exceeded: {leverage:.2f} > {self.max_leverage:.2f}")
            return False
            
        return True
    
    def check_drawdown_limit(self, portfolio_value: float) -> bool:
        """Check if drawdown is within limits."""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            
        if self.current_drawdown > self.max_drawdown:
            logger.error(f"Drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}")
            return False
            
        return True
    
    def calculate_var(self, confidence: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(self.daily_pnl_history) < 30:  # Need at least 30 days
            return 0.0
            
        return np.percentile(self.daily_pnl_history, confidence * 100)
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L history for risk calculations."""
        self.daily_pnl_history.append(pnl)
        
        # Keep only last 252 days (1 year)
        if len(self.daily_pnl_history) > 252:
            self.daily_pnl_history = self.daily_pnl_history[-252:]


class Portfolio:
    """
    Portfolio management system for tracking positions, P&L, and generating orders.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.001,
        risk_manager: Optional[RiskManager] = None
    ):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.commission_per_share = commission_per_share
        self.risk_manager = risk_manager or RiskManager()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.open_trades: Dict[str, List[Dict]] = defaultdict(list)  # Track partial fills
        
        # P&L tracking
        self.daily_pnl: Dict[datetime, float] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.total_commission_paid = 0.0
        self.total_slippage_paid = 0.0
        
        # Performance metrics
        self.metrics_cache: Dict[str, Any] = {}
        self.last_metrics_update: Optional[datetime] = None
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    @property
    def total_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_cash + positions_value
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.total_portfolio_value - self.initial_capital
    
    @property
    def total_exposure(self) -> float:
        """Calculate total notional exposure."""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_market_value(self, event: MarketEvent) -> None:
        """Update market values based on market event."""
        position = self.get_position(event.symbol)
        
        # Use the most appropriate price
        market_price = event.last or event.close or event.mid_price
        if market_price:
            position.update_market_price(market_price, event.timestamp)
            
        # Update equity curve
        portfolio_value = self.total_portfolio_value
        self.equity_curve.append((event.timestamp, portfolio_value))
        
        # Update risk manager
        self.risk_manager.check_drawdown_limit(portfolio_value)
    
    def check_risk_limits(self, signal: SignalEvent) -> Tuple[bool, List[str]]:
        """Check if signal passes risk limits."""
        reasons = []
        
        if signal.target_position is None:
            return True, reasons
            
        # Estimate position value (rough approximation)
        position = self.get_position(signal.symbol)
        estimated_price = position.market_price or 100.0  # Fallback price
        estimated_quantity = signal.target_position
        
        # Position size check
        if not self.risk_manager.check_position_limit(
            signal.symbol, 
            estimated_quantity, 
            self.total_portfolio_value
        ):
            reasons.append("Position size limit exceeded")
            
        # Leverage check
        total_exposure = self.total_exposure + abs(estimated_quantity * estimated_price)
        if not self.risk_manager.check_leverage_limit(total_exposure, self.total_portfolio_value):
            reasons.append("Leverage limit exceeded")
            
        # Drawdown check
        if not self.risk_manager.check_drawdown_limit(self.total_portfolio_value):
            reasons.append("Drawdown limit exceeded")
            
        return len(reasons) == 0, reasons
    
    def generate_orders(self, signal: SignalEvent) -> List[OrderEvent]:
        """Generate orders based on trading signal."""
        orders = []
        
        # Check risk limits first
        risk_ok, risk_reasons = self.check_risk_limits(signal)
        if not risk_ok:
            logger.warning(f"Signal for {signal.symbol} blocked by risk limits: {', '.join(risk_reasons)}")
            return orders
            
        current_position = self.get_position(signal.symbol)
        
        if signal.target_position is None:
            logger.warning(f"Signal for {signal.symbol} has no target position")
            return orders
            
        # Calculate required trade
        position_change = signal.target_position - current_position.quantity
        
        if abs(position_change) < 1e-6:  # No meaningful change
            return orders
            
        # Determine order direction and size
        direction = 'BUY' if position_change > 0 else 'SELL'
        quantity = abs(position_change)
        
        # Check if we have enough cash for buy orders
        if direction == 'BUY':
            estimated_cost = quantity * current_position.market_price
            if estimated_cost > self.current_cash * 0.95:  # Keep 5% cash buffer
                # Reduce order size to available cash
                quantity = (self.current_cash * 0.95) / current_position.market_price
                if quantity < 1:  # Not enough cash
                    logger.warning(f"Insufficient cash for {signal.symbol} order")
                    return orders
        
        # Create order
        order = OrderEvent(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            order_type='MARKET',  # Default to market orders
            quantity=quantity,
            direction=direction,
            strategy_id=signal.strategy_id,
            metadata={
                'signal_strength': signal.strength,
                'target_position': signal.target_position,
                'current_position': current_position.quantity
            }
        )
        
        orders.append(order)
        logger.info(f"Generated order: {direction} {quantity:.2f} {signal.symbol}")
        
        return orders
    
    def update_fill(self, fill: FillEvent) -> None:
        """Update portfolio with fill event."""
        position = self.get_position(fill.symbol)
        
        # Calculate new position metrics
        old_quantity = position.quantity
        fill_quantity = fill.quantity  # Already signed (+ for buy, - for sell)
        new_quantity = old_quantity + fill_quantity
        
        # Update average price using weighted average
        if new_quantity == 0:
            # Position closed
            position.avg_price = 0.0
            position.quantity = 0.0
            
            # Record realized P&L
            if old_quantity != 0:
                realized_pnl = -old_quantity * (fill.fill_price - position.avg_price)
                position.realized_pnl += realized_pnl
                
                # Close any open trades
                self._close_trades(fill)
                
        elif old_quantity == 0:
            # New position
            position.quantity = new_quantity
            position.avg_price = fill.fill_price
            
            # Start new trade tracking
            self._start_trade(fill)
            
        elif np.sign(old_quantity) == np.sign(fill_quantity):
            # Adding to position
            total_cost = (old_quantity * position.avg_price) + (fill_quantity * fill.fill_price)
            position.quantity = new_quantity
            position.avg_price = total_cost / new_quantity if new_quantity != 0 else 0.0
            
            # Update trade tracking
            self._update_trade(fill)
            
        else:
            # Reducing position or reversing
            if abs(fill_quantity) <= abs(old_quantity):
                # Partial close
                closed_quantity = -fill_quantity  # Amount being closed
                realized_pnl = closed_quantity * (fill.fill_price - position.avg_price)
                position.realized_pnl += realized_pnl
                position.quantity = new_quantity
                
                # Keep same average price for remaining position
                self._close_partial_trades(fill, closed_quantity)
                
            else:
                # Close and reverse
                # First close the existing position
                close_quantity = -old_quantity
                realized_pnl = close_quantity * (fill.fill_price - position.avg_price)
                position.realized_pnl += realized_pnl
                
                # Then open new position in opposite direction
                remaining_quantity = fill_quantity + old_quantity
                position.quantity = remaining_quantity
                position.avg_price = fill.fill_price
                
                # Close all trades and start new one
                self._close_trades(fill)
                if remaining_quantity != 0:
                    self._start_trade(fill)
        
        # Update cash position
        cash_change = -fill_quantity * fill.fill_price - fill.commission
        self.current_cash += cash_change
        
        # Track costs
        self.total_commission_paid += fill.commission
        self.total_slippage_paid += abs(fill.slippage)
        
        # Update market price
        position.update_market_price(fill.fill_price, fill.timestamp)
        
        logger.info(f"Fill processed: {fill.symbol} {fill.quantity:.2f}@{fill.fill_price:.4f} "
                   f"(pos: {position.quantity:.2f}, cash: ${self.current_cash:,.2f})")
    
    def _start_trade(self, fill: FillEvent) -> None:
        """Start tracking a new trade."""
        trade_record = {
            'entry_time': fill.timestamp,
            'entry_price': fill.fill_price,
            'quantity': fill.quantity,
            'commission': fill.commission
        }
        self.open_trades[fill.symbol].append(trade_record)
    
    def _update_trade(self, fill: FillEvent) -> None:
        """Update existing trade with additional fill."""
        if fill.symbol in self.open_trades and self.open_trades[fill.symbol]:
            # Add to most recent trade
            trade = self.open_trades[fill.symbol][-1]
            
            # Weighted average entry price
            total_quantity = trade['quantity'] + fill.quantity
            total_cost = (trade['quantity'] * trade['entry_price'] + 
                         fill.quantity * fill.fill_price)
            
            trade['entry_price'] = total_cost / total_quantity if total_quantity != 0 else 0
            trade['quantity'] = total_quantity
            trade['commission'] += fill.commission
    
    def _close_trades(self, fill: FillEvent) -> None:
        """Close all open trades for a symbol."""
        if fill.symbol not in self.open_trades:
            return
            
        for trade_record in self.open_trades[fill.symbol]:
            completed_trade = Trade(
                symbol=fill.symbol,
                entry_time=trade_record['entry_time'],
                exit_time=fill.timestamp,
                quantity=trade_record['quantity'],
                entry_price=trade_record['entry_price'],
                exit_price=fill.fill_price,
                commission=trade_record['commission'] + fill.commission,
                slippage=fill.slippage
            )
            self.closed_trades.append(completed_trade)
            
        self.open_trades[fill.symbol].clear()
    
    def _close_partial_trades(self, fill: FillEvent, closed_quantity: float) -> None:
        """Close partial trades FIFO."""
        if fill.symbol not in self.open_trades:
            return
            
        remaining_to_close = abs(closed_quantity)
        trades_to_remove = []
        
        for i, trade_record in enumerate(self.open_trades[fill.symbol]):
            if remaining_to_close <= 0:
                break
                
            trade_quantity = abs(trade_record['quantity'])
            
            if trade_quantity <= remaining_to_close:
                # Close entire trade
                completed_trade = Trade(
                    symbol=fill.symbol,
                    entry_time=trade_record['entry_time'],
                    exit_time=fill.timestamp,
                    quantity=trade_record['quantity'],
                    entry_price=trade_record['entry_price'],
                    exit_price=fill.fill_price,
                    commission=trade_record['commission'] + 
                              (fill.commission * trade_quantity / abs(fill.quantity)),
                    slippage=fill.slippage * trade_quantity / abs(fill.quantity)
                )
                self.closed_trades.append(completed_trade)
                trades_to_remove.append(i)
                remaining_to_close -= trade_quantity
                
            else:
                # Partially close trade
                close_proportion = remaining_to_close / trade_quantity
                
                completed_trade = Trade(
                    symbol=fill.symbol,
                    entry_time=trade_record['entry_time'],
                    exit_time=fill.timestamp,
                    quantity=trade_record['quantity'] * close_proportion,
                    entry_price=trade_record['entry_price'],
                    exit_price=fill.fill_price,
                    commission=trade_record['commission'] * close_proportion + 
                              fill.commission * close_proportion,
                    slippage=fill.slippage * close_proportion
                )
                self.closed_trades.append(completed_trade)
                
                # Update remaining trade
                trade_record['quantity'] *= (1 - close_proportion)
                trade_record['commission'] *= (1 - close_proportion)
                remaining_to_close = 0
        
        # Remove fully closed trades
        for i in sorted(trades_to_remove, reverse=True):
            del self.open_trades[fill.symbol][i]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        positions_summary = []
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                positions_summary.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'market_price': position.market_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'total_pnl': position.total_pnl
                })
        
        return {
            'timestamp': datetime.now(),
            'cash': self.current_cash,
            'total_value': self.total_portfolio_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl / self.initial_capital * 100,
            'total_exposure': self.total_exposure,
            'leverage': self.total_exposure / self.total_portfolio_value if self.total_portfolio_value > 0 else 0,
            'positions': positions_summary,
            'num_positions': len([p for p in self.positions.values() if p.quantity != 0]),
            'total_trades': len(self.closed_trades),
            'total_commission': self.total_commission_paid,
            'total_slippage': self.total_slippage_paid,
            'current_drawdown': self.risk_manager.current_drawdown
        }