"""
Main event-driven backtesting engine.

This module orchestrates the entire backtesting process by coordinating
events between data handlers, strategies, portfolio, and execution components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import time
from pathlib import Path
import json

from events import (
    Event, EventQueue, EventType, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, RiskEvent
)
from data_handler import DataHandler
from strategy import Strategy
from portfolio import Portfolio
from execution import ExecutionHandler

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    benchmark: Optional[str] = None
    
    # Risk parameters
    max_drawdown: float = 0.2
    max_position_size: float = 0.1
    max_leverage: float = 1.0
    
    # Execution parameters
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0
    
    # Output parameters
    save_results: bool = True
    output_dir: str = "./results/"
    generate_plots: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'benchmark': self.benchmark,
            'max_drawdown': self.max_drawdown,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage,
            'commission_per_share': self.commission_per_share,
            'slippage_bps': self.slippage_bps,
            'save_results': self.save_results,
            'output_dir': self.output_dir,
            'generate_plots': self.generate_plots
        }


@dataclass
class BacktestResults:
    """Container for backtest results."""
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    duration: timedelta
    
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Cost analysis
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    cost_as_pct_of_pnl: float = 0.0
    
    # Detailed data
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'config': self.config.to_dict(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration.total_seconds(),
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_costs,
            'cost_as_pct_of_pnl': self.cost_as_pct_of_pnl
        }


class EventDrivenBacktester:
    """
    Main event-driven backtesting engine that coordinates all components.
    """
    
    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
        config: BacktestConfig
    ):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.config = config
        
        # Event processing
        self.event_queue = EventQueue()
        self.events_processed = 0
        self.current_time = None
        
        # Results tracking
        self.signals_generated = []
        self.orders_placed = []
        self.fills_executed = []
        self.risk_events = []
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.heartbeat_count = 0
        
        logger.info(f"Event-driven backtester initialized")
        logger.info(f"Strategy: {self.strategy.__class__.__name__}")
        logger.info(f"Period: {config.start_date} to {config.end_date}")
        logger.info(f"Initial capital: ${config.initial_capital:,.2f}")
    
    def run(self) -> BacktestResults:
        """
        Run the complete backtest.
        
        Returns:
            BacktestResults containing all performance metrics and data
        """
        logger.info("Starting backtest execution...")
        self.start_time = datetime.now()
        
        try:
            # Initialize data handler
            self.data_handler.initialize()
            
            # Main event loop
            self._run_event_loop()
            
            # Calculate final results
            results = self._calculate_results()
            
            # Save results if requested
            if self.config.save_results:
                self._save_results(results)
                
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            
            logger.info(f"Backtest completed successfully in {duration.total_seconds():.2f}s")
            logger.info(f"Processed {self.events_processed} events")
            logger.info(f"Final portfolio value: ${results.equity_curve.iloc[-1]:,.2f}")
            logger.info(f"Total return: {results.total_return:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _run_event_loop(self) -> None:
        """Main event processing loop."""
        logger.info("Starting event loop...")
        
        while self.data_handler.continue_backtest:
            try:
                # Update market data
                market_events = self.data_handler.update_bars()
                
                # Add market events to queue
                for event in market_events:
                    self.event_queue.put(event, priority=1)  # High priority for market data
                    
                # Process all events in queue
                while not self.event_queue.empty():
                    event = self.event_queue.get()
                    self._handle_event(event)
                    self.events_processed += 1
                    
                self.heartbeat_count += 1
                
                # Log progress periodically
                if self.heartbeat_count % 1000 == 0:
                    portfolio_value = self.portfolio.total_portfolio_value
                    logger.info(f"Heartbeat {self.heartbeat_count}: Portfolio value ${portfolio_value:,.2f}")
                    
            except Exception as e:
                logger.error(f"Error in event loop: {e}")
                break
                
        logger.info(f"Event loop completed. Processed {self.events_processed} events")
    
    def _handle_event(self, event: Event) -> None:
        """Handle individual events based on type."""
        try:
            if event.event_type == EventType.MARKET:
                self._handle_market_event(event)
            elif event.event_type == EventType.SIGNAL:
                self._handle_signal_event(event)
            elif event.event_type == EventType.ORDER:
                self._handle_order_event(event)
            elif event.event_type == EventType.FILL:
                self._handle_fill_event(event)
            elif event.event_type == EventType.RISK:
                self._handle_risk_event(event)
            else:
                logger.warning(f"Unknown event type: {event.event_type}")
                
        except Exception as e:
            logger.error(f"Error handling {event.event_type} event: {e}")
    
    def _handle_market_event(self, event: MarketEvent) -> None:
        """Process market data events."""
        self.current_time = event.timestamp
        
        # Update portfolio with new market prices
        self.portfolio.update_market_value(event)
        
        # Generate trading signals
        try:
            signals = self.strategy.calculate_signals(event)
            
            for signal in signals:
                self.signals_generated.append({
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'strategy_id': signal.strategy_id
                })
                
                # Add signal to event queue
                self.event_queue.put(signal, priority=2)
                
        except Exception as e:
            logger.error(f"Error generating signals for {event.symbol}: {e}")
    
    def _handle_signal_event(self, event: SignalEvent) -> None:
        """Process trading signals."""
        try:
            # Check risk limits
            risk_ok, risk_reasons = self.portfolio.check_risk_limits(event)
            
            if not risk_ok:
                # Generate risk event
                risk_event = RiskEvent(
                    timestamp=event.timestamp,
                    risk_type='SIGNAL_BLOCKED',
                    severity='WARNING',
                    message=f"Signal blocked for {event.symbol}: {', '.join(risk_reasons)}",
                    action_required=False
                )
                self.event_queue.put(risk_event, priority=3)
                return
            
            # Generate orders from signal
            orders = self.portfolio.generate_orders(event)
            
            for order in orders:
                self.orders_placed.append({
                    'timestamp': order.timestamp,
                    'symbol': order.symbol,
                    'order_type': order.order_type,
                    'quantity': order.quantity,
                    'direction': order.direction,
                    'order_id': order.order_id
                })
                
                # Add order to event queue
                self.event_queue.put(order, priority=3)
                
        except Exception as e:
            logger.error(f"Error handling signal for {event.symbol}: {e}")
    
    def _handle_order_event(self, event: OrderEvent) -> None:
        """Process order placement."""
        try:
            # Get current market data for execution
            latest_bar = self.data_handler.get_latest_bar(event.symbol)
            if latest_bar is None:
                logger.warning(f"No market data available for {event.symbol} order")
                return
            
            # Create market event from latest bar
            market_event = MarketEvent(
                symbol=event.symbol,
                timestamp=self.current_time or event.timestamp,
                last=latest_bar.get('close'),
                bid=latest_bar.get('close', 0) * 0.9999,
                ask=latest_bar.get('close', 0) * 1.0001,
                volume=latest_bar.get('volume'),
                open=latest_bar.get('open'),
                high=latest_bar.get('high'),
                low=latest_bar.get('low'),
                close=latest_bar.get('close')
            )
            
            # Execute order
            fill = self.execution_handler.execute_order(event, market_event)
            
            if fill:
                # Add fill to event queue
                self.event_queue.put(fill, priority=4)
            else:
                logger.debug(f"Order {event.order_id} not filled")
                
        except Exception as e:
            logger.error(f"Error executing order {event.order_id}: {e}")
    
    def _handle_fill_event(self, event: FillEvent) -> None:
        """Process order fills."""
        try:
            # Update portfolio with fill
            self.portfolio.update_fill(event)
            
            # Update strategy position tracking
            self.strategy.update_position(event.symbol, 
                                        self.portfolio.get_position(event.symbol).quantity)
            
            # Track fill for reporting
            self.fills_executed.append({
                'timestamp': event.timestamp,
                'symbol': event.symbol,
                'quantity': event.quantity,
                'fill_price': event.fill_price,
                'commission': event.commission,
                'slippage': event.slippage,
                'order_id': event.order_id
            })
            
        except Exception as e:
            logger.error(f"Error processing fill {event.order_id}: {e}")
    
    def _handle_risk_event(self, event: RiskEvent) -> None:
        """Process risk management events."""
        self.risk_events.append({
            'timestamp': event.timestamp,
            'risk_type': event.risk_type,
            'severity': event.severity,
            'message': event.message,
            'action_required': event.action_required
        })
        
        # Log based on severity
        if event.severity == 'CRITICAL':
            logger.critical(f"RISK EVENT: {event.message}")
        elif event.severity == 'ERROR':
            logger.error(f"RISK EVENT: {event.message}")
        elif event.severity == 'WARNING':
            logger.warning(f"RISK EVENT: {event.message}")
        else:
            logger.info(f"RISK EVENT: {event.message}")
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        logger.info("Calculating backtest results...")
        
        # Extract equity curve from portfolio
        if self.portfolio.equity_curve:
            equity_data = pd.DataFrame(self.portfolio.equity_curve, 
                                     columns=['timestamp', 'portfolio_value'])
            equity_data.set_index('timestamp', inplace=True)
            equity_curve = equity_data['portfolio_value']
        else:
            equity_curve = pd.Series([self.config.initial_capital], 
                                   index=[pd.to_datetime(self.config.start_date)])
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Performance metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = ((1 + total_return) ** (365.25 / days)) - 1 if days > 0 else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Drawdown analysis
        drawdown_series = self._calculate_drawdown(equity_curve)
        max_drawdown = drawdown_series.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading metrics
        trades_df = pd.DataFrame(self.portfolio.closed_trades) if self.portfolio.closed_trades else pd.DataFrame()
        
        if not trades_df.empty:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            winning_pnl = trades_df[trades_df['pnl'] > 0]['pnl']
            losing_pnl = trades_df[trades_df['pnl'] < 0]['pnl']
            
            avg_win = winning_pnl.mean() if len(winning_pnl) > 0 else 0
            avg_loss = losing_pnl.mean() if len(losing_pnl) > 0 else 0
            
            gross_profit = winning_pnl.sum() if len(winning_pnl) > 0 else 0
            gross_loss = abs(losing_pnl.sum()) if len(losing_pnl) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Cost analysis
        total_commission = self.portfolio.total_commission_paid
        total_slippage = self.portfolio.total_slippage_paid
        total_costs = total_commission + total_slippage
        
        total_pnl = equity_curve.iloc[-1] - equity_curve.iloc[0]
        cost_as_pct_of_pnl = (total_costs / abs(total_pnl)) * 100 if total_pnl != 0 else 0
        
        # Create position history
        positions_data = []
        for timestamp, portfolio_value in self.portfolio.equity_curve:
            for symbol, position in self.portfolio.positions.items():
                if position.quantity != 0:
                    positions_data.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'quantity': position.quantity,
                        'market_value': position.market_value,
                        'unrealized_pnl': position.unrealized_pnl
                    })
        
        positions_df = pd.DataFrame(positions_data)
        
        # Create signals dataframe
        signals_df = pd.DataFrame(self.signals_generated)
        
        # Create results object
        results = BacktestResults(
            config=self.config,
            start_time=self.start_time,
            end_time=datetime.now(),
            duration=datetime.now() - self.start_time,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_costs=total_costs,
            cost_as_pct_of_pnl=cost_as_pct_of_pnl,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            positions=positions_df,
            trades=trades_df,
            signals=signals_df
        )
        
        return results
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sortino ratio."""
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
            
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate rolling drawdown series."""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown
    
    def _save_results(self, results: BacktestResults) -> None:
        """Save backtest results to files."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"backtest_{timestamp}"
            
            # Save JSON summary
            summary_file = output_dir / f"{base_filename}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            
            # Save detailed data
            if not results.equity_curve.empty:
                equity_file = output_dir / f"{base_filename}_equity_curve.csv"
                results.equity_curve.to_csv(equity_file)
            
            if not results.trades.empty:
                trades_file = output_dir / f"{base_filename}_trades.csv"
                results.trades.to_csv(trades_file, index=False)
            
            if not results.positions.empty:
                positions_file = output_dir / f"{base_filename}_positions.csv"
                results.positions.to_csv(positions_file, index=False)
            
            if not results.signals.empty:
                signals_file = output_dir / f"{base_filename}_signals.csv"
                results.signals.to_csv(signals_file, index=False)
            
            logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_live_stats(self) -> Dict[str, Any]:
        """Get live statistics during backtest execution."""
        return {
            'events_processed': self.events_processed,
            'heartbeat_count': self.heartbeat_count,
            'current_time': self.current_time,
            'portfolio_value': self.portfolio.total_portfolio_value,
            'cash': self.portfolio.current_cash,
            'positions': len([p for p in self.portfolio.positions.values() if p.quantity != 0]),
            'signals_generated': len(self.signals_generated),
            'orders_placed': len(self.orders_placed),
            'fills_executed': len(self.fills_executed),
            'risk_events': len(self.risk_events)
        }