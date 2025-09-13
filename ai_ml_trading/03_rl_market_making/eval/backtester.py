"""Backtesting infrastructure for market making strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
from dataclasses import dataclass
import logging
import time
from abc import ABC, abstractmethod

from ..rl.env_lob import MarketMakingEnv
from ..rl.market_simulator import MarketSimulator, MarketState
from ..agents.baseline_agents import BaselineAgent
from ..utils.data_generator import load_synthetic_data, MarketParameters
from ..utils.metrics import MarketMakingMetrics, TradeMetrics
from .performance_analysis import MarketMakingAnalyzer, PerformanceReport
from .adverse_selection import AdverseSelectionTracker, AdverseSelectionMetrics


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    initial_cash: float = 100000
    max_inventory: int = 1000
    tick_size: float = 0.01
    transaction_costs: bool = True
    slippage_model: str = "linear"  # "linear", "sqrt", "none"
    latency_ms: float = 1.0
    
    # Risk controls
    max_position_limit: int = 1500
    stop_loss_threshold: float = -1000
    daily_loss_limit: float = -5000
    
    # Evaluation settings
    benchmark_strategy: Optional[str] = "buy_and_hold"
    evaluation_frequency: str = "1D"  # pandas frequency string


@dataclass
class BacktestResult:
    """Results from backtesting."""
    
    # Basic results
    total_pnl: float
    total_return: float
    num_trades: int
    sharpe_ratio: float
    max_drawdown: float
    
    # Detailed metrics
    performance_report: PerformanceReport
    adverse_selection_metrics: AdverseSelectionMetrics
    
    # Time series
    pnl_series: pd.Series
    inventory_series: pd.Series
    trades_df: pd.DataFrame
    
    # Metadata
    config: BacktestConfig
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    execution_time_seconds: float
    
    # Benchmark comparison
    benchmark_pnl: Optional[float] = None
    excess_return: Optional[float] = None


class TradingAgent(Protocol):
    """Protocol for trading agents that can be backtested."""
    
    def select_action(self, state: MarketState, inventory: int, cash: float, **kwargs) -> np.ndarray:
        """Select trading action."""
        ...
    
    def get_name(self) -> str:
        """Get agent name."""
        ...


class MarketMakingBacktester:
    """Comprehensive backtesting engine for market making strategies."""
    
    def __init__(self, 
                 config: BacktestConfig,
                 market_data: Optional[List[Dict]] = None,
                 random_seed: Optional[int] = None):
        
        self.config = config
        self.market_data = market_data
        self.random_seed = random_seed
        
        # Components
        self.analyzer = MarketMakingAnalyzer()
        self.adverse_selection_tracker = AdverseSelectionTracker()
        self.metrics = MarketMakingMetrics()
        
        # State tracking
        self.current_time = None
        self.inventory = 0
        self.cash = config.initial_cash
        self.positions = []
        self.trades = []
        self.pnl_history = []
        self.inventory_history = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.0
    
    def run_backtest(self, 
                    agent: Union[TradingAgent, BaselineAgent],
                    start_episode: int = 0,
                    num_episodes: int = 100,
                    save_results: bool = True) -> BacktestResult:
        """Run backtest for given agent."""
        
        start_time = time.time()
        self.reset_state()
        
        logging.info(f"Starting backtest for {agent.get_name()} over {num_episodes} episodes")
        
        # Use market data or generate synthetic data
        if self.market_data is None:
            logging.info("No market data provided, using synthetic data")
            episodes_data = self._generate_synthetic_episodes(num_episodes)
        else:
            episodes_data = self.market_data[start_episode:start_episode + num_episodes]
        
        # Run backtest episodes
        for episode_idx, episode_data in enumerate(episodes_data):
            try:
                self._run_episode(agent, episode_data, episode_idx)
            except Exception as e:
                logging.error(f"Error in episode {episode_idx}: {e}")
                continue
        
        execution_time = time.time() - start_time
        
        # Generate results
        result = self._generate_backtest_result(agent, execution_time, start_time)
        
        if save_results:
            self._save_backtest_results(result)
        
        logging.info(f"Backtest completed in {execution_time:.2f} seconds")
        logging.info(f"Total P&L: ${result.total_pnl:,.2f}, Sharpe: {result.sharpe_ratio:.3f}")
        
        return result
    
    def reset_state(self):
        """Reset backtest state."""
        self.inventory = 0
        self.cash = self.config.initial_cash
        self.positions.clear()
        self.trades.clear()
        self.pnl_history.clear()
        self.inventory_history.clear()
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.0
        
        # Reset components
        self.adverse_selection_tracker.reset()
        self.metrics.reset()
    
    def _generate_synthetic_episodes(self, num_episodes: int) -> List[Dict]:
        """Generate synthetic market data episodes."""
        
        from ..utils.data_generator import generate_synthetic_lob_data, MarketParameters
        
        market_params = MarketParameters(
            initial_price=100.0,
            volatility=0.02,
            tick_size=self.config.tick_size
        )
        
        return generate_synthetic_lob_data(
            num_episodes=num_episodes,
            episode_length=1000,
            params=market_params
        )
    
    def _run_episode(self, 
                    agent: TradingAgent, 
                    episode_data: Dict, 
                    episode_idx: int):
        """Run single episode of backtesting."""
        
        order_books = episode_data['order_books']
        episode_start_cash = self.cash
        episode_start_inventory = self.inventory
        
        for step, book_state in enumerate(order_books):
            # Create market state
            market_state = self._create_market_state(book_state, step)
            
            # Update adverse selection tracker
            self.adverse_selection_tracker.update_market_data(
                timestamp=book_state['timestamp'],
                mid_price=book_state['mid_price'],
                market_state=market_state
            )
            
            # Check risk limits
            if self._check_risk_limits():
                logging.warning(f"Risk limits breached in episode {episode_idx}, step {step}")
                break
            
            # Get agent action
            try:
                action = agent.select_action(
                    state=market_state,
                    inventory=self.inventory,
                    cash=self.cash,
                    episode=episode_idx,
                    step=step
                )
            except Exception as e:
                logging.error(f"Agent action error: {e}")
                continue
            
            # Execute action
            step_pnl = self._execute_action(action, book_state, market_state)
            
            # Update tracking
            self.pnl_history.append(sum(self.pnl_history) + step_pnl if self.pnl_history else step_pnl)
            self.inventory_history.append(self.inventory)
            self.daily_pnl += step_pnl
            
            # Update metrics
            self.metrics.add_step_metrics(step, {
                'inventory': self.inventory,
                'cash': self.cash,
                'pnl': self.pnl_history[-1],
                'step_pnl': step_pnl,
                'mid_price': book_state['mid_price']
            })
        
        # End of episode processing
        episode_pnl = sum(self.pnl_history) - (sum(self.pnl_history[:-len(order_books)]) if len(self.pnl_history) > len(order_books) else 0)
        logging.debug(f"Episode {episode_idx} completed: P&L ${episode_pnl:.2f}, Inventory {self.inventory}")
    
    def _create_market_state(self, book_state: Dict, step: int) -> MarketState:
        """Create MarketState from order book state."""
        
        # Calculate basic market microstructure features
        bid_depth = book_state['depth']['bids']
        ask_depth = book_state['depth']['asks']
        
        # Imbalance
        total_bid_qty = sum(qty for _, qty in bid_depth)
        total_ask_qty = sum(qty for _, qty in ask_depth)
        imbalance = (total_bid_qty - total_ask_qty) / max(total_bid_qty + total_ask_qty, 1)
        
        # Spread
        if bid_depth and ask_depth:
            spread = ask_depth[0][0] - bid_depth[0][0]
        else:
            spread = self.config.tick_size
        
        # Simplified market state
        return MarketState(
            timestamp=book_state['timestamp'],
            mid_price=book_state['mid_price'],
            spread=spread,
            imbalance=imbalance,
            volatility=0.02,  # Default
            trend=0.0,  # Would need price history to calculate
            regime=0,  # Default to normal
            informed_flow=0.0,  # Default
            noise_level=0.0,  # Default
            liquidity_index=1.0,  # Default
            trade_volume_1min=0.0,  # Would need trade history
            price_change_1min=0.0,  # Would need price history
            order_arrival_rate=0.1,  # Default
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            avg_queue_depth=5.0,  # Default
            queue_decay_rate=0.1  # Default
        )
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached."""
        
        # Position limit
        if abs(self.inventory) > self.config.max_position_limit:
            return True
        
        # Stop loss
        current_pnl = self.pnl_history[-1] if self.pnl_history else 0
        if current_pnl < self.config.stop_loss_threshold:
            return True
        
        # Daily loss limit
        if self.daily_pnl < self.config.daily_loss_limit:
            return True
        
        return False
    
    def _execute_action(self, 
                       action: np.ndarray, 
                       book_state: Dict, 
                       market_state: MarketState) -> float:
        """Execute trading action and return step P&L."""
        
        # Parse action: [bid_offset, ask_offset, bid_size, ask_size, skew]
        bid_offset, ask_offset, bid_size, ask_size, skew = action
        
        # Calculate quote prices
        mid_price = book_state['mid_price']
        bid_price = mid_price - bid_offset * self.config.tick_size
        ask_price = mid_price + ask_offset * self.config.tick_size
        
        step_pnl = 0.0
        
        # Simulate order execution (simplified)
        # In practice, would use proper order book simulation
        
        # Simulate fills based on market conditions
        fill_probability = self._calculate_fill_probability(
            bid_price, ask_price, book_state, bid_size, ask_size
        )
        
        # Execute bid
        if np.random.random() < fill_probability['bid'] and bid_size > 0:
            trade_qty = int(bid_size * 100)  # Convert to shares
            if abs(self.inventory + trade_qty) <= self.config.max_inventory:
                step_pnl += self._execute_trade('BUY', bid_price, trade_qty, book_state['timestamp'])
        
        # Execute ask
        if np.random.random() < fill_probability['ask'] and ask_size > 0:
            trade_qty = int(ask_size * 100)  # Convert to shares
            if abs(self.inventory - trade_qty) <= self.config.max_inventory:
                step_pnl += self._execute_trade('SELL', ask_price, trade_qty, book_state['timestamp'])
        
        return step_pnl
    
    def _calculate_fill_probability(self, 
                                  bid_price: float, 
                                  ask_price: float, 
                                  book_state: Dict,
                                  bid_size: float, 
                                  ask_size: float) -> Dict[str, float]:
        """Calculate probability of order fills."""
        
        # Simple model based on price aggressiveness
        bid_depth = book_state['depth']['bids']
        ask_depth = book_state['depth']['asks']
        
        # Bid fill probability
        bid_prob = 0.0
        if bid_depth:
            best_bid = bid_depth[0][0]
            if bid_price >= best_bid:
                bid_prob = 0.8  # High probability if at or above best bid
            else:
                # Lower probability for passive orders
                bid_prob = 0.1 * (bid_price / best_bid)
        
        # Ask fill probability
        ask_prob = 0.0
        if ask_depth:
            best_ask = ask_depth[0][0]
            if ask_price <= best_ask:
                ask_prob = 0.8  # High probability if at or below best ask
            else:
                # Lower probability for passive orders
                ask_prob = 0.1 * (best_ask / ask_price)
        
        return {'bid': min(1.0, bid_prob), 'ask': min(1.0, ask_prob)}
    
    def _execute_trade(self, 
                      side: str, 
                      price: float, 
                      quantity: int, 
                      timestamp: pd.Timestamp) -> float:
        """Execute trade and return P&L."""
        
        # Apply slippage
        execution_price = self._apply_slippage(price, quantity, side)
        
        # Calculate transaction costs
        transaction_cost = 0.0
        if self.config.transaction_costs:
            transaction_cost = 0.0002 * execution_price * quantity  # 2 bps
        
        # Execute trade
        if side == 'BUY':
            self.inventory += quantity
            cash_change = -(execution_price * quantity + transaction_cost)
            trade_pnl = -transaction_cost  # Immediate cost
        else:  # SELL
            self.inventory -= quantity
            cash_change = execution_price * quantity - transaction_cost
            trade_pnl = -transaction_cost  # Immediate cost
        
        self.cash += cash_change
        
        # Record trade
        trade_data = {
            'timestamp': timestamp,
            'side': side,
            'price': execution_price,
            'quantity': quantity,
            'pnl': trade_pnl,
            'inventory_before': self.inventory - (quantity if side == 'BUY' else -quantity),
            'inventory_after': self.inventory,
            'cash_change': cash_change
        }
        
        self.trades.append(trade_data)
        
        # Update adverse selection tracker
        self.adverse_selection_tracker.record_trade(
            trade_time=timestamp,
            side=side,
            price=execution_price,
            quantity=quantity,
            trade_pnl=trade_pnl
        )
        
        return trade_pnl
    
    def _apply_slippage(self, price: float, quantity: int, side: str) -> float:
        """Apply slippage model."""
        
        if self.config.slippage_model == "none":
            return price
        
        # Simple linear slippage model
        slippage_bp = 0.1 * (quantity / 1000)  # 0.1 bp per 1000 shares
        
        if side == 'BUY':
            # Slippage increases execution price
            return price * (1 + slippage_bp / 10000)
        else:
            # Slippage decreases execution price
            return price * (1 - slippage_bp / 10000)
    
    def _generate_backtest_result(self, 
                                agent: TradingAgent, 
                                execution_time: float,
                                start_timestamp: float) -> BacktestResult:
        """Generate comprehensive backtest result."""
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Performance analysis
        performance_report = self.analyzer.analyze_session(
            trades_df=trades_df,
            inventory_history=self.inventory_history,
            pnl_history=self.pnl_history,
            market_states=[],  # Would need to store these
            orders_df=None
        )
        
        # Adverse selection analysis
        adverse_selection_metrics = self.adverse_selection_tracker.calculate_adverse_selection()
        
        # Create time series
        timestamps = [trade['timestamp'] for trade in self.trades] if self.trades else []
        pnl_series = pd.Series(self.pnl_history, name='pnl')
        inventory_series = pd.Series(self.inventory_history, name='inventory')
        
        # Basic metrics
        total_pnl = self.pnl_history[-1] if self.pnl_history else 0.0
        total_return = total_pnl / self.config.initial_cash
        
        return BacktestResult(
            total_pnl=total_pnl,
            total_return=total_return,
            num_trades=len(self.trades),
            sharpe_ratio=performance_report.sharpe_ratio,
            max_drawdown=performance_report.max_drawdown,
            performance_report=performance_report,
            adverse_selection_metrics=adverse_selection_metrics,
            pnl_series=pnl_series,
            inventory_series=inventory_series,
            trades_df=trades_df,
            config=self.config,
            start_time=pd.Timestamp.fromtimestamp(start_timestamp),
            end_time=pd.Timestamp.now(),
            execution_time_seconds=execution_time
        )
    
    def _save_backtest_results(self, result: BacktestResult):
        """Save backtest results to files."""
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades
        if not result.trades_df.empty:
            result.trades_df.to_csv(f"backtest_trades_{timestamp}.csv", index=False)
        
        # Save time series
        result.pnl_series.to_csv(f"backtest_pnl_{timestamp}.csv")
        result.inventory_series.to_csv(f"backtest_inventory_{timestamp}.csv")
        
        logging.info(f"Backtest results saved with timestamp {timestamp}")


def run_agent_backtest(agent: TradingAgent,
                      config: BacktestConfig,
                      market_data: Optional[List[Dict]] = None,
                      num_episodes: int = 100) -> BacktestResult:
    """Convenience function to run backtest for single agent."""
    
    backtester = MarketMakingBacktester(config, market_data)
    return backtester.run_backtest(agent, num_episodes=num_episodes)


def compare_strategies(agents: List[TradingAgent],
                      config: BacktestConfig,
                      market_data: Optional[List[Dict]] = None,
                      num_episodes: int = 100) -> Dict[str, BacktestResult]:
    """Compare multiple strategies using backtesting."""
    
    results = {}
    
    for agent in agents:
        logging.info(f"Backtesting {agent.get_name()}")
        
        try:
            backtester = MarketMakingBacktester(config, market_data)
            result = backtester.run_backtest(agent, num_episodes=num_episodes, save_results=False)
            results[agent.get_name()] = result
            
        except Exception as e:
            logging.error(f"Error backtesting {agent.get_name()}: {e}")
            continue
    
    return results


def create_comparison_report(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Create comparison report from multiple backtest results."""
    
    comparison_data = []
    
    for agent_name, result in results.items():
        comparison_data.append({
            'Agent': agent_name,
            'Total_PnL': result.total_pnl,
            'Total_Return': result.total_return,
            'Sharpe_Ratio': result.sharpe_ratio,
            'Max_Drawdown': result.max_drawdown,
            'Num_Trades': result.num_trades,
            'Win_Rate': result.performance_report.win_rate,
            'Avg_Trade_PnL': result.performance_report.avg_trade_pnl,
            'Adverse_Selection_Rate': result.adverse_selection_metrics.adverse_selection_rate_5s,
            'Execution_Time': result.execution_time_seconds
        })
    
    return pd.DataFrame(comparison_data).sort_values('Sharpe_Ratio', ascending=False)


class WalkForwardBacktester:
    """Walk-forward backtesting for more robust evaluation."""
    
    def __init__(self, 
                 config: BacktestConfig,
                 train_periods: int = 10,
                 test_periods: int = 2):
        
        self.config = config
        self.train_periods = train_periods
        self.test_periods = test_periods
    
    def run_walk_forward(self, 
                        agent_factory: callable,
                        market_data: List[Dict]) -> Dict[str, Any]:
        """Run walk-forward backtest."""
        
        if len(market_data) < self.train_periods + self.test_periods:
            raise ValueError("Insufficient data for walk-forward testing")
        
        results = []
        
        # Walk through data
        for i in range(0, len(market_data) - self.train_periods - self.test_periods, self.test_periods):
            train_data = market_data[i:i + self.train_periods]
            test_data = market_data[i + self.train_periods:i + self.train_periods + self.test_periods]
            
            # Train agent (if applicable)
            agent = agent_factory(train_data)
            
            # Test agent
            backtester = MarketMakingBacktester(self.config, test_data)
            test_result = backtester.run_backtest(agent, num_episodes=len(test_data), save_results=False)
            
            results.append({
                'period': i // self.test_periods,
                'train_start': i,
                'train_end': i + self.train_periods,
                'test_start': i + self.train_periods,
                'test_end': i + self.train_periods + self.test_periods,
                'result': test_result
            })
        
        return {
            'walk_forward_results': results,
            'avg_sharpe': np.mean([r['result'].sharpe_ratio for r in results]),
            'avg_return': np.mean([r['result'].total_return for r in results]),
            'consistency': np.std([r['result'].sharpe_ratio for r in results])
        }