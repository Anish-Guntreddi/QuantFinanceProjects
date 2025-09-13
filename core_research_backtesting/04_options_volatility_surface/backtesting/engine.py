"""Options backtesting engine"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from strategies.delta_hedge import DeltaHedger


class OptionsBacktester:
    """Backtest options strategies"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
        
    def backtest_delta_hedge(
        self,
        price_data: pd.DataFrame,
        option_params: Dict,
        rehedge_freq: str = 'daily'
    ) -> pd.DataFrame:
        """
        Backtest delta hedging strategy
        
        Args:
            price_data: DataFrame with columns [date, price, realized_vol]
            option_params: Dict with strike, maturity, etc.
        """
        
        results = []
        
        # Rolling window for options
        window = option_params['days_to_expiry']
        
        for i in range(0, len(price_data) - window, window):
            # Get price path for this option's lifetime
            S_path = price_data['price'].iloc[i:i+window].values
            
            # Get realized vol
            if 'realized_vol' in price_data.columns:
                rv_path = price_data['realized_vol'].iloc[i:i+window].values
            else:
                # Calculate realized vol
                returns = np.log(S_path[1:] / S_path[:-1])
                rv_path = pd.Series(returns).rolling(20).std() * np.sqrt(252)
                rv_path = rv_path.fillna(method='bfill').values
            
            # Initial IV (could be from surface)
            initial_iv = rv_path[0] * 1.1  # Assume IV premium
            
            # Run hedge simulation
            hedger = DeltaHedger(
                rehedge_frequency=rehedge_freq,
                transaction_cost=self.transaction_cost
            )
            
            hedge_result = hedger.simulate_hedge(
                option_type=option_params['type'],
                S_path=S_path,
                K=option_params['strike'],
                T=window/252,
                r=option_params['rate'],
                sigma_initial=initial_iv,
                realized_vol=rv_path
            )
            
            results.append({
                'start_date': price_data.index[i] if isinstance(price_data.index, pd.DatetimeIndex) else i,
                'end_date': price_data.index[i+window-1] if isinstance(price_data.index, pd.DatetimeIndex) else i+window-1,
                'initial_spot': S_path[0],
                'final_spot': S_path[-1],
                'initial_iv': initial_iv,
                'avg_rv': np.mean(rv_path),
                'pnl': hedge_result['net_pnl'],
                'hedge_error': hedge_result['hedge_error'],
                'total_costs': hedge_result['total_costs']
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        
        if self.results is None:
            raise ValueError("Run backtest first")
        
        # Calculate returns
        self.results['returns'] = self.results['pnl'] / self.initial_capital
        
        # Metrics
        total_return = self.results['returns'].sum()
        avg_return = self.results['returns'].mean()
        volatility = self.results['returns'].std()
        sharpe = avg_return / volatility * np.sqrt(252/30) if volatility > 0 else 0  # Monthly Sharpe
        
        # Win rate
        win_rate = (self.results['pnl'] > 0).mean()
        
        # Average hedge error
        avg_hedge_error = self.results['hedge_error'].mean()
        
        # Costs as % of P&L
        cost_ratio = self.results['total_costs'].sum() / abs(self.results['pnl'].sum()) if self.results['pnl'].sum() != 0 else 0
        
        # Max drawdown
        cumulative_returns = (1 + self.results['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': avg_return * 12,
            'volatility': volatility * np.sqrt(12),
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_hedge_error': avg_hedge_error,
            'cost_ratio': cost_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.results)
        }