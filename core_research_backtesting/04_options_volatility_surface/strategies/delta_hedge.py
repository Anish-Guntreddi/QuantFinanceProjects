"""Delta hedging strategy implementation"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vol.models.black_scholes import BlackScholes


class DeltaHedger:
    """Implement delta hedging strategies"""
    
    def __init__(
        self,
        rehedge_frequency: str = 'daily',
        transaction_cost: float = 0.0005,
        borrow_rate: float = 0.02
    ):
        self.rehedge_frequency = rehedge_frequency
        self.transaction_cost = transaction_cost
        self.borrow_rate = borrow_rate
        self.hedge_history = []
        
    def simulate_hedge(
        self,
        option_type: str,
        S_path: np.ndarray,
        K: float,
        T: float,
        r: float,
        sigma_initial: float,
        realized_vol: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Simulate delta hedging over price path
        
        Args:
            S_path: Stock price path
            K: Strike price
            T: Initial time to maturity
            r: Risk-free rate
            sigma_initial: Initial implied volatility
            realized_vol: Actual realized volatility path (if different from IV)
        """
        
        n_steps = len(S_path)
        dt = T / n_steps
        
        # Initialize
        portfolio_value = []
        hedge_deltas = []
        hedge_costs = []
        option_values = []
        
        # Initial setup
        S0 = S_path[0]
        
        # Calculate initial option value and delta
        if option_type.lower() == 'call':
            V0 = BlackScholes.call_price(S0, K, T, r, sigma_initial)
            delta0 = BlackScholes.delta(S0, K, T, r, sigma_initial, 'call')
        else:
            V0 = BlackScholes.put_price(S0, K, T, r, sigma_initial)
            delta0 = BlackScholes.delta(S0, K, T, r, sigma_initial, 'put')
        
        # Initial hedge: short option, buy delta shares
        shares_held = delta0
        cash_position = V0 - delta0 * S0
        
        portfolio_value.append(V0)
        hedge_deltas.append(delta0)
        option_values.append(V0)
        
        # Simulate hedging
        for i in range(1, n_steps):
            S = S_path[i]
            time_remaining = T - i * dt
            
            # Use realized vol if provided, otherwise use initial IV
            if realized_vol is not None:
                sigma = realized_vol[i]
            else:
                sigma = sigma_initial
            
            # Calculate new option value and delta
            if time_remaining > 0:
                if option_type.lower() == 'call':
                    V = BlackScholes.call_price(S, K, time_remaining, r, sigma)
                    delta = BlackScholes.delta(S, K, time_remaining, r, sigma, 'call')
                else:
                    V = BlackScholes.put_price(S, K, time_remaining, r, sigma)
                    delta = BlackScholes.delta(S, K, time_remaining, r, sigma, 'put')
            else:
                # At expiry
                if option_type.lower() == 'call':
                    V = max(S - K, 0)
                    delta = 1 if S > K else 0
                else:
                    V = max(K - S, 0)
                    delta = -1 if S < K else 0
            
            # Rehedge
            shares_to_trade = delta - shares_held
            trade_cost = abs(shares_to_trade) * S * self.transaction_cost
            
            # Update positions
            cash_position = cash_position * np.exp(r * dt) - shares_to_trade * S - trade_cost
            shares_held = delta
            
            # Portfolio value (short option + hedge)
            port_value = shares_held * S + cash_position - V
            
            # Store results
            portfolio_value.append(port_value)
            hedge_deltas.append(delta)
            hedge_costs.append(trade_cost)
            option_values.append(V)
        
        # Calculate P&L
        total_pnl = portfolio_value[-1] - portfolio_value[0]
        total_costs = sum(hedge_costs)
        
        # Analyze hedge effectiveness
        hedge_error = np.std(np.diff(portfolio_value))
        
        results = {
            'total_pnl': total_pnl,
            'total_costs': total_costs,
            'net_pnl': total_pnl - total_costs,
            'hedge_error': hedge_error,
            'portfolio_values': portfolio_value,
            'deltas': hedge_deltas,
            'option_values': option_values,
            'hedge_costs': hedge_costs,
            'final_portfolio_value': portfolio_value[-1]
        }
        
        return results
    
    def analyze_pnl(
        self,
        hedge_results: Dict,
        S_path: np.ndarray
    ) -> pd.DataFrame:
        """Decompose P&L into components"""
        
        n = len(S_path)
        pnl_components = pd.DataFrame()
        
        # Delta P&L (from stock moves)
        delta_pnl = []
        for i in range(1, n):
            delta = hedge_results['deltas'][i-1]
            price_change = S_path[i] - S_path[i-1]
            delta_pnl.append(delta * price_change)
        
        pnl_components['delta_pnl'] = [0] + delta_pnl
        
        # Gamma P&L (from convexity)
        gamma_pnl = []
        for i in range(1, n):
            price_change = S_path[i] - S_path[i-1]
            # Approximate gamma at midpoint
            S_mid = (S_path[i] + S_path[i-1]) / 2
            # Calculate gamma (would need option parameters)
            gamma = 0.01  # Placeholder
            gamma_pnl.append(0.5 * gamma * price_change**2)
        
        pnl_components['gamma_pnl'] = [0] + gamma_pnl
        
        # Theta P&L (time decay)
        theta_pnl = []
        option_values = hedge_results['option_values']
        for i in range(1, n):
            # Approximate theta from option value changes
            theta = (option_values[i] - option_values[i-1]) / n
            theta_pnl.append(theta)
        
        pnl_components['theta_pnl'] = [0] + theta_pnl
        
        # Transaction costs
        pnl_components['transaction_costs'] = hedge_results['hedge_costs']
        
        # Total P&L
        pnl_components['total_pnl'] = (
            pnl_components['delta_pnl'] +
            pnl_components['gamma_pnl'] +
            pnl_components['theta_pnl'] -
            pnl_components['transaction_costs']
        )
        
        return pnl_components