# Options Strategy (Straddles / Covered Calls)

## Overview
Options strategies including earnings straddles, systematic covered calls, and IV-RV spread trading.

## Project Structure
```
05_options_strategy/
├── options/
│   ├── straddle_backtest.py
│   ├── greeks.py
│   ├── iv_surface.py
│   └── covered_calls.py
├── backtests/
│   └── options_backtest.ipynb
└── tests/
    └── test_options.py
```

## Implementation

### options/straddle_backtest.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StraddleConfig:
    days_before_earnings: int = 5
    days_after_earnings: int = 2
    min_iv_rank: float = 0.5
    max_iv_rank: float = 1.0
    position_size: float = 0.1
    delta_hedge: bool = True
    hedge_frequency: int = 1

class BlackScholes:
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price using Black-Scholes"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price using Black-Scholes"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put
    
    @staticmethod
    def implied_volatility(option_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson"""
        def objective(sigma):
            if option_type == 'call':
                model_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                model_price = BlackScholes.put_price(S, K, T, r, sigma)
            return abs(model_price - option_price)
        
        result = minimize_scalar(objective, bounds=(0.001, 5), method='bounded')
        return result.x
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, 
             option_type: str = 'call') -> float:
        """Calculate option delta"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Vega per 1% change
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = 'call') -> float:
        """Calculate option theta"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 + term2) / 365  # Daily theta
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365
        
        return theta

class EarningsStraddleStrategy:
    def __init__(self, config: StraddleConfig = StraddleConfig()):
        self.config = config
        self.bs = BlackScholes()
        self.positions = []
        
    def calculate_iv_rank(self, iv_series: pd.Series, lookback: int = 252) -> pd.Series:
        """Calculate IV rank (percentile over lookback period)"""
        iv_rank = iv_series.rolling(window=lookback).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )
        return iv_rank
    
    def identify_earnings_events(self, earnings_dates: pd.Series,
                                prices: pd.Series) -> pd.DataFrame:
        """Identify tradeable earnings events"""
        events = []
        
        for earnings_date in earnings_dates:
            # Get price window around earnings
            start_date = earnings_date - pd.Timedelta(days=self.config.days_before_earnings)
            end_date = earnings_date + pd.Timedelta(days=self.config.days_after_earnings)
            
            price_window = prices.loc[start_date:end_date]
            
            if len(price_window) > 0:
                events.append({
                    'earnings_date': earnings_date,
                    'entry_date': start_date,
                    'exit_date': end_date,
                    'entry_price': price_window.iloc[0] if len(price_window) > 0 else None
                })
        
        return pd.DataFrame(events)
    
    def calculate_straddle_value(self, S: float, K: float, T: float,
                                r: float, iv: float) -> Dict:
        """Calculate straddle value and greeks"""
        call_price = self.bs.call_price(S, K, T, r, iv)
        put_price = self.bs.put_price(S, K, T, r, iv)
        
        straddle_value = call_price + put_price
        
        # Calculate Greeks
        call_delta = self.bs.delta(S, K, T, r, iv, 'call')
        put_delta = self.bs.delta(S, K, T, r, iv, 'put')
        
        gamma = self.bs.gamma(S, K, T, r, iv) * 2  # Both options have same gamma
        vega = self.bs.vega(S, K, T, r, iv) * 2
        
        call_theta = self.bs.theta(S, K, T, r, iv, 'call')
        put_theta = self.bs.theta(S, K, T, r, iv, 'put')
        
        return {
            'value': straddle_value,
            'call_value': call_price,
            'put_value': put_price,
            'delta': call_delta + put_delta,  # Net delta (should be ~0 for ATM)
            'gamma': gamma,
            'vega': vega,
            'theta': call_theta + put_theta
        }
    
    def simulate_straddle_pnl(self, entry_price: float, exit_price: float,
                             entry_iv: float, exit_iv: float,
                             days_held: int, r: float = 0.02) -> Dict:
        """Simulate P&L for a straddle position"""
        # Time to expiration at entry and exit
        T_entry = 30 / 365  # Assume 30 days to expiration
        T_exit = (30 - days_held) / 365
        
        # Calculate straddle values
        entry_straddle = self.calculate_straddle_value(
            entry_price, entry_price, T_entry, r, entry_iv
        )
        
        exit_straddle = self.calculate_straddle_value(
            exit_price, entry_price, T_exit, r, exit_iv
        )
        
        # P&L calculation
        pnl = exit_straddle['value'] - entry_straddle['value']
        pnl_pct = pnl / entry_straddle['value']
        
        # P&L attribution
        price_pnl = entry_straddle['gamma'] * ((exit_price - entry_price) ** 2) / 2
        vol_pnl = entry_straddle['vega'] * (exit_iv - entry_iv) * 100
        theta_pnl = entry_straddle['theta'] * days_held
        
        return {
            'total_pnl': pnl,
            'pnl_pct': pnl_pct,
            'price_pnl': price_pnl,
            'vol_pnl': vol_pnl,
            'theta_pnl': theta_pnl,
            'entry_value': entry_straddle['value'],
            'exit_value': exit_straddle['value']
        }
    
    def backtest_earnings_straddles(self, prices: pd.Series,
                                   implied_vols: pd.Series,
                                   earnings_dates: pd.Series,
                                   realized_vols: pd.Series = None) -> Dict:
        """Backtest earnings straddle strategy"""
        # Calculate IV rank
        iv_rank = self.calculate_iv_rank(implied_vols)
        
        # Identify earnings events
        events = self.identify_earnings_events(earnings_dates, prices)
        
        trades = []
        
        for _, event in events.iterrows():
            entry_date = event['entry_date']
            exit_date = event['exit_date']
            
            if entry_date not in prices.index or exit_date not in prices.index:
                continue
            
            # Check IV rank filter
            if entry_date in iv_rank.index:
                current_iv_rank = iv_rank.loc[entry_date]
                
                if (current_iv_rank < self.config.min_iv_rank or
                    current_iv_rank > self.config.max_iv_rank):
                    continue
            
            # Get prices and IVs
            entry_price = prices.loc[entry_date]
            exit_price = prices.loc[exit_date]
            entry_iv = implied_vols.loc[entry_date]
            exit_iv = implied_vols.loc[exit_date]
            
            # Calculate P&L
            days_held = (exit_date - entry_date).days
            
            pnl_result = self.simulate_straddle_pnl(
                entry_price, exit_price, entry_iv, exit_iv, days_held
            )
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_iv': entry_iv,
                'exit_iv': exit_iv,
                'iv_rank': current_iv_rank,
                'pnl': pnl_result['total_pnl'],
                'pnl_pct': pnl_result['pnl_pct'],
                'price_pnl': pnl_result['price_pnl'],
                'vol_pnl': pnl_result['vol_pnl'],
                'theta_pnl': pnl_result['theta_pnl']
            })
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) == 0:
            return {'error': 'No trades executed'}
        
        # Calculate performance metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        win_rate = (trades_df['pnl'] > 0).mean()
        
        sharpe = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252 / self.config.days_before_earnings)
        
        return {
            'trades': trades_df,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'num_trades': len(trades_df),
            'avg_price_pnl': trades_df['price_pnl'].mean(),
            'avg_vol_pnl': trades_df['vol_pnl'].mean(),
            'avg_theta_pnl': trades_df['theta_pnl'].mean()
        }

class CoveredCallStrategy:
    def __init__(self, strike_offset: float = 1.05, dte: int = 30):
        """
        strike_offset: Strike as percentage of spot (1.05 = 5% OTM)
        dte: Days to expiration for options
        """
        self.strike_offset = strike_offset
        self.dte = dte
        self.bs = BlackScholes()
        
    def select_strike(self, spot: float, offset: float = None) -> float:
        """Select strike price for covered call"""
        if offset is None:
            offset = self.strike_offset
        
        # Round to nearest standard strike
        strike = spot * offset
        strike = round(strike / 5) * 5  # Round to nearest $5
        
        return strike
    
    def calculate_covered_call_return(self, spot: float, strike: float,
                                     premium: float, final_price: float) -> Dict:
        """Calculate return for covered call position"""
        # Stock P&L
        stock_pnl = final_price - spot
        
        # Option P&L (we're short the call)
        if final_price > strike:
            # Option is exercised
            option_pnl = premium - (final_price - strike)
            total_pnl = strike - spot + premium
        else:
            # Option expires worthless
            option_pnl = premium
            total_pnl = final_price - spot + premium
        
        # Calculate returns
        total_return = total_pnl / spot
        
        # Annualized return
        annual_return = total_return * (365 / self.dte)
        
        return {
            'stock_pnl': stock_pnl,
            'option_pnl': option_pnl,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'annual_return': annual_return,
            'assigned': final_price > strike
        }
    
    def backtest(self, prices: pd.DataFrame, iv: pd.DataFrame,
                r: float = 0.02) -> Dict:
        """Backtest covered call strategy"""
        results = []
        
        # Monthly rebalancing
        rebalance_dates = pd.date_range(
            start=prices.index[0],
            end=prices.index[-1],
            freq='MS'
        )
        
        for i in range(len(rebalance_dates) - 1):
            entry_date = rebalance_dates[i]
            exit_date = rebalance_dates[i + 1]
            
            if entry_date not in prices.index or exit_date not in prices.index:
                continue
            
            for symbol in prices.columns:
                spot = prices.loc[entry_date, symbol]
                final_price = prices.loc[exit_date, symbol]
                
                if pd.isna(spot) or pd.isna(final_price):
                    continue
                
                # Select strike
                strike = self.select_strike(spot)
                
                # Calculate option premium
                T = self.dte / 365
                sigma = iv.loc[entry_date, symbol] if entry_date in iv.index else 0.25
                
                premium = self.bs.call_price(spot, strike, T, r, sigma)
                
                # Calculate returns
                result = self.calculate_covered_call_return(
                    spot, strike, premium, final_price
                )
                
                result['symbol'] = symbol
                result['entry_date'] = entry_date
                result['exit_date'] = exit_date
                result['spot'] = spot
                result['strike'] = strike
                result['premium'] = premium
                result['final_price'] = final_price
                
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Calculate portfolio metrics
        avg_return = results_df['total_return'].mean()
        total_return = (1 + results_df.groupby('entry_date')['total_return'].mean()).prod() - 1
        volatility = results_df['total_return'].std() * np.sqrt(12)
        sharpe = avg_return * 12 / volatility
        
        # Assignment statistics
        assignment_rate = results_df['assigned'].mean()
        
        return {
            'trades': results_df,
            'avg_monthly_return': avg_return,
            'total_return': total_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe,
            'assignment_rate': assignment_rate,
            'num_trades': len(results_df)
        }

class IVRVSpreadTrading:
    """Trade the spread between implied and realized volatility"""
    
    def __init__(self, lookback: int = 20, entry_zscore: float = 2.0):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        
    def calculate_realized_vol(self, prices: pd.Series, window: int = None) -> pd.Series:
        """Calculate realized volatility"""
        if window is None:
            window = self.lookback
        
        returns = prices.pct_change()
        rv = returns.rolling(window=window).std() * np.sqrt(252)
        
        return rv
    
    def calculate_iv_rv_spread(self, iv: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """Calculate IV-RV spread"""
        rv = self.calculate_realized_vol(prices)
        
        spread = pd.DataFrame({
            'iv': iv,
            'rv': rv,
            'spread': iv - rv,
            'spread_pct': (iv - rv) / rv
        })
        
        # Calculate z-score
        spread['spread_zscore'] = (
            spread['spread'] - spread['spread'].rolling(self.lookback).mean()
        ) / spread['spread'].rolling(self.lookback).std()
        
        return spread
    
    def generate_signals(self, spread_df: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on IV-RV spread"""
        signals = pd.Series(index=spread_df.index, data=0)
        
        # Long volatility when IV << RV (spread very negative)
        signals[spread_df['spread_zscore'] < -self.entry_zscore] = 1
        
        # Short volatility when IV >> RV (spread very positive)
        signals[spread_df['spread_zscore'] > self.entry_zscore] = -1
        
        # Exit when spread normalizes
        signals[abs(spread_df['spread_zscore']) < 0.5] = 0
        
        # Forward fill positions
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return signals
```

## Deliverables
- `options/straddle_backtest.py`: Complete earnings straddle strategy implementation
- Black-Scholes pricing and Greeks calculation
- Covered call systematic strategy
- IV-RV spread trading framework