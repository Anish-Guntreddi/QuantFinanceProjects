"""
Generate sample portfolio returns data for visualization
Copy this code into your Jupyter notebook to create the portfolio_returns variable
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_strategy_returns(n_days=252, base_return=0.10, volatility=0.15, strategy_type='momentum'):
    """Generate synthetic returns for a strategy"""
    daily_return = base_return / 252
    daily_vol = volatility / np.sqrt(252)
    
    if strategy_type == 'momentum':
        # Momentum has trending behavior
        returns = []
        trend = 0
        for _ in range(n_days):
            trend = 0.8 * trend + np.random.randn() * daily_vol
            daily_ret = daily_return + trend
            returns.append(daily_ret)
        return np.array(returns)
    
    elif strategy_type == 'value':
        # Value has mean-reverting behavior
        returns = []
        deviation = 0
        for _ in range(n_days):
            deviation = -0.2 * deviation + np.random.randn() * daily_vol
            daily_ret = daily_return + deviation
            returns.append(daily_ret)
        return np.array(returns)
    
    elif strategy_type == 'quality':
        # Quality has stable returns
        returns = np.random.randn(n_days) * daily_vol * 0.8 + daily_return
        return returns
    
    elif strategy_type == 'low_vol':
        # Low volatility has lower but steadier returns
        returns = np.random.randn(n_days) * daily_vol * 0.5 + daily_return * 0.7
        return returns
    
    else:  # combined
        # Combined strategy averages others
        mom = generate_strategy_returns(n_days, base_return, volatility, 'momentum')
        val = generate_strategy_returns(n_days, base_return, volatility, 'value')
        qual = generate_strategy_returns(n_days, base_return, volatility, 'quality')
        return (mom + val + qual) / 3

# Generate portfolio returns
n_days = 252 * 2  # 2 years of daily data

portfolio_returns = {
    'Momentum': generate_strategy_returns(n_days, 0.12, 0.18, 'momentum'),
    'Value': generate_strategy_returns(n_days, 0.10, 0.15, 'value'),
    'Quality': generate_strategy_returns(n_days, 0.09, 0.12, 'quality'),
    'Low_Volatility': generate_strategy_returns(n_days, 0.07, 0.10, 'low_vol'),
    'Combined': generate_strategy_returns(n_days, 0.11, 0.14, 'combined'),
}

# Generate factor scores
dates = pd.date_range('2022-01-01', periods=252, freq='D')

factor_scores = pd.DataFrame({
    'Momentum': np.random.randn(252).cumsum() / 10,
    'Value': np.sin(np.linspace(0, 4*np.pi, 252)) + np.random.randn(252) * 0.1,
    'Quality': np.random.randn(252) * 0.5 + 0.2,
    'Low_Vol': -np.abs(np.random.randn(252)) + 0.5,
}, index=dates)

# Generate IC values
ic_values = pd.DataFrame({
    'Momentum': np.random.randn(252) * 0.03 + 0.04,
    'Value': np.random.randn(252) * 0.025 + 0.03,
    'Quality': np.random.randn(252) * 0.02 + 0.035,
    'Low_Vol': np.random.randn(252) * 0.02 + 0.025,
}, index=dates)

# Add some autocorrelation to IC values
for col in ic_values.columns:
    ic_values[col] = ic_values[col].rolling(5).mean().fillna(method='bfill')

print("Sample data generated successfully!")
print(f"portfolio_returns: {len(portfolio_returns)} strategies, {n_days} days each")
print(f"factor_scores: {factor_scores.shape}")
print(f"ic_values: {ic_values.shape}")
print("\nVariables created: portfolio_returns, factor_scores, ic_values")