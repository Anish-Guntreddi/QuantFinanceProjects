"""Volatility and risk factor implementations"""

import pandas as pd
import numpy as np
from typing import Optional
from .base import BaseFactor


class RealizedVolatility(BaseFactor):
    """Realized volatility factor"""
    
    def __init__(self, window: int = 252, annualize: bool = True):
        """
        Parameters:
        -----------
        window : int
            Rolling window for volatility calculation (default 252 days = 1 year)
        annualize : bool
            Whether to annualize volatility
        """
        super().__init__(f'realized_vol_{window}d')
        self.window = window
        self.annualize = annualize
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate realized volatility"""
        if 'returns' not in data.columns:
            if 'price' in data.columns:
                data['returns'] = data['price'].pct_change()
            else:
                raise ValueError("Required data for volatility calculation not found")
        
        # Calculate rolling standard deviation
        vol = data['returns'].rolling(window=self.window).std()
        
        # Annualize if requested
        if self.annualize:
            vol = vol * np.sqrt(252)
        
        return vol


class MarketBeta(BaseFactor):
    """Market beta factor"""
    
    def __init__(self, window: int = 252):
        """
        Parameters:
        -----------
        window : int
            Rolling window for beta calculation
        """
        super().__init__(f'market_beta_{window}d')
        self.window = window
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market beta"""
        if 'returns' not in data.columns or 'market_returns' not in data.columns:
            raise ValueError("Required columns 'returns' and 'market_returns' not found")
        
        # Calculate rolling beta
        beta = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.window, len(data)):
            window_data = data.iloc[i-self.window:i]
            stock_returns = window_data['returns'].dropna()
            market_returns = window_data['market_returns'].dropna()
            
            if len(stock_returns) > self.window * 0.5:  # Require at least 50% data
                cov = np.cov(stock_returns, market_returns)[0, 1]
                var = np.var(market_returns)
                if var > 0:
                    beta.iloc[i] = cov / var
        
        return beta


class IdiosyncraticVolatility(BaseFactor):
    """Idiosyncratic volatility (residual volatility after market factor)"""
    
    def __init__(self, window: int = 252):
        """
        Parameters:
        -----------
        window : int
            Rolling window for calculation
        """
        super().__init__(f'idio_vol_{window}d')
        self.window = window
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate idiosyncratic volatility"""
        if 'returns' not in data.columns or 'market_returns' not in data.columns:
            raise ValueError("Required columns 'returns' and 'market_returns' not found")
        
        # Calculate rolling residual volatility
        idio_vol = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.window, len(data)):
            window_data = data.iloc[i-self.window:i]
            stock_returns = window_data['returns'].dropna()
            market_returns = window_data['market_returns'].dropna()
            
            if len(stock_returns) > self.window * 0.5:
                # Simple regression to get residuals
                try:
                    beta = np.cov(stock_returns, market_returns)[0, 1] / np.var(market_returns)
                    alpha = stock_returns.mean() - beta * market_returns.mean()
                    residuals = stock_returns - (alpha + beta * market_returns)
                    idio_vol.iloc[i] = residuals.std() * np.sqrt(252)
                except:
                    continue
        
        return idio_vol


class Downside_Volatility(BaseFactor):
    """Downside volatility (semi-deviation)"""
    
    def __init__(self, window: int = 252, threshold: float = 0.0):
        """
        Parameters:
        -----------
        window : int
            Rolling window
        threshold : float
            Threshold for downside (default 0)
        """
        super().__init__(f'downside_vol_{window}d')
        self.window = window
        self.threshold = threshold
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate downside volatility"""
        if 'returns' not in data.columns:
            if 'price' in data.columns:
                data['returns'] = data['price'].pct_change()
            else:
                raise ValueError("Required data for volatility calculation not found")
        
        # Calculate rolling downside deviation
        def downside_std(returns):
            downside_returns = returns[returns < self.threshold]
            if len(downside_returns) > 1:
                return downside_returns.std()
            return np.nan
        
        downside_vol = data['returns'].rolling(window=self.window).apply(downside_std)
        
        # Annualize
        downside_vol = downside_vol * np.sqrt(252)
        
        return downside_vol


class MaxDrawdown(BaseFactor):
    """Maximum drawdown over rolling window"""
    
    def __init__(self, window: int = 252):
        """
        Parameters:
        -----------
        window : int
            Rolling window for max drawdown
        """
        super().__init__(f'max_drawdown_{window}d')
        self.window = window
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate maximum drawdown"""
        if 'price' not in data.columns:
            raise ValueError("Required column 'price' not found")
        
        # Calculate rolling maximum drawdown
        def calculate_max_dd(prices):
            cummax = prices.expanding().max()
            drawdown = (prices - cummax) / cummax
            return drawdown.min()
        
        max_dd = data['price'].rolling(window=self.window).apply(calculate_max_dd)
        
        # Return absolute value (so higher = worse)
        return np.abs(max_dd)


class VolatilityOfVolatility(BaseFactor):
    """Volatility of volatility (vol clustering measure)"""
    
    def __init__(self, vol_window: int = 20, meta_window: int = 252):
        """
        Parameters:
        -----------
        vol_window : int
            Window for volatility calculation
        meta_window : int
            Window for volatility of volatility
        """
        super().__init__(f'vol_of_vol_{vol_window}d_{meta_window}d')
        self.vol_window = vol_window
        self.meta_window = meta_window
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility of volatility"""
        if 'returns' not in data.columns:
            if 'price' in data.columns:
                data['returns'] = data['price'].pct_change()
            else:
                raise ValueError("Required data for volatility calculation not found")
        
        # First calculate short-term rolling volatility
        short_vol = data['returns'].rolling(window=self.vol_window).std()
        
        # Then calculate volatility of this volatility
        vol_of_vol = short_vol.rolling(window=self.meta_window).std()
        
        return vol_of_vol