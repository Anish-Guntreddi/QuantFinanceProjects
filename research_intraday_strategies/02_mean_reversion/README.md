# Mean Reversion (Pairs/Bands/RSI) Strategy

## Overview
Implementation of mean reversion strategies using z-score reversion, Bollinger Bands, and RSI indicators.

## Project Structure
```
02_mean_reversion/
├── meanrev/
│   ├── pairs.py
│   ├── indicators.py
│   └── signals.py
├── plots/
│   └── spread_zscore.png
├── backtests/
│   └── mean_reversion_backtest.ipynb
└── tests/
    └── test_pairs.py
```

## Implementation

### meanrev/pairs.py
```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

@dataclass
class PairsConfig:
    lookback_period: int = 60
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 3.5
    min_half_life: int = 5
    max_half_life: int = 120
    position_size: float = 1.0

class PairsTradingStrategy:
    def __init__(self, config: PairsConfig = PairsConfig()):
        self.config = config
        self.positions = {}
        self.trades = []
        
    def find_cointegrated_pairs(self, data: pd.DataFrame, 
                               p_value_threshold: float = 0.05) -> List[Tuple[str, str, float]]:
        """Find cointegrated pairs using Engle-Granger test"""
        pairs = []
        symbols = data.columns.tolist()
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                # Test for cointegration
                score, p_value, _ = coint(data[symbols[i]], data[symbols[j]])
                
                if p_value < p_value_threshold:
                    pairs.append((symbols[i], symbols[j], p_value))
        
        return sorted(pairs, key=lambda x: x[2])
    
    def calculate_spread(self, series1: pd.Series, series2: pd.Series, 
                        window: Optional[int] = None) -> Tuple[pd.Series, float]:
        """Calculate spread using rolling OLS regression"""
        if window is None:
            window = self.config.lookback_period
        
        # Rolling regression to get dynamic hedge ratio
        hedge_ratios = []
        
        for i in range(window, len(series1)):
            X = series1.iloc[i-window:i].values.reshape(-1, 1)
            y = series2.iloc[i-window:i].values
            
            model = LinearRegression()
            model.fit(X, y)
            hedge_ratios.append(model.coef_[0])
        
        # Pad the beginning with the first hedge ratio
        hedge_ratios = [hedge_ratios[0]] * window + hedge_ratios
        hedge_ratios = pd.Series(hedge_ratios, index=series1.index)
        
        # Calculate spread
        spread = series2 - hedge_ratios * series1
        
        return spread, hedge_ratios.iloc[-1]
    
    def calculate_zscore(self, spread: pd.Series, window: Optional[int] = None) -> pd.Series:
        """Calculate rolling z-score of spread"""
        if window is None:
            window = self.config.lookback_period
        
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        
        zscore = (spread - spread_mean) / spread_std
        return zscore
    
    def calculate_half_life(self, spread: pd.Series) -> int:
        """Calculate half-life of mean reversion using OLS"""
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Remove NaN values
        mask = ~(spread_lag.isna() | spread_diff.isna())
        spread_lag = spread_lag[mask].values.reshape(-1, 1)
        spread_diff = spread_diff[mask].values
        
        if len(spread_lag) < 2:
            return np.inf
        
        # OLS regression: spread_diff = lambda * spread_lag
        model = LinearRegression()
        model.fit(spread_lag, spread_diff)
        
        lambda_coef = model.coef_[0]
        
        if lambda_coef >= 0:
            return np.inf
        
        half_life = -np.log(2) / lambda_coef
        return int(half_life)
    
    def generate_signals(self, data1: pd.Series, data2: pd.Series) -> pd.DataFrame:
        """Generate trading signals for a pair"""
        # Calculate spread and z-score
        spread, current_hedge_ratio = self.calculate_spread(data1, data2)
        zscore = self.calculate_zscore(spread)
        
        # Calculate half-life
        half_life = self.calculate_half_life(spread)
        
        # Check if half-life is within acceptable range
        if half_life < self.config.min_half_life or half_life > self.config.max_half_life:
            print(f"Half-life {half_life} outside acceptable range")
            return pd.DataFrame()
        
        # Generate signals
        signals = pd.DataFrame(index=data1.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['hedge_ratio'] = current_hedge_ratio
        signals['position'] = 0
        
        # Entry and exit logic
        signals.loc[zscore > self.config.entry_zscore, 'position'] = -1  # Short spread
        signals.loc[zscore < -self.config.entry_zscore, 'position'] = 1  # Long spread
        
        # Exit positions
        signals.loc[abs(zscore) < self.config.exit_zscore, 'position'] = 0
        
        # Stop loss
        signals.loc[zscore > self.config.stop_loss_zscore, 'position'] = 0
        signals.loc[zscore < -self.config.stop_loss_zscore, 'position'] = 0
        
        # Forward fill positions
        signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def backtest_pair(self, data1: pd.Series, data2: pd.Series, 
                     pair_name: str = "Pair") -> Dict:
        """Backtest a single pair"""
        signals = self.generate_signals(data1, data2)
        
        if signals.empty:
            return {}
        
        # Calculate returns
        returns1 = data1.pct_change()
        returns2 = data2.pct_change()
        
        # Calculate portfolio returns
        # Long spread = long asset2, short asset1
        # Short spread = short asset2, long asset1
        signals['returns'] = signals['position'].shift(1) * (
            returns2 - signals['hedge_ratio'].shift(1) * returns1
        )
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
        
        # Calculate performance metrics
        total_return = signals['cumulative_returns'].iloc[-1] - 1
        sharpe_ratio = signals['returns'].mean() / signals['returns'].std() * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(signals['cumulative_returns'])
        num_trades = (signals['position'].diff() != 0).sum()
        
        # Plot spread and z-score
        self.plot_spread_zscore(signals, pair_name)
        
        return {
            'pair': pair_name,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'signals': signals
        }
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def plot_spread_zscore(self, signals: pd.DataFrame, pair_name: str):
        """Plot spread and z-score with entry/exit levels"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot spread
        ax1.plot(signals.index, signals['spread'], label='Spread', color='blue')
        ax1.set_ylabel('Spread')
        ax1.set_title(f'{pair_name} - Spread')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot z-score with entry/exit levels
        ax2.plot(signals.index, signals['zscore'], label='Z-Score', color='black')
        ax2.axhline(y=self.config.entry_zscore, color='r', linestyle='--', 
                   label=f'Entry ({self.config.entry_zscore})')
        ax2.axhline(y=-self.config.entry_zscore, color='r', linestyle='--')
        ax2.axhline(y=self.config.exit_zscore, color='g', linestyle='--', 
                   label=f'Exit ({self.config.exit_zscore})')
        ax2.axhline(y=-self.config.exit_zscore, color='g', linestyle='--')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Z-Score')
        ax2.set_title(f'{pair_name} - Z-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot positions and cumulative returns
        ax3.plot(signals.index, signals['cumulative_returns'], 
                label='Cumulative Returns', color='green')
        ax3.set_ylabel('Cumulative Returns')
        ax3.set_xlabel('Date')
        ax3.set_title(f'{pair_name} - Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/spread_zscore_{pair_name.replace(" ", "_")}.png', dpi=150)
        plt.show()

class BollingerBandsStrategy:
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
        
    def calculate_bands(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        bands = pd.DataFrame(index=prices.index)
        
        bands['price'] = prices
        bands['sma'] = prices.rolling(window=self.window).mean()
        bands['std'] = prices.rolling(window=self.window).std()
        bands['upper'] = bands['sma'] + (bands['std'] * self.num_std)
        bands['lower'] = bands['sma'] - (bands['std'] * self.num_std)
        bands['bandwidth'] = (bands['upper'] - bands['lower']) / bands['sma']
        bands['percent_b'] = (prices - bands['lower']) / (bands['upper'] - bands['lower'])
        
        return bands
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """Generate mean reversion signals using Bollinger Bands"""
        bands = self.calculate_bands(prices)
        
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        signals['position'] = 0
        
        # Mean reversion signals
        # Buy when price touches lower band (oversold)
        signals.loc[bands['price'] <= bands['lower'], 'position'] = 1
        
        # Sell when price touches upper band (overbought)
        signals.loc[bands['price'] >= bands['upper'], 'position'] = -1
        
        # Exit when price crosses SMA
        cross_sma = ((bands['price'] > bands['sma']) & 
                     (bands['price'].shift(1) <= bands['sma'].shift(1))) | \
                    ((bands['price'] < bands['sma']) & 
                     (bands['price'].shift(1) >= bands['sma'].shift(1)))
        signals.loc[cross_sma, 'position'] = 0
        
        # Forward fill positions
        signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0)
        
        # Add band information
        signals = pd.concat([signals, bands[['upper', 'sma', 'lower', 'percent_b']]], axis=1)
        
        return signals

class RSIMeanReversionStrategy:
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """Generate mean reversion signals using RSI"""
        rsi = self.calculate_rsi(prices)
        
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        signals['rsi'] = rsi
        signals['position'] = 0
        
        # Mean reversion signals
        # Buy when RSI is oversold
        signals.loc[rsi < self.oversold, 'position'] = 1
        
        # Sell when RSI is overbought
        signals.loc[rsi > self.overbought, 'position'] = -1
        
        # Exit when RSI returns to neutral zone (45-55)
        signals.loc[(rsi > 45) & (rsi < 55), 'position'] = 0
        
        # Forward fill positions
        signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def backtest(self, prices: pd.Series) -> Dict:
        """Backtest RSI mean reversion strategy"""
        signals = self.generate_signals(prices)
        
        # Calculate returns
        returns = prices.pct_change()
        signals['returns'] = signals['position'].shift(1) * returns
        signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
        
        # Performance metrics
        total_return = signals['cumulative_returns'].iloc[-1] - 1
        sharpe_ratio = signals['returns'].mean() / signals['returns'].std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'signals': signals
        }
```

### meanrev/indicators.py
```python
import numpy as np
import pandas as pd
from typing import Optional

class MeanReversionIndicators:
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, 
                        num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        return pd.DataFrame({
            'upper': sma + (std * num_std),
            'middle': sma,
            'lower': sma - (std * num_std),
            'bandwidth': 2 * std * num_std / sma,
            'percent_b': (prices - (sma - std * num_std)) / (2 * std * num_std)
        })
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                         ema_period: int = 20, atr_period: int = 10,
                         multiplier: float = 2) -> pd.DataFrame:
        """Calculate Keltner Channels"""
        # Calculate EMA of close
        ema = close.ewm(span=ema_period, adjust=False).mean()
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        return pd.DataFrame({
            'upper': ema + (atr * multiplier),
            'middle': ema,
            'lower': ema - (atr * multiplier)
        })
    
    @staticmethod
    def mean_reversion_index(prices: pd.Series, lookback: int = 20) -> pd.Series:
        """Calculate mean reversion index"""
        sma = prices.rolling(window=lookback).mean()
        deviation = (prices - sma) / sma
        
        # Normalize to [-1, 1]
        mri = np.tanh(deviation * 10)
        
        return mri
    
    @staticmethod
    def ornstein_uhlenbeck_params(prices: pd.Series) -> Dict:
        """Estimate Ornstein-Uhlenbeck process parameters"""
        log_prices = np.log(prices)
        
        # Calculate differences
        dx = log_prices.diff().dropna()
        x_lag = log_prices.shift(1).dropna()
        
        # Estimate parameters using OLS
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(x_lag.values.reshape(-1, 1), dx.values)
        
        theta = -model.coef_[0]  # Mean reversion speed
        mu = model.intercept_ / theta  # Long-term mean
        
        # Estimate volatility
        residuals = dx - model.predict(x_lag.values.reshape(-1, 1))
        sigma = np.std(residuals) * np.sqrt(252)
        
        # Half-life
        half_life = np.log(2) / theta if theta > 0 else np.inf
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life
        }
```

## Deliverables
- `meanrev/pairs.py`: Pairs trading strategy with cointegration testing
- `plots/spread_zscore.png`: Visualization of spread and z-score with entry/exit levels
- Bollinger Bands and RSI mean reversion strategies
- Half-life calculation for mean reversion speed