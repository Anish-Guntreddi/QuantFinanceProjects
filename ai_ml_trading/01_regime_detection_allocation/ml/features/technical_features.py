"""
Technical feature extraction for regime detection.

This module provides comprehensive technical analysis features for regime detection
including trend, momentum, volatility, volume, and pattern indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    warnings.warn("TA-Lib not available, using simplified technical indicators")

warnings.filterwarnings("ignore")


class TechnicalFeatureExtractor:
    """Extract technical indicators for regime detection"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        use_talib: bool = True
    ) -> pd.DataFrame:
        """
        Extract technical features from price data
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with OHLC data or single price series
        volume_data : Optional[pd.DataFrame]
            Volume data (if available)
        use_talib : bool
            Whether to use TA-Lib for calculations
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with technical features
        """
        
        features = pd.DataFrame(index=price_data.index)
        
        # Determine price series structure
        if len(price_data.columns) >= 4:
            # OHLC data
            open_ = price_data.iloc[:, 0]
            high = price_data.iloc[:, 1]
            low = price_data.iloc[:, 2]
            close = price_data.iloc[:, 3]
        elif 'close' in price_data.columns:
            close = price_data['close']
            high = price_data.get('high', close)
            low = price_data.get('low', close)
            open_ = price_data.get('open', close)
        else:
            # Single series - assume it's close price
            close = price_data.iloc[:, 0]
            high = low = open_ = close
            
        # Extract different types of features
        trend_features = self._calculate_trend_indicators(close, high, low, use_talib)
        momentum_features = self._calculate_momentum_indicators(close, high, low, use_talib)
        volatility_features = self._calculate_volatility_indicators(close, high, low, use_talib)
        
        # Combine all features
        for feature_dict in [trend_features, momentum_features, volatility_features]:
            for name, values in feature_dict.items():
                features[name] = values
                
        # Volume features if available
        if volume_data is not None:
            volume_features = self._calculate_volume_indicators(close, volume_data, use_talib)
            for name, values in volume_features.items():
                features[name] = values
                
        # Pattern recognition features
        if HAS_TALIB and use_talib and len(price_data.columns) >= 4:
            pattern_features = self._calculate_patterns(open_, high, low, close)
            for name, values in pattern_features.items():
                features[name] = values
                
        # Market microstructure features
        micro_features = self._calculate_microstructure(high, low, close)
        for name, values in micro_features.items():
            features[name] = values
            
        return features
    
    def _calculate_trend_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        use_talib: bool = True
    ) -> Dict[str, pd.Series]:
        """Calculate trend indicators"""
        
        features = {}
        
        # Moving averages
        for period in [10, 20, 50, 200]:
            if HAS_TALIB and use_talib:
                features[f'sma_{period}'] = pd.Series(
                    talib.SMA(close.values, timeperiod=period), index=close.index
                )
                features[f'ema_{period}'] = pd.Series(
                    talib.EMA(close.values, timeperiod=period), index=close.index
                )
            else:
                features[f'sma_{period}'] = close.rolling(period).mean()
                features[f'ema_{period}'] = close.ewm(span=period).mean()
                
        # MA crossovers and ratios
        if 'sma_10' in features and 'sma_20' in features:
            features['ma_cross_short'] = (
                features['sma_10'] - features['sma_20']
            ) / features['sma_20']
            
        if 'sma_50' in features and 'sma_200' in features:
            features['ma_cross_long'] = (
                features['sma_50'] - features['sma_200']
            ) / features['sma_200']
            
        # Price relative to moving averages
        if 'sma_20' in features:
            features['price_to_sma20'] = close / features['sma_20'] - 1
            
        # ADX (Average Directional Index)
        if HAS_TALIB and use_talib:
            features['adx'] = pd.Series(
                talib.ADX(high.values, low.values, close.values, timeperiod=14),
                index=close.index
            )
            features['plus_di'] = pd.Series(
                talib.PLUS_DI(high.values, low.values, close.values, timeperiod=14),
                index=close.index
            )
            features['minus_di'] = pd.Series(
                talib.MINUS_DI(high.values, low.values, close.values, timeperiod=14),
                index=close.index
            )
        else:
            # Simplified ADX calculation
            features['adx'] = self._simple_adx(high, low, close, 14)
            
        # Trend strength
        returns = close.pct_change()
        for period in [10, 20]:
            # Trend consistency
            positive_days = (returns > 0).rolling(period).sum()
            features[f'trend_consistency_{period}'] = positive_days / period
            
            # Directional movement
            up_moves = returns.where(returns > 0, 0).rolling(period).sum()
            down_moves = (-returns).where(returns < 0, 0).rolling(period).sum()
            features[f'directional_strength_{period}'] = (up_moves - down_moves) / (up_moves + down_moves + 1e-10)
            
        return features
    
    def _calculate_momentum_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        use_talib: bool = True
    ) -> Dict[str, pd.Series]:
        """Calculate momentum indicators"""
        
        features = {}
        
        # RSI
        for period in [9, 14, 21]:
            if HAS_TALIB and use_talib:
                features[f'rsi_{period}'] = pd.Series(
                    talib.RSI(close.values, timeperiod=period), index=close.index
                )
            else:
                features[f'rsi_{period}'] = self._simple_rsi(close, period)
                
        # Stochastic Oscillator
        if HAS_TALIB and use_talib:
            stoch_k, stoch_d = talib.STOCH(
                high.values, low.values, close.values,
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            features['stoch_k'] = pd.Series(stoch_k, index=close.index)
            features['stoch_d'] = pd.Series(stoch_d, index=close.index)
        else:
            features['stoch_k'] = self._simple_stochastic(high, low, close, 14)
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()
            
        # MACD
        if HAS_TALIB and use_talib:
            macd, macd_signal, macd_hist = talib.MACD(
                close.values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            features['macd'] = pd.Series(macd, index=close.index)
            features['macd_signal'] = pd.Series(macd_signal, index=close.index)
            features['macd_histogram'] = pd.Series(macd_hist, index=close.index)
        else:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
        # Williams %R
        if HAS_TALIB and use_talib:
            features['williams_r'] = pd.Series(
                talib.WILLR(high.values, low.values, close.values, timeperiod=14),
                index=close.index
            )
        else:
            highest_high = high.rolling(14).max()
            lowest_low = low.rolling(14).min()
            features['williams_r'] = (highest_high - close) / (highest_high - lowest_low) * -100
            
        # Rate of Change
        for period in [10, 20]:
            if HAS_TALIB and use_talib:
                features[f'roc_{period}'] = pd.Series(
                    talib.ROC(close.values, timeperiod=period), index=close.index
                )
            else:
                features[f'roc_{period}'] = close.pct_change(period) * 100
                
        # Commodity Channel Index
        if HAS_TALIB and use_talib:
            features['cci'] = pd.Series(
                talib.CCI(high.values, low.values, close.values, timeperiod=14),
                index=close.index
            )
        else:
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(14).mean()
            mad = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
            features['cci'] = (tp - sma_tp) / (0.015 * mad)
            
        return features
    
    def _calculate_volatility_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        use_talib: bool = True
    ) -> Dict[str, pd.Series]:
        """Calculate volatility indicators"""
        
        features = {}
        
        # ATR (Average True Range)
        if HAS_TALIB and use_talib:
            features['atr'] = pd.Series(
                talib.ATR(high.values, low.values, close.values, timeperiod=14),
                index=close.index
            )
        else:
            features['atr'] = self._simple_atr(high, low, close, 14)
            
        features['atr_percent'] = features['atr'] / close
        
        # Bollinger Bands
        if HAS_TALIB and use_talib:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close.values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            features['bb_upper'] = pd.Series(bb_upper, index=close.index)
            features['bb_middle'] = pd.Series(bb_middle, index=close.index)
            features['bb_lower'] = pd.Series(bb_lower, index=close.index)
        else:
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            features['bb_upper'] = sma20 + 2 * std20
            features['bb_middle'] = sma20
            features['bb_lower'] = sma20 - 2 * std20
            
        # Bollinger Band derived features
        features['bb_width'] = (
            features['bb_upper'] - features['bb_lower']
        ) / features['bb_middle']
        
        features['bb_position'] = (
            close - features['bb_lower']
        ) / (features['bb_upper'] - features['bb_lower'])
        
        # Historical Volatility
        returns = close.pct_change()
        for period in [10, 20, 60]:
            features[f'hvol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
        # Volatility ratio
        if 'hvol_10' in features and 'hvol_60' in features:
            features['vol_ratio'] = features['hvol_10'] / (features['hvol_60'] + 1e-10)
            
        # Volatility percentile
        for period in [20, 60]:
            vol_col = f'hvol_{period}'
            if vol_col in features:
                features[f'vol_percentile_{period}'] = (
                    features[vol_col].rolling(252).rank(pct=True)
                )
                
        return features
    
    def _calculate_volume_indicators(
        self,
        close: pd.Series,
        volume: Union[pd.Series, pd.DataFrame],
        use_talib: bool = True
    ) -> Dict[str, pd.Series]:
        """Calculate volume indicators"""
        
        features = {}
        
        # Extract volume series
        if isinstance(volume, pd.DataFrame):
            vol = volume.iloc[:, 0]
        else:
            vol = volume
            
        # OBV (On-Balance Volume)
        if HAS_TALIB and use_talib:
            features['obv'] = pd.Series(
                talib.OBV(close.values, vol.values), index=close.index
            )
        else:
            price_change = close.diff()
            obv_values = []
            obv = 0
            
            for i, change in enumerate(price_change):
                if pd.notna(change):
                    if change > 0:
                        obv += vol.iloc[i]
                    elif change < 0:
                        obv -= vol.iloc[i]
                obv_values.append(obv)
                
            features['obv'] = pd.Series(obv_values, index=close.index)
            
        # Volume moving averages
        features['volume_ma'] = vol.rolling(20).mean()
        features['volume_ratio'] = vol / (features['volume_ma'] + 1e-10)
        
        # Volume trend
        features['volume_trend'] = vol.rolling(10).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
        # VWAP (Volume Weighted Average Price)
        features['vwap'] = (close * vol).cumsum() / vol.cumsum()
        features['vwap_deviation'] = (close - features['vwap']) / features['vwap']
        
        return features
    
    def _calculate_patterns(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Calculate candlestick patterns using TA-Lib"""
        
        features = {}
        
        if not HAS_TALIB:
            return features
            
        # Common candlestick patterns
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'hanging_man': talib.CDLHANGINGMAN,
            'inverted_hammer': talib.CDLINVERTEDHAMMER
        }
        
        for name, func in patterns.items():
            try:
                pattern_values = func(open_.values, high.values, low.values, close.values)
                features[f'pattern_{name}'] = pd.Series(pattern_values, index=close.index)
            except:
                continue
                
        # Aggregate pattern score
        pattern_cols = [col for col in features.keys() if col.startswith('pattern_')]
        if pattern_cols:
            pattern_df = pd.DataFrame(features)[pattern_cols]
            features['pattern_score'] = pattern_df.sum(axis=1)
            features['bullish_patterns'] = pattern_df[pattern_df > 0].sum(axis=1)
            features['bearish_patterns'] = pattern_df[pattern_df < 0].abs().sum(axis=1)
            
        return features
    
    def _calculate_microstructure(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Calculate market microstructure features"""
        
        features = {}
        
        # Spread proxy
        features['spread_proxy'] = (high - low) / close
        features['spread_ma'] = features['spread_proxy'].rolling(20).mean()
        
        # Efficiency ratio (Kaufman)
        net_change = close.diff(10).abs()
        total_change = close.diff().abs().rolling(10).sum()
        features['efficiency_ratio'] = net_change / (total_change + 1e-10)
        
        # High-low range features
        features['hl_range'] = (high - low) / close
        features['hl_range_ma'] = features['hl_range'].rolling(20).mean()
        features['hl_range_std'] = features['hl_range'].rolling(20).std()
        
        # Close position within range
        features['close_position'] = (close - low) / (high - low + 1e-10)
        
        # Intraday momentum
        features['intraday_momentum'] = (close - close.shift(1)) / (high - low + 1e-10)
        
        # Gap analysis
        features['gap'] = close / close.shift(1) - 1
        features['gap_abs'] = features['gap'].abs()
        features['gap_direction'] = np.sign(features['gap'])
        
        return features
    
    # Helper methods for simplified calculations when TA-Lib is not available
    
    def _simple_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Simplified RSI calculation"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _simple_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Simplified Stochastic calculation"""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        
        return k_percent
    
    def _simple_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Simplified ATR calculation"""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _simple_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Simplified ADX calculation"""
        # Calculate directional movement
        dm_pos = high.diff()
        dm_neg = (-low.diff())
        
        dm_pos = dm_pos.where((dm_pos > dm_neg) & (dm_pos > 0), 0)
        dm_neg = dm_neg.where((dm_neg > dm_pos) & (dm_neg > 0), 0)
        
        # Calculate true range
        atr = self._simple_atr(high, low, close, period)
        
        # Calculate directional indicators
        di_pos = 100 * dm_pos.rolling(period).mean() / atr
        di_neg = 100 * dm_neg.rolling(period).mean() / atr
        
        # Calculate ADX
        dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get feature categories for analysis"""
        
        return {
            'Trend': [
                'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'ema_10', 'ema_20', 'ema_50', 'ema_200',
                'ma_cross_short', 'ma_cross_long',
                'adx', 'plus_di', 'minus_di'
            ],
            'Momentum': [
                'rsi_9', 'rsi_14', 'rsi_21',
                'stoch_k', 'stoch_d',
                'macd', 'macd_signal', 'macd_histogram',
                'williams_r', 'cci',
                'roc_10', 'roc_20'
            ],
            'Volatility': [
                'atr', 'atr_percent',
                'bb_upper', 'bb_middle', 'bb_lower',
                'bb_width', 'bb_position',
                'hvol_10', 'hvol_20', 'hvol_60',
                'vol_ratio', 'vol_percentile_20', 'vol_percentile_60'
            ],
            'Volume': [
                'obv', 'volume_ma', 'volume_ratio',
                'volume_trend', 'vwap', 'vwap_deviation'
            ],
            'Patterns': [
                'pattern_doji', 'pattern_hammer', 'pattern_shooting_star',
                'pattern_engulfing', 'pattern_morning_star', 'pattern_evening_star',
                'pattern_score', 'bullish_patterns', 'bearish_patterns'
            ],
            'Microstructure': [
                'spread_proxy', 'efficiency_ratio',
                'hl_range', 'close_position', 'intraday_momentum',
                'gap', 'gap_abs', 'gap_direction'
            ]
        }