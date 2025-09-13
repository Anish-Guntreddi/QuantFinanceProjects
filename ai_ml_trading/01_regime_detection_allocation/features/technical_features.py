"""
Technical analysis features extraction with TA-Lib integration.

This module provides comprehensive technical analysis indicators for regime detection,
including trend, momentum, volatility, and volume-based features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Optional TA-Lib import with fallback implementations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None
    
warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)


class TechnicalFeatureExtractor:
    """
    Comprehensive technical analysis feature extractor with TA-Lib integration.
    
    Features include:
    - Trend indicators (MA, MACD, ADX, etc.)
    - Momentum indicators (RSI, Stochastic, Williams %R, etc.)
    - Volatility indicators (Bollinger Bands, ATR, etc.)
    - Volume indicators (OBV, MFI, etc.)
    - Price patterns and fractals
    - Market microstructure indicators
    - Custom regime-specific indicators
    """
    
    def __init__(
        self,
        use_talib: bool = True,
        fillna_method: str = 'forward',
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0
    ):
        """
        Initialize technical feature extractor.
        
        Parameters:
        -----------
        use_talib : bool
            Whether to use TA-Lib (if available) or fallback implementations
        fillna_method : str
            Method for handling NaN values: 'forward', 'backward', 'drop', 'zero'
        remove_outliers : bool
            Whether to remove statistical outliers
        outlier_threshold : float
            Z-score threshold for outlier detection
        """
        self.use_talib = use_talib and TALIB_AVAILABLE
        self.fillna_method = fillna_method
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        if not TALIB_AVAILABLE and use_talib:
            logger.warning("TA-Lib not available, using fallback implementations")
            
        logger.info(f"Technical feature extractor initialized (TA-Lib: {self.use_talib})")
    
    def extract_features(
        self,
        data: pd.DataFrame,
        feature_groups: Optional[List[str]] = None,
        custom_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Extract technical analysis features from OHLCV data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume (optional)
        feature_groups : list, optional
            Groups of features to extract
        custom_params : dict, optional
            Custom parameters for indicators
            
        Returns:
        --------
        pd.DataFrame
            Technical analysis features
        """
        
        if data.empty:
            logger.warning("No data provided for technical feature extraction")
            return pd.DataFrame()
            
        # Validate input data
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Extracting technical features from {len(data)} observations")
        
        # Default feature groups
        if feature_groups is None:
            feature_groups = [
                'price_features',
                'trend_indicators', 
                'momentum_indicators',
                'volatility_indicators',
                'volume_indicators',
                'pattern_features',
                'market_structure'
            ]
            
        # Initialize parameters
        if custom_params is None:
            custom_params = {}
            
        # Extract OHLCV arrays
        open_prices = data['Open'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        volume = data['Volume'].values if 'Volume' in data.columns else None
        
        all_features = pd.DataFrame(index=data.index)
        
        # Extract different feature groups
        for group in feature_groups:
            try:
                if group == 'price_features':
                    features = self._extract_price_features(
                        data, open_prices, high_prices, low_prices, close_prices, custom_params
                    )
                elif group == 'trend_indicators':
                    features = self._extract_trend_indicators(
                        close_prices, high_prices, low_prices, data.index, custom_params
                    )
                elif group == 'momentum_indicators':
                    features = self._extract_momentum_indicators(
                        close_prices, high_prices, low_prices, data.index, custom_params
                    )
                elif group == 'volatility_indicators':
                    features = self._extract_volatility_indicators(
                        close_prices, high_prices, low_prices, data.index, custom_params
                    )
                elif group == 'volume_indicators' and volume is not None:
                    features = self._extract_volume_indicators(
                        close_prices, high_prices, low_prices, volume, data.index, custom_params
                    )
                elif group == 'pattern_features':
                    features = self._extract_pattern_features(
                        open_prices, high_prices, low_prices, close_prices, data.index, custom_params
                    )
                elif group == 'market_structure':
                    features = self._extract_market_structure_features(
                        data, custom_params
                    )
                else:
                    logger.warning(f"Unknown feature group: {group}")
                    continue
                    
                # Combine features
                for col in features.columns:
                    if col not in all_features.columns:
                        all_features[col] = features[col]
                        
            except Exception as e:
                logger.error(f"Error extracting {group}: {e}")
                continue
                
        # Post-process features
        all_features = self._post_process_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} technical features")
        
        return all_features
    
    def _extract_price_features(
        self,
        data: pd.DataFrame,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        params: Dict
    ) -> pd.DataFrame:
        """Extract basic price-based features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # Returns
            features['returns'] = np.concatenate([[np.nan], np.diff(np.log(close_prices))])
            
            # Price ratios
            features['hl_ratio'] = high_prices / low_prices
            features['oc_ratio'] = open_prices / close_prices
            features['hc_ratio'] = high_prices / close_prices
            features['lc_ratio'] = low_prices / close_prices
            
            # Intraday ranges
            features['true_range'] = self._calculate_true_range(high_prices, low_prices, close_prices)
            features['high_low_pct'] = (high_prices - low_prices) / close_prices
            features['open_close_pct'] = (close_prices - open_prices) / open_prices
            
            # Price position in range
            features['close_position'] = (close_prices - low_prices) / (high_prices - low_prices)
            features['close_position'] = np.where(high_prices == low_prices, 0.5, features['close_position'])
            
            # Multiple timeframe returns
            windows = params.get('return_windows', [1, 2, 3, 5, 10, 21])
            for window in windows:
                if window < len(close_prices):
                    features[f'return_{window}d'] = pd.Series(close_prices, index=data.index).pct_change(window)
                    
            # Log returns for different periods
            for window in [5, 21, 63]:
                if window < len(close_prices):
                    features[f'log_return_{window}d'] = pd.Series(close_prices, index=data.index).pct_change(window).apply(lambda x: np.log(1 + x) if x > -1 else np.nan)
                    
        except Exception as e:
            logger.error(f"Error in price feature extraction: {e}")
            
        return features
    
    def _extract_trend_indicators(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        index: pd.Index,
        params: Dict
    ) -> pd.DataFrame:
        """Extract trend-based indicators"""
        
        features = pd.DataFrame(index=index)
        
        try:
            # Moving averages
            ma_windows = params.get('ma_windows', [5, 10, 21, 50, 200])
            
            for window in ma_windows:
                if window < len(close_prices):
                    if self.use_talib:
                        ma = talib.SMA(close_prices, timeperiod=window)
                    else:
                        ma = pd.Series(close_prices).rolling(window).mean().values
                        
                    features[f'sma_{window}'] = ma
                    features[f'price_sma_{window}_ratio'] = close_prices / ma
                    features[f'sma_{window}_slope'] = self._calculate_slope(ma, window=5)
                    
            # Exponential moving averages
            ema_windows = params.get('ema_windows', [12, 26, 50])
            for window in ema_windows:
                if window < len(close_prices):
                    if self.use_talib:
                        ema = talib.EMA(close_prices, timeperiod=window)
                    else:
                        ema = pd.Series(close_prices).ewm(span=window).mean().values
                        
                    features[f'ema_{window}'] = ema
                    features[f'price_ema_{window}_ratio'] = close_prices / ema
                    
            # MACD
            if self.use_talib:
                macd, macd_signal, macd_hist = talib.MACD(
                    close_prices,
                    fastperiod=params.get('macd_fast', 12),
                    slowperiod=params.get('macd_slow', 26),
                    signalperiod=params.get('macd_signal', 9)
                )
            else:
                macd, macd_signal, macd_hist = self._calculate_macd_fallback(
                    close_prices,
                    params.get('macd_fast', 12),
                    params.get('macd_slow', 26),
                    params.get('macd_signal', 9)
                )
                
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            features['macd_signal_cross'] = np.where(macd > macd_signal, 1, -1)
            
            # ADX (Directional Movement Index)
            adx_period = params.get('adx_period', 14)
            if self.use_talib:
                adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=adx_period)
                plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=adx_period)
                minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=adx_period)
            else:
                adx, plus_di, minus_di = self._calculate_adx_fallback(
                    high_prices, low_prices, close_prices, adx_period
                )
                
            features['adx'] = adx
            features['plus_di'] = plus_di
            features['minus_di'] = minus_di
            features['dx_diff'] = plus_di - minus_di
            
            # Parabolic SAR
            if self.use_talib:
                sar = talib.SAR(high_prices, low_prices,
                              acceleration=params.get('sar_accel', 0.02),
                              maximum=params.get('sar_max', 0.2))
            else:
                sar = self._calculate_sar_fallback(
                    high_prices, low_prices,
                    params.get('sar_accel', 0.02),
                    params.get('sar_max', 0.2)
                )
                
            features['sar'] = sar
            features['sar_trend'] = np.where(close_prices > sar, 1, -1)
            
        except Exception as e:
            logger.error(f"Error in trend indicator extraction: {e}")
            
        return features
    
    def _extract_momentum_indicators(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        index: pd.Index,
        params: Dict
    ) -> pd.DataFrame:
        """Extract momentum-based indicators"""
        
        features = pd.DataFrame(index=index)
        
        try:
            # RSI
            rsi_period = params.get('rsi_period', 14)
            if self.use_talib:
                rsi = talib.RSI(close_prices, timeperiod=rsi_period)
            else:
                rsi = self._calculate_rsi_fallback(close_prices, rsi_period)
                
            features['rsi'] = rsi
            features['rsi_overbought'] = (rsi > 70).astype(float)
            features['rsi_oversold'] = (rsi < 30).astype(float)
            features['rsi_momentum'] = np.concatenate([[np.nan], np.diff(rsi)])
            
            # Stochastic Oscillator
            stoch_k_period = params.get('stoch_k_period', 14)
            stoch_d_period = params.get('stoch_d_period', 3)
            
            if self.use_talib:
                slowk, slowd = talib.STOCH(
                    high_prices, low_prices, close_prices,
                    fastk_period=stoch_k_period,
                    slowk_period=stoch_d_period,
                    slowk_matype=0,
                    slowd_period=stoch_d_period,
                    slowd_matype=0
                )
            else:
                slowk, slowd = self._calculate_stoch_fallback(
                    high_prices, low_prices, close_prices, stoch_k_period, stoch_d_period
                )
                
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            features['stoch_signal'] = np.where(slowk > slowd, 1, -1)
            
            # Williams %R
            willr_period = params.get('willr_period', 14)
            if self.use_talib:
                willr = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=willr_period)
            else:
                willr = self._calculate_willr_fallback(high_prices, low_prices, close_prices, willr_period)
                
            features['williams_r'] = willr
            
            # ROC (Rate of Change)
            roc_periods = params.get('roc_periods', [5, 10, 21])
            for period in roc_periods:
                if period < len(close_prices):
                    if self.use_talib:
                        roc = talib.ROC(close_prices, timeperiod=period)
                    else:
                        roc = self._calculate_roc_fallback(close_prices, period)
                        
                    features[f'roc_{period}'] = roc
                    
            # CCI (Commodity Channel Index)
            cci_period = params.get('cci_period', 20)
            if self.use_talib:
                cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=cci_period)
            else:
                cci = self._calculate_cci_fallback(high_prices, low_prices, close_prices, cci_period)
                
            features['cci'] = cci
            features['cci_overbought'] = (cci > 100).astype(float)
            features['cci_oversold'] = (cci < -100).astype(float)
            
            # Ultimate Oscillator
            if self.use_talib:
                uo = talib.ULTOSC(
                    high_prices, low_prices, close_prices,
                    timeperiod1=params.get('uo_period1', 7),
                    timeperiod2=params.get('uo_period2', 14),
                    timeperiod3=params.get('uo_period3', 28)
                )
                features['ultimate_oscillator'] = uo
                
        except Exception as e:
            logger.error(f"Error in momentum indicator extraction: {e}")
            
        return features
    
    def _extract_volatility_indicators(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        index: pd.Index,
        params: Dict
    ) -> pd.DataFrame:
        """Extract volatility-based indicators"""
        
        features = pd.DataFrame(index=index)
        
        try:
            # Average True Range
            atr_period = params.get('atr_period', 14)
            true_range = self._calculate_true_range(high_prices, low_prices, close_prices)
            
            if self.use_talib:
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
            else:
                atr = pd.Series(true_range).rolling(atr_period).mean().values
                
            features['atr'] = atr
            features['atr_ratio'] = atr / close_prices
            
            # Bollinger Bands
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2)
            
            if self.use_talib:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close_prices, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
                )
            else:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands_fallback(
                    close_prices, bb_period, bb_std
                )
                
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).quantile(0.1)).astype(float)
            
            # Volatility measures
            vol_windows = params.get('vol_windows', [5, 10, 21, 63])
            returns = np.concatenate([[np.nan], np.diff(np.log(close_prices))])
            
            for window in vol_windows:
                if window < len(close_prices):
                    vol = pd.Series(returns).rolling(window).std().values * np.sqrt(252)
                    features[f'volatility_{window}d'] = vol
                    
            # GARCH-like volatility
            features['garch_vol'] = self._calculate_garch_vol(returns)
            
            # Keltner Channels
            kc_period = params.get('kc_period', 20)
            kc_multiplier = params.get('kc_multiplier', 2.0)
            
            ema = pd.Series(close_prices).ewm(span=kc_period).mean().values
            kc_upper = ema + kc_multiplier * atr
            kc_lower = ema - kc_multiplier * atr
            
            features['kc_upper'] = kc_upper
            features['kc_lower'] = kc_lower
            features['kc_position'] = (close_prices - kc_lower) / (kc_upper - kc_lower)
            
        except Exception as e:
            logger.error(f"Error in volatility indicator extraction: {e}")
            
        return features
    
    def _extract_volume_indicators(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        volume: np.ndarray,
        index: pd.Index,
        params: Dict
    ) -> pd.DataFrame:
        """Extract volume-based indicators"""
        
        features = pd.DataFrame(index=index)
        
        try:
            # On-Balance Volume
            if self.use_talib:
                obv = talib.OBV(close_prices, volume)
            else:
                obv = self._calculate_obv_fallback(close_prices, volume)
                
            features['obv'] = obv
            features['obv_sma'] = pd.Series(obv).rolling(params.get('obv_sma_period', 10)).mean().values
            
            # Volume SMA ratio
            volume_sma_periods = params.get('volume_sma_periods', [10, 21, 50])
            for period in volume_sma_periods:
                if period < len(volume):
                    vol_sma = pd.Series(volume).rolling(period).mean().values
                    features[f'volume_sma_{period}_ratio'] = volume / vol_sma
                    
            # Money Flow Index
            mfi_period = params.get('mfi_period', 14)
            if self.use_talib:
                mfi = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=mfi_period)
            else:
                mfi = self._calculate_mfi_fallback(high_prices, low_prices, close_prices, volume, mfi_period)
                
            features['mfi'] = mfi
            
            # Volume Price Trend
            vpt = self._calculate_vpt(close_prices, volume)
            features['vpt'] = vpt
            features['vpt_sma'] = pd.Series(vpt).rolling(params.get('vpt_sma_period', 10)).mean().values
            
            # Accumulation/Distribution Line
            if self.use_talib:
                ad = talib.AD(high_prices, low_prices, close_prices, volume)
            else:
                ad = self._calculate_ad_fallback(high_prices, low_prices, close_prices, volume)
                
            features['ad_line'] = ad
            
            # Chaikin Money Flow
            cmf_period = params.get('cmf_period', 21)
            cmf = self._calculate_cmf(high_prices, low_prices, close_prices, volume, cmf_period)
            features['cmf'] = cmf
            
            # Volume Oscillator
            vo_short = params.get('vo_short_period', 5)
            vo_long = params.get('vo_long_period', 10)
            
            vol_short_ma = pd.Series(volume).rolling(vo_short).mean().values
            vol_long_ma = pd.Series(volume).rolling(vo_long).mean().values
            features['volume_oscillator'] = ((vol_short_ma - vol_long_ma) / vol_long_ma) * 100
            
        except Exception as e:
            logger.error(f"Error in volume indicator extraction: {e}")
            
        return features
    
    def _extract_pattern_features(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        index: pd.Index,
        params: Dict
    ) -> pd.DataFrame:
        """Extract pattern recognition features"""
        
        features = pd.DataFrame(index=index)
        
        try:
            # Price patterns using TA-Lib if available
            if self.use_talib:
                # Candlestick patterns
                patterns = [
                    ('CDLDOJI', 'doji'),
                    ('CDLHAMMER', 'hammer'),
                    ('CDLHANGINGMAN', 'hanging_man'),
                    ('CDLENGULFING', 'engulfing'),
                    ('CDLHARAMI', 'harami'),
                    ('CDLPIERCING', 'piercing'),
                    ('CDLDARKCLOUDCOVER', 'dark_cloud'),
                    ('CDLMORNINGSTAR', 'morning_star'),
                    ('CDLEVENINGSTAR', 'evening_star'),
                    ('CDLTHREEWHITESOLDIERS', 'three_white_soldiers'),
                    ('CDLTHREEBLACKCROWS', 'three_black_crows')
                ]
                
                for ta_func, name in patterns:
                    try:
                        pattern_func = getattr(talib, ta_func)
                        pattern_result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                        features[f'pattern_{name}'] = pattern_result
                    except Exception:
                        continue
                        
            # Custom pattern features
            # Gap detection
            features['gap_up'] = ((open_prices > high_prices) * (open_prices > np.roll(high_prices, 1))).astype(float)
            features['gap_down'] = ((open_prices < low_prices) * (open_prices < np.roll(low_prices, 1))).astype(float)
            
            # Inside/Outside bars
            prev_high = np.roll(high_prices, 1)
            prev_low = np.roll(low_prices, 1)
            
            features['inside_bar'] = ((high_prices < prev_high) & (low_prices > prev_low)).astype(float)
            features['outside_bar'] = ((high_prices > prev_high) & (low_prices < prev_low)).astype(float)
            
            # Higher highs, lower lows
            features['higher_high'] = (high_prices > np.roll(high_prices, 1)).astype(float)
            features['lower_low'] = (low_prices < np.roll(low_prices, 1)).astype(float)
            features['higher_low'] = (low_prices > np.roll(low_prices, 1)).astype(float)
            features['lower_high'] = (high_prices < np.roll(high_prices, 1)).astype(float)
            
            # Fractal patterns
            fractal_period = params.get('fractal_period', 5)
            features['fractal_high'] = self._detect_fractals(high_prices, fractal_period, 'high')
            features['fractal_low'] = self._detect_fractals(low_prices, fractal_period, 'low')
            
            # Support and resistance levels
            sr_window = params.get('sr_window', 20)
            features['near_resistance'] = self._near_support_resistance(close_prices, high_prices, sr_window, 'resistance')
            features['near_support'] = self._near_support_resistance(close_prices, low_prices, sr_window, 'support')
            
        except Exception as e:
            logger.error(f"Error in pattern feature extraction: {e}")
            
        return features
    
    def _extract_market_structure_features(
        self,
        data: pd.DataFrame,
        params: Dict
    ) -> pd.DataFrame:
        """Extract market microstructure and regime-specific features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            # Market efficiency measures
            # Hurst exponent (simplified)
            hurst_window = params.get('hurst_window', 100)
            features['hurst_exponent'] = self._calculate_rolling_hurst(close_prices, hurst_window)
            
            # Autocorrelation of returns
            returns = np.concatenate([[np.nan], np.diff(np.log(close_prices))])
            autocorr_lags = params.get('autocorr_lags', [1, 2, 5, 10])
            
            for lag in autocorr_lags:
                if lag < len(returns):
                    autocorr = self._rolling_autocorr(returns, lag, window=60)
                    features[f'autocorr_lag_{lag}'] = autocorr
                    
            # Variance ratio test statistic
            vr_periods = params.get('vr_periods', [5, 10])
            for period in vr_periods:
                if period < len(returns):
                    vr_stat = self._rolling_variance_ratio(returns, period, window=100)
                    features[f'variance_ratio_{period}'] = vr_stat
                    
            # Regime stability measures
            # Price trend consistency
            trend_window = params.get('trend_window', 20)
            price_changes = np.diff(close_prices)
            trend_consistency = pd.Series(price_changes).rolling(trend_window).apply(
                lambda x: np.sum(np.sign(x)) / len(x) if len(x) > 0 else 0
            ).values
            features['trend_consistency'] = trend_consistency
            
            # Volatility clustering
            vol_cluster = self._detect_volatility_clustering(returns, window=30)
            features['volatility_clustering'] = vol_cluster
            
            # Mean reversion tendency
            mean_reversion = self._calculate_mean_reversion_tendency(close_prices, window=30)
            features['mean_reversion_tendency'] = mean_reversion
            
            # Momentum persistence
            momentum_persist = self._calculate_momentum_persistence(returns, window=20)
            features['momentum_persistence'] = momentum_persist
            
            # Distribution features
            # Skewness and kurtosis of returns
            for window in [21, 63]:
                if window < len(returns):
                    rolling_returns = pd.Series(returns).rolling(window)
                    features[f'returns_skewness_{window}d'] = rolling_returns.skew().values
                    features[f'returns_kurtosis_{window}d'] = rolling_returns.kurt().values
                    
            # Tail risk measures
            # Value at Risk approximation
            var_confidence = params.get('var_confidence', 0.05)
            for window in [21, 63]:
                if window < len(returns):
                    var = pd.Series(returns).rolling(window).quantile(var_confidence).values
                    features[f'var_{int(var_confidence*100)}_{window}d'] = var
                    
        except Exception as e:
            logger.error(f"Error in market structure feature extraction: {e}")
            
        return features
    
    def _post_process_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Post-process extracted features"""
        
        if features.empty:
            return features
            
        # Handle missing values
        if self.fillna_method == 'forward':
            features = features.fillna(method='ffill')
        elif self.fillna_method == 'backward':
            features = features.fillna(method='bfill')
        elif self.fillna_method == 'zero':
            features = features.fillna(0)
        elif self.fillna_method == 'drop':
            features = features.dropna()
            
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        # Remove outliers if requested
        if self.remove_outliers:
            features = self._remove_statistical_outliers(features)
            
        # Remove constant features
        constant_features = []
        for col in features.columns:
            if features[col].std() == 0 or features[col].nunique() == 1:
                constant_features.append(col)
                
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            features = features.drop(columns=constant_features)
            
        return features
    
    def _remove_statistical_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers using z-score"""
        
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(features[col].fillna(features[col].mean())))
            features.loc[z_scores > self.outlier_threshold, col] = np.nan
            
        # Forward fill the NaN values created by outlier removal
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    # Fallback implementations for when TA-Lib is not available
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first value
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def _calculate_macd_fallback(self, close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD fallback implementation"""
        ema_fast = pd.Series(close).ewm(span=fast).mean()
        ema_slow = pd.Series(close).ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        
        return macd.values, macd_signal.values, macd_hist.values
    
    def _calculate_rsi_fallback(self, close: np.ndarray, period: int) -> np.ndarray:
        """RSI fallback implementation"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Prepend NaN to maintain array length
        gain = np.concatenate([[np.nan], gain])
        loss = np.concatenate([[np.nan], loss])
        
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values
    
    def _calculate_adx_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ADX fallback implementation"""
        # Simplified ADX calculation
        tr = self._calculate_true_range(high, low, close)
        
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                          np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.values, plus_di.values, minus_di.values
    
    def _calculate_stoch_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic oscillator fallback implementation"""
        lowest_low = pd.Series(low).rolling(k_period).min()
        highest_high = pd.Series(high).rolling(k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent.values, d_percent.values
    
    def _calculate_bollinger_bands_fallback(self, close: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands fallback implementation"""
        sma = pd.Series(close).rolling(period).mean()
        std = pd.Series(close).rolling(period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.values, sma.values, lower.values
    
    # Additional helper methods
    
    def _calculate_slope(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        """Calculate slope of values over rolling window"""
        slopes = np.full(len(values), np.nan)
        
        for i in range(window, len(values)):
            y = values[i-window+1:i+1]
            x = np.arange(len(y))
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                slopes[i] = slope
                
        return slopes
    
    def _calculate_garch_vol(self, returns: np.ndarray, alpha: float = 0.1, beta: float = 0.85) -> np.ndarray:
        """Simple GARCH(1,1) volatility estimation"""
        vol = np.full(len(returns), np.nan)
        
        # Initialize with sample standard deviation
        vol[0] = np.nanstd(returns[:30]) if len(returns) > 30 else np.nanstd(returns)
        
        for i in range(1, len(returns)):
            if not np.isnan(returns[i-1]) and not np.isnan(vol[i-1]):
                vol[i] = np.sqrt(alpha * returns[i-1]**2 + beta * vol[i-1]**2)
            else:
                vol[i] = vol[i-1] if not np.isnan(vol[i-1]) else vol[0]
                
        return vol
    
    def _rolling_autocorr(self, series: np.ndarray, lag: int, window: int) -> np.ndarray:
        """Calculate rolling autocorrelation"""
        autocorr = np.full(len(series), np.nan)
        
        for i in range(window, len(series)):
            data = series[i-window+1:i+1]
            if len(data) > lag:
                lagged_data = data[:-lag] if lag > 0 else data
                current_data = data[lag:] if lag > 0 else data
                
                if len(lagged_data) > 0 and len(current_data) > 0:
                    corr = np.corrcoef(lagged_data, current_data)[0, 1]
                    autocorr[i] = corr if not np.isnan(corr) else 0
                    
        return autocorr
    
    def _rolling_variance_ratio(self, returns: np.ndarray, period: int, window: int) -> np.ndarray:
        """Calculate rolling variance ratio test statistic"""
        vr_stat = np.full(len(returns), np.nan)
        
        for i in range(window, len(returns)):
            data = returns[i-window+1:i+1]
            data = data[~np.isnan(data)]
            
            if len(data) > period * 2:
                # Variance of period-period returns
                period_returns = np.array([np.sum(data[j:j+period]) for j in range(0, len(data)-period+1, period)])
                period_var = np.var(period_returns) if len(period_returns) > 1 else 0
                
                # Variance of single-period returns
                single_var = np.var(data) if len(data) > 1 else 0
                
                if single_var > 0:
                    vr_stat[i] = period_var / (period * single_var)
                    
        return vr_stat
    
    def _rolling_hurst(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling Hurst exponent (simplified)"""
        hurst = np.full(len(prices), np.nan)
        
        for i in range(window, len(prices)):
            data = prices[i-window+1:i+1]
            if len(data) > 10:
                try:
                    # Simplified Hurst calculation
                    lags = range(2, min(20, len(data)//2))
                    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
                    
                    if len(tau) > 2:
                        poly = np.polyfit(np.log(lags), np.log(tau), 1)
                        hurst[i] = poly[0]
                except:
                    hurst[i] = 0.5  # Default to 0.5 (random walk)
                    
        return hurst
    
    def _detect_volatility_clustering(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Detect volatility clustering using rolling correlation of squared returns"""
        squared_returns = returns ** 2
        clustering = np.full(len(returns), np.nan)
        
        for i in range(window, len(returns)):
            data = squared_returns[i-window+1:i+1]
            data = data[~np.isnan(data)]
            
            if len(data) > 5:
                # Autocorrelation of squared returns
                autocorr = np.corrcoef(data[:-1], data[1:])[0, 1] if len(data) > 1 else 0
                clustering[i] = autocorr if not np.isnan(autocorr) else 0
                
        return clustering
    
    def _calculate_mean_reversion_tendency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate mean reversion tendency"""
        mean_reversion = np.full(len(prices), np.nan)
        
        for i in range(window, len(prices)):
            data = prices[i-window+1:i+1]
            if len(data) > 5:
                # Calculate how often price returns to mean
                mean_price = np.mean(data)
                deviations = data - mean_price
                
                # Count sign changes (crossings of mean)
                sign_changes = np.sum(np.diff(np.sign(deviations)) != 0)
                max_possible = len(data) - 1
                
                mean_reversion[i] = sign_changes / max_possible if max_possible > 0 else 0
                
        return mean_reversion
    
    def _calculate_momentum_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate momentum persistence"""
        persistence = np.full(len(returns), np.nan)
        
        for i in range(window, len(returns)):
            data = returns[i-window+1:i+1]
            data = data[~np.isnan(data)]
            
            if len(data) > 5:
                # Calculate persistence of momentum direction
                signs = np.sign(data)
                sign_changes = np.sum(np.diff(signs) != 0)
                max_changes = len(data) - 1
                
                persistence[i] = 1 - (sign_changes / max_changes) if max_changes > 0 else 1
                
        return persistence
    
    # Additional fallback methods for less common indicators
    
    def _calculate_willr_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Williams %R fallback implementation"""
        highest_high = pd.Series(high).rolling(period).max()
        lowest_low = pd.Series(low).rolling(period).min()
        
        willr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return willr.values
    
    def _calculate_roc_fallback(self, close: np.ndarray, period: int) -> np.ndarray:
        """Rate of Change fallback implementation"""
        roc = ((close - np.roll(close, period)) / np.roll(close, period)) * 100
        roc[:period] = np.nan
        return roc
    
    def _calculate_cci_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Commodity Channel Index fallback implementation"""
        typical_price = (high + low + close) / 3
        sma_tp = pd.Series(typical_price).rolling(period).mean()
        mad = pd.Series(typical_price).rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci.values
    
    def _calculate_sar_fallback(self, high: np.ndarray, low: np.ndarray, accel: float, max_accel: float) -> np.ndarray:
        """Parabolic SAR fallback implementation (simplified)"""
        sar = np.full(len(high), np.nan)
        ep = np.full(len(high), np.nan)
        af = np.full(len(high), accel)
        trend = np.full(len(high), 1)  # 1 for up, -1 for down
        
        if len(high) > 0:
            sar[0] = low[0]
            ep[0] = high[0]
            
        for i in range(1, len(high)):
            # Simplified SAR calculation
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + accel, max_accel)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                    
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = accel
                else:
                    trend[i] = 1
                    
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + accel, max_accel)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                    
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = accel
                else:
                    trend[i] = -1
                    
        return sar
    
    def _calculate_obv_fallback(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On-Balance Volume fallback implementation"""
        obv = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        return obv
    
    def _calculate_mfi_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """Money Flow Index fallback implementation"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = np.where(np.diff(np.concatenate([[typical_price[0]], typical_price])) > 0, money_flow, 0)
        negative_flow = np.where(np.diff(np.concatenate([[typical_price[0]], typical_price])) < 0, money_flow, 0)
        
        pos_mf = pd.Series(positive_flow).rolling(period).sum()
        neg_mf = pd.Series(negative_flow).rolling(period).sum()
        
        money_ratio = pos_mf / neg_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi.values
    
    def _calculate_vpt(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Volume Price Trend"""
        vpt = np.zeros(len(close))
        
        for i in range(1, len(close)):
            vpt[i] = vpt[i-1] + volume[i] * (close[i] - close[i-1]) / close[i-1]
            
        return vpt
    
    def _calculate_ad_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Accumulation/Distribution Line fallback implementation"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = np.where(high == low, 0, clv)  # Handle division by zero
        
        money_flow_volume = clv * volume
        ad_line = np.cumsum(money_flow_volume)
        
        return ad_line
    
    def _calculate_cmf(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """Chaikin Money Flow"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = np.where(high == low, 0, clv)
        
        money_flow_volume = clv * volume
        cmf = pd.Series(money_flow_volume).rolling(period).sum() / pd.Series(volume).rolling(period).sum()
        
        return cmf.values
    
    def _detect_fractals(self, prices: np.ndarray, period: int, fractal_type: str) -> np.ndarray:
        """Detect fractal patterns"""
        fractals = np.zeros(len(prices))
        half_period = period // 2
        
        for i in range(half_period, len(prices) - half_period):
            window = prices[i-half_period:i+half_period+1]
            
            if fractal_type == 'high':
                if prices[i] == np.max(window):
                    fractals[i] = 1
            else:  # low
                if prices[i] == np.min(window):
                    fractals[i] = 1
                    
        return fractals
    
    def _near_support_resistance(self, close: np.ndarray, extreme_prices: np.ndarray, window: int, level_type: str) -> np.ndarray:
        """Detect when price is near support/resistance levels"""
        near_level = np.zeros(len(close))
        
        for i in range(window, len(close)):
            recent_extremes = extreme_prices[i-window:i]
            
            if level_type == 'resistance':
                level = np.max(recent_extremes)
                # Consider "near" as within 2% of the level
                near_level[i] = 1 if close[i] > level * 0.98 else 0
            else:  # support
                level = np.min(recent_extremes)
                near_level[i] = 1 if close[i] < level * 1.02 else 0
                
        return near_level