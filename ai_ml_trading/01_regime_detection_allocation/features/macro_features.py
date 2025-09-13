"""
Macro-economic feature extraction with FRED API integration.

This module provides comprehensive macro-economic indicators for regime detection,
including yield curves, economic indicators, volatility measures, and custom composites.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
import time

# Optional imports with fallbacks
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    Fred = None

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


class MacroFeatureExtractor:
    """
    Comprehensive macro-economic feature extractor using FRED API and other sources.
    
    Features include:
    - Yield curve indicators
    - Economic growth indicators  
    - Inflation measures
    - Employment data
    - Monetary policy indicators
    - Market stress indicators
    - Custom composite indices
    """
    
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_data: bool = True,
        request_delay: float = 0.1,
        max_retries: int = 3
    ):
        """
        Initialize macro feature extractor.
        
        Parameters:
        -----------
        fred_api_key : str, optional
            FRED API key for accessing data
        cache_data : bool
            Whether to cache downloaded data
        request_delay : float
            Delay between API requests in seconds
        max_retries : int
            Maximum number of retry attempts for failed requests
        """
        self.fred_api_key = fred_api_key
        self.cache_data = cache_data
        self.request_delay = request_delay
        self.max_retries = max_retries
        
        # Initialize FRED client
        if FRED_AVAILABLE and fred_api_key:
            try:
                self.fred = Fred(api_key=fred_api_key)
                logger.info("FRED API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize FRED client: {e}")
                self.fred = None
        else:
            self.fred = None
            if not FRED_AVAILABLE:
                logger.warning("fredapi not available, FRED data will be unavailable")
                
        # Data cache
        self._data_cache = {}
        
        # Define indicator mappings
        self._setup_indicators()
    
    def _setup_indicators(self):
        """Setup indicator mappings and metadata"""
        
        # FRED series mappings
        self.fred_series = {
            # Yield Curve
            'DGS3MO': '3-Month Treasury',
            'DGS6MO': '6-Month Treasury', 
            'DGS1': '1-Year Treasury',
            'DGS2': '2-Year Treasury',
            'DGS3': '3-Year Treasury',
            'DGS5': '5-Year Treasury',
            'DGS7': '7-Year Treasury',
            'DGS10': '10-Year Treasury',
            'DGS20': '20-Year Treasury',
            'DGS30': '30-Year Treasury',
            
            # Economic Growth
            'GDP': 'Real GDP',
            'GDPC1': 'Real GDP (Quarterly)',
            'INDPRO': 'Industrial Production Index',
            'PAYEMS': 'Total Nonfarm Payrolls',
            'UNRATE': 'Unemployment Rate',
            'CIVPART': 'Labor Force Participation Rate',
            'HOUST': 'Housing Starts',
            'PERMIT': 'Building Permits',
            
            # Inflation
            'CPIAUCSL': 'Consumer Price Index',
            'CPILFESL': 'Core CPI',
            'PCEPI': 'PCE Price Index',
            'PCEPILFE': 'Core PCE Price Index',
            'DFEDTARU': 'Federal Funds Target Rate Upper Limit',
            'DFEDTARL': 'Federal Funds Target Rate Lower Limit',
            
            # Money Supply & Credit
            'M1SL': 'M1 Money Supply',
            'M2SL': 'M2 Money Supply',
            'BOGMBASE': 'Monetary Base',
            'TOTRESNS': 'Total Bank Reserves',
            'DEXUSEU': 'USD/EUR Exchange Rate',
            'DTWEXBGS': 'Trade Weighted USD Index - Broad',
            
            # Market Indicators
            'NASDAQCOM': 'NASDAQ Composite Index',
            'SP500': 'S&P 500',
            'VIXCLS': 'VIX Volatility Index',
            'DCOILWTICO': 'WTI Crude Oil Price',
            'DCOILBRENTEU': 'Brent Oil Price',
            'GOLDAMGBD228NLBM': 'Gold Price',
            
            # Credit & Risk
            'BAMLH0A0HYM2': 'High Yield Credit Spread',
            'BAMLC0A1CAAAEY': 'AAA Credit Spread',
            'BAMLC0A2CAAEY': 'AA Credit Spread',
            'BAMLC0A3CAEY': 'A Credit Spread',
            'BAMLC0A4CBBBEY': 'BBB Credit Spread',
            'DGS10': '10-Year Treasury Yield',
            'TB3MS': '3-Month Treasury Bill',
            
            # Economic Policy Uncertainty
            'USEPUINDXD': 'Economic Policy Uncertainty Index',
            
            # Consumer & Business Sentiment
            'UMCSENT': 'Consumer Sentiment',
            'BSCICP02USM665S': 'Business Confidence',
        }
        
        # Alternative data sources for non-FRED indicators
        self.yahoo_tickers = {
            '^VIX': 'VIX',
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            '^TNX': '10-Year Treasury Yield',
            'GLD': 'Gold ETF',
            'TLT': 'Long Treasury ETF',
            'HYG': 'High Yield ETF',
            'LQD': 'Investment Grade ETF',
            'DXY': 'Dollar Index',
            'CL=F': 'WTI Oil Futures',
            'BZ=F': 'Brent Oil Futures'
        }
    
    def fetch_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        indicators: Optional[List[str]] = None,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Fetch macro-economic data from various sources.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        indicators : list, optional
            Specific indicators to fetch (if None, fetches all available)
        frequency : str
            Data frequency: 'daily', 'weekly', 'monthly'
            
        Returns:
        --------
        pd.DataFrame
            Raw macro-economic data
        """
        
        logger.info(f"Fetching macro data from {start_date} to {end_date}")
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        all_data = {}
        
        # Fetch FRED data
        if self.fred is not None:
            fred_data = self._fetch_fred_data(start_date, end_date, indicators, frequency)
            all_data.update(fred_data)
            
        # Fetch Yahoo Finance data
        if YFINANCE_AVAILABLE:
            yahoo_data = self._fetch_yahoo_data(start_date, end_date, indicators, frequency)
            all_data.update(yahoo_data)
            
        if not all_data:
            logger.warning("No macro data could be fetched")
            return pd.DataFrame()
            
        # Combine all data
        combined_df = pd.DataFrame(all_data)
        
        # Handle missing values
        combined_df = self._handle_missing_values(combined_df, frequency)
        
        logger.info(f"Fetched {len(combined_df.columns)} indicators, {len(combined_df)} observations")
        
        return combined_df
    
    def _fetch_fred_data(
        self,
        start_date: datetime,
        end_date: datetime,
        indicators: Optional[List[str]],
        frequency: str
    ) -> Dict[str, pd.Series]:
        """Fetch data from FRED API"""
        
        fred_data = {}
        
        # Determine which series to fetch
        series_to_fetch = list(self.fred_series.keys())
        if indicators:
            series_to_fetch = [s for s in series_to_fetch if s in indicators or 
                             self.fred_series.get(s) in indicators]
            
        logger.info(f"Fetching {len(series_to_fetch)} FRED series")
        
        for i, series_id in enumerate(series_to_fetch):
            try:
                # Check cache first
                cache_key = f"fred_{series_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{frequency}"
                
                if self.cache_data and cache_key in self._data_cache:
                    fred_data[series_id] = self._data_cache[cache_key]
                    continue
                    
                # Add delay between requests
                if i > 0:
                    time.sleep(self.request_delay)
                    
                # Fetch data with retries
                for attempt in range(self.max_retries):
                    try:
                        series_data = self.fred.get_series(
                            series_id, 
                            start_date, 
                            end_date
                        )
                        
                        if len(series_data) > 0:
                            # Resample if needed
                            if frequency == 'weekly':
                                series_data = series_data.resample('W').last()
                            elif frequency == 'monthly':
                                series_data = series_data.resample('M').last()
                                
                            fred_data[series_id] = series_data
                            
                            # Cache the data
                            if self.cache_data:
                                self._data_cache[cache_key] = series_data
                                
                            logger.debug(f"Fetched {series_id}: {len(series_data)} observations")
                            break
                            
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed for {series_id}: {e}")
                            time.sleep(self.request_delay * (attempt + 1))
                        else:
                            logger.error(f"Failed to fetch {series_id} after {self.max_retries} attempts: {e}")
                            
            except Exception as e:
                logger.error(f"Error fetching {series_id}: {e}")
                continue
                
        return fred_data
    
    def _fetch_yahoo_data(
        self,
        start_date: datetime,
        end_date: datetime,
        indicators: Optional[List[str]],
        frequency: str
    ) -> Dict[str, pd.Series]:
        """Fetch data from Yahoo Finance"""
        
        yahoo_data = {}
        
        # Determine which tickers to fetch
        tickers_to_fetch = list(self.yahoo_tickers.keys())
        if indicators:
            tickers_to_fetch = [t for t in tickers_to_fetch if t in indicators or 
                              self.yahoo_tickers.get(t) in indicators]
            
        if not tickers_to_fetch:
            return yahoo_data
            
        logger.info(f"Fetching {len(tickers_to_fetch)} Yahoo Finance tickers")
        
        try:
            # Fetch multiple tickers at once
            tickers_str = ' '.join(tickers_to_fetch)
            data = yf.download(
                tickers_str,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if len(tickers_to_fetch) == 1:
                # Single ticker - data is a DataFrame
                if 'Adj Close' in data.columns:
                    yahoo_data[tickers_to_fetch[0]] = data['Adj Close']
                elif 'Close' in data.columns:
                    yahoo_data[tickers_to_fetch[0]] = data['Close']
            else:
                # Multiple tickers - data is MultiIndex
                if 'Adj Close' in data.columns.get_level_values(0):
                    adj_close_data = data['Adj Close']
                    for ticker in tickers_to_fetch:
                        if ticker in adj_close_data.columns:
                            yahoo_data[ticker] = adj_close_data[ticker]
                            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            
        return yahoo_data
    
    def _handle_missing_values(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Handle missing values in the combined dataset"""
        
        if df.empty:
            return df
            
        logger.info("Handling missing values in macro data")
        
        # Forward fill then backward fill
        df_filled = df.fillna(method='ffill').fillna(method='bfill')
        
        # For remaining NaNs, interpolate if reasonable
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:
                try:
                    # Only interpolate if less than 30% missing
                    missing_pct = df_filled[col].isnull().sum() / len(df_filled)
                    if missing_pct < 0.3:
                        df_filled[col] = df_filled[col].interpolate(method='linear', limit_direction='both')
                    else:
                        logger.warning(f"High missing data percentage for {col}: {missing_pct:.1%}")
                except Exception as e:
                    logger.warning(f"Interpolation failed for {col}: {e}")
                    
        # Drop columns that are still mostly NaN
        cols_to_drop = []
        for col in df_filled.columns:
            missing_pct = df_filled[col].isnull().sum() / len(df_filled)
            if missing_pct > 0.5:
                cols_to_drop.append(col)
                logger.warning(f"Dropping {col} due to high missing data: {missing_pct:.1%}")
                
        df_filled = df_filled.drop(columns=cols_to_drop)
        
        return df_filled
    
    def extract_features(
        self,
        data: pd.DataFrame,
        feature_types: Optional[List[str]] = None,
        lookback_windows: List[int] = [5, 21, 63, 252]
    ) -> pd.DataFrame:
        """
        Extract engineered features from raw macro data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw macro-economic data
        feature_types : list, optional
            Types of features to extract
        lookback_windows : list
            Lookback windows for rolling calculations
            
        Returns:
        --------
        pd.DataFrame
            Engineered macro features
        """
        
        if data.empty:
            logger.warning("No data provided for feature extraction")
            return pd.DataFrame()
            
        logger.info("Extracting macro features")
        
        if feature_types is None:
            feature_types = [
                'yield_curve',
                'growth_momentum', 
                'inflation_signals',
                'credit_risk',
                'market_stress',
                'policy_indicators',
                'composite_indices'
            ]
            
        all_features = pd.DataFrame(index=data.index)
        
        # Extract different types of features
        for feature_type in feature_types:
            try:
                if feature_type == 'yield_curve':
                    features = self._extract_yield_curve_features(data, lookback_windows)
                elif feature_type == 'growth_momentum':
                    features = self._extract_growth_features(data, lookback_windows)
                elif feature_type == 'inflation_signals':
                    features = self._extract_inflation_features(data, lookback_windows)
                elif feature_type == 'credit_risk':
                    features = self._extract_credit_features(data, lookback_windows)
                elif feature_type == 'market_stress':
                    features = self._extract_stress_features(data, lookback_windows)
                elif feature_type == 'policy_indicators':
                    features = self._extract_policy_features(data, lookback_windows)
                elif feature_type == 'composite_indices':
                    features = self._extract_composite_features(data, lookback_windows)
                else:
                    logger.warning(f"Unknown feature type: {feature_type}")
                    continue
                    
                # Add features to main DataFrame
                for col in features.columns:
                    if col not in all_features.columns:
                        all_features[col] = features[col]
                        
            except Exception as e:
                logger.error(f"Error extracting {feature_type} features: {e}")
                continue
                
        # Clean up features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} macro features")
        
        return all_features
    
    def _extract_yield_curve_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Extract yield curve related features"""
        
        features = pd.DataFrame(index=data.index)
        
        # Identify yield columns
        yield_cols = [col for col in data.columns if any(term in col.upper() for term in 
                     ['DGS', 'TNX', 'TREASURY', 'YIELD', 'TB3MS'])]
        
        if len(yield_cols) < 2:
            logger.warning("Insufficient yield curve data")
            return features
            
        try:
            # Term spreads
            long_yields = [col for col in yield_cols if any(term in col for term in ['DGS10', 'DGS30', 'DGS20', 'TNX'])]
            short_yields = [col for col in yield_cols if any(term in col for term in ['DGS3MO', 'DGS6MO', 'DGS1', 'TB3MS'])]
            
            if long_yields and short_yields:
                long_yield = data[long_yields[0]]
                short_yield = data[short_yields[0]]
                
                # 10Y-2Y spread (recession indicator)
                if 'DGS10' in data.columns and 'DGS2' in data.columns:
                    features['yield_spread_10y2y'] = data['DGS10'] - data['DGS2']
                    
                    # Inverted yield curve indicator
                    features['yield_curve_inverted'] = (features['yield_spread_10y2y'] < 0).astype(float)
                    
                # General term spread
                features['term_spread'] = long_yield - short_yield
                
                # Term spread momentum
                for window in windows:
                    if window < len(data):
                        features[f'term_spread_momentum_{window}d'] = features['term_spread'].diff(window)
                        
            # Yield curve level (average of available yields)
            available_yields = [col for col in yield_cols if col in data.columns]
            if available_yields:
                features['yield_curve_level'] = data[available_yields].mean(axis=1)
                
                # Yield curve slope (approximate)
                if len(available_yields) >= 2:
                    # Sort by maturity (approximate)
                    sorted_yields = sorted(available_yields, key=lambda x: 
                                         int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
                    if len(sorted_yields) >= 2:
                        features['yield_curve_slope'] = data[sorted_yields[-1]] - data[sorted_yields[0]]
                        
                # Yield volatility
                for window in windows:
                    if window < len(data):
                        features[f'yield_volatility_{window}d'] = features['yield_curve_level'].rolling(window).std()
                        
        except Exception as e:
            logger.error(f"Error in yield curve feature extraction: {e}")
            
        return features
    
    def _extract_growth_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Extract economic growth related features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # Industrial production momentum
            if 'INDPRO' in data.columns:
                for window in windows:
                    if window < len(data):
                        features[f'indpro_momentum_{window}d'] = data['INDPRO'].pct_change(window)
                        
            # Employment momentum
            if 'PAYEMS' in data.columns:
                for window in windows:
                    if window < len(data):
                        features[f'payems_momentum_{window}d'] = data['PAYEMS'].pct_change(window)
                        
            # Unemployment rate changes
            if 'UNRATE' in data.columns:
                features['unemployment_change'] = data['UNRATE'].diff()
                
                for window in windows:
                    if window < len(data):
                        features[f'unemployment_momentum_{window}d'] = data['UNRATE'].diff(window)
                        
            # Housing market indicators
            housing_cols = [col for col in data.columns if any(term in col.upper() for term in 
                          ['HOUST', 'PERMIT', 'HOUSING'])]
            
            for col in housing_cols:
                for window in windows:
                    if window < len(data):
                        features[f'{col.lower()}_momentum_{window}d'] = data[col].pct_change(window)
                        
            # GDP indicators (if available)
            gdp_cols = [col for col in data.columns if 'GDP' in col.upper()]
            for col in gdp_cols:
                for window in windows:
                    if window < len(data):
                        features[f'{col.lower()}_growth_{window}d'] = data[col].pct_change(window)
                        
        except Exception as e:
            logger.error(f"Error in growth feature extraction: {e}")
            
        return features
    
    def _extract_inflation_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Extract inflation related features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # CPI momentum
            cpi_cols = [col for col in data.columns if 'CPI' in col.upper()]
            for col in cpi_cols:
                for window in windows:
                    if window < len(data):
                        # Annualized inflation rate
                        annual_factor = 252 / window  # Assuming daily data
                        features[f'{col.lower()}_inflation_{window}d'] = data[col].pct_change(window) * annual_factor
                        
            # PCE momentum
            pce_cols = [col for col in data.columns if 'PCE' in col.upper()]
            for col in pce_cols:
                for window in windows:
                    if window < len(data):
                        annual_factor = 252 / window
                        features[f'{col.lower()}_inflation_{window}d'] = data[col].pct_change(window) * annual_factor
                        
            # Breakeven inflation (if Treasury and TIPS data available)
            # This would require TIPS data which isn't in our basic setup
            
            # Commodity price inflation proxy
            commodity_cols = [col for col in data.columns if any(term in col.upper() for term in 
                            ['OIL', 'GOLD', 'COMMODITY', 'CL=F', 'BZ=F'])]
            
            if commodity_cols:
                commodity_inflation = pd.DataFrame(index=data.index)
                for col in commodity_cols:
                    for window in [21, 63]:  # Monthly and quarterly
                        if window < len(data):
                            annual_factor = 252 / window
                            commodity_inflation[f'{col}_inflation'] = data[col].pct_change(window) * annual_factor
                            
                # Composite commodity inflation
                if len(commodity_inflation.columns) > 0:
                    features['commodity_inflation'] = commodity_inflation.mean(axis=1)
                    
        except Exception as e:
            logger.error(f"Error in inflation feature extraction: {e}")
            
        return features
    
    def _extract_credit_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Extract credit and risk related features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # Credit spreads
            spread_cols = [col for col in data.columns if any(term in col.upper() for term in 
                         ['BAML', 'SPREAD', 'CREDIT', 'HYG', 'LQD'])]
            
            for col in spread_cols:
                # Spread levels
                features[f'{col.lower()}_level'] = data[col]
                
                # Spread changes
                for window in windows:
                    if window < len(data):
                        features[f'{col.lower()}_change_{window}d'] = data[col].diff(window)
                        
                # Spread volatility
                for window in [21, 63]:
                    if window < len(data):
                        features[f'{col.lower()}_volatility_{window}d'] = data[col].rolling(window).std()
                        
            # Credit spread differentials (if multiple spreads available)
            if len(spread_cols) >= 2:
                # High yield vs investment grade
                hy_cols = [col for col in spread_cols if any(term in col.upper() for term in ['HY', 'HIGH', 'HYG'])]
                ig_cols = [col for col in spread_cols if any(term in col.upper() for term in ['AAA', 'AA', 'LQD', 'IG'])]
                
                if hy_cols and ig_cols:
                    hy_spread = data[hy_cols[0]]
                    ig_spread = data[ig_cols[0]]
                    features['credit_quality_spread'] = hy_spread - ig_spread
                    
                    for window in windows:
                        if window < len(data):
                            features[f'credit_quality_spread_change_{window}d'] = features['credit_quality_spread'].diff(window)
                            
        except Exception as e:
            logger.error(f"Error in credit feature extraction: {e}")
            
        return features
    
    def _extract_stress_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Extract market stress and volatility features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # VIX features
            vix_cols = [col for col in data.columns if 'VIX' in col.upper()]
            for col in vix_cols:
                features[f'{col.lower()}_level'] = data[col]
                
                # VIX percentiles (stress indicators)
                for window in [63, 252]:
                    if window < len(data):
                        features[f'{col.lower()}_percentile_{window}d'] = data[col].rolling(window).rank(pct=True)
                        
                # VIX spikes
                for window in [5, 21]:
                    if window < len(data):
                        features[f'{col.lower()}_spike_{window}d'] = (data[col] > data[col].rolling(window).quantile(0.9)).astype(float)
                        
            # Market volatility from price data
            market_cols = [col for col in data.columns if any(term in col.upper() for term in 
                         ['^GSPC', 'SP500', '^IXIC', 'NASDAQ', '^DJI'])]
            
            for col in market_cols:
                # Realized volatility
                returns = data[col].pct_change()
                for window in [21, 63]:
                    if window < len(data):
                        features[f'{col.lower().replace("^", "")}_realized_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
                        
            # Currency volatility (if USD index available)
            fx_cols = [col for col in data.columns if any(term in col.upper() for term in 
                     ['DXY', 'DEXUSEU', 'DTWEXBGS', 'USD'])]
            
            for col in fx_cols:
                returns = data[col].pct_change()
                for window in [21, 63]:
                    if window < len(data):
                        features[f'{col.lower()}_volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
                        
        except Exception as e:
            logger.error(f"Error in stress feature extraction: {e}")
            
        return features
    
    def _extract_policy_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Extract monetary and fiscal policy indicators"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # Federal funds rate
            fed_cols = [col for col in data.columns if any(term in col.upper() for term in 
                      ['DFEDTAR', 'FEDFUNDS', 'FED', 'RATE'])]
            
            for col in fed_cols:
                features[f'{col.lower()}_level'] = data[col]
                
                # Rate changes
                for window in [1, 21, 63]:
                    if window < len(data):
                        features[f'{col.lower()}_change_{window}d'] = data[col].diff(window)
                        
            # Money supply growth
            money_cols = [col for col in data.columns if any(term in col.upper() for term in 
                        ['M1SL', 'M2SL', 'BOGMBASE', 'MONEY'])]
            
            for col in money_cols:
                for window in [63, 252]:  # Quarterly and annual
                    if window < len(data):
                        features[f'{col.lower()}_growth_{window}d'] = data[col].pct_change(window)
                        
            # Bank reserves
            reserve_cols = [col for col in data.columns if 'RESERVE' in col.upper() or 'TOTRESNS' in col]
            for col in reserve_cols:
                for window in [63, 252]:
                    if window < len(data):
                        features[f'{col.lower()}_growth_{window}d'] = data[col].pct_change(window)
                        
            # Economic Policy Uncertainty
            if 'USEPUINDXD' in data.columns:
                features['policy_uncertainty'] = data['USEPUINDXD']
                
                for window in windows:
                    if window < len(data):
                        features[f'policy_uncertainty_change_{window}d'] = data['USEPUINDXD'].diff(window)
                        features[f'policy_uncertainty_percentile_{window}d'] = data['USEPUINDXD'].rolling(window).rank(pct=True)
                        
        except Exception as e:
            logger.error(f"Error in policy feature extraction: {e}")
            
        return features
    
    def _extract_composite_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Extract composite indices and cross-asset features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # Financial Conditions Index (simplified)
            components = {}
            
            # Yield curve component
            if 'term_spread' in data.columns:
                components['yield_curve'] = -data['term_spread']  # Inverted for stress direction
            elif any('DGS10' in col and 'DGS2' in col for col in data.columns):
                if 'DGS10' in data.columns and 'DGS2' in data.columns:
                    components['yield_curve'] = -(data['DGS10'] - data['DGS2'])
                    
            # Credit component
            credit_cols = [col for col in data.columns if any(term in col.upper() for term in 
                         ['BAML', 'HYG', 'CREDIT']) and 'SPREAD' in col.upper()]
            if credit_cols:
                components['credit'] = data[credit_cols[0]]
                
            # Volatility component  
            vix_cols = [col for col in data.columns if 'VIX' in col.upper()]
            if vix_cols:
                components['volatility'] = data[vix_cols[0]]
                
            # Currency component
            fx_cols = [col for col in data.columns if any(term in col.upper() for term in ['DXY', 'DTWEXBGS'])]
            if fx_cols:
                # Dollar strength indicates tightening financial conditions
                components['currency'] = data[fx_cols[0]].pct_change(21)  # Monthly change
                
            # Create composite Financial Conditions Index
            if len(components) >= 2:
                # Standardize components
                standardized_components = pd.DataFrame(index=data.index)
                for name, series in components.items():
                    standardized_components[name] = (series - series.rolling(252).mean()) / series.rolling(252).std()
                    
                features['financial_conditions_index'] = standardized_components.mean(axis=1)
                
                # FCI momentum
                for window in [5, 21, 63]:
                    if window < len(data):
                        features[f'fci_momentum_{window}d'] = features['financial_conditions_index'].diff(window)
                        
            # Risk-On/Risk-Off Index
            risk_on_components = {}
            
            # Stock market performance
            equity_cols = [col for col in data.columns if any(term in col.upper() for term in 
                         ['^GSPC', 'SP500', '^IXIC', 'NASDAQ'])]
            if equity_cols:
                risk_on_components['equity'] = data[equity_cols[0]].pct_change(21)
                
            # High yield credit performance  
            hy_cols = [col for col in data.columns if any(term in col.upper() for term in ['HYG']) 
                      and 'SPREAD' not in col.upper()]
            if hy_cols:
                risk_on_components['high_yield'] = data[hy_cols[0]].pct_change(21)
                
            # Commodity performance
            commodity_cols = [col for col in data.columns if any(term in col.upper() for term in 
                            ['OIL', 'GOLD', 'CL=F']) and 'PRICE' not in col.upper()]
            if commodity_cols:
                risk_on_components['commodities'] = data[commodity_cols[0]].pct_change(21)
                
            if len(risk_on_components) >= 2:
                standardized_risk = pd.DataFrame(index=data.index)
                for name, series in risk_on_components.items():
                    standardized_risk[name] = (series - series.rolling(252).mean()) / series.rolling(252).std()
                    
                features['risk_appetite_index'] = standardized_risk.mean(axis=1)
                
        except Exception as e:
            logger.error(f"Error in composite feature extraction: {e}")
            
        return features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and post-process extracted features"""
        
        if features.empty:
            return features
            
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values (up to 5 periods)
        features = features.fillna(method='ffill', limit=5)
        
        # Remove columns with too many missing values
        missing_threshold = 0.3
        cols_to_keep = []
        
        for col in features.columns:
            missing_pct = features[col].isnull().sum() / len(features)
            if missing_pct <= missing_threshold:
                cols_to_keep.append(col)
            else:
                logger.warning(f"Dropping feature {col} due to {missing_pct:.1%} missing values")
                
        features = features[cols_to_keep]
        
        # Winsorize extreme outliers
        for col in features.select_dtypes(include=[np.number]).columns:
            try:
                q01 = features[col].quantile(0.01)
                q99 = features[col].quantile(0.99)
                features[col] = features[col].clip(lower=q01, upper=q99)
            except Exception as e:
                logger.warning(f"Could not winsorize {col}: {e}")
                
        return features
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available features"""
        
        descriptions = {
            'yield_curve': 'Yield curve shape, term spreads, and interest rate indicators',
            'growth_momentum': 'Economic growth indicators and momentum measures',
            'inflation_signals': 'Price level changes and inflation expectations',
            'credit_risk': 'Credit spreads and risk appetite measures',
            'market_stress': 'Volatility indices and market stress indicators',
            'policy_indicators': 'Monetary policy and central bank indicators',
            'composite_indices': 'Custom composite indices combining multiple factors'
        }
        
        return descriptions


def create_sample_macro_data(
    start_date: str = '2010-01-01',
    end_date: str = '2023-12-31',
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    Create sample macro data for testing when FRED API is not available.
    
    Parameters:
    -----------
    start_date : str
        Start date
    end_date : str
        End date  
    frequency : str
        Data frequency
        
    Returns:
    --------
    pd.DataFrame
        Sample macro-economic data
    """
    
    logger.info(f"Creating sample macro data from {start_date} to {end_date}")
    
    # Create date range
    if frequency == 'daily':
        dates = pd.busi_date_range(start=start_date, end=end_date)
    elif frequency == 'weekly':
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
    else:  # monthly
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
    n_obs = len(dates)
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic macro data with realistic patterns
    data = pd.DataFrame(index=dates)
    
    # Yield curve data
    base_rate = 2.0 + 2.0 * np.sin(np.arange(n_obs) * 2 * np.pi / 252) + np.cumsum(np.random.normal(0, 0.01, n_obs))
    data['DGS3MO'] = np.maximum(base_rate + np.random.normal(0, 0.5, n_obs), 0.1)
    data['DGS2'] = np.maximum(data['DGS3MO'] + 0.5 + np.random.normal(0, 0.3, n_obs), 0.1)
    data['DGS10'] = np.maximum(data['DGS2'] + 1.0 + np.random.normal(0, 0.4, n_obs), 0.1)
    data['DGS30'] = np.maximum(data['DGS10'] + 0.3 + np.random.normal(0, 0.3, n_obs), 0.1)
    
    # Economic indicators
    data['UNRATE'] = np.maximum(4.0 + 2.0 * np.sin(np.arange(n_obs) * 2 * np.pi / (4 * 252)) + 
                               np.cumsum(np.random.normal(0, 0.02, n_obs)), 2.0)
    
    data['INDPRO'] = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, n_obs)))
    data['PAYEMS'] = 130000 + np.cumsum(np.random.normal(200, 100, n_obs))
    
    # Inflation indicators
    base_inflation = 2.0 + np.sin(np.arange(n_obs) * 2 * np.pi / (3 * 252))
    data['CPIAUCSL'] = 200 + np.cumsum(base_inflation / 252 + np.random.normal(0, 0.001, n_obs))
    data['CPILFESL'] = 200 + np.cumsum((base_inflation * 0.8) / 252 + np.random.normal(0, 0.0008, n_obs))
    
    # Market indicators
    data['SP500'] = 2000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, n_obs)))
    data['VIXCLS'] = np.maximum(10 + 15 * np.random.beta(2, 5, n_obs) + 
                               5 * np.abs(np.random.normal(0, 1, n_obs)), 5)
    
    # Credit spreads
    data['BAMLH0A0HYM2'] = np.maximum(300 + 200 * np.random.beta(2, 3, n_obs) + 
                                    100 * np.sin(np.arange(n_obs) * 2 * np.pi / (5 * 252)), 50)
    
    # Commodities
    data['DCOILWTICO'] = 50 + 30 * np.sin(np.arange(n_obs) * 2 * np.pi / (2 * 252)) + np.cumsum(np.random.normal(0, 0.02, n_obs))
    data['GOLDAMGBD228NLBM'] = 1200 + np.cumsum(np.random.normal(0.001, 0.015, n_obs))
    
    # Policy uncertainty
    data['USEPUINDXD'] = np.maximum(50 + 100 * np.random.gamma(2, 1, n_obs), 10)
    
    logger.info(f"Generated sample data with {len(data.columns)} indicators")
    
    return data