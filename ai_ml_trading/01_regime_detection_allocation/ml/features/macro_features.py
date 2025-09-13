"""
Macroeconomic feature extraction for regime detection.

This module provides functionality to extract and process macroeconomic indicators
from various data sources including FRED and Yahoo Finance.
"""

import numpy as np
import pandas as pd
from fredapi import Fred
import yfinance as yf
from typing import List, Optional, Dict, Union
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


class MacroFeatureExtractor:
    """Extract macroeconomic features for regime detection"""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize macro feature extractor
        
        Parameters:
        -----------
        fred_api_key : Optional[str]
            FRED API key for accessing Federal Reserve data
        """
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.feature_cache = {}
        self.data_sources = {
            'fred': self.fred,
            'yahoo': 'yahoo_finance'
        }
        
    def extract_features(
        self,
        start_date: str,
        end_date: str,
        features: Optional[List[str]] = None,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Extract macro features for given date range
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        features : Optional[List[str]]
            List of features to extract (None for default features)
        frequency : str
            Data frequency ('daily', 'weekly', 'monthly')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted macro features
        """
        
        if features is None:
            features = self.get_default_features()
            
        print(f"Extracting {len(features)} macro features...")
        
        df_features = pd.DataFrame()
        
        for feature in features:
            try:
                print(f"  Fetching {feature}...")
                
                if feature in self.feature_cache:
                    data = self.feature_cache[feature]
                    # Filter by date range
                    data = data.loc[start_date:end_date]
                else:
                    data = self._fetch_feature(feature, start_date, end_date)
                    if data is not None:
                        self.feature_cache[feature] = data
                        
                if data is not None and len(data) > 0:
                    df_features[feature] = data
                    
            except Exception as e:
                print(f"  Warning: Failed to fetch {feature}: {e}")
                continue
                
        if df_features.empty:
            print("Warning: No features were successfully extracted")
            return df_features
            
        # Resample to desired frequency if needed
        if frequency != 'daily':
            df_features = self._resample_data(df_features, frequency)
            
        # Calculate derived features
        print("  Calculating derived features...")
        df_features = self._calculate_derived_features(df_features)
        
        # Clean and fill missing data
        df_features = self._clean_data(df_features)
        
        print(f"Successfully extracted {len(df_features.columns)} features")
        
        return df_features
    
    def get_default_features(self) -> List[str]:
        """Get default macro features"""
        return [
            # Interest Rates
            'DGS10',      # 10-Year Treasury Rate
            'DGS2',       # 2-Year Treasury Rate
            'DGS3MO',     # 3-Month Treasury Rate
            'DFF',        # Federal Funds Rate
            
            # Exchange Rates
            'DEXUSEU',    # USD/EUR Exchange Rate
            'DEXJPUS',    # JPY/USD Exchange Rate
            'DEXCHUS',    # CNY/USD Exchange Rate
            
            # Economic Indicators
            'UNRATE',     # Unemployment Rate
            'CPIAUCSL',   # Consumer Price Index
            'INDPRO',     # Industrial Production Index
            'PAYEMS',     # Nonfarm Payrolls
            
            # Financial Markets
            'VIXCLS',     # VIX Volatility Index
            'DCOILWTICO', # WTI Crude Oil
            'GOLDAMGBD228NLBM',  # Gold Price
            
            # Money Supply
            'M2SL',       # M2 Money Supply
            'BOGMBASE',   # Monetary Base
            
            # Credit Markets
            'BAMLH0A0HYM2',  # High Yield Spread
            'T10Y2Y',     # Term Spread (10Y-2Y)
            'TEDRATE',    # TED Spread
            
            # Economic Growth
            'GDP',        # Gross Domestic Product
            'GDPC1',      # Real GDP
            
            # Consumer Sentiment
            'UMCSENT',    # University of Michigan Consumer Sentiment
            
            # Market Indices (via Yahoo Finance)
            'SPY',        # S&P 500 ETF
            'TLT',        # 20+ Year Treasury Bond ETF
            'GLD',        # Gold ETF
            'UUP',        # Dollar Index ETF
        ]
    
    def _fetch_feature(self, feature: str, start_date: str, end_date: str) -> Optional[pd.Series]:
        """Fetch feature from appropriate data source"""
        
        try:
            # Try FRED first for most features
            if self.fred and feature not in ['SPY', 'TLT', 'GLD', 'UUP', 'VIX']:
                try:
                    data = self.fred.get_series(feature, start_date, end_date)
                    return data
                except Exception as fred_error:
                    print(f"    FRED failed for {feature}: {fred_error}")
            
            # Try Yahoo Finance for market data
            yahoo_tickers = {
                'SPY': 'SPY',
                'TLT': 'TLT',
                'GLD': 'GLD',
                'UUP': 'UUP',
                'VIX': '^VIX',
                'VIXCLS': '^VIX',
                'DCOILWTICO': 'CL=F',
                'GOLDAMGBD228NLBM': 'GC=F'
            }
            
            if feature in yahoo_tickers:
                ticker = yf.Ticker(yahoo_tickers[feature])
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Use Close price for most securities
                    return hist['Close']
                    
        except Exception as e:
            print(f"    Error fetching {feature}: {e}")
            
        return None
    
    def _resample_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Resample data to specified frequency"""
        
        if frequency == 'weekly':
            return df.resample('W').last()
        elif frequency == 'monthly':
            return df.resample('M').last()
        else:
            return df
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived macro features"""
        
        df = df.copy()
        
        # Term structure features
        if 'DGS10' in df.columns and 'DGS2' in df.columns:
            df['term_spread'] = df['DGS10'] - df['DGS2']
            df['term_spread_ma'] = df['term_spread'].rolling(20, min_periods=10).mean()
            df['term_spread_std'] = df['term_spread'].rolling(60, min_periods=30).std()
            df['term_spread_percentile'] = df['term_spread'].rolling(252, min_periods=100).rank(pct=True)
            
        if 'DGS10' in df.columns and 'DGS3MO' in df.columns:
            df['yield_curve_slope'] = df['DGS10'] - df['DGS3MO']
            
        # Real interest rates
        if 'DGS10' in df.columns and 'CPIAUCSL' in df.columns:
            df['cpi_yoy'] = df['CPIAUCSL'].pct_change(252)
            df['real_rate_10y'] = df['DGS10'] - df['cpi_yoy'] * 100
            
        if 'DFF' in df.columns and 'CPIAUCSL' in df.columns:
            df['real_fed_funds'] = df['DFF'] - df.get('cpi_yoy', 0) * 100
            
        # Credit and risk spreads
        if 'BAMLH0A0HYM2' in df.columns:
            df['credit_spread_chg'] = df['BAMLH0A0HYM2'].diff()
            df['credit_spread_ma'] = df['BAMLH0A0HYM2'].rolling(20, min_periods=10).mean()
            df['credit_spread_z'] = (
                df['BAMLH0A0HYM2'] - df['BAMLH0A0HYM2'].rolling(252, min_periods=100).mean()
            ) / df['BAMLH0A0HYM2'].rolling(252, min_periods=100).std()
            
        # Dollar strength and momentum
        if 'DEXUSEU' in df.columns:
            df['dollar_eur_momentum'] = df['DEXUSEU'].pct_change(20)
            df['dollar_eur_volatility'] = df['DEXUSEU'].pct_change().rolling(20, min_periods=10).std()
            
        # Economic momentum indicators
        if 'INDPRO' in df.columns:
            df['indpro_momentum'] = df['INDPRO'].pct_change(3)
            df['indpro_acceleration'] = df['indpro_momentum'].diff()
            df['indpro_yoy'] = df['INDPRO'].pct_change(12)
            
        if 'PAYEMS' in df.columns:
            df['payrolls_momentum'] = df['PAYEMS'].diff()
            df['payrolls_ma'] = df['PAYEMS'].diff().rolling(6, min_periods=3).mean()
            
        # Unemployment rate features
        if 'UNRATE' in df.columns:
            df['unemployment_chg'] = df['UNRATE'].diff()
            df['unemployment_trend'] = df['UNRATE'].rolling(12, min_periods=6).mean()
            df['unemployment_acceleration'] = df['unemployment_chg'].diff()
            
        # Volatility regime indicators
        if 'VIXCLS' in df.columns:
            df['vix_percentile'] = df['VIXCLS'].rolling(252, min_periods=100).rank(pct=True)
            df['vix_change'] = df['VIXCLS'].pct_change(5)
            df['vix_ma'] = df['VIXCLS'].rolling(20, min_periods=10).mean()
            df['vix_regime'] = np.where(df['VIXCLS'] > 25, 1, 0)  # High vol regime
            
        # Commodity features
        if 'DCOILWTICO' in df.columns:
            df['oil_momentum'] = df['DCOILWTICO'].pct_change(20)
            df['oil_volatility'] = df['DCOILWTICO'].pct_change().rolling(20, min_periods=10).std()
            
        if 'GOLDAMGBD228NLBM' in df.columns:
            df['gold_momentum'] = df['GOLDAMGBD228NLBM'].pct_change(20)
            df['gold_real'] = df['GOLDAMGBD228NLBM'] / (1 + df.get('cpi_yoy', 0))
            
        # Combined commodity momentum
        if 'DCOILWTICO' in df.columns and 'GOLDAMGBD228NLBM' in df.columns:
            df['commodity_momentum'] = (
                df['oil_momentum'].fillna(0) + df['gold_momentum'].fillna(0)
            ) / 2
            
        # Monetary conditions
        if 'M2SL' in df.columns:
            df['m2_growth'] = df['M2SL'].pct_change(252)
            df['m2_acceleration'] = df['M2SL'].pct_change(252).diff()
            
        # GDP growth (when available)
        if 'GDPC1' in df.columns:
            df['gdp_growth'] = df['GDPC1'].pct_change(4)  # YoY growth
            
        # Consumer sentiment features
        if 'UMCSENT' in df.columns:
            df['consumer_sentiment_chg'] = df['UMCSENT'].diff()
            df['consumer_sentiment_ma'] = df['UMCSENT'].rolling(6, min_periods=3).mean()
            df['consumer_sentiment_z'] = (
                df['UMCSENT'] - df['UMCSENT'].rolling(252, min_periods=100).mean()
            ) / df['UMCSENT'].rolling(252, min_periods=100).std()
            
        # Market-based indicators
        if 'SPY' in df.columns:
            df['spy_returns'] = df['SPY'].pct_change()
            df['spy_volatility'] = df['spy_returns'].rolling(20, min_periods=10).std()
            df['spy_momentum'] = df['SPY'].pct_change(20)
            
        # Risk parity indicators
        risk_assets = ['SPY']
        safe_assets = ['TLT', 'GLD']
        
        risk_cols = [col for col in risk_assets if col in df.columns]
        safe_cols = [col for col in safe_assets if col in df.columns]
        
        if risk_cols and safe_cols:
            df['risk_on_momentum'] = df[risk_cols].pct_change(20).mean(axis=1)
            df['risk_off_momentum'] = df[safe_cols].pct_change(20).mean(axis=1)
            df['risk_on_off_spread'] = df['risk_on_momentum'] - df['risk_off_momentum']
            
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess macro data"""
        
        # Forward fill for variables that change infrequently
        monthly_vars = ['UNRATE', 'CPIAUCSL', 'INDPRO', 'PAYEMS', 'M2SL', 'GDP', 'GDPC1']
        for var in monthly_vars:
            if var in df.columns:
                df[var] = df[var].fillna(method='ffill')
                
        # Interpolate for daily series with gaps
        daily_vars = ['DGS10', 'DGS2', 'DFF', 'VIXCLS', 'SPY', 'TLT', 'GLD']
        for var in daily_vars:
            if var in df.columns:
                df[var] = df[var].interpolate(method='linear', limit=5)
                
        # Remove features with too many missing values
        missing_threshold = 0.5
        for col in df.columns:
            if df[col].isnull().sum() / len(df) > missing_threshold:
                print(f"  Warning: Removing {col} due to {df[col].isnull().sum()/len(df):.1%} missing values")
                df = df.drop(columns=[col])
                
        # Final forward fill and backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available features"""
        
        descriptions = {
            # Interest Rates
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'DGS2': '2-Year Treasury Constant Maturity Rate', 
            'DGS3MO': '3-Month Treasury Constant Maturity Rate',
            'DFF': 'Federal Funds Effective Rate',
            
            # Exchange Rates
            'DEXUSEU': 'US / Euro Foreign Exchange Rate',
            'DEXJPUS': 'Japan / US Foreign Exchange Rate',
            'DEXCHUS': 'China / US Foreign Exchange Rate',
            
            # Economic Indicators
            'UNRATE': 'Unemployment Rate',
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
            'INDPRO': 'Industrial Production Index',
            'PAYEMS': 'All Employees, Total Nonfarm Payrolls',
            
            # Financial Markets
            'VIXCLS': 'CBOE Volatility Index: VIX',
            'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate',
            'GOLDAMGBD228NLBM': 'Gold Fixing Price',
            
            # Money Supply
            'M2SL': 'M2 Money Supply',
            'BOGMBASE': 'St. Louis Adjusted Monetary Base',
            
            # Credit Markets
            'BAMLH0A0HYM2': 'ICE BofA US High Yield Index Option-Adjusted Spread',
            'T10Y2Y': '10-Year Treasury Constant Maturity Minus 2-Year',
            'TEDRATE': 'TED Spread',
            
            # Economic Growth
            'GDP': 'Gross Domestic Product',
            'GDPC1': 'Real Gross Domestic Product',
            
            # Consumer Sentiment
            'UMCSENT': 'University of Michigan: Consumer Sentiment',
            
            # Market ETFs
            'SPY': 'SPDR S&P 500 ETF Trust',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'GLD': 'SPDR Gold Trust',
            'UUP': 'Invesco DB US Dollar Index Bullish Fund'
        }
        
        return descriptions
    
    def calculate_feature_importance(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series,
        method: str = 'correlation'
    ) -> pd.Series:
        """
        Calculate feature importance for regime prediction
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix
        target : pd.Series
            Target variable (regime labels)
        method : str
            Method for calculating importance ('correlation', 'mutual_info')
            
        Returns:
        --------
        pd.Series
            Feature importance scores
        """
        
        # Align data
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]
        
        if method == 'correlation':
            # Calculate correlation with target
            importance = features_aligned.corrwith(target_aligned).abs()
            
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_aligned.fillna(0))
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X_scaled, target_aligned)
            importance = pd.Series(mi_scores, index=features_aligned.columns)
            
        else:
            raise ValueError(f"Unknown importance method: {method}")
            
        return importance.sort_values(ascending=False)
    
    def get_regime_sensitive_features(
        self, 
        features_df: pd.DataFrame, 
        regime_labels: pd.Series,
        top_k: int = 10
    ) -> List[str]:
        """
        Identify features most sensitive to regime changes
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix
        regime_labels : pd.Series
            Regime labels
        top_k : int
            Number of top features to return
            
        Returns:
        --------
        List[str]
            List of most regime-sensitive feature names
        """
        
        # Calculate feature importance
        importance = self.calculate_feature_importance(features_df, regime_labels)
        
        # Return top features
        return importance.head(top_k).index.tolist()
    
    def create_macro_report(self, features_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Generate macro features summary report"""
        
        report = "MACROECONOMIC FEATURES REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Date Range: {features_df.index[0]} to {features_df.index[-1]}\n"
        report += f"Total Features: {len(features_df.columns)}\n"
        report += f"Total Observations: {len(features_df)}\n\n"
        
        report += "FEATURE CATEGORIES:\n"
        report += "-" * 20 + "\n"
        
        categories = {
            'Interest Rates': ['DGS10', 'DGS2', 'DGS3MO', 'DFF', 'term_spread', 'real_rate'],
            'Exchange Rates': ['DEXUSEU', 'DEXJPUS', 'dollar_eur_momentum'],
            'Economic Growth': ['INDPRO', 'PAYEMS', 'GDP', 'GDPC1', 'indpro_momentum'],
            'Inflation': ['CPIAUCSL', 'cpi_yoy'],
            'Labor Market': ['UNRATE', 'unemployment_chg'],
            'Credit Markets': ['BAMLH0A0HYM2', 'TEDRATE', 'credit_spread'],
            'Volatility': ['VIXCLS', 'vix_percentile', 'vix_regime'],
            'Commodities': ['DCOILWTICO', 'GOLDAMGBD228NLBM', 'oil_momentum', 'gold_momentum'],
            'Money Supply': ['M2SL', 'm2_growth'],
            'Market Indicators': ['SPY', 'TLT', 'GLD', 'spy_returns', 'spy_volatility']
        }
        
        for category, feature_list in categories.items():
            available_features = [f for f in feature_list if f in features_df.columns]
            if available_features:
                report += f"{category}: {len(available_features)} features\n"
                
        # Data quality summary
        report += "\nDATA QUALITY:\n"
        report += "-" * 15 + "\n"
        
        missing_summary = features_df.isnull().sum() / len(features_df)
        high_missing = missing_summary[missing_summary > 0.1]
        
        if len(high_missing) > 0:
            report += f"Features with >10% missing data: {len(high_missing)}\n"
            for feature, pct in high_missing.head().items():
                report += f"  {feature}: {pct:.1%} missing\n"
        else:
            report += "All features have <10% missing data\n"
            
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report