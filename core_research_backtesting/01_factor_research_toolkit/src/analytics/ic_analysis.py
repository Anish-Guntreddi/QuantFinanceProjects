"""Information Coefficient analysis for factor evaluation"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple


class ICAnalyzer:
    """Analyze Information Coefficient for factors"""
    
    def __init__(self):
        """Initialize IC analyzer"""
        self.results = {}
        self.ic_series = {}
        
    def calculate_ic(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.Series:
        """
        Calculate Information Coefficient
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values (index: dates, columns: assets)
        forward_returns : pd.DataFrame
            Forward returns (same structure as factor)
        method : str
            Correlation method ('spearman', 'pearson')
            
        Returns:
        --------
        pd.Series
            IC values for each date
        """
        ic_values = []
        
        for date in factor.index:
            if date in forward_returns.index:
                # Get cross-sectional data for this date
                factor_cs = factor.loc[date].dropna()
                returns_cs = forward_returns.loc[date].dropna()
                
                # Find common assets
                common = factor_cs.index.intersection(returns_cs.index)
                
                if len(common) > 2:  # Need at least 3 points for correlation
                    if method == 'spearman':
                        corr, _ = stats.spearmanr(
                            factor_cs[common],
                            returns_cs[common]
                        )
                    else:  # pearson
                        corr, _ = stats.pearsonr(
                            factor_cs[common],
                            returns_cs[common]
                        )
                    
                    ic_values.append(corr)
                else:
                    ic_values.append(np.nan)
            else:
                ic_values.append(np.nan)
        
        ic_series = pd.Series(ic_values, index=factor.index)
        self.ic_series[f"{method}_ic"] = ic_series
        
        return ic_series
    
    def calculate_ic_decay(
        self,
        factor: pd.DataFrame,
        returns: Dict[int, pd.DataFrame],
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        Calculate IC decay over multiple horizons
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values
        returns : Dict[int, pd.DataFrame]
            Forward returns for different periods
        method : str
            Correlation method
            
        Returns:
        --------
        pd.DataFrame
            IC values for each period
        """
        ic_results = {}
        
        for period, ret in returns.items():
            ic = self.calculate_ic(factor, ret, method)
            ic_results[f"{period}d"] = ic
        
        ic_decay = pd.DataFrame(ic_results)
        self.results['ic_decay'] = ic_decay
        
        return ic_decay
    
    def calculate_monthly_ic(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        Calculate rolling monthly IC
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values
        forward_returns : pd.DataFrame
            Forward returns
        method : str
            Correlation method
            
        Returns:
        --------
        pd.DataFrame
            Monthly IC statistics
        """
        # Calculate daily IC
        daily_ic = self.calculate_ic(factor, forward_returns, method)
        
        # Group by month
        daily_ic.index = pd.to_datetime(daily_ic.index)
        monthly_ic = daily_ic.groupby(pd.Grouper(freq='M')).agg([
            'mean', 'std', 'count',
            lambda x: x.mean() / x.std() if x.std() > 0 else 0  # IC Sharpe
        ])
        monthly_ic.columns = ['mean_ic', 'std_ic', 'count', 'ic_sharpe']
        
        self.results['monthly_ic'] = monthly_ic
        
        return monthly_ic
    
    def calculate_ic_statistics(
        self,
        ic_series: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate comprehensive IC statistics
        
        Parameters:
        -----------
        ic_series : Optional[pd.Series]
            IC time series (if None, use last calculated)
            
        Returns:
        --------
        Dict
            IC statistics
        """
        if ic_series is None:
            if self.ic_series:
                ic_series = list(self.ic_series.values())[0]
            else:
                raise ValueError("No IC series available")
        
        # Remove NaN values
        ic_clean = ic_series.dropna()
        
        if len(ic_clean) == 0:
            return {}
        
        # Calculate statistics
        stats_dict = {
            'mean_ic': ic_clean.mean(),
            'std_ic': ic_clean.std(),
            'ic_sharpe': ic_clean.mean() / ic_clean.std() if ic_clean.std() > 0 else 0,
            'ic_skew': ic_clean.skew(),
            'ic_kurtosis': ic_clean.kurtosis(),
            'positive_ic_pct': (ic_clean > 0).mean(),
            'max_ic': ic_clean.max(),
            'min_ic': ic_clean.min(),
            'ic_t_stat': ic_clean.mean() / (ic_clean.std() / np.sqrt(len(ic_clean))),
            'ic_p_value': 2 * (1 - stats.norm.cdf(abs(ic_clean.mean() / (ic_clean.std() / np.sqrt(len(ic_clean))))))
        }
        
        # Calculate rolling statistics
        if len(ic_clean) > 20:
            rolling_mean = ic_clean.rolling(window=20).mean()
            rolling_std = ic_clean.rolling(window=20).std()
            stats_dict['rolling_ic_sharpe'] = (rolling_mean / rolling_std).mean()
        
        self.results['ic_statistics'] = stats_dict
        
        return stats_dict
    
    def calculate_factor_autocorrelation(
        self,
        factor: pd.DataFrame,
        lags: list = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Calculate factor autocorrelation
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values
        lags : list
            Lag periods to calculate
            
        Returns:
        --------
        pd.DataFrame
            Autocorrelations
        """
        autocorr_results = {}
        
        for lag in lags:
            autocorr_values = []
            
            for col in factor.columns:
                if factor[col].notna().sum() > lag:
                    autocorr = factor[col].autocorr(lag=lag)
                    autocorr_values.append(autocorr)
            
            if autocorr_values:
                autocorr_results[f"lag_{lag}"] = np.mean(autocorr_values)
        
        return pd.DataFrame(autocorr_results, index=['autocorrelation'])
    
    def calculate_quantile_returns(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        Calculate returns by factor quantile
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values
        forward_returns : pd.DataFrame
            Forward returns
        n_quantiles : int
            Number of quantiles
            
        Returns:
        --------
        pd.DataFrame
            Average returns by quantile
        """
        quantile_returns = []
        
        for date in factor.index:
            if date in forward_returns.index:
                # Get cross-sectional data
                factor_cs = factor.loc[date].dropna()
                returns_cs = forward_returns.loc[date].dropna()
                
                # Find common assets
                common = factor_cs.index.intersection(returns_cs.index)
                
                if len(common) > n_quantiles:
                    # Assign quantiles
                    quantiles = pd.qcut(factor_cs[common], n_quantiles, labels=False)
                    
                    # Calculate average return per quantile
                    for q in range(n_quantiles):
                        q_assets = quantiles[quantiles == q].index
                        q_return = returns_cs[q_assets].mean()
                        quantile_returns.append({
                            'date': date,
                            'quantile': q + 1,
                            'return': q_return
                        })
        
        quantile_df = pd.DataFrame(quantile_returns)
        
        if not quantile_df.empty:
            # Aggregate by quantile
            quantile_summary = quantile_df.groupby('quantile')['return'].agg([
                'mean', 'std', 'count'
            ])
            quantile_summary['sharpe'] = quantile_summary['mean'] / quantile_summary['std']
            
            self.results['quantile_returns'] = quantile_summary
            
            return quantile_summary
        
        return pd.DataFrame()