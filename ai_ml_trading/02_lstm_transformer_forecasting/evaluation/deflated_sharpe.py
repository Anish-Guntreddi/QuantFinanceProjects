"""
Deflated Sharpe Ratio and Related Statistical Tests

This module implements the deflated Sharpe ratio methodology from
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
by Marcos López de Prado and David Bailey.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy.stats import norm, t, chi2
from scipy.special import gamma
from scipy.optimize import minimize_scalar
import warnings


class DeflatedSharpe:
    """
    Implementation of deflated Sharpe ratio methodology.
    
    The deflated Sharpe ratio adjusts for multiple testing bias that occurs
    when testing many strategies and selecting the best performing one.
    """
    
    @staticmethod
    def calculate_deflated_sharpe(observed_sharpe: float,
                                n_trials: int,
                                n_observations: int,
                                skewness: float = 0.0,
                                kurtosis: float = 3.0,
                                frequency: str = 'daily') -> Dict[str, float]:
        """
        Calculate deflated Sharpe ratio.
        
        Args:
            observed_sharpe: Observed Sharpe ratio
            n_trials: Number of strategies tested
            n_observations: Number of observations
            skewness: Skewness of returns
            kurtosis: Kurtosis of returns
            frequency: Data frequency ('daily', 'monthly', 'annual')
            
        Returns:
            Dictionary containing deflated Sharpe statistics
        """
        # Frequency adjustment
        freq_multiplier = DeflatedSharpe._get_frequency_multiplier(frequency)
        
        # Annualized Sharpe ratio
        annualized_sharpe = observed_sharpe * np.sqrt(freq_multiplier)
        
        # Expected maximum Sharpe ratio under null hypothesis
        expected_max_sharpe = DeflatedSharpe._expected_maximum_sharpe(n_trials)
        
        # Standard error of Sharpe ratio accounting for higher moments
        sharpe_std_error = DeflatedSharpe._sharpe_standard_error(
            observed_sharpe, n_observations, skewness, kurtosis
        )
        
        # Deflated Sharpe ratio
        deflated_sharpe = (annualized_sharpe - expected_max_sharpe) / sharpe_std_error
        
        # P-value (probability of observing this Sharpe by chance)
        p_value = 1 - norm.cdf(deflated_sharpe)
        
        # Confidence levels
        confidence_95 = deflated_sharpe > norm.ppf(0.95)
        confidence_99 = deflated_sharpe > norm.ppf(0.99)
        
        return {
            'deflated_sharpe_ratio': deflated_sharpe,
            'p_value': p_value,
            'expected_max_sharpe': expected_max_sharpe,
            'sharpe_std_error': sharpe_std_error,
            'annualized_sharpe': annualized_sharpe,
            'significant_95': confidence_95,
            'significant_99': confidence_99,
            'n_trials': n_trials,
            'n_observations': n_observations
        }
    
    @staticmethod
    def _expected_maximum_sharpe(n_trials: int) -> float:
        """Calculate expected maximum Sharpe ratio under null hypothesis."""
        euler_mascheroni = 0.5772156649015329
        
        if n_trials <= 1:
            return 0.0
        
        # Expected maximum of n independent standard normal variables
        # Using asymptotic approximation for large n
        ln_n = np.log(n_trials)
        expected_max = np.sqrt(2 * ln_n) - (ln_n + np.log(4 * np.pi)) / (4 * np.sqrt(2 * ln_n))
        
        # More accurate formula for smaller n
        if n_trials <= 100:
            expected_max = np.sqrt(2 * ln_n) - (euler_mascheroni + np.log(2 * np.pi)) / (2 * np.sqrt(2 * ln_n))
        
        return expected_max
    
    @staticmethod
    def _sharpe_standard_error(sharpe: float, n_obs: int, 
                              skewness: float = 0.0, kurtosis: float = 3.0) -> float:
        """Calculate standard error of Sharpe ratio accounting for higher moments."""
        if n_obs <= 1:
            return np.inf
        
        # Adjustment for skewness and kurtosis
        # Based on Lo (2002) and Bailey & López de Prado (2012)
        
        # Variance of Sharpe ratio
        var_sharpe = (1 + 0.5 * sharpe**2 - 
                     skewness * sharpe + 
                     (kurtosis - 3) / 4 * sharpe**2) / n_obs
        
        return np.sqrt(var_sharpe)
    
    @staticmethod
    def _get_frequency_multiplier(frequency: str) -> float:
        """Get frequency multiplier for annualization."""
        freq_map = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annual': 1
        }
        
        return freq_map.get(frequency.lower(), 252)


class MinimumTrackRecord:
    """
    Calculate minimum track record length for statistical significance.
    
    Determines how long a strategy needs to run to achieve statistical
    significance for its Sharpe ratio.
    """
    
    @staticmethod
    def calculate_minimum_track_record(target_sharpe: float,
                                     confidence_level: float = 0.95,
                                     n_trials: int = 1,
                                     skewness: float = 0.0,
                                     kurtosis: float = 3.0,
                                     frequency: str = 'daily') -> Dict[str, float]:
        """
        Calculate minimum track record length.
        
        Args:
            target_sharpe: Target Sharpe ratio to test
            confidence_level: Desired confidence level
            n_trials: Number of strategies tested
            skewness: Return skewness
            kurtosis: Return kurtosis  
            frequency: Data frequency
            
        Returns:
            Dictionary with minimum track record statistics
        """
        # Critical value for confidence level
        z_critical = norm.ppf(confidence_level)
        
        # Expected maximum Sharpe under null
        expected_max_sharpe = DeflatedSharpe._expected_maximum_sharpe(n_trials)
        
        # Frequency adjustment
        freq_multiplier = DeflatedSharpe._get_frequency_multiplier(frequency)
        annual_target_sharpe = target_sharpe * np.sqrt(freq_multiplier)
        
        # Solve for minimum n using iterative approach
        def objective(n_obs):
            sharpe_std_error = DeflatedSharpe._sharpe_standard_error(
                target_sharpe, n_obs, skewness, kurtosis
            )
            
            deflated_sharpe = (annual_target_sharpe - expected_max_sharpe) / sharpe_std_error
            return abs(deflated_sharpe - z_critical)
        
        # Find minimum n_observations
        result = minimize_scalar(objective, bounds=(10, 10000), method='bounded')
        
        min_n_obs = result.x if result.success else np.inf
        
        # Convert to time periods based on frequency
        if frequency.lower() == 'daily':
            min_years = min_n_obs / 252
            min_months = min_n_obs / 21
        elif frequency.lower() == 'monthly':
            min_years = min_n_obs / 12
            min_months = min_n_obs
        else:
            min_years = min_n_obs / freq_multiplier
            min_months = min_years * 12
        
        return {
            'min_observations': min_n_obs,
            'min_years': min_years,
            'min_months': min_months,
            'target_sharpe': target_sharpe,
            'annual_target_sharpe': annual_target_sharpe,
            'confidence_level': confidence_level,
            'expected_max_sharpe': expected_max_sharpe
        }


class ProbabilisticSharpe:
    """
    Probabilistic Sharpe Ratio (PSR) calculations.
    
    PSR estimates the probability that the Sharpe ratio exceeds a benchmark.
    """
    
    @staticmethod
    def calculate_psr(observed_sharpe: float,
                     benchmark_sharpe: float,
                     n_observations: int,
                     skewness: float = 0.0,
                     kurtosis: float = 3.0) -> Dict[str, float]:
        """
        Calculate Probabilistic Sharpe Ratio.
        
        Args:
            observed_sharpe: Observed Sharpe ratio
            benchmark_sharpe: Benchmark Sharpe ratio
            n_observations: Number of observations
            skewness: Return skewness
            kurtosis: Return kurtosis
            
        Returns:
            Dictionary with PSR statistics
        """
        # Standard error of Sharpe ratio
        sharpe_std_error = DeflatedSharpe._sharpe_standard_error(
            observed_sharpe, n_observations, skewness, kurtosis
        )
        
        # PSR calculation
        if sharpe_std_error == 0:
            psr = 1.0 if observed_sharpe > benchmark_sharpe else 0.0
        else:
            z_score = (observed_sharpe - benchmark_sharpe) / sharpe_std_error
            psr = norm.cdf(z_score)
        
        # Confidence intervals for Sharpe ratio
        alpha = 0.05  # 95% confidence
        z_critical = norm.ppf(1 - alpha/2)
        
        sharpe_ci_lower = observed_sharpe - z_critical * sharpe_std_error
        sharpe_ci_upper = observed_sharpe + z_critical * sharpe_std_error
        
        return {
            'psr': psr,
            'observed_sharpe': observed_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'sharpe_std_error': sharpe_std_error,
            'sharpe_ci_lower': sharpe_ci_lower,
            'sharpe_ci_upper': sharpe_ci_upper,
            'n_observations': n_observations
        }
    
    @staticmethod
    def psr_time_series(returns: pd.Series,
                       benchmark_sharpe: float = 0.0,
                       window: int = 252) -> pd.Series:
        """
        Calculate rolling PSR over time.
        
        Args:
            returns: Return series
            benchmark_sharpe: Benchmark Sharpe ratio
            window: Rolling window size
            
        Returns:
            Series of PSR values
        """
        psr_values = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            
            # Calculate Sharpe ratio for window
            if window_returns.std() == 0:
                sharpe = 0
            else:
                sharpe = window_returns.mean() / window_returns.std()
            
            # Calculate higher moments
            skew = window_returns.skew()
            kurt = window_returns.kurtosis()
            
            # Calculate PSR
            psr_result = ProbabilisticSharpe.calculate_psr(
                sharpe, benchmark_sharpe, window, skew, kurt
            )
            
            psr_values.append(psr_result['psr'])
        
        # Create series with appropriate index
        psr_series = pd.Series(psr_values, index=returns.index[window-1:])
        
        return psr_series


class SharpeRatioStatistics:
    """
    Additional statistical tests and measures for Sharpe ratios.
    """
    
    @staticmethod
    def sharpe_ratio_difference_test(sharpe1: float, sharpe2: float,
                                   n_obs1: int, n_obs2: int,
                                   correlation: float = 0.0) -> Dict[str, float]:
        """
        Test statistical significance of difference between two Sharpe ratios.
        
        Args:
            sharpe1: First Sharpe ratio
            sharpe2: Second Sharpe ratio
            n_obs1: Number of observations for first strategy
            n_obs2: Number of observations for second strategy
            correlation: Correlation between the two strategies
            
        Returns:
            Dictionary with test statistics
        """
        # Standard errors
        se1 = np.sqrt((1 + 0.5 * sharpe1**2) / n_obs1)
        se2 = np.sqrt((1 + 0.5 * sharpe2**2) / n_obs2)
        
        # Standard error of difference
        se_diff = np.sqrt(se1**2 + se2**2 - 2 * correlation * se1 * se2)
        
        # Test statistic
        if se_diff == 0:
            z_stat = np.inf if sharpe1 != sharpe2 else 0
            p_value = 0 if sharpe1 != sharpe2 else 1
        else:
            z_stat = (sharpe1 - sharpe2) / se_diff
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))  # Two-tailed test
        
        return {
            'sharpe_difference': sharpe1 - sharpe2,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant_95': p_value < 0.05,
            'significant_99': p_value < 0.01,
            'se_difference': se_diff
        }
    
    @staticmethod
    def bootstrap_sharpe_confidence_interval(returns: np.ndarray,
                                           n_bootstrap: int = 1000,
                                           confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for Sharpe ratio.
        
        Args:
            returns: Return array
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Dictionary with confidence interval bounds
        """
        returns = np.array(returns)
        n_obs = len(returns)
        
        # Original Sharpe ratio
        original_sharpe = np.mean(returns) / np.std(returns, ddof=1) if np.std(returns, ddof=1) > 0 else 0
        
        # Bootstrap samples
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(returns, size=n_obs, replace=True)
            
            # Calculate Sharpe ratio
            if np.std(bootstrap_sample, ddof=1) > 0:
                bootstrap_sharpe = np.mean(bootstrap_sample) / np.std(bootstrap_sample, ddof=1)
            else:
                bootstrap_sharpe = 0
            
            bootstrap_sharpes.append(bootstrap_sharpe)
        
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_sharpes, lower_percentile)
        ci_upper = np.percentile(bootstrap_sharpes, upper_percentile)
        
        return {
            'original_sharpe': original_sharpe,
            'bootstrap_mean': np.mean(bootstrap_sharpes),
            'bootstrap_std': np.std(bootstrap_sharpes),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap
        }
    
    @staticmethod
    def haircut_sharpe_ratio(observed_sharpe: float,
                           n_observations: int,
                           confidence_level: float = 0.95) -> float:
        """
        Calculate haircut Sharpe ratio (lower confidence bound).
        
        Args:
            observed_sharpe: Observed Sharpe ratio
            n_observations: Number of observations
            confidence_level: Confidence level for haircut
            
        Returns:
            Haircut Sharpe ratio
        """
        # Standard error
        se = np.sqrt((1 + 0.5 * observed_sharpe**2) / n_observations)
        
        # Critical value
        z_critical = norm.ppf(confidence_level)
        
        # Haircut (lower confidence bound)
        haircut_sharpe = observed_sharpe - z_critical * se
        
        return haircut_sharpe
    
    @staticmethod
    def optimized_sharpe_threshold(n_trials: int,
                                  n_observations: int,
                                  significance_level: float = 0.05) -> float:
        """
        Calculate minimum Sharpe ratio threshold for significance after multiple testing.
        
        Args:
            n_trials: Number of strategies tested
            n_observations: Number of observations per strategy
            significance_level: Desired significance level
            
        Returns:
            Minimum Sharpe ratio threshold
        """
        # Bonferroni correction
        corrected_alpha = significance_level / n_trials
        
        # Critical value
        z_critical = norm.ppf(1 - corrected_alpha / 2)  # Two-tailed
        
        # Standard error for Sharpe ratio of 0
        se = np.sqrt(1 / n_observations)
        
        # Minimum threshold
        threshold = z_critical * se
        
        return threshold
    
    @staticmethod
    def multiple_testing_adjustment(p_values: List[float],
                                  method: str = 'bonferroni') -> Dict[str, List[float]]:
        """
        Apply multiple testing corrections to p-values.
        
        Args:
            p_values: List of p-values to adjust
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Dictionary with original and adjusted p-values
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            adjusted_p = np.minimum(p_values * n_tests, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            adjusted_p = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                adjustment_factor = n_tests - i
                adjusted_p[idx] = min(p_values[idx] * adjustment_factor, 1.0)
                
                # Ensure monotonicity
                if i > 0:
                    prev_idx = sorted_indices[i-1]
                    adjusted_p[idx] = max(adjusted_p[idx], adjusted_p[prev_idx])
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg method
            sorted_indices = np.argsort(p_values)
            adjusted_p = np.zeros_like(p_values)
            
            for i in range(n_tests - 1, -1, -1):
                idx = sorted_indices[i]
                adjustment_factor = n_tests / (i + 1)
                adjusted_p[idx] = min(p_values[idx] * adjustment_factor, 1.0)
                
                # Ensure monotonicity
                if i < n_tests - 1:
                    next_idx = sorted_indices[i + 1]
                    adjusted_p[idx] = min(adjusted_p[idx], adjusted_p[next_idx])
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return {
            'original_p_values': p_values.tolist(),
            'adjusted_p_values': adjusted_p.tolist(),
            'method': method,
            'significant_original': (p_values < 0.05).tolist(),
            'significant_adjusted': (adjusted_p < 0.05).tolist()
        }