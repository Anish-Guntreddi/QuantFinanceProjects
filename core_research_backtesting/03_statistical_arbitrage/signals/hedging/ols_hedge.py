"""
OLS Hedge Ratio Implementation

Ordinary Least Squares hedge ratio estimation for pairs trading.
Provides static hedge ratios that minimize the variance of the spread.

Mathematical Foundation:
For assets X and Y, the hedge ratio β minimizes:
min E[(Y - βX)²]

The solution is: β = Cov(X,Y) / Var(X)

Extensions include:
- Rolling OLS for time-varying ratios
- Multi-asset hedge ratio optimization
- Ridge regression for regularization
- Robust regression for outlier resistance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.metrics import r2_score
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')


class OLSHedgeRatio:
    """Calculate hedge ratios using Ordinary Least Squares regression"""
    
    def __init__(self):
        """Initialize OLS hedge ratio calculator"""
        self.history = []
        self.current_model = None
        self.current_params = {}
        
    def calculate_hedge_ratio(
        self,
        dependent: pd.Series,
        independent: Union[pd.Series, pd.DataFrame],
        method: str = 'standard',
        include_intercept: bool = True,
        **kwargs
    ) -> Dict:
        """
        Calculate hedge ratio using OLS
        
        Args:
            dependent: Dependent variable (Y)
            independent: Independent variable(s) (X)
            method: 'standard', 'ridge', 'robust'
            include_intercept: Whether to include intercept term
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with hedge ratios and statistics
        """
        
        # Prepare data
        if isinstance(independent, pd.Series):
            independent = pd.DataFrame({independent.name or 'X': independent})
        
        # Align data
        data = pd.concat([dependent.rename('Y'), independent], axis=1).dropna()
        
        if len(data) < 10:
            raise ValueError("Need at least 10 observations for hedge ratio calculation")
        
        Y = data['Y'].values
        X = data.drop('Y', axis=1).values
        
        if method == 'standard':
            result = self._standard_ols(Y, X, data.drop('Y', axis=1).columns, include_intercept)
        elif method == 'ridge':
            result = self._ridge_ols(Y, X, data.drop('Y', axis=1).columns, include_intercept, **kwargs)
        elif method == 'robust':
            result = self._robust_ols(Y, X, data.drop('Y', axis=1).columns, include_intercept, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add common statistics
        result.update({
            'method': method,
            'n_observations': len(data),
            'dependent_var': dependent.name,
            'independent_vars': list(data.drop('Y', axis=1).columns),
            'include_intercept': include_intercept,
            'calculation_date': pd.Timestamp.now()
        })
        
        # Store results
        self.current_params = result
        self.history.append(result)
        
        return result
    
    def _standard_ols(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        include_intercept: bool
    ) -> Dict:
        """Standard OLS estimation"""
        
        model = LinearRegression(fit_intercept=include_intercept)
        model.fit(X, Y)
        
        # Predictions and residuals
        Y_pred = model.predict(X)
        residuals = Y - Y_pred
        
        # Standard errors and t-statistics
        n, k = X.shape
        dof = n - k - (1 if include_intercept else 0)
        
        mse = np.sum(residuals**2) / dof
        
        # Design matrix for standard errors
        if include_intercept:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X.copy()
        
        try:
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
            var_coeff = mse * np.diag(XtX_inv)
            std_errors = np.sqrt(var_coeff)
            
            # t-statistics
            if include_intercept:
                coeff_full = np.concatenate([[model.intercept_], model.coef_])
            else:
                coeff_full = model.coef_
            
            t_stats = coeff_full / std_errors
            p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), dof))
            
        except np.linalg.LinAlgError:
            std_errors = np.full(len(model.coef_), np.nan)
            t_stats = np.full(len(model.coef_), np.nan)
            p_values = np.full(len(model.coef_), np.nan)
        
        # Hedge ratios
        hedge_ratios = pd.Series(model.coef_, index=feature_names)
        
        # Confidence intervals (95%)
        critical_value = t_dist.ppf(0.975, dof)
        if include_intercept:
            conf_intervals = {
                'intercept': [
                    model.intercept_ - critical_value * std_errors[0],
                    model.intercept_ + critical_value * std_errors[0]
                ]
            }
            hedge_errors = std_errors[1:]
            hedge_t_stats = t_stats[1:]
            hedge_p_values = p_values[1:]
        else:
            conf_intervals = {}
            hedge_errors = std_errors
            hedge_t_stats = t_stats
            hedge_p_values = p_values
        
        for i, name in enumerate(feature_names):
            conf_intervals[name] = [
                hedge_ratios[name] - critical_value * hedge_errors[i],
                hedge_ratios[name] + critical_value * hedge_errors[i]
            ]
        
        return {
            'hedge_ratios': hedge_ratios,
            'intercept': model.intercept_ if include_intercept else 0,
            'r_squared': r2_score(Y, Y_pred),
            'adjusted_r_squared': 1 - (1 - r2_score(Y, Y_pred)) * (n - 1) / dof,
            'residuals': pd.Series(residuals, name='residuals'),
            'fitted_values': pd.Series(Y_pred, name='fitted'),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'standard_errors': pd.Series(hedge_errors, index=feature_names),
            't_statistics': pd.Series(hedge_t_stats, index=feature_names),
            'p_values': pd.Series(hedge_p_values, index=feature_names),
            'confidence_intervals': conf_intervals,
            'f_statistic': self._calculate_f_statistic(Y, Y_pred, k, dof),
            'durbin_watson': self._durbin_watson(residuals),
            'model': model
        }
    
    def _ridge_ols(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        include_intercept: bool,
        alpha: float = 1.0,
        **kwargs
    ) -> Dict:
        """Ridge regression for regularized hedge ratios"""
        
        model = Ridge(alpha=alpha, fit_intercept=include_intercept)
        model.fit(X, Y)
        
        Y_pred = model.predict(X)
        residuals = Y - Y_pred
        
        hedge_ratios = pd.Series(model.coef_, index=feature_names)
        
        return {
            'hedge_ratios': hedge_ratios,
            'intercept': model.intercept_ if include_intercept else 0,
            'r_squared': r2_score(Y, Y_pred),
            'residuals': pd.Series(residuals, name='residuals'),
            'fitted_values': pd.Series(Y_pred, name='fitted'),
            'mse': np.mean(residuals**2),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'regularization_alpha': alpha,
            'model': model
        }
    
    def _robust_ols(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        include_intercept: bool,
        epsilon: float = 1.35,
        max_iter: int = 100,
        **kwargs
    ) -> Dict:
        """Robust regression using Huber loss"""
        
        model = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            fit_intercept=include_intercept
        )
        model.fit(X, Y)
        
        Y_pred = model.predict(X)
        residuals = Y - Y_pred
        
        hedge_ratios = pd.Series(model.coef_, index=feature_names)
        
        # Robust scale estimate
        robust_scale = model.scale_ if hasattr(model, 'scale_') else np.std(residuals)
        
        return {
            'hedge_ratios': hedge_ratios,
            'intercept': model.intercept_ if include_intercept else 0,
            'r_squared': r2_score(Y, Y_pred),
            'residuals': pd.Series(residuals, name='residuals'),
            'fitted_values': pd.Series(Y_pred, name='fitted'),
            'mse': np.mean(residuals**2),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'robust_scale': robust_scale,
            'epsilon': epsilon,
            'n_iterations': model.n_iter_ if hasattr(model, 'n_iter_') else None,
            'outliers_mask': np.abs(residuals) > epsilon * robust_scale,
            'model': model
        }
    
    def _calculate_f_statistic(
        self,
        Y: np.ndarray,
        Y_pred: np.ndarray,
        k: int,
        dof: int
    ) -> Dict:
        """Calculate F-statistic for overall model significance"""
        
        ss_res = np.sum((Y - Y_pred)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        
        if ss_tot == 0:
            return {'f_stat': np.nan, 'f_pvalue': np.nan}
        
        mse_model = (ss_tot - ss_res) / k
        mse_residual = ss_res / dof
        
        if mse_residual == 0:
            f_stat = np.inf
            f_pvalue = 0.0
        else:
            f_stat = mse_model / mse_residual
            from scipy.stats import f as f_dist
            f_pvalue = 1 - f_dist.cdf(f_stat, k, dof)
        
        return {'f_stat': f_stat, 'f_pvalue': f_pvalue}
    
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation"""
        
        if len(residuals) < 2:
            return np.nan
        
        diff_residuals = np.diff(residuals)
        return np.sum(diff_residuals**2) / np.sum(residuals**2)
    
    def rolling_hedge_ratio(
        self,
        dependent: pd.Series,
        independent: Union[pd.Series, pd.DataFrame],
        window: int = 60,
        min_periods: Optional[int] = None,
        method: str = 'standard',
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate rolling hedge ratios
        
        Args:
            dependent: Dependent variable time series
            independent: Independent variable(s) time series
            window: Rolling window size
            min_periods: Minimum observations needed
            method: OLS method to use
            
        Returns:
            DataFrame with rolling hedge ratios and statistics
        """
        
        if isinstance(independent, pd.Series):
            independent = pd.DataFrame({independent.name or 'X': independent})
        
        if min_periods is None:
            min_periods = max(10, window // 2)
        
        # Align data
        data = pd.concat([dependent.rename('Y'), independent], axis=1).dropna()
        
        results = []
        
        for i in range(len(data)):
            if i < min_periods - 1:
                continue
            
            start_idx = max(0, i - window + 1)
            window_data = data.iloc[start_idx:i+1]
            
            if len(window_data) < min_periods:
                continue
            
            try:
                window_dependent = window_data['Y']
                window_independent = window_data.drop('Y', axis=1)
                
                result = self.calculate_hedge_ratio(
                    window_dependent,
                    window_independent,
                    method=method,
                    **kwargs
                )
                
                # Extract key metrics
                row_data = {
                    'date': data.index[i],
                    'r_squared': result['r_squared'],
                    'mse': result['mse']
                }
                
                # Add hedge ratios
                for asset, ratio in result['hedge_ratios'].items():
                    row_data[f'hedge_ratio_{asset}'] = ratio
                
                if result.get('intercept', 0) != 0:
                    row_data['intercept'] = result['intercept']
                
                results.append(row_data)
                
            except Exception as e:
                # Skip problematic windows
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        
        return results_df
    
    def multi_asset_hedge_ratio(
        self,
        portfolio_weights: pd.Series,
        asset_prices: pd.DataFrame,
        hedge_assets: pd.DataFrame,
        method: str = 'standard',
        **kwargs
    ) -> Dict:
        """
        Calculate hedge ratios for a portfolio against hedge assets
        
        Args:
            portfolio_weights: Weights for portfolio construction
            asset_prices: Price data for portfolio assets
            hedge_assets: Price data for hedge assets (e.g., ETFs, futures)
            method: OLS method to use
            
        Returns:
            Dictionary with optimal hedge ratios
        """
        
        # Align data
        all_data = pd.concat([asset_prices, hedge_assets], axis=1).dropna()
        
        # Construct portfolio returns
        portfolio_assets = [col for col in portfolio_weights.index if col in all_data.columns]
        missing_assets = set(portfolio_weights.index) - set(portfolio_assets)
        
        if missing_assets:
            print(f"Warning: Missing price data for assets: {missing_assets}")
        
        # Calculate portfolio value
        portfolio_prices = (all_data[portfolio_assets] * portfolio_weights[portfolio_assets]).sum(axis=1)
        portfolio_returns = portfolio_prices.pct_change().dropna()
        
        # Calculate hedge asset returns
        hedge_returns = hedge_assets.pct_change().dropna()
        
        # Align returns
        aligned_data = pd.concat([portfolio_returns.rename('portfolio'), hedge_returns], axis=1).dropna()
        
        if len(aligned_data) < 10:
            raise ValueError("Insufficient aligned data for hedge ratio calculation")
        
        # Calculate hedge ratios
        result = self.calculate_hedge_ratio(
            aligned_data['portfolio'],
            aligned_data.drop('portfolio', axis=1),
            method=method,
            **kwargs
        )
        
        # Add portfolio-specific information
        result.update({
            'portfolio_weights': portfolio_weights,
            'portfolio_volatility': portfolio_returns.std() * np.sqrt(252),
            'hedge_effectiveness': self._calculate_hedge_effectiveness(
                aligned_data['portfolio'],
                aligned_data.drop('portfolio', axis=1),
                result['hedge_ratios']
            )
        })
        
        return result
    
    def _calculate_hedge_effectiveness(
        self,
        portfolio_returns: pd.Series,
        hedge_returns: pd.DataFrame,
        hedge_ratios: pd.Series
    ) -> Dict:
        """Calculate hedge effectiveness metrics"""
        
        # Unhedged portfolio variance
        unhedged_var = portfolio_returns.var()
        
        # Hedged portfolio
        hedge_portfolio = hedge_returns @ hedge_ratios
        hedged_returns = portfolio_returns - hedge_portfolio
        hedged_var = hedged_returns.var()
        
        # Hedge effectiveness
        effectiveness = (unhedged_var - hedged_var) / unhedged_var if unhedged_var > 0 else 0
        
        # Minimum variance hedge ratio (theoretical optimum)
        optimal_ratios = {}
        for hedge_asset in hedge_returns.columns:
            cov_ph = np.cov(portfolio_returns, hedge_returns[hedge_asset])[0, 1]
            var_h = hedge_returns[hedge_asset].var()
            optimal_ratios[hedge_asset] = cov_ph / var_h if var_h > 0 else 0
        
        return {
            'hedge_effectiveness': effectiveness,
            'variance_reduction': (unhedged_var - hedged_var) / unhedged_var * 100,
            'unhedged_volatility': np.sqrt(unhedged_var) * np.sqrt(252),
            'hedged_volatility': np.sqrt(hedged_var) * np.sqrt(252),
            'optimal_hedge_ratios': pd.Series(optimal_ratios),
            'hedge_ratio_efficiency': (hedge_ratios / pd.Series(optimal_ratios)).fillna(1.0)
        }
    
    def diagnostic_tests(
        self,
        result: Optional[Dict] = None
    ) -> Dict:
        """
        Run diagnostic tests on hedge ratio estimation
        
        Args:
            result: Hedge ratio estimation result (uses current if None)
            
        Returns:
            Dictionary with diagnostic test results
        """
        
        if result is None:
            result = self.current_params
        
        if not result or 'residuals' not in result:
            return {'error': 'No estimation results available'}
        
        residuals = result['residuals'].dropna()
        
        if len(residuals) < 10:
            return {'error': 'Insufficient residuals for testing'}
        
        diagnostics = {}
        
        # 1. Normality test (Jarque-Bera)
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(residuals)
        diagnostics['normality'] = {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        }
        
        # 2. Autocorrelation test (Ljung-Box)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
            diagnostics['autocorrelation'] = {
                'ljung_box_stat': lb_result['lb_stat'].iloc[-1],
                'ljung_box_pvalue': lb_result['lb_pvalue'].iloc[-1],
                'has_autocorrelation': lb_result['lb_pvalue'].iloc[-1] < 0.05
            }
        except Exception:
            diagnostics['autocorrelation'] = {'error': 'Ljung-Box test failed'}
        
        # 3. Heteroscedasticity test (Breusch-Pagan)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            if 'fitted_values' in result:
                fitted = result['fitted_values'].dropna()
                if len(fitted) == len(residuals):
                    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, fitted.values.reshape(-1, 1))
                    diagnostics['heteroscedasticity'] = {
                        'breusch_pagan_stat': bp_stat,
                        'breusch_pagan_pvalue': bp_pvalue,
                        'has_heteroscedasticity': bp_pvalue < 0.05
                    }
                else:
                    diagnostics['heteroscedasticity'] = {'error': 'Mismatched fitted values'}
            else:
                diagnostics['heteroscedasticity'] = {'error': 'No fitted values available'}
        except Exception:
            diagnostics['heteroscedasticity'] = {'error': 'Breusch-Pagan test failed'}
        
        # 4. Stability test (CUSUM)
        diagnostics['stability'] = self._cusum_test(residuals)
        
        return diagnostics
    
    def _cusum_test(self, residuals: pd.Series) -> Dict:
        """CUSUM test for parameter stability"""
        
        try:
            n = len(residuals)
            sigma = residuals.std()
            
            # Standardized residuals
            std_residuals = residuals / sigma
            
            # CUSUM statistic
            cusum = std_residuals.cumsum()
            
            # Critical values (approximate)
            critical_value = 0.948 * np.sqrt(n)  # 5% significance level
            
            # Test statistic
            max_cusum = np.max(np.abs(cusum))
            
            return {
                'cusum_statistic': max_cusum,
                'critical_value': critical_value,
                'is_stable': max_cusum < critical_value,
                'cusum_series': cusum
            }
        except Exception:
            return {'error': 'CUSUM test failed'}
    
    def get_summary(self, result: Optional[Dict] = None) -> str:
        """Get formatted summary of hedge ratio estimation"""
        
        if result is None:
            result = self.current_params
        
        if not result:
            return "No estimation results available"
        
        summary = []
        summary.append("=" * 50)
        summary.append("OLS HEDGE RATIO ESTIMATION SUMMARY")
        summary.append("=" * 50)
        
        # Basic info
        summary.append(f"Method: {result.get('method', 'Unknown')}")
        summary.append(f"Observations: {result.get('n_observations', 'Unknown')}")
        summary.append(f"Dependent variable: {result.get('dependent_var', 'Unknown')}")
        
        # Hedge ratios
        summary.append("\nHEDGE RATIOS:")
        summary.append("-" * 20)
        hedge_ratios = result.get('hedge_ratios', pd.Series())
        for asset, ratio in hedge_ratios.items():
            summary.append(f"{asset}: {ratio:.4f}")
        
        if result.get('intercept', 0) != 0:
            summary.append(f"Intercept: {result['intercept']:.4f}")
        
        # Model fit
        summary.append("\nMODEL FIT:")
        summary.append("-" * 15)
        summary.append(f"R-squared: {result.get('r_squared', np.nan):.4f}")
        summary.append(f"Adjusted R-squared: {result.get('adjusted_r_squared', np.nan):.4f}")
        summary.append(f"RMSE: {result.get('rmse', np.nan):.4f}")
        
        if 'f_statistic' in result:
            f_info = result['f_statistic']
            summary.append(f"F-statistic: {f_info.get('f_stat', np.nan):.2f} (p-value: {f_info.get('f_pvalue', np.nan):.4f})")
        
        # Significance tests
        if 'p_values' in result:
            summary.append("\nSIGNIFICANCE TESTS:")
            summary.append("-" * 20)
            for asset, p_val in result['p_values'].items():
                significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                summary.append(f"{asset}: p-value = {p_val:.4f} {significance}")
        
        summary.append("\nSignificance codes: *** p<0.01, ** p<0.05, * p<0.1")
        summary.append("=" * 50)
        
        return "\n".join(summary)