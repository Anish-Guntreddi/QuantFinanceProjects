"""
Dynamic Hedge Optimization

Advanced hedge ratio optimization that combines multiple methods and
adapts to changing market conditions. Includes portfolio-level hedging
and multi-objective optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class DynamicHedgeOptimizer:
    """Advanced dynamic hedge ratio optimization"""
    
    def __init__(self):
        """Initialize dynamic hedge optimizer"""
        self.optimization_history = []
        self.current_ratios = {}
        
    def optimize_hedge_ratios(
        self,
        portfolio_returns: pd.Series,
        hedge_instruments: pd.DataFrame,
        objective: str = 'minimize_variance',
        constraints: Optional[Dict] = None,
        method: str = 'portfolio_optimization'
    ) -> Dict:
        """
        Optimize hedge ratios using specified objective
        
        Args:
            portfolio_returns: Portfolio return series to hedge
            hedge_instruments: Available hedging instruments
            objective: Optimization objective
            constraints: Optimization constraints
            method: Optimization method
            
        Returns:
            Dictionary with optimal hedge ratios and metrics
        """
        
        if method == 'portfolio_optimization':
            return self._portfolio_optimization(
                portfolio_returns, hedge_instruments, objective, constraints
            )
        elif method == 'multi_objective':
            return self._multi_objective_optimization(
                portfolio_returns, hedge_instruments, constraints
            )
        elif method == 'regime_aware':
            return self._regime_aware_optimization(
                portfolio_returns, hedge_instruments, constraints
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _portfolio_optimization(
        self,
        portfolio_returns: pd.Series,
        hedge_instruments: pd.DataFrame,
        objective: str,
        constraints: Optional[Dict]
    ) -> Dict:
        """Portfolio-based hedge optimization"""
        
        # Align data
        data = pd.concat([
            portfolio_returns.rename('portfolio'),
            hedge_instruments
        ], axis=1).dropna()
        
        if len(data) < 30:
            raise ValueError("Need at least 30 observations for optimization")
        
        portfolio_ret = data['portfolio'].values
        hedge_ret = data[hedge_instruments.columns].values
        
        # Define optimization objectives
        def minimize_variance(weights):
            """Minimize hedged portfolio variance"""
            hedged_returns = portfolio_ret - hedge_ret @ weights
            return np.var(hedged_returns)
        
        def minimize_tracking_error(weights):
            """Minimize tracking error vs benchmark"""
            hedged_returns = portfolio_ret - hedge_ret @ weights
            benchmark_returns = np.zeros_like(hedged_returns)  # Zero return benchmark
            tracking_error = np.sqrt(np.mean((hedged_returns - benchmark_returns)**2))
            return tracking_error
        
        def maximize_sharpe(weights):
            """Maximize Sharpe ratio of hedged portfolio"""
            hedged_returns = portfolio_ret - hedge_ret @ weights
            mean_return = np.mean(hedged_returns)
            vol = np.std(hedged_returns)
            sharpe = mean_return / vol if vol > 0 else -np.inf
            return -sharpe  # Minimize negative Sharpe
        
        def minimize_downside_risk(weights):
            """Minimize downside deviation"""
            hedged_returns = portfolio_ret - hedge_ret @ weights
            downside_returns = hedged_returns[hedged_returns < 0]
            if len(downside_returns) == 0:
                return 0
            return np.sqrt(np.mean(downside_returns**2))
        
        # Select objective function
        objective_functions = {
            'minimize_variance': minimize_variance,
            'minimize_tracking_error': minimize_tracking_error,
            'maximize_sharpe': maximize_sharpe,
            'minimize_downside_risk': minimize_downside_risk
        }
        
        if objective not in objective_functions:
            raise ValueError(f"Unknown objective: {objective}")
        
        obj_func = objective_functions[objective]
        
        # Set up constraints
        n_instruments = hedge_ret.shape[1]
        bounds = [(-2.0, 2.0)] * n_instruments  # Default bounds
        
        if constraints:
            if 'bounds' in constraints:
                bounds = constraints['bounds']
        
        # Initial guess
        x0 = np.zeros(n_instruments)
        
        # Optimize
        try:
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_ratios = result.x
            else:
                # Fallback to least squares solution
                optimal_ratios = self._least_squares_hedge(portfolio_ret, hedge_ret)
        
        except Exception:
            optimal_ratios = self._least_squares_hedge(portfolio_ret, hedge_ret)
        
        # Calculate performance metrics
        hedged_returns = portfolio_ret - hedge_ret @ optimal_ratios
        
        metrics = self._calculate_hedge_metrics(
            portfolio_ret,
            hedged_returns,
            optimal_ratios
        )
        
        # Store results
        result_dict = {
            'optimal_ratios': pd.Series(optimal_ratios, index=hedge_instruments.columns),
            'objective': objective,
            'metrics': metrics,
            'optimization_success': result.success if 'result' in locals() else False,
            'method': 'portfolio_optimization'
        }
        
        self.current_ratios = result_dict
        self.optimization_history.append(result_dict)
        
        return result_dict
    
    def _least_squares_hedge(
        self,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray
    ) -> np.ndarray:
        """Fallback least squares hedge ratio calculation"""
        
        try:
            # Simple OLS: portfolio_ret = hedge_ret @ ratios + error
            ratios = np.linalg.lstsq(hedge_returns, portfolio_returns, rcond=None)[0]
            return ratios
        except:
            return np.zeros(hedge_returns.shape[1])
    
    def _multi_objective_optimization(
        self,
        portfolio_returns: pd.Series,
        hedge_instruments: pd.DataFrame,
        constraints: Optional[Dict]
    ) -> Dict:
        """Multi-objective optimization balancing multiple goals"""
        
        data = pd.concat([
            portfolio_returns.rename('portfolio'),
            hedge_instruments
        ], axis=1).dropna()
        
        portfolio_ret = data['portfolio'].values
        hedge_ret = data[hedge_instruments.columns].values
        
        def multi_objective(weights, alpha=0.5, beta=0.3, gamma=0.2):
            """
            Weighted combination of objectives:
            - alpha * variance minimization
            - beta * transaction cost minimization  
            - gamma * hedge effectiveness maximization
            """
            hedged_returns = portfolio_ret - hedge_ret @ weights
            
            # Objective 1: Minimize variance
            variance = np.var(hedged_returns)
            
            # Objective 2: Minimize transaction costs (proxy: sum of absolute weights)
            transaction_cost = np.sum(np.abs(weights))
            
            # Objective 3: Maximize hedge effectiveness
            unhedged_var = np.var(portfolio_ret)
            hedge_effectiveness = (unhedged_var - variance) / unhedged_var if unhedged_var > 0 else 0
            
            # Combined objective (minimize)
            return alpha * variance + beta * transaction_cost - gamma * hedge_effectiveness
        
        # Optimization
        n_instruments = hedge_ret.shape[1]
        x0 = np.zeros(n_instruments)
        bounds = [(-1.5, 1.5)] * n_instruments
        
        try:
            result = minimize(
                multi_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            optimal_ratios = result.x if result.success else self._least_squares_hedge(portfolio_ret, hedge_ret)
        except:
            optimal_ratios = self._least_squares_hedge(portfolio_ret, hedge_ret)
        
        hedged_returns = portfolio_ret - hedge_ret @ optimal_ratios
        metrics = self._calculate_hedge_metrics(portfolio_ret, hedged_returns, optimal_ratios)
        
        return {
            'optimal_ratios': pd.Series(optimal_ratios, index=hedge_instruments.columns),
            'objective': 'multi_objective',
            'metrics': metrics,
            'method': 'multi_objective_optimization'
        }
    
    def _regime_aware_optimization(
        self,
        portfolio_returns: pd.Series,
        hedge_instruments: pd.DataFrame,
        constraints: Optional[Dict]
    ) -> Dict:
        """Regime-aware hedge optimization"""
        
        # Simple regime detection based on volatility
        vol_window = 20
        rolling_vol = portfolio_returns.rolling(vol_window).std()
        high_vol_regime = rolling_vol > rolling_vol.median()
        
        data = pd.concat([
            portfolio_returns.rename('portfolio'),
            hedge_instruments,
            high_vol_regime.rename('high_vol_regime')
        ], axis=1).dropna()
        
        # Optimize separately for each regime
        regime_results = {}
        
        for regime in [True, False]:  # High vol, Low vol
            regime_mask = data['high_vol_regime'] == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) < 20:
                continue
            
            portfolio_ret = regime_data['portfolio'].values
            hedge_ret = regime_data[hedge_instruments.columns].values
            
            # Regime-specific optimization
            def regime_objective(weights):
                hedged_returns = portfolio_ret - hedge_ret @ weights
                return np.var(hedged_returns)
            
            n_instruments = hedge_ret.shape[1]
            x0 = np.zeros(n_instruments)
            bounds = [(-2.0, 2.0)] * n_instruments
            
            try:
                result = minimize(
                    regime_objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds
                )
                optimal_ratios = result.x if result.success else self._least_squares_hedge(portfolio_ret, hedge_ret)
            except:
                optimal_ratios = self._least_squares_hedge(portfolio_ret, hedge_ret)
            
            hedged_returns = portfolio_ret - hedge_ret @ optimal_ratios
            metrics = self._calculate_hedge_metrics(portfolio_ret, hedged_returns, optimal_ratios)
            
            regime_label = 'high_volatility' if regime else 'low_volatility'
            regime_results[regime_label] = {
                'ratios': pd.Series(optimal_ratios, index=hedge_instruments.columns),
                'metrics': metrics,
                'n_observations': len(regime_data)
            }
        
        # Current regime ratios (use most recent regime)
        current_regime = data['high_vol_regime'].iloc[-1]
        regime_label = 'high_volatility' if current_regime else 'low_volatility'
        current_ratios = regime_results.get(regime_label, {}).get('ratios', pd.Series(0, index=hedge_instruments.columns))
        
        return {
            'optimal_ratios': current_ratios,
            'regime_results': regime_results,
            'current_regime': regime_label,
            'method': 'regime_aware_optimization'
        }
    
    def _calculate_hedge_metrics(
        self,
        unhedged_returns: np.ndarray,
        hedged_returns: np.ndarray,
        hedge_ratios: np.ndarray
    ) -> Dict:
        """Calculate comprehensive hedge performance metrics"""
        
        # Basic statistics
        unhedged_vol = np.std(unhedged_returns) * np.sqrt(252)
        hedged_vol = np.std(hedged_returns) * np.sqrt(252)
        
        # Hedge effectiveness
        unhedged_var = np.var(unhedged_returns)
        hedged_var = np.var(hedged_returns)
        hedge_effectiveness = (unhedged_var - hedged_var) / unhedged_var if unhedged_var > 0 else 0
        
        # Risk metrics
        unhedged_var_95 = np.percentile(unhedged_returns, 5)
        hedged_var_95 = np.percentile(hedged_returns, 5)
        
        # Tracking error
        tracking_error = np.std(hedged_returns) * np.sqrt(252)
        
        # Maximum drawdown
        unhedged_cumret = np.cumprod(1 + unhedged_returns) - 1
        hedged_cumret = np.cumprod(1 + hedged_returns) - 1
        
        unhedged_dd = np.min(unhedged_cumret - np.maximum.accumulate(unhedged_cumret))
        hedged_dd = np.min(hedged_cumret - np.maximum.accumulate(hedged_cumret))
        
        return {
            'hedge_effectiveness': hedge_effectiveness,
            'variance_reduction_pct': hedge_effectiveness * 100,
            'unhedged_volatility': unhedged_vol,
            'hedged_volatility': hedged_vol,
            'volatility_reduction': (unhedged_vol - hedged_vol) / unhedged_vol * 100,
            'tracking_error': tracking_error,
            'unhedged_var_95': unhedged_var_95,
            'hedged_var_95': hedged_var_95,
            'var_improvement': (unhedged_var_95 - hedged_var_95) / abs(unhedged_var_95) * 100,
            'unhedged_max_drawdown': unhedged_dd,
            'hedged_max_drawdown': hedged_dd,
            'drawdown_improvement': (unhedged_dd - hedged_dd) / abs(unhedged_dd) * 100,
            'hedge_ratio_concentration': np.sum(np.abs(hedge_ratios)) / len(hedge_ratios),
            'total_hedge_exposure': np.sum(np.abs(hedge_ratios))
        }