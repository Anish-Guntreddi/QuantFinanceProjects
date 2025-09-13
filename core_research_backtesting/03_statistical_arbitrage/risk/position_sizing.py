"""
Risk Parity Position Sizing

Risk parity ensures each position contributes equally to portfolio risk,
rather than equal capital allocation. This is particularly important for
statistical arbitrage where pairs may have different volatilities and
correlations.

Mathematical Foundation:
Risk contribution of position i: RC_i = w_i * (Σ * w)_i / √(w' * Σ * w)
Risk parity condition: RC_i = 1/N for all i

Implementation uses iterative optimization to solve the non-linear system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import cvxpy as cp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class RiskParityOptimizer:
    """Risk parity position sizing for statistical arbitrage portfolio"""
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        max_leverage: float = 2.0,
        risk_lookback: int = 252,
        rebalance_threshold: float = 0.05
    ):
        """
        Initialize risk parity optimizer
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            max_leverage: Maximum leverage allowed
            risk_lookback: Days for risk estimation
            rebalance_threshold: Threshold for rebalancing
        """
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.risk_lookback = risk_lookback
        self.rebalance_threshold = rebalance_threshold
        
        self.current_weights = None
        self.optimization_history = []
        
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        method: str = 'risk_parity'
    ) -> Dict:
        """
        Optimize portfolio weights
        
        Args:
            returns: Historical returns data
            expected_returns: Expected returns (optional)
            current_weights: Current portfolio weights
            method: Optimization method
            
        Returns:
            Dictionary with optimal weights and diagnostics
        """
        
        # Estimate covariance matrix
        cov_matrix = self._estimate_covariance_matrix(returns)
        
        if method == 'risk_parity':
            return self._risk_parity_optimization(returns, cov_matrix, current_weights)
        elif method == 'target_vol':
            return self._target_volatility_optimization(returns, cov_matrix, expected_returns)
        elif method == 'risk_budgeting':
            return self._risk_budgeting_optimization(returns, cov_matrix, expected_returns)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _estimate_covariance_matrix(
        self,
        returns: pd.DataFrame,
        method: str = 'shrinkage'
    ) -> pd.DataFrame:
        """Estimate covariance matrix with regularization"""
        
        # Use recent data for covariance estimation
        recent_returns = returns.tail(self.risk_lookback)
        
        if method == 'sample':
            # Sample covariance
            cov_matrix = recent_returns.cov() * 252  # Annualize
            
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage
            sample_cov = recent_returns.cov() * 252
            
            # Target: diagonal matrix
            target = np.diag(np.diag(sample_cov))
            
            # Shrinkage intensity (simplified)
            n, p = recent_returns.shape
            rho = min(0.95, max(0.05, p / n))
            
            cov_matrix = (1 - rho) * sample_cov + rho * target
            
        elif method == 'robust':
            # Robust covariance estimation
            from sklearn.covariance import LedoitWolf
            
            lw = LedoitWolf()
            lw.fit(recent_returns)
            cov_matrix = pd.DataFrame(
                lw.covariance_ * 252,
                index=returns.columns,
                columns=returns.columns
            )
        
        else:
            raise ValueError(f"Unknown covariance method: {method}")
        
        return cov_matrix
    
    def _risk_parity_optimization(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series]
    ) -> Dict:
        """Pure risk parity optimization"""
        
        n_assets = len(returns.columns)
        
        # Use CVXPY for convex optimization
        weights = cp.Variable(n_assets)
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        portfolio_vol = cp.sqrt(portfolio_variance)
        
        # Risk contributions
        risk_contrib = cp.multiply(weights, cov_matrix.values @ weights) / portfolio_vol
        
        # Target equal risk contribution
        target_contrib = portfolio_variance / n_assets
        
        # Objective: minimize deviation from equal risk contribution
        objective = cp.sum_squares(risk_contrib - target_contrib)
        
        # Constraints
        constraints = [
            portfolio_vol <= self.target_volatility,
            cp.sum(cp.abs(weights)) <= self.max_leverage,
            weights >= -0.5,  # Max 50% short per position
            weights <= 0.5    # Max 50% long per position
        ]
        
        # Transaction cost penalty if rebalancing
        if current_weights is not None:
            turnover = cp.sum(cp.abs(weights - current_weights.values))
            objective += 0.01 * turnover  # Transaction cost penalty
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            
            if weights.value is not None:
                optimal_weights = pd.Series(weights.value, index=returns.columns)
                
                # Calculate risk contributions
                risk_contributions = self._calculate_risk_contributions(
                    optimal_weights, cov_matrix
                )
                
                # Portfolio metrics
                portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
                total_leverage = optimal_weights.abs().sum()
                
                result = {
                    'optimal_weights': optimal_weights,
                    'risk_contributions': risk_contributions,
                    'portfolio_volatility': portfolio_vol,
                    'target_volatility': self.target_volatility,
                    'total_leverage': total_leverage,
                    'optimization_status': 'success',
                    'method': 'risk_parity'
                }
                
            else:
                # Fallback: equal weight
                result = self._equal_weight_fallback(returns.columns)
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            result = self._equal_weight_fallback(returns.columns)
        
        # Store result
        self.current_weights = result['optimal_weights']
        self.optimization_history.append(result)
        
        return result
    
    def _target_volatility_optimization(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        expected_returns: Optional[pd.Series]
    ) -> Dict:
        """Optimize for target volatility with expected return maximization"""
        
        n_assets = len(returns.columns)
        
        weights = cp.Variable(n_assets)
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        
        # Objective: maximize expected return (if provided)
        if expected_returns is not None:
            objective = -expected_returns.values @ weights  # Minimize negative return
        else:
            # Fallback to minimum variance
            objective = portfolio_variance
        
        # Constraints
        constraints = [
            cp.sqrt(portfolio_variance) <= self.target_volatility,  # Vol constraint
            cp.sum(cp.abs(weights)) <= self.max_leverage,           # Leverage constraint
            weights >= -0.5,
            weights <= 0.5
        ]
        
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            
            if weights.value is not None:
                optimal_weights = pd.Series(weights.value, index=returns.columns)
                
                result = {
                    'optimal_weights': optimal_weights,
                    'portfolio_volatility': np.sqrt(optimal_weights @ cov_matrix @ optimal_weights),
                    'expected_return': optimal_weights @ expected_returns if expected_returns is not None else None,
                    'total_leverage': optimal_weights.abs().sum(),
                    'optimization_status': 'success',
                    'method': 'target_volatility'
                }
            else:
                result = self._equal_weight_fallback(returns.columns)
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            result = self._equal_weight_fallback(returns.columns)
        
        return result
    
    def _risk_budgeting_optimization(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        expected_returns: Optional[pd.Series],
        risk_budgets: Optional[pd.Series] = None
    ) -> Dict:
        """Risk budgeting optimization with custom risk allocations"""
        
        if risk_budgets is None:
            # Equal risk budgets
            risk_budgets = pd.Series(1.0/len(returns.columns), index=returns.columns)
        
        n_assets = len(returns.columns)
        
        # Use scipy for non-convex risk budgeting
        def objective(w):
            """Risk budgeting objective function"""
            weights = pd.Series(w, index=returns.columns)
            
            # Calculate risk contributions
            risk_contrib = self._calculate_risk_contributions(weights, cov_matrix)
            
            # Normalized risk contributions
            total_risk = risk_contrib.sum()
            if total_risk > 0:
                risk_contrib_norm = risk_contrib / total_risk
            else:
                risk_contrib_norm = risk_contrib
            
            # Squared deviation from target budgets
            deviation = np.sum((risk_contrib_norm - risk_budgets)**2)
            
            return deviation
        
        # Constraints
        def constraint_leverage(w):
            return self.max_leverage - np.sum(np.abs(w))
        
        def constraint_vol(w):
            portfolio_vol = np.sqrt(w @ cov_matrix.values @ w)
            return self.target_volatility - portfolio_vol
        
        constraints = [
            {'type': 'ineq', 'fun': constraint_leverage},
            {'type': 'ineq', 'fun': constraint_vol}
        ]
        
        # Bounds
        bounds = [(-0.5, 0.5)] * n_assets
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=returns.columns)
                
                result_dict = {
                    'optimal_weights': optimal_weights,
                    'risk_contributions': self._calculate_risk_contributions(optimal_weights, cov_matrix),
                    'risk_budgets': risk_budgets,
                    'portfolio_volatility': np.sqrt(optimal_weights @ cov_matrix @ optimal_weights),
                    'total_leverage': optimal_weights.abs().sum(),
                    'optimization_status': 'success',
                    'method': 'risk_budgeting'
                }
            else:
                result_dict = self._equal_weight_fallback(returns.columns)
                
        except Exception as e:
            print(f"Risk budgeting optimization failed: {e}")
            result_dict = self._equal_weight_fallback(returns.columns)
        
        return result_dict
    
    def _calculate_risk_contributions(
        self,
        weights: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """Calculate risk contribution of each position"""
        
        portfolio_var = weights @ cov_matrix @ weights
        
        if portfolio_var <= 0:
            return pd.Series(0, index=weights.index)
        
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)
        
        return risk_contrib
    
    def _equal_weight_fallback(self, assets: pd.Index) -> Dict:
        """Fallback to equal weighting"""
        
        n_assets = len(assets)
        equal_weights = pd.Series(1.0/n_assets, index=assets)
        
        return {
            'optimal_weights': equal_weights,
            'optimization_status': 'fallback_equal_weight',
            'method': 'equal_weight'
        }
    
    def check_rebalancing_need(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series
    ) -> Dict:
        """Check if rebalancing is needed"""
        
        # Calculate weight drift
        weight_drift = (current_weights - target_weights).abs()
        max_drift = weight_drift.max()
        avg_drift = weight_drift.mean()
        
        needs_rebalancing = max_drift > self.rebalance_threshold
        
        return {
            'needs_rebalancing': needs_rebalancing,
            'max_drift': max_drift,
            'avg_drift': avg_drift,
            'rebalance_threshold': self.rebalance_threshold,
            'weight_drifts': weight_drift
        }
    
    def dynamic_position_sizing(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        market_regime: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Dynamic position sizing based on market conditions"""
        
        # Base position sizes from risk parity
        base_result = self.optimize_portfolio(returns)
        base_weights = base_result['optimal_weights']
        
        # Adjust for market regime
        if market_regime is not None:
            regime_adjustments = self._calculate_regime_adjustments(market_regime)
        else:
            regime_adjustments = pd.Series(1.0, index=base_weights.index)
        
        # Adjust for signal strength
        signal_adjustments = self._calculate_signal_adjustments(signals)
        
        # Combine adjustments
        dynamic_weights = base_weights * regime_adjustments * signal_adjustments
        
        # Normalize to maintain target volatility
        target_vol_scalar = self.target_volatility / np.sqrt(
            dynamic_weights @ returns.cov() * 252 @ dynamic_weights
        )
        dynamic_weights *= min(target_vol_scalar, self.max_leverage / dynamic_weights.abs().sum())
        
        return pd.DataFrame({
            'base_weights': base_weights,
            'regime_adjustments': regime_adjustments,
            'signal_adjustments': signal_adjustments,
            'dynamic_weights': dynamic_weights
        })
    
    def _calculate_regime_adjustments(self, regime: pd.Series) -> pd.Series:
        """Calculate position adjustments based on market regime"""
        
        # Get current regime
        current_regime = regime.iloc[-1] if len(regime) > 0 else 1
        
        # Regime-based scaling
        if current_regime == 0:  # Bad regime
            regime_scalar = 0.5  # Reduce positions
        elif current_regime == 1:  # Good regime
            regime_scalar = 1.0  # Normal positions
        else:
            regime_scalar = 1.2  # Excellent regime - increase slightly
        
        # Apply to all positions uniformly
        # Could be made asset-specific
        return pd.Series(regime_scalar, index=self.current_weights.index if self.current_weights is not None else [])
    
    def _calculate_signal_adjustments(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate position adjustments based on signal strength"""
        
        # This is a simplified implementation
        # In practice, would analyze signal quality, timing, etc.
        
        if 'signal_strength' in signals.columns:
            avg_signal_strength = signals['signal_strength'].tail(20).mean()
            signal_scalar = np.clip(avg_signal_strength, 0.5, 1.5)
        else:
            signal_scalar = 1.0
        
        return pd.Series(signal_scalar, index=self.current_weights.index if self.current_weights is not None else [])
    
    def backtest_position_sizing(
        self,
        returns: pd.DataFrame,
        rebalance_frequency: str = 'monthly'
    ) -> Dict:
        """Backtest position sizing strategy"""
        
        # Generate rebalancing dates
        if rebalance_frequency == 'monthly':
            rebal_dates = returns.index[::21]  # Roughly monthly
        elif rebalance_frequency == 'weekly':
            rebal_dates = returns.index[::5]
        else:  # daily
            rebal_dates = returns.index
        
        portfolio_returns = []
        weight_history = []
        
        for i, rebal_date in enumerate(rebal_dates):
            if i == 0:
                continue
                
            # Historical data up to rebalancing date
            hist_returns = returns.loc[:rebal_date].tail(self.risk_lookback)
            
            # Optimize weights
            result = self.optimize_portfolio(hist_returns)
            weights = result['optimal_weights']
            
            # Calculate returns until next rebalancing
            start_date = rebal_date
            end_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else returns.index[-1]
            
            period_returns = returns.loc[start_date:end_date]
            period_portfolio_returns = (period_returns * weights).sum(axis=1)
            
            portfolio_returns.extend(period_portfolio_returns.tolist())
            weight_history.append({
                'date': rebal_date,
                'weights': weights,
                'portfolio_vol': result.get('portfolio_volatility', np.nan)
            })
        
        portfolio_returns = pd.Series(portfolio_returns, name='portfolio_returns')
        
        # Performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        return {
            'portfolio_returns': portfolio_returns,
            'weight_history': weight_history,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'rebalance_frequency': rebalance_frequency
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1).min()
        return drawdown