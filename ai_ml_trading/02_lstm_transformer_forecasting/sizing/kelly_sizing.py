"""
Kelly Criterion Position Sizing

This module implements various Kelly criterion-based position sizing methods
including classic Kelly, fractional Kelly, multi-asset Kelly, and dynamic Kelly.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import inv, pinv
import warnings


class KellySizing:
    """
    Classic Kelly criterion position sizing.
    
    The Kelly criterion maximizes the expected logarithm of wealth and provides
    optimal position sizes for betting/investment strategies.
    """
    
    def __init__(self, max_leverage: float = 1.0, kelly_fraction: float = 0.25,
                 min_kelly: float = 0.0, max_kelly: float = 1.0):
        """
        Initialize Kelly sizing calculator.
        
        Args:
            max_leverage: Maximum allowed leverage
            kelly_fraction: Fraction of Kelly criterion to use (safety factor)
            min_kelly: Minimum Kelly position size
            max_kelly: Maximum Kelly position size
        """
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.min_kelly = min_kelly
        self.max_kelly = max_kelly
    
    def calculate_binary_kelly(self, win_prob: float, win_return: float,
                              loss_return: float) -> float:
        """
        Calculate Kelly fraction for binary outcome bet.
        
        Args:
            win_prob: Probability of winning
            win_return: Return if winning (positive)
            loss_return: Return if losing (negative)
            
        Returns:
            Optimal Kelly fraction
        """
        if loss_return >= 0:
            warnings.warn("Loss return should be negative")
            return 0.0
        
        if win_prob <= 0 or win_prob >= 1:
            warnings.warn("Win probability should be between 0 and 1")
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where p = win probability, q = loss probability, b = odds ratio
        lose_prob = 1 - win_prob
        odds_ratio = win_return / abs(loss_return)
        
        kelly_fraction = (win_prob * odds_ratio - lose_prob) / odds_ratio
        
        # Apply constraints and safety factor
        kelly_fraction = max(self.min_kelly, min(self.max_kelly, kelly_fraction))
        kelly_fraction *= self.kelly_fraction
        kelly_fraction = min(kelly_fraction, self.max_leverage)
        
        return max(0, kelly_fraction)
    
    def calculate_continuous_kelly(self, expected_return: float, 
                                 variance: float) -> float:
        """
        Calculate Kelly fraction for continuous returns (log-normal distribution).
        
        Args:
            expected_return: Expected return
            variance: Return variance
            
        Returns:
            Optimal Kelly fraction
        """
        if variance <= 0:
            warnings.warn("Variance should be positive")
            return 0.0
        
        # For continuous case: f* = μ / σ²
        kelly_fraction = expected_return / variance
        
        # Apply constraints and safety factor
        kelly_fraction = max(self.min_kelly, min(self.max_kelly, kelly_fraction))
        kelly_fraction *= self.kelly_fraction
        kelly_fraction = min(kelly_fraction, self.max_leverage)
        
        return max(0, kelly_fraction)
    
    def calculate_empirical_kelly(self, returns: np.ndarray) -> float:
        """
        Calculate Kelly fraction from empirical return distribution.
        
        Args:
            returns: Array of historical returns
            
        Returns:
            Optimal Kelly fraction
        """
        returns = np.array(returns)
        
        if len(returns) == 0:
            return 0.0
        
        # Estimate parameters from returns
        mean_return = np.mean(returns)
        return_var = np.var(returns, ddof=1)
        
        return self.calculate_continuous_kelly(mean_return, return_var)
    
    def calculate_discrete_kelly(self, outcomes: np.ndarray, 
                               probabilities: np.ndarray) -> float:
        """
        Calculate Kelly fraction for discrete outcome distribution.
        
        Args:
            outcomes: Array of possible outcomes
            probabilities: Array of corresponding probabilities
            
        Returns:
            Optimal Kelly fraction
        """
        outcomes = np.array(outcomes)
        probabilities = np.array(probabilities)
        
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have same length")
        
        if not np.isclose(np.sum(probabilities), 1.0):
            warnings.warn("Probabilities don't sum to 1, normalizing")
            probabilities = probabilities / np.sum(probabilities)
        
        # Kelly fraction maximizes expected log growth
        def negative_expected_log_growth(f):
            if f <= 0:
                return np.inf
            
            expected_log = 0
            for outcome, prob in zip(outcomes, probabilities):
                wealth_ratio = 1 + f * outcome
                if wealth_ratio <= 0:
                    return np.inf
                expected_log += prob * np.log(wealth_ratio)
            
            return -expected_log
        
        # Optimize Kelly fraction
        result = minimize_scalar(
            negative_expected_log_growth,
            bounds=(0, self.max_leverage),
            method='bounded'
        )
        
        if result.success:
            kelly_fraction = result.x * self.kelly_fraction
        else:
            # Fallback to moment-based calculation
            mean_return = np.sum(outcomes * probabilities)
            variance = np.sum(probabilities * (outcomes - mean_return) ** 2)
            kelly_fraction = self.calculate_continuous_kelly(mean_return, variance)
        
        return max(0, min(self.max_leverage, kelly_fraction))
    
    def calculate_with_confidence(self, base_kelly: float, 
                                confidence: float) -> float:
        """
        Adjust Kelly fraction based on prediction confidence.
        
        Args:
            base_kelly: Base Kelly fraction
            confidence: Confidence in prediction (0 to 1)
            
        Returns:
            Confidence-adjusted Kelly fraction
        """
        # Sigmoid-like scaling based on confidence
        confidence_multiplier = 1 / (1 + np.exp(-5 * (confidence - 0.5)))
        
        adjusted_kelly = base_kelly * confidence_multiplier
        return min(adjusted_kelly, self.max_leverage)


class MultiAssetKelly:
    """
    Kelly criterion for multi-asset portfolios.
    
    Extends Kelly criterion to handle multiple correlated assets by
    accounting for covariance structure.
    """
    
    def __init__(self, max_leverage: float = 1.0, kelly_fraction: float = 0.25,
                 regularization: float = 1e-8):
        """
        Initialize multi-asset Kelly calculator.
        
        Args:
            max_leverage: Maximum total leverage
            kelly_fraction: Safety fraction of Kelly
            regularization: Regularization for matrix inversion
        """
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.regularization = regularization
    
    def calculate_portfolio_kelly(self, expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate Kelly weights for portfolio of assets.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Optimal Kelly weights for each asset
        """
        expected_returns = np.array(expected_returns)
        covariance_matrix = np.array(covariance_matrix)
        
        if len(expected_returns) != covariance_matrix.shape[0]:
            raise ValueError("Dimension mismatch between returns and covariance")
        
        # Add regularization to covariance matrix
        n_assets = len(expected_returns)
        reg_cov = covariance_matrix + self.regularization * np.eye(n_assets)
        
        try:
            # Kelly weights: w* = Σ^(-1) * μ
            inv_cov = inv(reg_cov)
            kelly_weights = inv_cov @ expected_returns
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            warnings.warn("Covariance matrix is singular, using pseudo-inverse")
            inv_cov = pinv(reg_cov)
            kelly_weights = inv_cov @ expected_returns
        
        # Apply safety fraction
        kelly_weights *= self.kelly_fraction
        
        # Apply leverage constraint
        total_leverage = np.sum(np.abs(kelly_weights))
        if total_leverage > self.max_leverage:
            kelly_weights *= self.max_leverage / total_leverage
        
        return kelly_weights
    
    def calculate_with_constraints(self, expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 min_weights: Optional[np.ndarray] = None,
                                 max_weights: Optional[np.ndarray] = None,
                                 long_only: bool = False) -> np.ndarray:
        """
        Calculate Kelly weights with additional constraints.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            min_weights: Minimum weights for each asset
            max_weights: Maximum weights for each asset
            long_only: Whether to allow only long positions
            
        Returns:
            Constrained optimal weights
        """
        n_assets = len(expected_returns)
        
        # Objective function: maximize expected log growth
        def objective(weights):
            portfolio_return = weights @ expected_returns
            portfolio_variance = weights @ covariance_matrix @ weights
            
            if portfolio_variance <= 0:
                return np.inf
            
            # Approximate expected log growth using Taylor expansion
            expected_log_growth = portfolio_return - 0.5 * portfolio_variance
            return -expected_log_growth  # Minimize negative
        
        # Set up constraints
        constraints = []
        
        # Leverage constraint
        def leverage_constraint(weights):
            return self.max_leverage - np.sum(np.abs(weights))
        
        constraints.append({'type': 'ineq', 'fun': leverage_constraint})
        
        # Set bounds
        if long_only:
            bounds = [(0, None) for _ in range(n_assets)]
        else:
            bounds = [(None, None) for _ in range(n_assets)]
        
        if min_weights is not None:
            bounds = [(max(bounds[i][0] or -np.inf, min_weights[i]), bounds[i][1]) 
                     for i in range(n_assets)]
        
        if max_weights is not None:
            bounds = [(bounds[i][0], min(bounds[i][1] or np.inf, max_weights[i])) 
                     for i in range(n_assets)]
        
        # Initial guess: unconstrained Kelly weights scaled down
        x0 = self.calculate_portfolio_kelly(expected_returns, covariance_matrix)
        x0 *= 0.5  # Start with conservative guess
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x * self.kelly_fraction
        else:
            warnings.warn("Optimization failed, returning unconstrained Kelly")
            return self.calculate_portfolio_kelly(expected_returns, covariance_matrix)


class DynamicKelly:
    """
    Dynamic Kelly sizing that adapts to changing market conditions.
    
    Adjusts Kelly fractions based on market regime, volatility, and
    recent performance.
    """
    
    def __init__(self, base_kelly_fraction: float = 0.25,
                 lookback_window: int = 252,
                 volatility_adjustment: bool = True,
                 drawdown_adjustment: bool = True):
        """
        Initialize dynamic Kelly calculator.
        
        Args:
            base_kelly_fraction: Base Kelly fraction
            lookback_window: Window for calculating statistics
            volatility_adjustment: Whether to adjust for volatility regime
            drawdown_adjustment: Whether to reduce size during drawdowns
        """
        self.base_kelly_fraction = base_kelly_fraction
        self.lookback_window = lookback_window
        self.volatility_adjustment = volatility_adjustment
        self.drawdown_adjustment = drawdown_adjustment
        
        # State variables
        self.performance_history = []
        self.volatility_history = []
    
    def update_history(self, return_value: float, volatility: Optional[float] = None):
        """
        Update performance and volatility history.
        
        Args:
            return_value: Most recent return
            volatility: Most recent volatility estimate
        """
        self.performance_history.append(return_value)
        if len(self.performance_history) > self.lookback_window:
            self.performance_history.pop(0)
        
        if volatility is not None:
            self.volatility_history.append(volatility)
            if len(self.volatility_history) > self.lookback_window:
                self.volatility_history.pop(0)
    
    def calculate_adaptive_kelly(self, base_kelly: float) -> float:
        """
        Calculate Kelly fraction adapted to current market conditions.
        
        Args:
            base_kelly: Base Kelly fraction to adjust
            
        Returns:
            Adapted Kelly fraction
        """
        adjusted_kelly = base_kelly * self.base_kelly_fraction
        
        # Volatility adjustment
        if self.volatility_adjustment and len(self.volatility_history) > 20:
            recent_vol = np.mean(self.volatility_history[-20:])
            long_term_vol = np.mean(self.volatility_history)
            
            if long_term_vol > 0:
                vol_ratio = recent_vol / long_term_vol
                # Reduce size in high volatility, increase in low volatility
                vol_adjustment = 2 / (1 + vol_ratio)
                adjusted_kelly *= vol_adjustment
        
        # Drawdown adjustment
        if self.drawdown_adjustment and len(self.performance_history) > 10:
            cumulative_returns = np.cumprod(1 + np.array(self.performance_history))
            current_dd = 1 - cumulative_returns[-1] / np.maximum.accumulate(cumulative_returns)[-1]
            
            if current_dd > 0.05:  # If in 5%+ drawdown
                dd_adjustment = max(0.5, 1 - current_dd * 2)  # Reduce size
                adjusted_kelly *= dd_adjustment
        
        return max(0, adjusted_kelly)
    
    def calculate_regime_aware_kelly(self, base_kelly: float,
                                   regime_probabilities: Dict[str, float],
                                   regime_parameters: Dict[str, Dict]) -> float:
        """
        Calculate Kelly fraction based on market regime probabilities.
        
        Args:
            base_kelly: Base Kelly fraction
            regime_probabilities: Probability of each regime
            regime_parameters: Parameters (mean, var) for each regime
            
        Returns:
            Regime-weighted Kelly fraction
        """
        weighted_kelly = 0
        
        for regime, prob in regime_probabilities.items():
            if regime in regime_parameters:
                params = regime_parameters[regime]
                mean_return = params.get('mean', 0)
                variance = params.get('variance', 1)
                
                if variance > 0:
                    regime_kelly = mean_return / variance
                    weighted_kelly += prob * regime_kelly
        
        # Apply base adjustments
        final_kelly = weighted_kelly * self.base_kelly_fraction
        return self.calculate_adaptive_kelly(final_kelly)


class FractionalKelly:
    """
    Fractional Kelly sizing with various risk management overlays.
    
    Implements various approaches to reduce the Kelly fraction for
    practical risk management.
    """
    
    def __init__(self, kelly_fraction: float = 0.25):
        """
        Initialize fractional Kelly calculator.
        
        Args:
            kelly_fraction: Fraction of full Kelly to use
        """
        self.kelly_fraction = kelly_fraction
    
    def calculate_fractional_kelly(self, full_kelly: float,
                                 confidence: Optional[float] = None,
                                 max_drawdown_tolerance: Optional[float] = None,
                                 utility_function: str = 'log') -> float:
        """
        Calculate fractional Kelly with various adjustments.
        
        Args:
            full_kelly: Full Kelly criterion result
            confidence: Confidence in the prediction (0 to 1)
            max_drawdown_tolerance: Maximum acceptable drawdown
            utility_function: Utility function type ('log', 'power', 'exponential')
            
        Returns:
            Adjusted Kelly fraction
        """
        # Start with base fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction
        
        # Confidence adjustment
        if confidence is not None:
            confidence_factor = self._calculate_confidence_factor(confidence)
            fractional_kelly *= confidence_factor
        
        # Drawdown tolerance adjustment
        if max_drawdown_tolerance is not None:
            dd_factor = self._calculate_drawdown_factor(full_kelly, max_drawdown_tolerance)
            fractional_kelly *= dd_factor
        
        # Utility function adjustment
        utility_factor = self._calculate_utility_factor(utility_function)
        fractional_kelly *= utility_factor
        
        return max(0, fractional_kelly)
    
    def _calculate_confidence_factor(self, confidence: float) -> float:
        """Calculate adjustment factor based on prediction confidence."""
        # Sigmoid transformation
        return 1 / (1 + np.exp(-10 * (confidence - 0.5)))
    
    def _calculate_drawdown_factor(self, kelly_fraction: float,
                                 max_dd_tolerance: float) -> float:
        """Calculate adjustment factor based on drawdown tolerance."""
        # Estimate maximum drawdown for given Kelly fraction
        # Using approximate formula: MaxDD ≈ Kelly² / (4 * edge)
        # For simplicity, use conservative approximation
        estimated_max_dd = kelly_fraction * 0.5  # Rough approximation
        
        if estimated_max_dd > max_dd_tolerance:
            return max_dd_tolerance / estimated_max_dd
        
        return 1.0
    
    def _calculate_utility_factor(self, utility_function: str) -> float:
        """Calculate adjustment factor based on utility function."""
        if utility_function == 'log':
            return 1.0  # Full Kelly is optimal for log utility
        elif utility_function == 'power':
            return 0.75  # Conservative for power utility
        elif utility_function == 'exponential':
            return 0.5  # Very conservative for exponential utility
        else:
            return 1.0
    
    def calculate_time_varying_fraction(self, lookback_returns: np.ndarray,
                                      base_fraction: Optional[float] = None) -> float:
        """
        Calculate time-varying Kelly fraction based on recent performance.
        
        Args:
            lookback_returns: Recent return history
            base_fraction: Base fraction to adjust (default: self.kelly_fraction)
            
        Returns:
            Time-varying Kelly fraction
        """
        if base_fraction is None:
            base_fraction = self.kelly_fraction
        
        if len(lookback_returns) < 5:
            return base_fraction
        
        # Calculate recent Sharpe ratio
        mean_return = np.mean(lookback_returns)
        std_return = np.std(lookback_returns, ddof=1)
        
        if std_return == 0:
            return base_fraction
        
        sharpe_ratio = mean_return / std_return
        
        # Adjust fraction based on Sharpe ratio
        if sharpe_ratio > 1.0:
            adjustment = min(1.5, 1 + 0.1 * (sharpe_ratio - 1))
        elif sharpe_ratio < 0:
            adjustment = max(0.1, 0.5 + 0.5 * sharpe_ratio)
        else:
            adjustment = 0.5 + 0.5 * sharpe_ratio
        
        return base_fraction * adjustment


class SimulatedKelly:
    """
    Kelly criterion calculation using Monte Carlo simulation.
    
    Uses simulation to handle complex return distributions that
    don't have analytical solutions.
    """
    
    def __init__(self, n_simulations: int = 10000, n_periods: int = 252):
        """
        Initialize simulated Kelly calculator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            n_periods: Number of periods to simulate
        """
        self.n_simulations = n_simulations
        self.n_periods = n_periods
    
    def calculate_simulated_kelly(self, return_generator: callable,
                                kelly_range: Tuple[float, float] = (0, 1),
                                n_kelly_points: int = 50) -> Tuple[float, Dict]:
        """
        Calculate optimal Kelly fraction using Monte Carlo simulation.
        
        Args:
            return_generator: Function that generates random returns
            kelly_range: Range of Kelly fractions to test
            n_kelly_points: Number of Kelly fractions to test
            
        Returns:
            Tuple of (optimal_kelly, results_dict)
        """
        kelly_fractions = np.linspace(kelly_range[0], kelly_range[1], n_kelly_points)
        final_wealths = np.zeros((len(kelly_fractions), self.n_simulations))
        
        for i, kelly_f in enumerate(kelly_fractions):
            for sim in range(self.n_simulations):
                wealth = 1.0  # Starting wealth
                
                for period in range(self.n_periods):
                    return_val = return_generator()
                    wealth *= (1 + kelly_f * return_val)
                    
                    if wealth <= 0:  # Bankruptcy
                        wealth = 1e-10  # Small positive value
                        break
                
                final_wealths[i, sim] = wealth
        
        # Calculate expected log growth for each Kelly fraction
        expected_log_growth = np.mean(np.log(final_wealths), axis=1)
        
        # Find optimal Kelly fraction
        optimal_idx = np.argmax(expected_log_growth)
        optimal_kelly = kelly_fractions[optimal_idx]
        
        results = {
            'kelly_fractions': kelly_fractions,
            'expected_log_growth': expected_log_growth,
            'final_wealths': final_wealths,
            'optimal_kelly': optimal_kelly,
            'max_expected_growth': expected_log_growth[optimal_idx]
        }
        
        return optimal_kelly, results
    
    def calculate_drawdown_constrained_kelly(self, return_generator: callable,
                                          max_drawdown: float = 0.2,
                                          confidence_level: float = 0.95) -> float:
        """
        Calculate Kelly fraction with drawdown constraints.
        
        Args:
            return_generator: Function that generates random returns
            max_drawdown: Maximum acceptable drawdown
            confidence_level: Confidence level for drawdown constraint
            
        Returns:
            Kelly fraction that satisfies drawdown constraint
        """
        def test_kelly_fraction(kelly_f):
            drawdowns = []
            
            for sim in range(self.n_simulations):
                wealth_path = [1.0]
                
                for period in range(self.n_periods):
                    return_val = return_generator()
                    new_wealth = wealth_path[-1] * (1 + kelly_f * return_val)
                    wealth_path.append(max(new_wealth, 1e-10))
                
                # Calculate maximum drawdown for this path
                wealth_path = np.array(wealth_path)
                peak = np.maximum.accumulate(wealth_path)
                drawdown_path = (wealth_path - peak) / peak
                max_dd = abs(np.min(drawdown_path))
                drawdowns.append(max_dd)
            
            # Check if drawdown constraint is satisfied
            percentile_dd = np.percentile(drawdowns, confidence_level * 100)
            return percentile_dd <= max_drawdown
        
        # Binary search for maximum Kelly fraction that satisfies constraint
        low, high = 0.0, 1.0
        tolerance = 1e-4
        
        while high - low > tolerance:
            mid = (low + high) / 2
            
            if test_kelly_fraction(mid):
                low = mid
            else:
                high = mid
        
        return low