"""
Probability-Aware Position Sizing

This module implements position sizing methods that incorporate prediction
probabilities, ensemble model outputs, and risk-adjusted sizing approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from scipy.stats import norm, t
from scipy.optimize import minimize_scalar, minimize
import warnings


class ProbabilityAwareSizing:
    """
    Position sizing based on prediction probabilities.
    
    Scales position sizes based on model confidence and prediction
    probabilities while incorporating risk management constraints.
    """
    
    def __init__(self, 
                 min_probability: float = 0.55,
                 max_position: float = 1.0,
                 probability_threshold: float = 0.5,
                 confidence_scaling: bool = True):
        """
        Initialize probability-aware sizing calculator.
        
        Args:
            min_probability: Minimum probability to take a position
            max_position: Maximum position size
            probability_threshold: Threshold for directional predictions
            confidence_scaling: Whether to scale by prediction confidence
        """
        self.min_probability = min_probability
        self.max_position = max_position
        self.probability_threshold = probability_threshold
        self.confidence_scaling = confidence_scaling
    
    def calculate_position_size(self, 
                               probability: float,
                               confidence: Optional[float] = None,
                               volatility: Optional[float] = None,
                               max_risk: Optional[float] = None) -> float:
        """
        Calculate position size based on probability and additional factors.
        
        Args:
            probability: Predicted probability of positive return
            confidence: Model confidence (0 to 1)
            volatility: Asset volatility
            max_risk: Maximum risk per trade
            
        Returns:
            Position size as fraction of capital
        """
        # Base position size from probability
        if probability < self.min_probability:
            return 0.0
        
        # Convert probability to position signal strength
        prob_score = self._probability_to_score(probability)
        position_size = prob_score * self.max_position
        
        # Apply confidence scaling
        if self.confidence_scaling and confidence is not None:
            confidence_multiplier = self._calculate_confidence_multiplier(confidence)
            position_size *= confidence_multiplier
        
        # Apply volatility adjustment
        if volatility is not None:
            vol_adjustment = self._calculate_volatility_adjustment(volatility)
            position_size *= vol_adjustment
        
        # Apply risk constraint
        if max_risk is not None and volatility is not None:
            risk_adjusted_size = max_risk / volatility
            position_size = min(position_size, risk_adjusted_size)
        
        return max(0, min(self.max_position, position_size))
    
    def _probability_to_score(self, probability: float) -> float:
        """Convert probability to position score."""
        # Linear scaling above threshold
        if probability > self.probability_threshold:
            score = 2 * (probability - self.probability_threshold)
        else:
            score = -2 * (self.probability_threshold - probability)
        
        return np.clip(score, -1, 1)
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate position multiplier based on model confidence."""
        # Sigmoid-like scaling
        return 2 / (1 + np.exp(-5 * (confidence - 0.5)))
    
    def _calculate_volatility_adjustment(self, volatility: float,
                                       target_volatility: float = 0.02) -> float:
        """Calculate volatility-based position adjustment."""
        return min(1.0, target_volatility / volatility) if volatility > 0 else 1.0
    
    def calculate_binary_kelly(self, probability: float, 
                              win_return: float, loss_return: float) -> float:
        """
        Calculate Kelly-optimal position size for binary outcome.
        
        Args:
            probability: Win probability
            win_return: Return if correct
            loss_return: Return if incorrect (negative)
            
        Returns:
            Kelly-optimal position size
        """
        if probability <= 0 or probability >= 1:
            return 0.0
        
        if loss_return >= 0:
            return 0.0
        
        # Kelly formula
        win_prob = probability
        lose_prob = 1 - probability
        odds_ratio = win_return / abs(loss_return)
        
        kelly_fraction = (win_prob * odds_ratio - lose_prob) / odds_ratio
        
        return max(0, min(self.max_position, kelly_fraction))
    
    def calculate_multi_outcome_sizing(self, 
                                     outcomes: np.ndarray,
                                     probabilities: np.ndarray,
                                     kelly_fraction: float = 0.25) -> float:
        """
        Calculate position size for multiple possible outcomes.
        
        Args:
            outcomes: Array of possible return outcomes
            probabilities: Array of outcome probabilities
            kelly_fraction: Fraction of Kelly criterion to use
            
        Returns:
            Optimal position size
        """
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have same length")
        
        if not np.isclose(np.sum(probabilities), 1.0):
            probabilities = probabilities / np.sum(probabilities)
        
        # Find Kelly-optimal fraction
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
        
        result = minimize_scalar(
            negative_expected_log_growth,
            bounds=(0, self.max_position),
            method='bounded'
        )
        
        if result.success:
            optimal_fraction = result.x * kelly_fraction
        else:
            # Fallback to moment-based calculation
            mean_return = np.sum(outcomes * probabilities)
            variance = np.sum(probabilities * (outcomes - mean_return) ** 2)
            optimal_fraction = (mean_return / variance) * kelly_fraction if variance > 0 else 0
        
        return max(0, min(self.max_position, optimal_fraction))


class EnsembleSizing:
    """
    Position sizing for ensemble models with multiple predictions.
    
    Combines predictions from multiple models and calculates position
    sizes that account for model agreement and uncertainty.
    """
    
    def __init__(self, 
                 aggregation_method: str = 'weighted_average',
                 uncertainty_penalty: float = 0.5):
        """
        Initialize ensemble sizing calculator.
        
        Args:
            aggregation_method: How to combine predictions ('weighted_average', 'majority_vote', 'median')
            uncertainty_penalty: Penalty for high prediction uncertainty
        """
        self.aggregation_method = aggregation_method
        self.uncertainty_penalty = uncertainty_penalty
    
    def calculate_ensemble_position(self, 
                                  predictions: Dict[str, float],
                                  model_weights: Optional[Dict[str, float]] = None,
                                  model_confidences: Optional[Dict[str, float]] = None,
                                  base_sizing_method: Any = None) -> Dict[str, Any]:
        """
        Calculate position size from ensemble predictions.
        
        Args:
            predictions: Dictionary mapping model names to predictions
            model_weights: Optional weights for each model
            model_confidences: Optional confidence scores for each model
            base_sizing_method: Base sizing method to apply to ensemble prediction
            
        Returns:
            Dictionary with position size and ensemble statistics
        """
        if not predictions:
            return {'position_size': 0, 'ensemble_prediction': 0, 'uncertainty': 1}
        
        # Set default weights
        if model_weights is None:
            model_weights = {model: 1.0/len(predictions) for model in predictions}
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        # Calculate ensemble prediction
        ensemble_pred = self._aggregate_predictions(predictions, model_weights)
        
        # Calculate prediction uncertainty
        uncertainty = self._calculate_uncertainty(predictions, model_weights)
        
        # Calculate base position size
        if base_sizing_method is not None:
            base_position = base_sizing_method.calculate_position_size(ensemble_pred)
        else:
            # Simple probability-based sizing
            base_position = 2 * abs(ensemble_pred - 0.5)
        
        # Apply uncertainty penalty
        uncertainty_multiplier = 1 - self.uncertainty_penalty * uncertainty
        final_position = base_position * uncertainty_multiplier
        
        # Calculate model agreement
        agreement_score = self._calculate_agreement_score(predictions)
        
        return {
            'position_size': max(0, final_position),
            'ensemble_prediction': ensemble_pred,
            'uncertainty': uncertainty,
            'model_agreement': agreement_score,
            'model_contributions': self._calculate_model_contributions(
                predictions, model_weights, ensemble_pred
            )
        }
    
    def _aggregate_predictions(self, 
                             predictions: Dict[str, float],
                             weights: Dict[str, float]) -> float:
        """Aggregate predictions using specified method."""
        pred_values = list(predictions.values())
        
        if self.aggregation_method == 'weighted_average':
            return sum(predictions[model] * weights[model] 
                      for model in predictions)
        
        elif self.aggregation_method == 'median':
            return np.median(pred_values)
        
        elif self.aggregation_method == 'majority_vote':
            # For binary classification
            votes = [1 if pred > 0.5 else 0 for pred in pred_values]
            majority = np.mean(votes)
            return majority
        
        else:
            # Default to weighted average
            return sum(predictions[model] * weights[model] 
                      for model in predictions)
    
    def _calculate_uncertainty(self, 
                             predictions: Dict[str, float],
                             weights: Dict[str, float]) -> float:
        """Calculate prediction uncertainty."""
        if len(predictions) == 1:
            return 0.0
        
        ensemble_pred = self._aggregate_predictions(predictions, weights)
        
        # Weighted variance of predictions
        weighted_variance = sum(
            weights[model] * (predictions[model] - ensemble_pred) ** 2
            for model in predictions
        )
        
        return min(1.0, np.sqrt(weighted_variance))
    
    def _calculate_agreement_score(self, predictions: Dict[str, float]) -> float:
        """Calculate how much models agree with each other."""
        pred_values = np.array(list(predictions.values()))
        
        if len(pred_values) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (inverse of agreement)
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        
        if mean_pred == 0:
            return 1.0 if std_pred == 0 else 0.0
        
        cv = std_pred / abs(mean_pred)
        agreement = 1 / (1 + cv)  # Higher agreement = lower CV
        
        return agreement
    
    def _calculate_model_contributions(self, 
                                     predictions: Dict[str, float],
                                     weights: Dict[str, float],
                                     ensemble_pred: float) -> Dict[str, float]:
        """Calculate each model's contribution to final prediction."""
        contributions = {}
        
        for model, pred in predictions.items():
            weight = weights[model]
            contribution = weight * (pred - 0.5)  # Contribution relative to neutral
            contributions[model] = contribution
        
        return contributions


class RiskAdjustedSizing:
    """
    Risk-adjusted position sizing with multiple risk metrics.
    
    Incorporates volatility targeting, Value-at-Risk constraints,
    and drawdown management into position sizing decisions.
    """
    
    def __init__(self, 
                 target_volatility: float = 0.15,
                 max_var: float = 0.05,
                 var_confidence: float = 0.95,
                 max_position: float = 1.0):
        """
        Initialize risk-adjusted sizing calculator.
        
        Args:
            target_volatility: Target portfolio volatility
            max_var: Maximum Value-at-Risk per position
            var_confidence: VaR confidence level
            max_position: Maximum position size
        """
        self.target_volatility = target_volatility
        self.max_var = max_var
        self.var_confidence = var_confidence
        self.max_position = max_position
    
    def calculate_volatility_targeted_size(self, 
                                         expected_return: float,
                                         asset_volatility: float) -> float:
        """
        Calculate position size for volatility targeting.
        
        Args:
            expected_return: Expected asset return
            asset_volatility: Asset volatility
            
        Returns:
            Position size to achieve target volatility
        """
        if asset_volatility <= 0:
            return 0.0
        
        # Position size = target_vol / asset_vol
        position_size = self.target_volatility / asset_volatility
        
        return min(self.max_position, position_size)
    
    def calculate_var_constrained_size(self, 
                                     expected_return: float,
                                     asset_volatility: float,
                                     distribution: str = 'normal') -> float:
        """
        Calculate position size constrained by Value-at-Risk.
        
        Args:
            expected_return: Expected asset return
            asset_volatility: Asset volatility
            distribution: Return distribution assumption
            
        Returns:
            Position size satisfying VaR constraint
        """
        if asset_volatility <= 0:
            return 0.0
        
        # Calculate VaR quantile
        if distribution == 'normal':
            var_quantile = norm.ppf(1 - self.var_confidence)
        elif distribution == 't':
            # Assume t-distribution with 5 degrees of freedom
            var_quantile = t.ppf(1 - self.var_confidence, df=5)
        else:
            var_quantile = norm.ppf(1 - self.var_confidence)
        
        # VaR for position: position_size * (expected_return + volatility * quantile)
        # Solve for position size such that VaR <= max_var
        position_var_factor = expected_return + asset_volatility * var_quantile
        
        if position_var_factor >= 0:
            # VaR is positive, no constraint needed (unlikely scenario)
            return self.max_position
        
        max_position_from_var = self.max_var / abs(position_var_factor)
        
        return min(self.max_position, max_position_from_var)
    
    def calculate_sharpe_optimized_size(self, 
                                      expected_return: float,
                                      asset_volatility: float,
                                      risk_free_rate: float = 0.0) -> float:
        """
        Calculate position size that maximizes Sharpe ratio.
        
        Args:
            expected_return: Expected asset return
            asset_volatility: Asset volatility
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe-optimal position size
        """
        if asset_volatility <= 0:
            return 0.0
        
        excess_return = expected_return - risk_free_rate
        
        if excess_return <= 0:
            return 0.0
        
        # For single asset, Sharpe-optimal position is unconstrained
        # Apply volatility targeting as practical constraint
        return self.calculate_volatility_targeted_size(expected_return, asset_volatility)
    
    def calculate_comprehensive_risk_adjusted_size(self, 
                                                 expected_return: float,
                                                 asset_volatility: float,
                                                 recent_returns: Optional[np.ndarray] = None,
                                                 max_drawdown_tolerance: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate position size using comprehensive risk management.
        
        Args:
            expected_return: Expected asset return
            asset_volatility: Asset volatility
            recent_returns: Recent return history
            max_drawdown_tolerance: Maximum acceptable drawdown
            
        Returns:
            Dictionary with position sizes from different methods
        """
        results = {}
        
        # Volatility targeting
        results['volatility_targeted'] = self.calculate_volatility_targeted_size(
            expected_return, asset_volatility
        )
        
        # VaR constrained
        results['var_constrained'] = self.calculate_var_constrained_size(
            expected_return, asset_volatility
        )
        
        # Sharpe optimized
        results['sharpe_optimized'] = self.calculate_sharpe_optimized_size(
            expected_return, asset_volatility
        )
        
        # Historical volatility adjustment
        if recent_returns is not None and len(recent_returns) > 10:
            historical_vol = np.std(recent_returns, ddof=1) * np.sqrt(252)  # Annualized
            vol_ratio = asset_volatility / historical_vol if historical_vol > 0 else 1
            vol_adjustment = min(1.5, max(0.5, 1 / vol_ratio))  # Inverse scaling
            
            results['history_adjusted'] = (
                results['volatility_targeted'] * vol_adjustment
            )
        
        # Take minimum position size across methods (conservative approach)
        position_sizes = [size for size in results.values() if size > 0]
        
        if position_sizes:
            results['conservative_size'] = min(position_sizes)
            results['aggressive_size'] = np.mean(position_sizes)
        else:
            results['conservative_size'] = 0
            results['aggressive_size'] = 0
        
        return results


class VolatilityTargetingSizing:
    """
    Volatility targeting position sizing with dynamic adjustments.
    
    Maintains constant portfolio volatility by adjusting position sizes
    based on realized and forecasted volatility.
    """
    
    def __init__(self, 
                 target_volatility: float = 0.15,
                 lookback_window: int = 60,
                 rebalancing_frequency: int = 5):
        """
        Initialize volatility targeting calculator.
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            lookback_window: Window for volatility estimation
            rebalancing_frequency: Days between rebalancing
        """
        self.target_volatility = target_volatility
        self.lookback_window = lookback_window
        self.rebalancing_frequency = rebalancing_frequency
        
        # State variables
        self.volatility_history = []
        self.position_history = []
        self.days_since_rebalance = 0
    
    def update_volatility_estimate(self, returns: np.ndarray) -> float:
        """
        Update volatility estimate with new returns.
        
        Args:
            returns: Recent return observations
            
        Returns:
            Updated volatility estimate
        """
        if len(returns) < 5:
            return self.target_volatility  # Default fallback
        
        # Use exponentially weighted volatility
        weights = np.exp(-0.05 * np.arange(len(returns)))
        weights = weights / np.sum(weights)
        
        weighted_variance = np.sum(weights * (returns - np.mean(returns)) ** 2)
        volatility = np.sqrt(weighted_variance * 252)  # Annualized
        
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > self.lookback_window:
            self.volatility_history.pop(0)
        
        return volatility
    
    def calculate_target_position(self, 
                                current_volatility: float,
                                forecasted_volatility: Optional[float] = None) -> float:
        """
        Calculate target position size for volatility targeting.
        
        Args:
            current_volatility: Current asset volatility
            forecasted_volatility: Optional forecasted volatility
            
        Returns:
            Target position size
        """
        # Use forecasted volatility if available, otherwise current
        vol_estimate = forecasted_volatility if forecasted_volatility else current_volatility
        
        if vol_estimate <= 0:
            return 0.0
        
        # Basic volatility targeting
        target_position = self.target_volatility / vol_estimate
        
        # Apply regime-based adjustments
        if len(self.volatility_history) > 10:
            regime_adjustment = self._calculate_regime_adjustment(current_volatility)
            target_position *= regime_adjustment
        
        return target_position
    
    def _calculate_regime_adjustment(self, current_volatility: float) -> float:
        """Calculate adjustment based on volatility regime."""
        if not self.volatility_history:
            return 1.0
        
        recent_avg_vol = np.mean(self.volatility_history[-10:])
        long_term_avg_vol = np.mean(self.volatility_history)
        
        # Regime identification
        if current_volatility > 1.5 * long_term_avg_vol:
            # High volatility regime - reduce position
            return 0.75
        elif current_volatility < 0.7 * long_term_avg_vol:
            # Low volatility regime - increase position slightly
            return 1.25
        else:
            # Normal regime
            return 1.0
    
    def should_rebalance(self) -> bool:
        """Determine if portfolio should be rebalanced."""
        self.days_since_rebalance += 1
        
        if self.days_since_rebalance >= self.rebalancing_frequency:
            self.days_since_rebalance = 0
            return True
        
        # Also rebalance if volatility has changed significantly
        if len(self.volatility_history) > 2:
            recent_vol = self.volatility_history[-1]
            prev_vol = self.volatility_history[-2]
            
            vol_change = abs(recent_vol - prev_vol) / prev_vol
            if vol_change > 0.2:  # 20% change
                self.days_since_rebalance = 0
                return True
        
        return False
    
    def calculate_multi_asset_volatility_target(self, 
                                              expected_returns: np.ndarray,
                                              covariance_matrix: np.ndarray,
                                              current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate volatility-targeted weights for multi-asset portfolio.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            current_weights: Current portfolio weights
            
        Returns:
            Target weights for volatility targeting
        """
        n_assets = len(expected_returns)
        
        # Objective: minimize deviation from target volatility
        def objective(weights):
            portfolio_variance = weights @ covariance_matrix @ weights
            portfolio_vol = np.sqrt(portfolio_variance)
            return (portfolio_vol - self.target_volatility) ** 2
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]  # Long-only
        
        # Initial guess
        if current_weights is not None:
            x0 = current_weights
        else:
            x0 = np.ones(n_assets) / n_assets  # Equal weights
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            # Fallback: inverse volatility weighting
            asset_vols = np.sqrt(np.diag(covariance_matrix))
            inv_vol_weights = (1 / asset_vols) / np.sum(1 / asset_vols)
            return inv_vol_weights