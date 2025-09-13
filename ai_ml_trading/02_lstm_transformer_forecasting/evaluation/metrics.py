"""
Comprehensive Metrics for Time Series Forecasting Evaluation

This module provides a wide range of metrics for evaluating time series
forecasting models, including regression, classification, and trading metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, brier_score_loss,
    log_loss
)
from scipy import stats
from scipy.stats import jarque_bera, normaltest
import warnings


class ForecastMetrics:
    """
    General forecasting metrics applicable to various prediction tasks.
    """
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Avoid division by zero
        non_zero_mask = y_true != 0
        if not np.any(non_zero_mask):
            return np.inf
        
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    @staticmethod
    def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        non_zero_mask = denominator != 0
        
        if not np.any(non_zero_mask):
            return 0.0
        
        return np.mean(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / denominator[non_zero_mask]) * 100
    
    @staticmethod
    def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray,
                                  y_train: np.ndarray, seasonality: int = 1) -> float:
        """Calculate Mean Absolute Scaled Error (MASE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_train = np.array(y_train)
        
        # Calculate MAE of forecast
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        
        # Calculate MAE of naive forecast on training data
        if len(y_train) <= seasonality:
            return np.inf
        
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        mae_naive = np.mean(naive_errors)
        
        if mae_naive == 0:
            return np.inf if mae_forecast > 0 else 0
        
        return mae_forecast / mae_naive
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (fraction of correct direction predictions)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        if len(y_true) <= 1:
            return np.nan
        
        # Calculate changes
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Check direction agreement
        correct_directions = (true_changes * pred_changes) > 0
        
        return np.mean(correct_directions)
    
    @staticmethod
    def theil_u_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        if len(y_true) <= 1:
            return np.inf
        
        # Calculate forecast errors
        forecast_error = np.sum((y_pred[1:] - y_true[1:]) ** 2)
        
        # Calculate naive forecast errors (no-change forecast)
        naive_error = np.sum((y_true[:-1] - y_true[1:]) ** 2)
        
        if naive_error == 0:
            return np.inf if forecast_error > 0 else 0
        
        return np.sqrt(forecast_error / naive_error)


class RegressionMetrics:
    """
    Metrics specifically for regression tasks.
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['mape'] = ForecastMetrics.mean_absolute_percentage_error(y_true, y_pred)
        metrics['smape'] = ForecastMetrics.symmetric_mape(y_true, y_pred)
        metrics['directional_accuracy'] = ForecastMetrics.directional_accuracy(y_true, y_pred)
        metrics['theil_u'] = ForecastMetrics.theil_u_statistic(y_true, y_pred)
        
        # MASE if training data provided
        if y_train is not None:
            metrics['mase'] = ForecastMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train)
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['skew_residual'] = stats.skew(residuals)
        metrics['kurtosis_residual'] = stats.kurtosis(residuals)
        
        # Normality tests on residuals
        try:
            _, jb_p_value = jarque_bera(residuals)
            metrics['residual_normality_p'] = jb_p_value
        except:
            metrics['residual_normality_p'] = np.nan
        
        # Correlation between true and predicted
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        return metrics
    
    @staticmethod
    def prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                           confidence_levels: List[float] = [0.95, 0.90, 0.80]) -> Dict[str, float]:
        """Calculate prediction interval coverage."""
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        results = {}
        
        for confidence in confidence_levels:
            # Assuming normal distribution of residuals
            alpha = 1 - confidence
            z_score = stats.norm.ppf(1 - alpha/2)
            
            # Calculate prediction intervals
            lower_bound = y_pred - z_score * residual_std
            upper_bound = y_pred + z_score * residual_std
            
            # Calculate coverage
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            
            results[f'coverage_{int(confidence*100)}'] = coverage
            results[f'interval_width_{int(confidence*100)}'] = np.mean(upper_bound - lower_bound)
        
        return results


class ClassificationMetrics:
    """
    Metrics for classification tasks.
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                            threshold: float = 0.5) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Probability-based metrics
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc_roc'] = np.nan
        
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        metrics['log_loss'] = log_loss(y_true, y_pred_proba, eps=1e-15)
        
        # Confusion matrix elements
        tn, fp, fn, tp = ClassificationMetrics._confusion_matrix_elements(y_true, y_pred)
        
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['true_negatives'] = tn
        metrics['false_negatives'] = fn
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Matthews correlation coefficient
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            metrics['mcc'] = 0
        else:
            metrics['mcc'] = (tp * tn - fp * fn) / denominator
        
        return metrics
    
    @staticmethod
    def _confusion_matrix_elements(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix elements."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return tn, fp, fn, tp
    
    @staticmethod
    def profit_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    costs: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, Any]:
        """
        Calculate profit curve for different probability thresholds.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            costs: (cost_false_positive, cost_false_negative)
            
        Returns:
            Dictionary with profit curve data
        """
        thresholds = np.linspace(0, 1, 101)
        profits = []
        
        cost_fp, cost_fn = costs
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calculate profit/loss
            profit = tp - fp * cost_fp - fn * cost_fn
            profits.append(profit)
        
        profits = np.array(profits)
        optimal_idx = np.argmax(profits)
        
        return {
            'thresholds': thresholds,
            'profits': profits,
            'optimal_threshold': thresholds[optimal_idx],
            'max_profit': profits[optimal_idx]
        }
    
    @staticmethod
    def precision_at_recall(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          target_recall: float = 0.8) -> float:
        """Calculate precision at specific recall level."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find closest recall to target
        idx = np.argmin(np.abs(recall - target_recall))
        return precision[idx]


class BacktestMetrics:
    """
    Metrics for backtesting trading strategies.
    """
    
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, 
                            benchmark: Optional[pd.Series] = None,
                            risk_free_rate: float = 0.0) -> Dict[str, float]:
        """Calculate comprehensive backtest metrics."""
        returns = pd.Series(returns)
        
        if benchmark is not None:
            benchmark = pd.Series(benchmark)
            # Align returns and benchmark
            common_index = returns.index.intersection(benchmark.index)
            returns = returns.loc[common_index]
            benchmark = benchmark.loc[common_index]
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = returns.mean() * 252
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        if metrics['annual_volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['annual_return'] - risk_free_rate) / metrics['annual_volatility']
        else:
            metrics['sharpe_ratio'] = np.inf if metrics['annual_return'] > risk_free_rate else 0
        
        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        metrics['drawdown_duration'] = BacktestMetrics._max_drawdown_duration(drawdown)
        
        # Calmar ratio
        if abs(metrics['max_drawdown']) > 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = np.inf if metrics['annual_return'] > 0 else 0
        
        # Value at Risk and Expected Shortfall
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Higher moments
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Downside metrics
        downside_returns = returns[returns < risk_free_rate/252]
        if len(downside_returns) > 0:
            metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = (metrics['annual_return'] - risk_free_rate) / metrics['downside_deviation']
        else:
            metrics['downside_deviation'] = 0
            metrics['sortino_ratio'] = np.inf if metrics['annual_return'] > risk_free_rate else 0
        
        # Win/loss statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        metrics['win_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics['avg_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        if len(negative_returns) > 0 and negative_returns.sum() != 0:
            metrics['profit_factor'] = positive_returns.sum() / abs(negative_returns.sum())
        else:
            metrics['profit_factor'] = np.inf if len(positive_returns) > 0 else 0
        
        # Benchmark relative metrics
        if benchmark is not None:
            excess_returns = returns - benchmark
            
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            
            if metrics['tracking_error'] > 0:
                metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            else:
                metrics['information_ratio'] = np.inf if excess_returns.mean() > 0 else 0
            
            # Beta and alpha
            if benchmark.std() > 0:
                metrics['beta'] = np.cov(returns, benchmark)[0, 1] / benchmark.var()
                metrics['alpha'] = metrics['annual_return'] - metrics['beta'] * (benchmark.mean() * 252)
            else:
                metrics['beta'] = 0
                metrics['alpha'] = metrics['annual_return']
        
        return metrics
    
    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        is_drawdown = drawdown < 0
        
        if not is_drawdown.any():
            return 0
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, is_dd in is_drawdown.items():
            if is_dd and not in_drawdown:
                # Start of drawdown
                start_date = date
                in_drawdown = True
            elif not is_dd and in_drawdown:
                # End of drawdown
                if start_date is not None:
                    duration = (date - start_date).days
                    drawdown_periods.append(duration)
                in_drawdown = False
        
        # Handle case where drawdown continues to end
        if in_drawdown and start_date is not None:
            duration = (drawdown.index[-1] - start_date).days
            drawdown_periods.append(duration)
        
        return max(drawdown_periods) if drawdown_periods else 0


class RiskMetrics:
    """
    Comprehensive risk metrics for portfolio evaluation.
    """
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """Calculate Value at Risk."""
        returns = np.array(returns)
        
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Assume normal distribution
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean_return + z_score * std_return
        
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            z = stats.norm.ppf(1 - confidence_level)
            
            # Cornish-Fisher adjustment
            cf_adjustment = (z + (z**2 - 1) * skew / 6 + 
                           (z**3 - 3*z) * kurt / 24 - 
                           (2*z**3 - 5*z) * skew**2 / 36)
            
            return mean_return + cf_adjustment * std_return
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = RiskMetrics.calculate_var(returns, confidence_level, method='historical')
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_maximum_drawdown_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Maximum Drawdown at Risk."""
        returns = np.array(returns)
        cumulative = np.cumprod(1 + returns)
        
        # Calculate all possible drawdowns
        drawdowns = []
        for i in range(len(cumulative)):
            for j in range(i + 1, len(cumulative)):
                dd = (cumulative[j] - cumulative[i]) / cumulative[i]
                if dd < 0:
                    drawdowns.append(dd)
        
        if not drawdowns:
            return 0
        
        return np.percentile(drawdowns, (1 - confidence_level) * 100)


class TradingMetrics:
    """
    Metrics specific to trading strategy evaluation.
    """
    
    @staticmethod
    def calculate_trade_statistics(trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistics from individual trades.
        
        Expected columns in trades DataFrame:
        - 'pnl': Profit/loss for each trade
        - 'duration': Duration of each trade
        - 'direction': 1 for long, -1 for short
        """
        if len(trades) == 0:
            return {}
        
        metrics = {}
        
        # Basic trade statistics
        metrics['total_trades'] = len(trades)
        metrics['total_pnl'] = trades['pnl'].sum()
        metrics['avg_pnl_per_trade'] = trades['pnl'].mean()
        
        # Win/loss statistics
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        metrics['num_winning_trades'] = len(winning_trades)
        metrics['num_losing_trades'] = len(losing_trades)
        metrics['win_rate'] = len(winning_trades) / len(trades)
        
        if len(winning_trades) > 0:
            metrics['avg_win'] = winning_trades['pnl'].mean()
            metrics['largest_win'] = winning_trades['pnl'].max()
        else:
            metrics['avg_win'] = 0
            metrics['largest_win'] = 0
        
        if len(losing_trades) > 0:
            metrics['avg_loss'] = losing_trades['pnl'].mean()
            metrics['largest_loss'] = losing_trades['pnl'].min()
        else:
            metrics['avg_loss'] = 0
            metrics['largest_loss'] = 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        if gross_loss > 0:
            metrics['profit_factor'] = gross_profit / gross_loss
        else:
            metrics['profit_factor'] = np.inf if gross_profit > 0 else 0
        
        # Average win to average loss ratio
        if metrics['avg_loss'] < 0:
            metrics['avg_win_loss_ratio'] = metrics['avg_win'] / abs(metrics['avg_loss'])
        else:
            metrics['avg_win_loss_ratio'] = np.inf if metrics['avg_win'] > 0 else 0
        
        # Trade duration statistics
        if 'duration' in trades.columns:
            metrics['avg_trade_duration'] = trades['duration'].mean()
            metrics['max_trade_duration'] = trades['duration'].max()
            metrics['min_trade_duration'] = trades['duration'].min()
        
        # Long vs short performance
        if 'direction' in trades.columns:
            long_trades = trades[trades['direction'] == 1]
            short_trades = trades[trades['direction'] == -1]
            
            if len(long_trades) > 0:
                metrics['long_win_rate'] = (long_trades['pnl'] > 0).mean()
                metrics['long_avg_pnl'] = long_trades['pnl'].mean()
            
            if len(short_trades) > 0:
                metrics['short_win_rate'] = (short_trades['pnl'] > 0).mean()
                metrics['short_avg_pnl'] = short_trades['pnl'].mean()
        
        return metrics
    
    @staticmethod
    def calculate_turnover_metrics(positions: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """Calculate portfolio turnover metrics."""
        position_changes = positions.diff().abs()
        
        # Portfolio turnover (sum of absolute position changes)
        turnover = position_changes.sum()
        
        # Average daily turnover
        avg_daily_turnover = position_changes.mean()
        
        # Turnover as fraction of portfolio value
        if len(prices) > 0:
            avg_portfolio_value = (positions * prices).abs().mean()
            if avg_portfolio_value > 0:
                turnover_ratio = turnover / avg_portfolio_value
            else:
                turnover_ratio = 0
        else:
            turnover_ratio = 0
        
        return {
            'total_turnover': turnover,
            'avg_daily_turnover': avg_daily_turnover,
            'turnover_ratio': turnover_ratio
        }
    
    @staticmethod
    def calculate_transaction_cost_impact(returns_gross: pd.Series, 
                                        returns_net: pd.Series) -> Dict[str, float]:
        """Calculate impact of transaction costs on returns."""
        cost_impact = returns_gross - returns_net
        
        return {
            'total_cost_impact': cost_impact.sum(),
            'avg_daily_cost': cost_impact.mean(),
            'cost_as_pct_of_gross_return': (cost_impact.sum() / returns_gross.sum()) * 100 if returns_gross.sum() != 0 else 0,
            'cost_drag_on_annual_return': cost_impact.mean() * 252
        }