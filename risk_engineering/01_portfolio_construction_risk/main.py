"""Main portfolio construction and risk management module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import risk modules
from risk.models import (
    EWMACovariance,
    LedoitWolfCovariance,
    NCOCovariance,
    RobustCovariance,
    FactorModel
)

# Import optimization modules
from opt import (
    KellyCriterion,
    ConvexPositionSizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    BlackLitterman,
    PortfolioConstraints
)

# Import risk metrics
from risk.metrics import (
    ValueAtRisk,
    ConditionalValueAtRisk,
    TrackingErrorAnalyzer,
    RiskAttribution
)


class PortfolioManager:
    """Main portfolio construction and risk management class."""
    
    def __init__(
        self,
        asset_names: List[str],
        risk_aversion: float = 1.0,
        max_leverage: float = 1.0
    ):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.risk_aversion = risk_aversion
        self.max_leverage = max_leverage
        
        # Components
        self.covariance_estimator = None
        self.optimizer = None
        self.risk_metrics = {}
        
        # Portfolio state
        self.weights = None
        self.expected_returns = None
        self.covariance = None
        
    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: str = 'ledoit_wolf',
        **kwargs
    ) -> np.ndarray:
        """Estimate covariance matrix."""
        
        if method == 'ewma':
            self.covariance_estimator = EWMACovariance(**kwargs)
        elif method == 'ledoit_wolf':
            self.covariance_estimator = LedoitWolfCovariance(**kwargs)
        elif method == 'nco':
            self.covariance_estimator = NCOCovariance(**kwargs)
        elif method == 'robust':
            self.covariance_estimator = RobustCovariance(**kwargs)
        elif method == 'factor':
            self.covariance_estimator = FactorModel(**kwargs)
        else:
            raise ValueError(f"Unknown covariance method: {method}")
            
        self.covariance_estimator.fit(returns)
        self.covariance = self.covariance_estimator.predict()
        
        return self.covariance
    
    def set_expected_returns(
        self,
        expected_returns: Optional[np.ndarray] = None,
        returns: Optional[pd.DataFrame] = None,
        method: str = 'historical'
    ):
        """Set expected returns."""
        
        if expected_returns is not None:
            self.expected_returns = expected_returns
        elif returns is not None:
            if method == 'historical':
                self.expected_returns = returns.mean().values
            elif method == 'exponential':
                # Exponentially weighted mean
                ewm = returns.ewm(span=60, adjust=False).mean()
                self.expected_returns = ewm.iloc[-1].values
            else:
                raise ValueError(f"Unknown returns method: {method}")
        else:
            raise ValueError("Must provide either expected_returns or returns")
    
    def optimize_portfolio(
        self,
        method: str = 'mean_variance',
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Optimize portfolio weights."""
        
        if self.expected_returns is None or self.covariance is None:
            raise ValueError("Must set expected returns and covariance first")
            
        results = {}
        
        if method == 'mean_variance':
            optimizer = MeanVarianceOptimizer(
                risk_aversion=self.risk_aversion,
                max_leverage=self.max_leverage
            )
            results = optimizer.optimize(
                self.expected_returns,
                self.covariance,
                constraints
            )
            self.weights = results['weights']
            
        elif method == 'kelly':
            kelly = KellyCriterion(
                max_leverage=self.max_leverage,
                **kwargs
            )
            self.weights = kelly.calculate_kelly_portfolio(
                self.expected_returns,
                self.covariance
            )
            results['weights'] = self.weights
            results['growth_rate'] = kelly.calculate_growth_rate(
                self.weights,
                self.expected_returns,
                self.covariance
            )
            
        elif method == 'risk_parity':
            rp_optimizer = RiskParityOptimizer(**kwargs)
            results = rp_optimizer.optimize(
                self.covariance,
                constraints=constraints
            )
            self.weights = results['weights']
            
        elif method == 'min_variance':
            sizer = ConvexPositionSizer(
                risk_aversion=self.risk_aversion,
                max_leverage=self.max_leverage
            )
            self.weights = sizer.minimum_variance_optimization(
                self.covariance,
                constraints
            )
            results['weights'] = self.weights
            
        elif method == 'max_sharpe':
            optimizer = MeanVarianceOptimizer(
                risk_aversion=self.risk_aversion,
                max_leverage=self.max_leverage
            )
            results = optimizer.maximize_sharpe_ratio(
                self.expected_returns,
                self.covariance,
                constraints=constraints,
                **kwargs
            )
            self.weights = results['weights']
            
        elif method == 'cvar':
            # Need return scenarios for CVaR
            n_scenarios = kwargs.get('n_scenarios', 1000)
            scenarios = np.random.multivariate_normal(
                self.expected_returns,
                self.covariance,
                n_scenarios
            )
            
            sizer = ConvexPositionSizer(
                risk_aversion=self.risk_aversion,
                max_leverage=self.max_leverage
            )
            self.weights = sizer.cvar_optimization(
                scenarios,
                **kwargs
            )
            results['weights'] = self.weights
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
        # Calculate portfolio metrics
        results.update(self.calculate_portfolio_metrics())
        
        return results
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio risk and return metrics."""
        
        if self.weights is None:
            raise ValueError("No portfolio weights set")
            
        metrics = {}
        
        # Basic metrics
        metrics['expected_return'] = self.weights @ self.expected_returns
        metrics['volatility'] = np.sqrt(self.weights @ self.covariance @ self.weights)
        metrics['sharpe_ratio'] = metrics['expected_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # Risk metrics
        var_calc = ValueAtRisk()
        metrics['var_95'] = var_calc.calculate_parametric_var(
            metrics['expected_return'],
            metrics['volatility']
        )
        
        cvar_calc = ConditionalValueAtRisk()
        metrics['cvar_95'], _ = cvar_calc.calculate_parametric_cvar(
            metrics['expected_return'],
            metrics['volatility']
        )
        
        # Risk attribution
        risk_attr = RiskAttribution()
        risk_contrib = risk_attr.calculate_risk_contributions(
            self.weights,
            self.covariance,
            self.asset_names
        )
        metrics['risk_contributions'] = risk_contrib
        
        # Concentration metrics
        metrics['concentration'] = risk_attr.calculate_concentration_metrics(self.weights)
        
        # Diversification ratio
        metrics['diversification_ratio'] = risk_attr.calculate_diversification_ratio(
            self.weights,
            self.covariance
        )
        
        self.risk_metrics = metrics
        
        return metrics
    
    def calculate_tracking_error(
        self,
        benchmark_weights: np.ndarray
    ) -> Dict:
        """Calculate tracking error metrics."""
        
        if self.weights is None or self.covariance is None:
            raise ValueError("Portfolio not optimized yet")
            
        te_analyzer = TrackingErrorAnalyzer()
        
        # Ex-ante tracking error
        tracking_error = te_analyzer.calculate_ex_ante_tracking_error(
            self.weights,
            benchmark_weights,
            self.covariance
        )
        
        # Decomposition
        te_decomposition = te_analyzer.decompose_tracking_error(
            self.weights,
            benchmark_weights,
            self.covariance,
            self.asset_names
        )
        
        # Active share
        active_share = te_analyzer.calculate_active_share(
            self.weights,
            benchmark_weights
        )
        
        return {
            'tracking_error': tracking_error,
            'decomposition': te_decomposition,
            'active_share': active_share
        }
    
    def apply_black_litterman(
        self,
        market_cap_weights: np.ndarray,
        views: List[Dict],
        tau: float = 0.05
    ) -> Dict:
        """Apply Black-Litterman model."""
        
        if self.covariance is None:
            raise ValueError("Covariance not estimated yet")
            
        bl = BlackLitterman(risk_aversion=self.risk_aversion, tau=tau)
        
        # Calculate equilibrium returns
        equilibrium_returns = bl.calculate_equilibrium_returns(
            market_cap_weights,
            self.covariance
        )
        
        # Create view matrices
        P, Q, confidence = bl.create_views(self.asset_names, views)
        
        # Incorporate views
        posterior_returns, posterior_cov = bl.incorporate_views(
            equilibrium_returns,
            self.covariance,
            P, Q,
            confidence=confidence
        )
        
        # Update expected returns
        self.expected_returns = posterior_returns
        
        # Optimize with posterior
        results = bl.optimal_portfolio(
            self.covariance,
            constraints={'long_only': True}
        )
        
        self.weights = results['weights']
        
        return results
    
    def backtest_portfolio(
        self,
        returns: pd.DataFrame,
        rebalance_freq: str = 'M'
    ) -> pd.DataFrame:
        """Backtest portfolio strategy."""
        
        if self.weights is None:
            raise ValueError("Portfolio not optimized yet")
            
        # Simple backtest - assumes constant weights
        portfolio_returns = returns @ self.weights
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol
        max_dd = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        results = pd.DataFrame({
            'returns': portfolio_returns,
            'cumulative': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }, index=returns.index)
        
        return results
    
    def generate_report(self) -> str:
        """Generate portfolio report."""
        
        if self.weights is None or self.risk_metrics is None:
            raise ValueError("Portfolio not optimized yet")
            
        report = f"""
# Portfolio Report

## Asset Allocation
"""
        
        # Weights table
        weights_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': self.weights * 100
        }).round(2)
        
        report += weights_df.to_string(index=False)
        
        report += f"""

## Performance Metrics
- Expected Return: {self.risk_metrics['expected_return']*252*100:.2f}%
- Volatility: {self.risk_metrics['volatility']*np.sqrt(252)*100:.2f}%
- Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}
- VaR (95%): {self.risk_metrics['var_95']*100:.2f}%
- CVaR (95%): {self.risk_metrics['cvar_95']*100:.2f}%

## Risk Metrics
- Diversification Ratio: {self.risk_metrics['diversification_ratio']:.2f}
- Effective N Assets: {self.risk_metrics['concentration']['effective_n_assets']:.1f}
- Max Weight: {self.risk_metrics['concentration']['max_weight']*100:.1f}%
- HHI: {self.risk_metrics['concentration']['herfindahl_index']:.3f}
"""
        
        return report


def example_usage():
    """Example usage of the portfolio manager."""
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_periods = 252
    
    asset_names = [f'Asset_{i}' for i in range(n_assets)]
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            np.random.uniform(-0.0005, 0.001, n_assets),
            np.eye(n_assets) * 0.01**2 + np.ones((n_assets, n_assets)) * 0.005**2,
            n_periods
        ),
        columns=asset_names
    )
    
    # Create portfolio manager
    pm = PortfolioManager(asset_names, risk_aversion=2.0)
    
    # Estimate covariance
    pm.estimate_covariance(returns, method='ledoit_wolf')
    
    # Set expected returns
    pm.set_expected_returns(returns=returns, method='historical')
    
    # Optimize portfolio
    results = pm.optimize_portfolio(
        method='mean_variance',
        constraints={'long_only': True, 'full_investment': True}
    )
    
    # Generate report
    report = pm.generate_report()
    print(report)
    
    return pm


if __name__ == "__main__":
    example_usage()