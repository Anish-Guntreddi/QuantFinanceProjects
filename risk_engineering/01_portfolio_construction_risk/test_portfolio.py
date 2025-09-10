"""Test script for portfolio construction and risk management."""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import PortfolioManager
from risk.models import LedoitWolfCovariance, EWMACovariance
from opt import KellyCriterion, RiskParityOptimizer
from risk.metrics import ValueAtRisk, RiskAttribution


def generate_test_data(n_assets=5, n_periods=252):
    """Generate test market data."""
    
    np.random.seed(42)
    
    # Asset names
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'][:n_assets]
    
    # Generate correlated returns
    mean_returns = np.array([0.0008, 0.0007, 0.0006, 0.0009, 0.0005])[:n_assets]
    volatilities = np.array([0.02, 0.018, 0.016, 0.022, 0.025])[:n_assets]
    
    # Correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation[i, j] = correlation[j, i] = 0.3 + np.random.uniform(-0.2, 0.3)
    
    # Covariance matrix
    covariance = np.outer(volatilities, volatilities) * correlation
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, covariance, n_periods)
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    return returns_df, asset_names


def test_covariance_estimation():
    """Test covariance estimation methods."""
    
    print("\n" + "="*60)
    print("Testing Covariance Estimation")
    print("="*60)
    
    returns_df, asset_names = generate_test_data()
    
    # Test Ledoit-Wolf
    print("\n1. Ledoit-Wolf Shrinkage:")
    lw = LedoitWolfCovariance(shrinkage_target='identity')
    lw.fit(returns_df)
    cov_lw = lw.predict()
    print(f"   Shrinkage parameter: {lw.shrinkage_:.4f}")
    print(f"   Condition number: {lw.get_risk_metrics()['condition_number']:.2f}")
    print(f"   Is positive definite: {lw.is_positive_definite()}")
    
    # Test EWMA
    print("\n2. EWMA Covariance:")
    ewma = EWMACovariance(lambda_param=0.94)
    ewma.fit(returns_df)
    cov_ewma = ewma.predict()
    print(f"   Trace: {np.trace(cov_ewma):.6f}")
    print(f"   Max eigenvalue: {ewma.get_risk_metrics()['max_eigenvalue']:.6f}")
    
    return cov_lw


def test_portfolio_optimization():
    """Test portfolio optimization methods."""
    
    print("\n" + "="*60)
    print("Testing Portfolio Optimization")
    print("="*60)
    
    returns_df, asset_names = generate_test_data()
    
    # Create portfolio manager
    pm = PortfolioManager(asset_names, risk_aversion=2.0)
    pm.estimate_covariance(returns_df, method='ledoit_wolf')
    pm.set_expected_returns(returns=returns_df, method='historical')
    
    # Test Mean-Variance
    print("\n1. Mean-Variance Optimization:")
    results_mv = pm.optimize_portfolio(
        method='mean_variance',
        constraints={'long_only': True, 'full_investment': True}
    )
    print(f"   Expected Return: {results_mv['expected_return']*252*100:.2f}%")
    print(f"   Volatility: {results_mv['volatility']*np.sqrt(252)*100:.2f}%")
    print(f"   Sharpe Ratio: {results_mv['sharpe_ratio']:.2f}")
    
    # Test Risk Parity
    print("\n2. Risk Parity Optimization:")
    rp = RiskParityOptimizer(method='scipy')
    results_rp = rp.optimize(pm.covariance)
    print(f"   Volatility: {results_rp['volatility']*np.sqrt(252)*100:.2f}%")
    print(f"   Convergence error: {results_rp['convergence_error']:.2e}")
    print(f"   Risk contributions std: {np.std(results_rp['risk_contributions']):.4f}")
    
    # Test Kelly Criterion
    print("\n3. Kelly Criterion:")
    kelly = KellyCriterion(max_leverage=1.0, use_half_kelly=True)
    kelly_weights = kelly.calculate_kelly_portfolio(
        pm.expected_returns,
        pm.covariance
    )
    growth_rate = kelly.calculate_growth_rate(
        kelly_weights,
        pm.expected_returns,
        pm.covariance
    )
    print(f"   Expected growth rate: {growth_rate*252*100:.2f}%")
    print(f"   Total leverage: {np.abs(kelly_weights).sum():.2f}")
    
    return results_mv


def test_risk_metrics():
    """Test risk metrics calculations."""
    
    print("\n" + "="*60)
    print("Testing Risk Metrics")
    print("="*60)
    
    returns_df, asset_names = generate_test_data()
    
    # Create portfolio
    pm = PortfolioManager(asset_names)
    pm.estimate_covariance(returns_df, method='ledoit_wolf')
    pm.set_expected_returns(returns=returns_df)
    results = pm.optimize_portfolio(method='min_variance', constraints={'long_only': True})
    
    # Test VaR
    print("\n1. Value at Risk:")
    var_calc = ValueAtRisk(confidence_level=0.95)
    var_95 = var_calc.calculate_portfolio_var(
        pm.weights,
        pm.expected_returns,
        pm.covariance,
        method='parametric'
    )
    print(f"   VaR (95%): {var_95*100:.2f}%")
    
    # Test Risk Attribution
    print("\n2. Risk Attribution:")
    risk_attr = RiskAttribution()
    risk_contrib = risk_attr.calculate_risk_contributions(
        pm.weights,
        pm.covariance,
        asset_names
    )
    print("\n   Risk Contributions:")
    print(risk_contrib[['Asset', 'Weight', 'Pct_Contribution']].round(2).to_string(index=False))
    
    # Test Concentration Metrics
    print("\n3. Concentration Metrics:")
    concentration = risk_attr.calculate_concentration_metrics(pm.weights)
    print(f"   Effective N: {concentration['effective_n_assets']:.1f}")
    print(f"   HHI: {concentration['herfindahl_index']:.3f}")
    print(f"   Max weight: {concentration['max_weight']*100:.1f}%")
    
    return risk_contrib


def test_black_litterman():
    """Test Black-Litterman model."""
    
    print("\n" + "="*60)
    print("Testing Black-Litterman Model")
    print("="*60)
    
    returns_df, asset_names = generate_test_data()
    
    # Create portfolio manager
    pm = PortfolioManager(asset_names)
    pm.estimate_covariance(returns_df, method='ledoit_wolf')
    
    # Market cap weights (example)
    market_cap_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Define views
    views = [
        {
            'assets': ['AAPL', 'MSFT'],
            'weights': [1, -1],  # AAPL will outperform MSFT
            'return': 0.02,  # by 2%
            'confidence': 0.8
        },
        {
            'assets': ['GOOGL'],
            'weights': [1],  # GOOGL absolute return
            'return': 0.15,  # 15% annual
            'confidence': 0.6
        }
    ]
    
    # Apply Black-Litterman
    print("\nApplying views:")
    for i, view in enumerate(views):
        print(f"   View {i+1}: {view['assets']} with return {view['return']*100:.1f}%")
    
    results = pm.apply_black_litterman(market_cap_weights, views)
    
    print("\nOptimal Portfolio:")
    weights_df = pd.DataFrame({
        'Asset': asset_names,
        'Market Weight': market_cap_weights * 100,
        'BL Weight': results['weights'] * 100
    }).round(2)
    print(weights_df.to_string(index=False))
    
    return results


def main():
    """Run all tests."""
    
    print("\n" + "="*60)
    print("PORTFOLIO CONSTRUCTION & RISK MANAGEMENT TEST SUITE")
    print("="*60)
    
    # Run tests
    test_covariance_estimation()
    test_portfolio_optimization()
    test_risk_metrics()
    test_black_litterman()
    
    # Final portfolio example
    print("\n" + "="*60)
    print("Complete Portfolio Example")
    print("="*60)
    
    returns_df, asset_names = generate_test_data()
    
    pm = PortfolioManager(asset_names, risk_aversion=2.0)
    pm.estimate_covariance(returns_df, method='ledoit_wolf')
    pm.set_expected_returns(returns=returns_df, method='exponential')
    
    results = pm.optimize_portfolio(
        method='max_sharpe',
        constraints={'long_only': True}
    )
    
    print(pm.generate_report())
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()