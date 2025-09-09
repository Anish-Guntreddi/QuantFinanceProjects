# Momentum vs Value Long-Short (Cross-Sectional)

## Overview
Cross-sectional momentum vs value strategy with ranking, neutralization, and long-short portfolio construction.

## Project Structure
```
04_momentum_value_long_short/
├── cross_sectional/
│   ├── longshort.py
│   ├── factor_models.py
│   └── neutralization.py
├── backtests/
│   └── cross_sectional_backtest.ipynb
└── tests/
    └── test_longshort.py
```

## Implementation

### cross_sectional/longshort.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

@dataclass
class LongShortConfig:
    momentum_lookback: int = 20
    value_lookback: int = 60
    rebalance_frequency: int = 5
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    market_neutral: bool = True
    sector_neutral: bool = True
    max_position_size: float = 0.05
    leverage: float = 2.0

class MomentumValueLongShort:
    def __init__(self, config: LongShortConfig = LongShortConfig()):
        self.config = config
        self.factor_exposures = {}
        self.portfolio_weights = {}
        
    def calculate_momentum_score(self, prices: pd.DataFrame, 
                                lookback: Optional[int] = None) -> pd.DataFrame:
        """Calculate cross-sectional momentum scores"""
        if lookback is None:
            lookback = self.config.momentum_lookback
        
        # Calculate returns over lookback period
        returns = prices.pct_change(lookback)
        
        # Skip most recent day (reversal effect)
        returns_adjusted = prices.shift(1).pct_change(lookback - 1)
        
        # Rank cross-sectionally
        momentum_rank = returns_adjusted.rank(axis=1, pct=True)
        
        # Z-score normalization
        momentum_zscore = (returns_adjusted - returns_adjusted.mean(axis=1).values.reshape(-1, 1)) / \
                         returns_adjusted.std(axis=1).values.reshape(-1, 1)
        
        return pd.DataFrame({
            'raw': returns_adjusted,
            'rank': momentum_rank,
            'zscore': momentum_zscore
        })
    
    def calculate_value_score(self, prices: pd.DataFrame, 
                            fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-sectional value scores"""
        # Price-to-book ratio (inverse for value)
        pb_ratio = prices / fundamentals['book_value']
        value_score = 1 / pb_ratio
        
        # Earnings yield
        earnings_yield = fundamentals['earnings'] / prices
        
        # Free cash flow yield
        fcf_yield = fundamentals['free_cash_flow'] / prices
        
        # Combine value metrics
        combined_value = (
            value_score.rank(axis=1, pct=True) * 0.4 +
            earnings_yield.rank(axis=1, pct=True) * 0.3 +
            fcf_yield.rank(axis=1, pct=True) * 0.3
        )
        
        # Z-score normalization
        value_zscore = (combined_value - combined_value.mean(axis=1).values.reshape(-1, 1)) / \
                      combined_value.std(axis=1).values.reshape(-1, 1)
        
        return pd.DataFrame({
            'raw': combined_value,
            'rank': combined_value.rank(axis=1, pct=True),
            'zscore': value_zscore
        })
    
    def combine_factors(self, momentum_scores: pd.DataFrame,
                       value_scores: pd.DataFrame,
                       momentum_weight: float = 0.5) -> pd.DataFrame:
        """Combine momentum and value factors"""
        # Momentum minus value (momentum long, value short)
        combined_score = (
            momentum_scores['zscore'] * momentum_weight - 
            value_scores['zscore'] * (1 - momentum_weight)
        )
        
        # Rank combined scores
        combined_rank = combined_score.rank(axis=1, pct=True)
        
        return pd.DataFrame({
            'score': combined_score,
            'rank': combined_rank
        })
    
    def neutralize_factors(self, scores: pd.DataFrame,
                          market_caps: pd.DataFrame,
                          sectors: pd.DataFrame = None) -> pd.DataFrame:
        """Neutralize factor exposures"""
        neutralized_scores = scores.copy()
        
        if self.config.market_neutral:
            # Market cap neutralization
            for date in scores.index:
                if date not in market_caps.index:
                    continue
                
                # Log market cap
                log_mcap = np.log(market_caps.loc[date])
                
                # Regress scores on market cap
                valid_idx = ~(scores.loc[date].isna() | log_mcap.isna())
                
                if valid_idx.sum() < 10:
                    continue
                
                X = log_mcap[valid_idx].values.reshape(-1, 1)
                y = scores.loc[date, valid_idx].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Remove market cap effect
                predictions = model.predict(X)
                residuals = y - predictions
                
                neutralized_scores.loc[date, valid_idx] = residuals
        
        if self.config.sector_neutral and sectors is not None:
            # Sector neutralization
            for date in scores.index:
                if date not in sectors.index:
                    continue
                
                date_scores = neutralized_scores.loc[date]
                date_sectors = sectors.loc[date]
                
                # Neutralize within each sector
                for sector in date_sectors.unique():
                    sector_mask = date_sectors == sector
                    sector_scores = date_scores[sector_mask]
                    
                    if len(sector_scores) > 0:
                        # Demean within sector
                        sector_mean = sector_scores.mean()
                        neutralized_scores.loc[date, sector_mask] = sector_scores - sector_mean
        
        return neutralized_scores
    
    def construct_portfolio(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Construct long-short portfolio weights"""
        weights = pd.DataFrame(index=scores.index, columns=scores.columns)
        
        for date in scores.index:
            date_scores = scores.loc[date].dropna()
            
            if len(date_scores) < 10:
                continue
            
            # Determine quantile thresholds
            top_threshold = date_scores.quantile(1 - self.config.top_quantile)
            bottom_threshold = date_scores.quantile(self.config.bottom_quantile)
            
            # Long positions (top quantile)
            long_mask = date_scores >= top_threshold
            num_long = long_mask.sum()
            
            # Short positions (bottom quantile)
            short_mask = date_scores <= bottom_threshold
            num_short = short_mask.sum()
            
            # Equal weight within long and short baskets
            if num_long > 0:
                long_weight = 1.0 / num_long
                weights.loc[date, long_mask] = long_weight
            
            if num_short > 0:
                short_weight = -1.0 / num_short
                weights.loc[date, short_mask] = short_weight
            
            # Apply position size limits
            weights.loc[date] = weights.loc[date].clip(
                -self.config.max_position_size,
                self.config.max_position_size
            )
            
            # Normalize to leverage
            total_exposure = weights.loc[date].abs().sum()
            if total_exposure > 0:
                weights.loc[date] *= self.config.leverage / total_exposure
        
        return weights.fillna(0)
    
    def calculate_factor_exposures(self, weights: pd.DataFrame,
                                  factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio's factor exposures"""
        exposures = pd.DataFrame(index=weights.index, 
                                columns=factor_returns.columns)
        
        for date in weights.index:
            if date in factor_returns.index:
                # Portfolio-weighted factor exposure
                date_weights = weights.loc[date]
                date_factors = factor_returns.loc[date]
                
                for factor in factor_returns.columns:
                    factor_values = date_factors[factor]
                    exposure = (date_weights * factor_values).sum()
                    exposures.loc[date, factor] = exposure
        
        return exposures
    
    def backtest(self, prices: pd.DataFrame,
                fundamentals: pd.DataFrame,
                market_caps: pd.DataFrame = None,
                sectors: pd.DataFrame = None) -> Dict:
        """Backtest the long-short strategy"""
        results = {
            'weights': [],
            'returns': [],
            'cumulative_returns': [],
            'factor_exposures': []
        }
        
        # Calculate factors
        momentum_scores = self.calculate_momentum_score(prices)
        value_scores = self.calculate_value_score(prices, fundamentals)
        
        # Combine factors
        combined_scores = self.combine_factors(momentum_scores, value_scores)
        
        # Neutralize if required
        if self.config.market_neutral or self.config.sector_neutral:
            neutralized_scores = self.neutralize_factors(
                combined_scores['score'],
                market_caps,
                sectors
            )
        else:
            neutralized_scores = combined_scores['score']
        
        # Construct portfolio
        weights = self.construct_portfolio(neutralized_scores)
        
        # Calculate returns
        asset_returns = prices.pct_change()
        portfolio_returns = (weights.shift(1) * asset_returns).sum(axis=1)
        
        # Transaction costs (simplified)
        weight_changes = weights.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * 0.001  # 10 bps
        portfolio_returns -= transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility
        
        # Maximum drawdown
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Information ratio (if benchmark provided)
        information_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        return {
            'weights': weights,
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'avg_long_positions': (weights > 0).sum(axis=1).mean(),
            'avg_short_positions': (weights < 0).sum(axis=1).mean(),
            'turnover': weight_changes.mean()
        }
    
    def risk_analysis(self, returns: pd.Series) -> Dict:
        """Analyze risk metrics of the strategy"""
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        # Sortino ratio (assuming 0% risk-free rate)
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)
        
        # Calmar ratio
        annual_return = returns.mean() * 252
        cumulative = (1 + returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        calmar_ratio = annual_return / abs(max_dd)
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
```

### cross_sectional/factor_models.py
```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, List, Optional

class FactorModel:
    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self.factor_loadings = None
        self.factor_returns = None
        
    def estimate_factors(self, returns: pd.DataFrame, 
                        method: str = 'pca') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Estimate latent factors from returns"""
        if method == 'pca':
            return self.pca_factors(returns)
        elif method == 'fundamental':
            return self.fundamental_factors(returns)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def pca_factors(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract factors using PCA"""
        # Standardize returns
        returns_std = (returns - returns.mean()) / returns.std()
        
        # Apply PCA
        pca = PCA(n_components=self.n_factors)
        factor_returns = pca.fit_transform(returns_std.fillna(0))
        
        # Factor loadings
        loadings = pca.components_.T
        
        # Create DataFrames
        factor_returns_df = pd.DataFrame(
            factor_returns,
            index=returns.index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        loadings_df = pd.DataFrame(
            loadings,
            index=returns.columns,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        self.factor_returns = factor_returns_df
        self.factor_loadings = loadings_df
        
        return factor_returns_df, loadings_df
    
    def fama_french_factors(self, returns: pd.DataFrame,
                           market_caps: pd.DataFrame,
                           book_values: pd.DataFrame) -> pd.DataFrame:
        """Construct Fama-French style factors"""
        factors = pd.DataFrame(index=returns.index)
        
        # Market factor (excess return)
        factors['MKT'] = returns.mean(axis=1)
        
        # SMB (Small Minus Big)
        for date in returns.index:
            if date in market_caps.index:
                mcap = market_caps.loc[date]
                median_mcap = mcap.median()
                
                small_stocks = mcap <= median_mcap
                big_stocks = mcap > median_mcap
                
                small_return = returns.loc[date, small_stocks].mean()
                big_return = returns.loc[date, big_stocks].mean()
                
                factors.loc[date, 'SMB'] = small_return - big_return
        
        # HML (High Minus Low book-to-market)
        for date in returns.index:
            if date in book_values.index:
                bm_ratio = book_values.loc[date] / market_caps.loc[date]
                
                high_bm = bm_ratio >= bm_ratio.quantile(0.7)
                low_bm = bm_ratio <= bm_ratio.quantile(0.3)
                
                high_return = returns.loc[date, high_bm].mean()
                low_return = returns.loc[date, low_bm].mean()
                
                factors.loc[date, 'HML'] = high_return - low_return
        
        # Momentum factor
        momentum = returns.rolling(window=12).mean()
        for date in returns.index:
            if date in momentum.index:
                mom_values = momentum.loc[date]
                
                high_mom = mom_values >= mom_values.quantile(0.7)
                low_mom = mom_values <= mom_values.quantile(0.3)
                
                high_return = returns.loc[date, high_mom].mean()
                low_return = returns.loc[date, low_mom].mean()
                
                factors.loc[date, 'MOM'] = high_return - low_return
        
        return factors
    
    def risk_attribution(self, portfolio_returns: pd.Series,
                        factor_returns: pd.DataFrame) -> Dict:
        """Attribute portfolio risk to factors"""
        # Regression of portfolio returns on factors
        from sklearn.linear_model import LinearRegression
        
        # Align data
        aligned_dates = portfolio_returns.index.intersection(factor_returns.index)
        y = portfolio_returns.loc[aligned_dates].values
        X = factor_returns.loc[aligned_dates].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Factor contributions
        factor_contributions = {}
        for i, factor in enumerate(factor_returns.columns):
            contribution = model.coef_[i] * factor_returns[factor].std()
            factor_contributions[factor] = contribution
        
        # R-squared
        r_squared = model.score(X, y)
        
        # Specific risk (residual)
        predictions = model.predict(X)
        residuals = y - predictions
        specific_risk = np.std(residuals)
        
        return {
            'factor_contributions': factor_contributions,
            'r_squared': r_squared,
            'specific_risk': specific_risk,
            'factor_betas': dict(zip(factor_returns.columns, model.coef_))
        }
```

## Deliverables
- `cross_sectional/longshort.py`: Complete momentum vs value long-short implementation
- Cross-sectional ranking and scoring
- Market and sector neutralization
- Risk analysis and factor attribution