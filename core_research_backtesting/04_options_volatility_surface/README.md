# Options Volatility Surface & Greeks

## Project Overview
A comprehensive options analytics framework for constructing arbitrage-free volatility surfaces, calculating Greeks, and implementing delta-hedged trading strategies with focus on volatility and skew factors.

## Implementation Guide

### Phase 1: Project Setup & Architecture

#### 1.1 Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 1.2 Required Dependencies
```python
# requirements.txt
pandas==2.1.0
numpy==1.24.0
scipy==1.11.0
yfinance==0.2.28
arch==6.2.0
QuantLib==1.31
py_vollib==1.0.1
matplotlib==3.7.0
plotly==5.17.0
scikit-learn==1.3.0
cvxpy==1.4.0
pytest==7.4.0
numba==0.58.0
joblib==1.3.0
```

#### 1.3 Directory Structure
```
04_options_volatility_surface/
├── vol/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── black_scholes.py         # BS model implementation
│   │   ├── svi.py                    # SVI calibration
│   │   ├── ssvi.py                   # SSVI (Surface SVI)
│   │   ├── sabr.py                   # SABR model
│   │   └── local_vol.py              # Dupire local volatility
│   ├── surface/
│   │   ├── __init__.py
│   │   ├── construction.py           # Surface construction
│   │   ├── interpolation.py          # Interpolation methods
│   │   ├── extrapolation.py          # Extrapolation techniques
│   │   └── arbitrage_check.py        # No-arbitrage constraints
│   ├── greeks/
│   │   ├── __init__.py
│   │   ├── analytical.py             # Analytical Greeks
│   │   ├── numerical.py              # Numerical Greeks
│   │   ├── higher_order.py           # Gamma, Vanna, Volga
│   │   └── portfolio_greeks.py       # Portfolio-level Greeks
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parser.py                 # Options data parser
│   │   ├── cleaner.py                # Data cleaning
│   │   └── quotes.py                 # Quote handling
│   └── calibration/
│       ├── __init__.py
│       ├── optimizer.py              # Calibration optimizer
│       ├── weights.py                # Weighting schemes
│       └── validation.py             # Calibration validation
├── strategies/
│   ├── __init__.py
│   ├── delta_hedge.py                # Delta hedging
│   ├── vol_trading.py                # Volatility trading
│   ├── skew_trading.py               # Skew strategies
│   └── dispersion.py                 # Dispersion trading
├── analytics/
│   ├── __init__.py
│   ├── term_structure.py             # Term structure analysis
│   ├── skew_analysis.py              # Skew metrics
│   ├── smile_dynamics.py             # Smile evolution
│   └── pnl_attribution.py            # P&L decomposition
├── backtesting/
│   ├── __init__.py
│   ├── engine.py                     # Backtesting engine
│   ├── hedging_simulator.py          # Hedging simulation
│   └── performance.py                # Performance metrics
├── tests/
│   ├── test_models.py
│   ├── test_arbitrage.py
│   ├── test_greeks.py
│   └── test_calibration.py
├── notebooks/
│   ├── surface_report.ipynb          # Surface analysis
│   ├── greeks_analysis.ipynb         # Greeks visualization
│   └── strategy_backtest.ipynb       # Strategy results
└── requirements.txt
```

### Phase 2: Core Options Models Implementation

#### 2.1 Black-Scholes Model (vol/models/black_scholes.py)
```python
import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional

class BlackScholes:
    """Black-Scholes-Merton model for European options"""
    
    @staticmethod
    def d1_d2(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def call_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """European call option price"""
        if T <= 0:
            return max(S - K, 0)
        
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        call_price = (
            S * np.exp(-q*T) * norm.cdf(d1) -
            K * np.exp(-r*T) * norm.cdf(d2)
        )
        
        return call_price
    
    @staticmethod
    def put_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """European put option price"""
        if T <= 0:
            return max(K - S, 0)
        
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        put_price = (
            K * np.exp(-r*T) * norm.cdf(-d2) -
            S * np.exp(-q*T) * norm.cdf(-d1)
        )
        
        return put_price
    
    @staticmethod
    def implied_volatility(
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        q: float = 0,
        max_iter: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson"""
        
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * option_price / S
        
        for i in range(max_iter):
            if option_type.lower() == 'call':
                price = BlackScholes.call_price(S, K, T, r, sigma, q)
                vega_val = BlackScholes.vega(S, K, T, r, sigma, q)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma, q)
                vega_val = BlackScholes.vega(S, K, T, r, sigma, q)
            
            price_diff = option_price - price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            if vega_val == 0:
                return None
            
            sigma = sigma + price_diff / vega_val
            
            # Bounds check
            if sigma <= 0:
                sigma = 0.001
            elif sigma > 5:
                sigma = 5
        
        return None
    
    # Greeks calculations
    @staticmethod
    def delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0
    ) -> float:
        """Option delta"""
        d1, _ = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            return np.exp(-q*T) * norm.cdf(d1)
        else:
            return -np.exp(-q*T) * norm.cdf(-d1)
    
    @staticmethod
    def gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """Option gamma"""
        d1, _ = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        return np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """Option vega"""
        d1, _ = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0
    ) -> float:
        """Option theta"""
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        term1 = -S * np.exp(-q*T) * norm.pdf(d1) * sigma / (2*np.sqrt(T))
        
        if option_type.lower() == 'call':
            term2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q*T) * norm.cdf(d1)
        else:
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q*T) * norm.cdf(-d1)
        
        return term1 + term2 + term3
    
    @staticmethod
    def rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0
    ) -> float:
        """Option rho"""
        _, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            return K * T * np.exp(-r*T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-r*T) * norm.cdf(-d2)
```

#### 2.2 SVI Model (vol/models/svi.py)
```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple

class SVIModel:
    """Stochastic Volatility Inspired (SVI) parameterization"""
    
    def __init__(self):
        self.params = None
        self.calibration_error = None
        
    @staticmethod
    def raw_svi(
        k: np.ndarray,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float
    ) -> np.ndarray:
        """
        Raw SVI parameterization
        w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
        
        where w = total implied variance = σ²T
        k = log-moneyness = log(K/F)
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    @staticmethod
    def natural_svi(
        k: np.ndarray,
        delta: float,
        mu: float,
        rho: float,
        omega: float,
        zeta: float
    ) -> np.ndarray:
        """
        Natural SVI parameterization (ensures no butterfly arbitrage)
        """
        # Convert to raw parameters
        a = delta + omega * zeta * (1 - rho * mu / np.sqrt(mu**2 + zeta**2))
        b = omega / np.sqrt(mu**2 + zeta**2)
        m = mu
        sigma = zeta
        
        return SVIModel.raw_svi(k, a, b, rho, m, sigma)
    
    def calibrate(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
        T: float,
        weights: Optional[np.ndarray] = None,
        param_type: str = 'raw'
    ) -> Dict:
        """
        Calibrate SVI model to market data
        
        Args:
            strikes: Strike prices
            ivs: Implied volatilities
            forward: Forward price
            T: Time to maturity
            weights: Calibration weights
            param_type: 'raw' or 'natural'
        """
        
        # Convert to log-moneyness
        k = np.log(strikes / forward)
        
        # Total variance
        w_market = ivs**2 * T
        
        if weights is None:
            # Use vega weighting
            weights = self._vega_weights(strikes, forward, T, ivs)
        
        # Initial parameter guess
        if param_type == 'raw':
            x0 = [np.min(w_market), 0.1, 0.0, 0.0, 0.1]
            bounds = [
                (0, np.max(w_market)),  # a
                (0, 1),                   # b
                (-0.999, 0.999),         # rho
                (np.min(k), np.max(k)),  # m
                (0.001, 1)               # sigma
            ]
            
            def objective(params):
                a, b, rho, m, sigma = params
                w_model = self.raw_svi(k, a, b, rho, m, sigma)
                return np.sum(weights * (w_model - w_market)**2)
        
        else:  # natural
            x0 = [0.1, 0.0, 0.0, 0.1, 0.1]
            bounds = [
                (0, 1),           # delta
                (-1, 1),          # mu
                (-0.999, 0.999),  # rho
                (0.001, 1),       # omega
                (0.001, 1)        # zeta
            ]
            
            def objective(params):
                delta, mu, rho, omega, zeta = params
                w_model = self.natural_svi(k, delta, mu, rho, omega, zeta)
                return np.sum(weights * (w_model - w_market)**2)
        
        # Add no-arbitrage constraints
        constraints = []
        if param_type == 'raw':
            constraints.append({
                'type': 'ineq',
                'fun': lambda p: self._no_butterfly_constraint(k, p)
            })
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.params = result.x
            self.calibration_error = result.fun
            
            # Calculate implied vols from calibrated model
            if param_type == 'raw':
                w_calibrated = self.raw_svi(k, *self.params)
            else:
                w_calibrated = self.natural_svi(k, *self.params)
            
            iv_calibrated = np.sqrt(w_calibrated / T)
            
            return {
                'params': self.params,
                'param_type': param_type,
                'error': self.calibration_error,
                'k': k,
                'iv_market': ivs,
                'iv_model': iv_calibrated,
                'rmse': np.sqrt(np.mean((iv_calibrated - ivs)**2))
            }
        else:
            raise ValueError(f"Calibration failed: {result.message}")
    
    def _no_butterfly_constraint(self, k: np.ndarray, params: np.ndarray) -> float:
        """Check no-butterfly arbitrage constraint"""
        a, b, rho, m, sigma = params
        
        # g(k) = (1 - rho*phi(k)/sqrt(1+phi(k)^2))^2 - (1 + phi(k)^2)/4
        # where phi(k) = (k - m) / sigma
        # Constraint: g(k) >= 0 for all k
        
        phi = (k - m) / sigma
        sqrt_term = np.sqrt(1 + phi**2)
        g = (1 - rho * phi / sqrt_term)**2 - (1 + phi**2) / 4
        
        return np.min(g)  # Should be >= 0
    
    def _vega_weights(
        self,
        strikes: np.ndarray,
        forward: float,
        T: float,
        ivs: np.ndarray
    ) -> np.ndarray:
        """Calculate vega-based weights for calibration"""
        vegas = []
        for K, iv in zip(strikes, ivs):
            vega = BlackScholes.vega(forward, K, T, 0, iv)
            vegas.append(vega)
        
        vegas = np.array(vegas)
        return vegas / np.sum(vegas)
```

#### 2.3 SSVI Model (vol/models/ssvi.py)
```python
class SSVIModel:
    """Surface SVI - consistent parameterization across expiries"""
    
    def __init__(self):
        self.theta_params = None  # ATM variance parameters
        self.phi_params = None    # Skew parameters
        
    @staticmethod
    def theta_function(
        T: np.ndarray,
        theta_0: float,
        theta_1: float,
        theta_2: float
    ) -> np.ndarray:
        """
        ATM total variance as function of time
        θ(T) = θ₀ + θ₁*T + θ₂*T²
        """
        return theta_0 + theta_1 * T + theta_2 * T**2
    
    @staticmethod
    def phi_function(
        theta: float,
        phi_0: float,
        phi_1: float
    ) -> float:
        """
        Skew function
        φ(θ) = φ₀ / θ^φ₁
        """
        return phi_0 / (theta ** phi_1)
    
    @staticmethod
    def rho_function(
        theta: float,
        rho_0: float,
        rho_1: float
    ) -> float:
        """
        Correlation function
        ρ(θ) = ρ₀ + ρ₁*θ
        """
        return np.clip(rho_0 + rho_1 * theta, -0.999, 0.999)
    
    def surface(
        self,
        k: np.ndarray,
        T: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Generate SSVI surface
        
        Args:
            k: Log-moneyness grid
            T: Time to maturity grid
            params: SSVI parameters
        """
        
        # Extract parameters
        theta_0 = params['theta_0']
        theta_1 = params['theta_1']
        theta_2 = params['theta_2']
        phi_0 = params['phi_0']
        phi_1 = params['phi_1']
        rho_0 = params['rho_0']
        rho_1 = params['rho_1']
        
        # Create meshgrid
        K, T_grid = np.meshgrid(k, T)
        
        # Calculate theta for each maturity
        theta = self.theta_function(T_grid, theta_0, theta_1, theta_2)
        
        # Calculate phi and rho
        phi = self.phi_function(theta, phi_0, phi_1)
        rho = self.rho_function(theta, rho_0, rho_1)
        
        # SSVI formula
        # w(k,T) = θ/2 * (1 + ρ*φ*k + sqrt((φ*k + ρ)² + (1-ρ²)))
        
        phi_k = phi * K
        w = theta / 2 * (
            1 + rho * phi_k +
            np.sqrt((phi_k + rho)**2 + (1 - rho**2))
        )
        
        return w
    
    def calibrate_surface(
        self,
        market_data: pd.DataFrame,
        weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calibrate SSVI to entire surface
        
        Args:
            market_data: DataFrame with columns [strike, maturity, forward, iv]
        """
        
        # Prepare data
        strikes = market_data['strike'].values
        maturities = market_data['maturity'].values
        forwards = market_data['forward'].values
        ivs = market_data['iv'].values
        
        # Convert to log-moneyness and total variance
        k = np.log(strikes / forwards)
        w_market = ivs**2 * maturities
        
        # Initial parameter guess
        x0 = [
            0.01,   # theta_0
            0.01,   # theta_1
            0.001,  # theta_2
            0.1,    # phi_0
            0.5,    # phi_1
            0.0,    # rho_0
            0.0     # rho_1
        ]
        
        # Bounds
        bounds = [
            (0.001, 0.5),    # theta_0
            (-0.1, 0.5),     # theta_1
            (-0.1, 0.1),     # theta_2
            (0.001, 1),      # phi_0
            (0, 2),          # phi_1
            (-0.999, 0.999), # rho_0
            (-1, 1)          # rho_1
        ]
        
        def objective(params):
            param_dict = {
                'theta_0': params[0],
                'theta_1': params[1],
                'theta_2': params[2],
                'phi_0': params[3],
                'phi_1': params[4],
                'rho_0': params[5],
                'rho_1': params[6]
            }
            
            # Calculate model total variance
            w_model = []
            for k_val, T_val in zip(k, maturities):
                theta = self.theta_function(T_val, params[0], params[1], params[2])
                phi = self.phi_function(theta, params[3], params[4])
                rho = self.rho_function(theta, params[5], params[6])
                
                phi_k = phi * k_val
                w = theta / 2 * (
                    1 + rho * phi_k +
                    np.sqrt((phi_k + rho)**2 + (1 - rho**2))
                )
                w_model.append(w)
            
            w_model = np.array(w_model)
            
            if weights is None:
                return np.sum((w_model - w_market)**2)
            else:
                return np.sum(weights * (w_model - w_market)**2)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            self.theta_params = result.x[:3]
            self.phi_params = result.x[3:5]
            self.rho_params = result.x[5:7]
            
            return {
                'params': result.x,
                'error': result.fun,
                'success': True
            }
        else:
            raise ValueError(f"Calibration failed: {result.message}")
```

### Phase 3: Volatility Surface Construction

#### 3.1 Surface Construction (vol/surface/construction.py)
```python
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator

class VolatilitySurface:
    """Construct and manage volatility surface"""
    
    def __init__(self, spot: float, rate: float, div_yield: float = 0):
        self.spot = spot
        self.rate = rate
        self.div_yield = div_yield
        self.surface_data = None
        self.interpolator = None
        
    def build_surface(
        self,
        market_quotes: pd.DataFrame,
        method: str = 'svi',
        interpolation: str = 'rbf'
    ) -> Dict:
        """
        Build volatility surface from market quotes
        
        Args:
            market_quotes: DataFrame with [strike, maturity, bid_iv, ask_iv, mid_iv]
            method: 'svi', 'ssvi', 'sabr', or 'local_vol'
            interpolation: 'linear', 'cubic', 'rbf'
        """
        
        # Clean data
        cleaned_data = self._clean_quotes(market_quotes)
        
        # Calculate forward prices
        cleaned_data['forward'] = self.spot * np.exp(
            (self.rate - self.div_yield) * cleaned_data['maturity']
        )
        
        # Build surface based on method
        if method == 'svi':
            surface = self._build_svi_surface(cleaned_data)
        elif method == 'ssvi':
            surface = self._build_ssvi_surface(cleaned_data)
        elif method == 'sabr':
            surface = self._build_sabr_surface(cleaned_data)
        else:
            surface = self._build_interpolated_surface(cleaned_data, interpolation)
        
        # Check arbitrage
        arbitrage_violations = self._check_arbitrage(surface)
        
        return {
            'surface': surface,
            'method': method,
            'arbitrage_violations': arbitrage_violations,
            'data': cleaned_data
        }
    
    def _clean_quotes(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter option quotes"""
        
        # Remove quotes with wide bid-ask spreads
        quotes['spread'] = quotes['ask_iv'] - quotes['bid_iv']
        quotes = quotes[quotes['spread'] < 0.05]  # Max 5% spread
        
        # Use mid IV
        quotes['iv'] = (quotes['bid_iv'] + quotes['ask_iv']) / 2
        
        # Remove extreme strikes (< 50% or > 200% of spot)
        quotes = quotes[
            (quotes['strike'] > 0.5 * self.spot) &
            (quotes['strike'] < 2.0 * self.spot)
        ]
        
        # Remove very short-dated options (< 1 day)
        quotes = quotes[quotes['maturity'] > 1/365]
        
        return quotes
    
    def _build_svi_surface(self, data: pd.DataFrame) -> Dict:
        """Build surface using SVI calibration per maturity"""
        
        surface = {}
        maturities = data['maturity'].unique()
        
        for T in sorted(maturities):
            slice_data = data[data['maturity'] == T]
            
            svi = SVIModel()
            calibration = svi.calibrate(
                strikes=slice_data['strike'].values,
                ivs=slice_data['iv'].values,
                forward=slice_data['forward'].iloc[0],
                T=T
            )
            
            surface[T] = {
                'model': svi,
                'params': calibration['params'],
                'strikes': slice_data['strike'].values,
                'ivs': calibration['iv_model']
            }
        
        self.surface_data = surface
        return surface
    
    def _build_interpolated_surface(
        self,
        data: pd.DataFrame,
        method: str
    ) -> callable:
        """Build interpolated surface"""
        
        # Prepare data points
        strikes = data['strike'].values
        maturities = data['maturity'].values
        ivs = data['iv'].values
        
        # Convert to moneyness and sqrt(T) for better interpolation
        moneyness = strikes / self.spot
        sqrt_t = np.sqrt(maturities)
        
        points = np.column_stack([moneyness, sqrt_t])
        
        if method == 'rbf':
            # Radial basis function interpolation
            self.interpolator = RBFInterpolator(
                points,
                ivs,
                kernel='multiquadric',
                epsilon=2
            )
        else:
            # Linear or cubic interpolation
            self.interpolator = lambda p: griddata(
                points,
                ivs,
                p,
                method=method,
                fill_value=np.nan
            )
        
        return self.interpolator
    
    def get_vol(
        self,
        strike: float,
        maturity: float,
        extrapolate: bool = True
    ) -> float:
        """Get implied volatility for given strike and maturity"""
        
        if self.interpolator is not None:
            moneyness = strike / self.spot
            sqrt_t = np.sqrt(maturity)
            point = np.array([[moneyness, sqrt_t]])
            
            iv = self.interpolator(point)[0]
            
            if np.isnan(iv) and extrapolate:
                # Simple flat extrapolation
                iv = self._extrapolate(strike, maturity)
            
            return iv
        
        elif self.surface_data is not None:
            # SVI-based surface
            return self._interpolate_svi(strike, maturity)
        
        else:
            raise ValueError("Surface not built yet")
    
    def _check_arbitrage(self, surface: Dict) -> List[str]:
        """Check for arbitrage violations"""
        
        violations = []
        
        # Check calendar arbitrage
        if self._check_calendar_arbitrage(surface):
            violations.append("Calendar arbitrage detected")
        
        # Check butterfly arbitrage
        if self._check_butterfly_arbitrage(surface):
            violations.append("Butterfly arbitrage detected")
        
        return violations
    
    def _check_calendar_arbitrage(self, surface: Dict) -> bool:
        """Total variance should be increasing in time"""
        
        if isinstance(surface, dict):
            maturities = sorted(surface.keys())
            
            for i in range(len(maturities) - 1):
                T1, T2 = maturities[i], maturities[i+1]
                
                # Check ATM total variance
                atm_iv1 = surface[T1]['ivs'][len(surface[T1]['ivs'])//2]
                atm_iv2 = surface[T2]['ivs'][len(surface[T2]['ivs'])//2]
                
                if atm_iv1**2 * T1 > atm_iv2**2 * T2:
                    return True
        
        return False
    
    def _check_butterfly_arbitrage(self, surface: Dict) -> bool:
        """Check convexity in strike (butterfly spread)"""
        
        if isinstance(surface, dict):
            for T, data in surface.items():
                strikes = data['strikes']
                ivs = data['ivs']
                
                if len(strikes) < 3:
                    continue
                
                # Check second derivative of call prices
                for i in range(1, len(strikes) - 1):
                    K1, K2, K3 = strikes[i-1:i+2]
                    iv1, iv2, iv3 = ivs[i-1:i+2]
                    
                    # Convert to prices
                    F = self.spot * np.exp((self.rate - self.div_yield) * T)
                    c1 = BlackScholes.call_price(F, K1, T, self.rate, iv1)
                    c2 = BlackScholes.call_price(F, K2, T, self.rate, iv2)
                    c3 = BlackScholes.call_price(F, K3, T, self.rate, iv3)
                    
                    # Butterfly condition
                    butterfly = c1 - 2*c2 + c3
                    
                    if butterfly < -1e-6:  # Small tolerance
                        return True
        
        return False
```

### Phase 4: Greeks Calculation

#### 4.1 Higher-Order Greeks (vol/greeks/higher_order.py)
```python
class HigherOrderGreeks:
    """Calculate higher-order Greeks for options"""
    
    @staticmethod
    def vanna(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Vanna: ∂²V/∂S∂σ = ∂Delta/∂σ = ∂Vega/∂S
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        vanna = -np.exp(-q*T) * norm.pdf(d1) * d2 / sigma
        
        return vanna
    
    @staticmethod
    def volga(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Volga (Vomma): ∂²V/∂σ² = ∂Vega/∂σ
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        vega = BlackScholes.vega(S, K, T, r, sigma, q)
        volga = vega * d1 * d2 / sigma
        
        return volga
    
    @staticmethod
    def charm(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0
    ) -> float:
        """
        Charm: ∂²V/∂S∂T = -∂Delta/∂T
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            charm = -q * np.exp(-q*T) * norm.cdf(d1) + np.exp(-q*T) * norm.pdf(d1) * (
                2*(r-q)*T - d2*sigma*np.sqrt(T)
            ) / (2*T*sigma*np.sqrt(T))
        else:
            charm = q * np.exp(-q*T) * norm.cdf(-d1) + np.exp(-q*T) * norm.pdf(d1) * (
                2*(r-q)*T - d2*sigma*np.sqrt(T)
            ) / (2*T*sigma*np.sqrt(T))
        
        return charm
    
    @staticmethod
    def speed(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Speed: ∂³V/∂S³ = ∂Gamma/∂S
        """
        d1, _ = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        gamma = BlackScholes.gamma(S, K, T, r, sigma, q)
        
        speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
        
        return speed
    
    @staticmethod
    def color(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Color: ∂³V/∂S²∂T = ∂Gamma/∂T
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        color = -np.exp(-q*T) * norm.pdf(d1) / (2*S*T*sigma*np.sqrt(T)) * (
            2*q*T + 1 + d1 * (
                2*(r-q)*T - d2*sigma*np.sqrt(T)
            ) / (sigma*np.sqrt(T))
        )
        
        return color
    
    @staticmethod
    def ultima(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Ultima: ∂³V/∂σ³ = ∂Volga/∂σ
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        vega = BlackScholes.vega(S, K, T, r, sigma, q)
        
        ultima = -vega / (sigma**2) * (
            d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2
        )
        
        return ultima
```

### Phase 5: Delta Hedging Strategy

#### 5.1 Delta Hedging Implementation (strategies/delta_hedge.py)
```python
class DeltaHedger:
    """Implement delta hedging strategies"""
    
    def __init__(
        self,
        rehedge_frequency: str = 'daily',
        transaction_cost: float = 0.0005,
        borrow_rate: float = 0.02
    ):
        self.rehedge_frequency = rehedge_frequency
        self.transaction_cost = transaction_cost
        self.borrow_rate = borrow_rate
        self.hedge_history = []
        
    def simulate_hedge(
        self,
        option_type: str,
        S_path: np.ndarray,
        K: float,
        T: float,
        r: float,
        sigma_initial: float,
        realized_vol: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Simulate delta hedging over price path
        
        Args:
            S_path: Stock price path
            K: Strike price
            T: Initial time to maturity
            r: Risk-free rate
            sigma_initial: Initial implied volatility
            realized_vol: Actual realized volatility path (if different from IV)
        """
        
        n_steps = len(S_path)
        dt = T / n_steps
        
        # Initialize
        portfolio_value = []
        hedge_deltas = []
        hedge_costs = []
        option_values = []
        
        # Initial setup
        S0 = S_path[0]
        
        # Calculate initial option value and delta
        if option_type.lower() == 'call':
            V0 = BlackScholes.call_price(S0, K, T, r, sigma_initial)
            delta0 = BlackScholes.delta(S0, K, T, r, sigma_initial, 'call')
        else:
            V0 = BlackScholes.put_price(S0, K, T, r, sigma_initial)
            delta0 = BlackScholes.delta(S0, K, T, r, sigma_initial, 'put')
        
        # Initial hedge: short option, buy delta shares
        shares_held = delta0
        cash_position = V0 - delta0 * S0
        
        portfolio_value.append(V0)
        hedge_deltas.append(delta0)
        option_values.append(V0)
        
        # Simulate hedging
        for i in range(1, n_steps):
            S = S_path[i]
            time_remaining = T - i * dt
            
            # Use realized vol if provided, otherwise use initial IV
            if realized_vol is not None:
                sigma = realized_vol[i]
            else:
                sigma = sigma_initial
            
            # Calculate new option value and delta
            if time_remaining > 0:
                if option_type.lower() == 'call':
                    V = BlackScholes.call_price(S, K, time_remaining, r, sigma)
                    delta = BlackScholes.delta(S, K, time_remaining, r, sigma, 'call')
                else:
                    V = BlackScholes.put_price(S, K, time_remaining, r, sigma)
                    delta = BlackScholes.delta(S, K, time_remaining, r, sigma, 'put')
            else:
                # At expiry
                if option_type.lower() == 'call':
                    V = max(S - K, 0)
                    delta = 1 if S > K else 0
                else:
                    V = max(K - S, 0)
                    delta = -1 if S < K else 0
            
            # Rehedge
            shares_to_trade = delta - shares_held
            trade_cost = abs(shares_to_trade) * S * self.transaction_cost
            
            # Update positions
            cash_position = cash_position * np.exp(r * dt) - shares_to_trade * S - trade_cost
            shares_held = delta
            
            # Portfolio value (short option + hedge)
            port_value = shares_held * S + cash_position - V
            
            # Store results
            portfolio_value.append(port_value)
            hedge_deltas.append(delta)
            hedge_costs.append(trade_cost)
            option_values.append(V)
        
        # Calculate P&L
        total_pnl = portfolio_value[-1] - portfolio_value[0]
        total_costs = sum(hedge_costs)
        
        # Analyze hedge effectiveness
        hedge_error = np.std(np.diff(portfolio_value))
        
        results = {
            'total_pnl': total_pnl,
            'total_costs': total_costs,
            'net_pnl': total_pnl - total_costs,
            'hedge_error': hedge_error,
            'portfolio_values': portfolio_value,
            'deltas': hedge_deltas,
            'option_values': option_values,
            'hedge_costs': hedge_costs,
            'final_portfolio_value': portfolio_value[-1]
        }
        
        return results
    
    def analyze_pnl(
        self,
        hedge_results: Dict,
        S_path: np.ndarray
    ) -> pd.DataFrame:
        """Decompose P&L into components"""
        
        n = len(S_path)
        pnl_components = pd.DataFrame()
        
        # Delta P&L (from stock moves)
        delta_pnl = []
        for i in range(1, n):
            delta = hedge_results['deltas'][i-1]
            price_change = S_path[i] - S_path[i-1]
            delta_pnl.append(delta * price_change)
        
        pnl_components['delta_pnl'] = [0] + delta_pnl
        
        # Gamma P&L (from convexity)
        gamma_pnl = []
        for i in range(1, n):
            price_change = S_path[i] - S_path[i-1]
            # Approximate gamma at midpoint
            S_mid = (S_path[i] + S_path[i-1]) / 2
            # Calculate gamma (would need option parameters)
            gamma = 0.01  # Placeholder
            gamma_pnl.append(0.5 * gamma * price_change**2)
        
        pnl_components['gamma_pnl'] = [0] + gamma_pnl
        
        # Theta P&L (time decay)
        theta_pnl = []
        option_values = hedge_results['option_values']
        for i in range(1, n):
            # Approximate theta from option value changes
            theta = (option_values[i] - option_values[i-1]) / n
            theta_pnl.append(theta)
        
        pnl_components['theta_pnl'] = [0] + theta_pnl
        
        # Transaction costs
        pnl_components['transaction_costs'] = hedge_results['hedge_costs']
        
        # Total P&L
        pnl_components['total_pnl'] = (
            pnl_components['delta_pnl'] +
            pnl_components['gamma_pnl'] +
            pnl_components['theta_pnl'] -
            pnl_components['transaction_costs']
        )
        
        return pnl_components
```

### Phase 6: Volatility Trading Strategies

#### 6.1 Volatility and Skew Trading (strategies/vol_trading.py)
```python
class VolatilityTrader:
    """Trade volatility using options"""
    
    def __init__(self):
        self.positions = {}
        self.trades = []
        
    def straddle_strategy(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        current_iv: float,
        forecast_rv: float,
        position_size: float = 1.0
    ) -> Dict:
        """
        Long/short straddle based on IV vs RV forecast
        """
        
        # Calculate straddle value
        call_price = BlackScholes.call_price(S, K, T, r, current_iv)
        put_price = BlackScholes.put_price(S, K, T, r, current_iv)
        straddle_price = call_price + put_price
        
        # Calculate breakeven points
        upper_breakeven = K + straddle_price
        lower_breakeven = K - straddle_price
        
        # Decision logic
        if forecast_rv > current_iv * 1.1:  # RV expected to be 10% higher
            direction = 'long'
            position = position_size
        elif forecast_rv < current_iv * 0.9:  # RV expected to be 10% lower
            direction = 'short'
            position = -position_size
        else:
            direction = 'neutral'
            position = 0
        
        # Calculate Greeks
        delta = BlackScholes.delta(S, K, T, r, current_iv, 'call') + \
                BlackScholes.delta(S, K, T, r, current_iv, 'put')
        gamma = 2 * BlackScholes.gamma(S, K, T, r, current_iv)
        vega = 2 * BlackScholes.vega(S, K, T, r, current_iv)
        theta = BlackScholes.theta(S, K, T, r, current_iv, 'call') + \
                BlackScholes.theta(S, K, T, r, current_iv, 'put')
        
        return {
            'strategy': 'straddle',
            'direction': direction,
            'position': position,
            'cost': straddle_price * position,
            'breakeven_upper': upper_breakeven,
            'breakeven_lower': lower_breakeven,
            'greeks': {
                'delta': delta * position,
                'gamma': gamma * position,
                'vega': vega * position,
                'theta': theta * position
            },
            'iv': current_iv,
            'forecast_rv': forecast_rv
        }
    
    def butterfly_spread(
        self,
        S: float,
        K_low: float,
        K_mid: float,
        K_high: float,
        T: float,
        r: float,
        sigma: float,
        position_size: float = 1.0
    ) -> Dict:
        """
        Butterfly spread for betting on low volatility
        """
        
        # Calculate option prices
        c_low = BlackScholes.call_price(S, K_low, T, r, sigma)
        c_mid = BlackScholes.call_price(S, K_mid, T, r, sigma)
        c_high = BlackScholes.call_price(S, K_high, T, r, sigma)
        
        # Butterfly: long 1 low, short 2 mid, long 1 high
        cost = c_low - 2*c_mid + c_high
        
        # Max profit at K_mid
        max_profit = K_mid - K_low - cost
        
        # Calculate net Greeks
        delta = (
            BlackScholes.delta(S, K_low, T, r, sigma, 'call') -
            2 * BlackScholes.delta(S, K_mid, T, r, sigma, 'call') +
            BlackScholes.delta(S, K_high, T, r, sigma, 'call')
        ) * position_size
        
        gamma = (
            BlackScholes.gamma(S, K_low, T, r, sigma) -
            2 * BlackScholes.gamma(S, K_mid, T, r, sigma) +
            BlackScholes.gamma(S, K_high, T, r, sigma)
        ) * position_size
        
        vega = (
            BlackScholes.vega(S, K_low, T, r, sigma) -
            2 * BlackScholes.vega(S, K_mid, T, r, sigma) +
            BlackScholes.vega(S, K_high, T, r, sigma)
        ) * position_size
        
        return {
            'strategy': 'butterfly',
            'strikes': [K_low, K_mid, K_high],
            'position': position_size,
            'cost': cost * position_size,
            'max_profit': max_profit * position_size,
            'max_loss': cost * position_size,
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega
            }
        }
    
    def skew_trade(
        self,
        surface: VolatilitySurface,
        T: float,
        percentiles: List[float] = [0.25, 0.75]
    ) -> Dict:
        """
        Trade volatility skew
        """
        
        # Get current spot
        S = surface.spot
        F = S * np.exp((surface.rate - surface.div_yield) * T)
        
        # Calculate strikes at percentiles
        K_low = F * np.exp(-2 * surface.get_vol(F*0.9, T) * np.sqrt(T) * norm.ppf(percentiles[0]))
        K_high = F * np.exp(2 * surface.get_vol(F*1.1, T) * np.sqrt(T) * norm.ppf(percentiles[1]))
        
        # Get implied vols
        iv_low = surface.get_vol(K_low, T)
        iv_high = surface.get_vol(K_high, T)
        
        # Calculate skew
        skew = (iv_low - iv_high) / (iv_low + iv_high)
        
        # Historical skew (would need historical data)
        historical_skew = 0.05  # Placeholder
        
        # Trading signal
        if skew > historical_skew * 1.5:
            # Skew too high, sell low strike vol, buy high strike vol
            signal = 'sell_skew'
        elif skew < historical_skew * 0.5:
            # Skew too low, buy low strike vol, sell high strike vol
            signal = 'buy_skew'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'current_skew': skew,
            'historical_skew': historical_skew,
            'iv_low': iv_low,
            'iv_high': iv_high,
            'K_low': K_low,
            'K_high': K_high
        }
```

### Phase 7: Backtesting Framework

#### 7.1 Options Backtester (backtesting/engine.py)
```python
class OptionsBacktester:
    """Backtest options strategies"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
        
    def backtest_delta_hedge(
        self,
        price_data: pd.DataFrame,
        option_params: Dict,
        rehedge_freq: str = 'daily'
    ) -> pd.DataFrame:
        """
        Backtest delta hedging strategy
        
        Args:
            price_data: DataFrame with columns [date, price, realized_vol]
            option_params: Dict with strike, maturity, etc.
        """
        
        results = []
        
        # Rolling window for options
        window = option_params['days_to_expiry']
        
        for i in range(0, len(price_data) - window, window):
            # Get price path for this option's lifetime
            S_path = price_data['price'].iloc[i:i+window].values
            
            # Get realized vol
            if 'realized_vol' in price_data.columns:
                rv_path = price_data['realized_vol'].iloc[i:i+window].values
            else:
                # Calculate realized vol
                returns = np.log(S_path[1:] / S_path[:-1])
                rv_path = pd.Series(returns).rolling(20).std() * np.sqrt(252)
                rv_path = rv_path.fillna(method='bfill').values
            
            # Initial IV (could be from surface)
            initial_iv = rv_path[0] * 1.1  # Assume IV premium
            
            # Run hedge simulation
            hedger = DeltaHedger(
                rehedge_frequency=rehedge_freq,
                transaction_cost=self.transaction_cost
            )
            
            hedge_result = hedger.simulate_hedge(
                option_type=option_params['type'],
                S_path=S_path,
                K=option_params['strike'],
                T=window/252,
                r=option_params['rate'],
                sigma_initial=initial_iv,
                realized_vol=rv_path
            )
            
            results.append({
                'start_date': price_data.index[i],
                'end_date': price_data.index[i+window-1],
                'initial_spot': S_path[0],
                'final_spot': S_path[-1],
                'initial_iv': initial_iv,
                'avg_rv': np.mean(rv_path),
                'pnl': hedge_result['net_pnl'],
                'hedge_error': hedge_result['hedge_error'],
                'total_costs': hedge_result['total_costs']
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        
        if self.results is None:
            raise ValueError("Run backtest first")
        
        # Calculate returns
        self.results['returns'] = self.results['pnl'] / self.initial_capital
        
        # Metrics
        total_return = self.results['returns'].sum()
        avg_return = self.results['returns'].mean()
        volatility = self.results['returns'].std()
        sharpe = avg_return / volatility * np.sqrt(252/30)  # Monthly Sharpe
        
        # Win rate
        win_rate = (self.results['pnl'] > 0).mean()
        
        # Average hedge error
        avg_hedge_error = self.results['hedge_error'].mean()
        
        # Costs as % of P&L
        cost_ratio = self.results['total_costs'].sum() / abs(self.results['pnl'].sum())
        
        return {
            'total_return': total_return,
            'annualized_return': avg_return * 12,
            'volatility': volatility * np.sqrt(12),
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_hedge_error': avg_hedge_error,
            'cost_ratio': cost_ratio,
            'num_trades': len(self.results)
        }
```

### Phase 8: Testing Framework

#### 8.1 Arbitrage Tests (tests/test_arbitrage.py)
```python
import pytest
import numpy as np
from vol.surface.arbitrage_check import ArbitrageChecker

class TestArbitrage:
    
    def test_calendar_spread_arbitrage(self):
        """Test calendar spread arbitrage detection"""
        
        checker = ArbitrageChecker()
        
        # Create test data with calendar arbitrage
        ivs_t1 = np.array([0.20, 0.18, 0.19])  # 1 month
        ivs_t2 = np.array([0.15, 0.14, 0.15])  # 3 months (violation)
        
        T1, T2 = 1/12, 3/12
        
        # Total variance should increase with time
        tv1 = ivs_t1**2 * T1
        tv2 = ivs_t2**2 * T2
        
        assert checker.check_calendar_arbitrage(tv1, tv2, T1, T2) == True
        
    def test_butterfly_arbitrage(self):
        """Test butterfly spread arbitrage detection"""
        
        checker = ArbitrageChecker()
        
        # Create test strikes and prices
        strikes = np.array([90, 100, 110])
        
        # Prices that violate butterfly (convexity)
        call_prices = np.array([15, 8, 6])  # Middle too low
        
        assert checker.check_butterfly_arbitrage(strikes, call_prices) == True
        
        # Valid butterfly
        call_prices_valid = np.array([15, 8, 2])
        assert checker.check_butterfly_arbitrage(strikes, call_prices_valid) == False
        
    def test_svi_no_arbitrage(self):
        """Test SVI parameter constraints"""
        
        from vol.models.svi import SVIModel
        
        svi = SVIModel()
        
        # Parameters that could cause arbitrage
        k = np.linspace(-0.5, 0.5, 100)
        
        # Good parameters
        good_params = [0.04, 0.1, 0.1, 0.0, 0.1]  # a, b, rho, m, sigma
        w_good = svi.raw_svi(k, *good_params)
        
        # Check positive total variance
        assert np.all(w_good > 0)
        
        # Check convexity (no butterfly)
        assert np.all(np.diff(np.diff(w_good)) > -1e-10)
```

### Phase 9: Example Usage

#### 9.1 Complete Surface Analysis (examples/surface_analysis.py)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_vol_surface():
    """Complete volatility surface analysis"""
    
    # 1. Load market data
    print("Loading market data...")
    market_data = pd.read_csv('data/option_quotes.csv')
    
    # 2. Build surface
    print("Building volatility surface...")
    surface = VolatilitySurface(
        spot=100,
        rate=0.05,
        div_yield=0.02
    )
    
    surface_result = surface.build_surface(
        market_data,
        method='ssvi',
        interpolation='rbf'
    )
    
    # 3. Check arbitrage
    print("Checking arbitrage conditions...")
    violations = surface_result['arbitrage_violations']
    if violations:
        print(f"Warning: {violations}")
    else:
        print("No arbitrage violations detected")
    
    # 4. Visualize surface
    print("Generating surface plots...")
    
    # Create grid
    strikes = np.linspace(80, 120, 50)
    maturities = np.linspace(0.1, 2, 50)
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    
    # Calculate IVs
    iv_grid = np.zeros_like(K_grid)
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            iv_grid[i, j] = surface.get_vol(strikes[j], maturities[i])
    
    # 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        K_grid, T_grid, iv_grid,
        cmap='viridis',
        alpha=0.8
    )
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Volatility Surface')
    
    plt.colorbar(surf)
    plt.savefig('reports/vol_surface.png')
    
    # 5. Term structure analysis
    print("Analyzing term structure...")
    
    # ATM term structure
    atm_vols = []
    terms = np.linspace(0.1, 2, 20)
    
    for T in terms:
        atm_vol = surface.get_vol(surface.spot, T)
        atm_vols.append(atm_vol)
    
    plt.figure(figsize=(10, 6))
    plt.plot(terms, atm_vols, 'b-', linewidth=2)
    plt.xlabel('Maturity (years)')
    plt.ylabel('ATM Implied Volatility')
    plt.title('ATM Volatility Term Structure')
    plt.grid(True)
    plt.savefig('reports/term_structure.png')
    
    # 6. Skew analysis
    print("Analyzing volatility skew...")
    
    # Calculate skew for different maturities
    maturities_skew = [0.25, 0.5, 1.0]
    
    plt.figure(figsize=(10, 6))
    
    for T in maturities_skew:
        moneyness = np.linspace(0.8, 1.2, 50)
        strikes = surface.spot * moneyness
        ivs = [surface.get_vol(K, T) for K in strikes]
        
        plt.plot(moneyness, ivs, label=f'T={T}y')
    
    plt.xlabel('Moneyness (K/S)')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Smile')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/vol_smile.png')
    
    # 7. Greeks surface
    print("Calculating Greeks surface...")
    
    # Vega surface
    vega_grid = np.zeros_like(K_grid)
    
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            vega_grid[i, j] = BlackScholes.vega(
                surface.spot,
                strikes[j],
                maturities[i],
                surface.rate,
                iv_grid[i, j],
                surface.div_yield
            )
    
    plt.figure(figsize=(10, 8))
    plt.contourf(K_grid, T_grid, vega_grid, levels=20, cmap='RdYlBu')
    plt.colorbar(label='Vega')
    plt.xlabel('Strike')
    plt.ylabel('Maturity')
    plt.title('Vega Surface')
    plt.savefig('reports/vega_surface.png')
    
    print("Analysis complete. Reports saved to reports/")
    
    return surface

def run_delta_hedge_backtest():
    """Run delta hedging backtest"""
    
    print("Running delta hedge backtest...")
    
    # Generate sample price data
    np.random.seed(42)
    n_days = 252
    S0 = 100
    mu = 0.10
    sigma = 0.20
    
    dt = 1/252
    prices = [S0]
    
    for _ in range(n_days-1):
        dW = np.random.randn() * np.sqrt(dt)
        S_new = prices[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
        prices.append(S_new)
    
    price_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_days, freq='B'),
        'price': prices
    })
    
    # Calculate realized vol
    returns = np.log(price_data['price'] / price_data['price'].shift(1))
    price_data['realized_vol'] = returns.rolling(20).std() * np.sqrt(252)
    price_data['realized_vol'] = price_data['realized_vol'].fillna(sigma)
    
    # Backtest parameters
    option_params = {
        'type': 'call',
        'strike': 100,
        'days_to_expiry': 30,
        'rate': 0.05
    }
    
    # Run backtest
    backtester = OptionsBacktester(
        initial_capital=100000,
        transaction_cost=0.001
    )
    
    results = backtester.backtest_delta_hedge(
        price_data,
        option_params,
        rehedge_freq='daily'
    )
    
    # Calculate metrics
    metrics = backtester.calculate_metrics()
    
    print("\nBacktest Results:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot P&L distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['pnl'], bins=30, edgecolor='black')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.title('Delta Hedging P&L Distribution')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('reports/hedge_pnl_dist.png')
    
    return results

if __name__ == "__main__":
    # Run analysis
    surface = analyze_vol_surface()
    hedge_results = run_delta_hedge_backtest()
```

## Testing & Validation Checklist

- [ ] Black-Scholes prices match benchmark values
- [ ] Implied volatility calculation converges
- [ ] SVI calibration produces valid parameters
- [ ] No-arbitrage constraints are satisfied
- [ ] Greeks calculations are numerically stable
- [ ] Delta hedging reduces portfolio volatility
- [ ] Surface interpolation is smooth
- [ ] Butterfly and calendar spreads are arbitrage-free
- [ ] Higher-order Greeks match finite difference approximations
- [ ] Backtest results are reproducible

## Performance Benchmarks

1. **IV Calculation**
   - Speed: < 1ms per option
   - Accuracy: Within 0.0001 of true IV

2. **Surface Calibration**
   - SVI: < 100ms per maturity slice
   - SSVI: < 1s for entire surface

3. **Greeks Calculation**
   - Analytical: < 0.1ms per Greek
   - Portfolio Greeks: < 10ms for 100 options

4. **Delta Hedging**
   - Hedge effectiveness: > 90% variance reduction
   - Transaction costs: < 10% of gross P&L

## Next Steps

1. Implement American option pricing (binomial/Monte Carlo)
2. Add exotic options support (barriers, Asians)
3. Develop jump-diffusion models
4. Build real-time Greeks monitoring
5. Create volatility forecasting models
6. Implement more sophisticated hedging strategies