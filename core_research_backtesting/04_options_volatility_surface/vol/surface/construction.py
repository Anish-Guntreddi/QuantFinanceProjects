"""Volatility surface construction and management"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from typing import Dict, List, Optional, Union
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vol.models.black_scholes import BlackScholes
from vol.models.svi import SVIModel
from vol.models.ssvi import SSVIModel


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
        
        # Make a copy to avoid modifying original
        quotes = quotes.copy()
        
        # Check if we have bid/ask or just mid
        if 'bid_iv' in quotes.columns and 'ask_iv' in quotes.columns:
            # Remove quotes with wide bid-ask spreads
            quotes['spread'] = quotes['ask_iv'] - quotes['bid_iv']
            quotes = quotes[quotes['spread'] < 0.05]  # Max 5% spread
            
            # Use mid IV
            quotes['iv'] = (quotes['bid_iv'] + quotes['ask_iv']) / 2
        elif 'mid_iv' in quotes.columns:
            quotes['iv'] = quotes['mid_iv']
        elif 'iv' not in quotes.columns:
            raise ValueError("Need either bid_iv/ask_iv, mid_iv, or iv columns")
        
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
    
    def _build_ssvi_surface(self, data: pd.DataFrame) -> Dict:
        """Build surface using SSVI calibration"""
        
        ssvi = SSVIModel()
        calibration = ssvi.calibrate_surface(data)
        
        self.surface_data = {
            'model': ssvi,
            'params': calibration['params'],
            'method': 'ssvi'
        }
        
        return self.surface_data
    
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
    
    def _interpolate_svi(self, strike: float, maturity: float) -> float:
        """Interpolate using SVI models"""
        
        if 'method' in self.surface_data and self.surface_data['method'] == 'ssvi':
            # SSVI model
            model = self.surface_data['model']
            forward = self.spot * np.exp((self.rate - self.div_yield) * maturity)
            k = np.log(strike / forward)
            
            params = {
                'theta_0': model.theta_params[0],
                'theta_1': model.theta_params[1],
                'theta_2': model.theta_params[2],
                'phi_0': model.phi_params[0],
                'phi_1': model.phi_params[1],
                'rho_0': model.rho_params[0],
                'rho_1': model.rho_params[1]
            }
            
            w = model.surface(np.array([k]), np.array([maturity]), params)[0, 0]
            return np.sqrt(w / maturity)
        
        else:
            # Regular SVI per maturity
            maturities = sorted(self.surface_data.keys())
            
            # Find bracketing maturities
            if maturity <= maturities[0]:
                # Use first maturity
                T = maturities[0]
                model = self.surface_data[T]['model']
                forward = self.spot * np.exp((self.rate - self.div_yield) * T)
                k = np.log(strike / forward)
                w = model.raw_svi(np.array([k]), *model.params)[0]
                return np.sqrt(w / T)
            
            elif maturity >= maturities[-1]:
                # Use last maturity
                T = maturities[-1]
                model = self.surface_data[T]['model']
                forward = self.spot * np.exp((self.rate - self.div_yield) * T)
                k = np.log(strike / forward)
                w = model.raw_svi(np.array([k]), *model.params)[0]
                return np.sqrt(w / T)
            
            else:
                # Linear interpolation between two maturities
                for i in range(len(maturities) - 1):
                    if maturities[i] <= maturity <= maturities[i+1]:
                        T1, T2 = maturities[i], maturities[i+1]
                        
                        # Get vols from both models
                        model1 = self.surface_data[T1]['model']
                        forward1 = self.spot * np.exp((self.rate - self.div_yield) * T1)
                        k1 = np.log(strike / forward1)
                        w1 = model1.raw_svi(np.array([k1]), *model1.params)[0]
                        iv1 = np.sqrt(w1 / T1)
                        
                        model2 = self.surface_data[T2]['model']
                        forward2 = self.spot * np.exp((self.rate - self.div_yield) * T2)
                        k2 = np.log(strike / forward2)
                        w2 = model2.raw_svi(np.array([k2]), *model2.params)[0]
                        iv2 = np.sqrt(w2 / T2)
                        
                        # Linear interpolation
                        alpha = (maturity - T1) / (T2 - T1)
                        return iv1 * (1 - alpha) + iv2 * alpha
        
        return 0.20  # Default fallback
    
    def _extrapolate(self, strike: float, maturity: float) -> float:
        """Simple extrapolation for out-of-range values"""
        # For now, just return a default volatility
        return 0.20
    
    def _check_arbitrage(self, surface: Union[Dict, callable]) -> List[str]:
        """Check for arbitrage violations"""
        
        violations = []
        
        # Check calendar arbitrage
        if self._check_calendar_arbitrage(surface):
            violations.append("Calendar arbitrage detected")
        
        # Check butterfly arbitrage
        if self._check_butterfly_arbitrage(surface):
            violations.append("Butterfly arbitrage detected")
        
        return violations
    
    def _check_calendar_arbitrage(self, surface: Union[Dict, callable]) -> bool:
        """Total variance should be increasing in time"""
        
        if isinstance(surface, dict) and 'model' not in surface:
            maturities = sorted(surface.keys())
            
            if len(maturities) < 2:
                return False
            
            for i in range(len(maturities) - 1):
                T1, T2 = maturities[i], maturities[i+1]
                
                # Check ATM total variance
                atm_iv1 = surface[T1]['ivs'][len(surface[T1]['ivs'])//2]
                atm_iv2 = surface[T2]['ivs'][len(surface[T2]['ivs'])//2]
                
                if atm_iv1**2 * T1 > atm_iv2**2 * T2:
                    return True
        
        return False
    
    def _check_butterfly_arbitrage(self, surface: Union[Dict, callable]) -> bool:
        """Check convexity in strike (butterfly spread)"""
        
        if isinstance(surface, dict) and 'model' not in surface:
            for T, data in surface.items():
                if 'strikes' not in data or 'ivs' not in data:
                    continue
                    
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