"""Portfolio constraints management."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import cvxpy as cp


class PortfolioConstraints:
    """Manage and validate portfolio constraints."""
    
    def __init__(self):
        self.constraints = {}
        
    def add_constraint(self, name: str, constraint_type: str, value: Union[float, Dict, List]):
        """Add a constraint to the portfolio."""
        
        valid_types = [
            'long_only', 'full_investment', 'max_leverage', 'max_position',
            'min_position', 'sector_limits', 'factor_exposure', 'turnover',
            'concentration', 'cardinality'
        ]
        
        if constraint_type not in valid_types:
            raise ValueError(f"Invalid constraint type: {constraint_type}")
            
        self.constraints[name] = {
            'type': constraint_type,
            'value': value
        }
        
    def validate_weights(self, weights: np.ndarray, asset_names: Optional[List[str]] = None) -> Dict:
        """Validate if weights satisfy all constraints."""
        
        violations = []
        
        for name, constraint in self.constraints.items():
            constraint_type = constraint['type']
            value = constraint['value']
            
            if constraint_type == 'long_only':
                if value and np.any(weights < -1e-6):
                    violations.append({
                        'constraint': name,
                        'violation': f"Negative weights found: min={weights.min():.4f}"
                    })
                    
            elif constraint_type == 'full_investment':
                weight_sum = np.sum(weights)
                if value and abs(weight_sum - 1.0) > 1e-6:
                    violations.append({
                        'constraint': name,
                        'violation': f"Weights sum to {weight_sum:.4f}, not 1.0"
                    })
                    
            elif constraint_type == 'max_leverage':
                leverage = np.sum(np.abs(weights))
                if leverage > value + 1e-6:
                    violations.append({
                        'constraint': name,
                        'violation': f"Leverage {leverage:.4f} exceeds limit {value}"
                    })
                    
            elif constraint_type == 'max_position':
                max_weight = np.max(np.abs(weights))
                if max_weight > value + 1e-6:
                    violations.append({
                        'constraint': name,
                        'violation': f"Max position {max_weight:.4f} exceeds limit {value}"
                    })
                    
            elif constraint_type == 'min_position':
                non_zero_weights = weights[np.abs(weights) > 1e-6]
                if len(non_zero_weights) > 0:
                    min_non_zero = np.min(np.abs(non_zero_weights))
                    if min_non_zero < value - 1e-6:
                        violations.append({
                            'constraint': name,
                            'violation': f"Min non-zero position {min_non_zero:.4f} below limit {value}"
                        })
                        
        return {
            'valid': len(violations) == 0,
            'violations': violations
        }
    
    def to_cvxpy(self, w: cp.Variable, n_assets: int) -> List:
        """Convert constraints to CVXPY format."""
        
        cvxpy_constraints = []
        
        for name, constraint in self.constraints.items():
            constraint_type = constraint['type']
            value = constraint['value']
            
            if constraint_type == 'long_only' and value:
                cvxpy_constraints.append(w >= 0)
                
            elif constraint_type == 'full_investment' and value:
                cvxpy_constraints.append(cp.sum(w) == 1)
                
            elif constraint_type == 'max_leverage':
                cvxpy_constraints.append(cp.sum(cp.abs(w)) <= value)
                
            elif constraint_type == 'max_position':
                cvxpy_constraints.append(w <= value)
                cvxpy_constraints.append(w >= -value)
                
            elif constraint_type == 'turnover' and 'current_weights' in value:
                current = value['current_weights']
                max_turnover = value['max_turnover']
                cvxpy_constraints.append(
                    cp.sum(cp.abs(w - current)) <= max_turnover
                )
                
            elif constraint_type == 'cardinality' and value:
                # Note: Cardinality constraints require mixed-integer programming
                # This is a simplified version
                max_assets = value
                # Add constraint that limits number of non-zero positions
                # This requires binary variables, simplified here
                pass
                
        return cvxpy_constraints
    
    def get_feasible_region(
        self,
        n_assets: int,
        n_samples: int = 1000
    ) -> np.ndarray:
        """Sample feasible portfolio weights."""
        
        feasible_weights = []
        
        for _ in range(n_samples):
            # Generate random weights
            if 'long_only' in [c['type'] for c in self.constraints.values()]:
                # Long-only portfolios
                weights = np.random.dirichlet(np.ones(n_assets))
            else:
                # Long-short portfolios
                weights = np.random.randn(n_assets)
                weights = weights / np.sum(np.abs(weights))
                
            # Check if feasible
            validation = self.validate_weights(weights)
            if validation['valid']:
                feasible_weights.append(weights)
                
        return np.array(feasible_weights) if feasible_weights else np.array([])
    
    def relax_constraints(self, relaxation_factor: float = 0.1) -> 'PortfolioConstraints':
        """Create relaxed version of constraints."""
        
        relaxed = PortfolioConstraints()
        
        for name, constraint in self.constraints.items():
            constraint_type = constraint['type']
            value = constraint['value']
            
            if constraint_type in ['max_leverage', 'max_position']:
                # Increase limits
                relaxed_value = value * (1 + relaxation_factor)
            elif constraint_type == 'min_position':
                # Decrease limits
                relaxed_value = value * (1 - relaxation_factor)
            else:
                relaxed_value = value
                
            relaxed.add_constraint(name, constraint_type, relaxed_value)
            
        return relaxed
    
    def to_dict(self) -> Dict:
        """Convert constraints to dictionary format."""
        
        result = {}
        
        for name, constraint in self.constraints.items():
            constraint_type = constraint['type']
            value = constraint['value']
            
            if constraint_type in ['long_only', 'full_investment']:
                result[constraint_type] = value
            elif constraint_type in ['max_leverage', 'max_position', 'min_position']:
                result[constraint_type] = value
            elif constraint_type == 'sector_limits':
                result['sector_limits'] = value
                
        return result
    
    @classmethod
    def from_dict(cls, constraints_dict: Dict) -> 'PortfolioConstraints':
        """Create PortfolioConstraints from dictionary."""
        
        pc = cls()
        
        for key, value in constraints_dict.items():
            pc.add_constraint(key, key, value)
            
        return pc
    
    def summary(self) -> pd.DataFrame:
        """Get summary of all constraints."""
        
        summary_data = []
        
        for name, constraint in self.constraints.items():
            summary_data.append({
                'name': name,
                'type': constraint['type'],
                'value': str(constraint['value'])
            })
            
        return pd.DataFrame(summary_data)