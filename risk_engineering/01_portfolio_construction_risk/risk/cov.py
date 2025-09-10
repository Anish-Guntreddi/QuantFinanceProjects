"""Base covariance estimation framework."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import linalg


class CovarianceEstimator(ABC):
    """Base class for covariance estimation."""
    
    def __init__(self, min_periods: int = 20):
        self.min_periods = min_periods
        self.covariance_ = None
        self.mean_ = None
        
    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> 'CovarianceEstimator':
        """Fit covariance model."""
        pass
    
    def predict(self) -> np.ndarray:
        """Return estimated covariance matrix."""
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
        return self.covariance_
    
    def get_correlation(self) -> np.ndarray:
        """Get correlation matrix from covariance."""
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
            
        std = np.sqrt(np.diag(self.covariance_))
        corr = self.covariance_ / np.outer(std, std)
        return corr
    
    def is_positive_definite(self) -> bool:
        """Check if covariance matrix is positive definite."""
        if self.covariance_ is None:
            return False
            
        try:
            np.linalg.cholesky(self.covariance_)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def regularize(self, epsilon: float = 1e-8):
        """Regularize covariance matrix to ensure positive definiteness."""
        if self.covariance_ is None:
            return
            
        # Add small value to diagonal
        n = self.covariance_.shape[0]
        self.covariance_ += epsilon * np.eye(n)
        
        # Ensure symmetry
        self.covariance_ = (self.covariance_ + self.covariance_.T) / 2
        
    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics from covariance."""
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
            
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)
        
        # Condition number
        condition_number = eigenvalues.max() / eigenvalues.min()
        
        # Effective rank
        explained_variance = eigenvalues / eigenvalues.sum()
        effective_rank = np.exp(-np.sum(explained_variance * np.log(explained_variance + 1e-10)))
        
        return {
            'condition_number': condition_number,
            'effective_rank': effective_rank,
            'max_eigenvalue': eigenvalues.max(),
            'min_eigenvalue': eigenvalues.min(),
            'trace': np.trace(self.covariance_),
            'determinant': np.linalg.det(self.covariance_)
        }