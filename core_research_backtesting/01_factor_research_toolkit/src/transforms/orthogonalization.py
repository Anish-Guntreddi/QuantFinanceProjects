"""Factor orthogonalization utilities"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Optional


class Orthogonalizer:
    """Orthogonalize factors to remove redundancy"""
    
    def __init__(self, method: str = 'gram-schmidt'):
        """
        Initialize orthogonalizer
        
        Parameters:
        -----------
        method : str
            Orthogonalization method ('gram-schmidt', 'pca', 'symmetric')
        """
        self.method = method
        self.transform_matrix = None
        
    def orthogonalize(
        self,
        factors: pd.DataFrame,
        priority_order: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Orthogonalize factors
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factor values (columns are different factors)
        priority_order : Optional[List[str]]
            Order of factors for Gram-Schmidt (first factor unchanged)
            
        Returns:
        --------
        pd.DataFrame
            Orthogonalized factors
        """
        if self.method == 'gram-schmidt':
            return self._gram_schmidt(factors, priority_order)
        elif self.method == 'pca':
            return self._pca_orthogonalize(factors)
        elif self.method == 'symmetric':
            return self._symmetric_orthogonalize(factors)
        else:
            raise ValueError(f"Unknown orthogonalization method: {self.method}")
    
    def _gram_schmidt(
        self,
        factors: pd.DataFrame,
        priority_order: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Gram-Schmidt orthogonalization
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factors to orthogonalize
        priority_order : Optional[List[str]]
            Priority ordering of factors
            
        Returns:
        --------
        pd.DataFrame
            Orthogonalized factors
        """
        # Order columns by priority
        if priority_order:
            ordered_cols = priority_order + [c for c in factors.columns if c not in priority_order]
            factors = factors[ordered_cols]
        
        # Initialize result
        result = pd.DataFrame(index=factors.index)
        
        # Process each factor
        for i, col in enumerate(factors.columns):
            # Start with original factor
            orthogonal = factors[col].copy()
            
            # Remove projection on all previous factors
            for j in range(i):
                prev_col = factors.columns[j]
                if prev_col in result.columns:
                    # Calculate projection
                    prev_factor = result[prev_col]
                    projection = (orthogonal @ prev_factor) / (prev_factor @ prev_factor)
                    
                    # Remove projection
                    orthogonal = orthogonal - projection * prev_factor
            
            result[col] = orthogonal
        
        return result
    
    def _pca_orthogonalize(
        self,
        factors: pd.DataFrame,
        n_components: Optional[int] = None
    ) -> pd.DataFrame:
        """
        PCA-based orthogonalization
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factors to orthogonalize
        n_components : Optional[int]
            Number of principal components to keep
            
        Returns:
        --------
        pd.DataFrame
            PCA factors
        """
        # Remove NaN values
        factors_clean = factors.dropna()
        
        if len(factors_clean) == 0:
            return pd.DataFrame(index=factors.index)
        
        # Standardize factors
        factors_std = (factors_clean - factors_clean.mean()) / factors_clean.std()
        
        # Apply PCA
        n_comp = n_components or len(factors.columns)
        pca = PCA(n_components=n_comp)
        pca_factors = pca.fit_transform(factors_std)
        
        # Create result DataFrame
        result = pd.DataFrame(
            index=factors_clean.index,
            columns=[f'PC{i+1}' for i in range(pca_factors.shape[1])],
            data=pca_factors
        )
        
        # Store transformation for later use
        self.transform_matrix = pca.components_
        self.explained_variance_ratio = pca.explained_variance_ratio_
        
        # Reindex to original index
        result = result.reindex(factors.index)
        
        return result
    
    def _symmetric_orthogonalize(
        self,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Symmetric orthogonalization (all factors treated equally)
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factors to orthogonalize
            
        Returns:
        --------
        pd.DataFrame
            Symmetrically orthogonalized factors
        """
        # Calculate correlation matrix
        corr_matrix = factors.corr()
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        
        # Construct transformation matrix
        # W = V * D^(-1/2) * V'
        D_sqrt_inv = np.diag(1.0 / np.sqrt(eigenvalues))
        transform = eigenvectors @ D_sqrt_inv @ eigenvectors.T
        
        # Apply transformation
        factors_std = (factors - factors.mean()) / factors.std()
        orthogonal = factors_std @ transform
        
        # Create result DataFrame
        result = pd.DataFrame(
            index=factors.index,
            columns=factors.columns,
            data=orthogonal
        )
        
        return result
    
    def decorrelate(
        self,
        factors: pd.DataFrame,
        target_correlation: float = 0.0
    ) -> pd.DataFrame:
        """
        Decorrelate factors to target correlation level
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factors to decorrelate
        target_correlation : float
            Target correlation (0 = uncorrelated)
            
        Returns:
        --------
        pd.DataFrame
            Decorrelated factors
        """
        if target_correlation == 0:
            # Full orthogonalization
            return self._symmetric_orthogonalize(factors)
        
        # Partial decorrelation
        current_corr = factors.corr()
        target_corr = np.eye(len(factors.columns)) * (1 - target_correlation) + \
                     target_correlation * np.ones_like(current_corr)
        
        # Find transformation to achieve target correlation
        # This is a simplified approach
        weight = target_correlation
        decorrelated = factors * (1 - weight) + \
                      self._symmetric_orthogonalize(factors) * weight
        
        return decorrelated