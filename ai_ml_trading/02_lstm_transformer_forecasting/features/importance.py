"""
Feature Importance Analysis

This module provides various methods for analyzing feature importance in time series
forecasting models, including SHAP, permutation importance, and integrated gradients.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn("Captum not available. Install with: pip install captum")


class FeatureImportance:
    """
    Unified interface for feature importance analysis.
    """
    
    def __init__(self, model: Any, model_type: str = 'sklearn'):
        """
        Initialize feature importance analyzer.
        
        Args:
            model: Trained model
            model_type: Type of model ('sklearn', 'torch', 'xgboost', 'lightgbm')
        """
        self.model = model
        self.model_type = model_type
        self.importance_scores_ = {}
        
    def calculate_importance(self, X: Union[np.ndarray, pd.DataFrame], 
                           y: Union[np.ndarray, pd.Series],
                           method: str = 'permutation',
                           **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate feature importance using specified method.
        
        Args:
            X: Input features
            y: Target values
            method: Importance method ('permutation', 'shap', 'integrated_gradients')
            **kwargs: Method-specific arguments
            
        Returns:
            Dictionary containing importance scores
        """
        if method == 'permutation':
            return self.permutation_importance(X, y, **kwargs)
        elif method == 'shap':
            return self.shap_importance(X, y, **kwargs)
        elif method == 'integrated_gradients':
            return self.integrated_gradients_importance(X, **kwargs)
        elif method == 'built_in':
            return self.built_in_importance(**kwargs)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def permutation_importance(self, X: Union[np.ndarray, pd.DataFrame],
                             y: Union[np.ndarray, pd.Series],
                             n_repeats: int = 10,
                             random_state: int = 42,
                             scoring: str = 'neg_mean_squared_error') -> Dict[str, np.ndarray]:
        """Calculate permutation importance."""
        if self.model_type == 'torch':
            # Custom implementation for PyTorch models
            return self._torch_permutation_importance(X, y, n_repeats, random_state)
        else:
            # Use sklearn's permutation_importance
            result = permutation_importance(
                self.model, X, y, 
                n_repeats=n_repeats, 
                random_state=random_state,
                scoring=scoring
            )
            
            return {
                'importances_mean': result.importances_mean,
                'importances_std': result.importances_std,
                'importances': result.importances
            }
    
    def _torch_permutation_importance(self, X: Union[np.ndarray, pd.DataFrame],
                                    y: Union[np.ndarray, pd.Series],
                                    n_repeats: int = 10,
                                    random_state: int = 42) -> Dict[str, np.ndarray]:
        """Permutation importance for PyTorch models."""
        np.random.seed(random_state)
        
        # Convert to tensors if needed
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values)
            feature_names = X.columns
        else:
            X_tensor = torch.FloatTensor(X)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y_tensor = torch.FloatTensor(y.values)
        else:
            y_tensor = torch.FloatTensor(y)
        
        # Baseline score
        self.model.eval()
        with torch.no_grad():
            baseline_pred = self.model(X_tensor)
            if isinstance(baseline_pred, dict):
                baseline_pred = baseline_pred['predictions']
            baseline_score = torch.mean((baseline_pred.squeeze() - y_tensor) ** 2).item()
        
        n_features = X_tensor.shape[1]
        importance_scores = np.zeros((n_features, n_repeats))
        
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # Create permuted dataset
                X_permuted = X_tensor.clone()
                permutation = torch.randperm(X_permuted.shape[0])
                X_permuted[:, feature_idx] = X_permuted[permutation, feature_idx]
                
                # Calculate score with permuted feature
                with torch.no_grad():
                    permuted_pred = self.model(X_permuted)
                    if isinstance(permuted_pred, dict):
                        permuted_pred = permuted_pred['predictions']
                    permuted_score = torch.mean((permuted_pred.squeeze() - y_tensor) ** 2).item()
                
                # Importance is the increase in error
                importance_scores[feature_idx, repeat] = permuted_score - baseline_score
        
        return {
            'importances_mean': importance_scores.mean(axis=1),
            'importances_std': importance_scores.std(axis=1),
            'importances': importance_scores,
            'feature_names': feature_names
        }
    
    def shap_importance(self, X: Union[np.ndarray, pd.DataFrame],
                       y: Union[np.ndarray, pd.Series] = None,
                       background_samples: int = 100) -> Dict[str, np.ndarray]:
        """Calculate SHAP importance values."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for this method")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns
        else:
            X_array = X
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        try:
            if self.model_type == 'torch':
                # PyTorch model wrapper
                def model_predict(x):
                    self.model.eval()
                    with torch.no_grad():
                        x_tensor = torch.FloatTensor(x)
                        pred = self.model(x_tensor)
                        if isinstance(pred, dict):
                            pred = pred['predictions']
                        return pred.numpy()
                
                # Use KernelExplainer for PyTorch models
                background = shap.sample(X_array, min(background_samples, len(X_array)))
                explainer = shap.KernelExplainer(model_predict, background)
                
            elif self.model_type == 'xgboost':
                explainer = shap.TreeExplainer(self.model)
                
            elif self.model_type == 'lightgbm':
                explainer = shap.TreeExplainer(self.model)
                
            else:
                # General case - use KernelExplainer
                background = shap.sample(X_array, min(background_samples, len(X_array)))
                explainer = shap.KernelExplainer(self.model.predict, background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_array[:min(100, len(X_array))])
            
            if isinstance(shap_values, list):
                # Multi-output case
                shap_values = shap_values[0]
            
            # Calculate importance as mean absolute SHAP value
            importance_scores = np.abs(shap_values).mean(axis=0)
            
            return {
                'shap_values': shap_values,
                'importances_mean': importance_scores,
                'feature_names': feature_names
            }
            
        except Exception as e:
            warnings.warn(f"SHAP calculation failed: {e}")
            return {'importances_mean': np.zeros(X_array.shape[1])}
    
    def integrated_gradients_importance(self, X: Union[np.ndarray, pd.DataFrame],
                                      baseline: Optional[torch.Tensor] = None,
                                      n_steps: int = 50) -> Dict[str, np.ndarray]:
        """Calculate integrated gradients importance."""
        if not CAPTUM_AVAILABLE or self.model_type != 'torch':
            raise ImportError("Captum and PyTorch are required for this method")
        
        # Convert to tensor
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values)
            feature_names = X.columns
        else:
            X_tensor = torch.FloatTensor(X)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        X_tensor.requires_grad = True
        
        if baseline is None:
            baseline = torch.zeros_like(X_tensor)
        
        # Create Integrated Gradients explainer
        ig = IntegratedGradients(self.model)
        
        try:
            # Calculate attributions
            attributions = ig.attribute(X_tensor, baseline, n_steps=n_steps)
            
            # Convert to numpy and calculate mean absolute attribution
            attributions_np = attributions.detach().numpy()
            importance_scores = np.abs(attributions_np).mean(axis=0)
            
            return {
                'attributions': attributions_np,
                'importances_mean': importance_scores,
                'feature_names': feature_names
            }
            
        except Exception as e:
            warnings.warn(f"Integrated gradients calculation failed: {e}")
            return {'importances_mean': np.zeros(X_tensor.shape[1])}
    
    def built_in_importance(self) -> Dict[str, np.ndarray]:
        """Extract built-in feature importance from tree-based models."""
        if self.model_type in ['xgboost', 'lightgbm']:
            if hasattr(self.model, 'feature_importances_'):
                return {'importances_mean': self.model.feature_importances_}
        elif self.model_type == 'sklearn':
            if hasattr(self.model, 'feature_importances_'):
                return {'importances_mean': self.model.feature_importances_}
            elif hasattr(self.model, 'coef_'):
                return {'importances_mean': np.abs(self.model.coef_)}
        
        return {'importances_mean': np.array([])}
    
    def plot_importance(self, importance_dict: Dict[str, np.ndarray],
                       feature_names: Optional[List[str]] = None,
                       top_k: int = 20) -> None:
        """Plot feature importance."""
        try:
            import matplotlib.pyplot as plt
            
            importances = importance_dict['importances_mean']
            
            if feature_names is None:
                if 'feature_names' in importance_dict:
                    feature_names = importance_dict['feature_names']
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:top_k]
            
            plt.figure(figsize=(10, max(6, top_k * 0.3)))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance Score')
            plt.title('Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


class PermutationImportance:
    """
    Advanced permutation importance with time-aware permutation for time series.
    """
    
    def __init__(self, model: Any, scoring_func=None, n_repeats: int = 10):
        """
        Initialize permutation importance analyzer.
        
        Args:
            model: Trained model
            scoring_func: Scoring function (default: MSE)
            n_repeats: Number of permutation repeats
        """
        self.model = model
        self.scoring_func = scoring_func or self._default_scoring
        self.n_repeats = n_repeats
        
    def _default_scoring(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default scoring function (negative MSE)."""
        return -mean_squared_error(y_true, y_pred)
    
    def calculate_importance(self, X: pd.DataFrame, y: pd.Series,
                           time_aware: bool = True,
                           block_size: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate permutation importance.
        
        Args:
            X: Features DataFrame
            y: Target series
            time_aware: Whether to use time-aware permutation
            block_size: Size of blocks for time-aware permutation
            
        Returns:
            DataFrame with importance scores
        """
        # Baseline score
        baseline_pred = self.model.predict(X)
        baseline_score = self.scoring_func(y, baseline_pred)
        
        n_features = X.shape[1]
        importance_scores = np.zeros((n_features, self.n_repeats))
        
        for feature_idx, feature_name in enumerate(X.columns):
            for repeat in range(self.n_repeats):
                # Create permuted dataset
                X_permuted = X.copy()
                
                if time_aware and block_size:
                    # Time-aware block permutation
                    X_permuted.iloc[:, feature_idx] = self._block_permute(
                        X.iloc[:, feature_idx], block_size
                    )
                else:
                    # Standard random permutation
                    X_permuted.iloc[:, feature_idx] = np.random.permutation(
                        X.iloc[:, feature_idx]
                    )
                
                # Calculate score with permuted feature
                permuted_pred = self.model.predict(X_permuted)
                permuted_score = self.scoring_func(y, permuted_pred)
                
                # Importance is the decrease in performance
                importance_scores[feature_idx, repeat] = baseline_score - permuted_score
        
        # Create results DataFrame
        results = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': importance_scores.mean(axis=1),
            'importance_std': importance_scores.std(axis=1)
        })
        
        # Add confidence intervals
        from scipy.stats import t
        alpha = 0.05  # 95% confidence interval
        df = self.n_repeats - 1
        t_critical = t.ppf(1 - alpha/2, df)
        
        results['ci_lower'] = (results['importance_mean'] - 
                              t_critical * results['importance_std'] / np.sqrt(self.n_repeats))
        results['ci_upper'] = (results['importance_mean'] + 
                              t_critical * results['importance_std'] / np.sqrt(self.n_repeats))
        
        return results.sort_values('importance_mean', ascending=False)
    
    def _block_permute(self, series: pd.Series, block_size: int) -> pd.Series:
        """Perform block-wise permutation for time series."""
        n = len(series)
        n_blocks = n // block_size
        
        # Create blocks
        blocks = []
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            blocks.append(series.iloc[start_idx:end_idx])
        
        # Add remaining data as final block
        if n % block_size != 0:
            blocks.append(series.iloc[n_blocks * block_size:])
        
        # Randomly permute block order
        permuted_blocks = np.random.permutation(blocks)
        
        # Concatenate permuted blocks
        return pd.concat(permuted_blocks, ignore_index=True)


class ShapAnalyzer:
    """
    SHAP analysis wrapper with time series specific features.
    """
    
    def __init__(self, model: Any, model_type: str = 'sklearn'):
        """Initialize SHAP analyzer."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required")
        
        self.model = model
        self.model_type = model_type
        self.explainer = None
        
    def fit_explainer(self, X_background: pd.DataFrame, 
                     background_size: int = 100) -> None:
        """Fit SHAP explainer."""
        if isinstance(X_background, pd.DataFrame):
            X_bg = X_background.values
        else:
            X_bg = X_background
        
        # Sample background data
        if len(X_bg) > background_size:
            X_bg = shap.sample(X_bg, background_size)
        
        try:
            if self.model_type in ['xgboost', 'lightgbm']:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, X_bg)
        except Exception as e:
            warnings.warn(f"Failed to create SHAP explainer: {e}")
    
    def explain_instance(self, X: pd.DataFrame, 
                        max_samples: int = 100) -> Dict[str, Any]:
        """Explain individual predictions."""
        if self.explainer is None:
            raise ValueError("Explainer must be fitted first")
        
        # Limit samples for computational efficiency
        X_sample = X.iloc[:max_samples] if len(X) > max_samples else X
        
        try:
            shap_values = self.explainer.shap_values(X_sample.values)
            
            return {
                'shap_values': shap_values,
                'feature_names': X.columns.tolist(),
                'base_values': getattr(self.explainer, 'expected_value', None),
                'data': X_sample.values
            }
        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}")
            return {}
    
    def summary_plot(self, shap_results: Dict[str, Any]) -> None:
        """Create SHAP summary plot."""
        try:
            shap.summary_plot(
                shap_results['shap_values'],
                shap_results['data'],
                feature_names=shap_results['feature_names']
            )
        except Exception as e:
            warnings.warn(f"Failed to create summary plot: {e}")


class IntegratedGradientsAnalyzer:
    """
    Integrated Gradients analyzer for PyTorch models.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize analyzer."""
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is required")
        
        self.model = model
        self.ig = IntegratedGradients(model)
        
    def analyze_batch(self, inputs: torch.Tensor,
                     baseline: Optional[torch.Tensor] = None,
                     n_steps: int = 50) -> Dict[str, torch.Tensor]:
        """Analyze a batch of inputs."""
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        inputs.requires_grad = True
        
        try:
            attributions = self.ig.attribute(inputs, baseline, n_steps=n_steps)
            
            return {
                'attributions': attributions,
                'inputs': inputs,
                'baseline': baseline
            }
        except Exception as e:
            warnings.warn(f"Integrated gradients analysis failed: {e}")
            return {}
    
    def temporal_importance_analysis(self, sequence: torch.Tensor) -> Dict[str, Any]:
        """Analyze importance of different time steps in sequence."""
        self.model.eval()
        sequence.requires_grad = True
        
        seq_length = sequence.shape[1]
        importance_scores = []
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(sequence)
            if isinstance(baseline_output, dict):
                baseline_output = baseline_output['predictions']
        
        # Mask each time step and measure importance
        for t in range(seq_length):
            masked_sequence = sequence.clone()
            masked_sequence[:, t, :] = 0  # Zero out time step
            
            with torch.no_grad():
                masked_output = self.model(masked_sequence)
                if isinstance(masked_output, dict):
                    masked_output = masked_output['predictions']
                
                # Calculate importance as difference from baseline
                importance = torch.abs(baseline_output - masked_output).mean().item()
                importance_scores.append(importance)
        
        return {
            'time_step_importance': np.array(importance_scores),
            'most_important_steps': np.argsort(importance_scores)[::-1]
        }