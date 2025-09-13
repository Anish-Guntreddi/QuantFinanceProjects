"""
Model Interpretability Tools

This module provides comprehensive interpretability analysis for time series
forecasting models, including attention analysis, saliency maps, and local explanations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import warnings

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelInterpreter:
    """
    Unified interface for model interpretability analysis.
    """
    
    def __init__(self, model: Any, model_type: str = 'torch'):
        """
        Initialize model interpreter.
        
        Args:
            model: Trained model
            model_type: Type of model ('torch', 'sklearn', 'xgboost')
        """
        self.model = model
        self.model_type = model_type
        
        # Initialize specialized analyzers based on model type
        if model_type == 'torch' and hasattr(model, 'forward'):
            self.attention_analyzer = AttentionAnalyzer(model)
            self.saliency_analyzer = SaliencyAnalyzer(model)
        
    def comprehensive_analysis(self, X: Union[pd.DataFrame, torch.Tensor],
                             feature_names: Optional[List[str]] = None,
                             sample_size: int = 100) -> Dict[str, Any]:
        """
        Perform comprehensive interpretability analysis.
        
        Args:
            X: Input data
            feature_names: Names of features
            sample_size: Number of samples to analyze
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {}
        
        # Limit sample size for computational efficiency
        if len(X) > sample_size:
            if isinstance(X, pd.DataFrame):
                X_sample = X.iloc[:sample_size]
            else:
                X_sample = X[:sample_size]
        else:
            X_sample = X
        
        # Attention analysis for transformer-like models
        if self.model_type == 'torch' and hasattr(self.model, 'attention'):
            try:
                results['attention_analysis'] = self.attention_analyzer.analyze_attention_patterns(X_sample)
            except Exception as e:
                warnings.warn(f"Attention analysis failed: {e}")
        
        # Saliency analysis for neural networks
        if self.model_type == 'torch':
            try:
                results['saliency_analysis'] = self.saliency_analyzer.compute_saliency_maps(X_sample)
            except Exception as e:
                warnings.warn(f"Saliency analysis failed: {e}")
        
        # Local explanations using LIME
        if LIME_AVAILABLE and isinstance(X, pd.DataFrame):
            try:
                lime_explainer = LimeExplainer(self.model, self.model_type)
                results['lime_explanations'] = lime_explainer.explain_instances(X_sample)
            except Exception as e:
                warnings.warn(f"LIME analysis failed: {e}")
        
        # Surrogate model analysis
        try:
            surrogate = LocalSurrogateModel(self.model, self.model_type)
            results['surrogate_analysis'] = surrogate.fit_surrogate(X_sample)
        except Exception as e:
            warnings.warn(f"Surrogate model analysis failed: {e}")
        
        return results
    
    def generate_report(self, analysis_results: Dict[str, Any],
                       output_file: Optional[str] = None) -> str:
        """Generate interpretability report."""
        report_lines = []
        report_lines.append("# Model Interpretability Report")
        report_lines.append("")
        
        # Attention analysis
        if 'attention_analysis' in analysis_results:
            report_lines.append("## Attention Analysis")
            attention_results = analysis_results['attention_analysis']
            
            if 'attention_entropy' in attention_results:
                avg_entropy = np.mean(attention_results['attention_entropy'])
                report_lines.append(f"- Average attention entropy: {avg_entropy:.4f}")
            
            if 'head_importance' in attention_results:
                top_heads = np.argsort(attention_results['head_importance'])[-3:]
                report_lines.append(f"- Most important attention heads: {top_heads.tolist()}")
            
            report_lines.append("")
        
        # Saliency analysis
        if 'saliency_analysis' in analysis_results:
            report_lines.append("## Saliency Analysis")
            saliency_results = analysis_results['saliency_analysis']
            
            if 'feature_saliency' in saliency_results:
                top_features = np.argsort(saliency_results['feature_saliency'])[-5:]
                report_lines.append(f"- Most salient features: {top_features.tolist()}")
            
            if 'temporal_saliency' in saliency_results:
                avg_temporal = np.mean(saliency_results['temporal_saliency'], axis=0)
                most_important_time = np.argmax(avg_temporal)
                report_lines.append(f"- Most important time step: {most_important_time}")
            
            report_lines.append("")
        
        # LIME analysis
        if 'lime_explanations' in analysis_results:
            report_lines.append("## Local Explanations (LIME)")
            lime_results = analysis_results['lime_explanations']
            
            if 'feature_importance_stats' in lime_results:
                stats = lime_results['feature_importance_stats']
                report_lines.append(f"- Average feature importance range: {stats['mean_range']:.4f}")
                report_lines.append(f"- Most consistently important features: {stats['top_features']}")
            
            report_lines.append("")
        
        # Surrogate model analysis
        if 'surrogate_analysis' in analysis_results:
            report_lines.append("## Surrogate Model Analysis")
            surrogate_results = analysis_results['surrogate_analysis']
            
            if 'fidelity_score' in surrogate_results:
                report_lines.append(f"- Model fidelity (R² with surrogate): {surrogate_results['fidelity_score']:.4f}")
            
            if 'feature_importance' in surrogate_results:
                top_features = surrogate_results['feature_importance'][:5]
                report_lines.append(f"- Top 5 important features (surrogate): {top_features}")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report


class AttentionAnalyzer:
    """
    Analyzer for attention mechanisms in neural networks.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize attention analyzer."""
        self.model = model
        
    def analyze_attention_patterns(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze attention patterns in the model.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Dictionary containing attention analysis results
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            # Forward pass to get attention weights
            outputs = self.model(inputs, return_attention=True)
            
            if 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights']
                
                # Attention entropy (measure of attention concentration)
                results['attention_entropy'] = self._calculate_attention_entropy(attention_weights)
                
                # Attention head analysis (for multi-head attention)
                if len(attention_weights.shape) == 4:  # (batch, heads, seq, seq)
                    results['head_importance'] = self._analyze_attention_heads(attention_weights)
                    results['head_diversity'] = self._calculate_head_diversity(attention_weights)
                
                # Temporal attention patterns
                results['temporal_patterns'] = self._analyze_temporal_attention(attention_weights)
                
                # Attention rollout (cumulative attention)
                results['attention_rollout'] = self._compute_attention_rollout(attention_weights)
        
        return results
    
    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Calculate entropy of attention distributions."""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_weights = attention_weights + eps
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        
        return entropy.cpu().numpy()
    
    def _analyze_attention_heads(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Analyze importance of different attention heads."""
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        head_importance = []
        for head in range(n_heads):
            head_attention = attention_weights[:, head, :, :]
            
            # Calculate head importance as variance of attention weights
            importance = torch.var(head_attention).item()
            head_importance.append(importance)
        
        return np.array(head_importance)
    
    def _calculate_head_diversity(self, attention_weights: torch.Tensor) -> float:
        """Calculate diversity between attention heads."""
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Flatten attention matrices for each head
        heads_flattened = attention_weights.view(batch_size, n_heads, -1)
        
        # Calculate pairwise correlations between heads
        correlations = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                head_i = heads_flattened[:, i, :]
                head_j = heads_flattened[:, j, :]
                
                # Calculate correlation coefficient
                correlation = torch.corrcoef(torch.stack([head_i.mean(0), head_j.mean(0)]))[0, 1]
                correlations.append(correlation.item())
        
        # Diversity is 1 - average correlation
        avg_correlation = np.mean(correlations) if correlations else 0
        return 1 - avg_correlation
    
    def _analyze_temporal_attention(self, attention_weights: torch.Tensor) -> Dict[str, np.ndarray]:
        """Analyze temporal patterns in attention."""
        # Average attention across batch and heads (if multi-head)
        if len(attention_weights.shape) == 4:  # (batch, heads, seq, seq)
            avg_attention = attention_weights.mean(dim=(0, 1))
        else:  # (batch, seq, seq)
            avg_attention = attention_weights.mean(dim=0)
        
        # Attention to past vs future
        seq_len = avg_attention.shape[0]
        past_attention = torch.triu(avg_attention, diagonal=1).sum(dim=1)  # Upper triangle
        future_attention = torch.tril(avg_attention, diagonal=-1).sum(dim=1)  # Lower triangle
        
        return {
            'attention_matrix': avg_attention.cpu().numpy(),
            'past_attention': past_attention.cpu().numpy(),
            'future_attention': future_attention.cpu().numpy()
        }
    
    def _compute_attention_rollout(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Compute attention rollout (cumulative attention flow)."""
        # For simplicity, we'll compute rollout for single-head or average multi-head
        if len(attention_weights.shape) == 4:  # Multi-head
            attention = attention_weights.mean(dim=1)  # Average across heads
        else:
            attention = attention_weights
        
        # Average across batch
        attention = attention.mean(dim=0)  # (seq_len, seq_len)
        
        # Add identity matrix to account for residual connections
        seq_len = attention.shape[0]
        identity = torch.eye(seq_len, device=attention.device)
        attention_with_residual = 0.5 * attention + 0.5 * identity
        
        # Compute rollout by matrix multiplication
        rollout = attention_with_residual
        for _ in range(seq_len - 1):
            rollout = torch.matmul(rollout, attention_with_residual)
        
        return rollout.cpu().numpy()


class SaliencyAnalyzer:
    """
    Saliency analysis for neural networks.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize saliency analyzer."""
        self.model = model
        
    def compute_saliency_maps(self, inputs: torch.Tensor,
                             target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute saliency maps for inputs.
        
        Args:
            inputs: Input tensor
            target_class: Target class for classification (None for regression)
            
        Returns:
            Dictionary containing saliency analysis results
        """
        results = {}
        
        # Vanilla gradients
        results['vanilla_gradients'] = self._vanilla_gradients(inputs, target_class)
        
        # Integrated gradients (simplified version)
        results['integrated_gradients'] = self._integrated_gradients(inputs, target_class)
        
        # Guided backpropagation
        results['guided_backprop'] = self._guided_backpropagation(inputs, target_class)
        
        # Feature-wise saliency
        results['feature_saliency'] = np.abs(results['vanilla_gradients']).mean(axis=(0, 1))
        
        # Temporal saliency (for sequence data)
        if len(inputs.shape) == 3:  # (batch, seq, features)
            results['temporal_saliency'] = np.abs(results['vanilla_gradients']).mean(axis=(0, 2))
        
        return results
    
    def _vanilla_gradients(self, inputs: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Compute vanilla gradients."""
        inputs.requires_grad = True
        self.model.eval()
        
        # Forward pass
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
            outputs = outputs['predictions']
        
        # Compute gradients
        self.model.zero_grad()
        
        if target_class is not None:
            # Classification case
            loss = outputs[:, target_class].sum()
        else:
            # Regression case
            loss = outputs.sum()
        
        loss.backward()
        
        gradients = inputs.grad.detach().cpu().numpy()
        inputs.grad = None
        
        return gradients
    
    def _integrated_gradients(self, inputs: torch.Tensor, target_class: Optional[int] = None,
                            n_steps: int = 20) -> np.ndarray:
        """Compute integrated gradients."""
        # Baseline (zeros)
        baseline = torch.zeros_like(inputs)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps, device=inputs.device)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            outputs = self.model(interpolated)
            if isinstance(outputs, dict):
                outputs = outputs['predictions']
            
            # Compute gradients
            self.model.zero_grad()
            
            if target_class is not None:
                loss = outputs[:, target_class].sum()
            else:
                loss = outputs.sum()
            
            loss.backward()
            
            gradients.append(interpolated.grad.detach())
            interpolated.grad = None
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = avg_gradients * (inputs - baseline)
        
        return integrated_gradients.cpu().numpy()
    
    def _guided_backpropagation(self, inputs: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Compute guided backpropagation (simplified version)."""
        # This is a simplified version - full implementation would require
        # modifying ReLU backward pass
        return self._vanilla_gradients(inputs, target_class)


class LimeExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) wrapper.
    """
    
    def __init__(self, model: Any, model_type: str = 'sklearn'):
        """Initialize LIME explainer."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required")
        
        self.model = model
        self.model_type = model_type
        self.explainer = None
        
    def fit_explainer(self, training_data: pd.DataFrame) -> None:
        """Fit LIME explainer on training data."""
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data.values,
            feature_names=training_data.columns,
            class_names=['prediction'],
            mode='regression',
            discretize_continuous=True
        )
    
    def explain_instances(self, X: pd.DataFrame, n_samples: int = 10,
                         n_features: int = 10) -> Dict[str, Any]:
        """
        Explain multiple instances.
        
        Args:
            X: Input data to explain
            n_samples: Number of instances to explain
            n_features: Number of top features to include in explanation
            
        Returns:
            Dictionary containing explanation results
        """
        if self.explainer is None:
            self.fit_explainer(X)
        
        # Create prediction function
        if self.model_type == 'torch':
            def predict_fn(x):
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    pred = self.model(x_tensor)
                    if isinstance(pred, dict):
                        pred = pred['predictions']
                    return pred.numpy()
        else:
            predict_fn = self.model.predict
        
        explanations = []
        feature_importance_matrix = []
        
        sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        for idx in sample_indices:
            try:
                explanation = self.explainer.explain_instance(
                    X.iloc[idx].values,
                    predict_fn,
                    num_features=n_features
                )
                
                explanations.append(explanation)
                
                # Extract feature importance
                feature_importance = dict(explanation.as_list())
                importance_vector = [feature_importance.get(feat, 0) for feat in X.columns]
                feature_importance_matrix.append(importance_vector)
                
            except Exception as e:
                warnings.warn(f"LIME explanation failed for instance {idx}: {e}")
        
        # Calculate statistics across explanations
        feature_importance_matrix = np.array(feature_importance_matrix)
        
        results = {
            'explanations': explanations,
            'feature_importance_matrix': feature_importance_matrix,
            'feature_importance_stats': {
                'mean_importance': np.mean(feature_importance_matrix, axis=0),
                'std_importance': np.std(feature_importance_matrix, axis=0),
                'mean_range': np.mean(np.max(feature_importance_matrix, axis=1) - 
                                    np.min(feature_importance_matrix, axis=1)),
                'top_features': X.columns[np.argsort(np.abs(np.mean(feature_importance_matrix, axis=0)))[-5:]].tolist()
            }
        }
        
        return results


class LocalSurrogateModel:
    """
    Local surrogate model for interpretability.
    """
    
    def __init__(self, model: Any, model_type: str = 'sklearn'):
        """Initialize surrogate model."""
        self.model = model
        self.model_type = model_type
        self.surrogate = None
        
    def fit_surrogate(self, X: pd.DataFrame, surrogate_type: str = 'tree') -> Dict[str, Any]:
        """
        Fit surrogate model to approximate the black-box model.
        
        Args:
            X: Input data
            surrogate_type: Type of surrogate model ('tree', 'forest')
            
        Returns:
            Dictionary containing surrogate analysis results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for surrogate models")
        
        # Get predictions from original model
        if self.model_type == 'torch':
            self.model.eval()
            with torch.no_grad():
                if isinstance(X, pd.DataFrame):
                    X_tensor = torch.FloatTensor(X.values)
                else:
                    X_tensor = torch.FloatTensor(X)
                
                predictions = self.model(X_tensor)
                if isinstance(predictions, dict):
                    predictions = predictions['predictions']
                y_pred = predictions.numpy()
        else:
            y_pred = self.model.predict(X)
        
        # Fit surrogate model
        if surrogate_type == 'tree':
            self.surrogate = DecisionTreeRegressor(max_depth=10, random_state=42)
        elif surrogate_type == 'forest':
            self.surrogate = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")
        
        self.surrogate.fit(X, y_pred.ravel())
        
        # Calculate fidelity (how well surrogate approximates original model)
        surrogate_pred = self.surrogate.predict(X)
        fidelity_score = 1 - np.mean((y_pred.ravel() - surrogate_pred) ** 2) / np.var(y_pred)
        
        # Extract feature importance from surrogate
        if hasattr(self.surrogate, 'feature_importances_'):
            feature_importance = self.surrogate.feature_importances_
            feature_ranking = np.argsort(feature_importance)[::-1]
            
            if isinstance(X, pd.DataFrame):
                top_features = X.columns[feature_ranking].tolist()
            else:
                top_features = feature_ranking.tolist()
        else:
            feature_importance = np.array([])
            top_features = []
        
        return {
            'surrogate_model': self.surrogate,
            'fidelity_score': fidelity_score,
            'feature_importance': feature_importance,
            'feature_ranking': feature_ranking,
            'top_features': top_features
        }