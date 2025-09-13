"""
Temperature Scaling Calibration for Neural Networks

This module implements temperature scaling and its variants for calibrating
neural network predictions, particularly effective for deep learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize_scalar, minimize
from typing import Optional, Union, Tuple
import warnings


class TemperatureScaling(BaseEstimator, TransformerMixin):
    """
    Temperature scaling for neural network calibration.
    
    Applies a single temperature parameter to scale logits before softmax,
    improving calibration without changing the model's accuracy.
    """
    
    def __init__(self, method: str = 'nll', max_iter: int = 50):
        """
        Initialize temperature scaling calibrator.
        
        Args:
            method: Optimization method ('nll' for negative log-likelihood, 'ece' for ECE)
            max_iter: Maximum iterations for optimization
        """
        self.method = method
        self.max_iter = max_iter
        self.temperature_ = 1.0
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> 'TemperatureScaling':
        """
        Fit temperature parameter.
        
        Args:
            logits: Model logits (before softmax)
            labels: True class labels (integers)
            
        Returns:
            self
        """
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        
        if len(logits.shape) == 1:
            # Binary classification case
            logits = np.column_stack([np.zeros_like(logits), logits])
        
        if self.method == 'nll':
            self.temperature_ = self._optimize_temperature_nll(logits, labels)
        elif self.method == 'ece':
            self.temperature_ = self._optimize_temperature_ece(logits, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted_ = True
        return self
    
    def _optimize_temperature_nll(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Optimize temperature using negative log-likelihood."""
        
        def neg_log_likelihood(T):
            if T <= 0:
                return np.inf
            
            # Apply temperature scaling
            scaled_logits = logits / T
            
            # Compute softmax probabilities
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Compute negative log-likelihood
            n_samples = len(labels)
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            
            if len(probs.shape) == 1:  # Binary case
                nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            else:  # Multi-class case
                nll = -np.mean(np.log(probs[np.arange(n_samples), labels]))
            
            return nll
        
        # Optimize temperature
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(0.01, 100.0),
            method='bounded',
            options={'maxiter': self.max_iter}
        )
        
        return result.x if result.success else 1.0
    
    def _optimize_temperature_ece(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Optimize temperature using Expected Calibration Error."""
        
        def expected_calibration_error(T):
            if T <= 0:
                return np.inf
            
            # Apply temperature scaling and get probabilities
            scaled_logits = logits / T
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Get predicted classes and confidences
            if len(probs.shape) == 1:  # Binary case
                confidences = np.maximum(probs, 1 - probs)
                predictions = (probs > 0.5).astype(int)
            else:  # Multi-class case
                confidences = np.max(probs, axis=1)
                predictions = np.argmax(probs, axis=1)
            
            # Calculate ECE
            return self._calculate_ece(confidences, predictions, labels)
        
        # Optimize temperature
        result = minimize_scalar(
            expected_calibration_error,
            bounds=(0.01, 100.0),
            method='bounded',
            options={'maxiter': self.max_iter}
        )
        
        return result.x if result.success else 1.0
    
    def _calculate_ece(self, confidences: np.ndarray, predictions: np.ndarray, 
                      labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge for last bin
                bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            
            if bin_mask.sum() > 0:
                bin_accuracy = (predictions[bin_mask] == labels[bin_mask]).mean()
                bin_confidence = confidences[bin_mask].mean()
                bin_size = bin_mask.sum()
                
                ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Temperature scaling must be fitted before transform")
        
        logits = np.asarray(logits)
        
        if len(logits.shape) == 1:
            # Binary classification case
            logits = np.column_stack([np.zeros_like(logits), logits])
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature_
        
        # Compute softmax probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Return single column for binary classification
        if probs.shape[1] == 2:
            return probs[:, 1]
        
        return probs
    
    def fit_transform(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(logits, labels).transform(logits)


class VectorScaling(BaseEstimator, TransformerMixin):
    """
    Vector scaling calibration.
    
    Learns a separate temperature parameter for each class,
    providing more flexibility than single temperature scaling.
    """
    
    def __init__(self, max_iter: int = 50):
        """
        Initialize vector scaling calibrator.
        
        Args:
            max_iter: Maximum iterations for optimization
        """
        self.max_iter = max_iter
        self.temperatures_ = None
        self.n_classes_ = None
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> 'VectorScaling':
        """
        Fit temperature parameters for each class.
        
        Args:
            logits: Model logits
            labels: True class labels
            
        Returns:
            self
        """
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        
        if len(logits.shape) == 1:
            # Binary classification case
            logits = np.column_stack([np.zeros_like(logits), logits])
        
        self.n_classes_ = logits.shape[1]
        
        def neg_log_likelihood(temperatures):
            if np.any(temperatures <= 0):
                return np.inf
            
            # Apply vector scaling
            scaled_logits = logits / temperatures.reshape(1, -1)
            
            # Compute softmax probabilities
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Compute negative log-likelihood
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            
            return nll
        
        # Initialize temperatures
        initial_temps = np.ones(self.n_classes_)
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=initial_temps,
            method='L-BFGS-B',
            bounds=[(0.01, 100)] * self.n_classes_,
            options={'maxiter': self.max_iter}
        )
        
        self.temperatures_ = result.x if result.success else initial_temps
        self.is_fitted_ = True
        
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply vector scaling to logits.
        
        Args:
            logits: Model logits to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Vector scaling must be fitted before transform")
        
        logits = np.asarray(logits)
        
        if len(logits.shape) == 1:
            # Binary classification case
            logits = np.column_stack([np.zeros_like(logits), logits])
        
        # Apply vector scaling
        scaled_logits = logits / self.temperatures_.reshape(1, -1)
        
        # Compute softmax probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Return single column for binary classification
        if probs.shape[1] == 2:
            return probs[:, 1]
        
        return probs


class MatrixScaling(BaseEstimator, TransformerMixin):
    """
    Matrix scaling calibration.
    
    Most general form of scaling that learns a full matrix transformation
    of the logits before applying softmax.
    """
    
    def __init__(self, max_iter: int = 100, regularization: float = 0.01):
        """
        Initialize matrix scaling calibrator.
        
        Args:
            max_iter: Maximum iterations for optimization
            regularization: L2 regularization strength
        """
        self.max_iter = max_iter
        self.regularization = regularization
        self.weight_matrix_ = None
        self.bias_vector_ = None
        self.n_classes_ = None
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> 'MatrixScaling':
        """
        Fit matrix scaling parameters.
        
        Args:
            logits: Model logits
            labels: True class labels
            
        Returns:
            self
        """
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        
        if len(logits.shape) == 1:
            # Binary classification case
            logits = np.column_stack([np.zeros_like(logits), logits])
        
        self.n_classes_ = logits.shape[1]
        
        def neg_log_likelihood_with_reg(params):
            # Unpack parameters
            n_params_per_class = self.n_classes_
            weight_params = params[:self.n_classes_ * n_params_per_class]
            bias_params = params[self.n_classes_ * n_params_per_class:]
            
            W = weight_params.reshape(self.n_classes_, self.n_classes_)
            b = bias_params.reshape(1, self.n_classes_)
            
            # Apply matrix scaling
            scaled_logits = np.dot(logits, W.T) + b
            
            # Compute softmax probabilities
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Compute negative log-likelihood
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            
            # Add regularization
            reg_term = self.regularization * (np.sum(W**2) + np.sum(b**2))
            
            return nll + reg_term
        
        # Initialize parameters (start with identity transformation)
        initial_weight = np.eye(self.n_classes_).flatten()
        initial_bias = np.zeros(self.n_classes_)
        initial_params = np.concatenate([initial_weight, initial_bias])
        
        # Optimize
        result = minimize(
            neg_log_likelihood_with_reg,
            x0=initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )
        
        if result.success:
            # Unpack optimized parameters
            n_params_per_class = self.n_classes_
            weight_params = result.x[:self.n_classes_ * n_params_per_class]
            bias_params = result.x[self.n_classes_ * n_params_per_class:]
            
            self.weight_matrix_ = weight_params.reshape(self.n_classes_, self.n_classes_)
            self.bias_vector_ = bias_params.reshape(1, self.n_classes_)
        else:
            # Fall back to identity transformation
            warnings.warn("Matrix scaling optimization failed. Using identity transformation.")
            self.weight_matrix_ = np.eye(self.n_classes_)
            self.bias_vector_ = np.zeros((1, self.n_classes_))
        
        self.is_fitted_ = True
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply matrix scaling to logits.
        
        Args:
            logits: Model logits to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Matrix scaling must be fitted before transform")
        
        logits = np.asarray(logits)
        
        if len(logits.shape) == 1:
            # Binary classification case
            logits = np.column_stack([np.zeros_like(logits), logits])
        
        # Apply matrix scaling
        scaled_logits = np.dot(logits, self.weight_matrix_.T) + self.bias_vector_
        
        # Compute softmax probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Return single column for binary classification
        if probs.shape[1] == 2:
            return probs[:, 1]
        
        return probs


class TorchTemperatureScaling(nn.Module):
    """
    PyTorch implementation of temperature scaling for integration with PyTorch models.
    """
    
    def __init__(self, init_temperature: float = 1.0):
        """
        Initialize temperature scaling layer.
        
        Args:
            init_temperature: Initial temperature value
        """
        super(TorchTemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling.
        
        Args:
            logits: Model logits
            
        Returns:
            Temperature-scaled probabilities
        """
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)
    
    def fit(self, model: nn.Module, val_loader, criterion, optimizer_class=torch.optim.LBFGS, 
            max_iter: int = 50, device: str = 'cpu'):
        """
        Fit temperature parameter using validation data.
        
        Args:
            model: Base model to calibrate
            val_loader: Validation data loader
            criterion: Loss criterion (e.g., nn.CrossEntropyLoss())
            optimizer_class: Optimizer class
            max_iter: Maximum iterations
            device: Device to use
        """
        self.to(device)
        model.to(device)
        model.eval()
        
        # Only optimize temperature parameter
        optimizer = optimizer_class([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            loss = 0.0
            count = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Get model logits
                    logits = model(inputs)
                    if isinstance(logits, dict):
                        logits = logits['predictions']
                    
                    # Apply temperature scaling
                    scaled_probs = self.forward(logits)
                    
                    # Calculate loss
                    loss += criterion(torch.log(scaled_probs + 1e-8), targets).item()
                    count += 1
            
            return loss / count if count > 0 else float('inf')
        
        def closure():
            optimizer.zero_grad()
            loss = eval_loss()
            loss_tensor = torch.tensor(loss, requires_grad=True)
            loss_tensor.backward()
            return loss_tensor
        
        # Optimize
        for _ in range(max_iter // 10):  # LBFGS does multiple iterations per step
            try:
                optimizer.step(closure)
            except:
                break
        
        # Ensure temperature is positive
        with torch.no_grad():
            self.temperature.clamp_(min=0.01)


class EnsembleTemperatureScaling(BaseEstimator, TransformerMixin):
    """
    Ensemble of different temperature scaling methods.
    """
    
    def __init__(self):
        """Initialize ensemble of temperature scaling methods."""
        self.calibrators = {
            'temperature': TemperatureScaling(method='nll'),
            'vector': VectorScaling(),
            'matrix': MatrixScaling(regularization=0.01)
        }
        self.best_calibrator_ = None
        self.calibrator_scores_ = {}
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray, 
            validation_split: float = 0.2) -> 'EnsembleTemperatureScaling':
        """
        Fit all calibrators and select the best one based on validation performance.
        
        Args:
            logits: Model logits
            labels: True class labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            self
        """
        # Split data for validation
        n_val = int(len(logits) * validation_split)
        indices = np.random.permutation(len(logits))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        logits_train = logits[train_indices]
        labels_train = labels[train_indices]
        logits_val = logits[val_indices]
        labels_val = labels[val_indices]
        
        best_score = np.inf
        
        # Fit and evaluate each calibrator
        for name, calibrator in self.calibrators.items():
            try:
                # Fit calibrator
                calibrator.fit(logits_train, labels_train)
                
                # Evaluate on validation set
                probs_val = calibrator.transform(logits_val)
                
                # Calculate validation score (negative log-likelihood)
                eps = 1e-15
                if len(probs_val.shape) == 1:  # Binary case
                    probs_val = np.clip(probs_val, eps, 1 - eps)
                    score = -np.mean(labels_val * np.log(probs_val) + 
                                   (1 - labels_val) * np.log(1 - probs_val))
                else:  # Multi-class case
                    probs_val = np.clip(probs_val, eps, 1 - eps)
                    score = -np.mean(np.log(probs_val[np.arange(len(labels_val)), labels_val]))
                
                self.calibrator_scores_[name] = score
                
                if score < best_score:
                    best_score = score
                    self.best_calibrator_ = name
                    
            except Exception as e:
                warnings.warn(f"Calibrator {name} failed: {e}")
                self.calibrator_scores_[name] = np.inf
        
        # Refit best calibrator on full data
        if self.best_calibrator_ is not None:
            self.calibrators[self.best_calibrator_].fit(logits, labels)
        else:
            # Fall back to simple temperature scaling
            self.best_calibrator_ = 'temperature'
            self.calibrators[self.best_calibrator_].fit(logits, labels)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply best calibrator to logits.
        
        Args:
            logits: Model logits to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before transform")
        
        return self.calibrators[self.best_calibrator_].transform(logits)