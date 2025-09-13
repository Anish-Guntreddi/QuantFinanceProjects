"""
Platt Scaling and Beta Calibration

This module implements Platt scaling (sigmoid calibration) and Beta calibration
methods for improving probability predictions.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import beta
from typing import Optional, Tuple, Dict
import warnings


class PlattScaling(BaseEstimator, TransformerMixin):
    """
    Platt scaling (sigmoid calibration) for probability predictions.
    
    Fits a sigmoid function to map predicted probabilities to calibrated
    probabilities. Originally developed for SVMs but applicable to any classifier.
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize Platt scaling calibrator.
        
        Args:
            max_iter: Maximum iterations for optimization
            tol: Tolerance for convergence
        """
        self.max_iter = max_iter
        self.tol = tol
        self.a_ = None
        self.b_ = None
        self.is_fitted_ = False
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """
        Fit Platt scaling parameters using maximum likelihood estimation.
        
        Args:
            probabilities: Predicted probabilities (or decision values)
            labels: True binary labels
            
        Returns:
            self
        """
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        
        if len(probabilities) != len(labels):
            raise ValueError("probabilities and labels must have same length")
        
        if len(np.unique(labels)) != 2:
            raise ValueError("labels must be binary")
        
        # Convert labels to 0/1 if they're not already
        unique_labels = np.unique(labels)
        if not np.array_equal(unique_labels, [0, 1]):
            labels = (labels == unique_labels[1]).astype(int)
        
        # Target probabilities (smoothed to avoid numerical issues)
        # Following Platt (1999) recommendations
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos
        
        # Smoothing parameters
        t = np.zeros_like(labels, dtype=float)
        t[labels == 1] = (n_pos + 1.0) / (n_pos + 2.0)
        t[labels == 0] = 1.0 / (n_neg + 2.0)
        
        # Optimize parameters using Newton-Raphson method
        self.a_, self.b_ = self._fit_sigmoid_newton(probabilities, t)
        self.is_fitted_ = True
        
        return self
    
    def _fit_sigmoid_newton(self, scores: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        """
        Fit sigmoid using Newton-Raphson method as in Platt (1999).
        
        Args:
            scores: Decision scores or probabilities
            targets: Smoothed target probabilities
            
        Returns:
            Tuple of (a, b) parameters for sigmoid
        """
        # Initialize parameters
        a = 0.0
        b = np.log((np.sum(targets) + 1e-10) / (len(targets) - np.sum(targets) + 1e-10))
        
        # Newton-Raphson iterations
        for iteration in range(self.max_iter):
            # Compute current predictions
            fApB = scores * a + b
            
            # Avoid overflow
            fApB = np.clip(fApB, -700, 700)
            p = 1.0 / (1.0 + np.exp(-fApB))
            
            # Compute gradient and Hessian
            d1 = targets - p
            d2 = p * (1 - p)
            
            # Gradient
            g1 = np.sum(d1 * scores)
            g2 = np.sum(d1)
            
            # Hessian
            h11 = np.sum(d2 * scores * scores)
            h22 = np.sum(d2)
            h12 = np.sum(d2 * scores)
            
            # Add regularization to avoid singular matrix
            h11 += 1e-12
            h22 += 1e-12
            
            det = h11 * h22 - h12 * h12
            
            if abs(det) < 1e-12:
                # Matrix is singular, use gradient descent step
                step_size = 0.01
                a += step_size * g1 / (h11 + 1e-12)
                b += step_size * g2 / (h22 + 1e-12)
            else:
                # Newton step
                da = (h22 * g1 - h12 * g2) / det
                db = (h11 * g2 - h12 * g1) / det
                
                # Line search for step size
                step_size = self._line_search(scores, targets, a, b, da, db)
                
                a += step_size * da
                b += step_size * db
            
            # Check convergence
            if abs(da) < self.tol and abs(db) < self.tol:
                break
        
        return a, b
    
    def _line_search(self, scores: np.ndarray, targets: np.ndarray,
                     a: float, b: float, da: float, db: float) -> float:
        """Simple line search for step size."""
        def obj_func(step_size):
            new_a = a + step_size * da
            new_b = b + step_size * db
            fApB = scores * new_a + new_b
            fApB = np.clip(fApB, -700, 700)
            p = 1.0 / (1.0 + np.exp(-fApB))
            
            # Log-likelihood (negative for minimization)
            eps = 1e-15
            p = np.clip(p, eps, 1 - eps)
            return -np.sum(targets * np.log(p) + (1 - targets) * np.log(1 - p))
        
        # Try different step sizes
        step_sizes = [1.0, 0.5, 0.1, 0.01]
        best_step = 1.0
        best_obj = obj_func(1.0)
        
        for step in step_sizes[1:]:
            obj = obj_func(step)
            if obj < best_obj:
                best_obj = obj
                best_step = step
        
        return best_step
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to probabilities.
        
        Args:
            probabilities: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before transform")
        
        probabilities = np.asarray(probabilities)
        
        # Apply sigmoid transformation
        fApB = probabilities * self.a_ + self.b_
        fApB = np.clip(fApB, -700, 700)  # Prevent overflow
        
        return 1.0 / (1.0 + np.exp(-fApB))
    
    def fit_transform(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            
        Returns:
            Calibrated probabilities
        """
        return self.fit(probabilities, labels).transform(probabilities)


class BetaCalibration(BaseEstimator, TransformerMixin):
    """
    Beta calibration for probability predictions.
    
    Uses a Beta distribution to model the relationship between predicted
    and true probabilities. More flexible than Platt scaling.
    """
    
    def __init__(self, method: str = 'mle', regularization: float = 0.0):
        """
        Initialize Beta calibrator.
        
        Args:
            method: Fitting method ('mle' for maximum likelihood, 'moments' for method of moments)
            regularization: L2 regularization parameter
        """
        self.method = method
        self.regularization = regularization
        self.alpha_ = None
        self.beta_ = None
        self.is_fitted_ = False
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'BetaCalibration':
        """
        Fit Beta distribution parameters.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            
        Returns:
            self
        """
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        
        if len(probabilities) != len(labels):
            raise ValueError("probabilities and labels must have same length")
        
        # Avoid boundary values
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)
        
        if self.method == 'mle':
            self.alpha_, self.beta_ = self._fit_beta_mle(probabilities, labels)
        elif self.method == 'moments':
            self.alpha_, self.beta_ = self._fit_beta_moments(probabilities, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted_ = True
        return self
    
    def _fit_beta_mle(self, probabilities: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Fit Beta distribution using maximum likelihood estimation."""
        
        def neg_log_likelihood(params):
            alpha, beta_param = params
            
            if alpha <= 0 or beta_param <= 0:
                return np.inf
            
            # Calculate calibrated probabilities
            calibrated = self._beta_transform(probabilities, alpha, beta_param)
            
            # Avoid log(0)
            eps = 1e-15
            calibrated = np.clip(calibrated, eps, 1 - eps)
            
            # Negative log-likelihood
            nll = -np.sum(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))
            
            # Add regularization
            if self.regularization > 0:
                nll += self.regularization * (alpha**2 + beta_param**2)
            
            return nll
        
        # Initialize parameters
        initial_alpha = 1.0
        initial_beta = 1.0
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[initial_alpha, initial_beta],
            method='L-BFGS-B',
            bounds=[(0.01, 100), (0.01, 100)]
        )
        
        if not result.success:
            warnings.warn("Beta MLE optimization did not converge. Using method of moments.")
            return self._fit_beta_moments(probabilities, labels)
        
        return result.x[0], result.x[1]
    
    def _fit_beta_moments(self, probabilities: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Fit Beta distribution using method of moments."""
        
        # Bin predictions and calculate empirical probabilities
        n_bins = min(20, len(probabilities) // 50)  # Adaptive number of bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        empirical_probs = []
        bin_centers = []
        
        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge for last bin
                mask = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])
            
            if mask.sum() > 0:
                empirical_prob = labels[mask].mean()
                bin_center = probabilities[mask].mean()
                
                empirical_probs.append(empirical_prob)
                bin_centers.append(bin_center)
        
        if len(empirical_probs) < 2:
            # Fall back to uniform prior
            return 1.0, 1.0
        
        empirical_probs = np.array(empirical_probs)
        bin_centers = np.array(bin_centers)
        
        # Method of moments for Beta distribution
        mean_emp = np.mean(empirical_probs)
        var_emp = np.var(empirical_probs)
        
        if var_emp == 0 or mean_emp == 0 or mean_emp == 1:
            return 1.0, 1.0
        
        # Convert to Beta parameters
        alpha = mean_emp * ((mean_emp * (1 - mean_emp)) / var_emp - 1)
        beta_param = (1 - mean_emp) * ((mean_emp * (1 - mean_emp)) / var_emp - 1)
        
        # Ensure positive parameters
        alpha = max(0.01, alpha)
        beta_param = max(0.01, beta_param)
        
        return alpha, beta_param
    
    def _beta_transform(self, probabilities: np.ndarray, alpha: float, beta_param: float) -> np.ndarray:
        """Apply Beta transformation to probabilities."""
        # Use the CDF of Beta distribution as the transformation
        return beta.cdf(probabilities, alpha, beta_param)
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply Beta calibration to probabilities.
        
        Args:
            probabilities: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before transform")
        
        probabilities = np.asarray(probabilities)
        
        # Avoid boundary values
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)
        
        return self._beta_transform(probabilities, self.alpha_, self.beta_)


class HistogramBinning(BaseEstimator, TransformerMixin):
    """
    Histogram binning calibration method.
    
    A simple but effective calibration method that bins predictions and
    uses empirical frequencies as calibrated probabilities.
    """
    
    def __init__(self, n_bins: int = 10, strategy: str = 'uniform'):
        """
        Initialize histogram binning calibrator.
        
        Args:
            n_bins: Number of bins
            strategy: Binning strategy ('uniform' or 'quantile')
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_ = None
        self.bin_calibrated_probs_ = None
        self.is_fitted_ = False
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'HistogramBinning':
        """
        Fit histogram binning calibrator.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            
        Returns:
            self
        """
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        
        if self.strategy == 'uniform':
            self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)
        elif self.strategy == 'quantile':
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            self.bin_edges_ = np.quantile(probabilities, quantiles)
            self.bin_edges_ = np.unique(self.bin_edges_)  # Remove duplicates
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Calculate calibrated probability for each bin
        self.bin_calibrated_probs_ = np.zeros(len(self.bin_edges_) - 1)
        
        for i in range(len(self.bin_edges_) - 1):
            mask = (probabilities >= self.bin_edges_[i]) & (probabilities < self.bin_edges_[i + 1])
            
            if i == len(self.bin_edges_) - 2:  # Last bin includes right edge
                mask = (probabilities >= self.bin_edges_[i]) & (probabilities <= self.bin_edges_[i + 1])
            
            if mask.sum() > 0:
                self.bin_calibrated_probs_[i] = labels[mask].mean()
            else:
                # Use bin midpoint as fallback
                self.bin_calibrated_probs_[i] = (self.bin_edges_[i] + self.bin_edges_[i + 1]) / 2
        
        self.is_fitted_ = True
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply histogram binning calibration.
        
        Args:
            probabilities: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before transform")
        
        probabilities = np.asarray(probabilities)
        calibrated = np.zeros_like(probabilities)
        
        for i in range(len(self.bin_edges_) - 1):
            mask = (probabilities >= self.bin_edges_[i]) & (probabilities < self.bin_edges_[i + 1])
            
            if i == len(self.bin_edges_) - 2:  # Last bin includes right edge
                mask = (probabilities >= self.bin_edges_[i]) & (probabilities <= self.bin_edges_[i + 1])
            
            calibrated[mask] = self.bin_calibrated_probs_[i]
        
        return calibrated


class EnsembleCalibrator(BaseEstimator, TransformerMixin):
    """
    Ensemble of different calibration methods.
    
    Combines multiple calibration approaches and weights their outputs
    for improved robustness.
    """
    
    def __init__(self, calibrators: Optional[list] = None, weights: Optional[np.ndarray] = None):
        """
        Initialize ensemble calibrator.
        
        Args:
            calibrators: List of calibrator objects
            weights: Weights for combining calibrators (default: equal weights)
        """
        if calibrators is None:
            self.calibrators = [
                PlattScaling(),
                HistogramBinning(n_bins=10),
                BetaCalibration()
            ]
        else:
            self.calibrators = calibrators
        
        if weights is None:
            self.weights = np.ones(len(self.calibrators)) / len(self.calibrators)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        self.is_fitted_ = False
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'EnsembleCalibrator':
        """
        Fit all calibrators in the ensemble.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            
        Returns:
            self
        """
        for calibrator in self.calibrators:
            try:
                calibrator.fit(probabilities, labels)
            except Exception as e:
                warnings.warn(f"Calibrator {type(calibrator).__name__} failed to fit: {e}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply ensemble calibration.
        
        Args:
            probabilities: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities (weighted average)
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before transform")
        
        calibrated_outputs = []
        valid_weights = []
        
        for calibrator, weight in zip(self.calibrators, self.weights):
            try:
                if hasattr(calibrator, 'is_fitted_') and calibrator.is_fitted_:
                    calibrated = calibrator.transform(probabilities)
                    calibrated_outputs.append(calibrated)
                    valid_weights.append(weight)
            except Exception as e:
                warnings.warn(f"Calibrator {type(calibrator).__name__} failed to transform: {e}")
        
        if not calibrated_outputs:
            raise ValueError("No calibrators succeeded")
        
        # Normalize weights
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)
        
        # Weighted average
        ensemble_output = np.average(calibrated_outputs, axis=0, weights=valid_weights)
        
        return ensemble_output