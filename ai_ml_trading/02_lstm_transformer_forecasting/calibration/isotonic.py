"""
Isotonic Regression Calibration

This module implements isotonic regression-based calibration methods for
improving probability predictions in machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.stats import chi2


class IsotonicCalibrator(BaseEstimator, TransformerMixin):
    """
    Isotonic regression calibrator for probability predictions.
    
    This calibrator fits a monotonic function to map predicted probabilities
    to calibrated probabilities that better reflect the true likelihood.
    """
    
    def __init__(self, out_of_bounds: str = 'clip', 
                 min_bin_size: int = 100,
                 y_min: Optional[float] = None,
                 y_max: Optional[float] = None):
        """
        Initialize isotonic calibrator.
        
        Args:
            out_of_bounds: How to handle out-of-bounds predictions ('clip' or 'nan')
            min_bin_size: Minimum samples per bin for reliability
            y_min: Minimum value for calibrated probabilities
            y_max: Maximum value for calibrated probabilities
        """
        self.out_of_bounds = out_of_bounds
        self.min_bin_size = min_bin_size
        self.y_min = y_min
        self.y_max = y_max
        self.calibrator = None
        self.calibration_curve = None
        self.is_fitted_ = False
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression calibrator.
        
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
        
        if len(np.unique(labels)) != 2:
            raise ValueError("labels must be binary")
        
        # Fit isotonic regression
        self.calibrator = IsotonicRegression(
            out_of_bounds=self.out_of_bounds,
            y_min=self.y_min,
            y_max=self.y_max
        )
        
        self.calibrator.fit(probabilities, labels)
        
        # Store calibration curve for analysis
        self.calibration_curve = self._compute_calibration_curve(probabilities, labels)
        self.is_fitted_ = True
        
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            probabilities: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before transform")
        
        probabilities = np.asarray(probabilities)
        return self.calibrator.transform(probabilities)
    
    def fit_transform(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fit calibrator and transform probabilities in one step.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            
        Returns:
            Calibrated probabilities
        """
        return self.fit(probabilities, labels).transform(probabilities)
    
    def _compute_calibration_curve(self, probabilities: np.ndarray, 
                                  labels: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
        """Compute calibration curve data for analysis."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fraction_pos = []
        mean_pred = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            
            if i == n_bins - 1:  # Include right edge for last bin
                mask = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])
            
            if mask.sum() >= self.min_bin_size:
                fraction_pos.append(labels[mask].mean())
                mean_pred.append(probabilities[mask].mean())
                bin_counts.append(mask.sum())
            else:
                fraction_pos.append(np.nan)
                mean_pred.append(np.nan)
                bin_counts.append(0)
        
        return {
            'bin_centers': bin_centers,
            'fraction_positive': np.array(fraction_pos),
            'mean_predicted': np.array(mean_pred),
            'bin_counts': np.array(bin_counts)
        }
    
    def plot_calibration_curve(self, n_bins: int = 10, 
                              show_histogram: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot calibration curve and histogram.
        
        Args:
            n_bins: Number of bins for calibration curve
            show_histogram: Whether to show prediction histogram
            
        Returns:
            Tuple of (figure, axes)
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before plotting")
        
        if show_histogram:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None
        
        # Plot calibration curve
        curve = self.calibration_curve
        valid_mask = ~np.isnan(curve['fraction_positive'])
        
        ax1.plot(curve['mean_predicted'][valid_mask], 
                curve['fraction_positive'][valid_mask], 
                'o-', label='Calibration curve', markersize=6)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve (Reliability Diagram)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Add bin counts as annotations
        for i, (x, y, count) in enumerate(zip(curve['mean_predicted'][valid_mask],
                                            curve['fraction_positive'][valid_mask],
                                            curve['bin_counts'][valid_mask])):
            if not np.isnan(x) and not np.isnan(y):
                ax1.annotate(f'n={int(count)}', (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Plot histogram of predictions
        if show_histogram:
            # This would require access to original predictions
            # For now, show bin counts
            ax2.bar(curve['bin_centers'], curve['bin_counts'], 
                   width=0.08, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Predicted Probability')
            ax2.set_ylabel('Count')
            ax2.set_title('Distribution of Predictions')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2) if ax2 is not None else fig, ax1
    
    def expected_calibration_error(self, probabilities: np.ndarray, 
                                  labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            n_bins: Number of bins
            
        Returns:
            ECE value
        """
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])
            
            if mask.sum() > 0:
                bin_accuracy = labels[mask].mean()
                bin_confidence = probabilities[mask].mean()
                bin_size = mask.sum()
                
                ece += (bin_size / len(probabilities)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def maximum_calibration_error(self, probabilities: np.ndarray,
                                 labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            n_bins: Number of bins
            
        Returns:
            MCE value
        """
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        max_error = 0.0
        
        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])
            
            if mask.sum() > 0:
                bin_accuracy = labels[mask].mean()
                bin_confidence = probabilities[mask].mean()
                
                error = abs(bin_accuracy - bin_confidence)
                max_error = max(max_error, error)
        
        return max_error
    
    def brier_score(self, probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate Brier score and its decomposition.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            
        Returns:
            Dictionary with Brier score components
        """
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        
        # Brier score
        brier_score = np.mean((probabilities - labels) ** 2)
        
        # Brier score decomposition
        # Reliability (calibration)
        curve = self._compute_calibration_curve(probabilities, labels)
        valid_mask = ~np.isnan(curve['fraction_positive'])
        
        reliability = 0.0
        for i, (mean_pred, frac_pos, count) in enumerate(zip(
            curve['mean_predicted'][valid_mask],
            curve['fraction_positive'][valid_mask], 
            curve['bin_counts'][valid_mask]
        )):
            if count > 0:
                reliability += (count / len(probabilities)) * (mean_pred - frac_pos) ** 2
        
        # Resolution
        base_rate = labels.mean()
        resolution = 0.0
        for i, (frac_pos, count) in enumerate(zip(
            curve['fraction_positive'][valid_mask],
            curve['bin_counts'][valid_mask]
        )):
            if count > 0:
                resolution += (count / len(probabilities)) * (frac_pos - base_rate) ** 2
        
        # Uncertainty (irreducible)
        uncertainty = base_rate * (1 - base_rate)
        
        return {
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'brier_skill_score': (resolution - reliability) / uncertainty if uncertainty > 0 else 0
        }


class BinnedCalibrator(BaseEstimator, TransformerMixin):
    """
    Binned calibration method.
    
    Divides predictions into bins and uses the empirical frequency
    in each bin as the calibrated probability.
    """
    
    def __init__(self, n_bins: int = 10, strategy: str = 'uniform'):
        """
        Initialize binned calibrator.
        
        Args:
            n_bins: Number of bins
            strategy: Binning strategy ('uniform' or 'quantile')
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_ = None
        self.bin_probabilities_ = None
        self.is_fitted_ = False
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'BinnedCalibrator':
        """
        Fit binned calibrator.
        
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
            # Use quantiles of the predictions
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            self.bin_edges_ = np.quantile(probabilities, quantiles)
            # Ensure edges are unique
            self.bin_edges_ = np.unique(self.bin_edges_)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Calculate empirical probability for each bin
        self.bin_probabilities_ = np.zeros(len(self.bin_edges_) - 1)
        
        for i in range(len(self.bin_edges_) - 1):
            mask = (probabilities >= self.bin_edges_[i]) & (probabilities < self.bin_edges_[i + 1])
            
            if i == len(self.bin_edges_) - 2:  # Last bin includes right edge
                mask = (probabilities >= self.bin_edges_[i]) & (probabilities <= self.bin_edges_[i + 1])
            
            if mask.sum() > 0:
                self.bin_probabilities_[i] = labels[mask].mean()
            else:
                # Use bin center as fallback
                self.bin_probabilities_[i] = (self.bin_edges_[i] + self.bin_edges_[i + 1]) / 2
        
        self.is_fitted_ = True
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply binned calibration.
        
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
            
            calibrated[mask] = self.bin_probabilities_[i]
        
        return calibrated


class AdaptiveIsotonicCalibrator(IsotonicCalibrator):
    """
    Adaptive isotonic calibrator that adjusts to different data regimes.
    
    Uses multiple isotonic regressors for different conditions or time periods.
    """
    
    def __init__(self, regime_column: Optional[str] = None, 
                 time_window: Optional[int] = None, **kwargs):
        """
        Initialize adaptive calibrator.
        
        Args:
            regime_column: Column name for regime identification
            time_window: Rolling window size for temporal adaptation
            **kwargs: Additional arguments for base IsotonicCalibrator
        """
        super().__init__(**kwargs)
        self.regime_column = regime_column
        self.time_window = time_window
        self.regime_calibrators_ = {}
        self.temporal_calibrators_ = []
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray,
            metadata: Optional[pd.DataFrame] = None) -> 'AdaptiveIsotonicCalibrator':
        """
        Fit adaptive calibrator.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            metadata: DataFrame with regime/time information
            
        Returns:
            self
        """
        if self.regime_column and metadata is not None:
            # Fit separate calibrators for each regime
            if self.regime_column in metadata.columns:
                regimes = metadata[self.regime_column].unique()
                
                for regime in regimes:
                    mask = metadata[self.regime_column] == regime
                    if mask.sum() > 100:  # Minimum samples per regime
                        calibrator = IsotonicCalibrator(**self.get_params())
                        calibrator.fit(probabilities[mask], labels[mask])
                        self.regime_calibrators_[regime] = calibrator
        
        elif self.time_window:
            # Fit calibrators for rolling time windows
            n_windows = len(probabilities) // self.time_window
            
            for i in range(n_windows):
                start_idx = i * self.time_window
                end_idx = min((i + 1) * self.time_window, len(probabilities))
                
                if end_idx - start_idx > 100:  # Minimum samples per window
                    calibrator = IsotonicCalibrator(**self.get_params())
                    calibrator.fit(probabilities[start_idx:end_idx], 
                                 labels[start_idx:end_idx])
                    self.temporal_calibrators_.append(calibrator)
        
        # Fit base calibrator on all data as fallback
        super().fit(probabilities, labels)
        
        return self
    
    def transform(self, probabilities: np.ndarray,
                 metadata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Apply adaptive calibration.
        
        Args:
            probabilities: Predicted probabilities to calibrate
            metadata: DataFrame with regime/time information
            
        Returns:
            Calibrated probabilities
        """
        calibrated = np.zeros_like(probabilities)
        
        if self.regime_column and metadata is not None:
            # Use regime-specific calibrators
            if self.regime_column in metadata.columns:
                for regime, calibrator in self.regime_calibrators_.items():
                    mask = metadata[self.regime_column] == regime
                    if mask.sum() > 0:
                        calibrated[mask] = calibrator.transform(probabilities[mask])
                
                # Use base calibrator for unknown regimes
                unmapped_mask = calibrated == 0
                if unmapped_mask.sum() > 0:
                    calibrated[unmapped_mask] = super().transform(probabilities[unmapped_mask])
                
                return calibrated
        
        # Fall back to base calibrator
        return super().transform(probabilities)