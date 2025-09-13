"""
Calibration Package for Probability Predictions

This package provides various calibration techniques for improving probability
predictions including isotonic regression, Platt scaling, and temperature scaling.
"""

from .isotonic import IsotonicCalibrator, BinnedCalibrator
from .platt import PlattScaling, BetaCalibration
from .temperature_scaling import TemperatureScaling, VectorScaling, MatrixScaling

__all__ = [
    'IsotonicCalibrator',
    'BinnedCalibrator', 
    'PlattScaling',
    'BetaCalibration',
    'TemperatureScaling',
    'VectorScaling',
    'MatrixScaling'
]