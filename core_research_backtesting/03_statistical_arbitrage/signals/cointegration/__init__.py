"""
Cointegration Testing Module

Contains implementations of various cointegration tests:
- Engle-Granger two-step test
- Johansen multivariate test  
- Phillips-Ouliaris test
- Automated pair finding
"""

from .engle_granger import EngleGrangerTest
from .johansen import JohansenTest
from .phillips_ouliaris import PhillipsOuliarisTest
from .pair_finder import PairFinder

__all__ = ['EngleGrangerTest', 'JohansenTest', 'PhillipsOuliarisTest', 'PairFinder']