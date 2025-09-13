"""
Strategy Allocation Policies
============================

Meta-policy framework for dynamic strategy allocation based on
detected market regimes.
"""

from .allocator import MetaPolicyAllocator, AllocationDecision
from . import strategies
from . import optimization

__all__ = [
    "MetaPolicyAllocator",
    "AllocationDecision",
    "strategies",
    "optimization"
]