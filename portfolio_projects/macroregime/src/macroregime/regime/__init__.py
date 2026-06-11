"""Regime detection and classification subpackage.

Public API (plan 03-04):
    align_regime_labels   — raw->aligned mapping via argsort(means[:, dim])
    transition_matrix     — empirical K×K transition matrix from causal seq
    dwell_times           — mean run length per state
    regime_run_lengths    — run-length encoding of a label sequence
    CausalRegimeDetector  — rolling re-fit causal regime detection (HMM+GMM)
"""
from macroregime.regime.alignment import align_regime_labels
from macroregime.regime.diagnostics import (
    dwell_times,
    regime_run_lengths,
    transition_matrix,
)

__all__ = [
    "align_regime_labels",
    "transition_matrix",
    "dwell_times",
    "regime_run_lengths",
]

# CausalRegimeDetector is imported lazily to avoid pulling in hmmlearn at
# collection time for tests that only need alignment/diagnostics.
def __getattr__(name: str):
    if name == "CausalRegimeDetector":
        from macroregime.regime.causal import CausalRegimeDetector
        return CausalRegimeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
