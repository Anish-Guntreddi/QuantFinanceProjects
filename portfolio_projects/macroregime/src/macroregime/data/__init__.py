"""Data loading and generation subpackage.

Public API
----------
From loader_base:
    MacroLoaderBase      — abstract base class for all macro loaders
    apply_release_lag    — shift observation index by lag_days → publication index
    as_of_view           — filter series to rows with pub_date <= as_of
    load_release_calendar — load per-series lag_days from release_calendar.yml
    SyntheticMacroLoader — default offline loader (backed by SyntheticMacroGenerator)

From synthetic:
    SyntheticMacroGenerator — 4-state Markov-switching DGP
    SyntheticMacroPanel     — output dataclass

From fred_loader (optional real-data path — requires FRED_API_KEY):
    FredMacroLoader      — ALFRED first-release vintage loader
"""

from macroregime.data.loader_base import (
    MacroLoaderBase,
    SyntheticMacroLoader,
    apply_release_lag,
    as_of_view,
    load_release_calendar,
)
from macroregime.data.synthetic import SyntheticMacroGenerator, SyntheticMacroPanel

__all__ = [
    # Loader base + utilities
    "MacroLoaderBase",
    "SyntheticMacroLoader",
    "apply_release_lag",
    "as_of_view",
    "load_release_calendar",
    # Synthetic DGP
    "SyntheticMacroGenerator",
    "SyntheticMacroPanel",
]
