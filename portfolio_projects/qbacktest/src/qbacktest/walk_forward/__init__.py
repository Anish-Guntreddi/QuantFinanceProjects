"""qbacktest.walk_forward — rolling walk-forward validation.

Public API:
  WalkForwardWindow   — dataclass with train/test date ranges
  WalkForwardResults  — dataclass with per-window and aggregate OOS results
  generate_windows()  — rolling window generator
  WalkForwardRunner   — orchestrator that calls engine_factory per window
"""

from qbacktest.walk_forward.runner import (
    WalkForwardWindow,
    WalkForwardResults,
    generate_windows,
    WalkForwardRunner,
)

__all__ = [
    "WalkForwardWindow",
    "WalkForwardResults",
    "generate_windows",
    "WalkForwardRunner",
]
