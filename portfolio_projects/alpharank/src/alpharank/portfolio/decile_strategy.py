"""PrecomputedWeightsStrategy — qbacktest Strategy adapter for precomputed decile weights.

Design notes
------------
- Stores a sorted list of rebalance dates and uses bisect_right for O(log n)
  as-of lookup per MarketEvent.  A sorted() call per bar event would be
  O(n log n) per bar and O(n^2 log n) total — unacceptable for daily bars with
  monthly rebalance over 10+ years.
- Emits signals only on direction changes (LONG → SHORT requires EXIT first;
  new entry after EXIT emits LONG/SHORT fresh).
- Emits EXIT when a symbol held in the prior portfolio drops to weight 0.0 or
  is absent from the current rebalance weights dict.
- Before the first rebalance date, no signals are emitted.
"""

from __future__ import annotations

import bisect
from typing import Literal

import pandas as pd

from qbacktest.events import MarketEvent, SignalEvent
from qbacktest.strategy.base import Strategy


# Direction state for each symbol
_Dir = Literal["LONG", "SHORT", "EXIT", None]


class PrecomputedWeightsStrategy(Strategy):
    """Strategy that replays precomputed decile portfolio weights.

    Parameters
    ----------
    weights:
        dict mapping pd.Timestamp rebalance dates → {symbol: weight}.
        Positive weights → LONG; negative weights → SHORT; 0.0 → EXIT.
        Pass the output of ``build_decile_weights`` directly.

    Notes
    -----
    - ``calculate_signals()`` is called once per MarketEvent (one symbol, one bar).
    - Signals are emitted only when the direction for a symbol changes from its
      last emitted direction, preventing redundant round-trips through the engine.
    - The as-of lookup uses ``bisect_right(rebal_keys, event.timestamp) - 1``
      which correctly resolves to the latest rebalance date ≤ event.timestamp.
      If no rebalance has occurred yet the result is -1 → return [] immediately.
    """

    def __init__(self, weights: dict[pd.Timestamp, dict[str, float]]) -> None:
        # Sort rebalance dates once for O(log n) bisect lookups later.
        # Sorting here (O(k log k) where k = number of rebalances, typically
        # ~12-120) vs sorting on every bar event (O(k log k) per bar × n_bars
        # × n_symbols) is critical for performance.
        self._rebal_keys: list[pd.Timestamp] = sorted(weights.keys())
        self._weights = weights

        # Track last emitted direction per symbol: None = never signalled
        self._last_direction: dict[str, _Dir] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        """Return zero or more SignalEvents for a single MarketEvent bar.

        Parameters
        ----------
        event:
            The latest bar for one symbol.

        Returns
        -------
        list[SignalEvent]
            - Empty when: no rebalance has occurred yet, or direction unchanged.
            - One SignalEvent with direction LONG/SHORT/EXIT otherwise.
        """
        # O(log k) as-of lookup — critical path
        idx = bisect.bisect_right(self._rebal_keys, event.timestamp) - 1
        if idx < 0:
            # No rebalance date ≤ event.timestamp yet
            return []

        rebal_date = self._rebal_keys[idx]
        current_weights = self._weights[rebal_date]
        sym = event.symbol

        raw_weight = current_weights.get(sym, None)

        # Determine target direction
        if raw_weight is None or raw_weight == 0.0:
            # Symbol absent or explicitly zero → EXIT if we hold it
            target: _Dir = "EXIT"
        elif raw_weight > 0.0:
            target = "LONG"
        else:
            target = "SHORT"

        prev_dir = self._last_direction.get(sym, None)

        # Case: we want to EXIT a symbol we never entered → no-op
        if target == "EXIT" and prev_dir is None:
            return []

        # Case: we want to EXIT a symbol that's already been exited → no-op
        if target == "EXIT" and prev_dir == "EXIT":
            return []

        # Case: direction unchanged (LONG→LONG or SHORT→SHORT) → no-op
        if target == prev_dir:
            return []

        # Direction changed → emit signal
        strength = abs(raw_weight) if raw_weight is not None and raw_weight != 0.0 else 1.0
        signal = SignalEvent(
            timestamp=event.timestamp,
            symbol=sym,
            direction=target,
            strength=strength,
        )
        self._last_direction[sym] = target
        return [signal]
