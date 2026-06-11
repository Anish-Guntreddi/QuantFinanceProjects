"""TargetWeightStrategy — as-of weight replay with weight-magnitude re-emission.

WHY THIS DIFFERS FROM PrecomputedWeightsStrategy (alpharank):
  alpharank's PrecomputedWeightsStrategy tracks _last_direction and returns []
  whenever ``target == prev_dir``. This means LONG 0.60 → LONG 0.30 emits NO
  signal — the direction is unchanged even though the weight changed. The position
  would silently remain at the old size with no rebalance.

  TargetWeightStrategy instead tracks _last_emitted (the signed weight, not the
  direction). A new signal is emitted whenever |new_weight - last_emitted| > 1e-9,
  i.e., ANY change in magnitude triggers a rebalance. TargetWeightPortfolio then
  computes the correct delta order from the new weight.

Design:
  - Uses bisect_right O(log k) as-of lookup over sorted rebalance keys (identical
    pattern to alpharank's PrecomputedWeightsStrategy.__init__).
  - Tracks _last_emitted: dict[str, float] — signed weight per symbol:
      +w = LONG at weight w, -w = SHORT at |w|, 0.0 = never held or exited.
  - On each MarketEvent:
      idx = bisect_right(rebal_keys, event.timestamp) - 1
      if idx < 0: return []  (before first rebalance)
      target_w = current_schedule[symbol] or 0.0 (absent → want to exit)
      if |target_w - last_emitted[symbol]| <= 1e-9: return []  (no change)
      emit signal, update _last_emitted

Signal encoding:
  target_w > 0  → SignalEvent(direction="LONG",  strength=target_w)
  target_w < 0  → SignalEvent(direction="SHORT", strength=|target_w|)
  target_w == 0 and last != 0  → SignalEvent(direction="EXIT", strength=1.0)
  target_w == 0 and last == 0  → no signal (never held this symbol)

This phase is long-only so negative weights are supported for completeness only.
"""

from __future__ import annotations

import bisect
import logging

import pandas as pd

from qbacktest.events import MarketEvent, SignalEvent
from qbacktest.strategy.base import Strategy

logger = logging.getLogger(__name__)


class TargetWeightStrategy(Strategy):
    """Replays precomputed target weights, re-emitting on ANY weight magnitude change.

    Unlike alpharank's PrecomputedWeightsStrategy (direction-change-only emission),
    this strategy re-emits whenever the WEIGHT changes — not just the direction.
    This ensures LONG 0.60 → LONG 0.30 generates a rebalance signal, resizing
    the position correctly via TargetWeightPortfolio.

    Parameters
    ----------
    weights:
        dict mapping pd.Timestamp rebalance dates → {symbol: signed_weight}.
        Positive weight → LONG; negative → SHORT; absent or 0.0 → EXIT.
        Pass the output of ``build_weight_schedule`` directly.

    Notes
    -----
    - Rebalance keys are sorted once in ``__init__``: O(k log k), k = n rebalances.
    - Each ``calculate_signals()`` call does O(log k) bisect lookup.
    - _last_emitted stores signed weights; 0.0 means "never held or exited".
    """

    def __init__(self, weights: dict[pd.Timestamp, dict[str, float]]) -> None:
        # Sort rebalance dates once for O(log k) bisect lookups later.
        # Sorting here (O(k log k) where k = number of rebalances, typically ~12-120)
        # avoids O(k log k) per bar, which would be catastrophic for daily bars.
        self._rebal_keys: list[pd.Timestamp] = sorted(weights.keys())
        self._weights = weights

        # Track last emitted SIGNED WEIGHT per symbol.
        # 0.0 = never held or fully exited.
        # +w = currently LONG at weight w.
        # -w = currently SHORT at weight |w|.
        self._last_emitted: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        """Return zero or one SignalEvent for this bar.

        Emits if and only if the target weight for this symbol has CHANGED
        (magnitude or sign) since the last emission. This closes the
        direction-change-only emission gap in PrecomputedWeightsStrategy.

        Parameters
        ----------
        event:
            The latest bar for one symbol.

        Returns
        -------
        list[SignalEvent]
            Empty if: no rebalance has occurred yet, or weight unchanged.
            One SignalEvent otherwise (LONG/SHORT/EXIT with strength=|weight|).
        """
        # O(log k) as-of lookup — critical path
        idx = bisect.bisect_right(self._rebal_keys, event.timestamp) - 1
        if idx < 0:
            # No rebalance date ≤ event.timestamp yet — warm-up period
            return []

        rebal_date = self._rebal_keys[idx]
        current_weights = self._weights[rebal_date]
        sym = event.symbol

        # Target signed weight: absent symbols → 0.0 (want to exit)
        target_w: float = current_weights.get(sym, 0.0)

        # Last emitted signed weight: default 0.0 = never held
        last_w: float = self._last_emitted.get(sym, 0.0)

        # Emit only if weight changed (magnitude OR sign)
        if abs(target_w - last_w) <= 1e-9:
            return []

        # Determine signal
        if target_w > 0.0:
            direction = "LONG"
            strength = target_w
        elif target_w < 0.0:
            direction = "SHORT"
            strength = abs(target_w)
        else:
            # target_w == 0 and last_w != 0 → EXIT
            if abs(last_w) <= 1e-9:
                # Never held — nothing to exit
                return []
            direction = "EXIT"
            strength = 1.0

        signal = SignalEvent(
            timestamp=event.timestamp,
            symbol=sym,
            direction=direction,
            strength=strength,
        )
        # Update last emitted BEFORE returning (signed weight)
        self._last_emitted[sym] = target_w
        return [signal]
