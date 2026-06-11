"""Regime→weights YAML loader and schedule builder.

Exports:
  - load_regime_weights: load strategy_params.yml and validate
  - build_weight_schedule: convert regime Series + rebalance dates → dated weight dicts
  - month_end_rebalance_dates: last business day per month for a DatetimeIndex
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Default config path relative to this file (package-internal, resolved at call time)
# Path chain: weights.py → allocation/ → macroregime/ → src/ → macroregime-project-root/
_DEFAULT_CONFIG: Path = (
    Path(__file__).parent.parent.parent.parent  # up to macroregime project root
    / "configs"
    / "strategy_params.yml"
)


def load_regime_weights(path: str | Path | None = None) -> dict[int, dict[str, float]]:
    """Load regime→weights mapping from strategy_params.yml.

    Parameters
    ----------
    path:
        Explicit path to a YAML file containing a ``regime_weights`` key.
        If None, resolves to ``configs/strategy_params.yml`` relative to the
        macroregime package root.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from integer regime label to {symbol: weight} dict.

    Raises
    ------
    ValueError
        If any regime's weights do not sum to 1.0 ± 1e-9 or contain negative values.
    FileNotFoundError
        If the resolved YAML file does not exist.
    """
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG

    if not resolved.exists():
        raise FileNotFoundError(f"strategy_params.yml not found at: {resolved}")

    with resolved.open("r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    raw: dict[Any, Any] = cfg.get("regime_weights", {})
    if not raw:
        raise ValueError("strategy_params.yml missing 'regime_weights' key")

    result: dict[int, dict[str, float]] = {}
    for regime_key, weights_dict in raw.items():
        regime = int(regime_key)
        weights: dict[str, float] = {str(k): float(v) for k, v in weights_dict.items()}

        # Validate non-negative
        for sym, w in weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Negative weight {w} for symbol '{sym}' in regime {regime}. "
                    "Long-only allocation requires all weights >= 0."
                )

        # Validate sum = 1.0 ± 1e-9
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Regime {regime} weights sum to {total:.10f}, not 1.0. "
                "Adjust weights so they sum to exactly 1.0."
            )

        result[regime] = weights

    return result


def build_weight_schedule(
    regimes: pd.Series,
    rebalance_dates: list[pd.Timestamp],
    regime_weights: dict[int, dict[str, float]],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Build a dated weight schedule from a regime Series and rebalance dates.

    For each rebalance date, takes the most recent available regime value AT OR
    BEFORE that date (``pd.Series.asof`` semantics). Rebalance dates where the
    regime is -1 (warm-up) or unavailable (NaN) are EXCLUDED from the output.

    Parameters
    ----------
    regimes:
        Series with a DatetimeIndex and integer regime labels. Label -1 denotes
        warm-up / unassigned; -1 entries are excluded from the schedule.
    rebalance_dates:
        Ordered list of pd.Timestamp rebalance dates to build the schedule for.
    regime_weights:
        Output of ``load_regime_weights``: {regime_int: {symbol: weight}}.

    Returns
    -------
    dict[pd.Timestamp, dict[str, float]]
        Mapping from rebalance timestamp to {symbol: weight} dict.
        Keys are sorted ascending by timestamp.
    """
    schedule: dict[pd.Timestamp, dict[str, float]] = {}

    for ts in rebalance_dates:
        # as-of lookup: most recent regime at or before ts
        regime_val = regimes.asof(ts)

        # Skip warm-up (-1), NaN, or missing
        if pd.isna(regime_val):
            continue
        regime_int = int(regime_val)
        if regime_int == -1:
            continue

        weights = regime_weights.get(regime_int)
        if weights is None:
            # Unknown regime label — skip silently (non-standard regime)
            continue

        schedule[pd.Timestamp(ts)] = dict(weights)  # shallow copy

    return dict(sorted(schedule.items()))


def month_end_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the last business day of each month covered by *index*.

    Parameters
    ----------
    index:
        DatetimeIndex of trading days (business days expected).

    Returns
    -------
    list[pd.Timestamp]
        Sorted list of last-business-day-of-month timestamps present in *index*.
    """
    # Group by (year, month) and take the maximum timestamp in each group.
    # This correctly identifies the last TRADING day (which may not be calendar month-end
    # if month-end falls on a weekend/holiday).
    if len(index) == 0:
        return []

    series = pd.Series(index, index=index)
    last_by_month = series.groupby([series.index.year, series.index.month]).last()
    return sorted(last_by_month.values.tolist())
