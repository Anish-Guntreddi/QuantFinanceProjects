"""FredMacroLoader — optional real-data path via FRED ALFRED vintage API.

IMPORTANT: This module is NEVER used in tests.
The SyntheticMacroLoader is the default everywhere (offline, CI, unit tests).
FredMacroLoader is the optional real-data path for live pipelines that have
a FRED API key configured.

Design conventions (locked):
- fredapi is an optional dependency: ``pip install macroregime[real-data]``
- fredapi is NEVER imported at module scope — lazy import inside _client().
  This mirrors alpharank's yfinance pattern and ensures this module is safe
  to import in offline/CI environments without fredapi installed.
- get_series_first_release() is the only FRED fetch method used.
  Plain get_series() returns REVISED data and is explicitly forbidden here —
  it would silently introduce point-in-time leakage (03-RESEARCH.md, §ALFRED).
- Release lags are applied from release_calendar.yml (never hardcoded).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from macroregime.data.loader_base import (
    MacroLoaderBase,
    apply_release_lag,
    as_of_view,
)


class FredMacroLoader(MacroLoaderBase):
    """Real-data macro loader using FRED ALFRED first-release vintages.

    This loader is an OPTIONAL real-data path. It requires:
    1. ``pip install macroregime[real-data]`` (installs fredapi>=0.5).
    2. A FRED API key (free; https://fred.stlouisfed.org/docs/api/api_key.html).

    If neither condition is met, a clear RuntimeError is raised at instantiation,
    directing users to the default offline path (SyntheticMacroLoader).

    Parameters
    ----------
    api_key : str, optional
        FRED API key. If not provided, resolved from os.environ["FRED_API_KEY"].
    release_calendar_path : str or Path, optional
        Override path to release_calendar.yml.

    Raises
    ------
    RuntimeError
        If no API key is available (argument + env both absent).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        release_calendar_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(release_calendar_path=release_calendar_path)

        resolved_key = api_key or os.environ.get("FRED_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "FRED_API_KEY not set — use SyntheticMacroLoader (default offline path). "
                "To use real FRED data: set the FRED_API_KEY environment variable "
                "or pass api_key= explicitly. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self._api_key: str = resolved_key

    def _client(self):
        """Return a lazily-imported Fred client.

        fredapi is an OPTIONAL dependency; it is imported HERE (never at module
        scope) to ensure this module is safe to import in offline environments.
        fredapi import belongs only inside method bodies — this is a locked
        convention mirroring alpharank's yfinance lazy-import pattern.
        """
        import fredapi  # noqa: PLC0415 — intentionally lazy, never module-scope

        return fredapi.Fred(api_key=self._api_key)

    def load_series(
        self,
        series_id: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Load first-release vintage from FRED ALFRED and apply release lag.

        Uses get_series_first_release() — the most point-in-time-correct ALFRED
        method. Plain get_series() is FORBIDDEN here: it returns revised data
        and would silently break PIT correctness.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g., "CPIAUCSL", "UNRATE").
        start : pd.Timestamp, optional
            Inclusive start date on observation-date space.
        end : pd.Timestamp, optional
            Inclusive end date on observation-date space.

        Returns
        -------
        pd.Series
            Publication-date indexed series. attrs["observation_dates"] set.
        """
        lag_days = self._calendar.get(series_id)
        if lag_days is None:
            raise KeyError(
                f"Series '{series_id}' not in release_calendar.yml. "
                f"Known series: {list(self._calendar.keys())}"
            )

        # Fetch first-release (most PIT-correct) vintage from ALFRED.
        # NEVER use self._client().get_series(series_id) — that returns revised data.
        raw: pd.Series = self._client().get_series_first_release(series_id)
        raw.index = pd.DatetimeIndex(raw.index)

        # Slice on observation dates before applying lag
        if start is not None:
            raw = raw[raw.index >= pd.Timestamp(start)]
        if end is not None:
            raw = raw[raw.index <= pd.Timestamp(end)]

        raw.name = series_id
        pit = apply_release_lag(raw, lag_days)
        return pit
