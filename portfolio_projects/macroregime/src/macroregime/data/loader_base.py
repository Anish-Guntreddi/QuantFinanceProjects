"""Point-in-time macro data loader base — MacroLoaderBase ABC, release-lag utilities.

Point-in-time (PIT) correctness is one of the two phase-defining leakage controls
(MCR-02). Release lag MUST be applied BEFORE any resample/ffill/join downstream;
ffill must happen AFTER lag (Pitfall 5 in 03-RESEARCH.md).

Public API
----------
load_release_calendar(path) -> dict[str, int]
    Load per-series lag_days from configs/release_calendar.yml.

apply_release_lag(series, lag_days) -> pd.Series
    Shift observation-date index forward by lag_days → publication-date index.
    Preserves original observation dates in series.attrs["observation_dates"].

as_of_view(series_pit, as_of) -> pd.Series
    Return rows with publication index <= as_of (no future data).

MacroLoaderBase(ABC)
    Abstract interface: load_series / load_panel.

SyntheticMacroLoader(MacroLoaderBase)
    Default offline loader backed by SyntheticMacroGenerator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Release calendar helpers
# ---------------------------------------------------------------------------

# Default path: package is at src/macroregime/data/loader_base.py
# configs/ is 3 levels up (data/ → macroregime/ → src/ → package root) + one more
# path: data/loader_base.py  → parents[0]=data, [1]=macroregime, [2]=src, [3]=package_root
_DEFAULT_CALENDAR_PATH = (
    Path(__file__).parents[3] / "configs" / "release_calendar.yml"
)


def load_release_calendar(
    path: Optional[Union[str, Path]] = None,
) -> dict[str, int]:
    """Load per-series release lag from release_calendar.yml.

    Parameters
    ----------
    path : str or Path, optional
        Override path to the YAML file. Defaults to
        ``<package_root>/configs/release_calendar.yml``.

    Returns
    -------
    dict[str, int]
        Mapping of series_id → lag_days (calendar days from observation
        month-end to public availability).
    """
    resolved = Path(path) if path is not None else _DEFAULT_CALENDAR_PATH
    with open(resolved, "r") as fh:
        raw = yaml.safe_load(fh)

    result: dict[str, int] = {}
    for series_id, info in raw.items():
        if isinstance(info, dict):
            result[series_id] = int(info["lag_days"])
        else:
            result[series_id] = int(info)
    return result


# ---------------------------------------------------------------------------
# Core PIT utilities
# ---------------------------------------------------------------------------


def apply_release_lag(series: pd.Series, lag_days: int) -> pd.Series:
    """Shift observation-date index by lag_days → publication-date index.

    The original observation dates are preserved in
    ``series.attrs["observation_dates"]`` so downstream code can assert
    PIT properties against both the publication index and the observation dates.

    Parameters
    ----------
    series : pd.Series
        Monthly (or other frequency) series with observation-date index.
    lag_days : int
        Calendar days between end of observation period and data availability.

    Returns
    -------
    pd.Series
        Same values; index is observation_index + Timedelta(lag_days).
        attrs["observation_dates"] = original DatetimeIndex.
    """
    observation_dates = pd.DatetimeIndex(series.index)
    new_index = observation_dates + pd.Timedelta(days=lag_days)
    pit = pd.Series(series.values, index=new_index, name=series.name)
    pit.attrs["observation_dates"] = observation_dates
    return pit


def as_of_view(series_pit: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Return rows with publication index <= as_of (mask future data).

    Parameters
    ----------
    series_pit : pd.Series
        Series with publication-date index (after apply_release_lag).
    as_of : pd.Timestamp
        The evaluation date. Only rows published on or before this date
        are returned.

    Returns
    -------
    pd.Series
        Subset of series_pit with index <= as_of.
    """
    return series_pit[series_pit.index <= as_of]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class MacroLoaderBase(ABC):
    """Abstract base class for macro data loaders.

    All concrete loaders (SyntheticMacroLoader, FredMacroLoader) share this
    interface. The public-date index convention is enforced here:

    - ``load_series`` MUST return a pd.Series with a publication-date index
      (i.e., observation_date + lag_days).
    - ``load_series`` MUST store original observation dates in
      ``series.attrs["observation_dates"]``.
    - ``load_panel`` is a concrete helper that calls ``load_series`` per series
      and outer-joins on publication dates (NO ffill — frequency handling and
      ffill are the pipeline's job in plan 03-07).
    """

    def __init__(
        self,
        release_calendar_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._calendar: dict[str, int] = load_release_calendar(
            release_calendar_path
        )

    @abstractmethod
    def load_series(
        self,
        series_id: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Load a single macro series with publication-date index.

        Parameters
        ----------
        series_id : str
            Identifier matching a key in release_calendar.yml.
        start : pd.Timestamp, optional
            Inclusive start date (observation-date space).
        end : pd.Timestamp, optional
            Inclusive end date (observation-date space).

        Returns
        -------
        pd.Series
            Publication-date indexed series. attrs["observation_dates"] set.
        """
        ...

    def load_panel(
        self,
        series_ids: Optional[list[str]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Load multiple series and outer-join on publication dates.

        NO ffill is applied — frequency alignment and forward-filling are
        the pipeline's responsibility (plan 03-07), after lag application.

        Parameters
        ----------
        series_ids : list[str], optional
            Series to load. Defaults to all series in the release calendar
            excluding USREC (evaluation-only flag).
        start, end : pd.Timestamp, optional
            Date range in observation-date space.
        as_of : pd.Timestamp, optional
            If provided, apply as_of_view masking after join.

        Returns
        -------
        pd.DataFrame
            Columns = series_ids; index = sorted union of all publication dates.
        """
        if series_ids is None:
            # Default: all feature series (exclude USREC — evaluation-only)
            series_ids = [
                sid for sid in self._calendar if sid != "USREC"
            ]

        frames: list[pd.Series] = []
        for sid in series_ids:
            try:
                s = self.load_series(sid, start=start, end=end)
                if as_of is not None:
                    s = as_of_view(s, as_of)
                # Clear attrs before concat: pd.concat compares attrs across
                # all Series using == which fails when attrs values are arrays
                # (e.g., DatetimeIndex). The observation_dates attr is
                # load_series metadata, not a DataFrame column property.
                s_clean = s.copy()
                s_clean.attrs = {}
                frames.append(s_clean)
            except (KeyError, ValueError):
                # Series not available in this loader; skip gracefully
                continue

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, axis=1, join="outer")
        panel.index.name = "pub_date"
        return panel


# ---------------------------------------------------------------------------
# SyntheticMacroLoader — default offline loader
# ---------------------------------------------------------------------------


class SyntheticMacroLoader(MacroLoaderBase):
    """Offline macro loader backed by SyntheticMacroGenerator.

    This is the DEFAULT loader used everywhere in tests and offline pipelines.
    No network access; fully deterministic given a seed.

    Parameters
    ----------
    generator : SyntheticMacroGenerator, optional
        Pre-built generator instance. If not provided, one is created from
        ``seed``.
    seed : int, optional
        RNG seed passed to a new SyntheticMacroGenerator. Ignored when
        ``generator`` is provided directly.
    release_calendar_path : str or Path, optional
        Override path to release_calendar.yml.
    """

    def __init__(
        self,
        generator=None,
        seed: int = 42,
        release_calendar_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(release_calendar_path=release_calendar_path)
        if generator is not None:
            self._generator = generator
        else:
            from macroregime.data.synthetic import SyntheticMacroGenerator

            self._generator = SyntheticMacroGenerator(seed=seed)
        # Generate once; cache the panel
        self._panel = self._generator.generate()

    def load_series(
        self,
        series_id: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Load a single synthetic series and apply its configured release lag.

        Parameters
        ----------
        series_id : str
            Must be a column in the synthetic macro panel
            (CPIAUCSL, UNRATE, GDPC1, T10Y2Y).
        start : pd.Timestamp, optional
            Slice on observation-date index (before lag application).
        end : pd.Timestamp, optional
            Slice on observation-date index (before lag application).

        Returns
        -------
        pd.Series
            Publication-date index; attrs["observation_dates"] = original index.
        """
        if series_id not in self._panel.macro.columns:
            raise KeyError(
                f"Series '{series_id}' not in synthetic panel. "
                f"Available: {list(self._panel.macro.columns)}"
            )

        lag_days = self._calendar.get(series_id)
        if lag_days is None:
            raise KeyError(
                f"Series '{series_id}' not found in release_calendar.yml. "
                f"Known series: {list(self._calendar.keys())}"
            )

        raw: pd.Series = self._panel.macro[series_id].copy()

        # Slice on observation dates (before lag)
        if start is not None:
            raw = raw[raw.index >= pd.Timestamp(start)]
        if end is not None:
            raw = raw[raw.index <= pd.Timestamp(end)]

        pit = apply_release_lag(raw, lag_days)
        return pit
