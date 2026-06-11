"""Point-in-time macro data loader tests.

Tests the PIT mask oracle, release-lag correctness, and loader interface.
All tests run fully offline — no FRED API key required, no network access.

Implements plan 03-02 Wave-0 stubs.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Task 1: MacroLoaderBase + release-lag/as-of utilities + SyntheticMacroLoader
# ---------------------------------------------------------------------------


def test_point_in_time_mask():
    """Release-lag mask: value for month m invisible before m+lag, visible at m+lag.

    Parametrized over all series in release_calendar.yml (reads actual lags).
    For each series the test:
    - Builds a 12-month synthetic PIT series using the configured lag.
    - Picks month-end m = first observation.
    - Checks: as_of_view at m + (lag - 1) days does NOT contain that value.
    - Checks: as_of_view at m + lag days DOES contain that value.
    """
    from macroregime.data.loader_base import (
        apply_release_lag,
        as_of_view,
        load_release_calendar,
    )

    calendar = load_release_calendar()
    assert len(calendar) > 0, "release_calendar.yml must define at least one series"

    for series_id, lag_days in calendar.items():
        # Build a small observation-date monthly series (12 months)
        obs_index = pd.date_range("2020-01-31", periods=12, freq="ME")
        values = np.arange(1.0, 13.0)
        obs_series = pd.Series(values, index=obs_index, name=series_id)

        # Apply release lag: index shifts by lag_days
        pit_series = apply_release_lag(obs_series, lag_days)

        # Use the first observation month-end as reference
        m = obs_index[0]  # e.g., 2020-01-31
        pub_date = m + pd.Timedelta(days=lag_days)

        # One day before publication: value must NOT be visible
        view_before = as_of_view(pit_series, pub_date - pd.Timedelta(days=1))
        assert m not in obs_series.index[: len(view_before)] or len(view_before) == 0, (
            f"{series_id} (lag={lag_days}d): value published at {pub_date} "
            f"incorrectly visible one day before"
        )
        # Exact check: publication date - 1 should have 0 rows (first pub date)
        assert len(view_before) == 0, (
            f"{series_id} (lag={lag_days}d): expected 0 rows at {pub_date - pd.Timedelta(days=1)}, "
            f"got {len(view_before)}"
        )

        # On publication date: exactly 1 row must be visible
        view_on = as_of_view(pit_series, pub_date)
        assert len(view_on) == 1, (
            f"{series_id} (lag={lag_days}d): expected 1 row at {pub_date}, "
            f"got {len(view_on)}"
        )
        assert view_on.iloc[0] == pytest.approx(1.0), (
            f"{series_id}: first row value mismatch"
        )


def test_no_future_observation():
    """SyntheticMacroLoader.load_panel: no row in PIT view has pub_date > as_of.

    Also checks that original observation dates are all <= as_of - lag_days
    (i.e., we can't see values whose observation period hasn't ended + lag passed).
    """
    from macroregime.data.loader_base import (
        SyntheticMacroLoader,
        load_release_calendar,
    )
    from macroregime.data.synthetic import SyntheticMacroGenerator

    gen = SyntheticMacroGenerator(n_years=5, seed=42)
    loader = SyntheticMacroLoader(generator=gen)
    calendar = load_release_calendar()

    # Sample a grid of as-of dates spanning the panel
    panel_full = loader.load_panel()
    all_pub_dates = panel_full.index
    # Pick every 13th row to keep test fast
    test_dates = all_pub_dates[::13]

    for as_of in test_dates:
        view = loader.load_panel(as_of=as_of)
        if view.empty:
            continue
        # All publication-date rows must be <= as_of
        assert (view.index <= as_of).all(), (
            f"Future publication dates found in as_of view at {as_of}: "
            f"{view.index[view.index > as_of].tolist()}"
        )

    # Spot-check a single series: observation dates must respect lag constraint
    for series_id, lag_days in list(calendar.items())[:2]:
        series_pit = loader.load_series(series_id)
        # Observation dates stored in attrs
        obs_dates = series_pit.attrs.get("observation_dates")
        assert obs_dates is not None, (
            f"load_series({series_id}) must store observation_dates in attrs"
        )
        # For each (pub_date, obs_date) pair: obs_date + lag_days should equal pub_date
        expected_pub = obs_dates + pd.Timedelta(days=lag_days)
        pd.testing.assert_index_equal(
            series_pit.index,
            pd.DatetimeIndex(expected_pub),
            check_names=False,
        )


def test_loader_interface():
    """MacroLoaderBase interface: isinstance checks, return types, attrs.

    FredMacroLoader subclasses the same ABC (class check only — no instantiation).
    """
    from macroregime.data.loader_base import MacroLoaderBase, SyntheticMacroLoader
    from macroregime.data.fred_loader import FredMacroLoader
    from macroregime.data.synthetic import SyntheticMacroGenerator

    gen = SyntheticMacroGenerator(n_years=5, seed=42)
    loader = SyntheticMacroLoader(generator=gen)

    # SyntheticMacroLoader is a MacroLoaderBase instance
    assert isinstance(loader, MacroLoaderBase), (
        "SyntheticMacroLoader must be an instance of MacroLoaderBase"
    )

    # FredMacroLoader is a subclass of MacroLoaderBase (class check only)
    assert issubclass(FredMacroLoader, MacroLoaderBase), (
        "FredMacroLoader must subclass MacroLoaderBase"
    )

    # load_series returns a pd.Series with publication-date index and attrs
    series_id = "CPIAUCSL"
    s = loader.load_series(series_id)
    assert isinstance(s, pd.Series), f"load_series must return pd.Series, got {type(s)}"
    assert isinstance(s.index, pd.DatetimeIndex), (
        "load_series index must be DatetimeIndex (publication dates)"
    )
    assert "observation_dates" in s.attrs, (
        "load_series must store observation_dates in series.attrs"
    )
    assert isinstance(s.attrs["observation_dates"], pd.DatetimeIndex), (
        "attrs['observation_dates'] must be a pd.DatetimeIndex"
    )

    # load_panel returns a DataFrame with DatetimeIndex
    panel = loader.load_panel()
    assert isinstance(panel, pd.DataFrame), (
        f"load_panel must return pd.DataFrame, got {type(panel)}"
    )
    assert isinstance(panel.index, pd.DatetimeIndex), (
        "load_panel index must be DatetimeIndex"
    )

    # load_panel with as_of filters correctly
    if len(panel) > 0:
        mid_date = panel.index[len(panel) // 2]
        view = loader.load_panel(as_of=mid_date)
        assert (view.index <= mid_date).all(), (
            "load_panel(as_of=...) must return only rows with index <= as_of"
        )


# ---------------------------------------------------------------------------
# Task 2: FredMacroLoader — offline tests (no network, no fredapi import)
# ---------------------------------------------------------------------------


def test_fred_loader_requires_key(monkeypatch):
    """FredMacroLoader raises RuntimeError when no API key is configured.

    No network is touched — the error must occur at instantiation before
    any fredapi call.
    """
    from macroregime.data.fred_loader import FredMacroLoader

    # Remove FRED_API_KEY from environment
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        FredMacroLoader()


def test_fredapi_not_imported_at_module_scope():
    """Importing fred_loader must NOT import fredapi at module scope.

    This ensures the optional dependency is truly lazy — the module is safe
    to import in offline/CI environments where fredapi is not installed.
    """
    # Remove fredapi from sys.modules if it was previously imported
    sys.modules.pop("fredapi", None)

    import importlib
    import macroregime.data.fred_loader  # noqa: F401

    # Force reimport so we can check state right after import
    importlib.reload(macroregime.data.fred_loader)

    assert "fredapi" not in sys.modules, (
        "fredapi must NOT be imported at module scope in fred_loader.py; "
        "use a lazy import inside a method body instead"
    )
