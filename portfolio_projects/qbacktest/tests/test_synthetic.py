"""Tests for SyntheticOHLCVGenerator (QBT-10).

All five behaviors tested:
  - test_determinism: same seed produces identical DataFrames
  - test_different_seeds: different seeds produce different close paths
  - test_multi_asset_independent_paths: 3 symbols produce mutually different close series
  - test_ohlc_sanity: high >= max(open, close) and low <= min(open, close); volume > 0
  - test_shape_and_index: n_bars rows, business-day DatetimeIndex, correct columns
"""

import pandas as pd
import pytest

from qbacktest.data.synthetic import SyntheticOHLCVGenerator


SYMBOLS = ["AAPL", "MSFT", "GOOG"]
N_BARS = 252
SEED = 42


def _make_gen(seed: int = SEED, symbols: list[str] | None = None) -> SyntheticOHLCVGenerator:
    return SyntheticOHLCVGenerator(
        symbols=symbols or SYMBOLS,
        n_bars=N_BARS,
        seed=seed,
    )


def test_determinism():
    """Two generators with seed=42 produce identical dict[str, DataFrame]."""
    gen1 = _make_gen()
    gen2 = _make_gen()
    bars1 = gen1.generate()
    bars2 = gen2.generate()

    assert set(bars1.keys()) == set(bars2.keys()), "Symbol keys differ"
    for symbol in SYMBOLS:
        pd.testing.assert_frame_equal(bars1[symbol], bars2[symbol])


def test_different_seeds():
    """seed=42 vs seed=43 produce different close paths."""
    bars_42 = _make_gen(seed=42).generate()
    bars_43 = _make_gen(seed=43).generate()

    # At least one symbol must differ in close prices
    any_differ = any(
        not bars_42[s]["close"].equals(bars_43[s]["close"]) for s in SYMBOLS
    )
    assert any_differ, "Different seeds produced identical close paths"


def test_multi_asset_independent_paths():
    """3 symbols produce mutually different close series."""
    bars = _make_gen().generate()
    closes = {s: bars[s]["close"].values for s in SYMBOLS}

    # All pairs must differ
    syms = list(SYMBOLS)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            assert not (closes[syms[i]] == closes[syms[j]]).all(), (
                f"Symbols {syms[i]} and {syms[j]} have identical close series"
            )


def test_ohlc_sanity():
    """high >= max(open, close) and low <= min(open, close); volume > 0 for every bar."""
    bars = _make_gen().generate()
    for symbol, df in bars.items():
        assert (df["high"] >= df["open"]).all(), f"{symbol}: high < open"
        assert (df["high"] >= df["close"]).all(), f"{symbol}: high < close"
        assert (df["low"] <= df["open"]).all(), f"{symbol}: low > open"
        assert (df["low"] <= df["close"]).all(), f"{symbol}: low > close"
        assert (df["volume"] > 0).all(), f"{symbol}: volume <= 0"


def test_shape_and_index():
    """n_bars rows, business-day DatetimeIndex, columns exactly [open, high, low, close, volume]."""
    bars = _make_gen().generate()
    expected_cols = ["open", "high", "low", "close", "volume"]

    for symbol, df in bars.items():
        assert len(df) == N_BARS, f"{symbol}: expected {N_BARS} rows, got {len(df)}"
        assert list(df.columns) == expected_cols, (
            f"{symbol}: columns {list(df.columns)} != {expected_cols}"
        )
        assert isinstance(df.index, pd.DatetimeIndex), (
            f"{symbol}: index is not DatetimeIndex"
        )
        # Verify business-day index (all dates are weekdays Mon-Fri)
        assert (df.index.dayofweek < 5).all(), (
            f"{symbol}: index contains non-business days"
        )
