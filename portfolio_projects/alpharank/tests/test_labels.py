"""Label construction tests — plan 02-03."""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: build a tiny monthly-prices DataFrame
# ---------------------------------------------------------------------------

def _prices_3assets():
    """3-asset 4-row monthly price series for hand-computed validation.

    Prices at t=0..3.  Forward 1-month returns at t=0:
      A: (120 - 100) / 100 = 0.20  (highest)
      B: (99  - 100) / 100 = -0.01 (lowest)
      C: (110 - 100) / 100 = 0.10  (middle)
    With 3 assets, percentile ranks are 1/3, 2/3, 3/3.
    Sorted ascending: B (-1%) rank=1/3, C (1%) rank=2/3, A (20%) rank=3/3.
    """
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    data = {
        "A": [100.0, 120.0, 115.0, 110.0],
        "B": [100.0,  99.0, 102.0, 101.0],
        "C": [100.0, 110.0, 108.0, 106.0],
    }
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------

def test_forward_rank_labels_hand_computed():
    """Labels at t=0 match the hand-computed percentile ranks for a 3-asset case."""
    from alpharank.labels.forward_returns import make_labels

    prices = _prices_3assets()
    labels = make_labels(prices, horizon=1)

    row0 = labels.iloc[0]  # labels at t=0 based on 1-month forward returns

    # Forward returns at t=0: A=+20%, B=-1%, C=+10%
    # Cross-sectional rank (pct=True, ascending): B=1/3, C=2/3, A=1.0
    assert abs(row0["A"] - 1.0) < 1e-10, f"Expected A=1.0, got {row0['A']}"
    assert abs(row0["B"] - 1 / 3) < 1e-10, f"Expected B=1/3, got {row0['B']}"
    assert abs(row0["C"] - 2 / 3) < 1e-10, f"Expected C=2/3, got {row0['C']}"


def test_label_nan_tail():
    """Last `horizon` rows of labels must be entirely NaN (no future data)."""
    from alpharank.labels.forward_returns import make_labels

    prices = _prices_3assets()

    for horizon in [1, 2, 3]:
        labels = make_labels(prices, horizon=horizon)
        tail = labels.iloc[-horizon:]
        assert tail.isna().all().all(), (
            f"horizon={horizon}: expected all-NaN tail rows, got\n{tail}"
        )
        # Non-tail should not be all NaN (sanity check)
        if len(labels) > horizon:
            non_tail = labels.iloc[: len(labels) - horizon]
            assert not non_tail.isna().all().all(), (
                f"horizon={horizon}: non-tail rows should contain valid labels"
            )


def test_label_ignores_delisted():
    """A delisted symbol (NaN forward prices) gets NaN label and does not distort others."""
    from alpharank.labels.forward_returns import make_labels

    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    # C is delisted: no price at t=2, so forward return at t=1 is NaN
    data = {
        "A": [100.0, 120.0, 115.0, 110.0],
        "B": [100.0,  99.0, 102.0, 101.0],
        "C": [100.0, 110.0, np.nan, np.nan],
    }
    prices = pd.DataFrame(data, index=idx)
    labels = make_labels(prices, horizon=1)

    # At t=1, C's forward return (t=2) is NaN — C should have NaN label
    assert np.isnan(labels.loc[prices.index[1], "C"]), "Delisted C should have NaN label at t=1"

    # A and B at t=1 should still have valid, different ranks (2 assets remain)
    a_label = labels.loc[prices.index[1], "A"]
    b_label = labels.loc[prices.index[1], "B"]
    assert not np.isnan(a_label), "A should have valid label at t=1"
    assert not np.isnan(b_label), "B should have valid label at t=1"
    # With 2 live assets, ranks should be 0.5 and 1.0
    assert set(np.round([a_label, b_label], 10)) == {0.5, 1.0}, (
        f"Expected {{0.5, 1.0}}, got A={a_label}, B={b_label}"
    )
