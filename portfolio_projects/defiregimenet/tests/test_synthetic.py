"""
Tests for defiregimenet.data.synthetic.

Covers DFR-01: 24/7 calendar, fat tails, vol clustering, determinism,
data-quality validation (warnings on injected anomalies), and lazy ccxt.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis

from defiregimenet.data.synthetic import (
    CryptoGenerator,
    CryptoPanel,
    inject_anomalies,
    validate_crypto_data,
)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def test_calendar_24_7(seeded_crypto_panel):
    """Generated index is freq='D' with no missing dates; contains weekends."""
    panel = seeded_crypto_panel
    for token in panel.tokens:
        idx = panel.ohlcv[token].index
        # No missing dates: length == (end - start).days + 1
        expected_len = (idx[-1] - idx[0]).days + 1
        assert len(idx) == expected_len, (
            f"{token}: index length {len(idx)} != expected {expected_len}"
        )
        # Must contain Saturdays (dayofweek==5) and Sundays (dayofweek==6)
        assert (idx.dayofweek == 5).any(), f"{token}: no Saturdays in index"
        assert (idx.dayofweek == 6).any(), f"{token}: no Sundays in index"
        # Frequency must be 'D'
        assert idx.freq is not None and idx.freq.freqstr == "D", (
            f"{token}: index.freq = {idx.freq!r}, expected 'D'"
        )


# ---------------------------------------------------------------------------
# Fat tails
# ---------------------------------------------------------------------------


def test_fat_tails(seeded_crypto_panel):
    """Excess kurtosis of close-to-close log returns > 1.0 for every token."""
    panel = seeded_crypto_panel
    for token in panel.tokens:
        close = panel.ohlcv[token]["close"]
        log_returns = np.log(close).diff().dropna()
        excess_kurt = kurtosis(log_returns, fisher=True)  # excess kurtosis
        assert excess_kurt > 1.0, (
            f"{token}: excess kurtosis {excess_kurt:.3f} not > 1.0"
        )


# ---------------------------------------------------------------------------
# Vol clustering
# ---------------------------------------------------------------------------


def test_vol_clustering(seeded_crypto_panel):
    """
    GARCH(1,1) fitted on one token's log returns should converge
    with alpha+beta > 0.9, confirming vol clustering.
    """
    from volsurfacelab.forecast import fit_garch_robust

    panel = seeded_crypto_panel
    token = panel.tokens[0]  # BTC
    close = panel.ohlcv[token]["close"]
    log_returns = np.log(close).diff().dropna()

    result = fit_garch_robust(log_returns)
    assert result.converged, f"{token}: GARCH did not converge"
    params = result.params
    # alpha = arch[1], beta = garch[1]
    alpha = params.get("alpha[1]", params.get("alpha1", None))
    beta = params.get("beta[1]", params.get("beta1", None))
    assert alpha is not None and beta is not None, (
        f"Could not find alpha/beta in GARCH params: {list(params.index)}"
    )
    assert alpha + beta > 0.9, (
        f"{token}: GARCH alpha+beta = {alpha+beta:.4f}, expected > 0.9"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism():
    """Two CryptoGenerator(seed=42) calls produce identical ohlcv and true_states."""
    gen1 = CryptoGenerator(seed=42, n_years=2, tokens=("BTC", "ETH"))
    gen2 = CryptoGenerator(seed=42, n_years=2, tokens=("BTC", "ETH"))
    panel1 = gen1.generate()
    panel2 = gen2.generate()

    for token in panel1.tokens:
        pd.testing.assert_frame_equal(
            panel1.ohlcv[token], panel2.ohlcv[token],
            check_exact=True,
            obj=f"ohlcv[{token}]",
        )
    np.testing.assert_array_equal(panel1.true_states, panel2.true_states)


# ---------------------------------------------------------------------------
# True states
# ---------------------------------------------------------------------------


def test_true_states_shape(seeded_crypto_panel):
    """true_states in {0,1,2,3}, all 4 states visited, mean dwell > 10 bars."""
    panel = seeded_crypto_panel
    states = panel.true_states
    # Shape matches time dimension
    n_bars = len(next(iter(panel.ohlcv.values())))
    assert states.shape == (n_bars,), (
        f"true_states shape {states.shape} != ({n_bars},)"
    )
    # All values in {0,1,2,3}
    assert set(np.unique(states)) == {0, 1, 2, 3}, (
        f"Expected states {{0,1,2,3}}, got {set(np.unique(states))}"
    )
    # Mean dwell > 10 bars: compute run lengths
    runs = []
    current_state = states[0]
    run_len = 1
    for s in states[1:]:
        if s == current_state:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
            current_state = s
    runs.append(run_len)
    mean_dwell = np.mean(runs)
    assert mean_dwell > 10, f"Mean dwell {mean_dwell:.1f} bars, expected > 10"


# ---------------------------------------------------------------------------
# Data quality validation
# ---------------------------------------------------------------------------


def test_gap_warning(small_crypto_panel):
    """inject_anomalies with gap_indices triggers UserWarning matching 'gap'."""
    panel = small_crypto_panel
    token = panel.tokens[0]
    df = panel.ohlcv[token].copy()
    df_with_gap = inject_anomalies(df, gap_indices=[10], volume_spike_indices=[])
    with pytest.warns(UserWarning, match="gap"):
        validate_crypto_data(df_with_gap)


def test_volume_anomaly_warning(small_crypto_panel):
    """Volume spike x50 triggers UserWarning matching 'volume'."""
    panel = small_crypto_panel
    token = panel.tokens[0]
    df = panel.ohlcv[token].copy()
    df_with_spike = inject_anomalies(df, gap_indices=[], volume_spike_indices=[20])
    with pytest.warns(UserWarning, match="volume"):
        validate_crypto_data(df_with_spike)


def test_clean_data_no_warnings(small_crypto_panel):
    """Pristine panel produces no UserWarnings from validate_crypto_data."""
    panel = small_crypto_panel
    token = panel.tokens[0]
    df = panel.ohlcv[token]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msgs = validate_crypto_data(df)
    assert msgs == [], f"Unexpected warnings on clean data: {msgs}"


# ---------------------------------------------------------------------------
# Lazy ccxt import
# ---------------------------------------------------------------------------


def test_ccxt_lazy():
    """
    Importing defiregimenet.data.real must NOT trigger a module-level import of ccxt.
    ccxt should not appear in sys.modules after the import.
    """
    # Remove ccxt from sys.modules if somehow pre-loaded by another test
    sys.modules.pop("ccxt", None)
    import importlib
    import defiregimenet.data.real  # noqa: F401 — side-effect import
    importlib.reload(defiregimenet.data.real)
    assert "ccxt" not in sys.modules, (
        "ccxt was imported at module scope in defiregimenet.data.real — "
        "it must be lazy (imported inside load_ccxt_panel body only)"
    )
