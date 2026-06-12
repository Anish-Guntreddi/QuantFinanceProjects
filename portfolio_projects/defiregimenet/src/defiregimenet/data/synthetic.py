"""
Deterministic synthetic crypto OHLCV generator.

Covers DFR-01: 24/7 calendar (freq='D', never 'B'), Student-t fat-tail
innovations (df=4), per-regime GARCH(1,1) vol clustering, shared 4-state
latent Markov regime sequence, market-factor + idiosyncratic composition,
OHLCV via cumprod of log-return exp.

Key conventions (locked by prior phases):
- Single np.random.default_rng(seed) drives all randomness.
- NO pct_change calls anywhere (FutureWarning-as-error in CI).
- NO freq='B' (business days) — crypto trades 24/7.
- true_states is a TEST ORACLE ONLY — never used in the pipeline for features
  or training targets (same discipline as volsurfacelab's true_iv).

State encoding (LOCKED — must match labels.py in plan 05-02):
  0 = bear / low-vol
  1 = bear / high-vol
  2 = bull / low-vol
  3 = bull / high-vol
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as scipy_t

# ---------------------------------------------------------------------------
# DGP constants
# ---------------------------------------------------------------------------

#: State encoding (LOCKED)
STATE_BEAR_LOW_VOL = 0
STATE_BEAR_HIGH_VOL = 1
STATE_BULL_LOW_VOL = 2
STATE_BULL_HIGH_VOL = 3

#: Per-state GARCH(1,1) parameters: (omega, alpha, beta)
GARCH_PARAMS: dict[int, tuple[float, float, float]] = {
    STATE_BEAR_LOW_VOL:  (0.015, 0.12, 0.85),   # bear / low-vol
    STATE_BEAR_HIGH_VOL: (0.050, 0.20, 0.75),   # bear / high-vol
    STATE_BULL_LOW_VOL:  (0.010, 0.10, 0.88),   # bull / low-vol
    STATE_BULL_HIGH_VOL: (0.020, 0.15, 0.82),   # bull / high-vol
}

#: Per-state daily drift (log-return mean)
STATE_DRIFT: dict[int, float] = {
    STATE_BEAR_LOW_VOL:  -0.0005,
    STATE_BEAR_HIGH_VOL: -0.0020,
    STATE_BULL_LOW_VOL:   0.0010,
    STATE_BULL_HIGH_VOL:  0.0008,
}

#: Vol level scaling for volume (higher vol state → more volume)
STATE_VOL_SCALE: dict[int, float] = {
    STATE_BEAR_LOW_VOL:  1.0,
    STATE_BEAR_HIGH_VOL: 2.5,
    STATE_BULL_LOW_VOL:  1.2,
    STATE_BULL_HIGH_VOL: 2.0,
}

# Markov transition matrix — 4x4, persistence ~0.97, all states ergodic.
# Row i: probability of transitioning FROM state i TO state j.
# Off-diagonal mass ensures all 4 states are visited (ergodicity guaranteed).
_TRANSITION_MATRIX: np.ndarray = np.array(
    [
        # to:  bear/lv  bear/hv  bull/lv  bull/hv
        [0.97,   0.01,   0.01,   0.01],  # from bear/lv
        [0.02,   0.96,   0.01,   0.01],  # from bear/hv
        [0.01,   0.01,   0.97,   0.01],  # from bull/lv
        [0.01,   0.01,   0.01,   0.97],  # from bull/hv
    ],
    dtype=float,
)

# Student-t degrees of freedom for fat-tail innovations
FAT_TAIL_DF: int = 4

# Market factor weight (30% idiosyncratic)
MARKET_FACTOR_WEIGHT: float = 0.7
IDIO_WEIGHT: float = 1.0 - MARKET_FACTOR_WEIGHT


# ---------------------------------------------------------------------------
# CryptoPanel frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CryptoPanel:
    """
    Immutable container for a multi-token OHLCV panel and oracle state sequence.

    Attributes:
        tokens: Token names in generation order.
        ohlcv: Dict mapping token name -> DataFrame with columns
               [open, high, low, close, volume] and DatetimeIndex freq='D'.
        true_states: 1-D array of shape (T,) — the shared latent regime
                     sequence. TEST ORACLE ONLY — never used for features
                     or training targets in the honest pipeline.
    """

    tokens: tuple[str, ...]
    ohlcv: dict[str, pd.DataFrame]
    true_states: np.ndarray


# ---------------------------------------------------------------------------
# CryptoGenerator
# ---------------------------------------------------------------------------


class CryptoGenerator:
    """
    Deterministic, seeded multi-token OHLCV generator.

    Parameters
    ----------
    seed : int
        RNG seed. Same seed → byte-identical output on repeated calls.
    n_years : int
        Approximate number of years (365 bars per year — 24/7 calendar).
    tokens : sequence of str
        Token names to generate (in order).
    fat_tail_df : int
        Student-t degrees of freedom for log-return innovations.
    market_factor_weight : float
        Weight on shared market component (0.7 → 70% market, 30% idio).
    """

    def __init__(
        self,
        seed: int = 42,
        n_years: int = 3,
        tokens: Sequence[str] = ("BTC", "ETH", "SOL", "AVAX"),
        fat_tail_df: int = FAT_TAIL_DF,
        market_factor_weight: float = MARKET_FACTOR_WEIGHT,
    ) -> None:
        self.seed = seed
        self.n_years = n_years
        self.tokens = tuple(tokens)
        self.fat_tail_df = fat_tail_df
        self.market_factor_weight = market_factor_weight

    def generate(self) -> CryptoPanel:
        """
        Generate the synthetic panel.

        Returns a CryptoPanel with:
        - ohlcv[token]: DataFrame(open, high, low, close, volume) freq='D'
        - true_states: np.ndarray shape (T,) with states in {0,1,2,3}
        """
        rng = np.random.default_rng(self.seed)
        n_bars = self.n_years * 365

        # 24/7 calendar — freq='D' includes weekends
        start = "2021-01-01"
        dates = pd.date_range(start=start, periods=n_bars, freq="D")

        # 1. Simulate shared Markov regime sequence
        true_states = _simulate_markov_chain(rng, n_bars, _TRANSITION_MATRIX)

        # 2. Simulate shared GARCH(1,1) vol path for market factor
        market_sigma = _simulate_garch_sigma(rng, true_states, self.fat_tail_df)

        # 3. Draw shared market factor innovations (Student-t)
        market_innovations = _draw_t_innovations(
            rng, n_bars, self.fat_tail_df
        )  # shape (T,)

        # 4. Compute market log-returns (drift + vol * innovation)
        market_drifts = np.array([STATE_DRIFT[s] for s in true_states])
        market_log_returns = market_drifts + market_sigma * market_innovations

        # 5. Per-token OHLCV
        ohlcv: dict[str, pd.DataFrame] = {}
        for token in self.tokens:
            # Per-token idiosyncratic sigma (redrawn per token from same regime)
            idio_sigma = _simulate_garch_sigma(rng, true_states, self.fat_tail_df)
            idio_innovations = _draw_t_innovations(rng, n_bars, self.fat_tail_df)
            idio_log_returns = market_drifts + idio_sigma * idio_innovations

            # Composite log returns
            log_returns = (
                self.market_factor_weight * market_log_returns
                + (1.0 - self.market_factor_weight) * idio_log_returns
            )

            # Build OHLCV from log-return path
            ohlcv[token] = _build_ohlcv(rng, dates, log_returns, true_states)

        return CryptoPanel(
            tokens=self.tokens,
            ohlcv=ohlcv,
            true_states=true_states,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_markov_chain(
    rng: np.random.Generator,
    n_bars: int,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """Simulate a discrete-time Markov chain of length n_bars."""
    n_states = transition_matrix.shape[0]
    states = np.empty(n_bars, dtype=np.int32)
    # Start in a uniformly random state
    states[0] = rng.integers(0, n_states)
    for t in range(1, n_bars):
        states[t] = rng.choice(n_states, p=transition_matrix[states[t - 1]])
    return states


def _simulate_garch_sigma(
    rng: np.random.Generator,
    true_states: np.ndarray,
    df: int,
) -> np.ndarray:
    """
    Simulate per-bar vol (sigma) using GARCH(1,1) recursion.
    Parameters are drawn from GARCH_PARAMS per regime state.
    Returns sigma array (not variance) of shape (T,).
    """
    n = len(true_states)
    sigma2 = np.empty(n)
    # Initialize sigma2[0] from the unconditional variance of the first state
    s0 = true_states[0]
    omega0, alpha0, beta0 = GARCH_PARAMS[s0]
    sigma2[0] = omega0 / max(1.0 - alpha0 - beta0, 1e-6)

    # Draw innovations for the recursion separately (don't consume main seed)
    innov = _draw_t_innovations(rng, n, df)

    for t in range(1, n):
        s = true_states[t]
        omega, alpha, beta = GARCH_PARAMS[s]
        r_prev = innov[t - 1] * np.sqrt(sigma2[t - 1])
        sigma2[t] = omega + alpha * r_prev**2 + beta * sigma2[t - 1]

    return np.sqrt(np.maximum(sigma2, 1e-10))


def _draw_t_innovations(
    rng: np.random.Generator,
    n: int,
    df: int,
) -> np.ndarray:
    """
    Draw n standardized Student-t innovations with given df.
    Uses scipy.stats.t.rvs with random_state=rng for seeded determinism.
    """
    raw = scipy_t.rvs(df=df, size=n, random_state=rng)
    # Standardize: Student-t has std = sqrt(df/(df-2))
    std_factor = np.sqrt(df / (df - 2))
    return raw / std_factor


def _build_ohlcv(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    log_returns: np.ndarray,
    true_states: np.ndarray,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """
    Construct a DataFrame with columns [open, high, low, close, volume].

    Close path: cumulative product of exp(log_return).
    OHLC: open = previous close * exp(small gap noise)
          high/low = close * exp(+/- |intraday noise|)
    Volume: log-normal baseline scaled by regime vol level.
    NO pct_change used anywhere.
    """
    n = len(dates)

    # Close path from cumulative log-returns (no pct_change)
    close_path = initial_price * np.exp(np.cumsum(log_returns))

    # Intraday noise amplitude: small fraction of daily vol
    intraday_noise = np.abs(rng.normal(0.0, 0.005, size=n))
    # Open: previous close * exp(gap noise), first bar uses initial_price
    gap_noise = rng.normal(0.0, 0.002, size=n)
    open_path = np.empty(n)
    open_path[0] = initial_price * np.exp(gap_noise[0])
    open_path[1:] = close_path[:-1] * np.exp(gap_noise[1:])

    high_path = np.maximum(close_path, open_path) * np.exp(intraday_noise)
    low_path = np.minimum(close_path, open_path) * np.exp(-intraday_noise)

    # Volume: log-normal baseline scaled by regime
    vol_scales = np.array([STATE_VOL_SCALE[s] for s in true_states])
    log_volume = rng.normal(loc=10.0, scale=0.5, size=n)
    volume = np.exp(log_volume) * vol_scales

    df = pd.DataFrame(
        {
            "open": open_path,
            "high": high_path,
            "low": low_path,
            "close": close_path,
            "volume": volume,
        },
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# Data quality validation
# ---------------------------------------------------------------------------


def validate_crypto_data(ohlcv: pd.DataFrame) -> list[str]:
    """
    Validate OHLCV data quality.

    Emits UserWarning for each issue found. Never raises.
    Returns the list of warning messages.

    Checks:
    - Calendar gaps (missing daily dates / NaN bars)
    - Non-positive prices (open, high, low, close)
    - Volume anomalies (volume > 20x rolling 30-bar median)
    - Zero-volume runs (5+ consecutive zero-volume bars)

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with columns [open, high, low, close, volume] and
        DatetimeIndex. Index must be sorted.

    Returns
    -------
    list of str
        Warning messages for all detected issues (empty if clean).
    """
    messages: list[str] = []

    idx = ohlcv.index
    if not isinstance(idx, pd.DatetimeIndex):
        msg = "calendar gap: index is not a DatetimeIndex"
        warnings.warn(msg, UserWarning, stacklevel=2)
        messages.append(msg)
        return messages

    # Calendar gap check: NaN rows or missing dates
    nan_rows = ohlcv.isna().any(axis=1)
    if nan_rows.any():
        n_nan = int(nan_rows.sum())
        msg = f"calendar gap: {n_nan} row(s) with NaN values detected"
        warnings.warn(msg, UserWarning, stacklevel=2)
        messages.append(msg)

    if len(idx) > 1:
        expected_len = (idx[-1] - idx[0]).days + 1
        if len(idx) < expected_len:
            n_missing = expected_len - len(idx)
            msg = (
                f"calendar gap: {n_missing} missing date(s) "
                f"in [{idx[0].date()} — {idx[-1].date()}]"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            messages.append(msg)

    # Non-positive prices
    price_cols = [c for c in ["open", "high", "low", "close"] if c in ohlcv.columns]
    for col in price_cols:
        bad = ohlcv[col] <= 0
        if bad.any():
            n_bad = int(bad.sum())
            msg = f"non-positive price: {n_bad} row(s) in column '{col}'"
            warnings.warn(msg, UserWarning, stacklevel=2)
            messages.append(msg)

    # Volume anomaly: volume > 20x rolling 30-bar median
    if "volume" in ohlcv.columns:
        volume = ohlcv["volume"]

        # Per-bar non-positive volume check (isolated zero or negative bars)
        nonpos_vol = volume <= 0
        if nonpos_vol.any():
            n_nonpos = int(nonpos_vol.sum())
            msg = f"volume anomaly: {n_nonpos} bar(s) with non-positive volume (<= 0)"
            warnings.warn(msg, UserWarning, stacklevel=2)
            messages.append(msg)

        rolling_med = volume.rolling(window=30, min_periods=1).median()
        spike_mask = volume > 20.0 * rolling_med
        if spike_mask.any():
            n_spikes = int(spike_mask.sum())
            msg = (
                f"volume anomaly: {n_spikes} bar(s) exceed 20x rolling "
                f"30-bar median"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            messages.append(msg)

        # Zero-volume runs of 5+
        zero_vol = (volume == 0.0).astype(int)
        if zero_vol.any():
            run_len = 0
            max_run = 0
            for v in zero_vol:
                run_len = run_len + 1 if v else 0
                max_run = max(max_run, run_len)
            if max_run >= 5:
                msg = f"volume anomaly: zero-volume run of {max_run} bars detected"
                warnings.warn(msg, UserWarning, stacklevel=2)
                messages.append(msg)

    return messages


# ---------------------------------------------------------------------------
# Anomaly injection (for testing data-quality validation)
# ---------------------------------------------------------------------------


def inject_anomalies(
    df: pd.DataFrame,
    gap_indices: list[int],
    volume_spike_indices: list[int],
) -> pd.DataFrame:
    """
    Inject synthetic anomalies into an OHLCV DataFrame (returns a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Clean OHLCV DataFrame.
    gap_indices : list of int
        Row positions to set to NaN (calendar gap simulation).
    volume_spike_indices : list of int
        Row positions to multiply volume by 50 (anomalous volume spike).

    Returns
    -------
    pd.DataFrame
        Modified copy; original is unchanged.
    """
    out = df.copy()
    for i in gap_indices:
        out.iloc[i] = np.nan
    if "volume" in out.columns:
        for i in volume_spike_indices:
            out.iloc[i, out.columns.get_loc("volume")] *= 50
    return out
