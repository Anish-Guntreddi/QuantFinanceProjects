"""Optional real crypto data loader via ccxt.

LAZY IMPORT CONTRACT: ccxt is imported INSIDE load_ccxt_panel only — never at
module scope. The test suite asserts that importing this module does not pull
ccxt into sys.modules. All tests run fully offline against the synthetic
generator; this loader exists for optional live research use only.
"""

from __future__ import annotations

import warnings

import pandas as pd

from defiregimenet.data.synthetic import validate_crypto_data

__all__ = ["load_ccxt_panel"]


def load_ccxt_panel(
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT"),
    exchange_id: str = "binance",
    timeframe: str = "1d",
    limit: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for ``symbols`` from a ccxt exchange.

    Network access — NEVER called from the test suite. Each fetched frame is
    routed through validate_crypto_data; any quality findings are surfaced as
    UserWarning (same contract as the synthetic path).

    Parameters
    ----------
    symbols:
        ccxt market symbols, e.g. ("BTC/USDT", "ETH/USDT").
    exchange_id:
        ccxt exchange identifier (default "binance").
    timeframe:
        ccxt OHLCV timeframe (default "1d" — matches the synthetic 24/7
        daily calendar).
    limit:
        Maximum number of bars per symbol.

    Returns
    -------
    dict[str, pd.DataFrame]
        {symbol: OHLCV frame} with columns open/high/low/close/volume and a
        UTC DatetimeIndex.

    Raises
    ------
    ImportError
        If ccxt is not installed (it is an optional dependency).
    """
    try:
        import ccxt  # noqa: PLC0415 — lazy by contract (see module docstring)
    except ImportError as exc:  # pragma: no cover — optional dependency
        raise ImportError(
            "ccxt is required for load_ccxt_panel but is not installed. "
            "Install it with `pip install ccxt`. All research and tests run "
            "offline via defiregimenet.data.synthetic.CryptoGenerator — ccxt "
            "is only needed for optional live-data runs."
        ) from exc

    exchange = getattr(ccxt, exchange_id)()
    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        findings = validate_crypto_data(df)
        for finding in findings:
            warnings.warn(f"{symbol}: {finding}", UserWarning, stacklevel=2)
        panel[symbol] = df

    return panel
