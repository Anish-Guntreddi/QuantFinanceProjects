"""Deterministic synthetic cross-sectional data generator with planted factor alpha.

Design (Pattern 6 — verified research design):
  - A single seeded np.random.default_rng instance drives all randomness.
    No global np.random calls are made anywhere in this module.
  - Planted alpha formula (LOCKED):
        alpha = IC_target * sigma_noise / sqrt(1 - IC_target**2)
    with sigma_noise = monthly_vol.
  - Monthly log-return:
        r[t, i] = alpha_mom * mom_loading[i] + alpha_val * val_loading[i]
                  + monthly_vol * standard_normal
    Because loadings are stable across time, the trailing 12-month price
    momentum (computed from prices in plan 02-02) is a noisy proxy for
    mom_loading — this is what makes the momentum FEATURE recover the planted
    IC.  The planted IC is *recoverable* from momentum features, not the
    loadings themselves.
  - Daily bars exactly decompose the monthly log-return:
        daily_log = monthly_log / n_days + eps   (eps re-centered to sum=0)
    So sum(daily_log) = monthly_log to floating-point precision.
  - Delist: each asset delists at most once; after its delist month the
    OHLCV DataFrame has no further rows (qbacktest expects per-symbol
    frames that simply end).
  - Fundamentals:
        book_to_market = val_loading[i] + 0.2 * noise  (slow-moving noisy proxy)
        quality        = quality_loading[i] + noise     (NO planted alpha —
                         quality IC ≈ 0 by construction — honest negative control)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPanel:
    """Container returned by CrossSectionalGenerator.generate().

    Attributes
    ----------
    ohlcv : dict[str, pd.DataFrame]
        Per-symbol DataFrames with columns [open, high, low, close, volume]
        and a DatetimeIndex of business days.  Assets that delist before the
        final month have shorter frames (they simply end — no NaN rows).
    fundamentals : pd.DataFrame
        Monthly frequency, MultiIndex (month_end, symbol), columns
        [book_to_market, quality].
    monthly_returns : pd.DataFrame
        month_end × symbol log-returns.  NaN for months after delist.
    mom_loading : pd.Series
        Ground-truth momentum loading per symbol (stable across time).
    val_loading : pd.Series
        Ground-truth value loading per symbol.
    delist_month : pd.Series
        Month-end timestamp of each symbol's delist event, or NaT if alive.
    """

    ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    mom_loading: pd.Series = field(default_factory=pd.Series)
    val_loading: pd.Series = field(default_factory=pd.Series)
    delist_month: pd.Series = field(default_factory=pd.Series)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CrossSectionalGenerator:
    """Generate a deterministic synthetic cross-sectional equity universe.

    Parameters
    ----------
    n_assets : int
        Number of assets in the universe.
    n_months : int
        Number of calendar months of data to generate.
    seed : int
        Seed for the single default_rng instance.
    momentum_ic_target : float
        Target Spearman IC between mom_loading and next-month returns.
    value_ic_target : float
        Target Spearman IC between val_loading and next-month returns.
    monthly_vol : float
        Idiosyncratic monthly return standard deviation (sigma_noise).
    delist_prob_annual : float
        Annual probability that any given asset delists.
    start : str
        First month-end date (YYYY-MM-DD).  Must be a valid month-end.
    """

    def __init__(
        self,
        n_assets: int = 50,
        n_months: int = 60,
        seed: int = 42,
        momentum_ic_target: float = 0.06,
        value_ic_target: float = 0.04,
        monthly_vol: float = 0.04,
        delist_prob_annual: float = 0.03,
        start: str = "2018-01-31",
    ) -> None:
        self.n_assets = n_assets
        self.n_months = n_months
        self.seed = seed
        self.momentum_ic_target = momentum_ic_target
        self.value_ic_target = value_ic_target
        self.monthly_vol = monthly_vol
        self.delist_prob_annual = delist_prob_annual
        self.start = start

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> SyntheticPanel:
        """Generate the full synthetic panel.

        Returns
        -------
        SyntheticPanel
        """
        rng = np.random.default_rng(self.seed)
        symbols = [f"A{i:03d}" for i in range(self.n_assets)]

        # 1. Stable per-asset factor loadings (demeaned, unit-variance)
        mom_loading = self._unit_normal_loading(rng, self.n_assets)
        val_loading = self._unit_normal_loading(rng, self.n_assets)
        quality_loading = self._unit_normal_loading(rng, self.n_assets)  # no planted alpha

        mom_loading_s = pd.Series(mom_loading, index=symbols, name="mom_loading")
        val_loading_s = pd.Series(val_loading, index=symbols, name="val_loading")

        # 2. Alpha coefficients (LOCKED formula)
        alpha_mom = self._alpha_coeff(self.momentum_ic_target, self.monthly_vol)
        alpha_val = self._alpha_coeff(self.value_ic_target, self.monthly_vol)

        # 3. Month-end date range
        month_ends = self._month_ends()

        # 4. Delist events
        monthly_delist_prob = 1.0 - (1.0 - self.delist_prob_annual) ** (1.0 / 12.0)
        delist_month_map: dict[str, pd.Timestamp | None] = {}
        for sym in symbols:
            delist_month_map[sym] = None
            for t, me in enumerate(month_ends):
                if rng.random() < monthly_delist_prob:
                    delist_month_map[sym] = me
                    break

        delist_month_s = pd.Series(
            {sym: (delist_month_map[sym] if delist_month_map[sym] else pd.NaT) for sym in symbols},
            name="delist_month",
        )

        # 5. Monthly log-returns for all assets
        monthly_ret_arr = np.full((self.n_months, self.n_assets), np.nan)
        for t in range(self.n_months):
            noise = rng.standard_normal(self.n_assets)
            ret_row = (
                alpha_mom * mom_loading
                + alpha_val * val_loading
                + self.monthly_vol * noise
            )
            for i, sym in enumerate(symbols):
                dm = delist_month_map[sym]
                if dm is None or month_ends[t] <= dm:
                    monthly_ret_arr[t, i] = ret_row[i]
                # else remains NaN

        monthly_returns = pd.DataFrame(
            monthly_ret_arr,
            index=month_ends,
            columns=symbols,
        )

        # 6. Daily OHLCV and fundamentals
        ohlcv: dict[str, pd.DataFrame] = {}
        fund_records: list[tuple] = []

        # Per-asset stable liquidity scale for volume
        liq_scale = rng.lognormal(mean=0.0, sigma=0.5, size=self.n_assets)

        for i, sym in enumerate(symbols):
            frames = []
            open_price = 100.0

            for t, me in enumerate(month_ends):
                month_ret = monthly_ret_arr[t, i]
                if np.isnan(month_ret):
                    # delisted — stop generating bars
                    break

                # Business days in this calendar month
                bdays = self._business_days_in_month(me)
                n_days = len(bdays)

                # Daily log-returns that sum exactly to month_ret
                daily_logs = self._split_monthly_log(rng, month_ret, n_days)

                # Reconstruct prices
                log_closes = np.log(open_price) + np.cumsum(daily_logs)
                closes = np.exp(log_closes)
                opens = np.empty(n_days)
                opens[0] = open_price
                opens[1:] = closes[:-1]

                # High / Low with small seeded noise
                e_h = np.abs(rng.normal(0.0, 0.002, n_days))
                e_l = np.abs(rng.normal(0.0, 0.002, n_days))
                highs = np.maximum(opens, closes) * (1.0 + e_h)
                lows = np.minimum(opens, closes) * (1.0 - e_l)
                # Ensure low > 0 (guaranteed since close > 0 and factor < 1)

                # Volume: log-normal scaled by asset liquidity
                base_vol = 1_000_000.0 * liq_scale[i]
                volumes = rng.lognormal(mean=np.log(base_vol), sigma=0.3, size=n_days)

                month_df = pd.DataFrame(
                    {
                        "open": opens,
                        "high": highs,
                        "low": lows,
                        "close": closes,
                        "volume": volumes,
                    },
                    index=bdays,
                )
                frames.append(month_df)

                # Next month opens at this month's last close
                open_price = closes[-1]

                # Fundamentals (monthly)
                btm = val_loading[i] + 0.2 * rng.standard_normal()
                qual = quality_loading[i] + 0.1 * rng.standard_normal()
                fund_records.append((me, sym, btm, qual))

            if frames:
                ohlcv[sym] = pd.concat(frames)
            else:
                # Delisted in month 0 — empty frame with correct schema
                ohlcv[sym] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"],
                    index=pd.DatetimeIndex([]),
                )

        # Build fundamentals DataFrame
        if fund_records:
            fund_idx = pd.MultiIndex.from_tuples(
                [(r[0], r[1]) for r in fund_records],
                names=["month_end", "symbol"],
            )
            fundamentals = pd.DataFrame(
                {"book_to_market": [r[2] for r in fund_records],
                 "quality": [r[3] for r in fund_records]},
                index=fund_idx,
            )
        else:
            fundamentals = pd.DataFrame(
                columns=["book_to_market", "quality"],
                index=pd.MultiIndex.from_tuples([], names=["month_end", "symbol"]),
            )

        return SyntheticPanel(
            ohlcv=ohlcv,
            fundamentals=fundamentals,
            monthly_returns=monthly_returns,
            mom_loading=mom_loading_s,
            val_loading=val_loading_s,
            delist_month=delist_month_s,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unit_normal_loading(rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n values, demean, and rescale to unit variance."""
        raw = rng.standard_normal(n)
        raw -= raw.mean()
        std = raw.std(ddof=0)
        if std > 0:
            raw /= std
        return raw

    @staticmethod
    def _alpha_coeff(ic_target: float, sigma_noise: float) -> float:
        """Compute alpha coefficient from IC target (LOCKED formula).

        alpha = IC_target * sigma_noise / sqrt(1 - IC_target**2)
        """
        denom = np.sqrt(1.0 - ic_target**2)
        return ic_target * sigma_noise / denom

    def _month_ends(self) -> pd.DatetimeIndex:
        """Generate n_months consecutive business month-end dates."""
        start_ts = pd.Timestamp(self.start)
        return pd.date_range(start=start_ts, periods=self.n_months, freq="BME")

    @staticmethod
    def _business_days_in_month(month_end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return all business days in the same calendar month as month_end."""
        month_start = month_end.replace(day=1)
        return pd.bdate_range(start=month_start, end=month_end)

    @staticmethod
    def _split_monthly_log(
        rng: np.random.Generator, monthly_log: float, n_days: int
    ) -> np.ndarray:
        """Split monthly log-return into daily increments that sum exactly.

        daily_log = monthly_log / n_days + eps
        where eps ~ N(0, monthly_vol/sqrt(21)) re-centered so sum(eps) == 0.
        This guarantees sum(daily_log) == monthly_log exactly.
        """
        if n_days == 1:
            return np.array([monthly_log])

        daily_vol = 0.04 / np.sqrt(21.0)  # approx daily volatility
        eps = rng.normal(0.0, daily_vol, n_days)
        eps -= eps.mean()  # re-center: sum(eps) == 0 exactly
        return monthly_log / n_days + eps
