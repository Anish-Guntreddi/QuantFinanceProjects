"""MacroRegimePipeline — end-to-end dual-frequency regime-to-backtest pipeline.

CAUSALITY CHAIN (never re-order these steps):
  1. PIT macro panel  (lag applied by loader — publication-date indexed)
  2. Expanding z-score standardization of macro features (NOT full-sample — full-sample
     leaks future mean/std into historical features; expanding window uses only data
     published at or before each bar)
  3. CausalRegimeDetector on the macro matrix → macro_regimes (publication-date indexed)
  4. build_market_features on daily OHLCV (each feature shifted by 1 bar)
  5. Expanding z-score standardization of daily market features (same reason as macro)
  6. CausalRegimeDetector on the DAILY matrix → market_regimes (daily)
  7. Forward-fill macro regimes to daily AFTER lag application (Pitfall 5:
     ffill must happen AFTER lag; lag is already applied by SyntheticMacroLoader so
     ffill here is safe)
  8. Combine: macro regime takes precedence when defined (publication date <= today);
     market regime used before macro warm-up ends
  9. month_end_rebalance_dates → build_weight_schedule → run_strategy_backtest

DESIGN DECISIONS (locked):
  - Macro and daily market features are fed to TWO SEPARATE CausalRegimeDetector instances.
    NEVER concatenate macro + daily features into one matrix — they have different
    frequencies, different stationarity properties, and mixing them creates alignment
    artifacts and look-ahead bias in the ffill step.
  - CausalRegimeDetector is the ONLY regime-production path.  No direct model-library
    classes (e.g., hmmlearn or sklearn mixture classes) are called in this file.
  - run_strategy_backtest (from macroregime.benchmarks) is imported at call time (inside
    methods) so that 03-07 and 03-06 can be executed in parallel without circular imports.
  - PipelineResults is a frozen dataclass for in-process testability (Phase 2 pattern).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResults:
    """Immutable results bundle returned by MacroRegimePipeline.run().

    Attributes
    ----------
    macro_regimes : pd.Series
        Monthly (publication-date indexed) causal regime labels from the macro model.
        Sentinel -1 denotes warm-up / unassigned.
    market_regimes : pd.Series
        Daily causal regime labels from the market-features model.
        Sentinel -1 denotes warm-up / unassigned.
    combined_regimes : pd.Series
        Daily regime labels combining macro (precedence) and market (fallback).
        Macro regime is forward-filled to daily after lag is applied (Pitfall 5 safe).
    weight_schedule : dict[pd.Timestamp, dict[str, float]]
        Dated weight schedule produced by build_weight_schedule.
    regime_backtest : Any
        BacktestResults from run_strategy_backtest.
    diagnostics : dict
        Keys: "macro", "market", "combined".  Each contains:
            "transition_matrix" : np.ndarray  (K, K) empirical transition matrix
            "dwell_times"       : dict[int, float]  mean run length per state
    config : dict
        Pipeline configuration used for this run.
    """

    macro_regimes: pd.Series
    market_regimes: pd.Series
    combined_regimes: pd.Series
    weight_schedule: dict
    regime_backtest: Any
    diagnostics: dict
    config: dict


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _expanding_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window z-score standardization of a feature DataFrame.

    For each column c and each row t:
        z[t] = (x[t] - expanding_mean[t]) / expanding_std[t]

    This uses only data up to and including t, which preserves causality.
    Full-sample standardization would leak future mean/std into historical
    feature values — hence this expanding implementation.

    The first few rows where std is zero (constant prefix) are replaced with 0.
    """
    mean = df.expanding(min_periods=2).mean()
    std = df.expanding(min_periods=2).std(ddof=1)
    # Guard: where std is zero or NaN, produce 0 rather than NaN/inf
    z = (df - mean) / std.where(std > 1e-12, other=np.nan)
    return z.fillna(0.0)


class MacroRegimePipeline:
    """Full research pipeline: synthetic macro → dual-frequency regimes → backtest.

    Parameters
    ----------
    seed : int
        Random seed for data generation and regime detection reproducibility.
    n_years : int
        Number of years for the synthetic macro panel.
        Overridden to 10 when quick=True.
    k : int
        Number of regime states K for both regime models (default 3).
    backend : {"hmm", "gmm"}
        Regime model backend (default "hmm").
    params_path : str or Path, optional
        Override path to strategy_params.yml.  If None, the default config path
        within the package is used.
    loader : MacroLoaderBase, optional
        Inject a custom macro loader (e.g. FredMacroLoader for live runs).
        If None, SyntheticMacroLoader is created from ``seed`` / ``n_years``.
    quick : bool
        Quick mode for fast testing: n_years=10, refit_every=63, n_restarts=2.
        Overrides n_years, refit_every, and n_restarts.
    """

    def __init__(
        self,
        seed: int = 42,
        n_years: int = 30,
        k: int = 3,
        backend: str = "hmm",
        params_path: str | None = None,
        loader=None,
        quick: bool = False,
    ) -> None:
        self.seed = seed
        self.n_years = 10 if quick else n_years
        self.k = k
        self.backend = backend
        self.params_path = params_path
        self._loader = loader
        self.quick = quick

        # quick mode: faster refit, fewer restarts
        self._refit_every = 63 if quick else 21
        self._n_restarts = 2 if quick else 3
        self._min_train_monthly = 24   # months (~2 years)
        self._min_train_daily = 126    # bars (~6 months)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineResults:
        """Execute the full pipeline and return a frozen PipelineResults.

        Steps (in causal order):
          1. Build / load synthetic macro panel + PIT loader
          2. MACRO branch: load_panel → drop NaN → expanding-z → CausalRegimeDetector → monthly regimes
          3. MARKET branch: build_market_features → drop NaN → expanding-z → CausalRegimeDetector → daily regimes
          4. COMBINE: ffill monthly macro regimes to daily (safe: lag applied upstream) → merge
          5. ALLOCATE: month_end_rebalance_dates → build_weight_schedule → run_strategy_backtest
          6. DIAGNOSE: transition_matrix + dwell_times for macro, market, combined
          7. Return PipelineResults

        Returns
        -------
        PipelineResults
        """
        # -----------------------------------------------------------------
        # Step 1: Build data
        # -----------------------------------------------------------------
        generator, asset_ohlcv = self._build_data()

        # -----------------------------------------------------------------
        # Step 2: MACRO branch (MONTHLY)
        # -----------------------------------------------------------------
        macro_regimes, macro_X_raw = self._run_macro_branch(generator)

        # -----------------------------------------------------------------
        # Step 3: MARKET branch (DAILY)
        # -----------------------------------------------------------------
        daily_index = next(iter(asset_ohlcv.values())).index
        market_regimes, _market_X_raw = self._run_market_branch(asset_ohlcv, daily_index)

        # -----------------------------------------------------------------
        # Step 4: COMBINE macro + market regimes
        # -----------------------------------------------------------------
        combined_regimes = self._combine_regimes(macro_regimes, market_regimes, daily_index)

        # -----------------------------------------------------------------
        # Step 5: ALLOCATE
        # -----------------------------------------------------------------
        from macroregime.benchmarks.benchmarks import (  # function-level import
            load_run_params,
            run_strategy_backtest,
        )
        from macroregime.allocation.weights import (
            load_regime_weights,
            build_weight_schedule,
            month_end_rebalance_dates,
        )

        rebal_dates = month_end_rebalance_dates(daily_index)
        # Normalise rebalance dates (month_end_rebalance_dates returns numpy int64 epoch-nanos)
        rebal_dates = [pd.Timestamp(ts) for ts in rebal_dates]

        regime_weights = load_regime_weights(self.params_path)
        weight_schedule = build_weight_schedule(combined_regimes, rebal_dates, regime_weights)

        # A custom params_path must govern BOTH regime weights (above) and
        # cost/engine params — otherwise overrides silently apply to only half
        # the configuration.
        run_params = None if self.params_path is None else load_run_params(self.params_path)
        regime_backtest = run_strategy_backtest(
            asset_ohlcv=asset_ohlcv,
            weight_schedule=weight_schedule,
            params=run_params,
        )

        # -----------------------------------------------------------------
        # Step 6: DIAGNOSTICS
        # -----------------------------------------------------------------
        from macroregime.regime.diagnostics import transition_matrix, dwell_times

        macro_seq = macro_regimes.values.astype(int)
        market_seq = market_regimes.values.astype(int)
        combined_seq = combined_regimes.values.astype(int)

        diagnostics = {
            "macro": {
                "transition_matrix": transition_matrix(macro_seq, self.k),
                "dwell_times": dwell_times(macro_seq, self.k),
            },
            "market": {
                "transition_matrix": transition_matrix(market_seq, self.k),
                "dwell_times": dwell_times(market_seq, self.k),
            },
            "combined": {
                "transition_matrix": transition_matrix(combined_seq, self.k),
                "dwell_times": dwell_times(combined_seq, self.k),
            },
        }

        config = {
            "seed": self.seed,
            "n_years": self.n_years,
            "k": self.k,
            "backend": self.backend,
            "quick": self.quick,
            "refit_every": self._refit_every,
            "n_restarts": self._n_restarts,
        }

        return PipelineResults(
            macro_regimes=macro_regimes,
            market_regimes=market_regimes,
            combined_regimes=combined_regimes,
            weight_schedule=weight_schedule,
            regime_backtest=regime_backtest,
            diagnostics=diagnostics,
            config=config,
        )

    # ------------------------------------------------------------------
    # Private: data construction
    # ------------------------------------------------------------------

    def _build_data(self):
        """Return (SyntheticMacroGenerator, asset_ohlcv dict)."""
        from macroregime.data.synthetic import SyntheticMacroGenerator

        gen = SyntheticMacroGenerator(n_years=self.n_years, seed=self.seed)
        panel = gen.generate()
        return gen, panel.asset_ohlcv

    # ------------------------------------------------------------------
    # Private: MACRO branch
    # ------------------------------------------------------------------

    def _run_macro_branch(self, generator) -> tuple[pd.Series, np.ndarray]:
        """Load PIT macro panel, standardize (expanding), detect monthly regimes.

        Returns
        -------
        macro_regimes : pd.Series
            Publication-date indexed monthly regime labels (sentinel -1 in warm-up).
        X : np.ndarray
            The raw (un-z-scored) feature matrix used, for diagnostics.
        """
        from macroregime.data.loader_base import SyntheticMacroLoader
        from macroregime.regime.causal import CausalRegimeDetector

        if self._loader is not None:
            loader = self._loader
        else:
            loader = SyntheticMacroLoader(generator=generator, seed=self.seed)

        # load_panel returns a DataFrame with publication-date index and NaN
        # gaps between publication dates (no ffill — plan 03-02 contract).
        macro_df = loader.load_panel()

        # Drop rows that are entirely NaN (publication-date gaps)
        macro_df = macro_df.dropna(how="all")

        # Forward-fill within macro_df to propagate values for the feature matrix.
        # This ffill is SAFE here because:
        #   (a) We are post-lag: indices are publication dates, not observation dates.
        #   (b) We never look ahead: ffill propagates the MOST RECENT PUBLISHED value.
        # Reference: Pitfall 5 in 03-RESEARCH.md — ffill must happen AFTER lag.
        macro_df = macro_df.ffill()

        # Drop any remaining NaNs (the very first row may still have NaN before ffill)
        macro_df = macro_df.dropna()

        X_raw = macro_df.values

        # Expanding z-score: uses only data published up to each bar (causal)
        X_z = _expanding_zscore(macro_df).values

        # observable_dim = index of GDPC1 (GDP growth): low growth = state 0
        # Column order follows MACRO_SERIES: CPIAUCSL=0, UNRATE=1, GDPC1=2, T10Y2Y=3
        gdpc1_dim = list(macro_df.columns).index("GDPC1") if "GDPC1" in macro_df.columns else 0

        detector = CausalRegimeDetector(
            backend=self.backend,
            n_components=self.k,
            min_train=self._min_train_monthly,
            refit_every=self._refit_every,
            n_restarts=self._n_restarts,
            observable_dim=gdpc1_dim,
            random_seed=self.seed,
        )
        raw_labels = detector.fit_predict_causal(X_z)

        macro_regimes = pd.Series(raw_labels, index=macro_df.index, name="macro_regime")
        return macro_regimes, X_raw

    # ------------------------------------------------------------------
    # Private: MARKET branch
    # ------------------------------------------------------------------

    def _run_market_branch(
        self,
        asset_ohlcv: dict[str, pd.DataFrame],
        daily_index: pd.DatetimeIndex,
    ) -> tuple[pd.Series, np.ndarray]:
        """Build daily market features, standardize (expanding), detect daily regimes.

        NOTE: Monthly and daily features are fed to SEPARATE detectors.
        NEVER concatenate monthly + daily features into one matrix.

        Returns
        -------
        market_regimes : pd.Series
            Daily regime labels (sentinel -1 in warm-up).
        X : np.ndarray
            The raw feature matrix used, for diagnostics.
        """
        from macroregime.features.market import build_market_features
        from macroregime.regime.causal import CausalRegimeDetector

        # build_market_features applies shift(1) to all features (causal)
        mkt_df = build_market_features(asset_ohlcv)

        # Drop warm-up NaN rows from the front
        mkt_df = mkt_df.dropna()

        X_raw = mkt_df.values

        # Expanding z-score: standardize using only data up to each daily bar
        X_z = _expanding_zscore(mkt_df).values

        # observable_dim = EQUITY momentum column
        # Columns: EQUITY_vol, EQUITY_mom, EQUITY_dd, BONDS_vol, ..., eq_bd_corr
        eq_mom_dim = 1  # EQUITY_mom is column 1 (after EQUITY_vol at 0)

        detector = CausalRegimeDetector(
            backend=self.backend,
            n_components=self.k,
            min_train=self._min_train_daily,
            refit_every=self._refit_every,
            n_restarts=self._n_restarts,
            observable_dim=eq_mom_dim,
            random_seed=self.seed + 1,  # different seed for independent model
        )
        raw_labels = detector.fit_predict_causal(X_z)

        market_regimes = pd.Series(raw_labels, index=mkt_df.index, name="market_regime")
        return market_regimes, X_raw

    # ------------------------------------------------------------------
    # Private: COMBINE
    # ------------------------------------------------------------------

    def _combine_regimes(
        self,
        macro_regimes: pd.Series,
        market_regimes: pd.Series,
        daily_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Combine macro and market regime series into a daily combined series.

        Combination rule (documented):
          - Forward-fill the monthly macro_regimes to daily (safe AFTER lag).
          - At each daily bar t:
              * If macro_ffill[t] >= 0 (i.e. macro warm-up has ended): use macro regime.
              * Else (macro still in warm-up / not yet published): use market regime.
          - Result: combined_regimes is a daily pd.Series.

        Causality note:
          - macro_regimes has publication-date index (lag already applied upstream by loader).
          - ffill propagates the most recently published macro regime — never future data.
          - market_regimes[t] uses only price data through t-1 (shift(1) in features).

        Parameters
        ----------
        macro_regimes : pd.Series
            Monthly publication-date indexed series.
        market_regimes : pd.Series
            Daily indexed series.
        daily_index : pd.DatetimeIndex
            Full daily business-day index of the asset OHLCV data.

        Returns
        -------
        pd.Series
            Daily combined regime labels aligned to daily_index.
            Sentinel -1 during warm-up of both models.
        """
        # Reindex macro_regimes to the full daily_index then ffill.
        # This is safe because the macro series already has publication-date index
        # (lag was applied by the loader in step 1 of the chain).
        macro_daily = macro_regimes.reindex(daily_index, method=None).ffill()
        # Convert to int-friendly: NaN → -1 sentinel
        macro_daily = macro_daily.fillna(-1).astype(int)

        # Align market_regimes to daily_index (it already covers most of daily_index
        # after dropna, so just reindex; missing bars get -1)
        market_daily = market_regimes.reindex(daily_index, fill_value=-1).astype(int)

        # Combination: macro takes precedence once defined; market fills warm-up gap
        combined = np.where(macro_daily.values >= 0, macro_daily.values, market_daily.values)

        return pd.Series(combined, index=daily_index, name="combined_regime")
