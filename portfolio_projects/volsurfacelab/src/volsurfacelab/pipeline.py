"""VolSurfaceLab end-to-end pipeline orchestrator.

Wires the four wave-2 modules in dependency order and returns a frozen
PipelineResults dataclass:

    1. SyntheticChainGenerator  -> ChainData          (chain.py)
    2. solve_chain_iv            -> iv_frame           (iv_solver.py)   HONEST PATH
    3. fit_svi_slice + validate  -> svi_fits           (svi.py)
    4. compare_forecasts         -> ForecastComparison (forecast.py)
    5. run_vrp_strategy          -> VRPResult          (strategy.py)

HONEST-PATH DISCIPLINE (VSL leakage-equivalent rule):
  The pipeline calls solve_chain_iv on OPTION PRICES to recover IVs.
  chain.options['true_iv'] is a ground-truth oracle for TESTS ONLY —
  it must NEVER be used as a pipeline input.  Using true_iv directly
  would be equivalent to using future realized vol in a forecast model:
  it bypasses the market microstructure (bid/ask spread, tick size, etc.)
  that creates the IV surface estimation problem we are studying.

This design mirrors the Phase 1 look-ahead-bias discipline:
  - QBacktest: no T+0 fill; data handler enforces causal bar access.
  - VolSurfaceLab: no true_iv in pipeline; iv_solver enforces honest pricing.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from volsurfacelab.chain import (
    SYNTHETIC_SVI_SURFACE,
    ChainData,
    SyntheticChainGenerator,
    generate_underlying_returns,
    validate_chain_coverage,
)
from volsurfacelab.forecast import ForecastComparison, compare_forecasts
from volsurfacelab.iv_solver import solve_chain_iv
from volsurfacelab.strategy import VRPResult, run_vrp_strategy
from volsurfacelab.svi import SVISliceFit, calibrate_surface, fit_svi_slice, validate_surface

__all__ = ["VolSurfacePipeline", "PipelineResults", "load_config"]

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

# Canonical location: configs/volsurfacelab.yaml relative to the project root.
# Walked up from this file: src/volsurfacelab/pipeline.py -> src/volsurfacelab/
# -> src/ -> volsurfacelab project root -> configs/
_THIS_DIR = Path(__file__).parent
# src/volsurfacelab/pipeline.py
# _THIS_DIR          = .../volsurfacelab/src/volsurfacelab/
# _THIS_DIR.parent   = .../volsurfacelab/src/
# _THIS_DIR.parent.parent = .../volsurfacelab/   (project root)
_DEFAULT_CONFIG_PATH = _THIS_DIR.parent.parent / "configs" / "volsurfacelab.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    """Load VolSurfaceLab configuration from YAML.

    Resolves the canonical configs/volsurfacelab.yaml relative to the package.
    Accepts an explicit path override for tests.

    Parameters
    ----------
    path : Path, optional
        Explicit path to a YAML config file.  If None, uses the default
        configs/volsurfacelab.yaml location.

    Returns
    -------
    dict
        Parsed YAML config.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    """
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"VolSurfaceLab config not found at: {resolved}\n"
            "Create configs/volsurfacelab.yaml or pass path= explicitly."
        )
    with resolved.open("r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# PipelineResults frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineResults:
    """Immutable container of all pipeline outputs.

    Attributes
    ----------
    chain : ChainData
        Synthetic options chain used for the run.
    iv_frame : pd.DataFrame
        chain.options enriched with a solved 'iv' column (from solve_chain_iv).
    svi_fits : dict[float, SVISliceFit]
        Validated SVI slice fits keyed by maturity T (only validated slices).
    excluded_slices : tuple
        Tuples of (T, reason) for slices excluded by the no-arb gate.
    forecast : ForecastComparison
        HAR / GARCH / EGARCH comparison on underlying returns.
    vrp : VRPResult
        Delta-hedged straddle strategy result.
    config_used : dict
        Config snapshot used for this run (includes seed and cost_rate).
    seed : int
        Random seed used for this run.
    """

    chain: ChainData
    iv_frame: pd.DataFrame
    svi_fits: Dict[float, SVISliceFit]
    excluded_slices: Tuple
    forecast: ForecastComparison
    vrp: VRPResult
    config_used: Dict[str, Any]
    seed: int


# ---------------------------------------------------------------------------
# VolSurfacePipeline
# ---------------------------------------------------------------------------

class VolSurfacePipeline:
    """End-to-end VolSurfaceLab pipeline orchestrator.

    Wires the four wave-2 modules:
      chain -> IV solve (honest path) -> SVI gate -> forecast -> strategy

    Parameters
    ----------
    config : dict, optional
        Config dict.  If None, loads from configs/volsurfacelab.yaml.
    seed : int
        RNG seed for chain generation and returns path (default 42).
    quick : bool
        If True, reduces n_days to 400 and n_restarts to 2 for fast
        integration testing (target: under 30 seconds).
    svi_surface : dict, optional
        Override for the SVI surface used by SyntheticChainGenerator.
        If None, uses SYNTHETIC_SVI_SURFACE (standard clean surface).
        Pass make_calendar_violating_surface() for gate testing.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        seed: int = 42,
        quick: bool = False,
        svi_surface: Optional[dict] = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        self.seed = seed
        self.quick = quick
        self.svi_surface = svi_surface  # dependency injection for tests

    def run(self) -> PipelineResults:
        """Execute the full pipeline and return a frozen PipelineResults.

        Steps
        -----
        1. Generate synthetic options chain and underlying returns.
        2. validate_chain_coverage — fail fast on malformed chains.
        3. solve_chain_iv on prices (HONEST PATH — not true_iv).
        4. Build per-maturity (k, w_obs = iv^2 * T) from solved IVs.
        5. fit_svi_slice per maturity + validate_surface gate.
           Raise RuntimeError if ALL slices are excluded.
        6. compare_forecasts on underlying returns.
        7. run_vrp_strategy.
        8. Assemble and return frozen PipelineResults.

        Returns
        -------
        PipelineResults

        Raises
        ------
        RuntimeError
            If the no-arb gate excludes ALL slices (nothing downstream is
            meaningful — would produce empty surface plots and NaN Greeks).
        """
        cfg = self.config
        chain_cfg = cfg.get("chain", {})
        underlying_cfg = cfg.get("underlying", {})
        forecast_cfg = cfg.get("forecast", {})
        strategy_cfg = cfg.get("strategy", {})

        # quick-mode overrides
        n_days = 400 if self.quick else int(underlying_cfg.get("n_days", 750))
        n_restarts = 2 if self.quick else int(forecast_cfg.get("garch_restarts", 5))

        # ------------------------------------------------------------------
        # Step 1: Generate chain and underlying returns
        # ------------------------------------------------------------------
        # If a custom svi_surface is injected, we need to use it for chain generation.
        # SyntheticChainGenerator reads SYNTHETIC_SVI_SURFACE at generate() time.
        # We patch the module-level constant temporarily if an override is provided.
        import volsurfacelab.chain as chain_mod

        if self.svi_surface is not None:
            # Temporarily override the surface for generation
            original_surface = chain_mod.SYNTHETIC_SVI_SURFACE
            chain_mod.SYNTHETIC_SVI_SURFACE = self.svi_surface
            try:
                generator = SyntheticChainGenerator(
                    spot=float(chain_cfg.get("spot", 100.0)),
                    risk_free=float(chain_cfg.get("risk_free", 0.05)),
                    maturities=tuple(chain_cfg.get("maturities", [0.25, 0.5, 1.0])),
                    n_strikes=int(chain_cfg.get("n_strikes", 13)),
                    k_min=float(chain_cfg.get("k_min", -1.5)),
                    k_max=float(chain_cfg.get("k_max", 1.5)),
                    seed=self.seed,
                )
                chain = generator.generate()
            finally:
                chain_mod.SYNTHETIC_SVI_SURFACE = original_surface
        else:
            generator = SyntheticChainGenerator(
                spot=float(chain_cfg.get("spot", 100.0)),
                risk_free=float(chain_cfg.get("risk_free", 0.05)),
                maturities=tuple(chain_cfg.get("maturities", [0.25, 0.5, 1.0])),
                n_strikes=int(chain_cfg.get("n_strikes", 13)),
                k_min=float(chain_cfg.get("k_min", -1.5)),
                k_max=float(chain_cfg.get("k_max", 1.5)),
                seed=self.seed,
            )
            chain = generator.generate()

        garch_cfg = underlying_cfg.get("garch", {})
        returns = generate_underlying_returns(
            seed=self.seed,
            n_days=n_days,
            omega=float(garch_cfg.get("omega", 2e-6)),
            alpha=float(garch_cfg.get("alpha", 0.08)),
            beta=float(garch_cfg.get("beta", 0.90)),
        )

        # ------------------------------------------------------------------
        # Step 2: Validate chain coverage
        # ------------------------------------------------------------------
        maturities = list(chain_cfg.get("maturities", [0.25, 0.5, 1.0]))
        k_min = float(chain_cfg.get("k_min", -1.5))
        k_max = float(chain_cfg.get("k_max", 1.5))
        validate_chain_coverage(chain, maturities, k_min, k_max)

        # ------------------------------------------------------------------
        # Step 3: Solve IVs from PRICES (honest path)
        # ------------------------------------------------------------------
        # IMPORTANT: We call solve_chain_iv on OPTION PRICES.
        # The 'true_iv' column in chain.options is the oracle for tests ONLY.
        # It must never be used here — using true_iv would bypass the market
        # microstructure and constitute a look-ahead analog.
        iv_frame = solve_chain_iv(chain)

        # ------------------------------------------------------------------
        # Step 4: Build per-maturity (k, w_obs) and calibrate SVI surface
        # ------------------------------------------------------------------
        calls_with_iv = iv_frame[iv_frame["flag"] == "c"].copy()
        calls_with_iv = calls_with_iv.dropna(subset=["iv"])

        if calls_with_iv.empty:
            raise RuntimeError(
                "VolSurfacePipeline: solve_chain_iv returned all-NaN IVs. "
                "Cannot calibrate SVI surface. Check the chain prices."
            )

        available_maturities = sorted(calls_with_iv["T"].unique())
        excluded_slices: list = []

        # Fit SVI per slice using solved IVs (not true_iv)
        raw_fits: dict = {}
        for T in available_maturities:
            slice_df = calls_with_iv[calls_with_iv["T"] == T]
            k_obs = slice_df["k"].values
            # w_obs = solved_iv^2 * T (total variance from solved IVs)
            w_obs = (slice_df["iv"].values ** 2) * T
            fit = fit_svi_slice(k_obs, w_obs, T=T)
            raw_fits[T] = fit
            if not fit.success:
                excluded_slices.append((T, "fit_failed"))

        # Build params dict for successful fits
        successful_params: dict = {}
        for T, fit in raw_fits.items():
            if fit.success:
                successful_params[T] = fit.params

        # Run no-arb gate
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", UserWarning)
            validated_params = validate_surface(successful_params)

        # Re-emit caught warnings
        for w in caught_warnings:
            warnings.warn(w.message, w.category, stacklevel=2)

        # Collect gate exclusions
        excluded_by_gate = set(successful_params.keys()) - set(validated_params.keys())
        for T in excluded_by_gate:
            reason_msg = "gate_excluded"
            for w in caught_warnings:
                msg = str(w.message)
                t_str = f"T={T:.4f}"
                if t_str in msg:
                    if "butterfly" in msg:
                        reason_msg = "butterfly_violation"
                    elif "Calendar" in msg or "calendar" in msg:
                        reason_msg = "calendar_violation"
                    elif "w(k)" in msg or "total variance" in msg:
                        reason_msg = "positivity_violation"
                    break
            excluded_slices.append((T, reason_msg))

        if not validated_params:
            raise RuntimeError(
                "VolSurfacePipeline: no-arb gate excluded ALL slices. "
                "Cannot produce meaningful surface plots or strategy entry. "
                f"Excluded: {excluded_slices}"
            )

        # Build final fits dict — only validated slices
        svi_fits: dict = {
            T: raw_fits[T] for T in validated_params.keys() if T in raw_fits
        }

        # ------------------------------------------------------------------
        # Step 5: compare_forecasts on underlying returns
        # ------------------------------------------------------------------
        train_frac = float(forecast_cfg.get("train_frac", 0.67))
        forecast = compare_forecasts(
            returns=returns,
            train_frac=train_frac,
            n_restarts=n_restarts,
        )

        # ------------------------------------------------------------------
        # Step 6: run_vrp_strategy
        # ------------------------------------------------------------------
        cost_rate = float(strategy_cfg.get("cost_rate", 0.001))
        delta_hedge_cost_rate = float(strategy_cfg.get("delta_hedge_cost_rate", 0.001))
        # Hand the strategy the IV-ENRICHED chain, BUT restricted to maturities
        # that survived the no-arb gate (validated_params keys).
        # This ensures run_vrp_strategy cannot select an excluded maturity as the
        # strategy entry (no-arb gate finding fixed in plan 04-08).
        validated_maturities = set(validated_params.keys())
        iv_frame_gated = iv_frame[iv_frame["T"].isin(validated_maturities)]
        chain_with_solved_iv = ChainData(
            options=iv_frame_gated,
            spot=chain.spot,
            risk_free=chain.risk_free,
            seed=chain.seed,
        )
        vrp = run_vrp_strategy(
            chain=chain_with_solved_iv,
            returns=returns,
            cost_rate=cost_rate,
            delta_hedge_cost_rate=delta_hedge_cost_rate,
            side="short",
            r=float(chain_cfg.get("risk_free", 0.05)),
        )

        # ------------------------------------------------------------------
        # Step 7: Assemble frozen PipelineResults
        # ------------------------------------------------------------------
        config_used = {
            "seed": self.seed,
            "quick": self.quick,
            "cost_rate": cost_rate,
            "delta_hedge_cost_rate": delta_hedge_cost_rate,
            "train_frac": train_frac,
            "n_days": n_days,
            "n_restarts": n_restarts,
        }

        return PipelineResults(
            chain=chain,
            iv_frame=iv_frame,
            svi_fits=svi_fits,
            excluded_slices=tuple(excluded_slices),
            forecast=forecast,
            vrp=vrp,
            config_used=config_used,
            seed=self.seed,
        )
