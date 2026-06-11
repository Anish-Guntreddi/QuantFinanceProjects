"""macroregime — Macro Regime Detection and Allocation Framework.

Public API
----------

Pipeline
~~~~~~~~
MacroRegimePipeline     — end-to-end macro regime pipeline (detect + allocate + backtest)
PipelineResults         — frozen dataclass returned by MacroRegimePipeline.run()

Data
~~~~
SyntheticMacroGenerator — 4-state Markov-switching DGP (offline / CI default)
SyntheticMacroLoader    — wrapper adapting SyntheticMacroGenerator to MacroLoaderBase
FredMacroLoader         — ALFRED first-release loader (optional; requires FRED_API_KEY)
MacroLoaderBase         — abstract base for all macro loaders
apply_release_lag       — shift observation-date index → publication-date index
as_of_view              — filter series to rows with pub_date <= as_of

Regime
~~~~~~
CausalRegimeDetector    — rolling-refit causal HMM/GMM regime detector
align_regime_labels     — raw→aligned label permutation (double-argsort on means)
transition_matrix       — empirical K×K row-stochastic transition matrix
dwell_times             — mean dwell time (run length) per regime state

Allocation
~~~~~~~~~~
TargetWeightPortfolio   — qbacktest Portfolio subclass driven by target weights
TargetWeightStrategy    — as-of weight replay strategy (weight-magnitude re-emission)

Benchmarks
~~~~~~~~~~
run_strategy_backtest   — assemble + run backtest via build_strategy_engine (cost parity)
build_strategy_engine   — assemble engine with shared cost params (cost parity)
build_60_40_weights     — 60/40 static allocation weight builder
build_equal_weight_weights — 1/N equal weight across universe
build_risk_parity_weights  — inverse-vol risk parity weight builder

Report (lazy — pulled only on first access; avoids pyplot at package init)
~~~~~~~
ReportBuilder           — figures + markdown summary for the pipeline

Version
~~~~~~~
__version__             — package version string
"""

__version__ = "0.1.0"

from macroregime.pipeline import MacroRegimePipeline, PipelineResults
from macroregime.data import (
    SyntheticMacroGenerator,
    SyntheticMacroPanel,
    SyntheticMacroLoader,
    MacroLoaderBase,
    apply_release_lag,
    as_of_view,
)
from macroregime.regime import (
    CausalRegimeDetector,
    align_regime_labels,
    transition_matrix,
    dwell_times,
)
from macroregime.allocation import (
    TargetWeightPortfolio,
    TargetWeightStrategy,
)
from macroregime.benchmarks import (
    run_strategy_backtest,
    build_60_40_weights,
    build_equal_weight_weights,
    build_risk_parity_weights,
)


def __getattr__(name: str):
    # Lazy exports — pull matplotlib-heavy modules only on first access.
    # This keeps `import macroregime` lightweight and headless-safe, mirroring
    # the qbacktest.TearsheetRenderer lazy-export pattern (Phase 1 locked).
    if name == "ReportBuilder":
        from macroregime.report import ReportBuilder
        return ReportBuilder
    if name == "FredMacroLoader":
        # Optional real-data path; lazy to keep package importable without fredapi.
        from macroregime.data.fred_loader import FredMacroLoader
        return FredMacroLoader
    if name == "build_strategy_engine":
        from macroregime.benchmarks.benchmarks import build_strategy_engine
        return build_strategy_engine
    raise AttributeError(f"module 'macroregime' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Pipeline
    "MacroRegimePipeline",
    "PipelineResults",
    # Data — eager
    "SyntheticMacroGenerator",
    "SyntheticMacroPanel",
    "SyntheticMacroLoader",
    "MacroLoaderBase",
    "apply_release_lag",
    "as_of_view",
    # Data — lazy (requires fredapi optional dep)
    "FredMacroLoader",
    # Regime
    "CausalRegimeDetector",
    "align_regime_labels",
    "transition_matrix",
    "dwell_times",
    # Allocation
    "TargetWeightPortfolio",
    "TargetWeightStrategy",
    # Benchmarks — eager
    "run_strategy_backtest",
    "build_60_40_weights",
    "build_equal_weight_weights",
    "build_risk_parity_weights",
    # Benchmarks — lazy (post-plan addition)
    "build_strategy_engine",
    # Report — lazy (pulls matplotlib/pyplot)
    "ReportBuilder",
]
