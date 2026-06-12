"""
defiregimenet — DeFi Regime Network: crypto-market regime detection, feature engineering,
classification, forecasting, evaluation, and reporting.

Public API (frozen, plan 05-09)
================================

EAGER (light import chains — numpy/pandas/scipy/sklearn only):
  Data:
    CryptoGenerator, CryptoPanel, validate_crypto_data, inject_anomalies
  Features:
    build_feature_matrix, build_feature_panel, expanding_zscore
  Regime detection:
    detect_regimes_per_token
    (CausalRegimeDetector is NOT re-exported at top level — it belongs to macroregime;
     import directly from defiregimenet.regime.detector if needed)
  Analytics:
    per_token_diagnostics, k_sensitivity_per_token, cramers_v,
    cross_token_regime_correlation
  Evaluation:
    RegimeCVEvaluator, labels_to_probas

LAZY via __getattr__ (each pulls xgboost, arch, matplotlib, or ccxt transitively):
  Classifiers (models.classifiers -> xgboost):
    LogisticRegimeClassifier, XGBRegimeClassifier
  Volatility forecast (forecast.vol_forecast -> arch):
    per_token_forecast_comparison, garch_studentst_variance
  Report (report.builder -> matplotlib.pyplot):
    ReportBuilder
  Real data (data.real -> ccxt, optional):
    load_ccxt_panel
  Pipeline (pipeline -> models.classifiers -> xgboost):
    run_pipeline, PipelineResults, load_config
    NOTE: pipeline is lazy by necessity — it imports models.classifiers which imports
    xgboost, so an eager pipeline import would drag xgboost into every `import defiregimenet`.

EXCLUDED from __all__ by design (label quarantine):
  make_regime_labels — internal DGP helper; must never be imported via the public surface
  to prevent label leakage into feature/evaluation paths. Import directly from
  defiregimenet.labels if required by internal test-oracle code.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Eager imports — verified light (numpy/pandas/scipy/sklearn chains only)
# ---------------------------------------------------------------------------

from defiregimenet.data.synthetic import (
    CryptoGenerator,
    CryptoPanel,
    validate_crypto_data,
    inject_anomalies,
)
from defiregimenet.features.crypto import (
    build_feature_matrix,
    build_feature_panel,
    expanding_zscore,
)
from defiregimenet.regime.detector import detect_regimes_per_token
from defiregimenet.analytics.diagnostics import (
    per_token_diagnostics,
    k_sensitivity_per_token,
)
from defiregimenet.analytics.cross_token import (
    cramers_v,
    cross_token_regime_correlation,
)
from defiregimenet.evaluation.cv_evaluator import (
    RegimeCVEvaluator,
    labels_to_probas,
)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # data
    "CryptoGenerator",
    "CryptoPanel",
    "validate_crypto_data",
    "inject_anomalies",
    # features
    "build_feature_matrix",
    "build_feature_panel",
    "expanding_zscore",
    # regime detection
    "detect_regimes_per_token",
    # analytics
    "per_token_diagnostics",
    "k_sensitivity_per_token",
    "cramers_v",
    "cross_token_regime_correlation",
    # evaluation
    "RegimeCVEvaluator",
    "labels_to_probas",
    # ---- LAZY (heavy deps deferred to first access) ----
    # classifiers -> xgboost
    "LogisticRegimeClassifier",
    "XGBRegimeClassifier",
    # forecast -> arch
    "per_token_forecast_comparison",
    "garch_studentst_variance",
    # report -> matplotlib.pyplot
    "ReportBuilder",
    # real data -> ccxt (optional)
    "load_ccxt_panel",
    # pipeline -> classifiers -> xgboost
    "run_pipeline",
    "PipelineResults",
    "load_config",
    # version
    "__version__",
]

# ---------------------------------------------------------------------------
# Lazy __getattr__ — defers heavy-module imports to first access
#
# All loader functions use fully-qualified, statically-known import paths
# (no dynamic strings) to avoid CWE-706 / arbitrary-module-load risk.
# ---------------------------------------------------------------------------

def _load_LogisticRegimeClassifier():
    from defiregimenet.models.classifiers import LogisticRegimeClassifier
    return LogisticRegimeClassifier


def _load_XGBRegimeClassifier():
    from defiregimenet.models.classifiers import XGBRegimeClassifier
    return XGBRegimeClassifier


def _load_per_token_forecast_comparison():
    from defiregimenet.forecast.vol_forecast import per_token_forecast_comparison
    return per_token_forecast_comparison


def _load_garch_studentst_variance():
    from defiregimenet.forecast.vol_forecast import garch_studentst_variance
    return garch_studentst_variance


def _load_ReportBuilder():
    from defiregimenet.report.builder import ReportBuilder
    return ReportBuilder


def _load_load_ccxt_panel():
    from defiregimenet.data.real import load_ccxt_panel
    return load_ccxt_panel


def _load_run_pipeline():
    from defiregimenet.pipeline import run_pipeline
    return run_pipeline


def _load_PipelineResults():
    from defiregimenet.pipeline import PipelineResults
    return PipelineResults


def _load_load_config():
    from defiregimenet.pipeline import load_config
    return load_config


# Static dispatch: name -> zero-arg callable that performs the import
_LAZY_LOADERS: dict[str, object] = {
    "LogisticRegimeClassifier":       _load_LogisticRegimeClassifier,
    "XGBRegimeClassifier":            _load_XGBRegimeClassifier,
    "per_token_forecast_comparison":  _load_per_token_forecast_comparison,
    "garch_studentst_variance":       _load_garch_studentst_variance,
    "ReportBuilder":                  _load_ReportBuilder,
    "load_ccxt_panel":                _load_load_ccxt_panel,
    "run_pipeline":                   _load_run_pipeline,
    "PipelineResults":                _load_PipelineResults,
    "load_config":                    _load_load_config,
}


def __getattr__(name: str):  # noqa: N802
    """Lazy import handler for heavy optional symbols.

    Resolves names present in _LAZY_LOADERS on first access via a static
    dispatch table (no dynamic importlib.import_module calls).  Raises
    AttributeError for any other name, including make_regime_labels which is
    quarantined from the public surface.
    """
    loader = _LAZY_LOADERS.get(name)
    if loader is not None:
        obj = loader()
        # Cache in module namespace so subsequent accesses bypass __getattr__
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'defiregimenet' has no attribute {name!r}")
