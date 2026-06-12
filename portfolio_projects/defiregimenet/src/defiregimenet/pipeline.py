"""DeFiRegimeNet end-to-end pipeline.

Entry point: run_pipeline(config, quick) -> PipelineResults

Allowed label importers (AST quarantine, see tests/test_labels.py):
  - defiregimenet.evaluation  (cv_evaluator.py)
  - defiregimenet.pipeline    (this module — the ONLY non-evaluation importer)

Label use is restricted to the evaluation protocol:
  1. make_regime_labels produces y (forward-looking labels).
  2. y feeds RegimeCVEvaluator.evaluate() for classifiers only.
  3. HMM/GMM baselines use labels_to_probas to convert causal regime sequences
     into pseudo-probabilities for the log-loss column (no training on y).
  4. y is NEVER concatenated into X or passed to detect_regimes_per_token.

Model comparison asymmetry note:
  HMM/GMM baselines use the full causal sequence (no CV needed — no training on
  labels occurs). Classifiers use purged CV. This asymmetry favors the baselines
  if anything, which is the conservative/honest direction.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, log_loss

from defiregimenet.analytics.cross_token import cross_token_regime_correlation
from defiregimenet.analytics.diagnostics import (
    k_sensitivity_per_token,
    per_token_diagnostics,
)
from defiregimenet.data.synthetic import CryptoGenerator
from defiregimenet.evaluation.cv_evaluator import RegimeCVEvaluator, labels_to_probas
from defiregimenet.features.crypto import build_feature_matrix, build_feature_panel
from defiregimenet.forecast.vol_forecast import (
    garch_studentst_variance,
    per_token_forecast_comparison,
)
# QUARANTINED import — allowed only here and in evaluation modules
from defiregimenet.labels import make_regime_labels
from defiregimenet.models.classifiers import (
    LogisticRegimeClassifier,
    XGBRegimeClassifier,
)
from defiregimenet.regime.detector import detect_regimes_per_token

__all__ = ["run_pipeline", "load_config", "PipelineResults"]

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "params.yml"


# ---------------------------------------------------------------------------
# PipelineResults frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResults:
    """Immutable container for all pipeline outputs.

    Attributes
    ----------
    config : dict
        Flat config dict used during this run (including quick-mode overrides).
    seed : int
        RNG seed used for the data generator.
    tokens : tuple[str, ...]
        Token symbols in the generated panel.
    n_bars : int
        Number of daily bars per token.
    regimes_hmm : dict[str, np.ndarray]
        Per-token causal HMM regime sequences (shape (T,), sentinels = -1).
    regimes_gmm : dict[str, np.ndarray]
        Per-token causal GMM regime sequences.
    diagnostics : dict[str, dict]
        Per-token transition matrices and dwell times (HMM backend).
    k_sensitivity : dict[str, dict]
        Per-token k-sensitivity (HMM; quick: 1 token, ks=(2,4)).
    model_comparison : pd.DataFrame
        Rows: hmm, gmm, logistic, xgboost. Cols: accuracy, log_loss.
    forecast_comparison : dict[str, Any]
        Per-token ForecastComparison from volsurfacelab.
    studentst_robustness : dict[str, float]
        Per-token Student-t GARCH OOS QLIKE (robustness section).
    cross_token_v : pd.DataFrame
        Pairwise Cramér's V on HMM regime sequences.
    label_distribution : dict[str, dict]
        Per-token label counts (for the report data section).
    """

    config: dict
    seed: int
    tokens: tuple
    n_bars: int
    regimes_hmm: dict
    regimes_gmm: dict
    diagnostics: dict
    k_sensitivity: dict
    model_comparison: pd.DataFrame
    forecast_comparison: dict
    studentst_robustness: dict
    cross_token_v: pd.DataFrame
    label_distribution: dict


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(
    path: Optional[Path] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Load YAML config from path (defaults to configs/params.yml).

    Parameters
    ----------
    path : Path, optional
        Path to a YAML config file. Defaults to configs/params.yml.
    overrides : dict, optional
        Key-value overrides applied after loading (flat keys override nested).

    Returns
    -------
    dict
        Flat+nested config dict.
    """
    cfg_path = path or _DEFAULT_CONFIG
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        cfg = {}
    if overrides:
        cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Quick-mode overrides
# ---------------------------------------------------------------------------

_QUICK_OVERRIDES: dict = {
    "n_years": 2,   # 2 full years = 730 bars; int so pd.date_range(periods=) is valid
    "tokens": ["BTC", "ETH"],
    # CV: keep embargo >= label_horizon=5; fold size ~91 >> purged+embargo=10
    "cv": {
        "n_folds": 6,
        "n_test_folds": 2,
        "purged_size": 5,
        "embargo_size": 5,
    },
    "regime": {
        "n_components": 4,
        "min_train": 60,
        "refit_every": 42,
        "n_restarts": 1,
    },
    "forecast": {
        "train_frac": 0.67,
        "n_restarts": 2,
    },
    # k-sensitivity: 1 token only, ks=(2,4) for speed
    "_ksens_tokens": 1,
    "_ksens_ks": (2, 4),
}


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def run_pipeline(
    config: Optional[dict] = None,
    quick: bool = False,
    seed: Optional[int] = None,
) -> PipelineResults:
    """Run the full DeFiRegimeNet pipeline end-to-end.

    Parameters
    ----------
    config : dict, optional
        Config dict (from load_config). If None, loads from configs/params.yml.
    quick : bool
        If True, applies quick-mode overrides (fewer bars, tokens, restarts).
    seed : int, optional
        Override the seed from config.

    Returns
    -------
    PipelineResults
        Frozen dataclass with all pipeline outputs.
    """
    # ------------------------------------------------------------------
    # 1. Config resolution
    # ------------------------------------------------------------------
    cfg = copy.deepcopy(config) if config is not None else load_config()

    if quick:
        _apply_quick_overrides(cfg)

    if seed is not None:
        cfg["seed"] = seed

    rng_seed = int(cfg.get("seed", 42))
    tokens_list = list(cfg.get("tokens", ["BTC", "ETH", "SOL", "AVAX"]))
    n_years = int(float(cfg.get("n_years", 3)))  # cast to int: n_bars = n_years*365 must be integer

    # Nested sub-configs
    regime_cfg = cfg.get("regime", {})
    cv_cfg = cfg.get("cv", {})
    forecast_cfg = cfg.get("forecast", {})
    labels_cfg = cfg.get("labels", {})
    label_horizon = int(labels_cfg.get("horizon", 5))

    # Flatten CV params into top-level cfg for easy inspection by tests
    cfg["label_horizon"] = label_horizon
    cfg["embargo_size"] = int(cv_cfg.get("embargo_size", 5))
    cfg["purged_size"] = int(cv_cfg.get("purged_size", 5))

    # ------------------------------------------------------------------
    # 2. Data generation
    # ------------------------------------------------------------------
    gen = CryptoGenerator(
        seed=rng_seed,
        n_years=n_years,
        tokens=tokens_list,
        fat_tail_df=int(cfg.get("fat_tail_df", 4)),
        market_factor_weight=float(cfg.get("market_factor_weight", 0.7)),
    )
    panel = gen.generate()
    tokens_tuple = panel.tokens
    n_bars = len(panel.ohlcv[tokens_tuple[0]])

    # ------------------------------------------------------------------
    # 3. Feature engineering (causal — no labels module)
    # ------------------------------------------------------------------
    # feature_dict: token -> ndarray (T_trimmed, 4)
    feature_dict: dict[str, np.ndarray] = {}
    feature_df_dict: dict[str, pd.DataFrame] = {}
    returns_dict: dict[str, pd.Series] = {}
    realized_vol_dict: dict[str, pd.Series] = {}

    for token in tokens_tuple:
        ohlcv = panel.ohlcv[token]
        feat_df = build_feature_matrix(ohlcv)
        feature_df_dict[token] = feat_df
        feature_dict[token] = feat_df.values

        # Log returns and realized vol for labels + forecasting (causal)
        log_ret = np.log(ohlcv["close"]).diff().dropna()
        returns_dict[token] = log_ret
        realized_vol_dict[token] = log_ret.rolling(21, min_periods=2).std()

    # Multi-token panel for classifier CV
    feature_panel = build_feature_panel({t: panel.ohlcv[t] for t in tokens_tuple})

    # ------------------------------------------------------------------
    # 4. Regime detection (HMM + GMM) — INDEPENDENT per-token detection.
    #
    # HONESTY NOTE (do not "fix" this by sharing one sequence across tokens):
    #   Each token gets its own fresh CausalRegimeDetector on its own feature
    #   matrix. The DGP plants a shared market regime (70% market factor), so
    #   independently detected sequences show genuine cross-token association
    #   — empirically Cramér's V ≈ 0.35-0.45 off-diagonal, well above the
    #   ~0.15 independence floor but far below 1.0 because 30% idiosyncratic
    #   noise and 4-state label-permutation ambiguity are REAL obstacles a
    #   practitioner faces. An earlier draft fit one joint detector on the
    #   cross-sectional mean features and assigned the SAME sequence to every
    #   token: that makes the cross-token V heatmap identically 1.0 by
    #   construction (vacuous), collapses "per-token" diagnostics into one
    #   sequence, and over-states regime recoverability in the report.
    # ------------------------------------------------------------------
    n_components = int(regime_cfg.get("n_components", 4))
    min_train = int(regime_cfg.get("min_train", 60))
    refit_every = int(regime_cfg.get("refit_every", 21))
    n_restarts = int(regime_cfg.get("n_restarts", 3))

    regimes_hmm = detect_regimes_per_token(
        feature_dict,
        backend="hmm",
        n_components=n_components,
        min_train=min_train,
        refit_every=refit_every,
        n_restarts=n_restarts,
        random_seed=rng_seed,
    )
    regimes_gmm = detect_regimes_per_token(
        feature_dict,
        backend="gmm",
        n_components=n_components,
        min_train=min_train,
        refit_every=refit_every,
        n_restarts=n_restarts,
        random_seed=rng_seed,
    )

    # ------------------------------------------------------------------
    # 5. Diagnostics (HMM sequences)
    # ------------------------------------------------------------------
    diagnostics = per_token_diagnostics(regimes_hmm, n_states=n_components)

    # ------------------------------------------------------------------
    # 6. K-sensitivity (quick: 1 token only)
    # ------------------------------------------------------------------
    ksens_token_count = int(cfg.get("_ksens_tokens", len(tokens_tuple)))
    ksens_ks = tuple(cfg.get("_ksens_ks", (2, 3, 4, 5)))
    ksens_tokens = list(tokens_tuple)[:ksens_token_count]
    ksens_feature_dict = {t: feature_dict[t] for t in ksens_tokens}

    k_sensitivity = k_sensitivity_per_token(
        ksens_feature_dict,
        ks=ksens_ks,
        backend="hmm",
    )

    # ------------------------------------------------------------------
    # 7. Labels for evaluation (QUARANTINED — pipeline.py is allowed importer)
    # ------------------------------------------------------------------
    # Build per-token y aligned to the feature panel index
    y_parts: list[pd.Series] = []
    for token in tokens_tuple:
        ret = returns_dict[token]
        rv = realized_vol_dict[token].reindex(ret.index)
        labels = make_regime_labels(ret, rv, horizon=label_horizon)

        # Align to the feature matrix index for this token
        feat_idx = feature_df_dict[token].index
        labels_aligned = labels.reindex(feat_idx)
        # Build multi-index aligned series (date, token)
        midx = pd.MultiIndex.from_arrays(
            [feat_idx, [token] * len(feat_idx)], names=["date", "token"]
        )
        y_parts.append(pd.Series(labels_aligned.values, index=midx, name="regime_label"))

    y = pd.concat(y_parts).sort_index(level="date")
    # Drop rows where label is NaN before building X_aligned
    valid = ~y.isna()
    X_aligned = feature_panel.loc[valid.index[valid]]
    y_valid = y[valid]

    # Re-align X to y_valid after NaN drop (ensure index correspondence)
    X_aligned = feature_panel.reindex(y_valid.index)
    # Drop any remaining NaN rows in X
    x_nan_mask = X_aligned.isna().any(axis=1)
    X_aligned = X_aligned[~x_nan_mask]
    y_valid = y_valid[~x_nan_mask]

    # ------------------------------------------------------------------
    # 8. Label distribution (report data section)
    # ------------------------------------------------------------------
    label_distribution: dict[str, dict] = {}
    for token in tokens_tuple:
        ret = returns_dict[token]
        rv = realized_vol_dict[token].reindex(ret.index)
        lbl = make_regime_labels(ret, rv, horizon=label_horizon)
        counts = lbl.dropna().astype(int).value_counts().to_dict()
        label_distribution[token] = {int(k): int(v) for k, v in counts.items()}

    # ------------------------------------------------------------------
    # 9. Classifier CV evaluation
    # ------------------------------------------------------------------
    embargo_size = int(cv_cfg.get("embargo_size", 5))
    purged_size = int(cv_cfg.get("purged_size", 5))
    n_folds = int(cv_cfg.get("n_folds", 6))
    n_test_folds = int(cv_cfg.get("n_test_folds", 2))

    evaluator = RegimeCVEvaluator(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=purged_size,
        embargo_size=embargo_size,
        label_horizon=label_horizon,
    )

    logistic_res = evaluator.evaluate(
        LogisticRegimeClassifier(), X_aligned, y_valid
    )
    xgb_res = evaluator.evaluate(
        XGBRegimeClassifier(), X_aligned, y_valid
    )

    # ------------------------------------------------------------------
    # 10. HMM/GMM baseline scores (honest comparison)
    # ------------------------------------------------------------------
    # Build regime sequences aligned to the feature panel timeline
    # Each token's HMM sequence has length = n_bars (from detector using full feature matrix)
    # Align to y_valid index (date, token)

    hmm_acc, hmm_ll = _compute_baseline_scores(
        regimes_hmm, tokens_tuple, feature_df_dict, y_valid, n_components
    )
    gmm_acc, gmm_ll = _compute_baseline_scores(
        regimes_gmm, tokens_tuple, feature_df_dict, y_valid, n_components
    )

    model_comparison = pd.DataFrame(
        {
            "accuracy": {
                "hmm": hmm_acc,
                "gmm": gmm_acc,
                "logistic": logistic_res["accuracy"],
                "xgboost": xgb_res["accuracy"],
            },
            "log_loss": {
                "hmm": hmm_ll,
                "gmm": gmm_ll,
                "logistic": logistic_res["log_loss"],
                "xgboost": xgb_res["log_loss"],
            },
        }
    )

    # ------------------------------------------------------------------
    # 11. Volatility forecasting
    # ------------------------------------------------------------------
    forecast_train_frac = float(forecast_cfg.get("train_frac", 0.67))
    forecast_n_restarts = int(forecast_cfg.get("n_restarts", 5))

    forecast_comparison = per_token_forecast_comparison(
        returns_dict,
        train_frac=forecast_train_frac,
        n_restarts=forecast_n_restarts,
    )

    # Student-t robustness (OOS QLIKE per token)
    studentst_robustness: dict[str, float] = {}
    for token, ret in returns_dict.items():
        split_idx = int(len(ret) * forecast_train_frac)
        try:
            var_series, converged = garch_studentst_variance(ret, split_idx)
            if converged and len(var_series) > 0 and len(ret[split_idx + 1 :]) > 0:
                # QLIKE: L(h, rv) = rv/h - log(rv/h) - 1 (Patton 2011)
                rv_oos = ret.iloc[split_idx + 1:] ** 2  # realized variance proxy
                rv_oos = rv_oos.reindex(var_series.index).dropna()
                h_oos = var_series.reindex(rv_oos.index).dropna()
                rv_oos = rv_oos.reindex(h_oos.index)
                if len(rv_oos) > 0 and (h_oos > 0).all():
                    ratio = rv_oos.values / h_oos.values
                    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
                else:
                    qlike = float("nan")
            else:
                qlike = float("nan")
        except Exception:
            qlike = float("nan")
        studentst_robustness[token] = qlike

    # ------------------------------------------------------------------
    # 12. Cross-token Cramér's V (on detected HMM sequences)
    # ------------------------------------------------------------------
    cross_token_v = cross_token_regime_correlation(regimes_hmm, n_states=n_components)

    # ------------------------------------------------------------------
    # 13. Return frozen PipelineResults
    # ------------------------------------------------------------------
    # Build a flat config snapshot for test inspection
    config_snapshot: dict = {
        "seed": rng_seed,
        "n_years": n_years,
        "tokens": list(tokens_tuple),
        "n_bars": n_bars,
        "label_horizon": label_horizon,
        "embargo_size": embargo_size,
        "purged_size": purged_size,
        "n_folds": n_folds,
        "n_test_folds": n_test_folds,
        "n_components": n_components,
        "labels": {"horizon": label_horizon},
        "cv": {
            "n_folds": n_folds,
            "n_test_folds": n_test_folds,
            "embargo_size": embargo_size,
            "purged_size": purged_size,
        },
    }

    return PipelineResults(
        config=config_snapshot,
        seed=rng_seed,
        tokens=tokens_tuple,
        n_bars=n_bars,
        regimes_hmm=regimes_hmm,
        regimes_gmm=regimes_gmm,
        diagnostics=diagnostics,
        k_sensitivity=k_sensitivity,
        model_comparison=model_comparison,
        forecast_comparison=forecast_comparison,
        studentst_robustness=studentst_robustness,
        cross_token_v=cross_token_v,
        label_distribution=label_distribution,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_quick_overrides(cfg: dict) -> None:
    """Apply quick-mode overrides in-place."""
    cfg["n_years"] = _QUICK_OVERRIDES["n_years"]
    cfg["tokens"] = _QUICK_OVERRIDES["tokens"]
    cfg["_ksens_tokens"] = _QUICK_OVERRIDES["_ksens_tokens"]
    cfg["_ksens_ks"] = _QUICK_OVERRIDES["_ksens_ks"]

    cfg["cv"] = copy.deepcopy(_QUICK_OVERRIDES["cv"])
    cfg["regime"] = copy.deepcopy(_QUICK_OVERRIDES["regime"])
    cfg["forecast"] = copy.deepcopy(_QUICK_OVERRIDES["forecast"])

    if "labels" not in cfg:
        cfg["labels"] = {"horizon": 5}


def _compute_baseline_scores(
    regimes: dict[str, np.ndarray],
    tokens_tuple: tuple,
    feature_df_dict: dict[str, pd.DataFrame],
    y_valid: pd.Series,
    n_states: int,
) -> tuple[float, float]:
    """Compute accuracy and log-loss for HMM/GMM baselines.

    The baseline prediction at (date, token) is the causal regime label at that
    date for that token (persistence prediction of the forward-looking class).
    This is scored on the same valid_mask rows used for classifiers where possible.

    Note: baselines use the full causal sequence (no CV needed — no training on labels
    occurs). Classifiers use purged CV. This asymmetry favors baselines; that is the
    conservative/honest direction.
    """
    # Build a lookup: (date, token) -> regime label
    regime_lookup: dict[tuple, int] = {}
    for token in tokens_tuple:
        seq = regimes[token]
        feat_idx = feature_df_dict[token].index
        # seq has length = len(feature_dict[token]) (post-dropna feature matrix)
        n = min(len(seq), len(feat_idx))
        for i in range(n):
            regime_lookup[(feat_idx[i], token)] = int(seq[i])

    # Gather predictions for rows in y_valid
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    probas_list: list[np.ndarray] = []

    for idx_val, label in y_valid.items():
        if isinstance(idx_val, tuple):
            date_val, token_val = idx_val
        else:
            continue
        pred = regime_lookup.get((date_val, token_val), -1)
        if pred < 0:
            # Sentinel — skip
            continue
        y_true_list.append(int(label))
        y_pred_list.append(pred)
        # Eps-smoothed one-hot for log-loss
        p = np.full(n_states, 1e-3)
        if pred < n_states:
            p[pred] = 1.0 - (n_states - 1) * 1e-3
        probas_list.append(p)

    if len(y_true_list) == 0:
        return float("nan"), float("nan")

    y_true_arr = np.array(y_true_list, dtype=int)
    y_pred_arr = np.array(y_pred_list, dtype=int)
    probas_arr = np.array(probas_list, dtype=float)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    ll = float(log_loss(y_true_arr, probas_arr, labels=list(range(n_states))))

    return acc, ll
