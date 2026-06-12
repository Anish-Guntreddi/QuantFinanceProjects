"""Per-token diagnostics — thin adapter over macroregime.regime.diagnostics
and macroregime.evaluation.k_sensitivity.

DESIGN NOTES
------------
This module contains zero reimplementation.  All computation is delegated:
- transition_matrix, dwell_times -> macroregime.regime.diagnostics
- k_sensitivity -> macroregime.evaluation.k_sensitivity

ANTI-FEATURES (explicitly forbidden):
- Return-based K selection (see macroregime evaluation module for the locked
  anti-feature rationale).  K is selected on economic interpretability and
  structural metrics (dwell times, transition matrices, BIC) only.
- Direct hmmlearn or sklearn.mixture imports.

The k_sensitivity_per_token function runs macroregime's k_sensitivity on each
token independently, preserving the per-token independence invariant established
in regime/detector.py.

K=3 agreement baseline: macroregime.evaluation.k_sensitivity uses K=3 as the
agreement reference internally (agreement_vs_k3).  DGP-true K=4 for DeFi
(bull/bear x hi/lo-vol) — document in report, do not override the baseline.
"""
from __future__ import annotations

import numpy as np

from macroregime.regime.diagnostics import dwell_times, transition_matrix
from macroregime.evaluation import k_sensitivity

__all__ = ["per_token_diagnostics", "k_sensitivity_per_token"]


def per_token_diagnostics(
    regimes: dict[str, np.ndarray],
    n_states: int = 4,
) -> dict[str, dict]:
    """Compute transition matrix and dwell times for each token's regime sequence.

    Delegates entirely to macroregime.regime.diagnostics.{transition_matrix,
    dwell_times}.  Sentinel values (-1) in the sequences are ignored by both
    functions.

    Parameters
    ----------
    regimes : dict[str, np.ndarray]
        Mapping from token symbol to causal label sequence (shape (T,)).
        Entries equal to -1 (warm-up sentinels) are ignored.
    n_states : int
        Number of regime states K.  Unvisited states receive uniform rows
        (1/K) in the transition matrix (row-stochastic by construction).

    Returns
    -------
    dict[str, dict]
        Token -> {
            "transition_matrix": np.ndarray (K, K),  # row-stochastic, empirical
            "dwell_times": dict[int, float],          # mean run length per state
        }
    """
    result: dict[str, dict] = {}

    for token, seq in regimes.items():
        seq_arr = np.asarray(seq, dtype=int)
        tm = transition_matrix(seq_arr, n_states=n_states)
        dt = dwell_times(seq_arr, n_states=n_states)
        result[token] = {
            "transition_matrix": tm,
            "dwell_times": dt,
        }

    return result


def k_sensitivity_per_token(
    feature_dict: dict[str, np.ndarray],
    ks: tuple[int, ...] = (2, 3, 4, 5),
    backend: str = "hmm",
) -> dict[str, dict[int, dict]]:
    """Run k-sensitivity analysis per token, reporting structural metrics only.

    For each token, delegates to macroregime.evaluation.k_sensitivity with
    the given K values and backend.  Tokens are processed independently.

    The agreement baseline inside macroregime.evaluation.k_sensitivity is K=3
    (by macroregime convention).  DGP-true K=4 for DeFiRegimeNet; this is
    reported in the research report, not enforced here.

    Parameters
    ----------
    feature_dict : dict[str, np.ndarray]
        Token -> feature matrix (T, n_features), chronologically ordered.
    ks : tuple[int, ...]
        K values to evaluate.  Recommended: (2, 3, 4, 5) for DeFi.
    backend : {"hmm", "gmm"}
        Model backend for CausalRegimeDetector.

    Returns
    -------
    dict[str, dict[int, dict]]
        Token -> K -> {
            "dwell_times": dict[int, float],
            "transition_matrix": np.ndarray (K, K),
            "agreement_vs_k3": float,   # 1.0 for K=3 (self-agreement)
        }

    Notes
    -----
    K selection must be done on structural metrics or economic interpretability
    only — not on return-based criteria (locked anti-feature from macroregime).
    """
    result: dict[str, dict[int, dict]] = {}

    for token, X in feature_dict.items():
        X_arr = np.asarray(X, dtype=float)
        result[token] = k_sensitivity(X_arr, ks=ks, backend=backend)

    return result
