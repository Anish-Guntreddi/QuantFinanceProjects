"""Per-token causal regime detection — thin adapter over macroregime.CausalRegimeDetector.

DESIGN NOTES
------------
This module is a thin wrapper. It does NOT call hmmlearn or sklearn.mixture
directly. All fitting, oracle-guarantee enforcement, and label alignment are
delegated to macroregime.regime.causal.CausalRegimeDetector.

Key invariants (by delegation):
- One fresh CausalRegimeDetector instance PER TOKEN: token feature streams are
  kept fully independent. Mixing tokens into a single matrix would create
  frequency-alignment artifacts and is semantically wrong.
- The oracle guarantee is inherited from macroregime: the label at bar t depends
  only on X[:t+1] (HMM) or X[t:t+1] with a model fitted on X[:t] (GMM).
  Appending future bars never changes any historical label.
- All seeds are pure functions of the constructor arguments — never of len(X).
  This makes the output deterministic across runs with the same random_seed.
- Warm-up sentinels: labels at t < min_train are -1.

CausalRegimeDetector is re-exported for downstream convenience (e.g. pipeline
modules that need direct access to the underlying detector).

ANTI-PATTERNS (forbidden):
  hmmlearn.hmm.GaussianHMM    <- use CausalRegimeDetector(backend="hmm")
  sklearn.mixture.GaussianMixture <- use CausalRegimeDetector(backend="gmm")
  model.predict(X)            <- FORBIDDEN smoothed pattern; delegated away
"""
from __future__ import annotations

from typing import Union

import numpy as np

# Re-export for downstream convenience
from macroregime.regime.causal import CausalRegimeDetector

__all__ = ["detect_regimes_per_token", "CausalRegimeDetector"]


def detect_regimes_per_token(
    feature_dict: dict[str, np.ndarray],
    backend: str = "hmm",
    n_components: int = 4,
    min_train: int = 60,
    refit_every: int = 21,
    n_restarts: int = 3,
    observable_dim: int = 0,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Detect causal regime sequences per token using CausalRegimeDetector.

    Each token receives its own fresh CausalRegimeDetector instance so that
    regime sequences are fully independent across tokens.

    Parameters
    ----------
    feature_dict : dict[str, np.ndarray]
        Mapping from token symbol to a pre-built feature matrix of shape
        (T, n_features), ordered chronologically.  Feature construction is
        the caller's responsibility — this function is deliberately decoupled
        from defiregimenet.features.crypto so that wave-2 plans can run in
        parallel.
    backend : {"hmm", "gmm"}
        Model family passed to CausalRegimeDetector.
    n_components : int
        Number of regime states K.  Recommended: 4 for crypto (bull/bear x
        hi/lo-vol, matching the DGP true_states convention in 05-01).
    min_train : int
        Minimum observations before the first fit.  Labels at t < min_train
        are -1 (sentinel).
    refit_every : int
        Re-fit interval in bars.
    n_restarts : int
        Number of independent random restarts for multi-start fitting.
    observable_dim : int
        Feature dimension used for label alignment (argsort by mean).
    random_seed : int
        Base random seed.  Seeds for restarts are pure functions of offset,
        not of data length — preserving the oracle invariant.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from token symbol to causal label sequence of shape (T,).
        Warm-up entries (t < min_train) are -1; post-warm-up labels are in
        {0, ..., n_components - 1}.

    Oracle guarantee
    ----------------
    Inherited from CausalRegimeDetector.fit_predict_causal: for any token,
    the label at bar t depends only on feature_dict[token][:t+1] (HMM) or
    feature_dict[token][t:t+1] with a model fitted on [:t] (GMM).  A run on
    a longer prefix of the same feature matrix produces identical labels for
    all previously-seen bars.
    """
    regimes: dict[str, np.ndarray] = {}

    for token, X in feature_dict.items():
        X_arr = np.asarray(X, dtype=float)

        # One fresh detector per token — never shared across tokens
        detector = CausalRegimeDetector(
            backend=backend,
            n_components=n_components,
            min_train=min_train,
            refit_every=refit_every,
            n_restarts=n_restarts,
            observable_dim=observable_dim,
            random_seed=random_seed,
        )

        regimes[token] = detector.fit_predict_causal(X_arr)

    return regimes
