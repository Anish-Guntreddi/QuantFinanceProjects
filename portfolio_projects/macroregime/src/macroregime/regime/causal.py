"""CausalRegimeDetector — rolling re-fit regime detection with the oracle guarantee.

RESEARCH CONTEXT (hmmlearn 0.3.3 verified empirically):
==========================================================
hmmlearn's predict() and predict_proba() use Viterbi / forward-backward on
the ENTIRE input sequence, i.e. they are SMOOTHED estimators.  If future data
is appended and predict() is called again on the full sequence, up to 55% of
historical labels can change.  This violates causality and makes any strategy
built on those labels look-ahead biased.

FORBIDDEN PATTERNS in any signal path:
    model.predict(X)          # FORBIDDEN — smoothed, sees future
    model.predict_proba(X)    # FORBIDDEN — smoothed, sees future
    model.score_samples(X)    # FORBIDDEN — same issue

THE ONLY SAFE CAUSAL PATTERN (HMM):
    # At time t, after fitting on X[:t]:
    regime_at_t = model.predict(X[:t + 1])[-1]
    # Prediction only uses observations 0..t inclusive.
    # Appending bars t+1..T to X and calling predict on X[:t+1] again
    # yields the SAME result because the prefix is unchanged.

THE ONLY SAFE CAUSAL PATTERN (GMM):
    # At time t, after fitting on X[:t] — GMM has fixed params:
    regime_at_t = gm.predict(X[t:t + 1])[0]
    # Single-sample scoring with fixed params is inherently causal.

ORACLE GUARANTEE (by construction):
The refit schedule and every random seed are PURE FUNCTIONS OF t — never of
len(X).  Therefore, a run on X[:N] and a run on X[:N+k] agree on every t < N.

CODEX AUDIT NOTE:
    grep -rn "predict_proba|score_samples" src/macroregime/regime/
must return zero matches in the signal path.  Only predict(X[:t+1])[-1] and
predict(X[t:t+1])[0] are permitted below.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from macroregime.regime.alignment import align_regime_labels

if TYPE_CHECKING:
    pass


class CausalRegimeDetector:
    """Rolling re-fit regime detector producing strictly causal label sequences.

    Supports two backends:
    - ``"hmm"`` : hmmlearn GaussianHMM with multi-start + warm-start selection
    - ``"gmm"`` : sklearn GaussianMixture with multi-start (n_init)

    Parameters
    ----------
    backend : {"hmm", "gmm"}
        Model family.
    n_components : int
        Number of regime states K (default 3).
    min_train : int
        Minimum number of observations required before the first fit.
        Labels at t < min_train are set to -1 (sentinel).
    refit_every : int
        Re-fit interval in bars (default 21 ≈ one month of daily data).
    n_restarts : int
        Number of independent random restarts for multi-start fitting.
        HMM: n_restarts cold starts PLUS 1 warm-start = n_restarts+1 candidates.
        GMM: passed directly as n_init to GaussianMixture.
    covariance_type : str
        Covariance structure passed to the underlying model (default "diag").
    n_iter : int
        Maximum EM iterations (default 200).
    tol : float
        EM convergence tolerance (default 1e-4).
    observable_dim : int
        Feature dimension used for label alignment (default 0).
    random_seed : int
        Base random seed.  Seeds for restarts are ``random_seed + offset``
        for offset in range(n_restarts) — a pure function of offset, not
        of len(X).

    Attributes
    ----------
    last_model_ : fitted model object (after at least one refit)
    refit_times_ : list[int] — bar indices where re-fits occurred
    alignments_ : list[np.ndarray] — raw->aligned mapping after each re-fit

    Notes
    -----
    The refit schedule is: t == min_train  OR  (t - min_train) % refit_every == 0.
    Both conditions depend ONLY on t, making them pure functions of position in X
    and therefore independent of len(X).  This is what guarantees the oracle.
    """

    def __init__(
        self,
        backend: str = "hmm",
        n_components: int = 3,
        min_train: int = 60,
        refit_every: int = 21,
        n_restarts: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 200,
        tol: float = 1e-4,
        observable_dim: int = 0,
        random_seed: int = 42,
    ) -> None:
        if backend not in ("hmm", "gmm"):
            raise ValueError(f"backend must be 'hmm' or 'gmm', got {backend!r}")
        self.backend = backend
        self.n_components = n_components
        self.min_train = min_train
        self.refit_every = refit_every
        self.n_restarts = n_restarts
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.observable_dim = observable_dim
        self.random_seed = random_seed

        # Set after fit_predict_causal
        self.last_model_ = None
        self.refit_times_: list[int] = []
        self.alignments_: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict_causal(self, X: np.ndarray) -> np.ndarray:
        """Produce a causal regime label sequence for all bars in X.

        Parameters
        ----------
        X : np.ndarray of shape (T, n_features)
            Feature matrix.  Rows must be ordered chronologically.

        Returns
        -------
        regimes : np.ndarray of shape (T,) dtype int
            Regime labels in [0, K).  Labels at t < min_train are -1.

        Causality guarantee
        -------------------
        For any t, ``regimes[t]`` is computed solely from ``X[:t+1]`` (HMM)
        or ``X[t:t+1]`` with a model fitted on ``X[:t]`` (GMM).  Appending
        future observations beyond T leaves all labels 0..T-1 unchanged.
        """
        X = np.asarray(X, dtype=float)
        T = len(X)

        regimes = np.full(T, -1, dtype=int)
        self.last_model_ = None
        self.refit_times_ = []
        self.alignments_ = []

        _current_model = None
        _current_alignment: np.ndarray | None = None

        for t in range(T):
            # Determine if this bar triggers a re-fit
            should_refit = (
                t == self.min_train
                or (
                    t > self.min_train
                    and (t - self.min_train) % self.refit_every == 0
                )
            )

            if should_refit:
                if self.backend == "hmm":
                    _current_model = self._fit_hmm(X[:t], _current_model)
                else:
                    _current_model = self._fit_gmm(X[:t])

                _current_alignment = align_regime_labels(
                    _current_model.means_, self.observable_dim
                )
                self.last_model_ = _current_model
                self.refit_times_.append(t)
                self.alignments_.append(_current_alignment.copy())

            # Only produce labels after min_train AND once a model exists
            if t < self.min_train or _current_model is None:
                continue

            # -----------------------------------------------------------------
            # Causal label production — THE ONLY PERMITTED PATTERN
            # -----------------------------------------------------------------
            if self.backend == "hmm":
                # Fit was on X[:t] (where t == min_train on first fit).
                # For subsequent bars (no refit), model was fit on X[:last_refit_t].
                # predict(X[:t+1]) uses Viterbi on the prefix ending at t.
                # NOTE: predict on a PREFIX of the same training window is safe
                # because no future bar beyond t is included.
                raw_label = int(_current_model.predict(X[: t + 1])[-1])
            else:
                # GMM: single-bar scoring with fixed model params — inherently causal
                raw_label = int(_current_model.predict(X[t : t + 1])[0])

            regimes[t] = int(_current_alignment[raw_label])

        return regimes

    # ------------------------------------------------------------------
    # Private: HMM fitting with multi-start + warm-start
    # ------------------------------------------------------------------

    def _fit_hmm(
        self,
        X_train: np.ndarray,
        prev_model: "object | None",
    ) -> "object":
        """Multi-start HMM fit with optional warm-start candidate.

        Candidate pool:
        - n_restarts cold starts with seeds random_seed+0, ..., random_seed+n_restarts-1
        - 1 warm-start from the previous model (if available)

        All seeds are pure functions of their offset (not of len(X_train)),
        preserving the oracle invariant.
        """
        from hmmlearn import hmm

        best_model = None
        best_score = -np.inf

        # Cold-start candidates
        for offset in range(self.n_restarts):
            seed = self.random_seed + offset
            candidate = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=seed,
            )
            try:
                candidate.fit(X_train)
                score = candidate.score(X_train)
                if score > best_score:
                    best_score = score
                    best_model = candidate
            except Exception:
                pass  # degenerate init — skip this candidate

        # Warm-start candidate (starts from previous model's parameters)
        if prev_model is not None:
            try:
                warm = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=self.random_seed,  # deterministic seed, not data-dependent
                )
                # Copy parameters from previous model; init_params='' tells hmmlearn
                # NOT to re-initialise parameters, so EM starts from prev state.
                warm.startprob_ = prev_model.startprob_.copy()
                warm.transmat_ = prev_model.transmat_.copy()
                warm.means_ = prev_model.means_.copy()
                warm.covars_ = prev_model.covars_.copy()
                warm.init_params = ""
                warm.fit(X_train)
                score = warm.score(X_train)
                if score > best_score:
                    best_score = score
                    best_model = warm
            except Exception:
                pass

        if best_model is None:
            raise RuntimeError(
                "All HMM fitting candidates failed.  Check X_train for degenerate data."
            )

        return best_model

    # ------------------------------------------------------------------
    # Private: GMM fitting
    # ------------------------------------------------------------------

    def _fit_gmm(self, X_train: np.ndarray) -> "object":
        """Fit GaussianMixture on X_train with n_init restarts.

        The random_state is fixed to self.random_seed — a pure function of
        the constructor argument, never of len(X_train).
        """
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_restarts,
            max_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_seed,
        )
        gm.fit(X_train)
        return gm
