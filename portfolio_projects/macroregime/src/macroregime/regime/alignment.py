"""Regime label alignment via argsort of state means.

After every HMM or GMM re-fit the component indices (raw states) are
arbitrary — the model can swap which component ID corresponds to, say,
'recession' vs 'expansion'.  To produce a *consistent* semantic ordering
we sort components by their mean along a chosen observable dimension and
return the **inverse permutation** that maps each raw component label to
its sorted rank.

Why the inverse permutation?
----------------------------
``np.argsort(means[:, d])`` returns the raw component indices ordered by
ascending mean on dimension d — i.e. ``sorted_raw_indices[rank] = raw``.
To convert a *raw label* to its *aligned rank* we need the inverse:
``mapping[raw] = rank``, which is ``np.argsort(np.argsort(means[:, d]))``.

Example
-------
>>> means = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
>>> align_regime_labels(means, observable_dim=0)
array([2, 0, 1])
# raw 0 (mean=3) → rank 2 (highest)
# raw 1 (mean=1) → rank 0 (lowest)
# raw 2 (mean=2) → rank 1 (middle)
"""
from __future__ import annotations

import numpy as np


def align_regime_labels(
    model_or_means: "np.ndarray | object",
    observable_dim: int = 0,
) -> np.ndarray:
    """Return the raw-to-aligned mapping (inverse argsort of means).

    Parameters
    ----------
    model_or_means : array-like of shape (K, n_features) OR any object
        with a ``.means_`` attribute (duck-type for GaussianHMM /
        GaussianMixture). If a 2-D numpy array is passed directly it is
        used as the means matrix.
    observable_dim : int
        Feature dimension on which to sort (default 0 — typically the
        first macro observable, e.g. CPI m/m growth).

    Returns
    -------
    mapping : np.ndarray of shape (K,), dtype int
        ``mapping[raw_state]`` is the aligned rank (0 = lowest mean on
        ``observable_dim``).  Apply to a raw label sequence via
        ``aligned = mapping[raw_sequence]``.

    Notes
    -----
    Compatible with both ``GaussianHMM.means_`` (hmmlearn) and
    ``GaussianMixture.means_`` (scikit-learn) — both expose a
    ``(K, n_features)`` float array.
    """
    # Duck-type: accept a fitted model or a raw means matrix
    if isinstance(model_or_means, np.ndarray):
        means = model_or_means
    else:
        means = model_or_means.means_

    means = np.asarray(means, dtype=float)
    if means.ndim != 2:
        raise ValueError(
            f"means must be 2-D (K, n_features), got shape {means.shape}"
        )

    # argsort gives: sorted_raw[rank] = raw_state_index
    # double argsort gives: mapping[raw_state] = rank  (the inverse permutation)
    order = np.argsort(means[:, observable_dim])        # raw indices by ascending mean
    mapping = np.argsort(order)                          # inverse: raw -> rank
    return mapping.astype(int)
