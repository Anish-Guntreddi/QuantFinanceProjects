"""
Cross-token regime association analytics.

Provides pairwise Cramér's V between per-token regime label sequences,
producing the symmetric correlation matrix consumed by the report heatmap.

Public API
----------
cramers_v(labels_a, labels_b, n_states=4) -> float
cross_token_regime_correlation(regime_sequences, n_states=4) -> pd.DataFrame

Design notes
------------
* -1 entries are warm-up sentinels — excluded from both sequences before any
  contingency computation (both sides masked simultaneously so alignment is
  preserved).
* The contingency table is built with np.add.at (O(n), vectorised).
* Rows/columns whose marginal sum is zero are dropped before chi2_contingency;
  scipy raises a ValueError on zero marginal totals.
* V = sqrt(chi2 / (n * (k - 1))) where k = min(reduced_rows, reduced_cols).
  Guards: empty table or k <= 1 → return 0.0.
* The result is clipped to [0, 1] to absorb floating-point rounding noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_states: int = 4,
) -> float:
    """
    Cramér's V association statistic between two integer regime-label sequences.

    Parameters
    ----------
    labels_a, labels_b : array-like of int
        Per-token regime label sequences.  -1 indicates warm-up (excluded).
        Must have the same length.
    n_states : int
        Number of distinct regime states (used to size the contingency table).

    Returns
    -------
    float
        Cramér's V in [0, 1].  Returns 0.0 for degenerate cases (empty table
        after sentinel removal, or only one distinct state on either axis).
    """
    a = np.asarray(labels_a, dtype=int)
    b = np.asarray(labels_b, dtype=int)

    # Mask warm-up sentinels (either side == -1)
    valid = (a != -1) & (b != -1)
    a = a[valid]
    b = b[valid]

    n = len(a)
    if n == 0:
        return 0.0

    # Build n_states x n_states contingency table (vectorised)
    table = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(table, (a, b), 1)

    # Drop all-zero rows and columns (scipy raises on zero marginals)
    row_mask = table.sum(axis=1) > 0
    col_mask = table.sum(axis=0) > 0
    table = table[np.ix_(row_mask, col_mask)]

    r, c = table.shape
    if r <= 1 or c <= 1:
        # Only one active row or column — no association to quantify
        return 0.0

    chi2, _, _, _ = chi2_contingency(table)

    k = min(r, c)
    v = np.sqrt(chi2 / (n * (k - 1)))

    # Clip to [0, 1] to absorb floating-point overshoot
    return float(np.clip(v, 0.0, 1.0))


def cross_token_regime_correlation(
    regime_sequences: dict[str, np.ndarray],
    n_states: int = 4,
) -> pd.DataFrame:
    """
    Pairwise Cramér's V matrix across all token regime sequences.

    Parameters
    ----------
    regime_sequences : dict[str, np.ndarray]
        Mapping of token name → integer regime-label sequence.
        Sequences need not be trimmed; -1 sentinels are handled internally.
    n_states : int
        Number of distinct regime states.

    Returns
    -------
    pd.DataFrame
        Square symmetric DataFrame indexed and columned by token names.
        Diagonal entries are 1.0.  All values are in [0, 1].
    """
    tokens = list(regime_sequences.keys())
    n = len(tokens)
    mat = np.eye(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            v = cramers_v(regime_sequences[tokens[i]], regime_sequences[tokens[j]], n_states)
            mat[i, j] = v
            mat[j, i] = v  # symmetric

    return pd.DataFrame(mat, index=tokens, columns=tokens)
