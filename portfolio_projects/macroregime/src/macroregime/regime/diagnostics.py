"""Regime persistence diagnostics.

These functions operate on *causal* regime sequences — the sequences that
strategies actually trade on — rather than a model's fitted transition matrix.
This distinction matters: model.transmat_ is a smoothed population estimate
while the empirical transition matrix reflects the actual regime assignments
produced by rolling re-fit.

Exported API
------------
- transition_matrix(seq, n_states) -> np.ndarray  shape (K, K)
- dwell_times(seq, n_states) -> dict[int, float]
- regime_run_lengths(seq) -> list[tuple[int, int]]

Optionally, a fitted model may be passed to also report model.transmat_ and
model.get_stationary_distribution() for comparison in the research report.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def transition_matrix(
    seq: np.ndarray,
    n_states: int,
    model: "Any | None" = None,
) -> np.ndarray:
    """Compute empirical transition matrix from a causal regime sequence.

    Parameters
    ----------
    seq : np.ndarray of int
        Causal label sequence. Entries equal to -1 (sentinel for
        pre-training period) are ignored.
    n_states : int
        Number of states K.  Rows for unvisited states receive a
        uniform distribution (1/K) rather than NaN or all-zero.
    model : optional
        If provided and has ``transmat_`` attribute, the method returns
        only the empirical matrix (the caller can inspect model.transmat_
        directly for comparison).

    Returns
    -------
    T : np.ndarray of shape (K, K)
        ``T[i, j]`` = probability of transitioning from state i to state j.
        Rows sum to 1.0.
    """
    seq = np.asarray(seq, dtype=int)
    valid = seq != -1
    valid_seq = seq[valid]

    counts = np.zeros((n_states, n_states), dtype=float)
    if len(valid_seq) > 1:
        from_states = valid_seq[:-1]
        to_states = valid_seq[1:]
        for f, t in zip(from_states, to_states):
            if 0 <= f < n_states and 0 <= t < n_states:
                counts[f, t] += 1.0

    # Normalize rows; unvisited rows → uniform
    row_sums = counts.sum(axis=1, keepdims=True)
    unvisited = (row_sums.squeeze() == 0)
    row_sums[unvisited] = 1.0          # avoid divide-by-zero
    T = counts / row_sums
    T[unvisited] = 1.0 / n_states      # uniform prior for unvisited rows
    return T


def dwell_times(
    seq: np.ndarray,
    n_states: int,
) -> dict[int, float]:
    """Mean run length (dwell time) per state in a causal regime sequence.

    Parameters
    ----------
    seq : np.ndarray of int
        Causal label sequence. Entries equal to -1 are ignored.
    n_states : int
        Number of states (used to initialise the output dict).

    Returns
    -------
    dt : dict[int, float]
        ``dt[state]`` = mean run length for that state.  States with no
        runs receive 0.0.
    """
    runs = regime_run_lengths(seq)
    # Accumulate run lengths per state
    lengths_per_state: dict[int, list[int]] = {s: [] for s in range(n_states)}
    for state, length in runs:
        if 0 <= state < n_states:
            lengths_per_state[state].append(length)

    return {
        s: float(np.mean(lens)) if lens else 0.0
        for s, lens in lengths_per_state.items()
    }


def regime_run_lengths(seq: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encoding of a regime sequence (ignoring -1 sentinels).

    Parameters
    ----------
    seq : np.ndarray of int
        Regime label sequence; -1 entries are skipped.

    Returns
    -------
    runs : list of (state, length) tuples
        E.g. seq=[0,0,0,1,1,2,0,0] → [(0,3),(1,2),(2,1),(0,2)]
    """
    seq = np.asarray(seq, dtype=int)
    valid = seq[seq != -1]
    if len(valid) == 0:
        return []

    runs: list[tuple[int, int]] = []
    current_state = int(valid[0])
    current_len = 1
    for label in valid[1:]:
        label = int(label)
        if label == current_state:
            current_len += 1
        else:
            runs.append((current_state, current_len))
            current_state = label
            current_len = 1
    runs.append((current_state, current_len))
    return runs
