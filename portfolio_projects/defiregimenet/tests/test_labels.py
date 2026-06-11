"""Label quarantine enforcement + forward-label tests (05-02).

The AST quarantine guard below is LIVE from Wave 0: it walks every source
file and fails if any module outside the allowed importers touches
defiregimenet.labels. It passes trivially before labels.py exists and
protects every wave-2 executor from accidentally wiring forward-looking
labels into a feature or model path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

QUARANTINED_MODULE = "defiregimenet.labels"
ALLOWED_IMPORTERS = {"defiregimenet.evaluation", "defiregimenet.pipeline"}


def test_label_quarantine():
    """No source module outside evaluation/pipeline may import labels.py.

    Forward-looking regime labels (built from FUTURE returns and FUTURE
    realized vol) are evaluation-only ground truth. Any import from a
    feature, model, or training module is look-ahead leakage by
    construction (DFR-02 strict causal separation).
    """
    src_root = Path(__file__).parents[1] / "src" / "defiregimenet"
    assert src_root.exists(), f"source root not found: {src_root}"

    violations: list[str] = []
    for path in src_root.rglob("*.py"):
        module_rel = path.relative_to(src_root)
        module_name = "defiregimenet." + ".".join(module_rel.with_suffix("").parts)
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        if any(allowed in module_name for allowed in ALLOWED_IMPORTERS):
            continue
        if module_name.split(".")[-1] == "labels":
            continue  # labels.py itself is allowed to exist

        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "labels" in node.module:
                    violations.append(f"{module_name}: imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "labels" in alias.name:
                        violations.append(f"{module_name}: imports {alias.name}")

    assert violations == [], f"Label quarantine violated: {violations}"


def test_labels_are_forward_looking():
    """Perturbing returns at bar t+1 changes the label at t for at least one t.
    Truncating the series at bar t (removing all future bars) leaves label at t NaN.
    """
    from defiregimenet.labels import make_regime_labels

    rng = np.random.default_rng(0)
    n = 60
    returns = pd.Series(rng.normal(0.0, 0.02, n))
    realized_vol = returns.abs().rolling(5, min_periods=1).mean()
    horizon = 5

    labels_base = make_regime_labels(returns, realized_vol, horizon=horizon)

    # Perturb returns at bar t+1 and confirm at least one label at t changes
    returns_perturbed = returns.copy()
    # Multiplying bar 1 by 10 changes the forward window for bar 0
    returns_perturbed.iloc[1] *= 10.0
    rv_perturbed = returns_perturbed.abs().rolling(5, min_periods=1).mean()
    labels_perturbed = make_regime_labels(returns_perturbed, rv_perturbed, horizon=horizon)

    # At least one label in [0..n-horizon-1] differs
    valid_mask = ~labels_base.isna() & ~labels_perturbed.isna()
    assert valid_mask.any(), "Expected at least some valid (non-NaN) labels"
    assert not (labels_base[valid_mask] == labels_perturbed[valid_mask]).all(), (
        "Perturbing future data should change at least one label"
    )

    # Truncating at bar t leaves label at t NaN: build with only t+1 bars (horizon away from end)
    # Use a short series where the last bar is < horizon away from end
    short_returns = returns.iloc[:horizon]  # only `horizon` bars — all labels are NaN
    short_rv = realized_vol.iloc[:horizon]
    short_labels = make_regime_labels(short_returns, short_rv, horizon=horizon)
    assert short_labels.isna().all(), (
        f"Labels on a series shorter than horizon should all be NaN; got {short_labels.values}"
    )


def test_last_horizon_nan():
    """labels.iloc[-horizon:] are all NaN."""
    from defiregimenet.labels import make_regime_labels

    rng = np.random.default_rng(1)
    n = 100
    horizon = 5
    returns = pd.Series(rng.normal(0.0, 0.02, n))
    realized_vol = returns.abs().rolling(21, min_periods=1).std()

    labels = make_regime_labels(returns, realized_vol, horizon=horizon)

    tail = labels.iloc[-horizon:]
    assert tail.isna().all(), (
        f"Last {horizon} bars should be NaN; got {tail.values}"
    )


def test_label_encoding():
    """
    Hand-built series with known forward up-move/low-vol window produces label 2 (bull/low).
    Known down-move/high-vol window produces label 1 (bear/high).

    Encoding: state = bull_flag * 2 + high_vol_flag
      0 = bear/low, 1 = bear/high, 2 = bull/low, 3 = bull/high
    """
    from defiregimenet.labels import make_regime_labels

    horizon = 5
    # Build a 20-bar series
    n = 20
    # Strong upward returns for bars 0..horizon-1 (these become the FORWARD window for bar -1)
    # Use a construction where we know bars 10..14 (the forward window of bar 9) are:
    #   - positive (bull) and low vol

    # Low vol, bullish forward window starting at bar 0 for label at bar -horizon
    # returns[horizon..2*horizon-1] = strong up, low vol  → label at bar 0 = 2 (bull/low)
    returns_bull_low = pd.Series([0.05] * n)  # constant strong up = low vol, bull
    realized_vol_bull_low = returns_bull_low.abs().rolling(5, min_periods=1).std()
    realized_vol_bull_low = realized_vol_bull_low.fillna(0.0)

    labels_bl = make_regime_labels(returns_bull_low, realized_vol_bull_low, horizon=horizon)

    # Only check valid (non-NaN) labels
    valid_bl = labels_bl.dropna()
    assert len(valid_bl) > 0, "Expected some valid labels for bull/low case"
    # All valid should be 2 (bull/low — positive return, near-zero vol)
    assert (valid_bl == 2).all(), (
        f"Expected all bull/low labels=2, got unique={valid_bl.unique()}"
    )

    # High vol, bearish forward window
    # Alternating large +/- creates high vol; negative mean = bear
    returns_bear_high = pd.Series([-0.05, 0.10] * (n // 2))
    # net drift is positive but median of forward return is negative? Use strongly negative returns
    returns_bear_high = pd.Series([-0.10, 0.05] * (n // 2))
    realized_vol_bear_high = returns_bear_high.abs().rolling(5, min_periods=1).std()
    realized_vol_bear_high = realized_vol_bear_high.fillna(0.0)

    labels_bh = make_regime_labels(returns_bear_high, realized_vol_bear_high, horizon=horizon)
    valid_bh = labels_bh.dropna()
    assert len(valid_bh) > 0, "Expected some valid labels for bear/high case"
    # The alternating pattern has nonzero std → high vol
    # Negative forward mean (more negative bars) → bear
    # Label should be 1 (bear/high)
    assert (valid_bh == 1).all(), (
        f"Expected all bear/high labels=1, got unique={valid_bh.unique()}"
    )


def test_label_distribution(seeded_crypto_panel):
    """On seeded_crypto_panel BTC, all 4 label values occur (non-degenerate)."""
    from defiregimenet.labels import make_regime_labels

    ohlcv = seeded_crypto_panel.ohlcv["BTC"]
    returns = np.log(ohlcv["close"]).diff().dropna()
    realized_vol = returns.rolling(21, min_periods=1).std()
    realized_vol = realized_vol.reindex(returns.index)

    labels = make_regime_labels(returns, realized_vol, horizon=5)
    valid = labels.dropna()

    unique_states = set(valid.unique().astype(int))
    assert unique_states == {0, 1, 2, 3}, (
        f"Expected all 4 label states on seeded BTC; got {unique_states}"
    )
