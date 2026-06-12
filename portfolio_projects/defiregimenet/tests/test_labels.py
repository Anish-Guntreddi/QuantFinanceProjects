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

    Design:
    - Bull/low case: constant positive returns → fwd_return > 0, fwd_rv = 0.
      The expanding median of fwd_rv stays 0 too, but high_vol_flag = (rv > median)
      which is (0 > 0) = False → low_vol. label = 2*1 + 0 = 2.
    - Bear/high case: first half has near-zero vol (small constant returns), second
      half has large-amplitude alternating returns (high vol, negative net fwd_return).
      For bars in the first half, fwd_rv is small → expanding_med is small → high_vol_flag=0.
      For bars in the second half, fwd_rv >> expanding_med built from first half → high_vol_flag=1,
      and fwd_return < 0 → bear. Label = 1.
    """
    from defiregimenet.labels import make_regime_labels

    horizon = 5

    # ---------------------------------------------------------------
    # BULL / LOW-VOL: constant positive returns, rv=0 throughout
    # bull_flag=1 (fwd_return=0.05*horizon>0), high_vol_flag=0 (rv=0=median)
    # label = 1*2 + 0 = 2
    # ---------------------------------------------------------------
    n = 30
    returns_bull_low = pd.Series([0.05] * n)
    rv_bull_low = pd.Series([0.0] * n)  # explicit zero vol

    labels_bl = make_regime_labels(returns_bull_low, rv_bull_low, horizon=horizon)
    valid_bl = labels_bl.dropna()
    assert len(valid_bl) > 0, "Expected some valid labels for bull/low case"
    assert (valid_bl == 2).all(), (
        f"Expected all bull/low labels=2, got unique={valid_bl.unique()}"
    )

    # ---------------------------------------------------------------
    # BEAR / HIGH-VOL
    # First 15 bars: small constant returns (+0.001) → low rv (≈0)
    # Last 15 bars: large alternating returns (-0.20, +0.10) → high rv AND
    #               fwd_return < 0 (net -0.10 per window of 5: 3*-0.20 + 2*0.10)
    # For bars in the first region, their forward window is also first-region
    # → low vol, positive (or ~0) → we only assert on bars whose forward window
    # falls entirely in the high-vol second region.
    # We check bar indices [0..9] whose forward window = bars[1..5] to bars[10..14]:
    # actually to get pure high-vol forward we look at bars whose forward falls in [15..29].
    # bar t's forward window = [t+1..t+horizon], so bars t<=9 (t+5<=14, in low-vol zone).
    # bars t=10..14: forward = [11..15]..[15..19] — partially in high-vol zone.
    # bars t=9: forward = [10..14] — still low-vol.
    # bars t=10..24: forward hits high-vol zone; once fully in high-vol zone (t>=15-horizon=10),
    # fwd_rv >> expanding_med_from_first_half.
    # Simplest: use bar index 10 explicitly (forward = [11..15]).
    # ---------------------------------------------------------------
    low_rv_bars = [0.001] * 15
    high_rv_bars = [-0.20, 0.10] * 8  # alternating: fwd of 5 = 3*(-0.20)+2*(0.10) = -0.40
    returns_mixed = pd.Series(low_rv_bars + high_rv_bars[:15])  # total 30

    # realized_vol: rolling std of returns (causal)
    rv_mixed = returns_mixed.abs().rolling(5, min_periods=1).std()

    labels_mixed = make_regime_labels(returns_mixed, rv_mixed, horizon=horizon)

    # For bar t=10..24, forward window = [11..15]..[25..29] — all in high-vol region
    # Check a bar deep in the high-vol forward zone
    # Bar 10 forward = [11..15]: returns = [0.001, -0.20, 0.10, -0.20, 0.10]
    # fwd_return = 0.001 -0.20 +0.10 -0.20 +0.10 = -0.199 < 0 → bear
    # fwd_rv = mean of rv[11..15] which includes high-vol alternating bars → high
    # At bar 10, expanding_med(fwd_rv) = median of fwd_rv[0..10], mostly low values → small
    # So high_vol_flag = 1, bull_flag = 0 → label = 0*2 + 1 = 1 (bear/high)
    bar10_label = labels_mixed.iloc[10]
    assert bar10_label == 1.0, (
        f"Expected bear/high label=1 at bar 10, got {bar10_label}"
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
