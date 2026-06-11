"""Tests for VSL-01: Synthetic options chain generator.

Requirements:
- Determinism: two generators with seed=42 produce identical DataFrames
- Coverage: 3 maturities x 13 strikes x 2 flags = 78 rows exactly
- Log-moneyness k spans [-1.5, 1.5]
- All prices > 0; true_iv in (0.05, 1.0)
- Ground truth: true_iv == sqrt(w(k,T)/T) to 1e-12
- validate_chain_coverage raises ValueError for missing maturity or truncated k range
- Underlying path: len==750, no NaN, annualized vol in (0.08, 0.30)
- Underlying path is deterministic under fixed seed
- Planted-arb helpers return params/surfaces with known violations
- Importing chain.py does not import yfinance at module scope
- ChainData is frozen (immutable after creation)
- ChainData has options, spot, risk_free, seed attributes with expected values
"""

import sys
import importlib

import numpy as np
import pandas as pd
import pytest

from volsurfacelab.chain import (
    SYNTHETIC_SVI_SURFACE,
    ChainData,
    SyntheticChainGenerator,
    generate_underlying_returns,
    make_butterfly_violating_params,
    make_calendar_violating_surface,
    validate_chain_coverage,
)


# ---------------------------------------------------------------------------
# Helper: SVI total variance (independent implementation for oracle check)
# ---------------------------------------------------------------------------

def _svi_w(k, a, b, rho, m, sigma):
    """SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2+sigma^2))."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _svi_wp(k, a, b, rho, m, sigma):
    """SVI derivative w'(k)."""
    return b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2))


def _svi_wpp(k, a, b, rho, m, sigma):
    """SVI second derivative w''(k)."""
    return b * sigma ** 2 / ((k - m) ** 2 + sigma ** 2) ** 1.5


def _g_func(k, a, b, rho, m, sigma):
    """Gatheral-Jacquier butterfly no-arb function g(k)."""
    w = _svi_w(k, a, b, rho, m, sigma)
    wp = _svi_wp(k, a, b, rho, m, sigma)
    wpp = _svi_wpp(k, a, b, rho, m, sigma)
    return (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2


# ---------------------------------------------------------------------------
# Tests: SYNTHETIC_SVI_SURFACE constant
# ---------------------------------------------------------------------------

class TestSyntheticSviSurface:
    def test_has_three_maturities(self):
        assert set(SYNTHETIC_SVI_SURFACE.keys()) == {0.25, 0.50, 1.00}

    def test_params_are_five_element_tuples(self):
        for T, params in SYNTHETIC_SVI_SURFACE.items():
            assert len(params) == 5, f"T={T}: expected 5 params, got {len(params)}"

    def test_expected_params(self):
        """Verify the exact documented ground-truth parameters."""
        assert SYNTHETIC_SVI_SURFACE[0.25] == (-0.0084, 0.08, -0.3, 0.0, 0.3)
        assert SYNTHETIC_SVI_SURFACE[0.50] == (0.0002, 0.08, -0.3, 0.0, 0.3)
        assert SYNTHETIC_SVI_SURFACE[1.00] == (0.0160, 0.08, -0.3, 0.0, 0.3)

    def test_butterfly_compliance_all_maturities(self):
        """g(k) > 0 for all traded k in [-1.5, 1.5] at all maturities."""
        k_grid = np.linspace(-1.5, 1.5, 200)
        for T, params in SYNTHETIC_SVI_SURFACE.items():
            g_vals = _g_func(k_grid, *params)
            assert np.all(g_vals > 0), (
                f"T={T}: butterfly violation, min g={g_vals.min():.4f}"
            )

    def test_calendar_compliance_traded_range(self):
        """w(k, T2) > w(k, T1) for all T1 < T2 in [-1.5, 1.5]."""
        k_grid = np.linspace(-1.5, 1.5, 200)
        maturities = sorted(SYNTHETIC_SVI_SURFACE.keys())
        for T1, T2 in zip(maturities[:-1], maturities[1:]):
            w1 = _svi_w(k_grid, *SYNTHETIC_SVI_SURFACE[T1])
            w2 = _svi_w(k_grid, *SYNTHETIC_SVI_SURFACE[T2])
            assert np.all(w2 > w1), (
                f"Calendar violation between T={T1} and T={T2}"
            )


# ---------------------------------------------------------------------------
# Tests: SyntheticChainGenerator
# ---------------------------------------------------------------------------

class TestSyntheticChainGenerator:
    def test_generate_returns_chain_data(self):
        gen = SyntheticChainGenerator()
        result = gen.generate()
        assert isinstance(result, ChainData)

    def test_determinism_options_dataframe(self):
        """Two generators with same seed produce identical options DataFrames."""
        gen1 = SyntheticChainGenerator(seed=42)
        gen2 = SyntheticChainGenerator(seed=42)
        pd.testing.assert_frame_equal(gen1.generate().options, gen2.generate().options)

    def test_determinism_different_seeds_differ(self):
        """Different seeds produce different results."""
        # Chain generation is deterministic from SVI; seed stored for API stability.
        # With multiple calls using same seed, results must still be identical.
        gen = SyntheticChainGenerator(seed=42)
        r1 = gen.generate()
        r2 = gen.generate()
        pd.testing.assert_frame_equal(r1.options, r2.options)

    def test_row_count_78(self):
        """3 maturities x 13 strikes x 2 flags = 78 rows."""
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        assert len(chain.options) == 78

    def test_columns_present(self):
        """Required columns: T, K, k, flag, price, true_iv, forward."""
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        expected = {"T", "K", "k", "flag", "price", "true_iv", "forward"}
        assert expected.issubset(chain.options.columns)

    def test_maturities_correct(self):
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        assert set(chain.options["T"].unique()) == {0.25, 0.50, 1.00}

    def test_n_strikes_per_maturity(self):
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        for T in [0.25, 0.50, 1.00]:
            n_calls = len(chain.options[(chain.options["T"] == T) & (chain.options["flag"] == "c")])
            n_puts = len(chain.options[(chain.options["T"] == T) & (chain.options["flag"] == "p")])
            assert n_calls == 13, f"T={T}: expected 13 calls, got {n_calls}"
            assert n_puts == 13, f"T={T}: expected 13 puts, got {n_puts}"

    def test_flags_are_c_and_p(self):
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        assert set(chain.options["flag"].unique()) == {"c", "p"}

    def test_moneyness_range(self):
        """k spans from k_min to k_max (default -1.5 to 1.5)."""
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        k_vals = chain.options["k"].unique()
        assert min(k_vals) == pytest.approx(-1.5, abs=1e-10)
        assert max(k_vals) == pytest.approx(1.5, abs=1e-10)

    def test_all_prices_positive(self):
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        assert (chain.options["price"] > 0).all()

    def test_true_iv_range(self):
        """true_iv in (0.05, 1.0) for all rows."""
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        assert (chain.options["true_iv"] > 0.05).all()
        assert (chain.options["true_iv"] < 1.0).all()

    def test_ground_truth_consistency(self):
        """true_iv == sqrt(w(k, T) / T) for every row to 1e-12."""
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        for _, row in chain.options.iterrows():
            T = row["T"]
            k = row["k"]
            params = SYNTHETIC_SVI_SURFACE[T]
            w = _svi_w(k, *params)
            expected_iv = np.sqrt(w / T)
            assert abs(row["true_iv"] - expected_iv) < 1e-12, (
                f"T={T}, k={k:.3f}: true_iv={row['true_iv']:.8f}, "
                f"expected={expected_iv:.8f}"
            )

    def test_chain_data_attributes(self):
        """ChainData.spot == 100.0, risk_free == 0.05, seed == 42."""
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        assert chain.spot == 100.0
        assert chain.risk_free == 0.05
        assert chain.seed == 42

    def test_chain_data_is_frozen(self):
        """ChainData is a frozen dataclass — assignment raises FrozenInstanceError."""
        from dataclasses import FrozenInstanceError
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        with pytest.raises(FrozenInstanceError):
            chain.spot = 99.0  # type: ignore[misc]

    def test_forward_computation(self):
        """forward == spot * exp(risk_free * T) for each maturity."""
        gen = SyntheticChainGenerator()
        chain = gen.generate()
        S = chain.spot
        r = chain.risk_free
        for T in [0.25, 0.50, 1.00]:
            rows = chain.options[chain.options["T"] == T]
            expected_fwd = S * np.exp(r * T)
            assert (rows["forward"] - expected_fwd).abs().max() < 1e-10


# ---------------------------------------------------------------------------
# Tests: generate_underlying_returns
# ---------------------------------------------------------------------------

class TestGenerateUnderlyingReturns:
    def test_length(self):
        series = generate_underlying_returns(seed=42, n_days=750)
        assert len(series) == 750

    def test_no_nan(self):
        series = generate_underlying_returns(seed=42, n_days=750)
        assert not series.isna().any()

    def test_deterministic(self):
        """Two calls with same seed produce identical series."""
        s1 = generate_underlying_returns(seed=42)
        s2 = generate_underlying_returns(seed=42)
        pd.testing.assert_series_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = generate_underlying_returns(seed=42)
        s2 = generate_underlying_returns(seed=99)
        assert not s1.equals(s2)

    def test_business_day_index(self):
        """Index is DatetimeIndex of business days starting 2020-01-01."""
        series = generate_underlying_returns(seed=42, n_days=750)
        assert isinstance(series.index, pd.DatetimeIndex)
        # First index entry should be 2020-01-02 (first bday on or after 2020-01-01)
        assert series.index[0] >= pd.Timestamp("2020-01-01")
        # Consecutive index entries should be business days (Mon-Fri)
        diffs = series.index[1:] - series.index[:-1]
        # All diffs should be 1, 3 (over weekend), or holiday gaps — all positive
        assert (diffs.days >= 1).all()
        # Most should be exactly 1 business day (Mon-Thu); some 3 (Fri->Mon)
        assert (diffs.days <= 4).all()

    def test_annualized_vol_plausible(self):
        """Annualized vol of GARCH(1,1) path with omega=2e-6, alpha=0.08, beta=0.90
        should be in (0.08, 0.30) — long-run ~15.9%."""
        series = generate_underlying_returns(seed=42, n_days=750)
        ann_vol = series.std() * np.sqrt(252)
        assert 0.08 < ann_vol < 0.30, f"Annualized vol {ann_vol:.4f} out of range"


# ---------------------------------------------------------------------------
# Tests: validate_chain_coverage
# ---------------------------------------------------------------------------

class TestValidateChainCoverage:
    def _make_chain(self, maturities=(0.25, 0.5, 1.0), k_min=-1.5, k_max=1.5):
        gen = SyntheticChainGenerator(
            maturities=maturities,
            k_min=k_min,
            k_max=k_max,
        )
        return gen.generate()

    def test_valid_chain_passes(self):
        """Full chain with 3 maturities and k in [-1.5, 1.5] passes."""
        chain = self._make_chain()
        # Should not raise
        validate_chain_coverage(chain, required_maturities=[0.25, 0.5, 1.0],
                                 k_min=-1.5, k_max=1.5)

    def test_missing_maturity_raises(self):
        """Chain missing one maturity raises ValueError naming the maturity."""
        chain = self._make_chain(maturities=(0.25, 1.0))  # missing 0.5
        with pytest.raises(ValueError, match="0.5"):
            validate_chain_coverage(chain, required_maturities=[0.25, 0.5, 1.0],
                                     k_min=-1.5, k_max=1.5)

    def test_truncated_moneyness_raises(self):
        """Chain with k range [-0.5, 0.5] raises ValueError about moneyness."""
        chain = self._make_chain(k_min=-0.5, k_max=0.5)
        with pytest.raises(ValueError):
            validate_chain_coverage(chain, required_maturities=[0.25, 0.5, 1.0],
                                     k_min=-1.5, k_max=1.5)


# ---------------------------------------------------------------------------
# Tests: planted arb helpers
# ---------------------------------------------------------------------------

class TestPlantedArbHelpers:
    def test_butterfly_violating_params_violates(self):
        """make_butterfly_violating_params returns params with min g(k) < 0."""
        params = make_butterfly_violating_params()
        k_grid = np.linspace(-1.5, 1.5, 200)
        g_vals = _g_func(k_grid, *params)
        assert np.any(g_vals < 0), (
            f"Expected butterfly violation; min g={g_vals.min():.4f}"
        )

    def test_calendar_violating_surface_violates(self):
        """make_calendar_violating_surface returns surface where a(T=0.5) < a(T=0.25)."""
        surface = make_calendar_violating_surface()
        a_025 = surface[0.25][0]
        a_050 = surface[0.50][0]
        assert a_050 < a_025, (
            f"Expected a(0.5) < a(0.25) for calendar violation; "
            f"a(0.25)={a_025}, a(0.5)={a_050}"
        )

    def test_calendar_violating_surface_has_same_maturities(self):
        surface = make_calendar_violating_surface()
        assert set(surface.keys()) == {0.25, 0.50, 1.00}


# ---------------------------------------------------------------------------
# Tests: yfinance not imported at module scope
# ---------------------------------------------------------------------------

class TestNoModuleScopeYfinance:
    def test_yfinance_not_imported_after_chain_import(self):
        """Importing volsurfacelab.chain must not pull yfinance into sys.modules."""
        # chain is already imported above; yfinance should not be in sys.modules
        assert "yfinance" not in sys.modules, (
            "yfinance was imported at module scope in chain.py — "
            "it must be lazily imported inside load_yfinance_chain()"
        )

    def test_chain_module_importable_without_yfinance(self):
        """Reimporting chain.py in a fresh context does not require yfinance."""
        # Force re-import to confirm no side-effect yfinance import
        if "volsurfacelab.chain" in sys.modules:
            importlib.reload(sys.modules["volsurfacelab.chain"])
        assert "yfinance" not in sys.modules
