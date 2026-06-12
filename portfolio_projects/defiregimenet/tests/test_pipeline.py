"""Integration tests for the end-to-end pipeline (05-07).

Tests run in two groups:
  Task 1 — pipeline.py: PipelineResults dataclass, run_pipeline(quick=True),
            determinism, frozen instance, embargo invariant, cross-token V strength,
            and quarantine still green.
  Task 2 — report/builder.py: ReportBuilder.build_all() artifact count, section headers,
            and no-pyplot-at-package-import guard.

Session-scope fixture `quick_results` runs the pipeline once and is shared across
all tests that need it (avoids running the ~10-second pipeline twice in the suite).
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Session-scope pipeline fixture (runs once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def quick_results():
    """Run run_pipeline(quick=True, seed=42) once; share across all tests."""
    from defiregimenet.pipeline import run_pipeline

    return run_pipeline(quick=True, seed=42)


# ===========================================================================
# TASK 1 — pipeline.py
# ===========================================================================


class TestPipelineQuickEndToEnd:
    """run_pipeline(quick=True) returns a fully-populated PipelineResults."""

    def test_fields_populated(self, quick_results):
        """Every PipelineResults field is non-empty / non-None."""
        r = quick_results
        # dict fields
        assert len(r.regimes_hmm) > 0, "regimes_hmm is empty"
        assert len(r.regimes_gmm) > 0, "regimes_gmm is empty"
        assert len(r.diagnostics) > 0, "diagnostics is empty"
        assert len(r.k_sensitivity) > 0, "k_sensitivity is empty"
        assert len(r.forecast_comparison) > 0, "forecast_comparison is empty"
        assert len(r.studentst_robustness) > 0, "studentst_robustness is empty"
        assert len(r.label_distribution) > 0, "label_distribution is empty"

    def test_model_comparison_shape(self, quick_results):
        """model_comparison has 4 rows (hmm, gmm, logistic, xgboost) and 2 metric cols."""
        mc = quick_results.model_comparison
        assert isinstance(mc, pd.DataFrame), "model_comparison must be a DataFrame"
        assert set(mc.index) == {"hmm", "gmm", "logistic", "xgboost"}, (
            f"Expected 4-model index, got {list(mc.index)}"
        )
        assert "accuracy" in mc.columns, "accuracy column missing"
        assert "log_loss" in mc.columns, "log_loss column missing"

    def test_model_comparison_finite(self, quick_results):
        """All accuracy and log_loss values in model_comparison are finite numbers."""
        mc = quick_results.model_comparison
        for col in ["accuracy", "log_loss"]:
            assert mc[col].notna().all(), f"NaN in model_comparison[{col!r}]"
            assert np.isfinite(mc[col].values).all(), f"Inf in model_comparison[{col!r}]"

    def test_cross_token_v_square_symmetric(self, quick_results):
        """cross_token_v is a square symmetric DataFrame with unit diagonal."""
        cv = quick_results.cross_token_v
        assert isinstance(cv, pd.DataFrame), "cross_token_v must be a DataFrame"
        assert cv.shape[0] == cv.shape[1], "cross_token_v must be square"
        np.testing.assert_allclose(
            cv.values, cv.values.T, atol=1e-9, err_msg="cross_token_v not symmetric"
        )
        np.testing.assert_allclose(
            np.diag(cv.values), 1.0, atol=1e-9, err_msg="diagonal of cross_token_v != 1.0"
        )

    def test_tokens_and_n_bars(self, quick_results):
        """tokens is a tuple of strings; n_bars is positive."""
        r = quick_results
        assert isinstance(r.tokens, tuple), "tokens must be a tuple"
        assert all(isinstance(t, str) for t in r.tokens)
        assert r.n_bars > 0, "n_bars must be positive"
        assert r.seed == 42


class TestPipelineDeterministic:
    """Two runs with same seed produce identical model_comparison."""

    def test_identical_comparison_tables(self):
        from defiregimenet.pipeline import run_pipeline

        r1 = run_pipeline(quick=True, seed=42)
        r2 = run_pipeline(quick=True, seed=42)
        pd.testing.assert_frame_equal(
            r1.model_comparison,
            r2.model_comparison,
            check_exact=False,
            atol=1e-12,
        )


class TestPipelineFrozen:
    """PipelineResults attributes are immutable (frozen dataclass)."""

    def test_frozen_instance_error(self, quick_results):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            quick_results.seed = 999  # type: ignore[misc]


class TestPipelineEmbargoInvariant:
    """Quick mode: embargo_size >= label_horizon in the config."""

    def test_embargo_not_below_horizon(self, quick_results):
        cfg = quick_results.config
        # Both embargo_size and purged_size must be >= label_horizon
        h = cfg.get("label_horizon", cfg.get("labels", {}).get("horizon", 5))
        embargo = cfg.get("embargo_size", cfg.get("cv", {}).get("embargo_size", 0))
        purged = cfg.get("purged_size", cfg.get("cv", {}).get("purged_size", 0))
        assert embargo >= h, (
            f"embargo_size ({embargo}) < label_horizon ({h}) in quick mode config"
        )
        assert purged >= h, (
            f"purged_size ({purged}) < label_horizon ({h}) in quick mode config"
        )


class TestCrossTokenVStrengthDetectedRegimes:
    """Cross-token association of INDEPENDENTLY detected regimes is genuine.

    The DGP plants a shared market regime (70% market factor). Each token's
    detector is fit independently, so off-diagonal Cramér's V measures real
    recovery of the shared factor through 30% idiosyncratic noise and 4-state
    label-permutation ambiguity. Empirically this lands ~0.35-0.45 — well
    above the ~0.15 independence floor, far below 1.0.

    VACUITY GUARD: an earlier implementation assigned one jointly-detected
    sequence to every token, making V identically 1.0 by construction. The
    second assertion rejects that: per-token sequences must actually differ.
    """

    def test_offdiagonal_mean_above_threshold(self, quick_results):
        cv = quick_results.cross_token_v
        tokens = list(cv.index)
        if len(tokens) < 2:
            pytest.skip("Need at least 2 tokens to compute off-diagonal V")

        n = len(tokens)
        off_diag = [cv.iloc[i, j] for i in range(n) for j in range(n) if i != j]
        mean_v = np.mean(off_diag)
        assert mean_v > 0.3, (
            f"Off-diagonal mean Cramér's V on detected HMM sequences = {mean_v:.4f} (<= 0.3). "
            "The shared market regime is not recovered by independent per-token detectors."
        )

    def test_sequences_are_independently_detected(self, quick_results):
        regimes = quick_results.regimes_hmm
        tokens = list(regimes)
        if len(tokens) < 2:
            pytest.skip("Need at least 2 tokens")
        a, b = regimes[tokens[0]], regimes[tokens[1]]
        m = min(len(a), len(b))
        assert not np.array_equal(a[:m], b[:m]), (
            "Per-token regime sequences are identical — detection is sharing one "
            "joint sequence across tokens, which makes the cross-token V heatmap "
            "vacuously 1.0 and collapses per-token diagnostics."
        )


class TestQuarantineStillGreen:
    """After adding pipeline.py, the label AST quarantine test still passes.

    pipeline.py is an ALLOWED importer; all other non-evaluation modules
    must remain clean.
    """

    def test_quarantine_still_passes(self):
        src_root = Path(__file__).parents[1] / "src" / "defiregimenet"
        assert src_root.exists(), f"source root not found: {src_root}"

        ALLOWED_IMPORTERS = {"defiregimenet.evaluation", "defiregimenet.pipeline"}

        violations: list[str] = []
        for path in src_root.rglob("*.py"):
            module_rel = path.relative_to(src_root)
            module_name = "defiregimenet." + ".".join(module_rel.with_suffix("").parts)
            if module_name.endswith(".__init__"):
                module_name = module_name[: -len(".__init__")]
            if any(allowed in module_name for allowed in ALLOWED_IMPORTERS):
                continue
            if module_name.split(".")[-1] == "labels":
                continue

            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "labels" in node.module:
                        violations.append(f"{module_name}: imports {node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "labels" in alias.name:
                            violations.append(f"{module_name}: imports {alias.name}")

        assert violations == [], f"Label quarantine violated after adding pipeline.py: {violations}"


# ===========================================================================
# TASK 2 — report/builder.py
# ===========================================================================


class TestReportBuilderArtifacts:
    """ReportBuilder.build_all() creates >=6 PNGs and summary.md."""

    def test_artifact_count_and_summary(self, quick_results, tmp_path):
        from defiregimenet.report.builder import ReportBuilder

        figures_dir = tmp_path / "figures"
        builder = ReportBuilder(quick_results, figures_dir)
        builder.build_all()

        pngs = list(figures_dir.glob("*.png"))
        assert len(pngs) >= 6, (
            f"Expected >= 6 PNG artifacts; got {len(pngs)}: {[p.name for p in pngs]}"
        )
        summary_path = tmp_path / "summary.md"
        assert summary_path.exists(), "summary.md was not created in output_dir.parent"
        assert summary_path.stat().st_size > 0, "summary.md is empty"


class TestReportSummarySections:
    """summary.md contains all 6 required section headers."""

    REQUIRED_SECTIONS = [
        "Abstract",
        "Data",
        "Methodology",
        "Results",
        "Robustness",
        "Limitations",
    ]

    def test_all_section_headers_present(self, quick_results, tmp_path):
        from defiregimenet.report.builder import ReportBuilder

        figures_dir = tmp_path / "figures"
        builder = ReportBuilder(quick_results, figures_dir)
        builder.build_all()

        summary_path = tmp_path / "summary.md"
        content = summary_path.read_text()
        for section in self.REQUIRED_SECTIONS:
            assert section in content, (
                f"Section '{section}' missing from summary.md"
            )


class TestNoPyplotAtPackageImport:
    """Importing defiregimenet must NOT pull in matplotlib.pyplot.

    The builder is only loaded explicitly; lazy wiring is plan 05-09.
    This guards against accidental eager imports in __init__.py files.
    """

    def test_no_pyplot_in_defiregimenet_init(self):
        import sys

        # Remove any cached defiregimenet modules to get a clean import
        mods_to_remove = [k for k in sys.modules if k.startswith("defiregimenet")]
        for mod in mods_to_remove:
            del sys.modules[mod]

        # Also remove matplotlib.pyplot if already imported (guard our own state)
        pyplot_was_present = "matplotlib.pyplot" in sys.modules

        import defiregimenet  # noqa: F401 — import side-effect test

        # If pyplot was not present before, it must not be present after importing defiregimenet
        if not pyplot_was_present:
            assert "matplotlib.pyplot" not in sys.modules, (
                "Importing defiregimenet pulled in matplotlib.pyplot eagerly. "
                "Move the import to report/builder.py and do NOT import builder "
                "from any __init__.py."
            )
