"""Purged cross-validation tests — implemented in plan 02-04."""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers for building synthetic MultiIndex panels inline
# ---------------------------------------------------------------------------

def _make_panel(n_months: int, n_assets: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Build a simple (date, symbol) MultiIndex panel for testing.

    Each month has exactly n_assets rows.  Returns (X, y) where X has two
    feature columns and y is the cross-sectional label column.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    symbols = [f"S{i:03d}" for i in range(n_assets)]

    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    X = pd.DataFrame(
        rng.standard_normal((len(idx), 2)), index=idx, columns=["f1", "f2"]
    )
    y = pd.Series(rng.standard_normal(len(idx)), index=idx, name="label")
    return X, y


def _make_variable_panel(n_months: int, n_assets_early: int, n_assets_late: int,
                          cutoff: int, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Build a panel where the last (n_months - cutoff) months have fewer symbols.

    Simulates delistings: the last block has n_assets_late < n_assets_early assets.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")

    rows_X: list[pd.DataFrame] = []
    rows_y: list[pd.Series] = []
    for i, dt in enumerate(dates):
        n = n_assets_early if i < cutoff else n_assets_late
        syms = [f"S{j:03d}" for j in range(n)]
        idx = pd.MultiIndex.from_arrays([[dt] * n, syms], names=["date", "symbol"])
        rows_X.append(pd.DataFrame(rng.standard_normal((n, 2)), index=idx, columns=["f1", "f2"]))
        rows_y.append(pd.Series(rng.standard_normal(n), index=idx, name="label"))

    X = pd.concat(rows_X)
    y = pd.concat(rows_y)
    return X, y


# ---------------------------------------------------------------------------
# Task 1 — behaviour tests
# ---------------------------------------------------------------------------

class TestSplitCount:
    """test_split_count: C(6,2)=15 splits over 60 months."""

    def test_split_count(self):
        from alpharank.validation import PurgedCVEvaluator

        months = pd.date_range("2015-01-31", periods=60, freq="ME")
        ev = PurgedCVEvaluator()
        splits = list(ev.split_months(months))
        assert len(splits) == 15, f"Expected 15 splits, got {len(splits)}"


class TestDummyModel:
    """test_evaluator_with_dummy_model: perfect model IC=1, random model |IC|<0.2."""

    def test_perfect_model_ic_equals_one(self):
        from alpharank.validation import PurgedCVEvaluator

        class PerfectModel:
            def fit(self, X, y):
                return self
            def predict(self, X):
                # Returns the stored true values (set during evaluate)
                return self._true_y

        # We can't inject true y into PerfectModel naively — use a model
        # that always returns y sorted rank (cross-sectional perfect order).
        # Instead, use the real y values via a closure model.
        class CheatingModel:
            """Returns the true y values verbatim — IC must be 1.0."""
            def fit(self, X, y):
                # Store a mapping of (date, symbol) -> y value for test-time lookup
                self._y_train = y.copy()
                return self
            def predict(self, X):
                # For each row in X, we need the true y value.
                # This only works when train == test (not useful) OR we pre-store
                # the full series. Use global reference hack via numpy seed.
                # Better: use a global y lookup.
                return np.ones(len(X))  # NOT IC=1 by itself

        # Real approach: model that returns y directly
        class OracleLookupModel:
            """At predict time, returns the true y value for each row.

            Works by storing the full true_y series in fit (the evaluator will
            call fit on train rows, but predict is called on test rows whose y
            values we still know from the global series).
            """
            def __init__(self, full_y: pd.Series):
                self._full_y = full_y

            def fit(self, X, y):
                return self  # nothing to learn

            def predict(self, X):
                return self._full_y.loc[X.index].values

        X, y = _make_panel(n_months=60, n_assets=10, seed=0)
        ev = PurgedCVEvaluator()
        oracle = OracleLookupModel(full_y=y)
        result = ev.evaluate(oracle, X, y)

        ic_series = result["ic_series"]
        assert isinstance(ic_series, pd.Series), "ic_series must be a pd.Series"
        assert len(ic_series) > 0, "ic_series must be non-empty"
        # Perfect predictions → IC = 1.0 in every OOS month
        assert (ic_series >= 0.99).all(), (
            f"Expected all ICs >= 0.99 for oracle model, got min={ic_series.min():.4f}"
        )

    def test_random_model_ic_small(self):
        """Random predictions should have mean |IC| well below 0.2.

        Auto-fix [Rule 1 - Bug]: With n=10 assets, E[|Spearman IC|] ≈ 0.27
        due to high variance in small cross-sections.  Using n=30 assets gives
        E[|IC|] ≈ 0.15 which reliably satisfies the < 0.2 threshold.
        """
        from alpharank.validation import PurgedCVEvaluator

        class RandomModel:
            def __init__(self, seed=99):
                self._rng = np.random.default_rng(seed)
            def fit(self, X, y):
                return self
            def predict(self, X):
                return self._rng.standard_normal(len(X))

        # n_assets=30 so that E[|Spearman IC|] ≈ 0.15 for truly random predictions
        X, y = _make_panel(n_months=60, n_assets=30, seed=0)
        ev = PurgedCVEvaluator()
        model = RandomModel(seed=77)
        result = ev.evaluate(model, X, y)
        ic_series = result["ic_series"]
        assert ic_series.abs().mean() < 0.2, (
            f"Expected mean |IC| < 0.2 for random model, got {ic_series.abs().mean():.4f}"
        )


class TestVariableUniverse:
    """test_variable_universe: variable universe (delistings) evaluates without error."""

    def test_variable_universe(self):
        from alpharank.validation import PurgedCVEvaluator

        class SimpleLinearModel:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return X.iloc[:, 0].values

        X, y = _make_variable_panel(
            n_months=60, n_assets_early=15, n_assets_late=8, cutoff=50
        )
        ev = PurgedCVEvaluator()
        result = ev.evaluate(SimpleLinearModel(), X, y)

        oos_scores = result["oos_scores"]
        # oos_scores must be aligned to X.index exactly
        assert oos_scores.index.equals(X.index), (
            "oos_scores.index must match X.index exactly"
        )


class TestReturnInterface:
    """Additional contract checks on evaluate() return dict."""

    def test_evaluate_returns_required_keys(self):
        from alpharank.validation import PurgedCVEvaluator

        class DummyModel:
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))

        X, y = _make_panel(n_months=60, n_assets=5, seed=42)
        ev = PurgedCVEvaluator()
        result = ev.evaluate(DummyModel(), X, y)
        for key in ("ic_series", "oos_scores", "n_splits"):
            assert key in result, f"Missing key '{key}' in evaluate() return"
        assert result["n_splits"] == 15


# ---------------------------------------------------------------------------
# Task 2 — property tests and KFold guard
# ---------------------------------------------------------------------------

class TestPurgedCVNoTrainTestOverlap:
    """test_purged_cv_no_train_test_overlap: all 15 splits have disjoint train/test."""

    def test_purged_cv_no_train_test_overlap(self):
        from alpharank.validation import PurgedCVEvaluator

        months = pd.date_range("2015-01-31", periods=60, freq="ME")
        ev = PurgedCVEvaluator()
        splits = list(ev.split_months(months))
        assert len(splits) == 15

        for i, (train_pos, test_pos) in enumerate(splits):
            train_set = set(train_pos.tolist())
            test_set = set(test_pos.tolist())
            overlap = train_set & test_set
            assert len(overlap) == 0, (
                f"Split {i} has train/test overlap at month positions: {overlap}"
            )

        # Also verify at the expanded panel-row level
        X, y = _make_panel(n_months=60, n_assets=10, seed=7)
        dates = X.index.get_level_values("date").unique().sort_values()
        all_dates = X.index.get_level_values("date")
        month_to_rows = {
            m: np.flatnonzero(all_dates == m) for m in dates
        }
        for i, (train_pos, test_pos) in enumerate(splits):
            train_rows = set(
                int(r) for p in train_pos for r in month_to_rows[dates[p]]
            )
            test_rows = set(
                int(r) for p in test_pos for r in month_to_rows[dates[p]]
            )
            overlap = train_rows & test_rows
            assert len(overlap) == 0, (
                f"Split {i} panel-row overlap: {list(overlap)[:5]}"
            )


class TestPurgeGap:
    """test_purge_gap: no t-1 (purge) or t+1 (embargo) neighbour of test month in train."""

    def test_purge_gap(self):
        """Verify purged_size=1 and embargo_size=1 boundary conditions.

        For each split and each test month position t, verify that:
        - t-1 is NOT in train (purge boundary)
        - t+1 is NOT in train (embargo boundary)
        when those indices are within the valid range [0, 60).

        CPCV boundary behaviour (first split layout):
        Split 0 test=[0..19] train=[22..59] — positions 20,21 are purged/embargoed gap.
        Interior test blocks (not at array boundaries) have both neighbours removed.
        """
        from alpharank.validation import PurgedCVEvaluator

        months = pd.date_range("2015-01-31", periods=60, freq="ME")
        ev = PurgedCVEvaluator()
        splits = list(ev.split_months(months))
        n = len(months)

        purge_violations = []
        embargo_violations = []
        for i, (train_pos, test_pos) in enumerate(splits):
            train_set = set(train_pos.tolist())
            for t in test_pos.tolist():
                if t - 1 >= 0 and (t - 1) in train_set:
                    purge_violations.append((i, t, t - 1))
                if t + 1 < n and (t + 1) in train_set:
                    embargo_violations.append((i, t, t + 1))

        assert len(purge_violations) == 0, (
            f"Purge violations (test month t, t-1 in train): {purge_violations[:5]}"
        )
        assert len(embargo_violations) == 0, (
            f"Embargo violations (test month t, t+1 in train): {embargo_violations[:5]}"
        )


class TestNoStandardKFoldAnywhere:
    """test_no_standard_kfold_anywhere: KFold must not appear in src/alpharank."""

    def test_no_standard_kfold_anywhere(self):
        """Walk src/alpharank/**/*.py and verify no 'KFold' string appears.

        Also checks run_pipeline.py at the alpharank project root if it exists.
        No sys.path manipulation needed — pure filesystem check via pathlib.
        """
        from pathlib import Path

        src_root = Path(__file__).parent.parent / "src" / "alpharank"
        assert src_root.is_dir(), f"src/alpharank not found at {src_root}"

        violations: list[str] = []
        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "KFold" in text:
                # Find lines for better diagnostics
                lines = [
                    f"  {py_file.name}:{lineno+1}: {line.rstrip()}"
                    for lineno, line in enumerate(text.splitlines())
                    if "KFold" in line
                ]
                violations.extend(lines)

        # Also check run_pipeline.py if it exists
        pipeline_file = Path(__file__).parent.parent / "run_pipeline.py"
        if pipeline_file.exists():
            text = pipeline_file.read_text(encoding="utf-8")
            if "KFold" in text:
                lines = [
                    f"  run_pipeline.py:{lineno+1}: {line.rstrip()}"
                    for lineno, line in enumerate(text.splitlines())
                    if "KFold" in line
                ]
                violations.extend(lines)

        assert len(violations) == 0, (
            "Found 'KFold' in src/alpharank — only CombinatorialPurgedCV is allowed:\n"
            + "\n".join(violations)
        )
