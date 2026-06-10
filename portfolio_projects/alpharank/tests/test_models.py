"""Model tests — plan 02-06.

Tests are written in RED (failing) state before any implementation.

TDD structure:
- Task 1 RED/GREEN: Four baseline models + fixed params
- Task 2 RED/GREEN: Comparison harness with identical CV protocol
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build (X, y) from synthetic panel
# ---------------------------------------------------------------------------

def _make_xy(n_assets: int = 50, n_months: int = 60, seed: int = 42):
    """Build (X, y) aligned feature/label pair for integration tests.

    Workflow (as specified in plan 02-06 interfaces note):
        1. CrossSectionalGenerator → SyntheticPanel
        2. build_feature_panel on daily OHLCV data
        3. make_labels on month-end closes
        4. stack labels → MultiIndex Series
        5. inner-join on feature index (left = features)
    """
    from alpharank.data.generator import CrossSectionalGenerator
    from alpharank.features.factors import build_feature_panel
    from alpharank.labels.forward_returns import make_labels

    gen = CrossSectionalGenerator(n_assets=n_assets, n_months=n_months, seed=seed)
    panel = gen.generate()

    # Feature panel — MultiIndex (date, symbol) x 6 factors
    X = build_feature_panel(panel)

    # Build wide daily close from ohlcv dict (same as build_feature_panel does internally)
    symbols = list(panel.ohlcv.keys())
    close_daily = pd.DataFrame(
        {sym: panel.ohlcv[sym]["close"] for sym in symbols}
    ).sort_index()

    # Use generator BME month-end dates for sampling (same freq as build_feature_panel
    # uses for rebalance_dates).  resample("ME") gives calendar month-ends which
    # diverge from BME on months where the last bday < last calendar day — causing a
    # 33/47-date date mismatch.  Sampling at panel.monthly_returns.index directly
    # gives perfect date alignment with X.
    bme_dates = panel.monthly_returns.index
    monthly_close = close_daily.reindex(bme_dates, method="ffill")

    # Labels: percentile ranks of 1-month forward returns
    labels_wide = make_labels(monthly_close, horizon=1)  # date x symbol, NaN at tail

    # Stack to MultiIndex Series (drop NaN tail rows from forward-return horizon)
    y_long = labels_wide.stack(future_stack=True).dropna()
    y_long.index.names = ["date", "symbol"]

    # Inner-join on feature index (left side = features) to avoid label leakage
    # (plan 02-06 interfaces note: "join on the FEATURE index — never the label index")
    X_aligned = X.loc[X.index.isin(y_long.index)]
    y_aligned = y_long.loc[X_aligned.index]

    return X_aligned, y_aligned


# ---------------------------------------------------------------------------
# Task 1 Tests: Four models with fixed parameters in baseline order
# ---------------------------------------------------------------------------


class TestCompositeModel:
    """Tests for EqualWeightComposite."""

    def test_composite_positive_ic(self):
        """EqualWeightComposite evaluated via PurgedCVEvaluator achieves mean OOS IC > 0.

        Uses the full-size synthetic panel (n_assets=50, n_months=60, seed=42)
        with planted alpha recoverable through the pipeline.
        """
        from alpharank.models.composite import EqualWeightComposite
        from alpharank.validation.purged_cv import PurgedCVEvaluator
        from alpharank.analytics.ic import newey_west_ic_tstat

        X, y = _make_xy(n_assets=50, n_months=60, seed=42)
        model = EqualWeightComposite()
        evaluator = PurgedCVEvaluator(n_folds=6, n_test_folds=2, purged_size=1, embargo_size=1)

        result = evaluator.evaluate(model, X, y)
        ic_series = result["ic_series"]

        mean_ic = float(ic_series.mean())
        assert mean_ic > 0.0, (
            f"EqualWeightComposite mean OOS IC={mean_ic:.4f} should be positive "
            f"(planted alpha in synthetic data must be recoverable)"
        )

        # NW t-stat must be computable without errors
        mean_ic_nw, t_stat, p_value = newey_west_ic_tstat(ic_series)
        assert np.isfinite(t_stat), "Newey-West t-stat must be finite"

    def test_composite_needs_no_fit(self):
        """EqualWeightComposite.predict gives identical output before and after fit.

        fit() is documented as a no-op.
        """
        from alpharank.models.composite import EqualWeightComposite

        X, y = _make_xy(n_assets=20, n_months=30, seed=1)

        model = EqualWeightComposite()
        preds_before = model.predict(X)

        model.fit(X, y)
        preds_after = model.predict(X)

        np.testing.assert_array_equal(
            preds_before, preds_after,
            err_msg="EqualWeightComposite.fit() must be a no-op: predictions differ before/after fit"
        )


class TestFixedParams:
    """Test that all model classes expose fixed, deterministic hyperparameters."""

    def test_models_fixed_params(self):
        """Each model exposes its params; LGBM uses exact locked constants.

        LGBM must use LGBMRegressor (NOT LGBMRanker) — see plan 02-06 Pitfall 5.
        No model module imports GridSearchCV/RandomizedSearchCV/optuna.
        """
        import alpharank.models.base as _base_mod
        import alpharank.models.composite as _composite_mod
        import alpharank.models.linear as _linear_mod
        import alpharank.models.elastic as _elastic_mod
        import alpharank.models.lgbm as _lgbm_mod

        # Read source of each module via its __file__ attribute (static, no dynamic import).
        # Forbidden: no hyperparameter search infrastructure in any model module.
        # Note: LGBMRanker is checked separately below (via isinstance) rather than
        # string scan, because the lgbm.py docstring legitimately mentions it as an
        # anti-pattern explanation.
        forbidden_search = ["GridSearchCV", "RandomizedSearchCV", "optuna"]
        for mod, label in [
            (_base_mod, "base"),
            (_composite_mod, "composite"),
            (_linear_mod, "linear"),
            (_elastic_mod, "elastic"),
            (_lgbm_mod, "lgbm"),
        ]:
            src_file = getattr(mod, "__file__", None)
            if src_file:
                with open(src_file) as f:
                    source = f.read()
                for bad in forbidden_search:
                    assert bad not in source, (
                        f"Module alpharank.models.{label} must not reference {bad!r}"
                    )

        # LGBMRanker check: the inner model must be an LGBMRegressor instance,
        # not an LGBMRanker instance (runtime check, not string scan).
        from lightgbm import LGBMRegressor
        from alpharank.models.lgbm import LGBMRankModel
        lgbm_instance = LGBMRankModel()
        assert isinstance(lgbm_instance._model, LGBMRegressor), (
            "LGBMRankModel._model must be LGBMRegressor (NOT LGBMRanker)"
        )

        # LGBM params must match locked constants exactly
        from alpharank.models.lgbm import LGBMRankModel
        lgbm = LGBMRankModel()
        params = lgbm.get_params()

        assert params["n_estimators"] == 200
        assert params["learning_rate"] == pytest.approx(0.05)
        assert params["num_leaves"] == 15
        assert params["max_depth"] == 3
        assert params["min_child_samples"] == 20
        assert params["subsample"] == pytest.approx(0.9)
        assert params["subsample_freq"] == 1
        assert params["colsample_bytree"] == pytest.approx(0.9)
        assert params["random_state"] == 42
        assert params.get("deterministic") is True
        assert params.get("force_row_wise") is True
        assert params.get("verbosity") == -1

    def test_baseline_order_exported(self):
        """BASELINE_ORDER exported from models/__init__.py has exactly 4 models in correct order."""
        from alpharank.models import BASELINE_ORDER
        from alpharank.models.composite import EqualWeightComposite
        from alpharank.models.linear import LinearRankModel
        from alpharank.models.elastic import ElasticNetRankModel
        from alpharank.models.lgbm import LGBMRankModel

        assert len(BASELINE_ORDER) == 4
        assert BASELINE_ORDER[0] is EqualWeightComposite
        assert BASELINE_ORDER[1] is LinearRankModel
        assert BASELINE_ORDER[2] is ElasticNetRankModel
        assert BASELINE_ORDER[3] is LGBMRankModel


# ---------------------------------------------------------------------------
# Task 2 Tests: Model comparison harness with identical protocol
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison_result():
    """Module-scoped fixture: runs full comparison once, shared across tests.

    Uses n_assets=30, n_months=48 for speed — planted IC tolerance still holds
    at smaller panel (documented choice: LGBM with 15 splits x 48 months x 30
    assets takes ~30-40s; n=50, T=60 would take ~90s+).
    """
    from alpharank.models.comparison import run_model_comparison
    from alpharank.validation.purged_cv import PurgedCVEvaluator

    X, y = _make_xy(n_assets=30, n_months=48, seed=42)
    evaluator = PurgedCVEvaluator(n_folds=6, n_test_folds=2, purged_size=1, embargo_size=1)
    table, oos_frames = run_model_comparison(X, y, evaluator)
    return table, oos_frames


class TestModelComparison:
    """Tests for run_model_comparison harness."""

    def test_all_models_positive_ic(self, comparison_result):
        """run_model_comparison returns 4-row DataFrame in exact baseline order.

        All models have mean_ic > 0.0 — all recover the planted linear alpha.

        Note: strict cross-model ordering is NOT asserted here. With linearly
        planted alpha, LinearRegression may legitimately beat LGBM because the
        planted signal is a linear combination of factors, and tree models do not
        necessarily dominate linear models on simple linear signals. This is
        expected behaviour and is NOT a bug.
        """
        from alpharank.models import BASELINE_ORDER

        table, _ = comparison_result

        # DataFrame shape and columns
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 4
        required_cols = {"model", "mean_ic", "icir", "nw_tstat", "p_value", "n_months"}
        assert required_cols.issubset(set(table.columns)), (
            f"Missing columns: {required_cols - set(table.columns)}"
        )

        # Exact baseline order
        expected_names = [cls().name for cls in BASELINE_ORDER]
        assert list(table["model"]) == expected_names, (
            f"Expected baseline order {expected_names}, got {list(table['model'])}"
        )

        # All models recover positive mean IC on planted-alpha data
        for _, row in table.iterrows():
            assert row["mean_ic"] > 0.0, (
                f"Model {row['model']!r} mean_ic={row['mean_ic']:.4f} should be > 0.0"
            )

    def test_comparison_returns_oos_scores(self, comparison_result):
        """run_model_comparison also returns per-model OOS score frames.

        Each frame is (date x symbol) suitable for build_decile_weights.
        Index/columns must match the label frame's shape.
        """
        from alpharank.models import BASELINE_ORDER

        table, oos_frames = comparison_result

        expected_names = [cls().name for cls in BASELINE_ORDER]
        assert set(oos_frames.keys()) == set(expected_names), (
            f"oos_frames keys {set(oos_frames.keys())} != model names {set(expected_names)}"
        )

        for model_name, frame in oos_frames.items():
            assert isinstance(frame, pd.DataFrame), (
                f"oos_frames[{model_name!r}] must be a DataFrame (date x symbol)"
            )
            assert frame.index.name == "date" or frame.index.name is None
            assert len(frame) > 0, f"oos_frames[{model_name!r}] is empty"
            assert len(frame.columns) > 0, f"oos_frames[{model_name!r}] has no columns"
