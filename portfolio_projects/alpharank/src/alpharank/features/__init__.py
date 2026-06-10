"""Features subpackage: cross-sectional feature engineering."""

from alpharank.features.base import (
    safe_shift,
    cross_sectional_zscore,
    FeatureLeakageValidator,
)

__all__ = [
    "safe_shift",
    "cross_sectional_zscore",
    "FeatureLeakageValidator",
    "build_feature_panel",
]


def build_feature_panel(*args, **kwargs):  # type: ignore[return]
    """Lazy forward to factors.build_feature_panel (avoids circular imports)."""
    from alpharank.features.factors import build_feature_panel as _bfp
    return _bfp(*args, **kwargs)
