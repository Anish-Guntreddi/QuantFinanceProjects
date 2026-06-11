"""VolSurfaceLab: Options volatility surface research system.

Self-contained options volatility research combining:
- Deterministic synthetic chain generator (SVI ground truth)
- Robust IV solver (LetsBeRational + brentq fallback)
- SVI calibration with butterfly + calendar no-arb validation
- HAR-RV, GARCH, EGARCH realized-vol forecasting
- Delta-hedged variance risk premium strategy
- One-command research report runner

Full public API frozen in plan 04-08.
"""

__version__ = "0.1.0"
