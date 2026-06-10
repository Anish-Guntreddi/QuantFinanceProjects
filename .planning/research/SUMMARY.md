# Research Summary — Five Flagship Quant Portfolio Projects

**Synthesized:** 2026-06-10
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md
**Overall confidence:** HIGH

## Executive Summary

Five interconnected quant research systems in a Python monorepo, anchored by QBacktest — a pip-installable event-driven backtesting engine that the research projects consume as a shared library. Canonical architecture: typed event hierarchy (MarketEvent → SignalEvent → OrderEvent → FillEvent), Strategy ABC plug-in interface, walk-forward validation with purged/embargoed cross-validation, and offline-first synthetic data generators as the default data path.

The environment already has the core scientific stack installed (numpy 2.2.6, pandas 2.3.2, scipy, scikit-learn, lightgbm, xgboost, hmmlearn, arch 7.2.0, cvxpy, QuantLib). New installs completed and validated 2026-06-10: `fredapi 0.5.2`, `skfolio 0.20.1`, `seaborn`. **`py-vollib-vectorized` was installed, found broken (numba typing failure on Python 3.11), and dropped** — plain `py_vollib 1.0.12` (LetsBeRational) is validated working and is the IV solver, with scipy `brentq` fallback for edge cases.

Orchestrator decisions shaping the design:
1. FRED unauthenticated CSV access was removed Nov 2025 — `fredapi` with a free key is the optional real-data path; a deterministic synthetic macro generator is the offline/default path.
2. VolSurfaceLab uses standalone options P&L accounting, NOT the QBacktest event engine.
3. QBacktest is built fresh under `portfolio_projects/qbacktest` (src layout, hatchling), informed by the existing backtester's design but superseding it.
4. Codex read-only review is a required gate per phase, with a dedicated leakage-audit focus after each data pipeline.

The dominant risk across all five projects is look-ahead bias in multiple forms: un-lagged forward return labels, FRED revised-vintage data without release lags, HMM smoothed-state decoding used for trading signals, and same-bar close used for both signal and fill. These are preventable with specific test assertions and architectural constraints — enforced from Phase 1, not retrofitted.

## Key Findings

**Stack:** Python 3.11 + pyproject.toml/hatchling/src-layout for QBacktest packaging. skfolio provides the only free pip-installable CombinatorialPurgedCV (mlfinlab is closed-source/paywalled). SVI calibration implemented directly with scipy differential_evolution + minimize(SLSQP) — no maintained library with arbitrage-free constraints exists. QuantLib for option Greeks. Custom matplotlib tearsheet (pyfolio abandoned; quantstats unnecessary weight). yfinance is optional enrichment only — never a test dependency.

**Features (P1 — non-negotiable):** Purged + embargoed walk-forward CV; point-in-time macro data with FRED release lags; no-arbitrage checks on the vol surface (butterfly + calendar); IC/Rank-IC/ICIR with t-stats; regime persistence diagnostics; QLIKE loss; transaction costs in every backtest with net-of-cost Sharpe beside every gross figure; statistical significance tests on all reported metrics; classical baselines before any ML.

**Architecture:** Hub-and-spoke monorepo. QBacktest is the hub; each research project declares it as a path dependency. Pipeline stages communicate via typed DataFrames only. `on_fill()` is the sole accounting update point. Default fill at T+1 bar open. WalkForwardRunner uses an `engine_factory()` for a fresh engine per window. VolSurfaceLab does not route through the event engine.

**Top pitfalls (build-order implications):**
1. Same-bar close fill → enforce T+1 in QBacktest Phase 1; oracle test confirms
2. Label leakage via un-lagged forward returns → feature/label leakage assertions before any AlphaRank model trains; codex audit gate
3. FRED release lag/vintage bias → point-in-time macro loader + release-lag config as first-class concepts
4. HMM smoothed-state leakage → causal `filtered_regime_sequence()` via rolling fit, never `predict(X_full)` for signals
5. HMM label switching → regime-label aligner (sort states by mean return/vol) before backtesting
6. PnL accounting bugs → accounting invariant test (cash + positions value = initial capital − costs ± realized, to 1e-6) after every fill
7. GARCH local maxima → robust fit wrapper with multi-restarts, assert stationarity + convergence flags
8. IV solver divergence → moneyness grid bounds, vega floor with bisection fallback, post-solve static-arbitrage gate
9. Walk-forward overfitting → Deflated Sharpe Ratio when hyperparameter search precedes OOS evaluation; document trial counts
10. Non-deterministic tests → conftest seed fixtures, random_state pinned everywhere

## Roadmap Implications

Suggested phases: 5 (one per project)

1. **QBacktest Foundation** — pip-installable event engine with T+1 fill, accounting invariant tests, WalkForwardRunner, metrics/tearsheet, seed/FutureWarning CI infrastructure; nothing downstream starts without it
2. **AlphaRank** — first consumer; validates Strategy ABC plug-in seam; proves purged CV implementation reused by DeFiRegimeNet
3. **MacroRegime** — most complex data infrastructure (release lags, synthetic macro generator); causal HMM regime detection + allocation through WalkForwardRunner
4. **VolSurfaceLab** — architecturally independent; IV solver + SVI no-arbitrage surface + HAR/GARCH forecasting; standalone P&L
5. **DeFiRegimeNet** — reuses purged-CV and regime patterns; adds crypto data-quality defenses

**Needs dedicated phase research:** MacroRegime (ALFRED vintage handling, filtered-vs-smoothed HMM architecture), VolSurfaceLab (SVI butterfly constraint formulation for SLSQP), DeFiRegimeNet (crypto data quality/synthetic generator realism).
**Skip phase research:** QBacktest, AlphaRank (well-documented canonical patterns).

## Confidence

| Area | Confidence | Caveat |
|------|------------|--------|
| Stack | HIGH | All new deps install-validated in the actual venv on 2026-06-10 |
| Features | HIGH | Cross-referenced against academic literature and reviewer expectations |
| Architecture | HIGH | Informed by existing repo backtester + Python Packaging User Guide |
| Pitfalls | HIGH | Cross-referenced peer-reviewed literature, official pandas/FRED docs |

**Remaining gaps:** (1) SVI constraint formulation details (resolve in Phase 4 research); (2) ALFRED release-lag calendar accuracy (resolve in Phase 3 research); (3) crypto public-API rate limits (synthetic generator must cover all tests regardless).
