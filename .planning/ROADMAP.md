# Roadmap: Quant Portfolio Build — Five Flagship Projects

## Overview

Five research-grade quantitative finance projects built sequentially under `portfolio_projects/`, anchored by QBacktest — a pip-installable event-driven backtesting engine that downstream projects import as a shared library. Each phase delivers one complete, runnable, tested project: QBacktest (shared infrastructure) → AlphaRank (first engine consumer) → MacroRegime (macro/regime allocation) → VolSurfaceLab (options research, standalone P&L) → DeFiRegimeNet (hybrid regime detection in crypto). Every phase passes its pytest suite offline, survives a codex read-only leakage audit, and ships a research-style report from a single runner command.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: QBacktest** - Build the pip-installable event-driven backtesting engine that all other projects depend on (completed 2026-06-10)
- [x] **Phase 2: AlphaRank** - Build the ML cross-sectional equity ranking pipeline using QBacktest as the portfolio execution layer (completed 2026-06-10)
- [x] **Phase 3: MacroRegime** - Build the regime-switching macro asset allocation system with point-in-time FRED data and HMM/GMM regime detection (completed 2026-06-11)
- [ ] **Phase 4: VolSurfaceLab** - Build the options volatility research system with IV surface fitting, RV forecasting, and standalone strategy P&L
- [ ] **Phase 5: DeFiRegimeNet** - Build the hybrid ML + econometric regime detection system for DeFi/crypto markets

## Phase Details

### Phase 1: QBacktest
**Goal**: A pip-installable, tested backtesting engine exists that any sibling project can import; it enforces T+1 fill, accounting invariants, walk-forward isolation, and produces a tearsheet from one runner command
**Depends on**: Nothing (first phase)
**Requirements**: QBT-01, QBT-02, QBT-03, QBT-04, QBT-05, QBT-06, QBT-07, QBT-08, QBT-09, QBT-10, QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05
**Note on QUAL requirements**: QUAL-01 through QUAL-05 are cross-cutting and first established here. They recur as active constraints in every subsequent phase — each phase must satisfy all five before being marked complete.
**Success Criteria** (what must be TRUE):
  1. `pip install -e portfolio_projects/qbacktest` succeeds and `import qbacktest` works from a sibling project's test suite
  2. pytest suite passes offline with no network calls (seeded fixtures throughout); accounting invariant assertion holds after every synthetic-data fill test
  3. An oracle test demonstrates that signal generated at bar T fills at T+1 open price — not the same-bar close — under all slippage/spread model configurations
  4. WalkForwardRunner produces aggregated out-of-sample results with no state bleed between windows (verified by injecting a sentinel into engine state and confirming it does not appear in the next window)
  5. One-command runner (`python run_demo.py`) produces a tearsheet PNG and summary table with gross and net-of-cost Sharpe side by side on synthetic data; codex read-only review passes with no unresolved findings
**Plans:** 9/9 plans complete

Plans:
- [ ] 01-01-PLAN.md — Package skeleton, synthetic OHLCV generator, conftest + Wave 0 test stubs (wave 1)
- [ ] 01-02-PLAN.md — Typed events, deterministic EventQueue, DataHandler + Strategy ABCs (wave 2)
- [ ] 01-03-PLAN.md — Metrics module: Sharpe/Sortino/MDD/turnover/hit rate, bootstrap CI, gross-vs-net report (wave 2)
- [ ] 01-04-PLAN.md — Portfolio accounting: on_fill sole mutation point, 1e-6 invariant, order generation risk seam (wave 3)
- [ ] 01-05-PLAN.md — Execution: slippage/spread/commission models, fill_at_open handler, RiskManager (wave 3)
- [ ] 01-06-PLAN.md — Engine assembly: T+1 pending buffer, oracle + determinism tests, EOD cancellation (wave 4)
- [ ] 01-07-PLAN.md — WalkForwardRunner: engine_factory isolation, sentinel state-bleed test, OOS aggregation (wave 5)
- [ ] 01-08-PLAN.md — Tearsheet renderer, demo MA strategy, run_demo.py, README (wave 5)
- [ ] 01-09-PLAN.md — Quality gate: public API freeze, strict full suite, codex read-only review (wave 6)

### Phase 2: AlphaRank
**Goal**: A complete ML cross-sectional equity ranking pipeline exists — factors to labels to models to long-short portfolio — backtested through QBacktest with purged CV and net-of-cost performance analytics in one runner command
**Depends on**: Phase 1
**Requirements**: ALR-01, ALR-02, ALR-03, ALR-04, ALR-05, ALR-06, ALR-07, ALR-08, ALR-09
**Note on QUAL requirements**: QUAL-01 through QUAL-05 apply to this phase (pytest offline, README, net-of-cost + significance, codex gate, shared conventions).
**Success Criteria** (what must be TRUE):
  1. Leakage assertions in the feature module confirm no feature at time t uses data timestamped after t; label unit tests match hand-computed next-period cross-sectional rank examples
  2. Validation pipeline uses purged/embargoed walk-forward CV (skfolio CombinatorialPurgedCV or equivalent) — no standard KFold present anywhere in training or evaluation code
  3. Analytics report shows IC, rank-IC, ICIR with Newey-West t-statistics, and IC decay curves across forecast horizons for all four models (equal-weight, linear, elastic net, LightGBM) in a consistent comparison table
  4. Long-short decile portfolio backtested through qbacktest reports turnover and net-of-cost Sharpe; factor attribution regression appears in the final research report
  5. One-command runner (`python run_pipeline.py`) produces a research report with figures covering question, methodology, results, and robustness; codex leakage audit passes with no unresolved findings
**Plans:** 8/8 plans complete

Plans:
- [ ] 02-01-PLAN.md — Package skeleton, CrossSectionalGenerator with planted IC, Wave 0 test stubs, optional yfinance loader (wave 1)
- [ ] 02-02-PLAN.md — Six lag-safe cross-sectional factors + FeatureLeakageValidator (wave 2)
- [ ] 02-03-PLAN.md — Forward-rank labels, IC/ICIR/Newey-West analytics, IC decay, factor attribution (wave 2)
- [ ] 02-04-PLAN.md — PurgedCVEvaluator wrapping skfolio CombinatorialPurgedCV(6,2,1,1), no-KFold guard (wave 2)
- [ ] 02-05-PLAN.md — Decile L/S construction, PrecomputedWeightsStrategy, qbacktest wiring with locked costs (wave 2)
- [ ] 02-06-PLAN.md — Four models in strict baseline order + identical-protocol comparison harness (wave 3)
- [ ] 02-07-PLAN.md — run_pipeline.py runner, ReportBuilder figures, README research report, integration tests (wave 4)
- [ ] 02-08-PLAN.md — Quality gate: strict suite x2, API freeze, codex read-only leakage audit (wave 5)

### Phase 3: MacroRegime
**Goal**: A complete macro regime-switching asset allocation system exists — point-in-time FRED/synthetic macro data, causal HMM/GMM regime detection, regime-conditional allocation through QBacktest — benchmarked against 60/40, equal weight, and risk parity in one runner command
**Depends on**: Phase 2
**Requirements**: MCR-01, MCR-02, MCR-03, MCR-04, MCR-05, MCR-06, MCR-07, MCR-08
**Note on QUAL requirements**: QUAL-01 through QUAL-05 apply to this phase. Dedicated phase research recommended before planning (ALFRED vintage handling, filtered-vs-smoothed HMM architecture).
**Success Criteria** (what must be TRUE):
  1. Macro data layer runs fully offline using the deterministic synthetic generator; a unit test proves that appending a future data point to a fitted HMM does not change the regime label at any historical time t (causal/filtered states, not smoothed)
  2. Every macro series has an explicit release-lag config; a unit test asserts that the strategy's data view at any date t contains only series published on or before t (point-in-time correctness)
  3. Regime diagnostics output includes transition matrix, mean dwell times per regime, and regime-label alignment documentation (states ordered by economically meaningful quantity) — visible in the research report
  4. Walk-forward backtest through qbacktest shows regime-conditional strategy versus 60/40, equal weight, and risk parity benchmarks over identical periods with identical cost assumptions; net-of-cost Sharpe with bootstrap CIs reported for all four strategies
  5. One-command runner (`python run_pipeline.py`) produces the research report; codex leakage audit (FRED release-lag focus) passes with no unresolved findings
**Plans:** 9/9 plans complete

Plans:
- [ ] 03-01-PLAN.md — Package skeleton, Wave-0 test stubs, SyntheticMacroGenerator (4-state Markov DGP) (wave 1)
- [ ] 03-02-PLAN.md — Point-in-time macro data layer: release-lag, as-of masking, Synthetic/Fred loaders (wave 2)
- [ ] 03-03-PLAN.md — Causal market features: realized vol, momentum, drawdown, rolling correlation (wave 2)
- [ ] 03-04-PLAN.md — CausalRegimeDetector (HMM+GMM), label alignment, causality oracle test (wave 2)
- [ ] 03-05-PLAN.md — TargetWeightPortfolio + TargetWeightStrategy, regime→weights through qbacktest (wave 2)
- [ ] 03-06-PLAN.md — Benchmarks: 60/40, equal weight, inverse-vol risk parity with identical costs (wave 3)
- [ ] 03-07-PLAN.md — Pipeline assembly, walk-forward, OOS regime stability + K sensitivity (wave 3)
- [ ] 03-08-PLAN.md — run_macroregime.py runner, ReportBuilder figures, README research report (wave 4)
- [ ] 03-09-PLAN.md — Quality gate: API freeze, strict suite x2, codex read-only leakage audit (wave 5)

### Phase 4: VolSurfaceLab
**Goal**: A complete options volatility research system exists — synthetic/real options chains, IV surface fitting with no-arbitrage validation, HAR/GARCH/EGARCH RV forecasting with QLIKE evaluation, and IV-vs-RV spread strategy P&L — all from one runner command
**Depends on**: Phase 3
**Note**: VolSurfaceLab is architecturally independent of QBacktest (standalone P&L accounting); this sequential ordering is for simplicity. Dedicated phase research recommended before planning (SVI butterfly constraint formulation for SLSQP).
**Requirements**: VSL-01, VSL-02, VSL-03, VSL-04, VSL-05, VSL-06, VSL-07, VSL-08
**Note on QUAL requirements**: QUAL-01 through QUAL-05 apply to this phase.
**Success Criteria** (what must be TRUE):
  1. Implied-vol solver recovers synthetic known vols round-trip to 1e-6 precision; deep OTM/ITM inputs that would cause solver divergence are handled gracefully (vega floor + bisection fallback) without raising unhandled exceptions
  2. SVI calibration per maturity slice passes static no-arbitrage validation (butterfly convexity and calendar spread monotonicity checks) before any downstream analysis proceeds; violated slices are logged and excluded with a clear warning
  3. Smile/skew plots per maturity and a 3D/heatmap surface plot are produced and appear in the research report; surface covers the moneyness/maturity range specified in the synthetic chain
  4. RV forecasting comparison table shows QLIKE and MSE for HAR baseline vs GARCH vs EGARCH with Diebold-Mariano test p-values; GARCH fitting uses multi-restart robust wrapper with convergence flags asserted
  5. One-command runner (`python run_pipeline.py`) produces the research report with surface figures, forecast comparison, and strategy P&L with Greeks risk summary; codex review passes with no unresolved findings
**Plans:** 2/8 plans executed

Plans:
- [ ] 04-01-PLAN.md — Package skeleton, synthetic SVI-surface chain + GARCH underlying path, Wave-0 test stubs (wave 1)
- [ ] 04-02-PLAN.md — Robust IV solver: LetsBeRational + brentq fallback, 1e-6 round-trip oracle (wave 2)
- [ ] 04-03-PLAN.md — SVI calibration (butterfly-constrained SLSQP) + static no-arb gate with planted-arb tests (wave 2)
- [ ] 04-04-PLAN.md — RV forecasting: HAR/GARCH/EGARCH, QLIKE + MSE, Diebold-Mariano (wave 2)
- [ ] 04-05-PLAN.md — VRP delta-hedged straddle strategy, standalone P&L accounting, Greeks (wave 2)
- [ ] 04-06-PLAN.md — Pipeline assembly + ReportBuilder: smile/3D/heatmap figures, tables (wave 3)
- [ ] 04-07-PLAN.md — run_pipeline.py runner, README research report, integration tests (wave 4)
- [ ] 04-08-PLAN.md — Quality gate: API freeze, strict suite x2, codex read-only audit (wave 5)

### Phase 5: DeFiRegimeNet
**Goal**: A complete hybrid ML + econometric regime detection system for DeFi/crypto markets exists — deterministic synthetic crypto data, causal HMM/GMM + ML classifiers, GARCH vol forecasting, purged/embargoed CV, per-token diagnostics — with a publication-style research report from one runner command
**Depends on**: Phase 4
**Note**: Reuses purged-CV patterns from AlphaRank and regime patterns from MacroRegime. Dedicated phase research recommended before planning (crypto data quality, synthetic generator realism).
**Requirements**: DFR-01, DFR-02, DFR-03, DFR-04, DFR-05, DFR-06, DFR-07
**Note on QUAL requirements**: QUAL-01 through QUAL-05 apply to this phase.
**Success Criteria** (what must be TRUE):
  1. All tests pass fully offline using the deterministic synthetic crypto generator (24/7 calendar, fat tails, vol clustering); data-quality validation triggers expected warnings when gaps or volume anomalies are injected into test data
  2. Forward-looking regime labels (bull/bear × high/low-vol) are used only in evaluation code; a test confirms they are never accessible to any feature pipeline or model at training time (strict causal separation)
  3. ML classifiers (logistic, XGBoost) are evaluated against HMM/GMM baselines using purged/embargoed CV — same protocol as AlphaRank; comparison table with accuracy and log-loss appears in the report
  4. Per-token diagnostics include Markov transition matrix, dwell times, and regime-count (k) sensitivity analysis; cross-token regime correlation heatmap appears in the research report
  5. One-command runner (`python run_pipeline.py`) produces a publication-style report covering abstract, data, methodology, results, robustness, and limitations; codex leakage audit (crypto data quality + label separation focus) passes with no unresolved findings
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. QBacktest | 8/9 | Complete    | 2026-06-10 |
| 2. AlphaRank | 7/8 | Complete    | 2026-06-10 |
| 3. MacroRegime | 9/9 | Complete   | 2026-06-11 |
| 4. VolSurfaceLab | 2/8 | In Progress|  |
| 5. DeFiRegimeNet | 0/TBD | Not started | - |
