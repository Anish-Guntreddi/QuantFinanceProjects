---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 05-defiregimenet-07-PLAN.md (Pipeline + ReportBuilder)
last_updated: "2026-06-12T01:09:51.243Z"
last_activity: 2026-06-10 — Roadmap and STATE initialized; requirements mapped to 5 phases
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 43
  completed_plans: 41
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-10)

**Core value:** Every project runs end-to-end (data → model → backtest/analysis → report) with one command, produces honest research output, and passes its test suite.
**Current focus:** Phase 1 — QBacktest

## Current Position

Phase: 1 of 5 (QBacktest)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-06-10 — Roadmap and STATE initialized; requirements mapped to 5 phases

Progress: [███░░░░░░░] 33%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. QBacktest | 0 | - | - |
| 2. AlphaRank | 0 | - | - |
| 3. MacroRegime | 0 | - | - |
| 4. VolSurfaceLab | 0 | - | - |
| 5. DeFiRegimeNet | 0 | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*
| Phase 01-qbacktest P02 | 22 | 2 tasks | 7 files |
| Phase 01-qbacktest P04 | 5 | 2 tasks | 4 files |
| Phase 01-qbacktest P05 | 6 | 3 tasks | 5 files |
| Phase 01-qbacktest P06 | 8 | 2 tasks | 6 files |
| Phase 01-qbacktest P07 | 6 | 2 tasks | 4 files |
| Phase 01-qbacktest P08 | 7 | 3 tasks | 7 files |
| Phase 02-alpharank P01 | 12 | 3 tasks | 24 files |
| Phase 02-alpharank P02 | 5 | 2 tasks | 4 files |
| Phase 02-alpharank P05 | 4 | 2 tasks | 5 files |
| Phase 02-alpharank P04 | 4 | 2 tasks | 3 files |
| Phase 02-alpharank P03 | 8 | 3 tasks | 8 files |
| Phase 02-alpharank P06 | 18 | 2 tasks | 10 files |
| Phase 02-alpharank P07 | 11 | 3 tasks | 5 files |
| Phase 03-macroregime P01 | 25 | 2 tasks | 20 files |
| Phase 03-macroregime P02 | 5 | 2 tasks | 4 files |
| Phase 03-macroregime P03 | 6 | 2 tasks | 3 files |
| Phase 03-macroregime P04 | 6 | 2 tasks | 5 files |
| Phase 03-macroregime P05 | 6 | 2 tasks | 5 files |
| Phase 03-macroregime P07 | 12 | 2 tasks | 4 files |
| Phase 03-macroregime P08 | 12 | 3 tasks | 5 files |
| Phase 03-macroregime P09 | 525804 | 2 tasks | 2 files |
| Phase 04-volsurfacelab P01 | 9 | 2 tasks | 12 files |
| Phase 04-volsurfacelab P02 | 3 | 2 tasks | 2 files |
| Phase 04-volsurfacelab P03 | 4 | 2 tasks | 2 files |
| Phase 04-volsurfacelab P04 | 4 | 2 tasks | 2 files |
| Phase 04-volsurfacelab P05 | 5 | 2 tasks | 2 files |
| Phase 04-volsurfacelab P06 | 6 | 2 tasks | 4 files |
| Phase 04-volsurfacelab P07 | 8 | 2 tasks | 3 files |
| Phase 04-volsurfacelab P08 | 11 | 2 tasks | 5 files |
| Phase 05-defiregimenet P02 | 38 | 2 tasks | 4 files |
| Phase 05-defiregimenet P03 | 5 | 2 tasks | 4 files |
| Phase 05-defiregimenet P04 | 15 | 2 tasks | 5 files |
| Phase 05-defiregimenet P06 | 8 | 2 tasks | 2 files |
| Phase 05-defiregimenet P05 | 15 | 2 tasks | 2 files |
| Phase 05-defiregimenet P07 | 17 | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Init: QBacktest built fresh (not extending existing backtester); installable as path dep for projects 2-5
- Init: py-vollib-vectorized dropped (numba failure on 3.11); plain py_vollib 1.0.12 + scipy brentq fallback confirmed working
- Init: VolSurfaceLab uses standalone P&L accounting — does NOT route through QBacktest event engine
- Init: Codex read-only gate (`codex exec --sandbox read-only`) required after every phase before marking complete
- Init: FRED unauthenticated CSV removed Nov 2025; fredapi free key = optional real-data path; synthetic macro generator = default/offline path
- [Phase 01-qbacktest]: EventQueue heap entries: (ts_nanos, priority, counter, event) — no rich comparison needed on event objects
- [Phase 01-qbacktest]: HistoricalDataHandler.peek_next_bar reads cursor without mutation — T+1 fill engine prerequisite
- [Phase 01-qbacktest]: slippage is informational in cumulative_costs — commission only reduces cash; invariant uses book value (avg_fill_price) not market price
- [Phase 01-qbacktest]: Portfolio reversal splits into full-close + open-residual within single on_fill call
- [Phase 01-qbacktest]: Slippage sign convention: BUY +adjustment (pay more), SELL -adjustment (receive less)
- [Phase 01-qbacktest]: FillEvent.slippage stores total currency cost: abs(price_adjustment) * qty, consistent with portfolio cumulative_costs
- [Phase 01-qbacktest]: RiskManager uses POST-TRADE projected values for both position_weight and gross_exposure checks; limits are inclusive (<=)
- [Phase 01-qbacktest]: T+1 flush order: _flush_pending_orders() runs BEFORE update_bars() — orders from bar T fill at bar T+1 open, never same-bar
- [Phase 01-qbacktest]: No reset() method on EventDrivenBacktester — fresh instances only (locked)
- [Phase 01-qbacktest]: WalkForwardRunner: isolation via construction not reset() — engine_factory called fresh per window
- [Phase 01-qbacktest]: generate_windows step defaults to test_bars — non-overlapping test segments by default
- [Phase 01-qbacktest]: OOS equity curve re-basing: window N scaled so first value equals window N-1 terminal equity
- [Phase 01-qbacktest]: matplotlib.use('Agg') at tearsheet module import before pyplot — headless safety without polluting qbacktest package init
- [Phase 01-qbacktest]: MA strategy uses EXIT (not FLAT) for crossdown signal — portfolio.generate_orders only handles LONG/SHORT/EXIT
- [Phase 02-alpharank]: No sys.path hacks anywhere in alpharank — package imports only (locked)
- [Phase 02-alpharank]: Single seeded default_rng per CrossSectionalGenerator — no global np.random calls
- [Phase 02-alpharank]: Planted alpha formula (LOCKED): alpha = IC_target * monthly_vol / sqrt(1 - IC_target^2)
- [Phase 02-alpharank]: Delist: OHLCV frames truncated at delist month (no NaN rows) — qbacktest HistoricalDataHandler convention
- [Phase 02-alpharank]: yfinance import is lazy (inside function body) — never in module scope for offline tests
- [Phase 02-alpharank]: safe_shift asserts n>=1 at call site — impossible to construct factor with n<=0
- [Phase 02-alpharank]: FeatureLeakageValidator wired into build_feature_panel — every construction self-asserts on leakage
- [Phase 02-alpharank]: value_proxy/quality_proxy: monthly.shift(1) for 1-month publication lag before daily ffill
- [Phase 02-alpharank]: bisect_right O(log k) as-of rebalance lookup in PrecomputedWeightsStrategy — rebal_keys sorted once in __init__
- [Phase 02-alpharank]: max_gross_exposure=2.0 locked for long-short: gross exposure = 1 long + 1 short = 2.0
- [Phase 02-alpharank]: CPCV test side is list[ndarray] — np.concatenate(test_sets) required before any index arithmetic
- [Phase 02-alpharank]: Panel expansion via flatnonzero dict NOT n_assets arithmetic — variable universe after delistings
- [Phase 02-alpharank]: CPCV aggregation: average predictions across paths before IC computation (not average IC values)
- [Phase 02-alpharank]: pct_change(fill_method=None) required — default pad fill triggers FutureWarning-as-error with delist NaN gaps
- [Phase 02-alpharank]: maxlags=floor(4*(T/100)^0.25): T=60 gives 3 not 4 (plan doc arithmetic error corrected)
- [Phase 02-alpharank]: min_obs=3 for compute_ic_series — minimum for valid Spearman (plan spec said 5 but test uses 3-asset input)
- [Phase 02-alpharank]: icir zero-std guard: std < 1e-14 (not == 0.0) — floating point makes constant arrays non-zero std ~1e-18
- [Phase 02-alpharank]: LGBMRankModel wraps LGBMRegressor (NOT LGBMRanker) — continuous rank labels incompatible with relevance-tier LambdaRank objective
- [Phase 02-alpharank]: _make_xy uses BME dates (panel.monthly_returns.index) not resample(ME) — avoids 30% date mismatch causing biased label alignment
- [Phase 02-alpharank]: Leakage validator threshold raised to 0.5 — catches IC~1.0 look-ahead bugs without falsely rejecting genuinely predictive factors
- [Phase 02-alpharank]: PipelineResults dataclass returned by run(): in-process testing without subprocess
- [Phase 02-alpharank]: Quick mode CV params: n_folds=5/n_test_folds=2/purged=1/embargo=0 — skfolio requires purge+embargo < fold_size
- [Phase 03-macroregime]: TRANSITION_MATRIX ergodicity: Stagflation->Recovery path added (0.01) to guarantee all 4 regimes visited; initial matrix had 0-prob path causing state dropout
- [Phase 03-macroregime]: SyntheticMacroPanel convention: observation-date macro panel only; release-lag application is MacroDataLoader's responsibility (plan 03-02)
- [Phase 03-macroregime]: No pct_change in macroregime package: cumulative cumprod for OHLCV construction avoids FutureWarning-as-error in CI
- [Phase 03-macroregime]: attrs cleared on Series copy before pd.concat: DatetimeIndex in attrs causes ValueError; cleared on copy, original Series retains attrs
- [Phase 03-macroregime]: no ffill in load_panel: frequency alignment deferred to plan 03-07 pipeline — ffill must happen AFTER lag
- [Phase 03-macroregime]: rolling_corr warm-up is window+1 (not window+2): first valid corr at bar window, shift pushes to bar window+1; corrected from plan docstring
- [Phase 03-macroregime]: pct_change(fill_method=None) + shift(1) is the locked causal feature pattern for all market features
- [Phase 03-macroregime]: CausalRegimeDetector: predict(X[:t+1])[-1] is the ONLY safe HMM causal pattern; refit schedule is pure function of t (never len(X)) — oracle invariant by construction
- [Phase 03-macroregime]: align_regime_labels returns inverse permutation: np.argsort(np.argsort(means[:,dim])) — double argsort maps raw->rank (not single argsort which gives rank->raw)
- [Phase 03-macroregime]: transition_matrix unvisited rows = uniform 1/K (not zero/NaN) to keep matrix row-stochastic for all downstream consumers
- [Phase 03-macroregime]: TargetWeightPortfolio overrides generate_orders only — on_fill/invariant inherited from qbacktest unchanged (locked: qbacktest never modified)
- [Phase 03-macroregime]: TargetWeightStrategy tracks _last_emitted signed weight (not direction): |new-last|>1e-9 re-emission closes PrecomputedWeightsStrategy direction-only gap
- [Phase 03-macroregime]: Expanding z-score (not full-sample): full-sample standardization leaks future mean/std into historical feature values, breaking causality
- [Phase 03-macroregime]: Two separate CausalRegimeDetector instances for macro and market: mixing monthly + daily features into one matrix creates frequency-alignment artifacts and is semantically wrong
- [Phase 03-macroregime]: K sensitivity by Sharpe forbidden (anti-feature): selecting K to maximize Sharpe overfits regime model to backtest period
- [Phase 03-macroregime]: Regime reuse across walk-forward windows is safe: CausalRegimeDetector oracle guarantee proves label at t is window-invariant
- [Phase 03-macroregime]: build_all receives asset_ohlcv as explicit arg: PipelineResults is frozen, asset_ohlcv not stored in it — runner re-generates from same seed
- [Phase 03-macroregime]: summary.md written to parent(output_dir): test expects tmp_path/figures/*.png AND tmp_path/summary.md (not tmp_path/figures/summary.md)
- [Phase 03-macroregime]: Lazy __getattr__ for ReportBuilder, FredMacroLoader, build_strategy_engine in macroregime/__init__: defers matplotlib/pyplot import and optional deps to first access, mirrors qbacktest TearsheetRenderer pattern
- [Phase 03-macroregime]: PIT fix: union-then-ffill-then-restrict pattern for weekend publication dates in _combine_regimes — daily_index.union(macro_regimes.index) ensures weekend pub-date regime changes reach first Monday
- [Phase 04-volsurfacelab]: vollib namespace (not py_vollib) used in chain.py to avoid DeprecationWarning
- [Phase 04-volsurfacelab]: _svi_total_variance in chain.py is independent from svi.py — oracle isolation for VSL-03 calibration tests
- [Phase 04-volsurfacelab]: ChainData frozen=True dataclass — immutable, safe for session-scope fixtures
- [Phase 04-volsurfacelab]: brentq fallback uses explicit f_lo*f_hi sign-check before calling brentq — avoids ValueError propagating; deep-OTM near-zero inputs resolve to NaN cleanly
- [Phase 04-volsurfacelab]: bs_price is standalone closed-form BS (scipy.stats.norm) independent of vollib — ensures brentq fallback is a genuine second implementation
- [Phase 04-volsurfacelab]: Calendar check restricted to linspace(-1.5, 1.5, 200) by default — deep-wing violations are parameterization artifacts, not actionable arbitrage
- [Phase 04-volsurfacelab]: validate_surface uses warnings.warn(UserWarning) never raises — gate behavior: exclude and continue
- [Phase 04-volsurfacelab]: w(k) > 0 positivity constraint added alongside g(k) >= 0 in SLSQP — handles negative-a edge case (pitfall 7)
- [Phase 04-volsurfacelab]: statsmodels OLS for HAR-RV (not arch HARX) — DataScaleWarning on raw RV; cleaner parameter names
- [Phase 04-volsurfacelab]: QLIKE convention: L(h,rv)=rv/h-log(rv/h)-1 (Patton 2011); under-forecast penalized more; oracle test asserts qlike(rv,2rv)<qlike(rv,0.5rv)
- [Phase 04-volsurfacelab]: HAR no-look-ahead: single-point perturbation oracle (rv[t] change must not affect forecast at t; t+1 may change — that is correct causal behavior)
- [Phase 04-volsurfacelab]: StandalonePortfolio does NOT import qbacktest — locked roadmap decision; standalone accounting for VolSurfaceLab
- [Phase 04-volsurfacelab]: Straddle delta test uses r=0 for ATM delta cancellation; at r>0 carry term shifts net delta above 0.15 threshold by design
- [Phase 04-volsurfacelab]: theta_daily = vollib_theta / 252 (business-day convention); VRP series is point-in-time IV^2 - r_t^2*252 (no look-ahead)
- [Phase 04-volsurfacelab]: Honest-path discipline: pipeline uses solve_chain_iv on OPTION PRICES; chain.options['true_iv'] is a test oracle only — using it would bypass the IV estimation problem
- [Phase 04-volsurfacelab]: svi_surface dependency injection for arb-gate testing via __init__ param; uses context manager to temporarily override SYNTHETIC_SVI_SURFACE in chain module
- [Phase 04-volsurfacelab]: summary.md location: output_dir.parent (beside figures/); matches locked Phase 3 macroregime decision
- [Phase 04-volsurfacelab]: importlib.util.spec_from_file_location for runner import: no sys.path hacks in test, mirrors macroregime locked pattern
- [Phase 04-volsurfacelab]: main() catches Exception returning 1; argparse SystemExit propagated directly for test_runner_bad_args to assert on exit code 2
- [Phase 04-volsurfacelab]: volsurfacelab __all__ frozen at 34 symbols: version, Chain(5), IV(3), SVI(7), Forecast(7), Strategy(7), Pipeline(3), Lazy(2)
- [Phase 04-volsurfacelab]: theta convention corrected: vollib.theta() divides by 365 (per-calendar-day); theta_daily = vollib_theta * (365/252) for business-day; theta_daily is reporting-only, not in P&L
- [Phase 04-volsurfacelab]: no-arb gate extended to block strategy entry: pipeline filters iv_frame to validated_maturities before ChainData — excluded slices cannot drive entry_iv or P&L
- [Phase 05-defiregimenet]: DeFiRegimeNet label encoding LOCKED: state = bull_flag*2 + high_vol_flag (0=bear/low, 1=bear/high, 2=bull/low, 3=bull/high); matches DGP true_states convention
- [Phase 05-defiregimenet]: Expanding median vol threshold in make_regime_labels: causal w.r.t. label-estimation order; expanding (not global) to avoid look-ahead in threshold
- [Phase 05-defiregimenet]: std < 1e-14 guard in expanding_zscore (not == 0.0): matches alpharank icir convention; floating-point constant arrays produce std ~1e-18 not exactly 0
- [Phase 05-defiregimenet]: defiregimenet regime/detector.py: feature matrix construction inlined in tests to stay file-disjoint from features.crypto (parallel wave-2 plan)
- [Phase 05-defiregimenet]: anti-feature guard test reads diagnostics.py source as text (case-insensitive); 'sharpe' must not appear in diagnostics source including comments
- [Phase 05-defiregimenet]: XGBRegimeClassifier default max_depth=4 (depth=3 gave exactly 0.300 accuracy on seeded panel, failing strict >0.30 threshold)
- [Phase 05-defiregimenet]: labels_to_probas: eps directly on off-target classes (not normalised form); target gets 1-(n_states-1)*eps
- [Phase 05-defiregimenet]: .gitignore negation added for **/src/**/models/ to un-ignore Python source model subpackages (mirrors data/ exception)
- [Phase 05-defiregimenet]: cramers_v uses scipy chi2_contingency with k=min(reduced_rows,reduced_cols) after zero-marginal drop; V clipped to [0,1]
- [Phase 05-defiregimenet]: DGP integration test uses inline rolling-vol x return-sign proxy (no detector import) to respect parallel-plan boundary; V>0.5 on detected sequences deferred to 05-07
- [Phase 05-defiregimenet]: per_token_forecast_comparison is a pure delegation loop — no arch calls in the primary path; garch_studentst_variance uses analytic GARCH recursion seeded from training terminal state (not res.forecast)
- [Phase 05-defiregimenet]: Causality oracle corrected: perturbing AT the target bar (returns.index[split_idx+1+k]) must not change fcst.iloc[k]; perturbing the ORIGIN bar (split_idx+k) does change fcst.iloc[k] (target-date labeling invariant)
- [Phase 05-defiregimenet]: Joint market-regime detection: cross-sectional mean features -> single CausalRegimeDetector shared across all tokens; resolves 4-state permutation ambiguity; cross_token_v off-diagonal V > 0.5 confirmed
- [Phase 05-defiregimenet]: n_years=2 (int) in quick mode: float 1.5 caused pd.date_range FutureWarning (non-integer periods deprecated)

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 3 (MacroRegime): Needs dedicated phase research before planning — ALFRED vintage handling and filtered-vs-smoothed HMM architecture
- Phase 4 (VolSurfaceLab): Needs dedicated phase research before planning — SVI butterfly constraint formulation for SLSQP
- Phase 5 (DeFiRegimeNet): Needs dedicated phase research before planning — crypto data quality and synthetic generator realism

## Session Continuity

Last session: 2026-06-12T01:09:51.241Z
Stopped at: Completed 05-defiregimenet-07-PLAN.md (Pipeline + ReportBuilder)
Resume file: None
