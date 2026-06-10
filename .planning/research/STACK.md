# Stack Research

**Domain:** Portfolio-grade quantitative finance research (event-driven backtesting engine, ML equity ranking, macro regime allocation, options vol surface, DeFi regime detection)
**Researched:** 2026-06-10
**Confidence:** HIGH (versions confirmed against installed environment + PyPI + official docs)

---

## Context: Existing vs New Installs

The environment already has (confirmed installed versions):

| Package | Installed |
|---------|-----------|
| numpy | 1.26.4 |
| pandas | 2.3.2 |
| scipy | 1.16.1 |
| statsmodels | 0.14.5 |
| scikit-learn | 1.7.2 |
| lightgbm | 4.6.0 |
| xgboost | 3.0.5 |
| hmmlearn | 0.3.3 |
| arch | 7.2.0 |
| yfinance | 0.2.65 |
| matplotlib | 3.10.6 |
| plotly | 6.3.0 |
| pytest | 8.4.2 |
| cvxpy | 1.7.2 |
| QuantLib | 1.39 |
| pandas-datareader | 0.10.0 |

Missing and needed: `fredapi`, `skfolio`, `py-vollib-vectorized`, `seaborn`

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11.13 | Runtime | Already pinned in repo; 3.11 is LTS stable through 2027; no reason to change |
| pyproject.toml + hatchling | hatchling >= 1.21 | QBacktest package build | Hatchling is the PyPA-recommended default for pure-Python packages with no C extensions; zero-config src layout support; `pip install -e .` works without setup.py; chosen over setuptools (verbose) and uv_build (no benefit without uv toolchain) |
| src layout | — | Package isolation | Forces import from installed copy not local folder, catching missing `__init__.py` and broken installs early; standard for any library that gets `import`ed by other projects |
| pytest | 8.4.2 (installed) | Testing | Already in use repo-wide; `pytest-cov` for coverage; no migration cost |
| pyyaml | >= 6.0 | Config files | Already in use via CLAUDE.md conventions; YAML configs in `configs/` are the established repo pattern |

### Packaging: QBacktest as a Pip-Installable Library (CRITICAL)

QBacktest must be importable by AlphaRank, MacroRegime, VolSurfaceLab, and DeFiRegimeNet. The correct approach is `src` layout + `pyproject.toml`, NOT `sys.path` hacks.

**Why NOT sys.path hacks:** The existing repo uses `sys.path.append(str(Path(__file__).parent / 'src'))` as a convention in runner scripts. This is acceptable for standalone scripts but breaks when another package tries to `import qbacktest` — there is no installed module to find. The new projects must use proper packaging.

**Recommended pyproject.toml skeleton for QBacktest:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "qbacktest"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pandas>=2.0", "numpy>=1.24"]

[tool.hatch.build.targets.wheel]
packages = ["src/qbacktest"]
```

Consumer projects install with: `pip install -e ../../qbacktest/` (editable, so changes propagate without reinstall).

**Confidence: HIGH** — Python Packaging User Guide (packaging.python.org) confirms hatchling as PyPA default; src layout guidance from same source.

---

### Data Sources

#### Equities / ETFs: yfinance

| Library | Version | Notes |
|---------|---------|-------|
| yfinance | 0.2.65 (installed) | Use for optional real-data path only |

**Reality check (MEDIUM confidence):** yfinance scrapes Yahoo Finance's undocumented JSON endpoints. As of 2025-2026 it faces periodic rate-limiting, IP blocking under heavy use, and data gaps. It works for occasional fetches of daily OHLCV (which is all these projects need) but is not production-reliable. The correct architecture is: synthetic data generator as the primary path (runs offline, deterministic, no API call) + yfinance as an optional enrichment path behind a try/except with a clear fallback. Do not gate any test suite on yfinance succeeding.

**Alternatives NOT recommended:** Polygon.io, Alpha Vantage, Alpaca — all require API keys; violates the no-paid-API constraint.

#### Macro Data: fredapi + synthetic fallback

| Library | Version | Notes |
|---------|---------|-------|
| fredapi | 0.5.2 (pip install needed) | Requires free FRED API key from fred.stlouisfed.org/docs/api/api_key.html |

**Critical finding:** As of November 2025, FRED enforced API key requirements on ALL endpoints including the previously-unauthenticated flat-file CSV endpoint at `fred.stlouisfed.org/data/`. The unauthenticated CSV scraping pattern no longer works. `pandas_datareader` FRED support also requires a key under the hood.

**Architecture decision:** MacroRegime must have a two-path design:
1. **Offline path (default, always works):** Deterministic synthetic macro data generator that produces realistic-looking series for UNRATE, CPI, FEDFUNDS, etc., with configurable regime transitions. All tests use this path.
2. **Live path (optional):** `fredapi.Fred(api_key=os.environ.get("FRED_API_KEY"))` guarded by `if api_key:` check. When an API key is present, fetch real series; cache to `data/` as CSV; subsequent runs use cached files.

`fredapi` 0.5.2 supports both FRED and ALFRED (Archival FRED) for point-in-time data — ALFRED is important for release-lag handling (macro data arrives with lags; using real-time values causes look-ahead bias).

**Confidence: HIGH** — Confirmed from FRED official API docs and search results dated November 2025.

#### Crypto OHLCV: ccxt (Binance public endpoint) + pycoingecko fallback

| Library | Version | Notes |
|---------|---------|-------|
| ccxt | already in existing codebase STACK.md | Use `ccxt.binance().fetch_ohlcv('BTC/USDT', '1d')` — no API key for public market data |
| pycoingecko | 3.x | Free public API, 5-15 calls/min without key; 30/min with free Demo registration |

**Why ccxt over python-binance:** ccxt is exchange-agnostic; DeFiRegimeNet needs multi-token data across potentially multiple sources. Single interface. Already listed in existing repo STACK.md.

**Rate limits:** CoinGecko public plan: 5-15 calls/min. Binance REST public: 1200 requests/min weight limit (OHLCV uses weight=2, so ~600 calls/min effectively). For a one-time data pull of 10 tokens × 3 years of daily data, Binance public is well within limits.

**Same offline-first rule applies:** Deterministic synthetic crypto OHLCV generator is the default; ccxt/pycoingecko are optional real-data paths.

**Confidence: MEDIUM** — ccxt public endpoint confirmed no-key-required for read-only market data; CoinGecko rate limits from official support docs.

---

### Project-Specific Libraries

#### QBacktest (event-driven engine)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| pandas | 2.3.2 | TimeIndex-based portfolio ledger, OHLCV data frames | Already present; `.copy()` discipline avoids SettingWithCopy warnings on 2.x |
| numpy | 1.26.4 | Array math for returns, slippage, position sizing | Already present |
| dataclasses / typing | stdlib | Typed event objects (MarketEvent, SignalEvent, etc.) | Zero-dependency approach; pydantic is overkill for internal event bus |

**Do NOT use:** `backtrader`, `zipline-reloaded` — these are competing frameworks; QBacktest replaces them for this project. Using them would defeat the purpose of building the engine.

#### AlphaRank (ML cross-sectional ranking)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| lightgbm | 4.6.0 | Gradient boosting ranker | Already installed; LightGBM's `rank` objective (`lambdarank`) is the correct formulation for cross-sectional ranking; outperforms regression on rank labels |
| xgboost | 3.0.5 | Gradient boosting (comparison baseline) | Already installed; use `rank:pairwise` objective for comparison |
| scikit-learn | 1.7.2 | Linear baselines (Ridge, ElasticNet), preprocessing, pipeline | Already installed |
| skfolio | >= 0.7.0 (pip install needed) | CombinatorialPurgedCV, risk-parity portfolio construction | **Critical:** This provides the only free, sklearn-compatible, pip-installable implementation of Combinatorial Purged Cross-Validation. mlfinlab (the original source) is now closed-source and requires a paid subscription — do NOT use. skfolio's `CombinatorialPurgedCV` is a drop-in sklearn CV splitter. |
| shap | >= 0.42 | SHAP feature importance for factor attribution | Already in existing codebase STACK.md |

**Purged CV implementation note:** The correct tool chain is:
- `skfolio.model_selection.CombinatorialPurgedCV` for combinatorial purged CV
- Build a simple `PurgedKFold` manually (10-20 lines) for simpler walk-forward purging — trivial to implement, avoids the full skfolio dependency if undesired

**Confidence: HIGH** — skfolio pip-installable confirmed; mlfinlab closed-source status confirmed via multiple sources.

#### MacroRegime (HMM/GMM regime allocation)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| hmmlearn | 0.3.3 | GaussianHMM, GMMHMM for regime detection | Already installed; 0.3.3 is numpy-2-compatible and Python 3.11 confirmed; use `GaussianHMM(n_components=3, covariance_type='full')` for 3-regime bull/bear/neutral |
| scikit-learn | 1.7.2 | GaussianMixture (GMM), StandardScaler, preprocessing | Already installed; `GaussianMixture` in sklearn is the GMM implementation — no separate library needed |
| fredapi | 0.5.2 | FRED macro data (ALFRED for point-in-time) | See Data Sources section |
| cvxpy | 1.7.2 | Regime-conditional mean-variance optimization, risk parity | Already installed |
| skfolio | >= 0.7.0 | Risk parity and MVO portfolio construction (optional enhancement) | Provides HRP, risk parity, MVO via sklearn-compatible estimators |

**Point-in-time handling:** Use `fredapi.Fred.get_series_all_releases()` (ALFRED endpoint) to get vintage data. This is critical — using real-time FRED data with historical timestamps introduces look-ahead bias because macro releases are revised and arrive with lags (e.g., GDP is released 30 days after quarter end, revised twice). Synthetic data generator must model this lag structure.

**Confidence: HIGH** — hmmlearn version confirmed via pip; sklearn GaussianMixture is documented API.

#### VolSurfaceLab (options vol surface + GARCH/HAR)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| arch | 7.2.0 (installed; 8.0.0 latest on PyPI) | GARCH, EGARCH, HAR mean model, realized vol forecasting | The authoritative Python ARCH library by Kevin Sheppard; `arch_model(returns, mean='HAR', lags=[1,5,22], vol='GARCH', p=1, q=1)` combines HAR mean with GARCH vol in one call; EGARCH via `vol='EGARCH'` |
| py-vollib-vectorized | 0.1.1 (pip install needed) | Vectorized implied vol solver across option chains | **Recommendation: use py-vollib-vectorized over scipy.brentq over QuantLib.** Rationale: (1) py-vollib uses Peter Jäckel's LetsBeRational — converges to machine precision in 1-2 iterations, far faster than Brent's method which may take 50+ iterations; (2) py-vollib-vectorized wraps it in numpy/pandas array operations, essential for solving IV across full option chains; (3) QuantLib IV solving (impliedVolatility method) is correct but verbose and not vectorized for array inputs. Note: py-vollib-vectorized 0.1.1 was last released 2021 but works with Python 3.11 and is still the standard community choice. |
| scipy | 1.16.1 | SVI surface fitting (scipy.optimize.minimize with SLSQP/differential_evolution) | SVI has no dedicated PyPI library with active maintenance and arbitrage-free constraints; implement SVI calibration directly using scipy constrained optimization — this is what industry practitioners do. The SVI parametric form is: `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))` with 5 params `(a, b, rho, m, sigma)`. Use `differential_evolution` for global search, then `minimize(method='SLSQP')` for local refinement with butterfly/calendar arbitrage constraints. |
| QuantLib | 1.39 | Option Greeks (delta, gamma, vega, theta), Black-Scholes pricing | Already installed; `QuantLib.BlackScholesProcess`, `QuantLib.EuropeanOption` for production-quality Greeks computation alongside the vol surface. |
| matplotlib | 3.10.6 | Static surface plots, smile/skew charts | See Reporting section |
| plotly | 6.3.0 | Interactive 3D vol surface visualization | See Reporting section |

**IV solver decision rationale:**
- `scipy.optimize.brentq`: correct, but scalar-only — requires a Python loop over each option; slow for chains of 100+ options; acceptable for learning/reference implementation only
- `py_vollib` (LetsBeRational): fastest, 1-2 iterations to machine precision, scalar interface
- `py-vollib-vectorized`: same LetsBeRational algorithm but accepts numpy arrays — the right choice for calibrating a full surface
- `QuantLib.impliedVolatility`: accurate and industry-standard for single-option pricing workflows; not designed for vectorized chain solving

**Confidence: HIGH** — arch HAR support confirmed from official arch docs; py-vollib LetsBeRational algorithm confirmed from vollib.org; QuantLib.impliedVolatility is standard documented API.

#### DeFiRegimeNet (hybrid ML + econometric crypto regimes)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| hmmlearn | 0.3.3 | HMM regime detection on crypto return/vol features | Same as MacroRegime |
| xgboost | 3.0.5 | ML regime classifier (XGBoost with `multi:softprob` objective) | Already installed; XGBoost on tabular features (returns, vol, correlation, on-chain metrics) is the standard ML baseline for regime classification |
| arch | 7.2.0 | GARCH vol forecasting for each token | Already installed |
| scikit-learn | 1.7.2 | GaussianMixture, StandardScaler, confusion matrix, classification report | Already installed |
| skfolio | >= 0.7.0 | CombinatorialPurgedCV with embargo for crypto time series | Embargo is especially important for crypto: correlated assets mean information leaks across tokens without it |
| ccxt | existing | Binance public OHLCV data fetching | See Data Sources section |

---

### Reporting: Static vs Interactive

**Decision: use both, for different purposes.**

| Output Type | Library | When |
|-------------|---------|------|
| Static PNG/PDF charts in research reports | matplotlib + seaborn | Publication-quality figures embedded in README/report; unambiguous rendering on any viewer; use `matplotlib.figure.Figure.savefig(dpi=150)` |
| Interactive HTML report | plotly | 3D vol surface visualization (VolSurfaceLab), equity curve dashboards, regime-shaded return charts; `plotly.io.write_html(fig, "report.html", include_plotlyjs='cdn')` produces standalone self-contained HTML |
| Tearsheet-style summary | quantstats OR manual matplotlib | **Recommendation: implement a lightweight custom tearsheet with matplotlib** (Sharpe, max drawdown, rolling vol, drawdown chart, monthly return heatmap) rather than pulling in quantstats or pyfolio-reloaded. Reason: `pyfolio` (original Quantopian) is abandoned; `pyfolio-reloaded` works but is heavy and couples you to specific return series formats; `quantstats` adds a dependency for functionality that is 50-100 lines to implement. Custom tearsheet = zero extra dependency + full control. |

**Confidence: HIGH** — plotly HTML standalone output is documented API; pyfolio abandonment confirmed via GitHub; quantstats is a valid alternative if tearsheet speed matters over dependency minimalism.

---

## Installation

```bash
# New installs required (all else already present)
pip install fredapi>=0.5.2
pip install skfolio>=0.7.0
pip install py-vollib-vectorized>=0.1.1
pip install seaborn>=0.13.0

# QBacktest editable install (run from each consumer project)
pip install -e ../qbacktest/

# Optional: upgrade arch to 8.0.0 if HAR features needed beyond 7.2.0
# pip install --upgrade arch
# Note: 7.2.0 already supports mean='HAR' — upgrade only if a specific 8.x feature is needed
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Package build backend | hatchling | setuptools | setuptools requires more boilerplate (setup.cfg or setup.py alongside pyproject.toml); hatchling is simpler for pure-Python packages with no C extensions |
| Package build backend | hatchling | uv_build | uv_build is excellent but assumes uv as the package manager; this repo uses pip; no benefit without switching the full toolchain |
| Packaging approach | pyproject.toml + src layout | sys.path hacks | sys.path hacks cannot be imported by other packages; fails the "pip-installable" requirement |
| Implied vol solver | py-vollib-vectorized | scipy.optimize.brentq | brentq is scalar-only and takes 50+ iterations; not suitable for vectorized chain calibration |
| Implied vol solver | py-vollib-vectorized | QuantLib.impliedVolatility | QuantLib IV is accurate but not vectorized; verbose setup per option; use QuantLib for Greeks, py-vollib-vectorized for IV solving |
| Purged CV | skfolio.CombinatorialPurgedCV | mlfinlab | mlfinlab is closed-source / subscription-only since 2020; not pip-installable from PyPI without paid key |
| Macro data | fredapi + synthetic fallback | pandas-datareader FRED | pandas-datareader FRED support also requires API key (calls FRED API under the hood); no advantage over fredapi directly |
| Macro data | fredapi | FRED unauthenticated CSV | FRED enforced API key on all endpoints in November 2025; unauthenticated access no longer works |
| SVI fitting | scipy.optimize (custom) | py_vollib for SVI | py_vollib implements Black-Scholes IV only, not SVI surface parametrization; SVI must be implemented directly |
| Crypto data | ccxt (Binance public) | pycoingecko | Both are viable; ccxt is preferred because it is exchange-agnostic and already in the existing repo; pycoingecko is a good fallback with its 5-15/min free tier |
| Tearsheet | custom matplotlib | pyfolio-reloaded | pyfolio-reloaded works but is heavyweight and tightly coupled to specific input formats; custom is 50-100 lines and has no version drift risk |
| Tearsheet | custom matplotlib | quantstats | quantstats is maintained but adds dependency weight for functionality trivially reproduced; use if tearsheet speed/completeness is prioritized over minimalism |
| HAR model | arch (mean='HAR') | statsmodels OLS | statsmodels OLS can fit HAR as a regression but does not combine it with GARCH vol estimation in one model; arch's HAR + GARCH combination is the correct tool for joint mean/vol modeling |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `zipline-reloaded` | Competing backtesting framework; defeats purpose of building QBacktest; complex data bundle system; slow to set up | QBacktest (build it) |
| `backtrader` | Same as zipline; not pip-installable as a dependency of another library cleanly; event loop is not extensible for custom execution models | QBacktest (build it) |
| `mlfinlab` (from hudson-and-thames) | Closed-source since 2020; requires paid subscription and custom pip install with hash key; not installable from PyPI | `skfolio.model_selection.CombinatorialPurgedCV` |
| `scipy.optimize.brentq` as primary IV solver | Scalar-only; too slow for full option chain calibration (100+ options per expiry × multiple expiries) | `py-vollib-vectorized` |
| `pyfolio` (original Quantopian) | Abandoned in 2020; broken with modern pandas | Custom matplotlib tearsheet or `pyfolio-reloaded` |
| `alphalens` (original Quantopian) | Abandoned; broken with modern pandas | Custom IC/rank-IC analytics (100-200 lines) using scipy.stats.spearmanr |
| `alphalens-reloaded` | Maintained but opinionated about input format; for a portfolio showcase it is better to show you can compute IC/IC-decay/turnover yourself | Custom implementation |
| `vectorbt` | Vectorized (not event-driven); fundamentally different paradigm; not compatible with QBacktest's design goals | QBacktest |
| Paid APIs (Polygon.io, Alpha Vantage paid, Refinitiv) | Violates repo constraint: every pipeline must run on free data | yfinance (optional), synthetic generators (default) |
| `torch` / `tensorflow` for these projects | None of the 5 projects needs deep learning; adds heavy install weight; XGBoost/LightGBM cover the ML requirements | lightgbm, xgboost, scikit-learn |
| `FRED unauthenticated CSV scraping` | Broke in November 2025 when FRED enforced API keys on all endpoints | `fredapi` (free key) + synthetic macro generator |

---

## Stack Patterns by Project

**QBacktest (event engine as library):**
- Mandatory: src layout + pyproject.toml + hatchling
- `pip install -e .` for development; `pip install -e path/to/qbacktest` for consumers
- Zero external dependencies in the engine core beyond pandas + numpy
- Optional extras for reporting: `[extras]` in pyproject.toml for plotly/matplotlib

**AlphaRank (ML ranking):**
- LightGBM `rank` objective (lambdarank) as primary model
- `skfolio.CombinatorialPurgedCV` with `purge_gap=5` (business days) and `embargo_ratio=0.01`
- Custom IC/rank-IC analytics rather than alphalens
- Uses QBacktest for portfolio simulation of long-short decile strategy

**MacroRegime (regime allocation):**
- `hmmlearn.GaussianHMM` with n_components=2 or 3, covariance_type='full'
- `sklearn.mixture.GaussianMixture` as the GMM alternative
- `fredapi` with ALFRED vintage data for point-in-time correctness
- `cvxpy` for regime-conditional MVO; risk parity via equal-risk-contribution (implementable in 30 lines with scipy.optimize or skfolio)
- Walk-forward validation through QBacktest

**VolSurfaceLab (options vol):**
- `py-vollib-vectorized` for IV solving across chains
- `scipy.optimize.differential_evolution` + `minimize(SLSQP)` for SVI surface calibration
- `arch_model(mean='HAR', lags=[1,5,22], vol='GARCH')` for realized vol forecasting
- `QuantLib` for Greeks (delta, gamma, vega) using calibrated vol surface
- `plotly` 3D surface plots; `matplotlib` for static smile/skew charts

**DeFiRegimeNet (crypto regime detection):**
- `ccxt.binance()` for Binance public OHLCV; `pycoingecko` as fallback
- `hmmlearn.GaussianHMM` for unsupervised regime detection
- `xgboost` with `multi:softprob` for supervised classification
- `arch` GARCH for per-token vol forecasting
- `skfolio.CombinatorialPurgedCV` with embargo (use embargo_ratio >= 0.05 for crypto; correlation structure means standard 1-day embargo is insufficient)

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| hmmlearn 0.3.3 | numpy 1.26.4, Python 3.11 | 0.3.3 added numpy-2 compatible wheels; 1.26.4 confirmed compatible |
| arch 7.2.0 | numpy 1.26.4, scipy 1.16.1 | arch 8.0.0 is on PyPI (released Oct 2025); upgrade optional — 7.2.0 already has HAR support |
| skfolio >= 0.7 | scikit-learn 1.7.2 | skfolio follows sklearn API; 1.7.2 is compatible; skfolio 1.0.0 planned stable release in 2025 (may already be released) |
| py-vollib-vectorized 0.1.1 | Python 3.11, numpy 1.26.4 | Last release 2021 but depends on py_lets_be_rational and numba; numba supports Python 3.11; install and test before committing to this version |
| cvxpy 1.7.2 | numpy 1.26.4, scipy 1.16.1 | Clarabel solver is the default and bundled; no extra solver install needed for MVO/risk-parity |
| QuantLib 1.39 | Python 3.11 | 1.39 is a 2025 release; confirmed Python 3.11 support |
| pandas-datareader 0.10.0 | pandas 2.3.2 | 0.10.0 is old (2021); pandas 2.x compatibility is fragile; use only for non-FRED data if needed, prefer fredapi directly |

---

## Research Integrity Constraints Affecting Stack Choices

These constraints (from PROJECT.md) directly force specific stack decisions:

1. **Offline-first / synthetic data mandatory:** Every data source (yfinance, fredapi, ccxt) must be wrapped in a try/except with deterministic synthetic generator fallback. The synthetic generators must be seeded (`np.random.default_rng(seed=42)`) for reproducibility.

2. **No look-ahead bias:** `skfolio.CombinatorialPurgedCV` (or manual PurgedKFold) is mandatory for AlphaRank and DeFiRegimeNet. Walk-forward evaluation in QBacktest (expanding window, not full-sample) for MacroRegime and VolSurfaceLab.

3. **Transaction costs in every backtest:** QBacktest must implement proportional commission + half-spread slippage model. All portfolio simulations must pass `commission_bps` and `slippage_bps` parameters.

4. **Classical baselines before ML:** MacroRegime must benchmark HMM/GMM against simple threshold rules (e.g., 12-month momentum > 0 = bull). AlphaRank must benchmark LightGBM against linear IC-weighted composite factor. This is a research design requirement, not a stack requirement, but it constrains that all models must produce comparable output formats.

---

## Sources

- Python Packaging User Guide, src layout vs flat layout — https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/ — HIGH confidence
- Python Packaging User Guide, writing pyproject.toml — https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ — HIGH confidence
- FRED API documentation — https://fred.stlouisfed.org/docs/api/fred/ — HIGH confidence (API key required; unauthenticated access removed Nov 2025)
- fredapi PyPI — https://pypi.org/project/fredapi/ — HIGH confidence (version 0.5.2 confirmed)
- hmmlearn 0.3.3 PyPI — https://pypi.org/project/hmmlearn/0.3.3/ — HIGH confidence (Python 3.11 wheels confirmed)
- arch PyPI — https://pypi.org/project/arch/ — HIGH confidence (7.2.0 installed; 8.0.0 on PyPI as of Oct 2025)
- arch documentation, HAR model — https://arch.readthedocs.io/en/stable/univariate/introduction.html — HIGH confidence
- vollib / py-vollib, LetsBeRational — https://vollib.org/ and https://github.com/vollib/py_vollib — HIGH confidence
- py-vollib-vectorized PyPI — https://pypi.org/project/py-vollib-vectorized/ — MEDIUM confidence (last release 2021; Python 3.11 compatibility inferred from dependency chain)
- skfolio installation and CombinatorialPurgedCV — https://skfolio.org/user_guide/install.html and https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html — HIGH confidence
- mlfinlab closed-source status — https://github.com/hudson-and-thames/mlfinlab — HIGH confidence (confirmed subscription-only)
- CoinGecko rate limits — https://support.coingecko.com/hc/en-us/articles/4538771776153 — MEDIUM confidence
- ccxt public endpoints — https://docs.ccxt.com/ — MEDIUM confidence (no-key confirmed for read-only market data)
- yfinance rate limiting issues — multiple Medium articles 2025-2026 — MEDIUM confidence
- Build backends comparison 2025 — https://medium.com/@dynamicy/python-build-backends-in-2025 — MEDIUM confidence

---
*Stack research for: Five flagship quant portfolio projects (QBacktest, AlphaRank, MacroRegime, VolSurfaceLab, DeFiRegimeNet)*
*Researched: 2026-06-10*
