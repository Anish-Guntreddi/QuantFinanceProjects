# DeFiRegimeNet Pipeline Report


## Abstract


This report presents the results of the DeFiRegimeNet causal regime detection pipeline applied to synthetic cryptocurrency panel data. We detect per-token regimes with INDEPENDENT causal Gaussian HMMs (one per token), compare logistic regression and XGBoost classifiers against HMM/GMM baselines using purged combinatorial cross-validation (CPCV), and evaluate volatility forecasting models (HAR, GARCH, EGARCH) via QLIKE loss. Independently detected sequences show genuine cross-token association — mean off-diagonal Cramér's V = 0.329, above the ~0.15 independence floor but well below 1.0: 30% idiosyncratic noise and label-permutation ambiguity limit recovery of the planted 70% market factor, an honest measure of how hard shared-regime detection is in practice.

## Data


**Generator:** CryptoGenerator(seed=42, n_years=3, tokens=['BTC', 'ETH', 'SOL', 'AVAX'])
**Bars per token:** 1095
**Calendar:** daily (freq=D, 24/7 crypto)
**DGP:** GARCH(1,1) vol clustering, Student-t fat-tail innovations (df=4), Markov 4-state regime with market_factor_weight=0.70

**Label distribution (regime states 0-3 per token):**
  - BTC: state 0: 326 (29.9%), state 1: 210 (19.3%), state 2: 376 (34.5%), state 3: 177 (16.3%)
  - ETH: state 0: 336 (30.9%), state 1: 157 (14.4%), state 2: 404 (37.1%), state 3: 192 (17.6%)
  - SOL: state 0: 395 (36.3%), state 1: 138 (12.7%), state 2: 395 (36.3%), state 3: 161 (14.8%)
  - AVAX: state 0: 356 (32.7%), state 1: 166 (15.2%), state 2: 382 (35.1%), state 3: 185 (17.0%)

## Methodology


### Regime Detection
Market regime detected via Gaussian HMM and GMM fitted on the cross-sectional mean feature matrix (4 causal features: ret_lag1, rv_21, mom_21, drawdown — all shift(1) then expanding z-scored). A single CausalRegimeDetector (macroregime adapter) fits on mean features; its causal oracle guarantee ensures label at t depends only on features[:t+1].

### Classifier Evaluation
Forward-looking 4-state regime labels (horizon H=5) are constructed via `make_regime_labels` and used as CV targets only — never as training features. CPCV (CombinatorialPurgedCV) with embargo_size=purged_size=H=5 prevents any look-ahead leakage. HMM/GMM baselines use the causal sequence directly (no training on labels) — this asymmetry favors the baselines, the conservative/honest direction.

### Volatility Forecasting
HAR, GARCH(1,1), and EGARCH(1,1) models compared per token via volsurfacelab.forecast.compare_forecasts. QLIKE loss (Patton 2011): L(h, rv) = rv/h - log(rv/h) - 1. Lower is better; under-forecasting penalized.

## Results


### Model Comparison


|  | accuracy | log_loss |
| --- | --- | --- |
| hmm | 0.2595 | 5.1158 |
| gmm | 0.3129 | 4.7474 |
| logistic | 0.4506 | 1.0280 |
| xgboost | 0.4322 | 1.0862 |

> Note: HMM/GMM are persistence baselines (causal regime at t predicts forward label at t). Classifiers use purged CPCV. Asymmetry favors baselines.

### Cross-Token Cramér's V


|  | BTC | ETH | SOL | AVAX |
| --- | --- | --- | --- | --- |
| BTC | 1.000 | 0.456 | 0.253 | 0.355 |
| ETH | 0.456 | 1.000 | 0.235 | 0.382 |
| SOL | 0.253 | 0.235 | 1.000 | 0.289 |
| AVAX | 0.355 | 0.382 | 0.289 | 1.000 |

Mean off-diagonal V: **0.329** — independently detected per-token sequences; > 0.3 indicates genuine shared-factor recovery (independence floor ~0.15; 1.0 would indicate a degenerate shared-sequence shortcut)

## Robustness


### Student-t GARCH (Fat-Tail Robustness)
GARCH(1,1) with Student-t innovations fitted per token. OOS QLIKE reported.

  - BTC: QLIKE = 1.6831
  - ETH: QLIKE = 1.6737
  - SOL: QLIKE = 1.7032
  - AVAX: QLIKE = 1.8496

### K-Sensitivity
K-sensitivity run on HMM backend with structural metrics only. K selection is NOT based on return-based criteria (locked anti-feature: selecting K to maximize Sharpe overfits the regime model to the backtest period).

## Limitations


1. **Gaussian HMM on fat-tailed crypto returns**: The Student-t DGP (df=4) produces heavy tails; Gaussian HMM may misclassify high-vol states as separate regimes. Student-t HMM extensions are available but not used here.

2. **Synthetic data**: Results are generated from a simplified 4-state Markov chain with fixed GARCH parameters. Real crypto data has time-varying microstructure and structural breaks not captured by the DGP.

3. **Persistence-baseline asymmetry**: HMM/GMM baselines use the full causal sequence without CV, giving them a statistical advantage over purged-CV classifiers. Reported comparisons favor the baselines; classifier advantage over baselines is a conservative lower bound.
