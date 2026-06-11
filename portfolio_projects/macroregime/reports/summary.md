# MacroRegime Pipeline Summary

## Strategy Comparison

> **Label-alignment rule**: Regime states are ordered ascending by the
> economic observable (observable_dim). State 0 = lowest value
> (contraction/recession/stress); state K-1 = highest (expansion/risk-on).
> Implementation: `permutation = np.argsort(np.argsort(means[:, dim]))`
> (double argsort maps raw->rank; single argsort gives rank->raw).

Net Sharpe CIs are 95% bootstrap (1000 resamples, percentile method).

| Strategy | Gross Sharpe | Net Sharpe | Net CI Low | Net CI High | Sortino | MaxDD | Turnover |
|----------|-------------|-----------|-----------|------------|---------|-------|----------|
| Regime | 0.235 | 0.226 | -0.418 | 0.862 | 0.334 | -0.404 | 1.094 |
| 60/40 | 0.023 | 0.022 | -0.594 | 0.656 | 0.032 | -0.298 | 0.108 |
| EqualWeight | 0.087 | 0.086 | -0.529 | 0.649 | 0.136 | -0.291 | 0.097 |
| RiskParity | 1.004 | 0.981 | 0.335 | 1.661 | 1.120 | -0.027 | 0.209 |

## Regime Stability & K Sensitivity

### HMM vs GMM Stability

- **HMM/GMM agreement (daily, aligned)**: 86.0%
- **Distribution drift (L1, first vs second half)**: 0.8141

**Market dwell times (mean bars per regime)**

| State | HMM | GMM |
|-------|-----|-----|
| R0 | 118.0 | 51.8 |
| R1 | 76.1 | 62.0 |
| R2 | 83.4 | 50.6 |

**Macro dwell times (mean observations per regime)**

| State | HMM | GMM |
|-------|-----|-----|
| R0 | 23.0 | 5.7 |
| R1 | 45.4 | 18.4 |
| R2 | 17.0 | 10.8 |

### K Sensitivity

> K was **not** selected by Sharpe (anti-feature: selecting K to maximize Sharpe overfits the regime model to the backtest period, invalidating the research hypothesis). K=3 is the default economic choice (contraction, neutral, expansion). Use BIC or dwell-time interpretability to select K.

| K | Mean Dwell R0 | Mean Dwell R1 | Mean Dwell R2+ | Agreement vs K=3 |
|---|--------------|--------------|---------------|-----------------|
| 2 | 113.4 | 124.3 | 0.0 | 59.6% |
| 3 | 59.7 | 63.9 | 105.5 | 100.0% |

