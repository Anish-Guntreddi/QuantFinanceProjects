"""Flagship project simulations — five seeded pure-numpy engines.

Each function accepts keyword arguments matching the card's interactive_params
and returns the standard dispatcher shape:
    {"metrics": {...}, "equity_curve": {"dates": [...], "values": [...], ...}}

All seeded with np.random.default_rng(42), pure numpy, <1 s runtime each.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. QBacktest sim — GBM + MA crossover with T+1 fill vs. same-bar fill
# ---------------------------------------------------------------------------

def run_qbacktest_sim(
    ma_fast: int = 20,
    ma_slow: int = 50,
    cost_bps: float = 10.0,
    n_bars: int = 504,
    seed: int = 42,
) -> dict:
    """GBM price path with MA crossover.  T+1-fill equity (honest) vs
    same-bar-fill equity (cheating benchmark) — the Sharpe gap quantifies
    look-ahead bias if you got it wrong.
    """
    rng = np.random.default_rng(seed)
    mu = 0.0002
    sigma = 0.015

    log_rets = rng.normal(mu, sigma, n_bars)
    price = 100.0 * np.exp(np.cumsum(log_rets))

    # Moving averages
    fast_sma = np.full(n_bars, np.nan)
    slow_sma = np.full(n_bars, np.nan)
    for i in range(n_bars):
        if i >= ma_fast - 1:
            fast_sma[i] = price[i - ma_fast + 1: i + 1].mean()
        if i >= ma_slow - 1:
            slow_sma[i] = price[i - ma_slow + 1: i + 1].mean()

    cost_per_trade = cost_bps / 10000.0

    # T+1 fill equity (honest): signal on bar t fills at bar t+1 open (≈ close[t])
    # We approximate T+1 open = close[t] (GBM has no gap model here)
    t1_equity = np.ones(n_bars)
    sb_equity = np.ones(n_bars)  # same-bar fill (cheating)
    position = 0
    sb_position = 0
    cost_acc = 0.0
    sb_cost_acc = 0.0

    pending_signal = 0  # T+1 fill: pending signal from previous bar

    for t in range(1, n_bars):
        if np.isnan(fast_sma[t]) or np.isnan(slow_sma[t]):
            t1_equity[t] = t1_equity[t - 1]
            sb_equity[t] = sb_equity[t - 1]
            continue

        # --- T+1 fill: apply pending signal from previous bar ---
        if pending_signal != 0 and pending_signal != position:
            cost_acc += cost_per_trade
            position = pending_signal

        # --- Same-bar fill: apply signal immediately (cheating) ---
        sb_signal = 1 if fast_sma[t] > slow_sma[t] else (
            -1 if fast_sma[t] < slow_sma[t] else sb_position
        )
        if sb_signal != sb_position:
            sb_cost_acc += cost_per_trade
            sb_position = sb_signal

        # --- Generate signal for T+1 ---
        pending_signal = 1 if fast_sma[t] > slow_sma[t] else (
            -1 if fast_sma[t] < slow_sma[t] else position
        )

        # Mark-to-market: 1-unit position
        ret = log_rets[t]
        t1_equity[t] = t1_equity[t - 1] * (1 + position * ret - cost_per_trade * abs(position != position))
        sb_equity[t] = sb_equity[t - 1] * (1 + sb_position * ret - cost_per_trade * abs(sb_position != sb_position))

    # Recompute properly using position series
    t1_equity = _compute_equity(position_series(fast_sma, slow_sma, n_bars, fill_mode="t1"), log_rets, cost_per_trade)
    sb_equity_arr = _compute_equity(position_series(fast_sma, slow_sma, n_bars, fill_mode="sb"), log_rets, cost_per_trade)

    # Dates
    import datetime
    start = datetime.date(2022, 1, 3)
    bdays = _bdays(start, n_bars)
    dates = [str(d) for d in bdays]

    # Downsample to 250 pts
    idx = np.round(np.linspace(0, n_bars - 1, min(250, n_bars))).astype(int)
    d_dates = [dates[i] for i in idx]
    d_t1 = [round(float(t1_equity[i]), 6) for i in idx]
    d_sb = [round(float(sb_equity_arr[i]), 6) for i in idx]

    # Metrics
    rets_t1 = np.diff(t1_equity) / t1_equity[:-1]
    rets_sb = np.diff(sb_equity_arr) / sb_equity_arr[:-1]
    sharpe_t1 = _sharpe(rets_t1)
    sharpe_sb = _sharpe(rets_sb)

    return {
        "metrics": {
            "sharpe_t1": round(sharpe_t1, 4),
            "sharpe_samebar": round(sharpe_sb, 4),
            "lookahead_sharpe_gap": round(sharpe_sb - sharpe_t1, 4),
            "total_return": round(float(t1_equity[-1]) - 1.0, 4),
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "cost_bps": cost_bps,
        },
        "equity_curve": {
            "dates": d_dates,
            "values": d_t1,
            "benchmark_values": d_sb,
        },
    }


def _bdays(start, n):
    import datetime
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def position_series(fast_sma, slow_sma, n_bars, fill_mode="t1"):
    pos = np.zeros(n_bars)
    pending = 0
    for t in range(1, n_bars):
        if np.isnan(fast_sma[t]) or np.isnan(slow_sma[t]):
            pos[t] = pos[t - 1]
            continue
        signal = 1.0 if fast_sma[t] > slow_sma[t] else (-1.0 if fast_sma[t] < slow_sma[t] else pos[t - 1])
        if fill_mode == "t1":
            pos[t] = pending
            pending = signal
        else:
            pos[t] = signal
    return pos


def _compute_equity(positions, log_rets, cost_per_trade):
    eq = np.ones(len(log_rets))
    for t in range(1, len(log_rets)):
        trade_cost = cost_per_trade if positions[t] != positions[t - 1] else 0.0
        eq[t] = eq[t - 1] * (1 + positions[t] * log_rets[t]) - trade_cost * eq[t - 1]
    return eq


def _sharpe(rets, ann=252):
    if len(rets) == 0 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(ann))


# ---------------------------------------------------------------------------
# 2. AlphaRank sim — planted-IC cross-section, decile long-short
# ---------------------------------------------------------------------------

def run_alpharank_sim(
    ic_strength: float = 0.06,
    n_assets: int = 50,
    n_months: int = 36,
    seed: int = 42,
) -> dict:
    """Planted-IC cross-section.  Decile long-short monthly equity curve."""
    rng = np.random.default_rng(seed)

    ic_strength = float(np.clip(ic_strength, 0.0, 0.3))
    n_assets = max(10, int(n_assets))
    n_months = max(6, int(n_months))

    monthly_vol = 0.04
    noise_sigma = monthly_vol

    # Planted alpha factor (momentum-like)
    factor_loading = rng.standard_normal(n_assets)
    ic_series = []
    equity = np.ones(n_months)

    for m in range(n_months):
        alpha = ic_strength * noise_sigma / max(np.sqrt(1 - ic_strength ** 2), 1e-6)
        rets = alpha * factor_loading + noise_sigma * rng.standard_normal(n_assets)
        scores = factor_loading + 0.3 * rng.standard_normal(n_assets)  # noisy signal

        # Rank correlation (IC)
        ranks_scores = np.argsort(np.argsort(scores))
        ranks_rets = np.argsort(np.argsort(rets))
        n = len(rets)
        ic = float(np.corrcoef(ranks_scores, ranks_rets)[0, 1])
        ic_series.append(ic)

        # Decile spread: top vs bottom decile
        n_decile = max(1, n_assets // 10)
        sorted_idx = np.argsort(scores)
        top_ret = rets[sorted_idx[-n_decile:]].mean()
        bot_ret = rets[sorted_idx[:n_decile]].mean()
        spread = top_ret - bot_ret
        cost = 0.001  # 10 bps round trip per month
        equity[m] = equity[m - 1] * (1 + spread - cost) if m > 0 else 1.0 + spread - cost

    ic_arr = np.array(ic_series)
    realized_ic = float(ic_arr.mean())
    icir = float(realized_ic / max(ic_arr.std(), 1e-9))

    # Monthly dates
    import datetime
    start = datetime.date(2022, 1, 31)
    dates = []
    d = start
    for _ in range(n_months):
        dates.append(str(d))
        # advance to next month-end
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1, day=31)
        else:
            import calendar
            last_day = calendar.monthrange(d.year, d.month + 1)[1]
            d = d.replace(month=d.month + 1, day=last_day)

    decile_spread_ann = realized_ic * np.sqrt(12)

    return {
        "metrics": {
            "realized_ic": round(realized_ic, 4),
            "icir": round(icir, 4),
            "decile_spread_ann": round(decile_spread_ann, 4),
            "ic_strength": ic_strength,
            "n_assets": n_assets,
        },
        "equity_curve": {
            "dates": dates,
            "values": [round(float(v), 6) for v in equity],
        },
    }


# ---------------------------------------------------------------------------
# 3. MacroRegime sim — 3-state sticky Markov + T+1 weight lag
# ---------------------------------------------------------------------------

def run_macroregime_sim(
    stay_prob: float = 0.92,
    risk_weight_expansion: float = 0.60,
    n_bars: int = 2520,
    seed: int = 42,
) -> dict:
    """3-state sticky Markov regime market.  Regime strategy (T+1 weight lag)
    vs buy-and-hold benchmark.
    """
    rng = np.random.default_rng(seed)
    stay_prob = float(np.clip(stay_prob, 0.70, 0.99))
    risk_weight_expansion = float(np.clip(risk_weight_expansion, 0.3, 0.9))

    # 3 regimes: contraction (0), neutral (1), expansion (2)
    regime_means = [-0.0008, 0.0003, 0.0010]
    regime_vols = [0.022, 0.012, 0.008]

    # Transition matrix (sticky)
    flip_prob = (1 - stay_prob) / 2
    T_mat = np.array([
        [stay_prob, flip_prob, flip_prob],
        [flip_prob, stay_prob, flip_prob],
        [flip_prob, flip_prob, stay_prob],
    ])

    # Simulate regimes
    regimes = np.zeros(n_bars, dtype=int)
    regimes[0] = 1  # start in neutral
    for t in range(1, n_bars):
        regimes[t] = rng.choice(3, p=T_mat[regimes[t - 1]])

    # Asset returns
    asset_rets = np.array([
        rng.normal(regime_means[r], regime_vols[r]) for r in regimes
    ])

    # Regime weights (T+1 lag: weight on bar t is based on regime at t-1)
    weights = np.ones(n_bars) * 0.5  # start at neutral
    regime_weights = {0: 0.20, 1: 0.50, 2: risk_weight_expansion}
    for t in range(1, n_bars):
        weights[t] = regime_weights[regimes[t - 1]]  # T+1 lag

    # Compute equity curves
    strategy_equity = np.ones(n_bars)
    bh_equity = np.ones(n_bars)
    for t in range(1, n_bars):
        strategy_equity[t] = strategy_equity[t - 1] * (1 + weights[t] * asset_rets[t])
        bh_equity[t] = bh_equity[t - 1] * (1 + 0.6 * asset_rets[t])

    # Downsample to 250 pts
    import datetime
    start = datetime.date(2000, 1, 3)
    bdays = _bdays(start, n_bars)
    dates_all = [str(d) for d in bdays]
    idx = np.round(np.linspace(0, n_bars - 1, 250)).astype(int)
    dates = [dates_all[i] for i in idx]
    values = [round(float(strategy_equity[i]), 6) for i in idx]
    bench = [round(float(bh_equity[i]), 6) for i in idx]

    # Metrics
    rets_s = np.diff(strategy_equity) / strategy_equity[:-1]
    sharpe = _sharpe(rets_s)
    dd = strategy_equity / np.maximum.accumulate(strategy_equity) - 1
    max_dd = float(dd.min())

    # Regime switches
    switches = int(np.sum(np.diff(regimes) != 0))
    avg_dwell = n_bars / max(switches + 1, 1)

    return {
        "metrics": {
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "n_regime_switches": switches,
            "avg_dwell_bars": round(avg_dwell, 1),
            "stay_prob": stay_prob,
            "equity_weight_expansion": risk_weight_expansion,
        },
        "equity_curve": {
            "dates": dates,
            "values": values,
            "benchmark_values": bench,
        },
    }


# ---------------------------------------------------------------------------
# 4. VolSurfaceLab sim — SVI smile + butterfly g(k) check
# ---------------------------------------------------------------------------

def run_volsurfacelab_sim(
    svi_b: float = 0.08,
    svi_rho: float = -0.30,
    svi_sigma: float = 0.30,
    svi_a: float = 0.02,
    svi_m: float = 0.0,
    n_strikes: int = 100,
    seed: int = 42,
) -> dict:
    """SVI smile with no-arb butterfly check g(k)≥0.

    equity_curve.dates = k-grid as strings, values = implied vol at each k.
    The Strategy page plots this as a line — effectively the smile curve.
    Metrics include ATM IV and min butterfly g value.
    """
    svi_b = float(np.clip(svi_b, 0.001, 0.5))
    svi_rho = float(np.clip(svi_rho, -0.999, 0.999))
    svi_sigma = float(np.clip(svi_sigma, 0.01, 2.0))

    # k-grid: log-moneyness
    k = np.linspace(-1.5, 1.5, n_strikes)

    def w(k_arr):
        """SVI total variance."""
        disc = np.sqrt((k_arr - svi_m) ** 2 + svi_sigma ** 2)
        return svi_a + svi_b * (svi_rho * (k_arr - svi_m) + disc)

    w_vals = w(k)

    # Implied vol = sqrt(w / T), T=0.25
    T = 0.25
    iv_vals = np.sqrt(np.maximum(w_vals, 0) / T)

    # Butterfly g(k) via finite differences
    dk = k[1] - k[0]
    w_prime = np.gradient(w_vals, dk)
    w_dbl = np.gradient(w_prime, dk)

    g = (1 - k * w_prime / (2 * np.maximum(w_vals, 1e-12))) ** 2 \
        - (w_prime ** 2 / 4) * (1 / np.maximum(w_vals, 1e-12) + 0.25) \
        + w_dbl / 2

    min_g = float(g.min())
    arb_free = 1 if min_g >= 0 else 0

    # ATM IV
    atm_idx = np.argmin(np.abs(k))
    atm_iv = float(iv_vals[atm_idx])

    # Format k as strings for the "dates" axis
    k_strs = [f"{kv:.3f}" for kv in k]

    return {
        "metrics": {
            "atm_iv": round(atm_iv, 6),
            "min_butterfly_g": round(min_g, 6),
            "butterfly_arbitrage_free": arb_free,
            "svi_b": svi_b,
            "svi_rho": svi_rho,
            "svi_sigma": svi_sigma,
        },
        "equity_curve": {
            "dates": k_strs,
            "values": [round(float(v), 6) for v in iv_vals],
        },
    }


# ---------------------------------------------------------------------------
# 5. DeFiRegimeNet sim — multi-token fat-tail market
# ---------------------------------------------------------------------------

def run_defiregimenet_sim(
    t_dof: float = 4.0,
    market_factor_weight: float = 0.70,
    n_bars: int = 1095,
    seed: int = 42,
) -> dict:
    """Multi-token fat-tail market with shared regime factor.

    values = cumulative BTC-like token path; benchmark_values = ETH-like.
    """
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(seed)
    t_dof = float(np.clip(t_dof, 2.5, 30.0))
    market_factor_weight = float(np.clip(market_factor_weight, 0.0, 1.0))

    # Shared market factor (fat-tail)
    market_factor = scipy_stats.t.rvs(df=t_dof, size=n_bars, random_state=rng.integers(0, 2**31))
    market_factor = market_factor * 0.02 / np.sqrt(t_dof / (t_dof - 2))  # normalize to ~2% daily vol

    # Token-specific idiosyncratic returns
    btc_idio = scipy_stats.t.rvs(df=t_dof, size=n_bars, random_state=rng.integers(0, 2**31))
    btc_idio = btc_idio * 0.03 / np.sqrt(t_dof / (t_dof - 2))

    eth_idio = scipy_stats.t.rvs(df=t_dof, size=n_bars, random_state=rng.integers(0, 2**31))
    eth_idio = eth_idio * 0.025 / np.sqrt(t_dof / (t_dof - 2))

    btc_rets = market_factor_weight * market_factor + (1 - market_factor_weight) * btc_idio
    eth_rets = market_factor_weight * market_factor + (1 - market_factor_weight) * eth_idio

    btc_equity = np.cumprod(1 + btc_rets)
    eth_equity = np.cumprod(1 + eth_rets)

    # Metrics
    excess_kurtosis = float(scipy_stats.kurtosis(btc_rets))
    corr = float(np.corrcoef(btc_rets, eth_rets)[0, 1])
    ann_vol = float(btc_rets.std() * np.sqrt(365))

    # Dates
    import datetime
    start = datetime.date(2021, 1, 1)
    dates = [str(start + datetime.timedelta(days=i)) for i in range(n_bars)]

    idx = np.round(np.linspace(0, n_bars - 1, 250)).astype(int)
    d_dates = [dates[i] for i in idx]
    d_btc = [round(float(btc_equity[i]), 6) for i in idx]
    d_eth = [round(float(eth_equity[i]), 6) for i in idx]

    return {
        "metrics": {
            "excess_kurtosis_tok1": round(excess_kurtosis, 4),
            "cross_token_return_corr": round(corr, 4),
            "realized_vol_ann": round(ann_vol, 4),
            "t_dof": t_dof,
            "market_factor_weight": market_factor_weight,
        },
        "equity_curve": {
            "dates": d_dates,
            "values": d_btc,
            "benchmark_values": d_eth,
        },
    }
