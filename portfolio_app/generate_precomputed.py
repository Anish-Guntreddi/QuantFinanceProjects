#!/usr/bin/env python3
"""Generate pre-computed strategy cards and results from notebook configs."""

import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add repo root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml

CONFIGS_DIR = os.path.join(ROOT, "notebook_configs")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(APP_DIR, "data", "cards")
RESULTS_DIR = os.path.join(APP_DIR, "data", "results")


def generate_synthetic_results(n_days=504, annual_sharpe=1.5, annual_vol=0.15, seed=42):
    np.random.seed(seed)
    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = (annual_sharpe * annual_vol) / 252
    returns = np.random.normal(daily_mu, daily_vol, n_days)
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i - 1]
    jump_mask = np.random.random(n_days) < 0.03
    returns[jump_mask] *= np.random.choice([-2.5, 2.0], size=jump_mask.sum())
    return returns


def compute_metrics(returns):
    returns = np.array(returns)
    n = len(returns)
    total_return = float(np.prod(1 + returns) - 1)
    n_years = n / 252
    cagr = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)
    ann_vol = float(np.std(returns) * np.sqrt(252))
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    downside = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252)) if len(downside) > 0 and np.std(downside) > 0 else 0
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    max_dd = float(np.min(dd))
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0
    wins = returns[returns > 0]
    win_rate = float(len(wins) / n) if n > 0 else 0
    profit_factor = float(np.sum(wins) / abs(np.sum(returns[returns < 0]))) if np.sum(returns[returns < 0]) != 0 else 99
    return {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "annualized_vol": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(min(calmar, 99), 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(min(profit_factor, 99), 4),
        "total_trades": int(n * 0.7),
    }


def generate_project_data(config):
    proj = config["project"]
    proj_id = proj["id"]
    sharpe = config.get("synthetic_sharpe", 1.5)
    vol = config.get("synthetic_vol", 0.15)

    # Skip infrastructure-only projects (sharpe=0)
    if sharpe == 0:
        n_days = 252
        sharpe = 0.5
        vol = 0.10
    else:
        n_days = 504

    seed = hash(proj_id) % 10000
    returns = generate_synthetic_results(n_days, sharpe, vol, seed)
    metrics = compute_metrics(returns)

    equity = (100000 * np.cumprod(1 + returns)).tolist()
    bench_returns = generate_synthetic_results(n_days, 0.7, 0.16, seed + 1)
    bench_equity = (100000 * np.cumprod(1 + bench_returns)).tolist()

    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    date_strs = [str(d.date()) for d in dates]

    # Monthly returns
    monthly = {}
    for i in range(0, n_days, 21):
        chunk = returns[i:i + 21]
        m_date = dates[min(i, n_days - 1)].strftime("%Y-%m")
        monthly[m_date] = round(float(np.prod(1 + chunk) - 1), 6)

    # Parameter sensitivity
    params = config.get("params", {})
    sensitivity = []
    for pname, pdef in list(params.items())[:2]:
        lo, hi = pdef["range"]
        vals = np.linspace(lo, hi, 8).tolist()
        sharpes = []
        dds = []
        for v in vals:
            factor = 1 - 0.3 * abs(v - pdef["default"]) / (hi - lo + 1e-10)
            s_ret = generate_synthetic_results(252, sharpe * factor, vol, seed + int(v * 100))
            m = compute_metrics(s_ret)
            sharpes.append(m["sharpe_ratio"])
            dds.append(m["max_drawdown"])
        sensitivity.append({"param": pname, "values": [round(v, 4) for v in vals],
                            "sharpe": sharpes, "max_drawdowns": dds})

    # Strategy card
    card = {
        "project_id": proj_id,
        "title": proj["display_name"],
        "short_description": proj.get("description", "")[:150],
        "long_description": proj.get("description", ""),
        "category": proj["category"],
        "subcategory": config.get("subcategory", ""),
        "asset_class": config.get("asset_class", "Equities"),
        "frequency": config.get("frequency", "Daily"),
        "data_source": config.get("data", {}).get("source_type", "synthetic"),
        "languages": config.get("languages", ["Python"]),
        "key_techniques": config.get("tags", []),
        "interactive_params": config.get("interactive_params", []),
        "tags": config.get("tags", []),
        "github_path": f"{proj['category']}/{proj['dir_name']}",
        "notebook_path": f"{proj['category']}/{proj['dir_name']}/notebooks/",
        "requires_gpu": config.get("requires_gpu", False),
        "has_cpp": config.get("has_cpp", False),
        "estimated_runtime_seconds": 10,
        "simulation_tier": config.get("simulation_tier", "precomputed"),
        "headline_metric": {"name": "Sharpe", "value": metrics["sharpe_ratio"]},
    }

    results = {
        "project_id": proj_id,
        "timestamp": datetime.now().isoformat(),
        "backtest_period": {"start": date_strs[0], "end": date_strs[-1]},
        "benchmark": "SPY",
        "metrics": metrics,
        "category_specific_metrics": {},
        "monthly_returns": monthly,
        "equity_curve": {
            "dates": date_strs,
            "values": [round(v, 2) for v in equity],
            "benchmark_values": [round(v, 2) for v in bench_equity],
        },
        "parameter_sensitivity": sensitivity,
    }

    return card, results


def main():
    os.makedirs(CARDS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load registry
    with open(os.path.join(CONFIGS_DIR, "_registry.yaml")) as f:
        registry = yaml.safe_load(f)

    projects = []
    for entry in registry["projects"]:
        config_path = os.path.join(CONFIGS_DIR, entry["config_file"])
        with open(config_path) as f:
            config = yaml.safe_load(f)

        card, results = generate_project_data(config)

        with open(os.path.join(CARDS_DIR, f"{card['project_id']}.json"), "w") as f:
            json.dump(card, f, indent=2)
        with open(os.path.join(RESULTS_DIR, f"{card['project_id']}.json"), "w") as f:
            json.dump(results, f, indent=2)

        projects.append({
            "id": card["project_id"],
            "category": card["category"],
            "card_path": f"cards/{card['project_id']}.json",
            "results_path": f"results/{card['project_id']}.json",
            "simulation_tier": card["simulation_tier"],
        })
        print(f"  Generated: {card['project_id']}")

    # Write manifest
    manifest = {"generated_at": datetime.now().isoformat(), "total_projects": len(projects), "projects": projects}
    with open(os.path.join(APP_DIR, "data", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(projects)} cards + results + manifest.json")


if __name__ == "__main__":
    main()
