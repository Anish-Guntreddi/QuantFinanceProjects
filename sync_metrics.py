"""Sync metrics from project-local notebooks/results.json
into portfolio_app/data/results/ and portfolio_app/data/cards/.

Usage:
    python sync_metrics.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_DATA = REPO_ROOT / "portfolio_app" / "data"


def sync():
    manifest_path = APP_DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    synced, skipped = 0, 0

    for project in manifest["projects"]:
        pid = project["id"]
        card_path = APP_DATA / project["card_path"]
        results_path = APP_DATA / project["results_path"]

        # Load card to get notebook_path
        card = json.loads(card_path.read_text(encoding="utf-8"))
        notebook_dir = card.get("notebook_path", "")
        if not notebook_dir:
            print(f"  SKIP {pid}: no notebook_path in card")
            skipped += 1
            continue

        # Find local results.json
        local_results_path = REPO_ROOT / notebook_dir / "results.json"
        if not local_results_path.exists():
            print(f"  SKIP {pid}: no notebooks/results.json")
            skipped += 1
            continue

        local_data = json.loads(local_results_path.read_text(encoding="utf-8"))
        local_metrics = local_data.get("metrics", {})
        if not local_metrics:
            print(f"  SKIP {pid}: empty metrics in notebooks/results.json")
            skipped += 1
            continue

        # Update portfolio_app results JSON
        app_results = json.loads(results_path.read_text(encoding="utf-8"))
        old_sharpe = app_results.get("metrics", {}).get("sharpe_ratio")
        app_results["metrics"] = local_metrics

        # Also sync equity_curve, monthly_returns, etc. if present
        for key in ("equity_curve", "monthly_returns", "parameter_sensitivity",
                     "backtest_period"):
            if key in local_data:
                app_results[key] = local_data[key]

        results_path.write_text(json.dumps(app_results, indent=2, ensure_ascii=False), encoding="utf-8")

        # Update card headline_metric
        new_sharpe = local_metrics.get("sharpe_ratio")
        if new_sharpe is not None and "headline_metric" in card:
            card["headline_metric"]["value"] = round(new_sharpe, 4)
            card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  SYNC {pid}: Sharpe {old_sharpe} -> {new_sharpe}")
        synced += 1

    print(f"\nDone: {synced} synced, {skipped} skipped")


if __name__ == "__main__":
    sync()
