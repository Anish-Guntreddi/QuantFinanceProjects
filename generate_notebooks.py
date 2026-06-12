#!/usr/bin/env python3
"""Generate Jupyter notebooks for all projects from templates.

Usage:
    python generate_notebooks.py                              # Generate all
    python generate_notebooks.py --project hft_01_adaptive_market_making  # Single project
    python generate_notebooks.py --category HFT_strategy_projects         # One category
    python generate_notebooks.py --dry-run                    # Validate only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat

# Add repo root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from notebook_templates import TEMPLATE_MAP

DATA_DIR = REPO_ROOT / "portfolio_app" / "data"


def load_manifest() -> list[dict]:
    with open(DATA_DIR / "manifest.json") as f:
        return json.load(f)["projects"]


def load_card(card_path: str) -> dict:
    with open(DATA_DIR / card_path) as f:
        return json.load(f)


def generate_notebook(project: dict, dry_run: bool = False) -> Path | None:
    """Generate a notebook for a single project. Returns output path or None."""
    card = load_card(project["card_path"])
    category = project["category"]
    project_id = project["id"]

    template_fn = TEMPLATE_MAP.get(category)
    if template_fn is None:
        print(f"  SKIP {project_id}: no template for category '{category}'")
        return None

    # Build the notebook
    nb = template_fn(card)

    # Output path
    github_path = card.get("github_path", "")
    if not github_path:
        print(f"  SKIP {project_id}: no github_path in card")
        return None

    output_dir = REPO_ROOT / github_path / "notebooks"
    output_file = output_dir / f"{project_id}_analysis.ipynb"

    if dry_run:
        # Validate the notebook structure
        nbformat.validate(nb)
        print(f"  OK   {project_id} -> {output_file.relative_to(REPO_ROOT)} ({len(nb.cells)} cells)")
        return output_file

    # Write
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"  WRITE {project_id} -> {output_file.relative_to(REPO_ROOT)} ({len(nb.cells)} cells)")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate Jupyter notebooks from templates")
    parser.add_argument("--project", type=str, help="Generate for a single project ID")
    parser.add_argument("--category", type=str, help="Generate for a single category")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't write files")
    args = parser.parse_args()

    projects = load_manifest()

    if args.project:
        projects = [p for p in projects if p["id"] == args.project]
        if not projects:
            print(f"ERROR: Project '{args.project}' not found in manifest.")
            sys.exit(1)

    if args.category:
        projects = [p for p in projects if p["category"] == args.category]
        if not projects:
            print(f"ERROR: No projects found for category '{args.category}'.")
            sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "GENERATING"
    print(f"{mode} notebooks for {len(projects)} projects...\n")

    success = 0
    errors = 0
    for p in projects:
        try:
            result = generate_notebook(p, dry_run=args.dry_run)
            if result:
                success += 1
        except Exception as e:
            print(f"  ERROR {p['id']}: {e}")
            errors += 1

    print(f"\nDone: {success} generated, {errors} errors, {len(projects) - success - errors} skipped")


if __name__ == "__main__":
    main()
