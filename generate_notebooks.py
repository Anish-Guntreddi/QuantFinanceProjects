#!/usr/bin/env python3
"""
Generate Jupyter notebooks for all projects from templates and YAML configs.

Usage:
    python generate_notebooks.py                         # Generate all
    python generate_notebooks.py --category hft          # One category
    python generate_notebooks.py --project hft_01        # One project (prefix match)
    python generate_notebooks.py --dry-run               # Preview only
"""

import argparse
import os
import sys
import yaml
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Template class mapping
TEMPLATE_MAP = {
    "hft": "notebook_templates.hft_template.HFTTemplate",
    "ml_trading": "notebook_templates.ml_trading_template.MLTradingTemplate",
    "backtesting": "notebook_templates.backtesting_template.BacktestingTemplate",
    "microstructure": "notebook_templates.microstructure_template.MicrostructureTemplate",
    "execution": "notebook_templates.execution_template.ExecutionTemplate",
    "intraday": "notebook_templates.intraday_template.IntradayTemplate",
    "risk": "notebook_templates.risk_template.RiskTemplate",
}

# Category prefix mapping for --category filter
CATEGORY_PREFIXES = {
    "hft": "hft_",
    "ml": "ml_",
    "research": "research_",
    "engines": "engines_",
    "execution": "exec_",
    "intraday": "intraday_",
    "risk": "risk_",
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT_DIR, "notebook_configs")


def load_template(template_name):
    """Dynamically load a template class."""
    class_path = TEMPLATE_MAP.get(template_name)
    if not class_path:
        print(f"  WARNING: Unknown template '{template_name}', using base template")
        from notebook_templates.common import BaseNotebookTemplate
        return BaseNotebookTemplate()

    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls()


def load_registry():
    """Load the project registry and all project configs."""
    registry_path = os.path.join(CONFIGS_DIR, "_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    configs = []
    for entry in registry["projects"]:
        config_path = os.path.join(CONFIGS_DIR, entry["config_file"])
        with open(config_path) as f:
            config = yaml.safe_load(f)
        configs.append(config)
    return configs


def generate_notebook(config, dry_run=False):
    """Generate a single notebook from a config."""
    proj = config["project"]
    category = proj["category"]
    dir_name = proj["dir_name"]
    proj_id = proj["id"]

    # Determine output path
    notebooks_dir = os.path.join(ROOT_DIR, category, dir_name, "notebooks")
    # Clean filename from display name
    safe_name = proj_id.replace("/", "_").replace(" ", "_")
    output_path = os.path.join(notebooks_dir, f"{safe_name}_analysis.ipynb")

    if dry_run:
        print(f"  [DRY RUN] Would generate: {output_path}")
        return True

    # Load template
    template = load_template(config["template"])

    # Generate cells
    cells = template.generate_cells(config)

    # Build notebook
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.10.0",
    }

    for cell_type, content in cells:
        if cell_type == "markdown":
            nb.cells.append(new_markdown_cell(content))
        else:
            nb.cells.append(new_code_cell(content))

    # Ensure output directory exists
    os.makedirs(notebooks_dir, exist_ok=True)

    # Write notebook
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"  Generated: {os.path.relpath(output_path, ROOT_DIR)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate Jupyter notebooks from templates")
    parser.add_argument("--category", type=str, help="Generate only for this category (hft, ml, research, engines, execution, intraday, risk)")
    parser.add_argument("--project", type=str, help="Generate only for this project (prefix match on project_id)")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be generated")
    args = parser.parse_args()

    # Load all configs
    configs = load_registry()
    print(f"Loaded {len(configs)} project configurations")

    # Filter
    if args.category:
        prefix = CATEGORY_PREFIXES.get(args.category, args.category + "_")
        configs = [c for c in configs if c["project"]["id"].startswith(prefix)]
        print(f"Filtered to {len(configs)} projects in category '{args.category}'")

    if args.project:
        configs = [c for c in configs if c["project"]["id"].startswith(args.project)]
        print(f"Filtered to {len(configs)} projects matching '{args.project}'")

    if not configs:
        print("No matching projects found.")
        return

    # Generate
    success = 0
    for config in configs:
        try:
            if generate_notebook(config, dry_run=args.dry_run):
                success += 1
        except Exception as e:
            print(f"  ERROR generating {config['project']['id']}: {e}")

    print(f"\n{'Would generate' if args.dry_run else 'Generated'} {success}/{len(configs)} notebooks")


if __name__ == "__main__":
    main()
