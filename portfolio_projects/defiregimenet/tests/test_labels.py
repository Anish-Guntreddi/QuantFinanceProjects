"""Label quarantine enforcement + forward-label tests (05-02).

The AST quarantine guard below is LIVE from Wave 0: it walks every source
file and fails if any module outside the allowed importers touches
defiregimenet.labels. It passes trivially before labels.py exists and
protects every wave-2 executor from accidentally wiring forward-looking
labels into a feature or model path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

QUARANTINED_MODULE = "defiregimenet.labels"
ALLOWED_IMPORTERS = {"defiregimenet.evaluation", "defiregimenet.pipeline"}


def test_label_quarantine():
    """No source module outside evaluation/pipeline may import labels.py.

    Forward-looking regime labels (built from FUTURE returns and FUTURE
    realized vol) are evaluation-only ground truth. Any import from a
    feature, model, or training module is look-ahead leakage by
    construction (DFR-02 strict causal separation).
    """
    src_root = Path(__file__).parents[1] / "src" / "defiregimenet"
    assert src_root.exists(), f"source root not found: {src_root}"

    violations: list[str] = []
    for path in src_root.rglob("*.py"):
        module_rel = path.relative_to(src_root)
        module_name = "defiregimenet." + ".".join(module_rel.with_suffix("").parts)
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        if any(allowed in module_name for allowed in ALLOWED_IMPORTERS):
            continue
        if module_name.split(".")[-1] == "labels":
            continue  # labels.py itself is allowed to exist

        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "labels" in node.module:
                    violations.append(f"{module_name}: imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "labels" in alias.name:
                        violations.append(f"{module_name}: imports {alias.name}")

    assert violations == [], f"Label quarantine violated: {violations}"


def test_labels_are_forward_looking():
    pytest.skip("Wave 0 stub — implemented in 05-02")
