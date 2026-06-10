"""qbacktest.tearsheet — 3-panel matplotlib tearsheet renderer.

Public API
----------
TearsheetRenderer:
    render(results, title, filename) -> Path | None
    summary_table(results) -> str
"""

from qbacktest.tearsheet.renderer import TearsheetRenderer

__all__ = ["TearsheetRenderer"]
