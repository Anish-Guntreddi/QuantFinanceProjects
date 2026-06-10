"""Packaging tests (QBT-01): qbacktest must import from any cwd, offline."""

import subprocess
import sys
import tempfile


def test_import_from_foreign_cwd():
    """Editable install means sibling projects can import qbacktest from anywhere."""
    result = subprocess.run(
        [sys.executable, "-c", "import qbacktest; print(qbacktest.__version__)"],
        cwd=tempfile.mkdtemp(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_import_pulls_no_network_modules():
    """Importing qbacktest must not import yfinance or other network clients."""
    code = (
        "import sys; import qbacktest; "
        "banned = [m for m in ('yfinance', 'requests', 'urllib3') if m in sys.modules]; "
        "sys.exit(1 if banned else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tempfile.mkdtemp(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, "qbacktest import pulled in network modules"
