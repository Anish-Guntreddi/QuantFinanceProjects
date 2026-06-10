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


def test_public_api_complete():
    """Every name in __all__ resolves from the qbacktest top level."""
    import qbacktest

    for name in qbacktest.__all__:
        assert getattr(qbacktest, name, None) is not None, f"missing export: {name}"


def test_import_does_not_load_matplotlib():
    """matplotlib loads lazily (only when TearsheetRenderer is accessed)."""
    code = (
        "import sys; import qbacktest; "
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tempfile.mkdtemp(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, "import qbacktest eagerly loaded matplotlib"


def test_no_skipped_stubs_remain():
    """All Wave 0 skip stubs have been replaced by real tests."""
    import pathlib

    tests_dir = pathlib.Path(__file__).parent
    marker = "W0 " + "stub"  # split so this file's own source doesn't match
    offenders = [
        p.name for p in tests_dir.glob("test_*.py") if marker in p.read_text()
    ]
    assert offenders == [], f"Wave-0 skip markers remain in: {offenders}"
