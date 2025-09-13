"""Test runner for options volatility surface project"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests():
    """Run all tests with coverage reporting"""
    
    print("="*60)
    print("Running Options Volatility Surface Tests")
    print("="*60)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(Path(__file__).parent),
        "-v",
        "--tb=short",
        "-p", "no:warnings"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Some tests failed. Please review the output above.")
        print("="*60)
    
    return result.returncode


if __name__ == "__main__":
    exit(run_tests())