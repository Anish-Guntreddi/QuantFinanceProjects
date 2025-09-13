#!/usr/bin/env python3
"""
Test runner for the event-driven backtesting framework.

This script runs all tests with coverage reporting and benchmarking.
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_basic_tests():
    """Run basic test suite without benchmarks."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(Path(__file__).parent),
        "-v",
        "--tb=short",
        "--durations=10"
    ]
    
    return subprocess.run(cmd).returncode


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(Path(__file__).parent),
        "-v",
        "--cov=../src",
        "--cov-report=html:../coverage_html",
        "--cov-report=term",
        "--tb=short"
    ]
    
    return subprocess.run(cmd).returncode


def run_benchmarks():
    """Run benchmark tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(Path(__file__).parent),
        "-v",
        "--benchmark-only",
        "--benchmark-sort=mean"
    ]
    
    return subprocess.run(cmd).returncode


def run_specific_test(test_file):
    """Run a specific test file."""
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return 1
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v",
        "--tb=short"
    ]
    
    return subprocess.run(cmd).returncode


def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    
    modules_to_check = [
        'events', 'data_handler', 'strategy', 'portfolio', 
        'execution', 'backtest_engine', 'performance', 'utils'
    ]
    
    failed_imports = []
    
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import {len(failed_imports)} modules.")
        return False
    else:
        print(f"\nAll {len(modules_to_check)} modules imported successfully.")
        return True


def run_integration_test():
    """Run a basic integration test."""
    print("Running integration test...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent.parent / "src"
        sys.path.insert(0, str(src_path))
        
        from datetime import datetime
        from events import MarketEvent, EventQueue
        from strategy import MovingAverageCrossoverStrategy, StrategyParameters
        from portfolio import Portfolio
        from execution import SimulatedExecutionHandler, LinearSlippageModel, FixedCommissionModel
        
        # Test basic event creation and queuing
        queue = EventQueue()
        market_event = MarketEvent("TEST", datetime.now(), close=100.0)
        queue.put(market_event)
        
        retrieved_event = queue.get()
        assert retrieved_event == market_event
        
        # Test strategy parameter handling
        params = StrategyParameters()
        params.set('test_param', 42)
        assert params.get('test_param') == 42
        
        # Test portfolio creation
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.total_portfolio_value == 100000
        
        # Test execution handler creation
        slippage_model = LinearSlippageModel()
        commission_model = FixedCommissionModel()
        handler = SimulatedExecutionHandler(slippage_model, commission_model)
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run backtesting framework tests')
    parser.add_argument('--coverage', action='store_true', 
                       help='Run tests with coverage reporting')
    parser.add_argument('--benchmarks', action='store_true',
                       help='Run benchmark tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration test only')
    parser.add_argument('--check-imports', action='store_true',
                       help='Check if all modules can be imported')
    parser.add_argument('--file', type=str,
                       help='Run specific test file')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests including coverage and benchmarks')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # Default: run basic tests
        args.basic = True
    
    # Check imports first
    if args.check_imports or args.all:
        if not check_imports():
            return 1
        print()
    
    # Run integration test
    if args.integration or args.all:
        if not run_integration_test():
            return 1
        print()
    
    # Run specific test file
    if args.file:
        return run_specific_test(args.file)
    
    # Run benchmarks
    if args.benchmarks or args.all:
        print("Running benchmark tests...")
        benchmark_result = run_benchmarks()
        if benchmark_result != 0:
            print("Some benchmarks failed")
        print()
    
    # Run tests with coverage
    if args.coverage or args.all:
        print("Running tests with coverage...")
        return run_tests_with_coverage()
    
    # Default: run basic tests
    print("Running basic test suite...")
    return run_basic_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)