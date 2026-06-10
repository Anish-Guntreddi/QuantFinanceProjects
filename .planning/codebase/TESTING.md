# Testing Patterns

**Analysis Date:** 2026-06-10

## Test Framework

**Runner:**
- pytest 7.4.0+ (specified in `requirements.txt`)
- Config: No explicit `pytest.ini` or `setup.cfg` found; uses standard pytest discovery
- Test discovery: Automatic (files named `test_*.py`)

**Assertion Library:**
- pytest built-in assertions (`assert` statements)
- pandas testing utilities: `pd.testing.assert_series_equal()`, `pd.testing.assert_frame_equal()`
- unittest assertions also used: `self.assertEqual()`, `self.assertIn()`, `self.assertTrue()` (in some test files)

**Run Commands:**
```bash
# Run all tests in a project
cd project_directory/
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_events.py -v

# Run specific test class/method
python -m pytest tests/test_events.py::TestEventCreation::test_market_event_creation -v

# Using the run_tests.py script
python tests/run_tests.py --coverage
python tests/run_tests.py --integration
```

## Test File Organization

**Location:**
- Co-located pattern: Tests are in separate `tests/` directory adjacent to `src/`
- Structure: `project_name/tests/` contains all test files

**Naming:**
- Pattern: `test_<module_name>.py` corresponds to `src/<module_name>.py`
- Examples: `test_events.py` tests `src/events.py`, `test_portfolio.py` tests `src/portfolio.py`
- Runner scripts: `run_tests.py` orchestrates test execution

**Structure:**
```
02_event_driven_backtester/
├── src/
│   ├── events.py
│   ├── strategy.py
│   ├── portfolio.py
│   └── ... (other modules)
└── tests/
    ├── test_events.py
    ├── test_strategy.py
    ├── test_portfolio.py
    ├── test_execution.py
    └── run_tests.py
```

## Test Structure

**Suite Organization:**
```python
class TestEventCreation:
    """Test basic event creation and validation."""
    
    def test_market_event_creation(self):
        """Test MarketEvent creation and properties."""
        # Setup
        timestamp = datetime.now()
        
        # Execute
        market_event = MarketEvent(
            symbol="AAPL",
            timestamp=timestamp,
            bid=150.0,
            ask=150.05
        )
        
        # Assert
        assert market_event.event_type == EventType.MARKET
        assert market_event.symbol == "AAPL"
```

**Patterns:**
- Setup: Test data creation (often using fixtures or setUp methods)
- Execute: Call the function/method being tested
- Assert: Verify the results

**Test class organization:**
- One test class per major component or concern
- Example from `test_events.py`:
  - `TestEventCreation`: Tests event object creation and properties
  - `TestEventQueue`: Tests event queue operations
  - `TestEventDispatcher`: Tests event dispatch mechanisms
  - `TestEventComparison`: Tests event comparison for sorting

## Mocking

**Framework:** 
- `unittest.mock.Mock` (standard library)
- Available in requirements: `pytest-mock>=3.11.0`

**Patterns:**
From `test_events.py` (lines 286-298):
```python
class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self):
        self.handled_events = []
        self.can_handle_types = [EventType.MARKET]
    
    def handle_event(self, event: Event):
        self.handled_events.append(event)
    
    def can_handle(self, event_type: EventType) -> bool:
        return event_type in self.can_handle_types
```

**What to Mock:**
- External dependencies: File I/O, network calls, database
- Complex collaborators: Other modules that are already tested separately
- Time-dependent behavior: Use `datetime.now()` or fixture-based time

**What NOT to Mock:**
- Core business logic (test the actual implementation)
- Data classes and simple value objects (events, positions)
- The code under test itself
- Simple utility functions (calculate, validate)

## Fixtures and Factories

**Test Data:**
From `test_portfolio.py` (lines 48-54):
```python
def test_position_update_market_price(self):
    """Test market price update and unrealized P&L."""
    position = Position(
        symbol="AAPL",
        quantity=100,
        avg_price=150.0
    )
    
    timestamp = datetime.now()
    position.update_market_price(155.0, timestamp)
```

From `test_basic.py` (lines 26-44):
```python
def setUp(self):
    """Set up test data"""
    np.random.seed(42)  # Reproducible randomness
    
    # Generate cointegrated series
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    common_factor = np.cumsum(np.random.normal(0, 0.01, 252))
    
    self.y1 = pd.Series(
        100 + 0.8 * common_factor + np.cumsum(np.random.normal(0, 0.005, 252)),
        index=dates,
        name='Y1'
    )
```

**Location:**
- Inline in test methods for simple cases
- `setUp()` method for complex shared fixtures
- Separate factory functions can be created but not observed in current codebase

## Coverage

**Requirements:** No explicit coverage targets enforced

**View Coverage:**
```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Terminal report with percentages
python -m pytest tests/ --cov=src --cov-report=term

# Using run_tests.py with coverage
python tests/run_tests.py --coverage
```

**Coverage tools:** `pytest-cov>=4.1.0` (in requirements.txt)

## Test Types

**Unit Tests:**
- Scope: Individual classes and functions
- Approach: Test single units in isolation
- Examples:
  - `TestEventCreation`: Tests individual event object creation
  - `TestPosition`: Tests position class methods
  - `test_book_to_price()`: Tests factor calculation function
- Isolation: Mock external dependencies, use synthetic test data

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Tests interaction between modules
- Examples from `run_tests.py` (lines 108-151):
  - `run_integration_test()`: Tests event creation → queue → strategy parameter handling
- Setup: No mocking; uses real objects from multiple modules
- Execution path: `run_tests.py --integration`

**E2E Tests:**
- Framework: Not explicitly used
- Alternative: Full backtest runs with sample data serve as end-to-end validation
- Pattern: `generate_sample_data.py` creates test datasets, then `run_backtest.py` executes full pipeline

## Common Patterns

**Async Testing:**
- Not applicable (single-threaded synchronous design)
- Backtesting runs sequentially through event loop in `backtest_engine.py`

**Error Testing:**
From `test_events.py` (lines 68-85):
```python
def test_signal_event_validation(self):
    """Test SignalEvent validation."""
    
    # Valid signal
    signal = SignalEvent(
        symbol="AAPL",
        timestamp=timestamp,
        signal_type="LONG",
        strength=0.8,
        confidence=0.9
    )
    
    assert signal.event_type == EventType.SIGNAL
    
    # Invalid strength should raise error
    with pytest.raises(ValueError):
        SignalEvent(
            symbol="AAPL",
            timestamp=timestamp,
            signal_type="LONG",
            strength=1.5  # Invalid: > 1
        )
```

**Validation Testing:**
- Test both valid inputs (assertions pass)
- Test invalid inputs (exceptions raised)
- Verify error messages contain helpful context

**Data Validation:**
From `test_factors.py` (lines 72-80):
```python
def test_factor_validation():
    """Test factor validation and cleaning"""
    data = pd.DataFrame({
        'book_value': [100, 200, np.inf, -np.inf, 150],
        'market_cap': [1000, 1500, 2000, 2500, 0]
    })
    
    factor = BookToPrice()
    raw_values = data['book_value'] / data['market_cap']
```

## Test Execution Flow

**Import Path Setup:**
All test files use this pattern:
```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

This allows importing modules from the `src/` directory without installing the package.

**Test Runner Script (`run_tests.py`):**
- `check_imports()`: Verifies all modules can be imported before running tests
- `run_integration_test()`: Runs a comprehensive integration test
- `run_basic_tests()`: Default pytest execution
- `run_tests_with_coverage()`: Pytest with coverage reporting
- `run_benchmarks()`: Performance benchmarks (pytest-benchmark)

**Benchmark Tests:**
- Framework: `pytest-benchmark>=4.0.0` (in requirements)
- Usage: Tests marked with `@pytest.mark.benchmark`
- Command: `python tests/run_tests.py --benchmarks`

## Test Examples from Codebase

**Example 1: Queue Operations Test**
From `test_events.py` (lines 181-204):
```python
def test_queue_operations(self):
    """Test basic queue operations."""
    queue = EventQueue()
    
    assert queue.empty()
    assert queue.size() == 0
    
    # Add events
    event1 = MarketEvent("AAPL", datetime.now())
    event2 = MarketEvent("GOOGL", datetime.now())
    
    queue.put(event1)
    queue.put(event2)
    
    assert not queue.empty()
    assert queue.size() == 2
    
    # Get events (should be FIFO for same priority)
    retrieved1 = queue.get()
    retrieved2 = queue.get()
    
    assert retrieved1 == event1
    assert retrieved2 == event2
    assert queue.empty()
```

**Example 2: Portfolio P&L Calculation**
From `test_portfolio.py` (lines 63-79):
```python
def test_position_pnl_calculation(self):
    """Test P&L calculation."""
    position = Position(
        symbol="AAPL",
        quantity=100,
        avg_price=150.0,
        market_price=155.0,
        realized_pnl=200.0
    )
    
    position.update_market_price(155.0, datetime.now())
    
    expected_unrealized = 500.0  # (155 - 150) * 100
    expected_total = 200.0 + 500.0  # realized + unrealized
    
    assert position.unrealized_pnl == expected_unrealized
    assert position.total_pnl == expected_total
```

**Example 3: Statistical Test**
From `test_basic.py` (lines 46-55):
```python
def test_engle_granger(self):
    """Test Engle-Granger cointegration test"""
    eg_test = EngleGrangerTest()
    is_coint, p_value, details = eg_test.test(self.y1, self.y2)
    
    # Should find cointegration (low p-value)
    self.assertTrue(p_value < 0.05, f"P-value {p_value} should be < 0.05")
    self.assertTrue(is_coint, "Should detect cointegration")
    self.assertIn('beta', details)
    self.assertTrue(abs(details['beta']) > 0, "Beta should be non-zero")
```

---

*Testing analysis: 2026-06-10*
