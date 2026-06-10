# Coding Conventions

**Analysis Date:** 2026-06-10

## Naming Patterns

**Files:**
- Module files: `lowercase_with_underscores.py`
- Examples: `data_handler.py`, `backtest_engine.py`, `execution.py`, `portfolio.py`
- Test files: `test_<module_name>.py` (e.g., `test_events.py`, `test_portfolio.py`)
- Runner scripts: `run_<action>.py` or `generate_<output>.py` (e.g., `run_tests.py`, `generate_sample_data.py`)

**Classes:**
- PascalCase for all class names
- Examples: `MarketEvent`, `MovingAverageCrossoverStrategy`, `HistoricalCSVDataHandler`, `RiskManager`, `Portfolio`, `Position`, `Trade`
- Abstract base classes: Still PascalCase (e.g., `Strategy`, `DataHandler`, `ExecutionHandler`, `EventHandler`)
- Data classes: PascalCase (e.g., `StrategyParameters`, `BacktestConfig`)

**Functions/Methods:**
- snake_case for all functions and methods
- Private methods: Prefix with single underscore (e.g., `_calculate_rsi`, `_calculate_trend_factor`)
- Special methods: Dunder style (e.g., `__init__`, `__post_init__`)
- Property methods: Use `@property` decorator with snake_case name (e.g., `mid_price`, `total_pnl`)
- Examples: `calculate_signals()`, `update_position()`, `get_position()`, `check_position_limit()`

**Variables:**
- Local variables: snake_case (e.g., `market_event`, `strategy_params`, `current_positions`)
- Instance variables: snake_case with underscore prefix for private (e.g., `self.symbols`, `self._internal_state`)
- Constants: UPPER_SNAKE_CASE (e.g., `EventType` enum members like `MARKET`, `SIGNAL`, `ORDER`)

**Types:**
- Type hints are used throughout: `Dict[str, Any]`, `List[float]`, `Optional[datetime]`, `Tuple[float, float]`
- Type imports: `from typing import Dict, List, Optional, Any, Tuple, Union`

## Code Style

**Formatting:**
- No explicit formatter in use (Black is in requirements.txt but not enforced)
- 4-space indentation (Python standard)
- Line length: Generally follows 100-120 character limits based on file review
- Imports are organized by groups but sorting varies

**Linting:**
- Tools available: `flake8` (in requirements.txt)
- `mypy` available for type checking but not enforced
- Code follows basic PEP 8 standards

## Import Organization

**Order:**
1. Standard library imports (e.g., `import sys`, `import os`, `from pathlib import Path`)
2. Third-party imports (e.g., `import pandas as pd`, `import numpy as np`, `from datetime import datetime`)
3. Local imports (e.g., `from events import MarketEvent`, `from data_handler import DataHandler`)

**Pattern observed:**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging

from events import MarketEvent, SignalEvent
from data_handler import DataHandler
```

**Path Aliases:**
- No explicit path aliases configured
- Instead, `sys.path.append()` is used to add `src` directories (see `run_backtest.py` lines 17-18)
- Common pattern: `sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))`

## Error Handling

**Patterns:**
- Validation errors: `raise ValueError()` with descriptive messages
- Examples from `events.py`:
  - Line 104: `raise ValueError(f"Signal strength must be between -1 and 1, got {self.strength}")`
  - Line 139: `raise ValueError(f"Order quantity must be positive, got {self.quantity}")`
  - Line 142: `raise ValueError("Limit orders must specify a price")`
- Post-initialization validation: `__post_init__()` method on dataclasses

**Try-except blocks:**
- Used sparingly in main execution paths
- Typically logged with `logger.error()` before exception handling
- Example from `run_backtest.py` (lines 371-389): Wraps strategy creation in try-except with logging

## Logging

**Framework:** Python's standard `logging` module

**Pattern:**
```python
import logging
logger = logging.getLogger(__name__)
```

**Usage levels:**
- `logger.info()`: General informational messages (strategy setup, data loaded, results summary)
- `logger.error()`: Error conditions and failures
- `logger.warning()`: Warnings and potential issues
- `logger.debug()`: Detailed debugging information (event creation, internal state changes)

**Examples:**
- `events.py` line 73: `logger.debug(f"Market event created for {self.symbol} at {self.timestamp}")`
- `data_handler.py`: `logger.info(f"Loaded {len(df)} bars for {symbol} from {csv_file}")`
- `run_backtest.py` line 220: `logger.error(f"Error running {strategy_name}: {e}")`

## Comments

**When to Comment:**
- Module-level docstrings: Always (3-4 line description of module purpose)
- Class docstrings: Always on abstract/important classes
- Method docstrings: On public methods, especially for complex logic
- Inline comments: Minimal; only for non-obvious logic

**JSDoc/TSDoc:**
- Not used (Python project)
- Uses standard Python docstrings instead

**Docstring format:**
```python
"""
Module description here.

This module provides [purpose and key components].
"""

class ClassName:
    """Short description of class."""
    
    def method(self, arg: str) -> Dict[str, Any]:
        """
        Description of what method does.
        
        Args:
            arg: Description of arg
            
        Returns:
            Description of return value
        """
```

**Examples:**
- `run_backtest.py` (lines 1-7): Module docstring
- `events.py` (lines 44-45): `"""Market data update event containing OHLCV and order book information."""`
- `strategy.py` (lines 71-81): Multi-line docstring with Args and Returns

## Function Design

**Size:** Functions are generally 10-60 lines
- Smaller utility functions: 5-15 lines
- Strategy calculation methods: 20-40 lines
- Configuration/setup functions: 30-60 lines
- Long functions (100+ lines): `run_backtest.py` functions like `create_multi_factor_backtest()` (103 lines) handle complex setup

**Parameters:**
- Type hints always used: `def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:`
- Defaults provided for optional parameters: `def __init__(self, symbols: List[str], parameters: Optional[StrategyParameters] = None)`
- Dataclass pattern for complex parameter passing: `StrategyParameters` class wraps dict-like parameters

**Return Values:**
- Always typed: `-> Dict[str, float]`, `-> bool`, `-> List[SignalEvent]`
- Multiple returns use Tuple: `-> Tuple[float, float]`
- Void/None returns explicit: `-> None`

## Module Design

**Exports:**
- No `__all__` defined in modules
- Everything public at module level is importable
- Private functions/classes prefixed with underscore if needed

**Barrel Files:**
- Not used in this codebase
- Each module imported explicitly (e.g., `from events import MarketEvent`)

**Module categories:**
- Core infrastructure: `events.py`, `data_handler.py`, `execution.py`, `portfolio.py`
- Strategy and analysis: `strategy.py`, `performance.py`
- Utilities and configuration: `utils.py`, `backtest_engine.py`
- Test modules: Separate `tests/` directory with corresponding test files

**Layered imports:**
- Lower layers: `events.py` (no imports from project modules)
- Middle layers: `data_handler.py` imports `events.py`
- Upper layers: `backtest_engine.py` imports from multiple modules
- Circular imports avoided through careful layering

---

*Convention analysis: 2026-06-10*
