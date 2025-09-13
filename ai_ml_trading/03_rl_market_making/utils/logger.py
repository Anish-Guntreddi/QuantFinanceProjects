"""Logging utilities for RL market making system."""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console logging."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['ENDC'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'exc_info',
                              'exc_text', 'stack_info', 'lineno', 'funcName',
                              'created', 'msecs', 'relativeCreated', 'thread',
                              'threadName', 'processName', 'process', 'message']:
                    log_entry[key] = value
        
        return json.dumps(log_entry)


class MarketMakingLogger:
    """Enhanced logger for market making system."""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.metrics_buffer = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
    
    def setup_logging(self, 
                     level: str = "INFO",
                     console_output: bool = True,
                     file_output: bool = True,
                     json_format: bool = False) -> None:
        """Setup logging configuration."""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if json_format:
                console_formatter = JSONFormatter()
            else:
                console_formatter = ColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_filename)
            
            if json_format:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_episode_start(self, episode: int, config: Dict[str, Any]) -> None:
        """Log episode start with configuration."""
        self.logger.info(f"Starting episode {episode}", extra={
            'episode': episode,
            'event_type': 'episode_start',
            'config': config
        })
    
    def log_episode_end(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log episode completion with metrics."""
        self.logger.info(f"Episode {episode} completed", extra={
            'episode': episode,
            'event_type': 'episode_end',
            'metrics': metrics
        })
    
    def log_trade(self, trade_info: Dict[str, Any]) -> None:
        """Log individual trade execution."""
        self.logger.debug("Trade executed", extra={
            'event_type': 'trade',
            'trade_info': trade_info
        })
    
    def log_order(self, order_info: Dict[str, Any]) -> None:
        """Log order placement/cancellation."""
        self.logger.debug("Order event", extra={
            'event_type': 'order',
            'order_info': order_info
        })
    
    def log_inventory_update(self, inventory: float, cash: float, pnl: float) -> None:
        """Log inventory and P&L updates."""
        self.logger.debug("Inventory updated", extra={
            'event_type': 'inventory_update',
            'inventory': inventory,
            'cash': cash,
            'pnl': pnl
        })
    
    def log_model_update(self, metrics: Dict[str, float]) -> None:
        """Log model training metrics."""
        self.logger.info("Model updated", extra={
            'event_type': 'model_update',
            'training_metrics': metrics
        })
    
    def log_evaluation(self, eval_metrics: Dict[str, float]) -> None:
        """Log evaluation results."""
        self.logger.info("Evaluation completed", extra={
            'event_type': 'evaluation',
            'eval_metrics': eval_metrics
        })
    
    def log_adverse_selection(self, adverse_metrics: Dict[str, float]) -> None:
        """Log adverse selection analysis."""
        self.logger.warning("Adverse selection detected", extra={
            'event_type': 'adverse_selection',
            'adverse_metrics': adverse_metrics
        })
    
    def buffer_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Buffer metrics for batch logging."""
        self.metrics_buffer.append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def flush_metrics(self, filename: Optional[str] = None) -> None:
        """Flush buffered metrics to file."""
        if not self.metrics_buffer:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, f"metrics_{timestamp}.jsonl")
        
        with open(filename, 'w') as f:
            for metric in self.metrics_buffer:
                f.write(json.dumps(metric) + '\n')
        
        self.logger.info(f"Flushed {len(self.metrics_buffer)} metrics to {filename}")
        self.metrics_buffer.clear()
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


# Global logger instances
_loggers = {}


def setup_logger(name: str,
                log_dir: str = "logs",
                level: str = "INFO",
                console_output: bool = True,
                file_output: bool = True,
                json_format: bool = False) -> MarketMakingLogger:
    """Setup and configure a logger."""
    
    if name not in _loggers:
        logger = MarketMakingLogger(name, log_dir)
        logger.setup_logging(level, console_output, file_output, json_format)
        _loggers[name] = logger
    
    return _loggers[name]


def get_logger(name: str) -> MarketMakingLogger:
    """Get existing logger or create default one."""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, logger: MarketMakingLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.get_logger().debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.get_logger().debug(
            f"Completed {self.operation_name}",
            extra={
                'operation': self.operation_name,
                'duration_seconds': duration,
                'event_type': 'performance'
            }
        )