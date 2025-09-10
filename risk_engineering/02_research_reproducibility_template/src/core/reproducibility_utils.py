"""Utilities for ensuring reproducibility."""

import os
import random
import json
import platform
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd


class ReproducibilityUtils:
    """Utilities for reproducible research."""
    
    @staticmethod
    def set_global_seed(seed: int):
        """Set random seed for all libraries."""
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # Set environment variable for hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Try to set seeds for ML libraries if available
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        except ImportError:
            pass
    
    @staticmethod
    def log_environment() -> Dict:
        """Log environment information."""
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            },
            'environment_variables': {
                k: v for k, v in os.environ.items() 
                if k.startswith(('PYTHON', 'PATH', 'VIRTUAL'))
            }
        }
        
        # Get installed packages
        try:
            import pkg_resources
            packages = {}
            for dist in pkg_resources.working_set:
                packages[dist.project_name] = dist.version
            env_info['packages'] = packages
        except:
            env_info['packages'] = {}
            
        # Save to file
        env_path = Path("environment_info.json")
        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)
            
        return env_info
    
    @staticmethod
    def create_checkpoint(state: Dict, path: str):
        """Create a checkpoint of current state."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'state': state
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use joblib for better serialization
        try:
            import joblib
            joblib.dump(checkpoint, path)
        except:
            # Fallback to pickle
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(checkpoint, f)
    
    @staticmethod
    def load_checkpoint(path: str) -> Dict:
        """Load checkpoint from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        try:
            import joblib
            return joblib.load(path)
        except:
            # Fallback to pickle
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    @staticmethod
    def compute_code_hash(code_dir: str = ".") -> str:
        """Compute hash of code files."""
        code_dir = Path(code_dir)
        
        # Get all Python files
        py_files = sorted(code_dir.rglob("*.py"))
        
        # Compute combined hash
        hasher = hashlib.sha256()
        
        for py_file in py_files:
            # Skip cache and test files
            if '__pycache__' in str(py_file) or 'test_' in py_file.name:
                continue
                
            with open(py_file, 'rb') as f:
                hasher.update(f.read())
                
        return hasher.hexdigest()
    
    @staticmethod
    def ensure_deterministic_operations():
        """Ensure all operations are deterministic."""
        # NumPy
        np.seterr(all='raise')  # Raise errors for numerical issues
        
        # Pandas
        pd.options.mode.chained_assignment = 'raise'
        
        # Warnings
        import warnings
        warnings.filterwarnings('error')  # Convert warnings to errors
        
        print("Deterministic mode enabled")
    
    @staticmethod
    def create_reproducibility_report(
        experiment_name: str,
        config: Dict,
        results: Dict,
        output_path: str = "reproducibility_report.json"
    ):
        """Create a reproducibility report."""
        report = {
            'experiment': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results': results,
            'environment': ReproducibilityUtils.log_environment(),
            'code_hash': ReproducibilityUtils.compute_code_hash(),
            'reproducibility_checklist': {
                'seed_set': 'seed' in config,
                'environment_logged': True,
                'code_versioned': True,
                'data_versioned': 'data_version' in config,
                'config_saved': True
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    @staticmethod
    def verify_reproducibility(
        original_results: Dict,
        new_results: Dict,
        tolerance: float = 1e-6
    ) -> Dict:
        """Verify if results are reproducible."""
        verification = {
            'reproducible': True,
            'differences': {}
        }
        
        for key in original_results:
            if key not in new_results:
                verification['reproducible'] = False
                verification['differences'][key] = 'Missing in new results'
                continue
                
            orig_value = original_results[key]
            new_value = new_results[key]
            
            # Compare based on type
            if isinstance(orig_value, (int, float)):
                if abs(orig_value - new_value) > tolerance:
                    verification['reproducible'] = False
                    verification['differences'][key] = {
                        'original': orig_value,
                        'new': new_value,
                        'diff': abs(orig_value - new_value)
                    }
            elif isinstance(orig_value, np.ndarray):
                if not np.allclose(orig_value, new_value, rtol=tolerance):
                    verification['reproducible'] = False
                    verification['differences'][key] = {
                        'max_diff': np.max(np.abs(orig_value - new_value))
                    }
            elif isinstance(orig_value, pd.DataFrame):
                if not orig_value.equals(new_value):
                    verification['reproducible'] = False
                    verification['differences'][key] = 'DataFrames differ'
                    
        return verification
    
    @staticmethod
    def setup_logging(log_dir: str = "logs"):
        """Setup logging for reproducibility."""
        import logging
        from logging.handlers import RotatingFileHandler
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('reproducibility')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = RotatingFileHandler(
            log_dir / 'experiment.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger