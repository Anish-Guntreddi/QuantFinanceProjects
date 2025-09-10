"""Configuration management for experiments."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import copy

from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """Manage experiment configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.config_history = []
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        path = Path(config_path)
        
        if not path.exists():
            # Try relative to config_dir
            path = self.config_dir / config_path
            
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
            
        # Store in cache
        self.configs[str(path)] = config
        
        return config
    
    def save_config(self, config: Dict, path: str):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def merge_configs(self, *configs: Dict) -> Dict:
        """Merge multiple configurations."""
        if not configs:
            return {}
            
        # Convert to OmegaConf for deep merging
        omega_configs = [OmegaConf.create(cfg) for cfg in configs]
        merged = OmegaConf.merge(*omega_configs)
        
        return OmegaConf.to_container(merged)
    
    def validate_config(self, config: Dict, schema: Optional[Dict] = None) -> bool:
        """Validate configuration against schema."""
        if schema is None:
            # Use default schema
            schema = self._get_default_schema()
            
        return self._validate_recursive(config, schema)
    
    def _validate_recursive(self, config: Dict, schema: Dict) -> bool:
        """Recursively validate configuration."""
        for key, requirements in schema.items():
            if requirements.get('required', False) and key not in config:
                raise ValueError(f"Missing required field: {key}")
                
            if key in config:
                value = config[key]
                
                # Check type
                if 'type' in requirements:
                    expected_type = requirements['type']
                    if expected_type == 'dict' and not isinstance(value, dict):
                        raise TypeError(f"Field '{key}' must be a dict")
                    elif expected_type == 'list' and not isinstance(value, list):
                        raise TypeError(f"Field '{key}' must be a list")
                    elif expected_type == 'str' and not isinstance(value, str):
                        raise TypeError(f"Field '{key}' must be a string")
                    elif expected_type == 'int' and not isinstance(value, int):
                        raise TypeError(f"Field '{key}' must be an integer")
                    elif expected_type == 'float' and not isinstance(value, (int, float)):
                        raise TypeError(f"Field '{key}' must be a number")
                        
                # Check nested schema
                if 'schema' in requirements and isinstance(value, dict):
                    self._validate_recursive(value, requirements['schema'])
                    
                # Check allowed values
                if 'allowed' in requirements and value not in requirements['allowed']:
                    raise ValueError(f"Field '{key}' must be one of {requirements['allowed']}")
                    
                # Check range
                if 'min' in requirements and value < requirements['min']:
                    raise ValueError(f"Field '{key}' must be >= {requirements['min']}")
                if 'max' in requirements and value > requirements['max']:
                    raise ValueError(f"Field '{key}' must be <= {requirements['max']}")
                    
        return True
    
    def _get_default_schema(self) -> Dict:
        """Get default configuration schema."""
        return {
            'experiment': {
                'required': True,
                'type': 'dict',
                'schema': {
                    'name': {'required': True, 'type': 'str'},
                    'version': {'required': True, 'type': 'str'},
                    'seed': {'required': True, 'type': 'int', 'min': 0}
                }
            },
            'data': {
                'required': True,
                'type': 'dict',
                'schema': {
                    'source': {'required': True, 'type': 'str'},
                    'start_date': {'type': 'str'},
                    'end_date': {'type': 'str'}
                }
            },
            'model': {
                'required': True,
                'type': 'dict',
                'schema': {
                    'type': {'required': True, 'type': 'str'},
                    'parameters': {'type': 'dict'}
                }
            }
        }
    
    def create_experiment_config(
        self,
        name: str,
        base_config: Optional[str] = None,
        overrides: Optional[Dict] = None
    ) -> Dict:
        """Create experiment configuration from base and overrides."""
        # Load base config if provided
        if base_config:
            config = self.load_config(base_config)
        else:
            config = self._get_default_config()
            
        # Set experiment name
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['name'] = name
        
        # Apply overrides
        if overrides:
            config = self.merge_configs(config, overrides)
            
        # Validate
        self.validate_config(config)
        
        # Track in history
        self.config_history.append({
            'name': name,
            'config': copy.deepcopy(config),
            'timestamp': datetime.now().isoformat()
        })
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default configuration template."""
        return {
            'experiment': {
                'name': 'default',
                'version': '1.0.0',
                'seed': 42
            },
            'data': {
                'source': 'data/default.parquet',
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'frequency': 'daily'
            },
            'features': {
                'window_sizes': [10, 20, 50],
                'indicators': ['sma', 'rsi'],
                'target': 'returns'
            },
            'model': {
                'type': 'linear_regression',
                'parameters': {
                    'fit_intercept': True
                }
            },
            'validation': {
                'method': 'time_series_cv',
                'n_splits': 5,
                'test_size': 252
            }
        }
    
    def get_config_diff(self, config1: Dict, config2: Dict) -> Dict:
        """Get differences between two configurations."""
        diff = {
            'added': {},
            'removed': {},
            'changed': {}
        }
        
        def compare_dicts(d1, d2, path=""):
            # Find added keys
            for key in d2:
                if key not in d1:
                    diff['added'][f"{path}.{key}" if path else key] = d2[key]
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], f"{path}.{key}" if path else key)
                elif d1[key] != d2[key]:
                    diff['changed'][f"{path}.{key}" if path else key] = {
                        'old': d1[key],
                        'new': d2[key]
                    }
                    
            # Find removed keys
            for key in d1:
                if key not in d2:
                    diff['removed'][f"{path}.{key}" if path else key] = d1[key]
                    
        compare_dicts(config1, config2)
        
        return diff
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration files."""
        configs = []
        
        if self.config_dir.exists():
            for config_file in self.config_dir.rglob('*.yaml'):
                configs.append(str(config_file.relative_to(self.config_dir)))
            for config_file in self.config_dir.rglob('*.yml'):
                configs.append(str(config_file.relative_to(self.config_dir)))
            for config_file in self.config_dir.rglob('*.json'):
                configs.append(str(config_file.relative_to(self.config_dir)))
                
        return sorted(configs)
    
    def export_config_history(self, path: str):
        """Export configuration history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.config_history, f, indent=2, default=str)