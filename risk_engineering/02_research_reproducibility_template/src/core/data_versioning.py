"""Data versioning and management."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np


class DataVersioning:
    """Version control for datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def track_data(self, data: pd.DataFrame, name: str, version: Optional[str] = None) -> Dict:
        """Track a dataset with versioning."""
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create versioned path
        versioned_dir = self.data_dir / name / version
        versioned_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = versioned_dir / f"{name}.parquet"
        data.to_parquet(data_path)
        
        # Compute hash
        data_hash = self._compute_data_hash(data)
        
        # Update metadata
        if name not in self.metadata:
            self.metadata[name] = {}
            
        self.metadata[name][version] = {
            'path': str(data_path),
            'hash': data_hash,
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'timestamp': datetime.now().isoformat(),
            'stats': self._compute_data_stats(data)
        }
        
        # Set as latest version
        self.metadata[name]['latest'] = version
        
        self._save_metadata()
        
        return self.metadata[name][version]
    
    def get_data(self, name: str, version: Optional[str] = None) -> pd.DataFrame:
        """Get a specific version of data."""
        if name not in self.metadata:
            raise ValueError(f"Dataset '{name}' not found")
            
        # Use latest version if not specified
        if version is None:
            version = self.metadata[name].get('latest')
            if version is None:
                raise ValueError(f"No versions found for dataset '{name}'")
                
        if version not in self.metadata[name]:
            raise ValueError(f"Version '{version}' not found for dataset '{name}'")
            
        data_path = Path(self.metadata[name][version]['path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        return pd.read_parquet(data_path)
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a dataset."""
        if name not in self.metadata:
            return []
            
        versions = [v for v in self.metadata[name].keys() if v != 'latest']
        return sorted(versions)
    
    def compare_versions(self, name: str, version1: str, version2: str) -> Dict:
        """Compare two versions of a dataset."""
        if name not in self.metadata:
            raise ValueError(f"Dataset '{name}' not found")
            
        v1_info = self.metadata[name].get(version1)
        v2_info = self.metadata[name].get(version2)
        
        if not v1_info or not v2_info:
            raise ValueError("Version not found")
            
        comparison = {
            'shape_changed': v1_info['shape'] != v2_info['shape'],
            'columns_added': list(set(v2_info['columns']) - set(v1_info['columns'])),
            'columns_removed': list(set(v1_info['columns']) - set(v2_info['columns'])),
            'hash_changed': v1_info['hash'] != v2_info['hash']
        }
        
        # Compare statistics
        stats_comparison = {}
        for col in set(v1_info['stats'].keys()) & set(v2_info['stats'].keys()):
            v1_stats = v1_info['stats'][col]
            v2_stats = v2_info['stats'][col]
            stats_comparison[col] = {
                'mean_diff': v2_stats.get('mean', 0) - v1_stats.get('mean', 0),
                'std_diff': v2_stats.get('std', 0) - v1_stats.get('std', 0)
            }
        
        comparison['stats_changes'] = stats_comparison
        
        return comparison
    
    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of dataframe."""
        # Use a subset for large datasets
        if len(data) > 10000:
            sample = data.sample(n=10000, random_state=42)
        else:
            sample = data
            
        # Convert to bytes and hash
        data_bytes = pd.util.hash_pandas_object(sample).values.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _compute_data_stats(self, data: pd.DataFrame) -> Dict:
        """Compute basic statistics for numerical columns."""
        stats = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'nulls': int(data[col].isnull().sum())
            }
            
        return stats
    
    def create_snapshot(self, data: pd.DataFrame, name: str, description: str = "") -> str:
        """Create a snapshot of current data."""
        version = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        info = self.track_data(data, name, version)
        info['description'] = description
        info['is_snapshot'] = True
        
        self.metadata[name][version] = info
        self._save_metadata()
        
        return version
    
    def rollback(self, name: str, version: str) -> pd.DataFrame:
        """Rollback to a specific version."""
        data = self.get_data(name, version)
        
        # Set as latest
        self.metadata[name]['latest'] = version
        self._save_metadata()
        
        return data
    
    def delete_version(self, name: str, version: str):
        """Delete a specific version."""
        if name not in self.metadata:
            raise ValueError(f"Dataset '{name}' not found")
            
        if version not in self.metadata[name]:
            raise ValueError(f"Version '{version}' not found")
            
        # Don't delete if it's the latest version
        if self.metadata[name].get('latest') == version:
            raise ValueError("Cannot delete the latest version")
            
        # Delete data file
        data_path = Path(self.metadata[name][version]['path'])
        if data_path.exists():
            data_path.unlink()
            
        # Remove from metadata
        del self.metadata[name][version]
        self._save_metadata()
    
    def export_lineage(self, name: str) -> Dict:
        """Export data lineage information."""
        if name not in self.metadata:
            return {}
            
        lineage = {
            'dataset': name,
            'versions': []
        }
        
        for version in self.list_versions(name):
            version_info = self.metadata[name][version]
            lineage['versions'].append({
                'version': version,
                'timestamp': version_info.get('timestamp'),
                'shape': version_info.get('shape'),
                'hash': version_info.get('hash')
            })
            
        return lineage