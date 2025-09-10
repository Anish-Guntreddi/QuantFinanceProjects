"""Experiment tracking with MLflow."""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

import mlflow
import numpy as np
import pandas as pd
import yaml


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    
    name: str
    version: str
    seed: int
    data_config: Dict[str, Any]
    model_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    
    def to_hash(self) -> str:
        """Generate hash of configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()


class ExperimentTracker:
    """Track experiments with MLflow."""
    
    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlruns.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_id = None
        self.config_hash = None
        
    def start_run(self, config: ExperimentConfig, tags: Optional[Dict] = None) -> str:
        """Start a new experiment run."""
        self.config_hash = config.to_hash()
        
        # Check for existing runs with same config
        existing_runs = self._find_existing_runs(self.config_hash)
        if existing_runs:
            print(f"Found {len(existing_runs)} existing runs with same config")
            
        self.run_id = str(uuid.uuid4())
        mlflow.start_run(run_id=self.run_id)
        
        # Log configuration
        mlflow.log_params({
            "experiment_name": config.name,
            "version": config.version,
            "seed": config.seed,
            "config_hash": self.config_hash
        })
        
        # Log nested configs
        for key, value in config.data_config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"data.{key}", value)
                
        for key, value in config.model_config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"model.{key}", value)
                
        for key, value in config.feature_config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"feature.{key}", value)
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
            
        # Log config file
        config_path = Path(f"experiments/{self.run_id}/config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f)
        mlflow.log_artifact(str(config_path))
        
        return self.run_id
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value, step=step)
    
    def log_artifacts(self, artifact_paths: List[str]):
        """Log artifacts to MLflow."""
        for path in artifact_paths:
            if Path(path).exists():
                mlflow.log_artifact(path)
    
    def log_dataset(self, data: pd.DataFrame, name: str):
        """Log dataset to MLflow."""
        data_path = Path(f"experiments/{self.run_id}/data/{name}.parquet")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(data_path)
        mlflow.log_artifact(str(data_path))
    
    def log_model(self, model, artifact_path: str):
        """Log model to MLflow."""
        try:
            mlflow.sklearn.log_model(model, artifact_path)
        except:
            # Fallback for non-sklearn models
            import joblib
            model_path = Path(f"experiments/{self.run_id}/models/{artifact_path}.pkl")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))
    
    def end_run(self):
        """End the current run."""
        mlflow.end_run()
    
    def _find_existing_runs(self, config_hash: str) -> List[str]:
        """Find existing runs with the same configuration hash."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return []
        
        try:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.config_hash = '{config_hash}'"
            )
            return runs['run_id'].tolist()
        except:
            return []
    
    def get_run_metrics(self, run_id: Optional[str] = None) -> Dict:
        """Get metrics for a run."""
        if run_id is None:
            run_id = self.run_id
            
        run = mlflow.get_run(run_id)
        return run.data.metrics
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare metrics across multiple runs."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return pd.DataFrame()
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"run_id IN {tuple(run_ids)}"
        )
        return runs
    
    def cleanup_old_runs(self, days: int = 30):
        """Clean up old experiment runs."""
        from datetime import timedelta
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return
            
        cutoff_date = datetime.now() - timedelta(days=days)
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"start_time < '{cutoff_date.isoformat()}'"
        )
        
        for run_id in runs['run_id']:
            try:
                mlflow.delete_run(run_id)
                print(f"Deleted old run: {run_id}")
            except:
                print(f"Failed to delete run: {run_id}")