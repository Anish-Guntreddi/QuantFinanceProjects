# Research Reproducibility Template

## Overview
Standardized template for reproducible quantitative research with experiment tracking, version control, and automated validation.

## Project Structure
```
research_reproducibility_template/
├── src/
│   ├── core/
│   │   ├── experiment_tracker.py
│   │   ├── data_versioning.py
│   │   ├── config_manager.py
│   │   └── reproducibility_utils.py
│   ├── validation/
│   │   ├── result_validator.py
│   │   ├── statistical_tests.py
│   │   └── performance_benchmarks.py
│   ├── pipelines/
│   │   ├── data_pipeline.py
│   │   ├── feature_pipeline.py
│   │   └── model_pipeline.py
│   └── reporting/
│       ├── report_generator.py
│       └── visualization.py
├── configs/
│   ├── experiment_configs/
│   ├── data_configs/
│   └── model_configs/
├── experiments/
│   └── template_experiment/
│       ├── config.yaml
│       ├── notebooks/
│       ├── results/
│       └── logs/
├── tests/
│   ├── test_reproducibility.py
│   └── test_pipelines.py
└── scripts/
    ├── run_experiment.py
    └── validate_results.py
```

## Implementation

### 1. Experiment Tracker
```python
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    name: str
    version: str
    seed: int
    data_config: Dict[str, Any]
    model_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    
    def to_hash(self) -> str:
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

class ExperimentTracker:
    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlruns.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_id = None
        self.config_hash = None
        
    def start_run(self, config: ExperimentConfig, tags: Optional[Dict] = None):
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
            mlflow.log_param(f"data.{key}", value)
        for key, value in config.model_config.items():
            mlflow.log_param(f"model.{key}", value)
        for key, value in config.feature_config.items():
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
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)
    
    def log_artifacts(self, artifact_paths: List[str]):
        for path in artifact_paths:
            mlflow.log_artifact(path)
    
    def log_dataset(self, data: pd.DataFrame, name: str):
        data_path = Path(f"experiments/{self.run_id}/data/{name}.parquet")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(data_path)
        mlflow.log_artifact(str(data_path))
    
    def log_model(self, model, artifact_path: str):
        mlflow.sklearn.log_model(model, artifact_path)
    
    def end_run(self):
        mlflow.end_run()
    
    def _find_existing_runs(self, config_hash: str) -> List[str]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return []
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.config_hash = '{config_hash}'"
        )
        return runs['run_id'].tolist()
```

### 2. Data Versioning
```python
import dvc.api
import git
import hashlib
from typing import Optional, Tuple

class DataVersioning:
    def __init__(self, repo_path: str = "."):
        self.repo = git.Repo(repo_path)
        self.repo_path = Path(repo_path)
        
    def track_data(self, data_path: str, remote: Optional[str] = None):
        # Add to DVC
        dvc.api.add(data_path)
        
        # Push to remote if specified
        if remote:
            dvc.api.push(data_path, remote=remote)
        
        # Get data hash
        data_hash = self._compute_file_hash(data_path)
        
        # Get git commit
        commit_hash = self.repo.head.commit.hexsha
        
        return {
            'data_path': data_path,
            'data_hash': data_hash,
            'commit_hash': commit_hash,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_data_version(self, data_path: str, version: Optional[str] = None):
        if version:
            # Checkout specific version
            self.repo.git.checkout(version)
            
        # Pull data from DVC
        dvc.api.pull(data_path)
        
        return pd.read_parquet(data_path)
    
    def _compute_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def create_data_snapshot(self, data: pd.DataFrame, name: str) -> Dict:
        snapshot_dir = self.repo_path / "data" / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = snapshot_dir / f"{name}_{timestamp}.parquet"
        
        data.to_parquet(snapshot_path)
        
        # Track with DVC
        version_info = self.track_data(str(snapshot_path))
        
        # Commit to git
        self.repo.index.add([str(snapshot_path) + ".dvc"])
        self.repo.index.commit(f"Add data snapshot: {name}_{timestamp}")
        
        return version_info
```

### 3. Config Manager
```python
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize

class ConfigManager:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_experiment_config(self, experiment_name: str) -> ExperimentConfig:
        config_path = self.config_dir / "experiment_configs" / f"{experiment_name}.yaml"
        
        with initialize(config_path=str(self.config_dir)):
            cfg = compose(config_name=experiment_name)
        
        return ExperimentConfig(
            name=cfg.experiment.name,
            version=cfg.experiment.version,
            seed=cfg.experiment.seed,
            data_config=OmegaConf.to_container(cfg.data),
            model_config=OmegaConf.to_container(cfg.model),
            feature_config=OmegaConf.to_container(cfg.features)
        )
    
    def validate_config(self, config: ExperimentConfig) -> bool:
        required_fields = {
            'data_config': ['source', 'start_date', 'end_date'],
            'model_config': ['type', 'parameters'],
            'feature_config': ['window_sizes', 'indicators']
        }
        
        for config_type, fields in required_fields.items():
            config_dict = getattr(config, config_type)
            for field in fields:
                if field not in config_dict:
                    raise ValueError(f"Missing required field '{field}' in {config_type}")
        
        return True
    
    def merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        base = OmegaConf.create(base_config)
        override = OmegaConf.create(override_config)
        return OmegaConf.to_container(OmegaConf.merge(base, override))
```

### 4. Reproducibility Utils
```python
import random
import torch
import tensorflow as tf

class ReproducibilityUtils:
    @staticmethod
    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tf.random.set_seed(seed)
        
        # Make operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    @staticmethod
    def log_environment():
        import pkg_resources
        import platform
        
        env_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
        }
        
        # Log installed packages
        packages = {}
        for dist in pkg_resources.working_set:
            packages[dist.project_name] = dist.version
        
        env_info['packages'] = packages
        
        # Save to file
        env_path = Path("environment_info.json")
        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)
        
        return env_info
    
    @staticmethod
    def create_checkpoint(state: Dict, path: str):
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'state': state
        }
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_checkpoint(path: str) -> Dict:
        return torch.load(path)
```

### 5. Result Validator
```python
from scipy import stats
import warnings

class ResultValidator:
    def __init__(self, baseline_results: Optional[Dict] = None):
        self.baseline_results = baseline_results
        self.validation_tests = []
        
    def validate_statistical_significance(
        self,
        results: np.ndarray,
        baseline: np.ndarray,
        test: str = 'wilcoxon',
        alpha: float = 0.05
    ) -> Dict:
        if test == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(results, baseline)
        elif test == 't-test':
            statistic, p_value = stats.ttest_rel(results, baseline)
        elif test == 'mann-whitney':
            statistic, p_value = stats.mannwhitneyu(results, baseline)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        return {
            'test': test,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha
        }
    
    def validate_performance_bounds(
        self,
        metrics: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict:
        violations = []
        
        for metric, value in metrics.items():
            if metric in bounds:
                lower, upper = bounds[metric]
                if value < lower or value > upper:
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'bounds': (lower, upper)
                    })
        
        return {
            'valid': len(violations) == 0,
            'violations': violations
        }
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict:
        issues = []
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            issues.append({
                'type': 'missing_values',
                'columns': missing[missing > 0].to_dict()
            })
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            issues.append({
                'type': 'duplicates',
                'count': duplicates
            })
        
        # Check for outliers (using IQR method)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | 
                       (data[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                issues.append({
                    'type': 'outliers',
                    'column': col,
                    'count': outliers
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
```

### 6. Data Pipeline
```python
from typing import Callable, List

class DataPipeline:
    def __init__(self, steps: List[Tuple[str, Callable]]):
        self.steps = steps
        self.history = []
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        for name, transform in self.steps:
            print(f"Running step: {name}")
            before_shape = result.shape
            result = transform(result)
            after_shape = result.shape
            
            self.history.append({
                'step': name,
                'before_shape': before_shape,
                'after_shape': after_shape,
                'timestamp': datetime.now().isoformat()
            })
        
        return result
    
    def add_step(self, name: str, transform: Callable):
        self.steps.append((name, transform))
    
    def save_pipeline(self, path: str):
        import dill
        with open(path, 'wb') as f:
            dill.dump(self.steps, f)
    
    @classmethod
    def load_pipeline(cls, path: str):
        import dill
        with open(path, 'rb') as f:
            steps = dill.load(f)
        return cls(steps)

# Example transforms
class DataTransforms:
    @staticmethod
    def remove_outliers(data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        result = data.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (result[col] >= (Q1 - 1.5 * IQR)) & (result[col] <= (Q3 + 1.5 * IQR))
                result = result[mask]
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs(stats.zscore(result[col]))
                result = result[z_scores < 3]
        
        return result
    
    @staticmethod
    def normalize_features(data: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        result = data.copy()
        
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            result[columns] = scaler.fit_transform(result[columns])
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            result[columns] = scaler.fit_transform(result[columns])
        
        return result
```

### 7. Report Generator
```python
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class ReportGenerator:
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path
        self.figures = []
        self.tables = []
        self.metrics = {}
        
    def add_metric(self, name: str, value: float, description: str = ""):
        self.metrics[name] = {
            'value': value,
            'description': description
        }
    
    def add_figure(self, fig: plt.Figure, caption: str):
        fig_path = f"figures/fig_{len(self.figures)}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        self.figures.append({
            'path': fig_path,
            'caption': caption
        })
    
    def add_table(self, df: pd.DataFrame, caption: str):
        self.tables.append({
            'data': df.to_html(classes='table table-striped'),
            'caption': caption
        })
    
    def generate_html_report(self, output_path: str):
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; }
                .figure { margin: 20px 0; text-align: center; }
                .figure img { max-width: 100%; }
                .caption { font-style: italic; margin-top: 10px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Research Report</h1>
            <p>Generated: {{ timestamp }}</p>
            
            <h2>Metrics</h2>
            {% for name, info in metrics.items() %}
            <div class="metric">
                <strong>{{ name }}:</strong> {{ info.value }}
                {% if info.description %}
                <br><small>{{ info.description }}</small>
                {% endif %}
            </div>
            {% endfor %}
            
            <h2>Figures</h2>
            {% for fig in figures %}
            <div class="figure">
                <img src="{{ fig.path }}" alt="{{ fig.caption }}">
                <div class="caption">{{ fig.caption }}</div>
            </div>
            {% endfor %}
            
            <h2>Tables</h2>
            {% for table in tables %}
            <div class="table-container">
                <div class="caption">{{ table.caption }}</div>
                {{ table.data|safe }}
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        template = Template(template_str)
        html = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=self.metrics,
            figures=self.figures,
            tables=self.tables
        )
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def generate_latex_report(self, output_path: str):
        template_str = r"""
        \documentclass{article}
        \usepackage{graphicx}
        \usepackage{booktabs}
        \usepackage{hyperref}
        
        \title{Research Report}
        \date{\today}
        
        \begin{document}
        \maketitle
        
        \section{Metrics}
        {% for name, info in metrics.items() %}
        \textbf{{ "{" }}{{ name }}{{ "}" }}: {{ info.value }}
        {% if info.description %}
        \\ \small{{ "{" }}{{ info.description }}{{ "}" }}
        {% endif %}
        \par
        {% endfor %}
        
        \section{Results}
        {% for fig in figures %}
        \begin{figure}[h]
            \centering
            \includegraphics[width=\textwidth]{{ "{" }}{{ fig.path }}{{ "}" }}
            \caption{{ "{" }}{{ fig.caption }}{{ "}" }}
        \end{figure}
        {% endfor %}
        
        \end{document}
        """
        
        template = Template(template_str)
        latex = template.render(
            metrics=self.metrics,
            figures=self.figures
        )
        
        with open(output_path, 'w') as f:
            f.write(latex)
```

### 8. Example Experiment Script
```python
# scripts/run_experiment.py
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.experiment_tracker import ExperimentTracker, ExperimentConfig
from src.core.config_manager import ConfigManager
from src.core.reproducibility_utils import ReproducibilityUtils
from src.pipelines.data_pipeline import DataPipeline, DataTransforms
from src.validation.result_validator import ResultValidator
from src.reporting.report_generator import ReportGenerator

def run_experiment(config_name: str, debug: bool = False):
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_experiment_config(config_name)
    
    # Set seed for reproducibility
    ReproducibilityUtils.set_global_seed(config.seed)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(config.name)
    run_id = tracker.start_run(config, tags={'debug': str(debug)})
    
    try:
        # Log environment
        env_info = ReproducibilityUtils.log_environment()
        tracker.log_artifacts(['environment_info.json'])
        
        # Create data pipeline
        pipeline = DataPipeline([
            ('remove_outliers', lambda df: DataTransforms.remove_outliers(df, ['returns'], 'iqr')),
            ('normalize', lambda df: DataTransforms.normalize_features(df, ['volume'], 'standard'))
        ])
        
        # Load and process data
        data = pd.read_parquet(config.data_config['source'])
        processed_data = pipeline.run(data)
        tracker.log_dataset(processed_data, 'processed_data')
        
        # Train model (placeholder)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = processed_data[config.feature_config['features']]
        y = processed_data[config.feature_config['target']]
        model.fit(X, y)
        
        # Evaluate
        predictions = model.predict(X)
        mse = np.mean((predictions - y) ** 2)
        r2 = model.score(X, y)
        
        # Log metrics
        tracker.log_metrics({
            'mse': mse,
            'r2': r2,
            'num_samples': len(X)
        })
        
        # Validate results
        validator = ResultValidator()
        validation_results = validator.validate_performance_bounds(
            {'mse': mse, 'r2': r2},
            {'mse': (0, 1.0), 'r2': (0, 1.0)}
        )
        
        # Generate report
        report_gen = ReportGenerator()
        report_gen.add_metric('MSE', mse, 'Mean Squared Error')
        report_gen.add_metric('R²', r2, 'Coefficient of Determination')
        
        # Create performance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y, predictions, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predictions vs Actual')
        report_gen.add_figure(fig, 'Prediction accuracy visualization')
        
        # Generate HTML report
        report_path = f"experiments/{run_id}/report.html"
        report_gen.generate_html_report(report_path)
        tracker.log_artifacts([report_path])
        
        # Save model
        tracker.log_model(model, "model")
        
        print(f"Experiment completed successfully. Run ID: {run_id}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        tracker.log_metrics({'failed': 1})
        raise
    finally:
        tracker.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config name')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    run_experiment(args.config, args.debug)
```

### 9. Example Config
```yaml
# configs/experiment_configs/baseline_momentum.yaml
experiment:
  name: baseline_momentum
  version: 1.0.0
  seed: 42

data:
  source: data/sp500_prices.parquet
  start_date: 2020-01-01
  end_date: 2023-12-31
  frequency: daily

features:
  window_sizes: [10, 20, 50]
  indicators:
    - sma
    - rsi
    - macd
  features:
    - returns_10d
    - volume_ratio
    - rsi_14
  target: forward_returns_1d

model:
  type: linear_regression
  parameters:
    fit_intercept: true
    normalize: false
  validation:
    method: time_series_cv
    n_splits: 5
    test_size: 252
```

## Build and Run

### Installation
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
mlflow>=2.0.0
dvc>=3.0.0
hydra-core>=1.3.0
omegaconf>=2.3.0
pyyaml>=6.0
pytest>=7.0.0
jinja2>=3.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
torch>=2.0.0
tensorflow>=2.15.0
dill>=0.3.6
GitPython>=3.1.0
```

### Running Experiments
```bash
# Run baseline experiment
python scripts/run_experiment.py --config baseline_momentum

# Run with debugging
python scripts/run_experiment.py --config baseline_momentum --debug

# Validate results
python scripts/validate_results.py --run-id <run_id>
```

## Key Features

1. **Experiment Tracking**: MLflow integration for comprehensive tracking
2. **Data Versioning**: DVC for data version control
3. **Config Management**: Hydra/OmegaConf for configuration
4. **Reproducibility**: Seed management and environment logging
5. **Validation**: Statistical tests and performance bounds
6. **Pipelines**: Modular data and feature pipelines
7. **Reporting**: Automated HTML/LaTeX report generation

## Deliverables

- Complete experiment tracking system
- Data versioning with DVC
- Reproducible research pipelines
- Automated validation and testing
- Professional report generation
- Config-driven experiments