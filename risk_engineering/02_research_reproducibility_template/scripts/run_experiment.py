#!/usr/bin/env python
"""Run reproducible experiment."""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from src.core import (
    ExperimentTracker,
    ExperimentConfig,
    ConfigManager,
    ReproducibilityUtils,
    DataVersioning
)
from src.validation import ResultValidator


def generate_sample_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """Generate sample data for testing."""
    np.random.seed(42)
    
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add target variable
    data['target'] = (
        0.5 * data['feature_0'] + 
        0.3 * data['feature_1'] + 
        np.random.randn(n_samples) * 0.1
    )
    
    return data


def run_experiment(config_name: str, debug: bool = False):
    """Run experiment with full reproducibility."""
    
    print(f"Starting experiment: {config_name}")
    print("=" * 60)
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Create default config for demo
    config_dict = {
        'experiment': {
            'name': config_name,
            'version': '1.0.0',
            'seed': 42
        },
        'data': {
            'source': 'generated',
            'n_samples': 1000,
            'n_features': 10
        },
        'model': {
            'type': 'linear_regression',
            'parameters': {
                'fit_intercept': True
            }
        },
        'features': {
            'selected': ['feature_0', 'feature_1', 'feature_2'],
            'target': 'target'
        }
    }
    
    config = ExperimentConfig(
        name=config_dict['experiment']['name'],
        version=config_dict['experiment']['version'],
        seed=config_dict['experiment']['seed'],
        data_config=config_dict['data'],
        model_config=config_dict['model'],
        feature_config=config_dict['features']
    )
    
    # Set seed for reproducibility
    ReproducibilityUtils.set_global_seed(config.seed)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(config.name)
    run_id = tracker.start_run(config, tags={'debug': str(debug)})
    
    print(f"Run ID: {run_id}")
    print(f"Config hash: {config.to_hash()}")
    
    try:
        # Log environment
        env_info = ReproducibilityUtils.log_environment()
        tracker.log_artifacts(['environment_info.json'])
        
        # Generate/Load data
        print("\nGenerating data...")
        data = generate_sample_data(
            n_samples=config.data_config['n_samples'],
            n_features=config.data_config['n_features']
        )
        
        # Version the data
        data_versioning = DataVersioning()
        data_version = data_versioning.track_data(data, 'experiment_data')
        print(f"Data version: {data_version['hash'][:8]}")
        
        # Validate data
        print("\nValidating data...")
        validator = ResultValidator()
        data_validation = validator.validate_data_integrity(data)
        print(f"Data valid: {data_validation['valid']}")
        
        # Prepare features
        X = data[config.feature_config['selected']]
        y = data[config.feature_config['target']]
        
        # Train model (simple example)
        print("\nTraining model...")
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.seed
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        predictions = model.predict(X_test)
        residuals = y_test - predictions
        mse = np.mean(residuals ** 2)
        
        print(f"Train R²: {train_score:.4f}")
        print(f"Test R²: {test_score:.4f}")
        print(f"MSE: {mse:.4f}")
        
        # Log metrics
        metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'mse': mse,
            'n_samples': len(X),
            'n_features': len(config.feature_config['selected'])
        }
        tracker.log_metrics(metrics)
        
        # Validate model assumptions
        print("\nValidating model assumptions...")
        assumptions = validator.validate_model_assumptions(residuals, predictions)
        print(f"All assumptions met: {assumptions['all_passed']}")
        
        # Validate performance bounds
        bounds = {
            'test_r2': (0.5, 1.0),
            'mse': (0, 0.5)
        }
        perf_validation = validator.validate_performance_bounds(metrics, bounds)
        print(f"Performance valid: {perf_validation['valid']}")
        
        # Save model
        tracker.log_model(model, "linear_model")
        
        # Create reproducibility report
        print("\nGenerating reproducibility report...")
        report = ReproducibilityUtils.create_reproducibility_report(
            config.name,
            config_dict,
            metrics,
            output_path=f"experiments/{run_id}/reproducibility_report.json"
        )
        tracker.log_artifacts([f"experiments/{run_id}/reproducibility_report.json"])
        
        print("\nExperiment completed successfully!")
        print(f"Final metrics: {metrics}")
        
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        tracker.log_metrics({'failed': 1})
        raise
        
    finally:
        tracker.end_run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run reproducible experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='demo_experiment',
        help='Configuration name'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Ensure reproducibility
    ReproducibilityUtils.ensure_deterministic_operations()
    
    # Run experiment
    run_experiment(args.config, args.debug)


if __name__ == "__main__":
    main()