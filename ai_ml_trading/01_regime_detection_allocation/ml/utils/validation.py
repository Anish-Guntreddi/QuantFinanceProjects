"""
Cross-validation utilities for regime detection models.

This module provides specialized validation techniques for regime detection including
time series cross-validation and regime-aware validation strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Iterator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")


class CrossValidator:
    """Cross-validation utilities for regime detection models"""
    
    def __init__(self):
        self.validation_results = {}
        self.performance_history = []
        
    def time_series_cross_validate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size_fraction: float = 0.2,
        min_train_size: Optional[int] = None
    ) -> Dict:
        """
        Perform time series cross-validation
        
        Parameters:
        -----------
        model : BaseRegimeDetector
            Model to validate
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target series (for supervised validation)
        n_splits : int
            Number of cross-validation splits
        test_size_fraction : float
            Fraction of data to use for testing in each split
        min_train_size : Optional[int]
            Minimum size of training set
            
        Returns:
        --------
        Dict
            Cross-validation results
        """
        
        print(f"Performing time series cross-validation with {n_splits} splits...")
        
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # Setup time series split
        test_size = int(len(X_aligned) * test_size_fraction)
        if min_train_size is None:
            min_train_size = int(len(X_aligned) * 0.3)  # At least 30% for training
            
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size
        )
        
        # Store results
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_aligned)):
            print(f"  Processing fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train = X_aligned.iloc[train_idx]
            X_test = X_aligned.iloc[test_idx]
            y_train = y_aligned.iloc[train_idx] if len(y_aligned) > 0 else None
            y_test = y_aligned.iloc[test_idx] if len(y_aligned) > 0 else None
            
            # Skip if training set too small
            if len(X_train) < min_train_size:
                print(f"    Skipping fold {fold + 1}: training set too small")
                continue
                
            # Fit model
            try:
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model_copy.predict(X_train)
                test_pred = model_copy.predict(X_test)
                
                train_proba = model_copy.predict_proba(X_train)
                test_proba = model_copy.predict_proba(X_test)
                
                # Calculate metrics
                fold_result = self._calculate_fold_metrics(
                    train_pred, test_pred, train_proba, test_proba,
                    y_train, y_test, fold
                )
                
                fold_result['train_size'] = len(X_train)
                fold_result['test_size'] = len(X_test)
                fold_result['train_period'] = (X_train.index[0], X_train.index[-1])
                fold_result['test_period'] = (X_test.index[0], X_test.index[-1])
                
                fold_results.append(fold_result)
                
            except Exception as e:
                print(f"    Error in fold {fold + 1}: {e}")
                continue
                
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results)
        
        self.validation_results['time_series_cv'] = cv_results
        
        return cv_results
    
    def regime_aware_cross_validate(
        self,
        model,
        X: pd.DataFrame,
        regime_labels: pd.Series,
        n_splits: int = 5
    ) -> Dict:
        """
        Cross-validation ensuring each fold has representation from all regimes
        
        Parameters:
        -----------
        model : BaseRegimeDetector
            Model to validate
        X : pd.DataFrame
            Feature matrix
        regime_labels : pd.Series
            True regime labels
        n_splits : int
            Number of splits
            
        Returns:
        --------
        Dict
            Validation results
        """
        
        print(f"Performing regime-aware cross-validation with {n_splits} splits...")
        
        # Align data
        common_index = X.index.intersection(regime_labels.index)
        X_aligned = X.loc[common_index]
        regime_aligned = regime_labels.loc[common_index]
        
        # Create stratified time series splits
        splits = self._create_regime_aware_splits(regime_aligned, n_splits)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"  Processing fold {fold + 1}/{n_splits}")
            
            X_train = X_aligned.iloc[train_idx]
            X_test = X_aligned.iloc[test_idx]
            y_train = regime_aligned.iloc[train_idx]
            y_test = regime_aligned.iloc[test_idx]
            
            try:
                # Fit model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train)
                
                # Predictions
                train_pred = model_copy.predict(X_train)
                test_pred = model_copy.predict(X_test)
                
                train_proba = model_copy.predict_proba(X_train)
                test_proba = model_copy.predict_proba(X_test)
                
                # Calculate metrics
                fold_result = self._calculate_supervised_metrics(
                    train_pred, test_pred, y_train, y_test, fold
                )
                
                # Add regime distribution info
                fold_result['train_regime_dist'] = y_train.value_counts(normalize=True).to_dict()
                fold_result['test_regime_dist'] = y_test.value_counts(normalize=True).to_dict()
                
                fold_results.append(fold_result)
                
            except Exception as e:
                print(f"    Error in fold {fold + 1}: {e}")
                continue
                
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results)
        
        self.validation_results['regime_aware_cv'] = cv_results
        
        return cv_results
    
    def walk_forward_validation(
        self,
        model,
        X: pd.DataFrame,
        initial_train_size: float = 0.5,
        step_size: int = 30,
        forecast_horizon: int = 30
    ) -> Dict:
        """
        Walk-forward validation for regime detection
        
        Parameters:
        -----------
        model : BaseRegimeDetector
            Model to validate
        X : pd.DataFrame
            Feature matrix
        initial_train_size : float
            Initial training set size as fraction
        step_size : int
            Step size for walking forward
        forecast_horizon : int
            Forecast horizon for each step
            
        Returns:
        --------
        Dict
            Walk-forward validation results
        """
        
        print("Performing walk-forward validation...")
        
        n_obs = len(X)
        initial_train_end = int(n_obs * initial_train_size)
        
        results = []
        current_start = initial_train_end
        
        while current_start + forecast_horizon < n_obs:
            # Training data: from beginning to current_start
            X_train = X.iloc[:current_start]
            
            # Test data: next forecast_horizon periods
            test_end = min(current_start + forecast_horizon, n_obs)
            X_test = X.iloc[current_start:test_end]
            
            try:
                # Fit model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train)
                
                # Predict
                test_pred = model_copy.predict(X_test)
                test_proba = model_copy.predict_proba(X_test)
                
                # Store results
                step_result = {
                    'step': len(results) + 1,
                    'train_end': current_start,
                    'test_start': current_start,
                    'test_end': test_end,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'predictions': test_pred,
                    'probabilities': test_proba,
                    'regime_stability': self._calculate_regime_stability(test_pred),
                    'confidence': np.mean([np.max(prob) for prob in test_proba])
                }
                
                results.append(step_result)
                
                print(f"  Step {len(results)}: Train size={len(X_train)}, Test size={len(X_test)}")
                
            except Exception as e:
                print(f"  Error in step {len(results) + 1}: {e}")
                
            # Move forward
            current_start += step_size
            
        # Aggregate results
        wf_results = self._aggregate_walkforward_results(results)
        
        self.validation_results['walk_forward'] = wf_results
        
        return wf_results
    
    def out_of_sample_validation(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict:
        """
        Out-of-sample validation
        
        Parameters:
        -----------
        model : BaseRegimeDetector
            Model to validate
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : Optional[pd.Series]
            Training labels (for supervised validation)
        y_test : Optional[pd.Series]
            Test labels (for supervised validation)
            
        Returns:
        --------
        Dict
            Out-of-sample validation results
        """
        
        print("Performing out-of-sample validation...")
        
        try:
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            if y_train is not None and y_test is not None:
                results = self._calculate_supervised_metrics(
                    train_pred, test_pred, y_train, y_test, fold_id='oos'
                )
            else:
                results = self._calculate_unsupervised_metrics(
                    train_pred, test_pred, train_proba, test_proba, fold_id='oos'
                )
                
            # Add period information
            results['train_period'] = (X_train.index[0], X_train.index[-1])
            results['test_period'] = (X_test.index[0], X_test.index[-1])
            results['train_size'] = len(X_train)
            results['test_size'] = len(X_test)
            
            self.validation_results['out_of_sample'] = results
            
            return results
            
        except Exception as e:
            print(f"Error in out-of-sample validation: {e}")
            return {}
    
    def _clone_model(self, model):
        """Create a copy of the model for validation"""
        
        # Simple cloning - create new instance with same parameters
        model_class = model.__class__
        
        # Try to extract initialization parameters
        init_params = {}
        
        if hasattr(model, 'n_regimes'):
            init_params['n_regimes'] = model.n_regimes
            
        # Model-specific parameters
        if hasattr(model, 'covariance_type'):
            init_params['covariance_type'] = model.covariance_type
        if hasattr(model, 'method'):
            init_params['method'] = model.method
        if hasattr(model, 'model_type'):
            init_params['model_type'] = model.model_type
            
        return model_class(**init_params)
    
    def _create_regime_aware_splits(
        self, 
        regime_labels: pd.Series, 
        n_splits: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create splits ensuring regime representation"""
        
        # Simple approach: divide time series into n_splits chunks
        # while trying to maintain temporal order
        
        n_obs = len(regime_labels)
        chunk_size = n_obs // n_splits
        
        splits = []
        
        for i in range(n_splits):
            test_start = i * chunk_size
            test_end = min((i + 1) * chunk_size, n_obs)
            
            # Test indices
            test_idx = np.arange(test_start, test_end)
            
            # Train indices: all data before test period
            train_idx = np.arange(0, test_start)
            
            # Ensure minimum training size
            if len(train_idx) < chunk_size:
                continue
                
            splits.append((train_idx, test_idx))
            
        return splits
    
    def _calculate_fold_metrics(
        self,
        train_pred: np.ndarray,
        test_pred: np.ndarray,
        train_proba: np.ndarray,
        test_proba: np.ndarray,
        y_train: Optional[pd.Series],
        y_test: Optional[pd.Series],
        fold_id: Union[int, str]
    ) -> Dict:
        """Calculate metrics for a single fold"""
        
        metrics = {'fold': fold_id}
        
        # Unsupervised metrics (always available)
        metrics.update(self._calculate_unsupervised_metrics(
            train_pred, test_pred, train_proba, test_proba, fold_id
        ))
        
        # Supervised metrics (if labels available)
        if y_train is not None and y_test is not None:
            supervised_metrics = self._calculate_supervised_metrics(
                train_pred, test_pred, y_train, y_test, fold_id
            )
            metrics.update(supervised_metrics)
            
        return metrics
    
    def _calculate_supervised_metrics(
        self,
        train_pred: np.ndarray,
        test_pred: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        fold_id: Union[int, str]
    ) -> Dict:
        """Calculate supervised validation metrics"""
        
        metrics = {}
        
        # Align predictions with labels
        y_train_aligned = y_train.iloc[:len(train_pred)] if len(y_train) > len(train_pred) else y_train
        y_test_aligned = y_test.iloc[:len(test_pred)] if len(y_test) > len(test_pred) else y_test
        
        try:
            # Training metrics
            metrics['train_accuracy'] = accuracy_score(y_train_aligned, train_pred)
            
            # Test metrics
            metrics['test_accuracy'] = accuracy_score(y_test_aligned, test_pred)
            
            # Precision, Recall, F1 (macro average for multi-class)
            metrics['test_precision'] = precision_score(y_test_aligned, test_pred, average='macro')
            metrics['test_recall'] = recall_score(y_test_aligned, test_pred, average='macro')
            metrics['test_f1'] = f1_score(y_test_aligned, test_pred, average='macro')
            
            # Confusion matrix
            cm = confusion_matrix(y_test_aligned, test_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Classification report
            metrics['classification_report'] = classification_report(
                y_test_aligned, test_pred, output_dict=True
            )
            
        except Exception as e:
            print(f"    Warning: Could not calculate supervised metrics: {e}")
            
        return metrics
    
    def _calculate_unsupervised_metrics(
        self,
        train_pred: np.ndarray,
        test_pred: np.ndarray,
        train_proba: np.ndarray,
        test_proba: np.ndarray,
        fold_id: Union[int, str]
    ) -> Dict:
        """Calculate unsupervised validation metrics"""
        
        metrics = {}
        
        try:
            # Regime stability
            metrics['train_regime_stability'] = self._calculate_regime_stability(train_pred)
            metrics['test_regime_stability'] = self._calculate_regime_stability(test_pred)
            
            # Average confidence
            metrics['train_confidence'] = np.mean([np.max(prob) for prob in train_proba])
            metrics['test_confidence'] = np.mean([np.max(prob) for prob in test_proba])
            
            # Regime distribution
            unique_regimes, regime_counts = np.unique(test_pred, return_counts=True)
            regime_distribution = dict(zip(unique_regimes, regime_counts / len(test_pred)))
            metrics['test_regime_distribution'] = regime_distribution
            
            # Regime transitions
            transitions = np.sum(np.diff(test_pred) != 0)
            metrics['test_regime_transitions'] = transitions
            metrics['test_transition_rate'] = transitions / len(test_pred) if len(test_pred) > 1 else 0
            
        except Exception as e:
            print(f"    Warning: Could not calculate unsupervised metrics: {e}")
            
        return metrics
    
    def _calculate_regime_stability(self, predictions: np.ndarray) -> float:
        """Calculate regime stability (persistence)"""
        
        if len(predictions) <= 1:
            return 1.0
            
        # Calculate average regime duration
        regime_durations = []
        current_regime = predictions[0]
        current_duration = 1
        
        for pred in predictions[1:]:
            if pred == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = pred
                current_duration = 1
                
        regime_durations.append(current_duration)
        
        # Stability = average duration / total periods
        avg_duration = np.mean(regime_durations)
        stability = avg_duration / len(predictions)
        
        return stability
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results across folds"""
        
        if not fold_results:
            return {}
            
        aggregated = {
            'n_folds': len(fold_results),
            'fold_results': fold_results
        }
        
        # Aggregate numeric metrics
        numeric_metrics = [
            'train_accuracy', 'test_accuracy', 'test_precision', 
            'test_recall', 'test_f1', 'train_confidence', 'test_confidence',
            'train_regime_stability', 'test_regime_stability', 'test_transition_rate'
        ]
        
        for metric in numeric_metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                
        return aggregated
    
    def _aggregate_walkforward_results(self, step_results: List[Dict]) -> Dict:
        """Aggregate walk-forward validation results"""
        
        if not step_results:
            return {}
            
        aggregated = {
            'n_steps': len(step_results),
            'step_results': step_results
        }
        
        # Calculate performance over time
        confidences = [step['confidence'] for step in step_results]
        stabilities = [step['regime_stability'] for step in step_results]
        
        aggregated['avg_confidence'] = np.mean(confidences)
        aggregated['confidence_trend'] = np.corrcoef(range(len(confidences)), confidences)[0, 1]
        aggregated['avg_stability'] = np.mean(stabilities)
        aggregated['stability_trend'] = np.corrcoef(range(len(stabilities)), stabilities)[0, 1]
        
        return aggregated
    
    def get_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        
        report = "REGIME DETECTION VALIDATION REPORT\n"
        report += "=" * 40 + "\n\n"
        
        if not self.validation_results:
            report += "No validation results available.\n"
            return report
            
        for validation_type, results in self.validation_results.items():
            report += f"{validation_type.upper()} RESULTS:\n"
            report += "-" * 25 + "\n"
            
            if validation_type == 'time_series_cv':
                self._add_cv_report(report, results)
            elif validation_type == 'regime_aware_cv':
                self._add_cv_report(report, results)
            elif validation_type == 'walk_forward':
                self._add_wf_report(report, results)
            elif validation_type == 'out_of_sample':
                self._add_oos_report(report, results)
                
            report += "\n"
            
        return report
    
    def _add_cv_report(self, report: str, results: Dict) -> str:
        """Add cross-validation results to report"""
        
        report += f"Number of folds: {results.get('n_folds', 0)}\n"
        
        # Key metrics
        key_metrics = [
            'test_accuracy', 'test_f1', 'test_confidence', 'test_regime_stability'
        ]
        
        for metric in key_metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            
            if mean_key in results:
                mean_val = results[mean_key]
                std_val = results.get(std_key, 0)
                report += f"{metric}: {mean_val:.3f} ± {std_val:.3f}\n"
                
        return report
    
    def _add_wf_report(self, report: str, results: Dict) -> str:
        """Add walk-forward results to report"""
        
        report += f"Number of steps: {results.get('n_steps', 0)}\n"
        report += f"Average confidence: {results.get('avg_confidence', 0):.3f}\n"
        report += f"Confidence trend: {results.get('confidence_trend', 0):.3f}\n"
        report += f"Average stability: {results.get('avg_stability', 0):.3f}\n"
        report += f"Stability trend: {results.get('stability_trend', 0):.3f}\n"
        
        return report
    
    def _add_oos_report(self, report: str, results: Dict) -> str:
        """Add out-of-sample results to report"""
        
        report += f"Training period: {results.get('train_period', 'N/A')}\n"
        report += f"Test period: {results.get('test_period', 'N/A')}\n"
        
        # Key metrics
        if 'test_accuracy' in results:
            report += f"Test accuracy: {results['test_accuracy']:.3f}\n"
        if 'test_f1' in results:
            report += f"Test F1: {results['test_f1']:.3f}\n"
        if 'test_confidence' in results:
            report += f"Test confidence: {results['test_confidence']:.3f}\n"
            
        return report