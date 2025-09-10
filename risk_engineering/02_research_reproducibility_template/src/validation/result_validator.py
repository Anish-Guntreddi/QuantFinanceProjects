"""Result validation for experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings


class ResultValidator:
    """Validate experiment results."""
    
    def __init__(self, baseline_results: Optional[Dict] = None):
        self.baseline_results = baseline_results
        self.validation_tests = []
        self.validation_results = {}
        
    def validate_statistical_significance(
        self,
        results: np.ndarray,
        baseline: np.ndarray,
        test: str = 'wilcoxon',
        alpha: float = 0.05
    ) -> Dict:
        """Test statistical significance of results."""
        validation = {
            'test': test,
            'alpha': alpha
        }
        
        try:
            if test == 'wilcoxon':
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(results, baseline)
            elif test == 't-test':
                # Paired t-test
                statistic, p_value = stats.ttest_rel(results, baseline)
            elif test == 'mann-whitney':
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(results, baseline)
            elif test == 'ks':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(results, baseline)
            else:
                raise ValueError(f"Unknown test: {test}")
                
            validation.update({
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'effect_size': self._calculate_effect_size(results, baseline)
            })
            
        except Exception as e:
            validation.update({
                'error': str(e),
                'significant': False
            })
            
        self.validation_results[f'significance_{test}'] = validation
        return validation
    
    def validate_performance_bounds(
        self,
        metrics: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict:
        """Validate if metrics are within expected bounds."""
        violations = []
        warnings_list = []
        
        for metric, value in metrics.items():
            if metric in bounds:
                lower, upper = bounds[metric]
                
                if value < lower:
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'bound': 'lower',
                        'threshold': lower,
                        'violation_pct': abs((value - lower) / lower * 100)
                    })
                elif value > upper:
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'bound': 'upper',
                        'threshold': upper,
                        'violation_pct': abs((value - upper) / upper * 100)
                    })
                    
                # Check for near-boundary warnings
                margin = 0.1  # 10% margin
                if lower * (1 + margin) > value > lower:
                    warnings_list.append({
                        'metric': metric,
                        'value': value,
                        'message': f'Close to lower bound ({lower})'
                    })
                elif upper > value > upper * (1 - margin):
                    warnings_list.append({
                        'metric': metric,
                        'value': value,
                        'message': f'Close to upper bound ({upper})'
                    })
                    
        validation = {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings_list,
            'metrics_checked': len(bounds),
            'metrics_valid': len(bounds) - len(violations)
        }
        
        self.validation_results['performance_bounds'] = validation
        return validation
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict:
        """Validate data integrity."""
        issues = []
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            missing_cols = missing[missing > 0]
            issues.append({
                'type': 'missing_values',
                'severity': 'warning',
                'details': {
                    'columns': missing_cols.to_dict(),
                    'total_missing': int(missing.sum()),
                    'pct_missing': float(missing.sum() / len(data) * 100)
                }
            })
            
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            issues.append({
                'type': 'duplicates',
                'severity': 'warning',
                'count': int(duplicates),
                'pct': float(duplicates / len(data) * 100)
            })
            
        # Check for constant columns
        constant_cols = [col for col in data.columns if data[col].nunique() == 1]
        if constant_cols:
            issues.append({
                'type': 'constant_columns',
                'severity': 'info',
                'columns': constant_cols
            })
            
        # Check for outliers
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            
            if outliers > 0:
                outlier_info[col] = {
                    'count': int(outliers),
                    'pct': float(outliers / len(data) * 100),
                    'bounds': (float(lower_bound), float(upper_bound))
                }
                
        if outlier_info:
            issues.append({
                'type': 'outliers',
                'severity': 'info',
                'columns': outlier_info
            })
            
        # Check data types
        expected_types = self._infer_expected_types(data)
        type_issues = []
        
        for col, expected_type in expected_types.items():
            actual_type = str(data[col].dtype)
            if not self._types_compatible(actual_type, expected_type):
                type_issues.append({
                    'column': col,
                    'expected': expected_type,
                    'actual': actual_type
                })
                
        if type_issues:
            issues.append({
                'type': 'type_mismatch',
                'severity': 'warning',
                'columns': type_issues
            })
            
        validation = {
            'valid': len([i for i in issues if i.get('severity') == 'error']) == 0,
            'issues': issues,
            'data_shape': data.shape,
            'memory_usage': float(data.memory_usage(deep=True).sum() / 1024**2)  # MB
        }
        
        self.validation_results['data_integrity'] = validation
        return validation
    
    def validate_model_assumptions(
        self,
        residuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict:
        """Validate model assumptions."""
        assumptions = {}
        
        # 1. Normality of residuals
        _, normality_p = stats.normaltest(residuals)
        assumptions['normality'] = {
            'p_value': float(normality_p),
            'passed': normality_p > 0.05,
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals))
        }
        
        # 2. Homoscedasticity (constant variance)
        # Breusch-Pagan test approximation
        residuals_squared = residuals ** 2
        correlation = np.corrcoef(predictions, residuals_squared)[0, 1]
        assumptions['homoscedasticity'] = {
            'correlation': float(correlation),
            'passed': abs(correlation) < 0.3
        }
        
        # 3. Independence (Durbin-Watson test)
        dw_stat = self._durbin_watson(residuals)
        assumptions['independence'] = {
            'durbin_watson': float(dw_stat),
            'passed': 1.5 < dw_stat < 2.5
        }
        
        # 4. Linearity (check residual pattern)
        # Simple check: correlation between residuals and predictions
        linearity_corr = np.corrcoef(predictions, residuals)[0, 1]
        assumptions['linearity'] = {
            'correlation': float(linearity_corr),
            'passed': abs(linearity_corr) < 0.1
        }
        
        validation = {
            'all_passed': all(a['passed'] for a in assumptions.values()),
            'assumptions': assumptions
        }
        
        self.validation_results['model_assumptions'] = validation
        return validation
    
    def validate_cross_validation(
        self,
        cv_scores: List[float],
        min_acceptable_score: float = 0.0
    ) -> Dict:
        """Validate cross-validation results."""
        cv_array = np.array(cv_scores)
        
        validation = {
            'n_folds': len(cv_scores),
            'mean_score': float(cv_array.mean()),
            'std_score': float(cv_array.std()),
            'min_score': float(cv_array.min()),
            'max_score': float(cv_array.max()),
            'cv_coefficient': float(cv_array.std() / cv_array.mean()) if cv_array.mean() != 0 else np.inf,
            'all_above_threshold': bool(cv_array.min() >= min_acceptable_score)
        }
        
        # Check for high variance
        if validation['cv_coefficient'] > 0.3:
            validation['warning'] = 'High variance in CV scores'
            
        self.validation_results['cross_validation'] = validation
        return validation
    
    def _calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
            
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return float(d)
    
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic."""
        diff = np.diff(residuals)
        return float(np.sum(diff ** 2) / np.sum(residuals ** 2))
    
    def _infer_expected_types(self, data: pd.DataFrame) -> Dict:
        """Infer expected data types."""
        expected = {}
        
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                expected[col] = 'datetime'
            elif 'id' in col.lower() or 'code' in col.lower():
                expected[col] = 'object'
            elif data[col].dtype in [np.float64, np.float32]:
                expected[col] = 'float'
            elif data[col].dtype in [np.int64, np.int32]:
                expected[col] = 'int'
            else:
                expected[col] = str(data[col].dtype)
                
        return expected
    
    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        if actual == expected:
            return True
            
        # Float can contain int
        if expected == 'int' and 'float' in actual:
            return True
            
        # Object can contain anything
        if expected == 'object' or actual == 'object':
            return True
            
        return False
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'validations_performed': list(self.validation_results.keys()),
            'overall_valid': all(
                v.get('valid', v.get('passed', True)) 
                for v in self.validation_results.values()
            ),
            'results': self.validation_results
        }
        
        return report