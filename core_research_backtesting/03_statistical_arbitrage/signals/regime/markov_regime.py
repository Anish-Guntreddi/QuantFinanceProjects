"""
Markov Regime Switching Model Implementation

Hidden Markov Models for detecting market regimes in statistical arbitrage.
Different regimes may have different cointegration parameters, volatilities,
and mean reversion speeds.

Mathematical Foundation:
- Hidden states S_t ∈ {1, 2, ..., K}
- Transition matrix P_{ij} = P(S_t = j | S_{t-1} = i)
- Emission distributions: X_t | S_t ~ f(x | θ_{S_t})

Applications:
1. Detect regime changes in spread behavior
2. Adjust trading parameters by regime
3. Filter signals during unstable regimes
4. Regime-dependent position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from hmmlearn import hmm
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')


class MarkovRegimeDetector:
    """Detect market regimes using Hidden Markov Models"""
    
    def __init__(self, n_regimes: int = 2, covariance_type: str = "full"):
        """
        Initialize Markov regime detector
        
        Args:
            n_regimes: Number of hidden regimes
            covariance_type: Type of covariance parameters ("full", "diag", "spherical")
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.model = None
        self.regime_history = []
        self.regime_characteristics = {}
        self.fitted_data = None
        
    def fit(
        self,
        data: Union[pd.Series, pd.DataFrame],
        n_iter: int = 100,
        random_state: Optional[int] = 42
    ) -> pd.Series:
        """
        Fit HMM to data and detect regimes
        
        Args:
            data: Time series data (univariate or multivariate)
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
            
        Returns:
            Series of regime labels
        """
        
        # Prepare data
        if isinstance(data, pd.Series):
            data_df = data.to_frame()
            is_univariate = True
        else:
            data_df = data.copy()
            is_univariate = False
        
        # Remove NaN values
        data_clean = data_df.dropna()
        if len(data_clean) < self.n_regimes * 10:
            raise ValueError(f"Need at least {self.n_regimes * 10} observations for {self.n_regimes} regimes")
        
        self.fitted_data = data_clean
        X = data_clean.values
        
        # Initialize and fit HMM
        if is_univariate:
            # For univariate data, reshape to (n_samples, 1)
            X = X.reshape(-1, 1)
            
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        
        try:
            self.model.fit(X)
        except Exception as e:
            raise ValueError(f"HMM fitting failed: {str(e)}")
        
        # Predict regime sequence
        regime_sequence = self.model.predict(X)
        
        # Create regime series
        regime_series = pd.Series(
            regime_sequence,
            index=data_clean.index,
            name='regime'
        )
        
        # Analyze regime characteristics
        self.regime_characteristics = self._analyze_regimes(data_clean, regime_series)
        
        # Store results
        self.regime_history.append({
            'timestamp': pd.Timestamp.now(),
            'regime_series': regime_series,
            'model_params': self._extract_model_parameters(),
            'characteristics': self.regime_characteristics
        })
        
        return regime_series
    
    def _analyze_regimes(
        self,
        data: pd.DataFrame,
        regimes: pd.Series
    ) -> Dict:
        """Analyze characteristics of each regime"""
        
        characteristics = {}
        
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Basic statistics
            regime_stats = {
                'frequency': regime_mask.mean(),
                'n_observations': len(regime_data),
                'mean': regime_data.mean(),
                'std': regime_data.std(),
                'min': regime_data.min(),
                'max': regime_data.max()
            }
            
            # Return statistics (if data looks like prices/returns)
            if len(regime_data) > 1:
                if isinstance(regime_data, pd.DataFrame):
                    returns = regime_data.pct_change().dropna()
                else:
                    returns = regime_data.pct_change().dropna()
                
                if len(returns) > 0:
                    regime_stats['return_mean'] = returns.mean()
                    regime_stats['return_std'] = returns.std()
                    regime_stats['annualized_vol'] = returns.std() * np.sqrt(252)
                    
                    if isinstance(returns, pd.DataFrame):
                        regime_stats['correlation_matrix'] = returns.corr()
                        regime_stats['sharpe_ratio'] = (returns.mean() / returns.std() * np.sqrt(252))
                    else:
                        regime_stats['sharpe_ratio'] = (returns.mean() / returns.std() * np.sqrt(252))
            
            # Duration analysis
            regime_stats['avg_duration'] = self._calculate_average_duration(regimes, regime)
            regime_stats['persistence'] = self._calculate_persistence(regimes, regime)
            
            characteristics[f'regime_{regime}'] = regime_stats
        
        # Cross-regime analysis
        characteristics['regime_transitions'] = self._analyze_transitions(regimes)
        characteristics['transition_matrix'] = self._calculate_transition_matrix(regimes)
        
        return characteristics
    
    def _calculate_average_duration(self, regimes: pd.Series, regime: int) -> float:
        """Calculate average duration of a regime"""
        
        regime_indicator = (regimes == regime).astype(int)
        
        # Find regime runs
        runs = []
        current_run = 0
        
        for indicator in regime_indicator:
            if indicator == 1:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
        
        # Don't forget the last run
        if current_run > 0:
            runs.append(current_run)
        
        return np.mean(runs) if runs else 0
    
    def _calculate_persistence(self, regimes: pd.Series, regime: int) -> float:
        """Calculate persistence probability P(S_t = regime | S_{t-1} = regime)"""
        
        if len(regimes) < 2:
            return 0
        
        regime_today = regimes.iloc[1:] == regime
        regime_yesterday = regimes.iloc[:-1] == regime
        
        # P(regime today | regime yesterday)
        both_regime = regime_today & regime_yesterday
        return both_regime.sum() / regime_yesterday.sum() if regime_yesterday.sum() > 0 else 0
    
    def _analyze_transitions(self, regimes: pd.Series) -> Dict:
        """Analyze regime transition patterns"""
        
        if len(regimes) < 2:
            return {}
        
        transitions = {}
        regime_changes = regimes.iloc[1:] != regimes.iloc[:-1]
        n_transitions = regime_changes.sum()
        
        transitions['total_transitions'] = n_transitions
        transitions['transition_frequency'] = n_transitions / len(regimes)
        
        # Transition timing analysis
        if n_transitions > 0:
            transition_indices = regimes.index[1:][regime_changes]
            if len(transition_indices) > 1:
                time_between_transitions = np.diff(transition_indices)
                if hasattr(time_between_transitions[0], 'days'):
                    # Handle datetime differences
                    time_between_transitions = [t.days for t in time_between_transitions]
                
                transitions['avg_time_between_transitions'] = np.mean(time_between_transitions)
                transitions['std_time_between_transitions'] = np.std(time_between_transitions)
        
        return transitions
    
    def _calculate_transition_matrix(self, regimes: pd.Series) -> np.ndarray:
        """Calculate regime transition matrix"""
        
        if len(regimes) < 2:
            return np.eye(self.n_regimes)
        
        transition_counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            transition_counts[from_regime, to_regime] += 1
        
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = np.divide(
            transition_counts,
            row_sums.reshape(-1, 1),
            out=np.zeros_like(transition_counts),
            where=row_sums.reshape(-1, 1) != 0
        )
        
        return transition_matrix
    
    def _extract_model_parameters(self) -> Dict:
        """Extract fitted HMM parameters"""
        
        if self.model is None:
            return {}
        
        params = {
            'means': self.model.means_,
            'covariances': self.model.covars_,
            'transition_matrix': self.model.transmat_,
            'start_probabilities': self.model.startprob_,
            'log_likelihood': getattr(self.model, 'score_', None)
        }
        
        return params
    
    def predict_regime_probabilities(
        self,
        data: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Predict regime probabilities for new data
        
        Args:
            data: New time series data
            
        Returns:
            DataFrame with regime probabilities
        """
        
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Prepare data
        if isinstance(data, pd.Series):
            data_array = data.values.reshape(-1, 1)
        else:
            data_array = data.values
        
        # Predict probabilities
        try:
            probabilities = self.model.predict_proba(data_array)
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
        
        # Create DataFrame
        prob_df = pd.DataFrame(
            probabilities,
            index=data.index,
            columns=[f'regime_{i}_prob' for i in range(self.n_regimes)]
        )
        
        return prob_df
    
    def detect_regime_changes(
        self,
        regime_series: pd.Series,
        min_duration: int = 5,
        confidence_threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Detect significant regime changes
        
        Args:
            regime_series: Series of regime labels
            min_duration: Minimum duration to consider a regime change significant
            confidence_threshold: Minimum confidence for regime assignment
            
        Returns:
            DataFrame with regime change events
        """
        
        changes = []
        current_regime = regime_series.iloc[0]
        regime_start = regime_series.index[0]
        
        for i in range(1, len(regime_series)):
            if regime_series.iloc[i] != current_regime:
                # Regime change detected
                duration = i - regime_series.index.get_loc(regime_start)
                
                if duration >= min_duration:
                    changes.append({
                        'change_date': regime_series.index[i],
                        'from_regime': current_regime,
                        'to_regime': regime_series.iloc[i],
                        'previous_duration': duration,
                        'regime_start': regime_start
                    })
                
                current_regime = regime_series.iloc[i]
                regime_start = regime_series.index[i]
        
        if changes:
            changes_df = pd.DataFrame(changes)
            changes_df.set_index('change_date', inplace=True)
            return changes_df
        else:
            return pd.DataFrame()
    
    def get_regime_filtered_signals(
        self,
        signals: pd.Series,
        regime_series: pd.Series,
        allowed_regimes: List[int]
    ) -> pd.Series:
        """
        Filter trading signals based on regime
        
        Args:
            signals: Trading signals
            regime_series: Regime labels
            allowed_regimes: List of regimes where trading is allowed
            
        Returns:
            Filtered signals
        """
        
        # Align series
        aligned_data = pd.concat([signals, regime_series], axis=1).dropna()
        aligned_data.columns = ['signals', 'regimes']
        
        # Filter signals
        regime_mask = aligned_data['regimes'].isin(allowed_regimes)
        filtered_signals = aligned_data['signals'].copy()
        filtered_signals[~regime_mask] = 0
        
        return filtered_signals
    
    def backtest_regime_model(
        self,
        test_data: Union[pd.Series, pd.DataFrame],
        regime_labels: pd.Series
    ) -> Dict:
        """
        Backtest regime model predictions
        
        Args:
            test_data: Out-of-sample test data
            regime_labels: True regime labels
            
        Returns:
            Dictionary with backtesting results
        """
        
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Predict regimes on test data
        predicted_regimes = self.model.predict(test_data.values.reshape(-1, 1) if isinstance(test_data, pd.Series) else test_data.values)
        predicted_series = pd.Series(predicted_regimes, index=test_data.index)
        
        # Align with true labels
        aligned_data = pd.concat([predicted_series.rename('predicted'), regime_labels.rename('actual')], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return {'error': 'No overlapping data for backtesting'}
        
        # Calculate accuracy metrics
        accuracy = (aligned_data['predicted'] == aligned_data['actual']).mean()
        
        # Confusion matrix
        confusion_matrix = pd.crosstab(
            aligned_data['actual'],
            aligned_data['predicted'],
            margins=True
        )
        
        # Regime-specific accuracy
        regime_accuracy = {}
        for regime in range(self.n_regimes):
            regime_mask = aligned_data['actual'] == regime
            if regime_mask.sum() > 0:
                regime_accuracy[f'regime_{regime}'] = (
                    aligned_data.loc[regime_mask, 'predicted'] == regime
                ).mean()
        
        return {
            'overall_accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'regime_accuracy': regime_accuracy,
            'n_test_observations': len(aligned_data)
        }
    
    def get_current_regime_characteristics(
        self,
        recent_data: Union[pd.Series, pd.DataFrame],
        lookback_periods: int = 50
    ) -> Dict:
        """
        Get characteristics of current regime
        
        Args:
            recent_data: Recent time series data
            lookback_periods: Number of recent periods to analyze
            
        Returns:
            Dictionary with current regime characteristics
        """
        
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Get recent data
        if len(recent_data) > lookback_periods:
            recent_subset = recent_data.tail(lookback_periods)
        else:
            recent_subset = recent_data
        
        # Predict current regime
        if isinstance(recent_subset, pd.Series):
            data_array = recent_subset.values.reshape(-1, 1)
        else:
            data_array = recent_subset.values
        
        regime_probs = self.model.predict_proba(data_array)
        current_regime = self.model.predict(data_array)[-1]
        
        # Get regime characteristics
        current_characteristics = {}
        
        if f'regime_{current_regime}' in self.regime_characteristics:
            current_characteristics = self.regime_characteristics[f'regime_{current_regime}'].copy()
        
        # Add current probabilities
        current_characteristics['current_regime'] = current_regime
        current_characteristics['regime_probabilities'] = {
            f'regime_{i}': regime_probs[-1, i] for i in range(self.n_regimes)
        }
        current_characteristics['regime_confidence'] = np.max(regime_probs[-1])
        
        return current_characteristics
    
    def plot_regime_analysis(
        self,
        data: Union[pd.Series, pd.DataFrame],
        regime_series: pd.Series
    ):
        """Plot regime analysis results"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            
            # Plot 1: Time series with regime coloring
            if isinstance(data, pd.Series):
                plot_data = data
            else:
                plot_data = data.iloc[:, 0]  # Use first column
            
            for regime in range(self.n_regimes):
                regime_mask = regime_series == regime
                axes[0, 0].scatter(
                    plot_data.index[regime_mask],
                    plot_data[regime_mask],
                    label=f'Regime {regime}',
                    alpha=0.6,
                    s=10
                )
            
            axes[0, 0].set_title('Time Series by Regime')
            axes[0, 0].legend()
            
            # Plot 2: Regime transitions
            axes[0, 1].plot(regime_series.index, regime_series, marker='o', markersize=2)
            axes[0, 1].set_title('Regime Sequence')
            axes[0, 1].set_ylabel('Regime')
            
            # Plot 3: Regime duration histogram
            durations = []
            current_regime = regime_series.iloc[0]
            current_duration = 1
            
            for i in range(1, len(regime_series)):
                if regime_series.iloc[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = regime_series.iloc[i]
                    current_duration = 1
            
            if durations:
                axes[1, 0].hist(durations, bins=20, alpha=0.7)
                axes[1, 0].set_title('Regime Duration Distribution')
                axes[1, 0].set_xlabel('Duration (periods)')
            
            # Plot 4: Transition matrix heatmap
            if 'transition_matrix' in self.regime_characteristics:
                transition_matrix = self.regime_characteristics['transition_matrix']
                sns.heatmap(
                    transition_matrix,
                    annot=True,
                    cmap='Blues',
                    ax=axes[1, 1]
                )
                axes[1, 1].set_title('Transition Matrix')
            
            # Plot 5 & 6: Regime statistics
            regime_stats_data = []
            for regime in range(self.n_regimes):
                if f'regime_{regime}' in self.regime_characteristics:
                    stats = self.regime_characteristics[f'regime_{regime}']
                    regime_stats_data.append({
                        'regime': regime,
                        'frequency': stats.get('frequency', 0),
                        'avg_duration': stats.get('avg_duration', 0),
                        'volatility': stats.get('return_std', 0) if 'return_std' in stats else stats.get('std', 0)
                    })
            
            if regime_stats_data:
                stats_df = pd.DataFrame(regime_stats_data)
                
                # Frequency bar plot
                axes[2, 0].bar(stats_df['regime'], stats_df['frequency'])
                axes[2, 0].set_title('Regime Frequencies')
                axes[2, 0].set_xlabel('Regime')
                axes[2, 0].set_ylabel('Frequency')
                
                # Volatility comparison
                axes[2, 1].bar(stats_df['regime'], stats_df['volatility'])
                axes[2, 1].set_title('Regime Volatilities')
                axes[2, 1].set_xlabel('Regime')
                axes[2, 1].set_ylabel('Volatility')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib and seaborn not available for plotting")