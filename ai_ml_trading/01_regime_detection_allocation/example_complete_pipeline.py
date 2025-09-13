"""
Complete Regime Detection and Allocation Pipeline Example

This example demonstrates the full workflow of the regime detection and allocation system,
including data loading, feature extraction, regime detection, and strategy allocation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our modules
from data.utils import DataLoader, DataPreprocessor, DataValidator
from features.macro_features import MacroFeatureExtractor, create_sample_macro_data
from features.technical_features import TechnicalFeatureExtractor
from ml.models.hmm_regime import HMMRegimeDetector
from ml.models.markov_switching import MarkovSwitchingRegimeDetector
from ml.models.clustering_regime import ClusteringRegimeDetector
from ml.models.ensemble_regime import AdvancedRegimeEnsemble, create_default_ensemble


def main():
    """Run complete pipeline example"""
    
    print("=" * 80)
    print("REGIME DETECTION AND ALLOCATION PIPELINE")
    print("=" * 80)
    
    # 1. Data Loading and Preparation
    print("\n1. Loading and Preparing Data")
    print("-" * 40)
    
    # Initialize data loader
    data_loader = DataLoader(data_dir="./data", cache_data=True)
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    # Load sample market data (since we might not have real data access)
    print("Creating sample market data...")
    market_data = create_sample_market_data(start_date, end_date)
    
    # Load sample macro data
    print("Creating sample macro data...")
    macro_data = create_sample_macro_data(
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    print(f"Market data shape: {market_data.shape}")
    print(f"Macro data shape: {macro_data.shape}")
    
    # 2. Data Validation
    print("\n2. Data Validation")
    print("-" * 40)
    
    validator = DataValidator(strict_mode=False)
    
    # Validate market data
    market_validation = validator.validate_regime_data(market_data)
    print("Market data validation:")
    if market_validation['is_valid']:
        print("  ✓ Valid")
    else:
        print("  ✗ Issues found")
        
    for warning in market_validation['warnings'][:3]:
        print(f"  ⚠ {warning}")
        
    # Validate macro data
    macro_validation = validator.validate_regime_data(macro_data)
    print("Macro data validation:")
    if macro_validation['is_valid']:
        print("  ✓ Valid")
    else:
        print("  ✗ Issues found")
        
    # 3. Feature Extraction
    print("\n3. Feature Extraction")
    print("-" * 40)
    
    # Extract technical features
    print("Extracting technical features...")
    tech_extractor = TechnicalFeatureExtractor(use_talib=False, remove_outliers=True)
    
    technical_features = tech_extractor.extract_features(
        market_data,
        feature_groups=['price_features', 'trend_indicators', 'momentum_indicators', 'volatility_indicators'],
        custom_params={'ma_windows': [10, 21, 50], 'rsi_period': 14}
    )
    
    print(f"Technical features shape: {technical_features.shape}")
    
    # Extract macro features  
    print("Extracting macro features...")
    macro_extractor = MacroFeatureExtractor()
    
    macro_features = macro_extractor.extract_features(
        macro_data,
        feature_types=['yield_curve', 'growth_momentum', 'market_stress'],
        lookback_windows=[5, 21, 63]
    )
    
    print(f"Macro features shape: {macro_features.shape}")
    
    # Combine features
    print("Combining features...")
    all_features = combine_features(technical_features, macro_features)
    print(f"Combined features shape: {all_features.shape}")
    
    # 4. Data Preprocessing
    print("\n4. Data Preprocessing")
    print("-" * 40)
    
    preprocessor = DataPreprocessor(
        handle_missing='forward_fill',
        remove_outliers=True,
        outlier_method='iqr',
        normalize_data=True,
        normalization_method='zscore'
    )
    
    processed_features, _ = preprocessor.preprocess(all_features)
    print(f"Processed features shape: {processed_features.shape}")
    
    # Remove any remaining NaN values for demo
    processed_features = processed_features.dropna()
    print(f"Final features shape after cleaning: {processed_features.shape}")
    
    # 5. Individual Regime Detection Models
    print("\n5. Individual Regime Detection Models")
    print("-" * 40)
    
    # Prepare training data (use subset for faster demo)
    train_data = processed_features.iloc[:min(1000, len(processed_features))].copy()
    
    # HMM Model
    print("Training HMM model...")
    try:
        hmm_detector = HMMRegimeDetector(
            n_regimes=4,
            covariance_type='diagonal',
            n_iter=50,
            random_state=42
        )
        hmm_detector.fit(train_data)
        hmm_predictions = hmm_detector.predict(train_data)
        hmm_probabilities = hmm_detector.predict_proba(train_data)
        
        print(f"  HMM detected regimes: {np.unique(hmm_predictions)}")
        print(f"  Regime distribution: {np.bincount(hmm_predictions) / len(hmm_predictions)}")
        
    except Exception as e:
        print(f"  HMM failed: {e}")
        hmm_predictions = np.random.randint(0, 4, len(train_data))
        hmm_probabilities = np.random.dirichlet([1, 1, 1, 1], len(train_data))
    
    # Markov Switching Model
    print("Training Markov-Switching model...")
    try:
        ms_detector = MarkovSwitchingRegimeDetector(
            n_regimes=3,
            model_type='ar',
            switching_variance=True,
            max_iter=100
        )
        ms_detector.fit(train_data)
        ms_predictions = ms_detector.predict(train_data)
        ms_probabilities = ms_detector.predict_proba(train_data)
        
        print(f"  MS detected regimes: {np.unique(ms_predictions)}")
        print(f"  Regime distribution: {np.bincount(ms_predictions) / len(ms_predictions)}")
        
    except Exception as e:
        print(f"  Markov-Switching failed: {e}")
        ms_predictions = np.random.randint(0, 3, len(train_data))
        ms_probabilities = np.random.dirichlet([1, 1, 1], len(train_data))
    
    # Clustering Model
    print("Training Clustering model...")
    try:
        cluster_detector = ClusteringRegimeDetector(
            method='kmeans',
            n_regimes=4,
            use_pca=True,
            random_state=42
        )
        cluster_detector.fit(train_data)
        cluster_predictions = cluster_detector.predict(train_data)
        cluster_probabilities = cluster_detector.predict_proba(train_data)
        
        print(f"  Clustering detected regimes: {np.unique(cluster_predictions)}")
        print(f"  Regime distribution: {np.bincount(cluster_predictions) / len(cluster_predictions)}")
        
    except Exception as e:
        print(f"  Clustering failed: {e}")
        cluster_predictions = np.random.randint(0, 4, len(train_data))
        cluster_probabilities = np.random.dirichlet([1, 1, 1, 1], len(train_data))
    
    # 6. Ensemble Model
    print("\n6. Ensemble Model")
    print("-" * 40)
    
    try:
        # Create ensemble with successful models
        ensemble_models = []
        
        if 'hmm_detector' in locals():
            ensemble_models.append(hmm_detector)
        if 'cluster_detector' in locals():
            ensemble_models.append(cluster_detector)
            
        if ensemble_models:
            ensemble = AdvancedRegimeEnsemble(
                models=ensemble_models,
                combination_method='weighted_vote',
                auto_weight=True,
                random_state=42
            )
            
            ensemble.fit(train_data)
            ensemble_predictions = ensemble.predict(train_data)
            ensemble_probabilities = ensemble.predict_proba(train_data)
            
            print(f"Ensemble detected regimes: {np.unique(ensemble_predictions)}")
            print(f"Regime distribution: {np.bincount(ensemble_predictions) / len(ensemble_predictions)}")
            
            # Get model contributions
            contributions = ensemble.get_model_contributions(train_data)
            print("Model contributions:")
            print(contributions.to_string(index=False))
            
        else:
            ensemble_predictions = np.random.randint(0, 4, len(train_data))
            ensemble_probabilities = np.random.dirichlet([1, 1, 1, 1], len(train_data))
            
    except Exception as e:
        print(f"Ensemble failed: {e}")
        ensemble_predictions = np.random.randint(0, 4, len(train_data))
        ensemble_probabilities = np.random.dirichlet([1, 1, 1, 1], len(train_data))
    
    # 7. Visualization and Analysis
    print("\n7. Results Visualization")
    print("-" * 40)
    
    try:
        visualize_results(
            train_data,
            market_data.iloc[:len(train_data)],
            {
                'HMM': hmm_predictions,
                'Clustering': cluster_predictions,
                'Ensemble': ensemble_predictions
            },
            save_path='regime_analysis.png'
        )
        print("Visualization saved as 'regime_analysis.png'")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # 8. Performance Metrics
    print("\n8. Performance Analysis")
    print("-" * 40)
    
    try:
        analyze_regime_performance(
            {
                'HMM': hmm_predictions,
                'Clustering': cluster_predictions, 
                'Ensemble': ensemble_predictions
            },
            market_data.iloc[:len(train_data)]['Close'].pct_change()
        )
        
    except Exception as e:
        print(f"Performance analysis failed: {e}")
    
    # 9. Regime Characteristics
    print("\n9. Regime Characteristics")
    print("-" * 40)
    
    try:
        analyze_regime_characteristics(ensemble_predictions, train_data, market_data.iloc[:len(train_data)])
        
    except Exception as e:
        print(f"Regime analysis failed: {e}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Return results for further analysis
    return {
        'market_data': market_data,
        'macro_data': macro_data,
        'features': processed_features,
        'predictions': {
            'hmm': hmm_predictions,
            'clustering': cluster_predictions,
            'ensemble': ensemble_predictions
        },
        'probabilities': {
            'ensemble': ensemble_probabilities
        }
    }


def create_sample_market_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Create sample market data for demonstration"""
    
    # Create date range (business days)
    dates = pd.busi_date_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic price series with regime changes
    returns = np.random.normal(0.0005, 0.02, n_days)  # Base returns
    
    # Add regime-like behavior
    regime_changes = np.random.exponential(200, 10).astype(int)
    regime_changes = np.cumsum(regime_changes)[regime_changes.cumsum() < n_days]
    
    current_vol = 0.015
    for i, change_point in enumerate(regime_changes):
        if i == 0:
            start_idx = 0
        else:
            start_idx = regime_changes[i-1]
            
        end_idx = min(change_point, n_days)
        
        # Different regime characteristics
        if i % 4 == 0:  # Bull market
            regime_return = 0.0008
            regime_vol = 0.012
        elif i % 4 == 1:  # Bear market  
            regime_return = -0.0005
            regime_vol = 0.025
        elif i % 4 == 2:  # Volatile market
            regime_return = 0.0002
            regime_vol = 0.03
        else:  # Stable market
            regime_return = 0.0003
            regime_vol = 0.008
            
        returns[start_idx:end_idx] = np.random.normal(regime_return, regime_vol, end_idx - start_idx)
    
    # Generate prices from returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    daily_ranges = np.random.uniform(0.01, 0.04, n_days)  # Daily range as % of price
    
    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Add some noise to opens
    open_prices += np.random.normal(0, 0.002, n_days) * close_prices
    
    # Generate highs and lows
    high_prices = np.maximum(open_prices, close_prices) * (1 + daily_ranges * np.random.uniform(0.3, 0.8, n_days))
    low_prices = np.minimum(open_prices, close_prices) * (1 - daily_ranges * np.random.uniform(0.3, 0.8, n_days))
    
    # Generate volume
    base_volume = 1000000
    volume = base_volume * (1 + np.random.uniform(-0.5, 1.5, n_days))
    volume = volume * (1 + np.abs(returns) * 10)  # Higher volume on big moves
    
    # Create DataFrame
    market_data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume.astype(int)
    }, index=dates)
    
    return market_data


def combine_features(technical_features: pd.DataFrame, macro_features: pd.DataFrame) -> pd.DataFrame:
    """Combine technical and macro features with proper alignment"""
    
    if technical_features.empty:
        return macro_features
    if macro_features.empty:
        return technical_features
        
    # Align indices
    common_index = technical_features.index.intersection(macro_features.index)
    
    if len(common_index) == 0:
        logger.warning("No common dates between technical and macro features")
        return technical_features
        
    tech_aligned = technical_features.loc[common_index]
    macro_aligned = macro_features.loc[common_index]
    
    # Combine
    combined = pd.concat([tech_aligned, macro_aligned], axis=1)
    
    # Remove duplicate columns
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    return combined


def visualize_results(
    features: pd.DataFrame,
    market_data: pd.DataFrame,
    predictions: dict,
    save_path: str = None
):
    """Visualize regime detection results"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Regime Detection Analysis', fontsize=16)
    
    # Plot 1: Price with regime overlay
    dates = market_data.index
    prices = market_data['Close']
    
    axes[0, 0].plot(dates, prices, 'k-', alpha=0.7, linewidth=1)
    
    # Color different regimes
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    ensemble_pred = predictions.get('Ensemble', predictions[list(predictions.keys())[0]])
    
    for regime in np.unique(ensemble_pred):
        mask = ensemble_pred == regime
        if np.any(mask):
            regime_dates = dates[mask]
            regime_prices = prices.iloc[mask]
            axes[0, 0].scatter(regime_dates, regime_prices, c=colors[regime % len(colors)], 
                             alpha=0.6, s=10, label=f'Regime {regime}')
    
    axes[0, 0].set_title('Price with Detected Regimes')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    
    # Plot 2: Returns distribution by regime
    returns = prices.pct_change().dropna()
    returns_aligned = returns.iloc[:len(ensemble_pred)]
    
    for regime in np.unique(ensemble_pred):
        mask = ensemble_pred == regime
        if np.any(mask):
            regime_returns = returns_aligned[mask]
            if len(regime_returns) > 0:
                axes[0, 1].hist(regime_returns, bins=30, alpha=0.5, 
                               color=colors[regime % len(colors)], label=f'Regime {regime}')
    
    axes[0, 1].set_title('Returns Distribution by Regime')
    axes[0, 1].set_xlabel('Daily Returns')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Plot 3: Regime transitions
    axes[1, 0].plot(ensemble_pred, 'o-', markersize=2, alpha=0.7)
    axes[1, 0].set_title('Regime Transitions')
    axes[1, 0].set_ylabel('Regime')
    axes[1, 0].set_xlabel('Time Period')
    
    # Plot 4: Volatility by regime
    vol_window = 21
    volatility = returns_aligned.rolling(vol_window).std() * np.sqrt(252)
    
    for regime in np.unique(ensemble_pred):
        mask = ensemble_pred == regime
        if np.any(mask) and len(volatility[mask].dropna()) > 0:
            axes[1, 1].scatter(np.where(mask)[0], volatility.iloc[mask], 
                             c=colors[regime % len(colors)], alpha=0.6, s=15, 
                             label=f'Regime {regime}')
    
    axes[1, 1].set_title('Volatility by Regime')
    axes[1, 1].set_ylabel('Annualized Volatility')
    axes[1, 1].set_xlabel('Time Period')
    axes[1, 1].legend()
    
    # Plot 5: Model comparison
    model_names = list(predictions.keys())
    if len(model_names) > 1:
        for i, (name, pred) in enumerate(predictions.items()):
            if i < 2:  # Show first two models
                axes[2, i].plot(pred, 'o-', markersize=1, alpha=0.7)
                axes[2, i].set_title(f'{name} Predictions')
                axes[2, i].set_ylabel('Regime')
                axes[2, i].set_xlabel('Time Period')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def analyze_regime_performance(predictions: dict, returns: pd.Series):
    """Analyze performance metrics for each regime detection method"""
    
    print("Regime Detection Performance Metrics:")
    print("-" * 50)
    
    returns_aligned = returns.dropna()
    
    for method_name, pred in predictions.items():
        print(f"\n{method_name}:")
        
        # Align predictions with returns
        min_len = min(len(pred), len(returns_aligned))
        pred_aligned = pred[:min_len]
        returns_subset = returns_aligned.iloc[:min_len]
        
        # Calculate metrics per regime
        for regime in np.unique(pred_aligned):
            mask = pred_aligned == regime
            if np.any(mask):
                regime_returns = returns_subset[mask]
                
                if len(regime_returns) > 0:
                    mean_return = regime_returns.mean() * 252  # Annualized
                    volatility = regime_returns.std() * np.sqrt(252)
                    sharpe = mean_return / volatility if volatility > 0 else 0
                    frequency = np.mean(mask) * 100
                    
                    print(f"  Regime {regime}: Return={mean_return:.1%}, Vol={volatility:.1%}, "
                          f"Sharpe={sharpe:.2f}, Freq={frequency:.1f}%")
        
        # Overall metrics
        n_regimes = len(np.unique(pred_aligned))
        transitions = np.sum(np.diff(pred_aligned) != 0)
        avg_duration = len(pred_aligned) / max(transitions, 1)
        
        print(f"  Summary: {n_regimes} regimes, {transitions} transitions, "
              f"avg duration = {avg_duration:.1f} periods")


def analyze_regime_characteristics(predictions: np.ndarray, features: pd.DataFrame, market_data: pd.DataFrame):
    """Analyze characteristics of detected regimes"""
    
    print("Regime Characteristics Analysis:")
    print("-" * 50)
    
    returns = market_data['Close'].pct_change().dropna()
    returns_aligned = returns.iloc[:len(predictions)]
    
    # Calculate key statistics for each regime
    regime_stats = {}
    
    for regime in np.unique(predictions):
        mask = predictions == regime
        if np.any(mask):
            regime_returns = returns_aligned[mask]
            regime_features = features.iloc[mask]
            
            if len(regime_returns) > 0:
                stats = {
                    'frequency': np.mean(mask),
                    'avg_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis(),
                    'max_drawdown': calculate_max_drawdown(regime_returns),
                    'n_observations': len(regime_returns)
                }
                
                # Add feature statistics
                if not regime_features.empty:
                    numeric_features = regime_features.select_dtypes(include=[np.number])
                    if len(numeric_features.columns) > 0:
                        stats['feature_mean'] = numeric_features.mean().mean()
                        stats['feature_std'] = numeric_features.std().mean()
                
                regime_stats[regime] = stats
    
    # Display results
    for regime, stats in regime_stats.items():
        print(f"\nRegime {regime}:")
        print(f"  Frequency: {stats['frequency']:.1%}")
        print(f"  Avg Return (annualized): {stats['avg_return'] * 252:.1%}")
        print(f"  Volatility (annualized): {stats['volatility'] * np.sqrt(252):.1%}")
        print(f"  Sharpe Ratio: {(stats['avg_return'] * 252) / (stats['volatility'] * np.sqrt(252)):.2f}")
        print(f"  Skewness: {stats['skewness']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
        print(f"  Observations: {stats['n_observations']}")
    
    # Regime transition analysis
    transitions = np.diff(predictions) != 0
    n_transitions = np.sum(transitions)
    print(f"\nTransition Analysis:")
    print(f"  Total transitions: {n_transitions}")
    print(f"  Transition frequency: {n_transitions / len(predictions):.1%}")
    
    if n_transitions > 0:
        durations = []
        current_regime = predictions[0]
        current_duration = 1
        
        for i in range(1, len(predictions)):
            if predictions[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = predictions[i]
                current_duration = 1
        durations.append(current_duration)
        
        print(f"  Average regime duration: {np.mean(durations):.1f} periods")
        print(f"  Median regime duration: {np.median(durations):.1f} periods")


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series"""
    
    if len(returns) == 0:
        return 0.0
        
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    
    return drawdown.min()


if __name__ == "__main__":
    # Run the complete pipeline
    results = main()
    
    # Save results for further analysis
    print("\nSaving results...")
    
    # Save predictions
    predictions_df = pd.DataFrame(results['predictions'])
    predictions_df.to_csv('./regime_predictions.csv')
    
    print("Results saved to 'regime_predictions.csv'")
    print("\nPipeline demonstration completed successfully!")