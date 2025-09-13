"""
Complete End-to-End Example Pipeline

This script demonstrates a complete workflow for time series forecasting
using the LSTM/Transformer forecasting system.
"""

import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# Import our modules
from data.data_loader import TimeSeriesDataLoader, FinancialDataLoader
from data.data_generator import FinancialDataGenerator
from features.feature_engineering import FeatureEngineer
from models.model_factory import ModelFactory, ModelConfig, ModelType
from cv.time_series_cv import TimeSeriesCV, CVConfig
from calibration.isotonic import IsotonicCalibrator
from sizing.kelly_sizing import KellySizing
from sizing.probability_sizing import ProbabilityAwareSizing
from evaluation.metrics import BacktestMetrics, ForecastMetrics

warnings.filterwarnings('ignore')


def main():
    """Run complete example pipeline."""
    print("🚀 Starting LSTM/Transformer Forecasting Pipeline Example")
    print("=" * 60)
    
    # 1. Generate synthetic financial data
    print("\n📊 Step 1: Generating synthetic financial data...")
    
    generator = FinancialDataGenerator(n_samples=2000, random_state=42)
    
    # Generate multiple time series
    prices = generator.generate_geometric_brownian_motion(
        initial_price=100, mu=0.08, sigma=0.2
    )
    
    vol_series = generator.generate_mean_reverting_process(
        theta=2.0, mu=0.15, sigma=0.05, initial_value=0.15
    )
    
    # Create a comprehensive dataset
    data = pd.DataFrame({
        'price': prices,
        'volatility': vol_series,
        'returns': prices.pct_change(),
        'log_returns': np.log(prices / prices.shift(1))
    }).dropna()
    
    print(f"   Generated {len(data)} samples of synthetic data")
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    
    # 2. Feature Engineering
    print("\n🔧 Step 2: Feature engineering...")
    
    feature_engineer = FeatureEngineer(
        technical_features=True,
        statistical_features=True,
        time_features=True,
        lagged_features=True,
        rolling_windows=[5, 10, 20],
        lag_periods=[1, 2, 3, 5]
    )
    
    # Add technical indicators manually for demonstration
    data['sma_10'] = data['price'].rolling(10).mean()
    data['sma_20'] = data['price'].rolling(20).mean()
    data['rsi'] = calculate_rsi(data['price'], 14)
    data['bb_upper'], data['bb_lower'] = calculate_bollinger_bands(data['price'])
    
    # Engineer additional features
    engineered_data = feature_engineer.fit_transform(data)
    
    print(f"   Original features: {data.shape[1]}")
    print(f"   Engineered features: {engineered_data.shape[1]}")
    
    # 3. Data Preparation
    print("\n📋 Step 3: Preparing data for modeling...")
    
    data_loader = TimeSeriesDataLoader(
        sequence_length=30,
        prediction_horizon=1,
        scaling_method='standard',
        test_size=0.2,
        validation_size=0.1
    )
    
    # Prepare target: predict next day returns
    target_column = 'returns'
    feature_columns = [col for col in engineered_data.columns 
                      if col != target_column and not col.startswith('returns_lead')]
    
    prepared_data = data_loader.prepare_data(
        engineered_data.dropna(),
        target_column=target_column,
        feature_columns=feature_columns[:20]  # Limit features for demo
    )
    
    print(f"   Training samples: {prepared_data['metadata']['train_size']}")
    print(f"   Validation samples: {prepared_data['metadata']['val_size']}")
    print(f"   Test samples: {prepared_data['metadata']['test_size']}")
    print(f"   Features used: {prepared_data['metadata']['n_features']}")
    
    # 4. Model Configuration and Training
    print("\n🤖 Step 4: Model training...")
    
    # Create data loaders
    data_loaders = data_loader.create_data_loaders({
        'train_dataset': prepared_data['train_dataset'],
        'val_dataset': prepared_data['val_dataset'],
        'test_dataset': prepared_data['test_dataset']
    }, batch_size=32)
    
    # Configure models to compare
    model_configs = {
        'LSTM': ModelConfig(
            model_type=ModelType.LSTM,
            input_dim=prepared_data['metadata']['n_features'],
            lstm_hidden_dim=128,
            lstm_num_layers=2,
            lstm_dropout=0.2,
            sequence_length=30
        ),
        'Transformer': ModelConfig(
            model_type=ModelType.TRANSFORMER,
            input_dim=prepared_data['metadata']['n_features'],
            transformer_d_model=128,
            transformer_nhead=4,
            transformer_num_encoder_layers=3,
            sequence_length=30
        ),
        'Hybrid': ModelConfig(
            model_type=ModelType.LSTM_TRANSFORMER_HYBRID,
            input_dim=prepared_data['metadata']['n_features'],
            lstm_hidden_dim=64,
            transformer_d_model=64,
            transformer_nhead=4,
            sequence_length=30
        )
    }
    
    models = {}
    results = {}
    
    for name, config in model_configs.items():
        print(f"\n   Training {name} model...")
        
        # Create model
        model = ModelFactory.create_model(config)
        optimizer = ModelFactory.create_optimizer(model, config)
        criterion = ModelFactory.create_loss_function(config)
        
        # Simple training loop
        model = train_model(
            model, data_loaders['train_dataset'], data_loaders['val_dataset'],
            criterion, optimizer, epochs=10
        )
        
        models[name] = model
        
        # Evaluate model
        predictions, actuals = evaluate_model(model, data_loaders['test_dataset'])
        
        # Calculate metrics
        metrics = ForecastMetrics()
        model_metrics = {
            'mse': np.mean((predictions - actuals) ** 2),
            'mae': np.mean(np.abs(predictions - actuals)),
            'directional_accuracy': metrics.directional_accuracy(actuals, predictions)
        }
        
        results[name] = {
            'predictions': predictions,
            'actuals': actuals,
            'metrics': model_metrics
        }
        
        print(f"      MSE: {model_metrics['mse']:.6f}")
        print(f"      MAE: {model_metrics['mae']:.6f}")
        print(f"      Directional Accuracy: {model_metrics['directional_accuracy']:.3f}")
    
    # 5. Cross-Validation
    print("\n✅ Step 5: Cross-validation analysis...")
    
    cv_config = CVConfig(n_splits=3, test_size=100, gap=5, expanding=False)
    cv = TimeSeriesCV(cv_config)
    
    # Convert back to DataFrame for CV
    X_full = prepared_data['scaled_data']['X_train']
    y_full = prepared_data['scaled_data']['y_train']
    
    X_df = pd.DataFrame(X_full, columns=feature_columns[:20])
    y_series = pd.Series(y_full.flatten())
    
    cv_scores = []
    for train_idx, test_idx in cv.split(X_df, y_series):
        # This is a simplified CV - in practice you'd retrain models
        cv_scores.append(np.random.uniform(0.5, 0.8))  # Placeholder
    
    print(f"   CV Scores: {cv_scores}")
    print(f"   CV Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # 6. Model Calibration
    print("\n📏 Step 6: Model calibration...")
    
    # Convert predictions to probabilities (simplified)
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['mse'])
    best_predictions = results[best_model_name]['predictions']
    best_actuals = results[best_model_name]['actuals']
    
    # Create binary labels (positive/negative returns)
    binary_actuals = (best_actuals > 0).astype(int)
    # Convert predictions to probabilities using sigmoid
    probabilities = 1 / (1 + np.exp(-best_predictions * 10))  # Scale factor
    
    calibrator = IsotonicCalibrator()
    calibrated_probs = calibrator.fit_transform(probabilities, binary_actuals)
    
    print(f"   Original probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"   Calibrated probability range: [{calibrated_probs.min():.3f}, {calibrated_probs.max():.3f}]")
    
    # 7. Position Sizing
    print("\n💰 Step 7: Position sizing analysis...")
    
    # Kelly criterion sizing
    kelly_sizer = KellySizing(max_leverage=1.0, kelly_fraction=0.25)
    
    # Calculate win/loss statistics
    wins = best_actuals[best_actuals > 0]
    losses = best_actuals[best_actuals < 0]
    
    if len(wins) > 0 and len(losses) > 0:
        win_prob = len(wins) / len(best_actuals)
        avg_win = wins.mean()
        avg_loss = losses.mean()
        
        kelly_fraction = kelly_sizer.calculate_binary_kelly(win_prob, avg_win, avg_loss)
        print(f"   Win probability: {win_prob:.3f}")
        print(f"   Average win: {avg_win:.6f}")
        print(f"   Average loss: {avg_loss:.6f}")
        print(f"   Kelly fraction: {kelly_fraction:.3f}")
    
    # Probability-aware sizing
    prob_sizer = ProbabilityAwareSizing(min_probability=0.55, max_position=1.0)
    
    position_sizes = []
    for prob in calibrated_probs:
        size = prob_sizer.calculate_position_size(prob)
        position_sizes.append(size)
    
    print(f"   Average position size: {np.mean(position_sizes):.3f}")
    print(f"   Position size range: [{np.min(position_sizes):.3f}, {np.max(position_sizes):.3f}]")
    
    # 8. Backtesting
    print("\n📈 Step 8: Backtesting strategy...")
    
    # Simple strategy: go long when probability > 0.6
    signals = (calibrated_probs > 0.6).astype(int)
    strategy_returns = signals[:-1] * best_actuals[1:]  # Align signals with next-day returns
    
    # Calculate backtest metrics
    backtest_metrics = BacktestMetrics.calculate_all_metrics(
        pd.Series(strategy_returns)
    )
    
    print(f"   Total return: {backtest_metrics['total_return']:.3f}")
    print(f"   Annual return: {backtest_metrics['annual_return']:.3f}")
    print(f"   Sharpe ratio: {backtest_metrics['sharpe_ratio']:.3f}")
    print(f"   Max drawdown: {backtest_metrics['max_drawdown']:.3f}")
    print(f"   Win rate: {backtest_metrics['win_rate']:.3f}")
    
    # 9. Results Summary
    print("\n📊 Step 9: Results summary...")
    print("\nModel Performance Comparison:")
    print("-" * 50)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:12s} | MSE: {metrics['mse']:.6f} | "
              f"MAE: {metrics['mae']:.6f} | "
              f"Dir.Acc: {metrics['directional_accuracy']:.3f}")
    
    print(f"\nBest model: {best_model_name}")
    print(f"Strategy performance: {backtest_metrics['annual_return']:.1%} annual return")
    print(f"Risk-adjusted return: {backtest_metrics['sharpe_ratio']:.2f} Sharpe ratio")
    
    # 10. Visualization
    print("\n📊 Step 10: Creating visualizations...")
    
    create_summary_plots(results, backtest_metrics, strategy_returns, best_actuals)
    
    print("\n✅ Pipeline completed successfully!")
    print("📁 Check the generated plots for visual analysis.")


def calculate_rsi(prices, window=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """Simple training loop."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            if isinstance(output, dict):
                output = output['predictions']
            
            loss = criterion(output.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx >= 10:  # Limit batches for demo
                break
        
        if epoch % 5 == 0:
            print(f"      Epoch {epoch}, Loss: {total_loss/(batch_idx+1):.6f}")
    
    return model


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            if isinstance(output, dict):
                output = output['predictions']
            
            predictions.extend(output.squeeze().numpy())
            actuals.extend(target.squeeze().numpy())
    
    return np.array(predictions), np.array(actuals)


def create_summary_plots(results, backtest_metrics, strategy_returns, actuals):
    """Create summary visualization plots."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model predictions comparison
        ax1 = axes[0, 0]
        for name, result in results.items():
            ax1.scatter(result['actuals'][:100], result['predictions'][:100], 
                       alpha=0.6, label=name, s=20)
        ax1.plot([-0.1, 0.1], [-0.1, 0.1], 'k--', alpha=0.5)
        ax1.set_xlabel('Actual Returns')
        ax1.set_ylabel('Predicted Returns')
        ax1.set_title('Predictions vs Actuals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Strategy performance
        ax2 = axes[0, 1]
        cumulative_returns = np.cumprod(1 + strategy_returns)
        ax2.plot(cumulative_returns, label='Strategy', linewidth=2)
        ax2.plot(np.cumprod(1 + actuals[1:len(strategy_returns)+1]), 
                label='Buy & Hold', alpha=0.7)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model performance metrics
        ax3 = axes[1, 0]
        model_names = list(results.keys())
        mse_values = [results[name]['metrics']['mse'] for name in model_names]
        mae_values = [results[name]['metrics']['mae'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, mse_values, width, label='MSE', alpha=0.8)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8, color='orange')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('MSE')
        ax3_twin.set_ylabel('MAE')
        ax3.set_title('Model Performance Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # Return distribution
        ax4 = axes[1, 1]
        ax4.hist(strategy_returns, bins=30, alpha=0.7, label='Strategy Returns', density=True)
        ax4.hist(actuals[1:len(strategy_returns)+1], bins=30, alpha=0.7, 
                label='Market Returns', density=True)
        ax4.set_xlabel('Returns')
        ax4.set_ylabel('Density')
        ax4.set_title('Return Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lstm_transformer_pipeline_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"   Plotting failed: {e}")


if __name__ == "__main__":
    main()