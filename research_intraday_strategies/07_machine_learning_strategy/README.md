# Machine Learning Strategy (Classification → LSTM/Transformer)

## Overview
Machine learning strategy implementation with up/down classifier, embargoed cross-validation, and calibration techniques.

## Project Structure
```
07_machine_learning_strategy/
├── ml/
│   ├── forecasting.py
│   ├── features.py
│   ├── models.py
│   └── validation.py
├── ml/
│   └── results.ipynb
├── configs/
│   └── model_config.yaml
└── tests/
    └── test_ml.py
```

## Implementation

### ml/forecasting.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ForecastConfig:
    sequence_length: int = 60
    forecast_horizon: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    embargo_period: int = 10

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                output_dim: int, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state
        out = attn_out[:, -1, :]
        
        # Final prediction
        predictions = self.fc(out)
        
        return predictions

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                num_layers: int = 4, output_dim: int = 1, dropout: float = 0.1):
        super(TransformerForecaster, self).__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Global pooling
        pooled = encoded.mean(dim=1)
        
        # Output projection
        output = self.output_projection(pooled)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class DirectionalClassifier:
    def __init__(self, config: ForecastConfig = ForecastConfig()):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series prediction"""
        features = data.drop(columns=[target_col]).values
        target = data[target_col].values
        
        X, y = [], []
        
        for i in range(self.config.sequence_length, 
                      len(features) - self.config.forecast_horizon):
            X.append(features[i - self.config.sequence_length:i])
            
            # Binary classification: up (1) or down (0)
            future_return = target[i + self.config.forecast_horizon] - target[i]
            y.append(1 if future_return > 0 else 0)
        
        return np.array(X), np.array(y)
    
    def create_model(self, input_dim: int, model_type: str = 'lstm') -> nn.Module:
        """Create neural network model"""
        if model_type == 'lstm':
            return LSTMForecaster(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                output_dim=2,  # Binary classification
                dropout=self.config.dropout
            )
        elif model_type == 'transformer':
            return TransformerForecaster(
                input_dim=input_dim,
                d_model=self.config.hidden_dim,
                output_dim=2,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        # Initialize model
        self.model = self.create_model(X_train.shape[-1]).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())
                
                # Calculate accuracy
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == y_val_tensor).float().mean().item()
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.epochs}], '
                     f'Train Loss: {avg_train_loss:.4f}, '
                     f'Val Loss: {val_loss:.4f}, '
                     f'Val Accuracy: {accuracy:.4f}')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': accuracy
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()

class EmbargoedCrossValidation:
    def __init__(self, n_splits: int = 5, embargo_period: int = 10):
        self.n_splits = n_splits
        self.embargo_period = embargo_period
        
    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits with embargo period"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        splits = []
        for train_idx, test_idx in tscv.split(X):
            # Apply embargo
            embargo_end = train_idx[-1] + self.embargo_period
            
            # Remove embargo period from training set
            train_idx = train_idx[train_idx < train_idx[-1] - self.embargo_period]
            
            # Adjust test set to start after embargo
            test_idx = test_idx[test_idx > embargo_end]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits

class ProbabilityCalibration:
    def __init__(self, method: str = 'isotonic'):
        """
        method: 'isotonic', 'sigmoid', or 'beta'
        """
        self.method = method
        self.calibrator = None
        
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """Fit calibration mapping"""
        if self.method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probabilities, true_labels)
            
        elif self.method == 'sigmoid':
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(probabilities.reshape(-1, 1), true_labels)
            
        elif self.method == 'beta':
            self.calibrator = self._fit_beta_calibration(probabilities, true_labels)
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration"""
        if self.method == 'isotonic':
            return self.calibrator.transform(probabilities)
        elif self.method == 'sigmoid':
            return self.calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        elif self.method == 'beta':
            return self._apply_beta_calibration(probabilities)
    
    def _fit_beta_calibration(self, probs: np.ndarray, labels: np.ndarray):
        """Fit beta calibration parameters"""
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            a, b = params
            # Beta calibration function
            calibrated = probs ** a / (probs ** a + (1 - probs) ** a)
            
            # Log likelihood
            eps = 1e-15
            calibrated = np.clip(calibrated, eps, 1 - eps)
            ll = labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated)
            
            return -ll.sum()
        
        result = minimize(neg_log_likelihood, x0=[1.0, 1.0], bounds=[(0.01, 10), (0.01, 10)])
        
        return result.x
    
    def _apply_beta_calibration(self, probs: np.ndarray) -> np.ndarray:
        """Apply beta calibration"""
        a, b = self.calibrator
        return probs ** a / (probs ** a + (1 - probs) ** a)
```

### ml/features.py
```python
import numpy as np
import pandas as pd
from typing import List, Optional
import talib

class FeatureEngineering:
    def __init__(self):
        self.feature_names = []
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Returns at different horizons
        for period in [1, 5, 10, 20]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Price position in range
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap features
        features['gap'] = df['open'] / df['close'].shift(1) - 1
        
        return features
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Volume ratios
        features['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(5).mean()
        features['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Dollar volume
        features['dollar_volume'] = df['close'] * df['volume']
        features['dollar_volume_ratio'] = (
            features['dollar_volume'] / features['dollar_volume'].rolling(20).mean()
        )
        
        # Volume-weighted price
        features['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['price_to_vwap'] = df['close'] / features['vwap']
        
        # On-balance volume
        features['obv'] = talib.OBV(df['close'], df['volume'])
        features['obv_change'] = features['obv'].pct_change(5)
        
        return features
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        features = pd.DataFrame(index=df.index)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = talib.SMA(df['close'], period)
            features[f'ema_{period}'] = talib.EMA(df['close'], period)
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
        
        # RSI
        features['rsi_14'] = talib.RSI(df['close'], 14)
        features['rsi_28'] = talib.RSI(df['close'], 28)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # ATR
        features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        features['atr_ratio'] = features['atr_14'] / df['close']
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        
        return features
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        features = pd.DataFrame(index=df.index)
        
        # Spread features
        if 'bid' in df.columns and 'ask' in df.columns:
            features['spread'] = df['ask'] - df['bid']
            features['spread_pct'] = features['spread'] / ((df['ask'] + df['bid']) / 2)
            features['spread_ratio'] = features['spread'] / features['spread'].rolling(20).mean()
        
        # Kyle's lambda (price impact)
        returns = df['close'].pct_change()
        signed_volume = df['volume'] * np.sign(returns)
        
        features['kyle_lambda'] = returns.rolling(20).apply(
            lambda x: np.cov(x, signed_volume[-len(x):])[0, 1] / np.var(signed_volume[-len(x):])
            if len(x) > 1 else 0
        )
        
        # Amihud illiquidity
        features['amihud'] = abs(returns) / df['volume']
        features['amihud_ratio'] = features['amihud'] / features['amihud'].rolling(20).mean()
        
        # Roll's spread estimator
        features['roll_spread'] = 2 * np.sqrt(-returns.rolling(20).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        ))
        
        return features
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        price_features = self.create_price_features(df)
        volume_features = self.create_volume_features(df)
        technical_features = self.create_technical_features(df)
        microstructure_features = self.create_microstructure_features(df)
        
        # Combine all features
        all_features = pd.concat([
            price_features,
            volume_features,
            technical_features,
            microstructure_features
        ], axis=1)
        
        # Store feature names
        self.feature_names = all_features.columns.tolist()
        
        return all_features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series,
                       method: str = 'mutual_info', top_k: int = 50) -> List[str]:
        """Feature selection"""
        from sklearn.feature_selection import mutual_info_regression, f_regression
        from sklearn.ensemble import RandomForestRegressor
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        if method == 'mutual_info':
            scores = mutual_info_regression(X, y)
        elif method == 'f_regression':
            scores, _ = f_regression(X, y)
        elif method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            scores = rf.feature_importances_
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top k features
        feature_scores = pd.Series(scores, index=features.columns)
        top_features = feature_scores.nlargest(top_k).index.tolist()
        
        return top_features
```

## Deliverables
- `ml/forecasting.py`: LSTM and Transformer models for directional prediction
- Embargoed cross-validation for time series
- Probability calibration methods (isotonic, sigmoid, beta)
- Comprehensive feature engineering pipeline