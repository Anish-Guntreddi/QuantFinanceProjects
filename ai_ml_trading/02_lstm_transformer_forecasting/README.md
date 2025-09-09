# LSTM/Transformer Forecasting with Proper Evaluation

## Overview
Advanced time series forecasting with sliding/expanding windows, embargoed cross-validation, calibration techniques, and probability-aware position sizing.

## Project Structure
```
02_lstm_transformer_forecasting/
├── models/
│   ├── lstm_model.py
│   ├── transformer_model.py
│   ├── attention_mechanisms.py
│   └── model_factory.py
├── cv/
│   ├── time_series_cv.py
│   ├── embargo.py
│   └── purged_cv.py
├── calibration/
│   ├── isotonic.py
│   ├── platt.py
│   └── temperature_scaling.py
├── features/
│   ├── feature_engineering.py
│   ├── importance.py
│   └── interpretability.py
├── sizing/
│   ├── kelly_sizing.py
│   └── probability_sizing.py
├── evaluation/
│   ├── metrics.py
│   └── deflated_sharpe.py
└── tests/
    ├── test_models.py
    └── test_cv.py
```

## Implementation

### models/lstm_model.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LSTMConfig:
    input_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    bidirectional: bool = False
    attention: bool = True
    output_dim: int = 1
    sequence_length: int = 60

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate attention weights
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)
        
        # Apply attention
        attended = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)
        
        return attended, attention_weights

class AdvancedLSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super(AdvancedLSTM, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Attention layer
        if config.attention:
            self.attention = AttentionLayer(lstm_output_dim)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(lstm_output_dim)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.fc2 = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4)
        self.fc3 = nn.Linear(lstm_output_dim // 4, config.output_dim)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout / 2)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if configured
        if self.config.attention:
            output, attention_weights = self.attention(lstm_out)
        else:
            output = lstm_out[:, -1, :]
            attention_weights = None
        
        # Batch normalization
        output = self.batch_norm(output)
        
        # Fully connected layers with residual connections
        residual = output
        output = F.relu(self.fc1(output))
        output = self.dropout1(output)
        
        output = F.relu(self.fc2(output))
        output = self.dropout2(output)
        
        # Final output
        predictions = self.fc3(output)
        
        result = {'predictions': predictions}
        
        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights
            
        return result
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get learned representations for interpretability"""
        lstm_out, _ = self.lstm(x)
        
        if self.config.attention:
            embeddings, _ = self.attention(lstm_out)
        else:
            embeddings = lstm_out[:, -1, :]
        
        return embeddings
```

### models/transformer_model.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_len: int = 5000
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Temporal attention aggregation
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, output_dim)
        )
        
        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask for autoregressive prediction"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Input projection
        src = self.input_projection(src)
        
        # Add positional encoding
        src = src.transpose(0, 1)  # (batch, seq, features) -> (seq, batch, features)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Back to (batch, seq, features)
        
        # Create attention mask if not provided
        if src_mask is None:
            src_mask = self.create_attention_mask(src.size(1)).to(src.device)
        
        # Transformer encoding
        memory = self.transformer_encoder(src, mask=src_mask)
        
        # Temporal attention aggregation
        batch_size = memory.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        attended_output, attention_weights = self.temporal_attention(
            query, memory, memory
        )
        
        # Squeeze the sequence dimension
        attended_output = attended_output.squeeze(1)
        
        # Output projection
        output = self.output_layers(attended_output)
        
        return {
            'predictions': output,
            'attention_weights': attention_weights.squeeze(1),
            'embeddings': memory
        }
    
    def get_attention_maps(self, src: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps for interpretability"""
        self.eval()
        with torch.no_grad():
            # Get attention weights from each layer
            attention_maps = []
            
            src = self.input_projection(src)
            src = src.transpose(0, 1)
            src = self.pos_encoder(src)
            src = src.transpose(0, 1)
            
            # Hook to capture attention weights
            def hook_fn(module, input, output):
                attention_maps.append(output[1])
            
            # Register hooks
            hooks = []
            for layer in self.transformer_encoder.layers:
                hook = layer.self_attn.register_forward_hook(hook_fn)
                hooks.append(hook)
            
            # Forward pass
            _ = self.transformer_encoder(src)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return {'layer_attention': attention_maps}
```

### cv/time_series_cv.py
```python
import numpy as np
import pandas as pd
from typing import Iterator, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CVConfig:
    n_splits: int = 5
    test_size: int = 252  # Trading days
    gap: int = 10  # Embargo period
    expanding: bool = False
    purge_pct: float = 0.01

class TimeSeriesCV:
    def __init__(self, config: CVConfig = CVConfig()):
        self.config = config
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for time series cross-validation"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.config.expanding:
            yield from self._expanding_window_split(indices)
        else:
            yield from self._sliding_window_split(indices)
    
    def _sliding_window_split(self, indices: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Sliding window cross-validation"""
        n_samples = len(indices)
        test_size = self.config.test_size
        gap = self.config.gap
        
        # Calculate split points
        split_size = (n_samples - test_size - gap) // self.config.n_splits
        
        for i in range(self.config.n_splits):
            test_end = n_samples - i * split_size
            test_start = test_end - test_size
            train_end = test_start - gap
            train_start = max(0, train_end - split_size * 2)  # Use 2x test size for training
            
            if train_start >= train_end or test_start >= test_end:
                continue
            
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def _expanding_window_split(self, indices: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Expanding window cross-validation"""
        n_samples = len(indices)
        test_size = self.config.test_size
        gap = self.config.gap
        
        # Calculate split points
        split_size = (n_samples - test_size - gap) // self.config.n_splits
        
        for i in range(self.config.n_splits):
            test_end = n_samples - i * split_size
            test_start = test_end - test_size
            train_end = test_start - gap
            train_start = 0  # Always start from beginning (expanding)
            
            if train_start >= train_end or test_start >= test_end:
                continue
            
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices

class PurgedKFold:
    """Purged K-Fold cross-validation for time series with embargo"""
    
    def __init__(self, n_splits: int = 5, embargo_td: pd.Timedelta = pd.Timedelta(days=10)):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
             events: pd.DataFrame = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices with purging and embargo
        events: DataFrame with columns ['start_time', 'end_time'] for each observation
        """
        if events is None:
            # Create default events (point-in-time)
            events = pd.DataFrame({
                'start_time': X.index,
                'end_time': X.index
            })
        
        indices = np.arange(len(X))
        embargoed_indices = []
        
        # Generate base splits
        test_ranges = [(i * len(X) // self.n_splits, (i + 1) * len(X) // self.n_splits) 
                      for i in range(self.n_splits)]
        
        for test_start, test_end in test_ranges:
            test_indices = indices[test_start:test_end]
            
            # Get test event times
            test_events = events.iloc[test_indices]
            min_test_start = test_events['start_time'].min()
            max_test_end = test_events['end_time'].max()
            
            # Apply embargo
            embargo_start = min_test_start - self.embargo_td
            embargo_end = max_test_end + self.embargo_td
            
            # Find training indices (exclude test and embargo periods)
            train_mask = np.ones(len(X), dtype=bool)
            
            for idx in range(len(X)):
                event_start = events.iloc[idx]['start_time']
                event_end = events.iloc[idx]['end_time']
                
                # Check if event overlaps with test or embargo period
                if not (event_end < embargo_start or event_start > embargo_end):
                    train_mask[idx] = False
            
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices

class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation (de Prado, 2018)"""
    
    def __init__(self, n_splits: int = 5, n_test_groups: int = 2, 
                embargo_td: pd.Timedelta = pd.Timedelta(days=10)):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_td = embargo_td
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial purged splits"""
        from itertools import combinations
        
        n_samples = len(X)
        group_size = n_samples // self.n_splits
        groups = [list(range(i * group_size, min((i + 1) * group_size, n_samples))) 
                 for i in range(self.n_splits)]
        
        # Generate all combinations of test groups
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        
        for test_groups in test_combinations:
            test_indices = []
            for g in test_groups:
                test_indices.extend(groups[g])
            
            # Apply embargo
            test_start = min(test_indices)
            test_end = max(test_indices)
            
            embargo_size = int(len(X) * 0.01)  # 1% embargo
            
            train_indices = []
            for i in range(n_samples):
                if i < test_start - embargo_size or i > test_end + embargo_size:
                    train_indices.append(i)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield np.array(train_indices), np.array(test_indices)
```

### calibration/isotonic.py
```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Optional, Tuple
import matplotlib.pyplot as plt

class IsotonicCalibrator:
    def __init__(self, min_bin_size: int = 100):
        self.min_bin_size = min_bin_size
        self.calibrator = None
        self.calibration_map = None
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """Fit isotonic regression calibrator"""
        # Fit isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probabilities, labels)
        
        # Store calibration mapping
        self.calibration_map = {
            'original': probabilities,
            'calibrated': self.calibrator.transform(probabilities),
            'labels': labels
        }
        
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted")
        
        return self.calibrator.transform(probabilities)
    
    def fit_transform(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(probabilities, labels)
        return self.transform(probabilities)
    
    def plot_calibration(self, n_bins: int = 10) -> Tuple[plt.Figure, plt.Axes]:
        """Plot calibration curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original calibration
        fraction_pos, mean_pred = self._calibration_curve(
            self.calibration_map['labels'],
            self.calibration_map['original'],
            n_bins
        )
        
        ax1.plot(mean_pred, fraction_pos, 's-', label='Original')
        ax1.plot([0, 1], [0, 1], 'k:', label='Perfect')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Original Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calibrated
        fraction_pos_cal, mean_pred_cal = self._calibration_curve(
            self.calibration_map['labels'],
            self.calibration_map['calibrated'],
            n_bins
        )
        
        ax2.plot(mean_pred_cal, fraction_pos_cal, 's-', label='Calibrated', color='green')
        ax2.plot([0, 1], [0, 1], 'k:', label='Perfect')
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Fraction of Positives')
        ax2.set_title('After Isotonic Calibration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def _calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                          n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate calibration curve"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fraction_pos = []
        mean_pred = []
        
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            
            if mask.sum() >= self.min_bin_size:
                fraction_pos.append(y_true[mask].mean())
                mean_pred.append(y_prob[mask].mean())
        
        return np.array(fraction_pos), np.array(mean_pred)
    
    def expected_calibration_error(self, probabilities: np.ndarray, 
                                  labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            
            if mask.sum() > 0:
                bin_accuracy = labels[mask].mean()
                bin_confidence = probabilities[mask].mean()
                bin_size = mask.sum()
                
                ece += (bin_size / len(probabilities)) * abs(bin_accuracy - bin_confidence)
        
        return ece

class PlattScaling:
    """Platt scaling (sigmoid calibration)"""
    
    def __init__(self):
        self.a = None
        self.b = None
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """Fit Platt scaling parameters"""
        from scipy.optimize import minimize
        
        def sigmoid(x, a, b):
            return 1 / (1 + np.exp(a * x + b))
        
        def neg_log_likelihood(params):
            a, b = params
            probs = sigmoid(probabilities, a, b)
            
            # Avoid numerical issues
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            
            # Binary cross-entropy
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return loss
        
        # Optimize parameters
        result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method='L-BFGS-B')
        self.a, self.b = result.x
        
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling"""
        if self.a is None or self.b is None:
            raise ValueError("Model not fitted")
        
        return 1 / (1 + np.exp(self.a * probabilities + self.b))

class TemperatureScaling:
    """Temperature scaling for neural network calibration"""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> 'TemperatureScaling':
        """Optimize temperature parameter"""
        from scipy.optimize import minimize_scalar
        
        def neg_log_likelihood(T):
            # Apply temperature scaling
            scaled_logits = logits / T
            
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # NLL
            n_samples = len(labels)
            nll = -np.sum(np.log(probs[np.arange(n_samples), labels] + 1e-15)) / n_samples
            
            return nll
        
        # Optimize temperature
        result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        scaled_logits = logits / self.temperature
        
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs
```

### sizing/kelly_sizing.py
```python
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.optimize import minimize

class KellySizing:
    def __init__(self, max_leverage: float = 1.0, kelly_fraction: float = 0.25):
        """
        max_leverage: Maximum allowed leverage
        kelly_fraction: Fraction of Kelly criterion to use (for safety)
        """
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
    
    def calculate_kelly_fraction(self, win_prob: float, win_return: float, 
                                loss_return: float) -> float:
        """Calculate Kelly fraction for binary outcome"""
        if loss_return >= 0:
            return 0  # No risk, invalid setup
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        q = 1 - win_prob
        b = win_return / abs(loss_return)
        
        kelly = (win_prob * b - q) / b
        
        # Apply safety fraction
        kelly *= self.kelly_fraction
        
        # Cap at max leverage
        kelly = min(kelly, self.max_leverage)
        
        # Ensure non-negative
        return max(0, kelly)
    
    def multi_asset_kelly(self, expected_returns: np.ndarray, 
                         covariance: np.ndarray) -> np.ndarray:
        """Calculate Kelly weights for multiple assets"""
        n_assets = len(expected_returns)
        
        # Inverse covariance
        try:
            inv_cov = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            inv_cov = np.linalg.pinv(covariance)
        
        # Kelly weights
        kelly_weights = inv_cov @ expected_returns
        
        # Apply fraction
        kelly_weights *= self.kelly_fraction
        
        # Apply leverage constraint
        total_exposure = np.sum(np.abs(kelly_weights))
        if total_exposure > self.max_leverage:
            kelly_weights *= self.max_leverage / total_exposure
        
        return kelly_weights
    
    def probability_weighted_kelly(self, probabilities: np.ndarray,
                                  outcomes: np.ndarray) -> float:
        """Kelly sizing with multiple probability-weighted outcomes"""
        # Expected return
        expected_return = np.sum(probabilities * outcomes)
        
        # Variance
        variance = np.sum(probabilities * (outcomes - expected_return) ** 2)
        
        if variance == 0:
            return 0
        
        # Kelly fraction
        kelly = expected_return / variance
        
        # Apply safety fraction and constraints
        kelly *= self.kelly_fraction
        kelly = min(kelly, self.max_leverage)
        
        return max(0, kelly)
    
    def dynamic_kelly_with_confidence(self, base_kelly: float, 
                                     confidence: float) -> float:
        """Adjust Kelly fraction based on prediction confidence"""
        # Scale Kelly by confidence
        # Use sigmoid-like scaling
        confidence_multiplier = 2 / (1 + np.exp(-5 * (confidence - 0.5)))
        
        adjusted_kelly = base_kelly * confidence_multiplier
        
        return min(adjusted_kelly, self.max_leverage)

class ProbabilityAwareSizing:
    def __init__(self, min_probability: float = 0.55, max_position: float = 1.0):
        self.min_probability = min_probability
        self.max_position = max_position
    
    def calculate_position_size(self, probability: float, 
                               volatility: float = None) -> float:
        """Calculate position size based on probability"""
        if probability < self.min_probability:
            return 0
        
        # Linear scaling above minimum threshold
        prob_score = (probability - 0.5) * 2  # Scale to [0, 1]
        
        # Optional volatility adjustment
        if volatility is not None:
            # Reduce size in high volatility
            vol_adjustment = 1 / (1 + volatility * 10)
            prob_score *= vol_adjustment
        
        # Apply maximum position constraint
        position_size = prob_score * self.max_position
        
        return np.clip(position_size, 0, self.max_position)
    
    def ensemble_sizing(self, probabilities: Dict[str, float],
                       model_weights: Dict[str, float] = None) -> float:
        """Combine predictions from multiple models"""
        if model_weights is None:
            model_weights = {k: 1/len(probabilities) for k in probabilities}
        
        # Weighted average probability
        weighted_prob = sum(prob * model_weights[model] 
                          for model, prob in probabilities.items())
        
        # Calculate position size
        return self.calculate_position_size(weighted_prob)
    
    def risk_adjusted_sizing(self, probability: float, sharpe_ratio: float,
                           max_drawdown: float) -> float:
        """Adjust position size based on risk metrics"""
        base_size = self.calculate_position_size(probability)
        
        # Sharpe adjustment (higher Sharpe = larger position)
        sharpe_multiplier = min(1.5, max(0.5, sharpe_ratio / 2))
        
        # Drawdown adjustment (larger drawdown = smaller position)
        dd_multiplier = max(0.3, 1 - abs(max_drawdown))
        
        adjusted_size = base_size * sharpe_multiplier * dd_multiplier
        
        return np.clip(adjusted_size, 0, self.max_position)
```

### evaluation/metrics.py
```python
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, precision_recall_curve

class ForecastMetrics:
    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier score"""
        return brier_score_loss(y_true, y_prob)
    
    @staticmethod
    def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate log loss"""
        return log_loss(y_true, y_prob)
    
    @staticmethod
    def profit_curve(y_true: np.ndarray, y_prob: np.ndarray,
                    costs: Tuple[float, float] = (1.0, 1.0)) -> Dict:
        """
        Calculate profit curve
        costs: (cost_false_positive, cost_false_negative)
        """
        thresholds = np.linspace(0, 1, 101)
        profits = []
        
        cost_fp, cost_fn = costs
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            # Calculate profit/loss
            profit = tp - fp * cost_fp - fn * cost_fn
            profits.append(profit)
        
        profits = np.array(profits)
        optimal_threshold = thresholds[np.argmax(profits)]
        max_profit = np.max(profits)
        
        return {
            'thresholds': thresholds,
            'profits': profits,
            'optimal_threshold': optimal_threshold,
            'max_profit': max_profit
        }
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy for returns"""
        return np.mean(np.sign(y_true) == np.sign(y_pred))
    
    @staticmethod
    def precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray,
                          target_recall: float = 0.8) -> float:
        """Calculate precision at specific recall level"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Find precision at target recall
        idx = np.argmin(np.abs(recall - target_recall))
        return precision[idx]

class DeflatedSharpe:
    """Deflated Sharpe Ratio (Lopez de Prado, 2018)"""
    
    @staticmethod
    def calculate(sharpe_ratio: float, n_trials: int, n_periods: int,
                 skewness: float = 0, kurtosis: float = 3) -> Dict:
        """
        Calculate probability that Sharpe ratio is due to chance
        
        sharpe_ratio: Observed Sharpe ratio
        n_trials: Number of strategies tested
        n_periods: Number of time periods
        skewness: Skewness of returns
        kurtosis: Excess kurtosis of returns
        """
        from scipy.stats import norm
        
        # Adjust for higher moments
        sr_adjusted = sharpe_ratio * np.sqrt(n_periods)
        
        # Standard error considering skewness and kurtosis
        se = np.sqrt((1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
                     (kurtosis - 3) / 4 * sharpe_ratio**2) / n_periods)
        
        # Expected maximum Sharpe ratio under null hypothesis
        euler_mascheroni = 0.5772156649
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) - \
                         (euler_mascheroni + np.log(2 * np.log(n_trials))) / \
                         (2 * np.sqrt(2 * np.log(n_trials)))
        
        # Deflated Sharpe ratio
        deflated_sr = (sr_adjusted - expected_max_sr) / se
        
        # Probability of observing this Sharpe by chance
        p_value = 1 - norm.cdf(deflated_sr)
        
        return {
            'deflated_sharpe': deflated_sr,
            'p_value': p_value,
            'expected_max_sharpe': expected_max_sr / np.sqrt(n_periods),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def minimum_track_record_length(sharpe_ratio: float, 
                                   confidence: float = 0.95) -> int:
        """Calculate minimum track record length for significance"""
        from scipy.stats import norm
        
        z_score = norm.ppf(confidence)
        
        # Minimum length formula
        min_length = 1 + (1 + 0.5 * sharpe_ratio**2) * (z_score / sharpe_ratio)**2
        
        return int(np.ceil(min_length))

class BacktestMetrics:
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, benchmark: pd.Series = None) -> Dict:
        """Calculate comprehensive backtest metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = returns.mean() * 252
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility']
        
        # Risk metrics
        metrics['max_drawdown'] = BacktestMetrics._max_drawdown(returns)
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # Higher moments
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = metrics['annual_return'] / metrics['downside_deviation']
        
        # Win/loss statistics
        metrics['win_rate'] = (returns > 0).mean()
        metrics['avg_win'] = returns[returns > 0].mean()
        metrics['avg_loss'] = returns[returns < 0].mean()
        metrics['profit_factor'] = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
        
        # Benchmark relative metrics
        if benchmark is not None:
            excess_returns = returns - benchmark
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            # Beta and alpha
            covariance = np.cov(returns, benchmark)
            metrics['beta'] = covariance[0, 1] / covariance[1, 1]
            metrics['alpha'] = metrics['annual_return'] - metrics['beta'] * (benchmark.mean() * 252)
        
        return metrics
    
    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

### features/importance.py
```python
import numpy as np
import pandas as pd
import torch
import shap
from typing import Dict, List, Optional
from captum.attr import IntegratedGradients, DeepLift, GradientShap

class FeatureImportance:
    def __init__(self, model, model_type: str = 'torch'):
        """
        model: Trained model
        model_type: 'torch', 'sklearn', 'xgboost'
        """
        self.model = model
        self.model_type = model_type
    
    def calculate_shap_values(self, X: np.ndarray, background_samples: int = 100) -> np.ndarray:
        """Calculate SHAP values for tabular data"""
        if self.model_type == 'torch':
            # For neural networks, use DeepExplainer
            explainer = shap.DeepExplainer(self.model, X[:background_samples])
        elif self.model_type == 'xgboost':
            explainer = shap.TreeExplainer(self.model)
        else:
            # For other models, use KernelExplainer
            explainer = shap.KernelExplainer(self.model.predict, X[:background_samples])
        
        shap_values = explainer.shap_values(X)
        
        return shap_values
    
    def integrated_gradients(self, model: torch.nn.Module, 
                           inputs: torch.Tensor,
                           baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate Integrated Gradients for sequence models"""
        model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        ig = IntegratedGradients(model)
        
        # Calculate attributions
        attributions = ig.attribute(inputs, baseline, n_steps=50)
        
        return attributions
    
    def attention_based_importance(self, model: torch.nn.Module,
                                 inputs: torch.Tensor) -> np.ndarray:
        """Extract feature importance from attention weights"""
        model.eval()
        
        with torch.no_grad():
            outputs = model(inputs, return_attention=True)
            attention_weights = outputs['attention_weights']
        
        # Average attention across batch
        avg_attention = attention_weights.mean(dim=0).cpu().numpy()
        
        return avg_attention
    
    def permutation_importance(self, X: pd.DataFrame, y: pd.Series,
                              scoring_func, n_repeats: int = 10) -> pd.DataFrame:
        """Calculate permutation importance"""
        baseline_score = scoring_func(X, y)
        importance_scores = {}
        
        for column in X.columns:
            scores = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[column] = np.random.permutation(X_permuted[column])
                
                permuted_score = scoring_func(X_permuted, y)
                scores.append(baseline_score - permuted_score)
            
            importance_scores[column] = {
                'importance': np.mean(scores),
                'std': np.std(scores)
            }
        
        return pd.DataFrame(importance_scores).T.sort_values('importance', ascending=False)
    
    def gradient_based_saliency(self, model: torch.nn.Module,
                              inputs: torch.Tensor,
                              target_class: int = 1) -> torch.Tensor:
        """Calculate gradient-based saliency maps"""
        model.eval()
        inputs.requires_grad = True
        
        # Forward pass
        outputs = model(inputs)
        
        # Select target class
        if len(outputs.shape) > 1:
            target_output = outputs[:, target_class]
        else:
            target_output = outputs
        
        # Backward pass
        model.zero_grad()
        target_output.backward(torch.ones_like(target_output))
        
        # Get gradients
        saliency = inputs.grad.abs()
        
        return saliency
    
    def temporal_importance_analysis(self, model: torch.nn.Module,
                                   sequence: torch.Tensor) -> np.ndarray:
        """Analyze importance of different time steps in sequence"""
        model.eval()
        
        seq_length = sequence.shape[1]
        importance_scores = []
        
        with torch.no_grad():
            # Get baseline prediction
            baseline_output = model(sequence)['predictions']
            
            # Mask each time step
            for t in range(seq_length):
                masked_sequence = sequence.clone()
                masked_sequence[:, t, :] = 0  # Zero out time step
                
                masked_output = model(masked_sequence)['predictions']
                
                # Calculate importance as difference from baseline
                importance = torch.abs(baseline_output - masked_output).mean().item()
                importance_scores.append(importance)
        
        return np.array(importance_scores)
```

## Deliverables Summary

### Models
- Advanced LSTM with attention mechanism
- Time Series Transformer with temporal attention
- Comprehensive model configurations

### Cross-Validation
- Sliding/expanding window CV
- Purged K-Fold with embargo
- Combinatorial purged CV

### Calibration
- Isotonic regression calibration
- Platt scaling
- Temperature scaling for neural networks

### Position Sizing
- Kelly criterion with safety fraction
- Probability-aware sizing
- Risk-adjusted position sizing

### Evaluation Metrics
- Brier score and log loss
- Profit curves with optimal thresholds
- Deflated Sharpe ratio (Lopez de Prado)
- Comprehensive backtest metrics

### Feature Importance
- SHAP values for tabular data
- Integrated Gradients for sequences
- Attention-based importance
- Temporal importance analysis