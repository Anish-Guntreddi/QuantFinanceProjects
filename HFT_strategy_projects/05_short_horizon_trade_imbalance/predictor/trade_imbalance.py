"""Short Horizon Trade Imbalance Predictor"""

import numpy as np
import pandas as pd
from typing import Dict, List
import lightgbm as lgb

class TradeImbalancePredictor:
    """Predict short-term trade imbalance"""
    
    def __init__(self, horizon: int = 100):  # milliseconds
        self.horizon = horizon
        self.model = None
        self.feature_names = []
        
    def create_features(self, trades: pd.DataFrame) -> np.ndarray:
        """Create features from recent trades"""
        
        features = []
        
        # Trade imbalance over different windows
        for window in [10, 50, 100, 500]:
            recent = trades.tail(window)
            buy_volume = recent[recent['side'] == 'buy']['size'].sum()
            sell_volume = recent[recent['side'] == 'sell']['size'].sum()
            
            if buy_volume + sell_volume > 0:
                imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            else:
                imbalance = 0
                
            features.append(imbalance)
            
        # Trade intensity
        if len(trades) > 1:
            time_diff = trades['timestamp'].diff().mean()
            intensity = 1 / time_diff if time_diff > 0 else 0
        else:
            intensity = 0
            
        features.append(intensity)
        
        # Price momentum
        if len(trades) > 10:
            price_change = (trades['price'].iloc[-1] - trades['price'].iloc[-10]) / trades['price'].iloc[-10]
        else:
            price_change = 0
            
        features.append(price_change)
        
        return np.array(features)
    
    def train(self, historical_data: pd.DataFrame):
        """Train the model on historical data"""
        
        # Create training features and labels
        X = []
        y = []
        
        for i in range(100, len(historical_data) - self.horizon):
            features = self.create_features(historical_data.iloc[:i])
            
            # Label: future trade imbalance
            future_trades = historical_data.iloc[i:i+self.horizon]
            future_buy = future_trades[future_trades['side'] == 'buy']['size'].sum()
            future_sell = future_trades[future_trades['side'] == 'sell']['size'].sum()
            
            if future_buy + future_sell > 0:
                label = (future_buy - future_sell) / (future_buy + future_sell)
            else:
                label = 0
                
            X.append(features)
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        # Train LightGBM
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=100)
        
    def predict(self, trades: pd.DataFrame) -> float:
        """Predict future trade imbalance"""
        
        if self.model is None:
            return 0
            
        features = self.create_features(trades).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        
        return np.clip(prediction, -1, 1)
