"""
AI Engine - Predictive models and decision making
Enhanced with machine learning capabilities
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import pickle
import os

class AIEngine:
    def __init__(self, data_layer=None, risk_engine=None, config=None):
        """
        Initialize AI Engine
        
        Args:
            data_layer: DataLayer instance for market data
            risk_engine: RiskEngine instance for risk assessment
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.data_layer = data_layer
        self.risk_engine = risk_engine
        self.config = config or {}
        
        # Model storage
        self.models = {}
        self.model_directory = self.config.get('model_directory', 'models/')
        
        # Feature configuration
        self.features = self.config.get('features', [
            'sma_20', 'sma_50', 'rsi', 'macd', 
            'bb_upper', 'bb_lower', 'volume'
        ])
        
        # Initialize models
        self.initialize_models()
        
        # Performance tracking
        self.predictions_history = []
        self.accuracy_history = []
        
        self.logger.info("AI Engine initialized")
    
    def initialize_models(self):
        """Initialize or load ML models"""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(self.model_directory, exist_ok=True)
            
            # Try to load existing models
            self.load_models()
            
            # If no models loaded, create default ones
            if not self.models:
                self.create_default_models()
                self.logger.info("Created default AI models")
            else:
                self.logger.info(f"Loaded {len(self.models)} AI models")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            self.create_default_models()
    
    def create_default_models(self):
        """Create default ML models for initial use"""
        try:
            # Price direction classifier (buy/sell/hold)
            from sklearn.ensemble import RandomForestClassifier
            self.models['direction_classifier'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Price regression model
            from sklearn.ensemble import GradientBoostingRegressor
            self.models['price_predictor'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            # Save models
            self.save_models()
            
        except ImportError:
            self.logger.warning("scikit-learn not available, using simple heuristic models")
            self.models['direction_classifier'] = None
            self.models['price_predictor'] = None
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_files = {
                'direction_classifier': 'direction_classifier.pkl',
                'price_predictor': 'price_predictor.pkl',
                'sentiment_model': 'sentiment_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.model_directory, filename)
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    self.logger.debug(f"Loaded {model_name} from {model_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = os.path.join(self.model_directory, f"{model_name}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    self.logger.debug(f"Saved {model_name} to {model_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def extract_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from market data for ML models
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with extracted features
        """
        try:
            features_df = pd.DataFrame(index=market_data.index)
            
            # Price-based features
            features_df['returns'] = market_data['close'].pct_change()
            features_df['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
            features_df['volatility'] = market_data['close'].rolling(window=20).std()
            
            # Technical indicators (already calculated in data layer)
            for feature in self.features:
                if feature in market_data.columns:
                    features_df[feature] = market_data[feature]
            
            # Additional derived features
            if 'sma_20' in market_data.columns and 'sma_50' in market_data.columns:
                features_df['sma_cross'] = market_data['sma_20'] - market_data['sma_50']
            
            if 'rsi' in market_data.columns:
                features_df['rsi_overbought'] = (market_data['rsi'] > 70).astype(int)
                features_df['rsi_oversold'] = (market_data['rsi'] < 30).astype(int)
            
            if 'bb_upper' in market_data.columns and 'bb_lower' in market_data.columns:
                bb_width = (market_data['bb_upper'] - market_data['bb_lower']) / market_data['bb_middle']
                features_df['bb_width'] = bb_width
                features_df['bb_position'] = (market_data['close'] - market_data['bb_lower']) / (market_data['bb_upper'] - market_data['bb_lower'])
            
            # Volume features
            features_df['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(window=20).mean()
            features_df['volume_spike'] = (market_data['volume'] > market_data['volume'].rolling(window=20).mean() * 2).astype(int)
            
            # Time-based features
            features_df['hour'] = market_data.index.hour
            features_df['day_of_week'] = market_data.index.dayofweek
            features_df['month'] = market_data.index.month
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
                features_df[f'volume_lag_{lag}'] = market_data['volume'].shift(lag)
            
            # Drop NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return pd.DataFrame()
    
    def prepare_training_data(self, market_data: pd.DataFrame) -> tuple:
        """
        Prepare data for model training
        
        Args:
            market_data: Historical market data
            
        Returns:
            Tuple of (X_features, y_target)
        """
        try:
            # Extract features
            features_df = self.extract_features(market_data)
            
            if features_df.empty:
                return np.array([]), np.array([])
            
            # Create target variable (1 if price goes up next period, 0 otherwise)
            future_returns = market_data['close'].pct_change().shift(-1)
            y_direction = (future_returns > 0).astype(int)
            
            # Align features with target
            common_idx = features_df.index.intersection(y_direction.index)
            X = features_df.loc[common_idx].values
            y = y_direction.loc[common_idx].values
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation error: {e}")
            return np.array([]), np.array([])
    
    def train_models(self, market_data: pd.DataFrame):
        """
        Train ML models on historical data
        
        Args:
            market_data: Historical market data for training
        """
        try:
            X, y = self.prepare_training_data(market_data)
            
            if len(X) == 0 or len(y) == 0:
                self.logger.warning("Insufficient data for training")
                return
            
            # Train direction classifier
            if self.models.get('direction_classifier') is not None:
                self.models['direction_classifier'].fit(X, y)
                
                # Calculate training accuracy
                train_pred = self.models['direction_classifier'].predict(X)
                train_accuracy = np.mean(train_pred == y)
                self.accuracy_history.append(train_accuracy)
                
                self.logger.info(f"Direction classifier trained. Accuracy: {train_accuracy:.2%}")
                self.logger.info(f"Training samples: {len(X)}")
            
            # Train price predictor (regression)
            if self.models.get('price_predictor') is not None and len(X) > 0:
                y_price = market_data['close'].values[-len(X):]
                if len(y_price) == len(X):
                    self.models['price_predictor'].fit(X, y_price)
                    self.logger.info(f"Price predictor trained on {len(X)} samples")
            
            # Save trained models
            self.save_models()
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
    
    def predict(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Make trading prediction based on market data
        
        Args:
            market_data: Recent market data
            symbol: Trading symbol
            
        Returns:
            Dictionary with prediction details
        """
        try:
            # Extract features from latest data
            features_df = self.extract_features(market_data)
            
            if features_df.empty:
                self.logger.warning("No features extracted, using heuristic prediction")
                return self._heuristic_prediction(market_data, symbol)
            
            # Get latest features
            latest_features = features_df.iloc[-1:].values
            
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'features_extracted': True,
                'confidence': 0.5,  # Default
                'action': 'hold',   # Default
                'reason': '',
                'metadata': {}
            }
            
            # Make direction prediction
            if self.models.get('direction_classifier') is not None:
                try:
                    direction_proba = self.models['direction_classifier'].predict_proba(latest_features)[0]
                    direction_pred = self.models['direction_classifier'].predict(latest_features)[0]
                    
                    # Buy if predicts up with high confidence
                    buy_confidence = direction_proba[1] if len(direction_proba) > 1 else 0.5
                    sell_confidence = direction_proba[0] if len(direction_proba) > 0 else 0.5
                    
                    confidence_threshold = self.config.get('confidence_threshold', 0.65)
                    
                    if buy_confidence > confidence_threshold:
                        prediction['action'] = 'buy'
                        prediction['confidence'] = buy_confidence
                        prediction['reason'] = f"AI predicts price increase (confidence: {buy_confidence:.2%})"
                    elif sell_confidence > confidence_threshold:
                        prediction['action'] = 'sell'
                        prediction['confidence'] = sell_confidence
                        prediction['reason'] = f"AI predicts price decrease (confidence: {sell_confidence:.2%})"
                    else:
                        prediction['action'] = 'hold'
                        prediction['confidence'] = max(buy_confidence, sell_confidence)
                        prediction['reason'] = f"Low confidence: buy={buy_confidence:.2%}, sell={sell_confidence:.2%}"
                    
                    prediction['metadata']['direction_proba'] = direction_proba.tolist()
                    prediction['metadata']['direction_pred'] = int(direction_pred)
                    
                except Exception as e:
                    self.logger.warning(f"Classifier prediction failed: {e}, using heuristic")
                    return self._heuristic_prediction(market_data, symbol)
            
            # Make price prediction
            if self.models.get('price_predictor') is not None:
                try:
                    predicted_price = self.models['price_predictor'].predict(latest_features)[0]
                    current_price = market_data['close'].iloc[-1]
                    price_change_pct = (predicted_price - current_price) / current_price * 100
                    
                    prediction['predicted_price'] = float(predicted_price)
                    prediction['current_price'] = float(current_price)
                    prediction['expected_return'] = float(price_change_pct)
                    prediction['metadata']['price_prediction'] = float(predicted_price)
                    
                except Exception as e:
                    self.logger.debug(f"Price prediction failed: {e}")
            
            # Add technical analysis signals
            prediction = self._add_technical_signals(prediction, market_data)
            
            # Store prediction history
            self.predictions_history.append(prediction)
            
            # Keep history limited
            if len(self.predictions_history) > 1000:
                self.predictions_history = self.predictions_history[-500:]
            
            self.logger.debug(f"Prediction: {prediction['action']} {symbol} (confidence: {prediction['confidence']:.2%})")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self._heuristic_prediction(market_data, symbol)
    
    def _heuristic_prediction(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Fallback heuristic prediction when ML models fail"""
        current_price = market_data['close'].iloc[-1]
        prev_price = market_data['close'].iloc[-2] if len(market_data) > 1 else current_price
        
        # Simple momentum heuristic
        if current_price > prev_price:
            action = 'buy'
            confidence = 0.6
            reason = "Price momentum positive"
        else:
            action = 'sell'
            confidence = 0.6
            reason = "Price momentum negative"
        
        # Check RSI if available
        if 'rsi' in market_data.columns:
            rsi = market_data['rsi'].iloc[-1]
            if rsi < 30:
                action = 'buy'
                confidence = 0.7
                reason = f"RSI oversold: {rsi:.1f}"
            elif rsi > 70:
                action = 'sell'
                confidence = 0.7
                reason = f"RSI overbought: {rsi:.1f}"
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'current_price': float(current_price),
            'features_extracted': False,
            'metadata': {'heuristic': True}
        }
    
    def _add_technical_signals(self, prediction: Dict, market_data: pd.DataFrame) -> Dict:
        """Add technical analysis signals to prediction"""
        signals = []
        
        try:
            # RSI signals
            if 'rsi' in market_data.columns:
                rsi = market_data['rsi'].iloc[-1]
                if rsi < 30:
                    signals.append(('RSI_OVERSOLD', rsi, 'bullish'))
                elif rsi > 70:
                    signals.append(('RSI_OVERBOUGHT', rsi, 'bearish'))
            
            # MACD signals
            if 'macd' in market_data.columns and 'macd_signal' in market_data.columns:
                macd = market_data['macd'].iloc[-1]
                signal = market_data['macd_signal'].iloc[-1]
                if macd > signal and market_data['macd'].iloc[-2] <= market_data['macd_signal'].iloc[-2]:
                    signals.append(('MACD_CROSS_UP', macd - signal, 'bullish'))
                elif macd < signal and market_data['macd'].iloc[-2] >= market_data['macd_signal'].iloc[-2]:
                    signals.append(('MACD_CROSS_DOWN', signal - macd, 'bearish'))
            
            # Moving average signals
            if 'sma_20' in market_data.columns and 'sma_50' in market_data.columns:
                sma_20 = market_data['sma_20'].iloc[-1]
                sma_50 = market_data['sma_50'].iloc[-1]
                price = market_data['close'].iloc[-1]
                
                if price > sma_20 > sma_50:
                    signals.append(('TRIPLE_BULLISH', price - sma_20, 'bullish'))
                elif price < sma_20 < sma_50:
                    signals.append(('TRIPLE_BEARISH', sma_20 - price, 'bearish'))
            
            # Bollinger Bands
            if all(col in market_data.columns for col in ['bb_upper', 'bb_lower', 'close']):
                price = market_data['close'].iloc[-1]
                bb_upper = market_data['bb_upper'].iloc[-1]
                bb_lower = market_data['bb_lower'].iloc[-1]
                
                if price <= bb_lower:
                    signals.append(('BB_OVERSOLD', (bb_lower - price) / price, 'bullish'))
                elif price >= bb_upper:
                    signals.append(('BB_OVERBOUGHT', (price - bb_upper) / price, 'bearish'))
            
            # Add signals to prediction
            if signals:
                prediction['technical_signals'] = signals
                # Adjust confidence based on signals
                signal_strength = len([s for s in signals if s[2] == prediction['action']])
                if signal_strength > 0:
                    prediction['confidence'] = min(0.95, prediction['confidence'] + 0.1 * signal_strength)
                    prediction['reason'] += f" | Supported by {signal_strength} technical signal(s)"
        
        except Exception as e:
            self.logger.debug(f"Technical signal analysis error: {e}")
        
        return prediction
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get AI engine performance metrics"""
        if not self.accuracy_history:
            return {'accuracy': 0, 'predictions_made': 0}
        
        return {
            'accuracy_avg': np.mean(self.accuracy_history) if self.accuracy_history else 0,
            'accuracy_recent': np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else np.mean(self.accuracy_history),
            'predictions_made': len(self.predictions_history),
            'model_count': len([m for m in self.models.values() if m is not None]),
            'last_training_samples': len(self.accuracy_history)
        }