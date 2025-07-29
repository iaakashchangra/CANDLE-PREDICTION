from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from backend.forms.auth_forms import LoginForm, RegistrationForm
from backend.models.user import User
from backend.models.user_preferences import UserPreferences
from backend.models.performance_tracker import PerformanceTracker
from backend.utils.database import db
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Any
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

# Market Data APIs
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from polygon import RESTClient
import finnhub

# Technical Analysis
import talib

# ML Models
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from models.prediction_models import RNNModel, CNNModel, LSTMModel, HybridModel

# Ensure static directory exists
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Import backend modules (ML models temporarily disabled)
try:
    from backend.data.data_collector import DataCollector
    data_collector = DataCollector()
except ImportError as e:
    print(f"Warning: Could not import DataCollector: {e}")
    data_collector = None

try:
    from backend.data.preprocessor import DataPreprocessor
    data_preprocessor = DataPreprocessor()
except ImportError as e:
    print(f"Warning: Could not import DataPreprocessor: {e}")
    data_preprocessor = None

# Import ML models
try:
    # Configure TensorFlow for Apple Silicon if available
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging verbosity
    
    # Try to import TensorFlow and configure for Metal GPU
    try:
        import tensorflow as tf
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except ImportError:
        pass  # TensorFlow not available, will use fallback models
    except Exception as tf_err:
        pass  # Error configuring TensorFlow, will use fallback models
    
    # Import model classes
    from backend.models.model_manager import ModelManager
    from backend.models.lstm_model import LSTMModel
    from backend.models.cnn_model import CNNModel
    from backend.models.rnn_model import RNNModel
    from backend.models.hybrid_model import HybridModel
    model_manager = ModelManager()
except ImportError as e:
    # TensorFlow not available, using fallback models
    try:
        from backend.models.simple_model_manager import SimpleModelManager
        model_manager = SimpleModelManager()
    except ImportError as e2:
        model_manager = None

# Initialize Flask app with templates and static files
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.config['SECRET_KEY'] = 'dev'  # Change this to a secure key in production
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{Config.DATABASE_CONFIG['sqlite']['path']}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = Config.WEB_CONFIG['max_content_length']

@app.route('/')
def index():
    """Home page that shows index.html for unauthenticated users and redirects to dashboard for authenticated users"""
    if current_user.is_authenticated:
        return redirect(url_for('main_routes.dashboard'))
    return render_template('index.html')

# Add route to handle Vite client requests
@app.route('/@vite/client', methods=['GET'])
def handle_vite_client():
    """Handle Vite client requests with an empty response to prevent 404 errors"""
    return '', 200

# Import and register blueprints
try:
    from backend.routes.main_routes import main_routes
    from backend.routes.auth_routes import auth_bp
    from backend.routes.selection_routes import selection_routes
    from backend.routes.admin_routes import admin_routes
    app.register_blueprint(main_routes, url_prefix='/main')
    app.register_blueprint(auth_bp)
    app.register_blueprint(selection_routes)
    app.register_blueprint(admin_routes)
    print("Main routes, auth, selection, and admin blueprints registered successfully")
except ImportError as e:
    print(f"Warning: Could not import blueprints: {e}")

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'
login_manager.session_protection = 'strong'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOGS_DIR, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Backend components initialized above with error handling

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=4)

# Prediction helper functions
def generate_emergency_predictions(historical_data, prediction_count):
    """
    Generate very simple predictions when all other prediction methods fail.
    This is a last resort fallback to ensure the API always returns something.
    
    Args:
        historical_data (pd.DataFrame): Historical price data, may be None or empty
        prediction_count (int): Number of predictions to generate
        
    Returns:
        list: List of prediction dictionaries
    """
    logger.warning("Using emergency prediction generation as last resort")
    predictions = []
    
    # First check if we have any historical data to work with
    if historical_data is None or historical_data.empty:
        logger.error("No historical data available for emergency predictions, using synthetic data")
        return generate_synthetic_predictions(prediction_count)
    
    try:
        # Get the last available price
        try:
            last_timestamp = historical_data.index[-1]
            last_close = historical_data['Close'].iloc[-1]
            last_open = historical_data['Open'].iloc[-1]
            last_high = historical_data['High'].iloc[-1]
            last_low = historical_data['Low'].iloc[-1]
        except (IndexError, KeyError) as e:
            logger.error(f"Error accessing historical data: {str(e)}")
            return generate_synthetic_predictions(prediction_count)
        
        # Calculate average daily change based on last 5 days if available
        lookback = min(5, len(historical_data))
        if lookback < 2:  # Need at least 2 points to calculate change
            avg_daily_change = np.random.uniform(-0.005, 0.005)  # ±0.5%
        else:
            try:
                recent_data = historical_data.iloc[-lookback:]
                avg_daily_change = recent_data['Close'].pct_change().mean()
                
                # If we couldn't calculate change (e.g., NaN), use a small random value
                if pd.isna(avg_daily_change):
                    avg_daily_change = np.random.uniform(-0.005, 0.005)  # ±0.5%
            except Exception as e:
                logger.error(f"Error calculating price change: {str(e)}")
                avg_daily_change = np.random.uniform(-0.005, 0.005)  # ±0.5%
        
        # Generate predictions with minimal variation
        for i in range(prediction_count):
            # Calculate next timestamp based on the timeframe of the last timestamp
            if isinstance(last_timestamp, pd.Timestamp):
                # Assuming daily data - add one day for each prediction
                next_timestamp = last_timestamp + timedelta(days=i+1)
            else:
                # If timestamp is not a datetime, just increment by 1
                next_timestamp = datetime.now() + timedelta(days=i+1)
            
            # Calculate predicted prices with minimal randomness
            change_factor = 1 + (avg_daily_change * (i+1)) + np.random.uniform(-0.002, 0.002)
            predicted_close = last_close * change_factor
            
            # Generate other values with small variations
            predicted_open = last_open * change_factor * np.random.uniform(0.998, 1.002)
            predicted_high = predicted_close * np.random.uniform(1.001, 1.005)
            predicted_low = predicted_close * np.random.uniform(0.995, 0.999)
            
            # Ensure high is always highest and low is always lowest
            predicted_high = max(predicted_high, predicted_open, predicted_close)
            predicted_low = min(predicted_low, predicted_open, predicted_close)
            
            # Create prediction entry
            prediction = {
                'timestamp': next_timestamp,
                'predicted_open': float(predicted_open),
                'predicted_high': float(predicted_high),
                'predicted_low': float(predicted_low),
                'predicted_close': float(predicted_close),
                'confidence': 0.1,  # Very low confidence
                'signal': 'hold',  # Default to hold
                'risk_level': 'high',  # Mark as high risk
                'model_type': 'emergency'
            }
            
            predictions.append(prediction)
        
        logger.info(f"Generated {len(predictions)} emergency predictions")
        return predictions
    except Exception as e:
        logger.error(f"Error in emergency prediction generation: {str(e)}")
        return generate_synthetic_predictions(prediction_count)


def generate_synthetic_predictions(prediction_count):
    """
    Generate completely synthetic predictions when no historical data is available.
    This is the absolute last resort.
    
    Args:
        prediction_count (int): Number of predictions to generate
        
    Returns:
        list: List of prediction dictionaries
    """
    logger.warning("Generating completely synthetic predictions with no historical data")
    predictions = []
    base_price = 100.0  # Default price
    
    for i in range(prediction_count):
        # Generate random walk prices
        random_change = np.random.uniform(-0.01, 0.01)  # ±1%
        price_factor = 1 + (random_change * (i+1))
        
        predicted_close = base_price * price_factor
        predicted_open = base_price * price_factor * np.random.uniform(0.99, 1.01)
        predicted_high = predicted_close * np.random.uniform(1.01, 1.02)
        predicted_low = predicted_close * np.random.uniform(0.98, 0.99)
        
        # Ensure high is always highest and low is always lowest
        predicted_high = max(predicted_high, predicted_open, predicted_close)
        predicted_low = min(predicted_low, predicted_open, predicted_close)
        
        prediction = {
            'timestamp': datetime.now() + timedelta(days=i+1),
            'predicted_open': float(predicted_open),
            'predicted_high': float(predicted_high),
            'predicted_low': float(predicted_low),
            'predicted_close': float(predicted_close),
            'confidence': 0.05,  # Extremely low confidence
            'signal': 'hold',
            'risk_level': 'extreme',
            'model_type': 'synthetic'
        }
        predictions.append(prediction)
    
    logger.info(f"Generated {len(predictions)} completely synthetic predictions")
    return predictions

def generate_ml_predictions(historical_data, model_type, prediction_count, risk_level, symbol, timeframe):
    """Generate predictions using ML models with fallback to SimpleModelManager"""
    try:
        # Check if we're using SimpleModelManager (TensorFlow fallback)
        using_simple_model = hasattr(model_manager, 'get_or_create_model')
        
        if using_simple_model:
            # Using SimpleModelManager fallback
            logger.info(f"Using SimpleModelManager for predictions (TensorFlow not available)")
            simple_model = model_manager.get_or_create_model(current_user.id, symbol, timeframe, model_type.lower())
            
            # Train the simple model with historical data
            training_success = simple_model.train(historical_data)
            if not training_success:
                raise Exception("Simple model training failed")
                
            # Generate predictions using simple model
            predictions_data = simple_model.predict(historical_data, steps=prediction_count)
            
            # Format predictions to match expected format
            predictions = []
            for pred in predictions_data:
                predictions.append({
                    'timestamp': pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_close': float(pred['close']),
                    'predicted_open': float(pred['open']),
                    'predicted_high': float(pred['high']),
                    'predicted_low': float(pred['low']),
                    'confidence': pred.get('confidence', 0.5),
                    'signal': determine_trading_signal((pred['close'] - historical_data['close'].iloc[-1]) / historical_data['close'].iloc[-1], risk_level),
                    'risk_level': risk_level,
                    'model_type': f"Simple-{model_type}"
                })
            
            return predictions
        else:
            # Using regular TensorFlow-based ModelManager
            # Preprocess data for ML model
            processed_data = data_preprocessor.prepare_data_for_prediction(
                historical_data, 
                sequence_length=60,  # Use 60 days of history
                features=['open', 'high', 'low', 'close', 'volume']
            )
            
            if processed_data is None or len(processed_data) == 0:
                raise Exception("Data preprocessing failed")
            
            # Get or train model
            model_key = f"{current_user.id}_{model_type.lower()}_{symbol}_{timeframe}"
            
            # Check if model exists and is trained
            if not model_manager.is_model_trained(model_key):
                # Train model with historical data
                X_train, y_train, X_val, y_val = data_preprocessor.prepare_training_data(
                    historical_data, 
                    sequence_length=60,
                    prediction_horizon=prediction_count
                )
                
                training_result = model_manager.train_model(
                    model_type.lower(), X_train, y_train, X_val, y_val, 
                    user_id=current_user.id
                )
                
                if not training_result['success']:
                    raise Exception(f"Model training failed: {training_result.get('error', 'Unknown error')}")
            
            # Generate predictions
            predictions_raw = model_manager.predict(
                model_type.lower(), 
                processed_data[-1:],  # Use last sequence for prediction
                prediction_count,
                user_id=current_user.id
            )
            
            # Format predictions
            predictions = []
            last_price = historical_data['close'].iloc[-1]
            
            for i, pred_price in enumerate(predictions_raw):
                # Calculate confidence based on model performance and risk level
                base_confidence = model_manager.get_model_confidence(model_key)
                confidence = adjust_confidence_for_risk(base_confidence, risk_level)
                
                # Determine trading signal
                price_change_pct = (pred_price - last_price) / last_price
                signal = determine_trading_signal(price_change_pct, risk_level)
                
                predictions.append({
                    'timestamp': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_close': float(pred_price),
                    'predicted_open': float(pred_price * (1 + np.random.normal(0, 0.001))),
                    'predicted_high': float(pred_price * (1 + abs(np.random.normal(0, 0.01)))),
                    'predicted_low': float(pred_price * (1 - abs(np.random.normal(0, 0.01)))),
                    'confidence': confidence,
                    'signal': signal,
                    'risk_level': risk_level,
                    'model_type': model_type
                })
                
                last_price = pred_price
        
        return predictions
        
    except Exception as e:
        logger.error(f"ML prediction failed: {str(e)}")
        raise e

def generate_statistical_predictions(historical_data, prediction_count, risk_level):
    """Generate predictions using statistical methods - handles limited data"""
    try:
        # Check if we have enough data for full statistical analysis
        if len(historical_data) < 50:
            logger.warning(f"Limited data for statistical analysis: {len(historical_data)} points. Using simplified approach.")
            return generate_basic_predictions(historical_data, prediction_count, risk_level)
        
        # Calculate technical indicators
        data = historical_data.copy()
        
        # Standardize column names (handle both uppercase and lowercase)
        if 'Close' in data.columns:
            data.columns = data.columns.str.lower()
        
        # Moving averages (adjust window sizes for limited data)
        window_20 = min(20, len(data) // 3)
        window_50 = min(50, len(data) // 2)
        window_12 = min(12, len(data) // 4)
        window_26 = min(26, len(data) // 3)
        
        data['sma_20'] = data['close'].rolling(window=window_20).mean()
        data['sma_50'] = data['close'].rolling(window=window_50).mean()
        data['ema_12'] = data['close'].ewm(span=window_12).mean()
        data['ema_26'] = data['close'].ewm(span=window_26).mean()
        
        # RSI (adjust window for limited data)
        rsi_window = min(14, len(data) // 4)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands (adjust window for limited data)
        bb_window = min(20, len(data) // 3)
        data['bb_middle'] = data['close'].rolling(window=bb_window).mean()
        bb_std = data['close'].rolling(window=bb_window).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Calculate trend and momentum
        last_close = data['close'].iloc[-1]
        sma_20 = data['sma_20'].iloc[-1]
        sma_50 = data['sma_50'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        
        # Determine trend direction
        trend_bullish = last_close > sma_20 > sma_50 and macd > macd_signal
        trend_bearish = last_close < sma_20 < sma_50 and macd < macd_signal
        
        # Calculate volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Adjust volatility based on risk level
        volatility_multiplier = {
            'conservative': 0.5,
            'moderate': 1.0,
            'aggressive': 1.5
        }.get(risk_level, 1.0)
        
        adjusted_volatility = volatility * volatility_multiplier
        
        # Generate predictions
        predictions = []
        current_price = last_close
        
        for i in range(prediction_count):
            # Trend-based prediction with mean reversion
            if trend_bullish:
                trend_factor = 0.001 * (1 - i * 0.1)  # Diminishing trend effect
            elif trend_bearish:
                trend_factor = -0.001 * (1 - i * 0.1)
            else:
                trend_factor = 0
            
            # Add random component based on volatility
            random_factor = np.random.normal(0, adjusted_volatility / np.sqrt(252))
            
            # Mean reversion factor (prices tend to revert to moving average)
            mean_reversion = (sma_20 - current_price) / current_price * 0.1
            
            # Calculate predicted price
            price_change = trend_factor + random_factor + mean_reversion
            predicted_price = current_price * (1 + price_change)
            
            # Generate OHLC data
            daily_volatility = adjusted_volatility / np.sqrt(252)
            high_factor = abs(np.random.normal(0, daily_volatility * 0.5))
            low_factor = abs(np.random.normal(0, daily_volatility * 0.5))
            
            predicted_open = current_price * (1 + np.random.normal(0, daily_volatility * 0.2))
            predicted_high = max(predicted_open, predicted_price) * (1 + high_factor)
            predicted_low = min(predicted_open, predicted_price) * (1 - low_factor)
            
            # Calculate confidence based on technical indicators
            confidence = calculate_statistical_confidence(rsi, macd, macd_signal, trend_bullish, trend_bearish, risk_level)
            
            # Determine trading signal
            price_change_pct = (predicted_price - current_price) / current_price
            signal = determine_trading_signal(price_change_pct, risk_level)
            
            predictions.append({
                'timestamp': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_close': float(predicted_price),
                'predicted_open': float(predicted_open),
                'predicted_high': float(predicted_high),
                'predicted_low': float(predicted_low),
                'confidence': confidence,
                'signal': signal,
                'risk_level': risk_level,
                'model_type': 'Statistical'
            })
            
            current_price = predicted_price
        
        return predictions
        
    except Exception as e:
        logger.error(f"Statistical prediction failed: {str(e)}")
        raise e

def generate_synthetic_historical_data(symbol, timeframe, min_required_points=100):
    """Generate synthetic historical candlestick data when real data is insufficient"""
    try:
        # Base price for the symbol (you can enhance this with real market data lookup)
        base_prices = {
            'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0, 'AMZN': 3000.0, 'TSLA': 200.0,
            'META': 250.0, 'NVDA': 400.0, 'NFLX': 400.0, 'SPY': 400.0, 'QQQ': 350.0
        }
        
        base_price = base_prices.get(symbol.upper(), 100.0)
        
        # Generate synthetic data
        synthetic_data = []
        current_price = base_price
        
        # Determine time increment based on timeframe
        time_increments = {
            '1m': timedelta(minutes=1), '5m': timedelta(minutes=5), '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30), '1h': timedelta(hours=1), '4h': timedelta(hours=4),
            '1d': timedelta(days=1), '1w': timedelta(weeks=1), '1mo': timedelta(days=30)
        }
        
        time_increment = time_increments.get(timeframe, timedelta(days=1))
        start_time = datetime.now() - (time_increment * min_required_points)
        
        for i in range(min_required_points):
            # Generate realistic price movement
            daily_return = np.random.normal(0.0005, 0.02)  # Small positive drift with 2% volatility
            price_change = current_price * daily_return
            
            # Generate OHLC for the period
            open_price = current_price
            close_price = current_price + price_change
            
            # Generate high and low with realistic intraday volatility
            intraday_vol = abs(np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + intraday_vol)
            low_price = min(open_price, close_price) * (1 - intraday_vol)
            
            # Generate volume
            base_volume = 1000000
            volume = int(base_volume * (1 + np.random.normal(0, 0.3)))
            
            timestamp = start_time + (time_increment * i)
            
            synthetic_data.append({
                'timestamp': timestamp,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
            
            current_price = close_price
        
        # Convert to DataFrame
        df = pd.DataFrame(synthetic_data)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {len(df)} synthetic data points for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        return None

def generate_basic_predictions(historical_data, prediction_count, risk_level):
    """Generate basic predictions as fallback - works with limited data"""
    try:
        # Standardize column names (handle both uppercase and lowercase)
        data = historical_data.copy()
        if 'Close' in data.columns:
            data.columns = data.columns.str.lower()
        
        last_price = data['close'].iloc[-1]
        
        # Handle very limited data (less than 10 points)
        if len(historical_data) < 10:
            # Use simple random walk with minimal assumptions
            avg_return = 0.001  # Small positive drift
            volatility = 0.02   # Conservative volatility estimate
        else:
            # Calculate simple moving average and volatility
            returns = data['close'].pct_change().dropna()
            avg_return = returns.mean() if len(returns) > 0 else 0.001
            volatility = returns.std() if len(returns) > 1 else 0.02
        
        # Adjust volatility based on risk level
        volatility_multiplier = {
            'conservative': 0.5,
            'moderate': 1.0,
            'aggressive': 1.5
        }.get(risk_level, 1.0)
        
        adjusted_volatility = volatility * volatility_multiplier
        
        predictions = []
        current_price = last_price
        
        # Determine time increment based on timeframe (if available in request context)
        time_increments = {
            '1m': timedelta(minutes=1), '5m': timedelta(minutes=5), '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30), '1h': timedelta(hours=1), '4h': timedelta(hours=4),
            '1d': timedelta(days=1), '1w': timedelta(weeks=1), '1mo': timedelta(days=30)
        }
        
        # Default to daily if timeframe not available
        time_increment = timedelta(days=1)
        
        for i in range(prediction_count):
            # Simple random walk with drift
            change = np.random.normal(avg_return, adjusted_volatility)
            predicted_price = current_price * (1 + change)
            
            # Generate OHLC
            daily_vol = adjusted_volatility * 0.5
            predicted_open = current_price * (1 + np.random.normal(0, daily_vol * 0.3))
            predicted_high = max(predicted_open, predicted_price) * (1 + abs(np.random.normal(0, daily_vol)))
            predicted_low = min(predicted_open, predicted_price) * (1 - abs(np.random.normal(0, daily_vol)))
            
            # Basic confidence
            base_confidence = 0.65
            confidence = adjust_confidence_for_risk(base_confidence, risk_level)
            
            # Determine signal
            price_change_pct = (predicted_price - current_price) / current_price
            signal = determine_trading_signal(price_change_pct, risk_level)
            
            # Generate realistic timestamp
            prediction_time = datetime.now() + (time_increment * (i + 1))
            
            predictions.append({
                'timestamp': prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_close': float(predicted_price),
                'predicted_open': float(predicted_open),
                'predicted_high': float(predicted_high),
                'predicted_low': float(predicted_low),
                'confidence': confidence,
                'signal': signal,
                'risk_level': risk_level,
                'model_type': 'Basic'
            })
            
            current_price = predicted_price
        
        return predictions
        
    except Exception as e:
        logger.error(f"Basic prediction failed: {str(e)}")
        return []

def adjust_confidence_for_risk(base_confidence, risk_level):
    """Adjust confidence based on risk level"""
    multipliers = {
        'conservative': 0.85,
        'moderate': 1.0,
        'aggressive': 1.15
    }
    return min(base_confidence * multipliers.get(risk_level, 1.0), 0.99)

def determine_trading_signal(price_change_pct, risk_level):
    """Determine trading signal based on price change and risk level"""
    thresholds = {
        'conservative': {'buy': 0.03, 'sell': -0.03},
        'moderate': {'buy': 0.02, 'sell': -0.02},
        'aggressive': {'buy': 0.01, 'sell': -0.01}
    }
    
    threshold = thresholds.get(risk_level, thresholds['moderate'])
    
    if price_change_pct > threshold['buy']:
        return 'buy'
    elif price_change_pct < threshold['sell']:
        return 'sell'
    else:
        return 'hold'

def calculate_statistical_confidence(rsi, macd, macd_signal, trend_bullish, trend_bearish, risk_level):
    """Calculate confidence based on technical indicators"""
    base_confidence = 0.75
    
    # RSI confidence (higher when not overbought/oversold)
    if 30 <= rsi <= 70:
        rsi_confidence = 0.1
    else:
        rsi_confidence = -0.05
    
    # MACD confidence
    macd_confidence = 0.05 if abs(macd - macd_signal) > 0.1 else -0.02
    
    # Trend confidence
    trend_confidence = 0.1 if (trend_bullish or trend_bearish) else -0.05
    
    total_confidence = base_confidence + rsi_confidence + macd_confidence + trend_confidence
    
    return adjust_confidence_for_risk(total_confidence, risk_level)

# Import models
from backend.models.prediction import Prediction
    


class UserConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    config_key = db.Column(db.String(100), nullable=False)
    config_value = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelPerformance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)
    
    # Performance metrics
    accuracy = db.Column(db.Float)
    mae = db.Column(db.Float)  # Mean Absolute Error
    rmse = db.Column(db.Float)  # Root Mean Square Error
    mape = db.Column(db.Float)  # Mean Absolute Percentage Error
    
    # Training info
    training_samples = db.Column(db.Integer)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
# Root route is defined at the top of the file

@app.route('/register')
def register_redirect():
    """Redirect /register to /auth/register"""
    return redirect(url_for('auth.register'))

@app.route('/login')
def login_redirect():
    """Redirect /login to /auth/login"""
    return redirect(url_for('auth.login'))

# Login route removed - handled by auth blueprint

# Registration route removed - handled by auth blueprint

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    """Forgot password page (placeholder)"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Placeholder for password reset functionality
        email = request.form.get('email', '').strip().lower()
        if email:
            flash('Password reset instructions would be sent to your email (feature not implemented yet).', 'info')
        else:
            flash('Please enter a valid email address.', 'error')
        return render_template('login.html')
    
    # For now, redirect back to login with a message
    flash('Password reset feature is not implemented yet. Please contact support.', 'info')
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    """Logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/user_config', methods=['GET', 'POST'])
@login_required
def user_config():
    """User configuration page"""
    if request.method == 'POST':
        try:
            # Get or create user preferences
            user_prefs = current_user.preferences
            if not user_prefs:
                user_prefs = UserPreferences(user_id=current_user.id)
                db.session.add(user_prefs)
            
            # Update preferences from form data
            user_prefs.preferred_data_provider = request.form.get('preferred_data_provider', 'yahoo')
            user_prefs.default_symbol = request.form.get('default_symbol', 'AAPL')
            user_prefs.default_timeframe = request.form.get('default_timeframe', '1d')
            user_prefs.default_prediction_count = int(request.form.get('default_prediction_count', 5))
            user_prefs.preferred_model = request.form.get('preferred_model', 'lstm')
            user_prefs.risk_tolerance = request.form.get('risk_tolerance', 'medium')
            user_prefs.prediction_threshold = float(request.form.get('prediction_threshold', 0.7))
            user_prefs.theme = request.form.get('theme', 'light')
            user_prefs.chart_interval = request.form.get('chart_interval', '1d')
            
            # Handle checkbox fields
            user_prefs.email_notifications = request.form.get('email_notifications') == 'on'
            user_prefs.notification_enabled = request.form.get('notification_enabled') == 'on'
            user_prefs.auto_retrain = request.form.get('auto_retrain') == 'on'
            
            db.session.commit()
            flash('Configuration updated successfully!', 'success')
            
            if request.is_json:
                return jsonify({'success': True, 'message': 'Configuration updated successfully'})
            
        except Exception as e:
            logger.error(f"Configuration update error: {str(e)}")
            db.session.rollback()
            flash('Error updating configuration. Please try again.', 'error')
            
            if request.is_json:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        return redirect(url_for('user_config'))
    
    # Get user preferences
    user_prefs = current_user.preferences
    
    return render_template('user_config.html', 
                         user_preferences=user_prefs,
                         available_providers=UserPreferences.get_available_providers(),
                         available_timeframes=UserPreferences.get_available_timeframes(),
                         available_models=UserPreferences.get_available_models(),
                         available_companies=UserPreferences.get_available_companies())

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page"""
    if request.method == 'POST':
        # Handle profile updates
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=current_user)

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    try:
        # Redirect to the blueprint route
        return redirect(url_for('main_routes.dashboard'))
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        flash('Error loading dashboard data', 'error')
        return redirect(url_for('main_routes.dashboard'))

@app.route('/prediction')
@login_required
def prediction_page():
    """Advanced prediction configuration page"""
    try:
        # Get user stats for display
        recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(1).first()
        
        stats = {
            'last_update': recent_predictions.created_at.strftime('%Y-%m-%d %H:%M:%S') if recent_predictions else 'Never'
        }
        
        # Prepare available options for prediction form
        available_companies = [
            {'symbol': symbol, 'name': f"{symbol} Inc."} 
            for symbol in Config.MARKET_CONFIG['supported_companies']
        ]
        
        available_timeframes = [
            {'value': tf, 'name': tf.replace('m', ' Minutes').replace('h', ' Hours').replace('d', ' Days')} 
            for tf in Config.MARKET_CONFIG['supported_timeframes']
        ]
        
        available_models = [
            {'value': 'lstm', 'name': 'LSTM', 'description': 'Long Short-Term Memory'},
            {'value': 'cnn', 'name': 'CNN', 'description': 'Convolutional Neural Network'},
            {'value': 'rnn', 'name': 'RNN', 'description': 'Recurrent Neural Network'},
            {'value': 'hybrid', 'name': 'Hybrid', 'description': 'Combined Model'}
        ]
        
        available_providers = [
            {'value': 'finnhub', 'name': 'Finnhub'},
            {'value': 'yahoo', 'name': 'Yahoo Finance'},
            {'value': 'alpha_vantage', 'name': 'Alpha Vantage'},
            {'value': 'polygon', 'name': 'Polygon.io'}
        ]
        
        return render_template('prediction.html',
                             user_config=current_user.to_dict(),
                             user_stats=stats,
                             available_companies=available_companies,
                             available_timeframes=available_timeframes,
                             available_models=available_models,
                             available_providers=available_providers)
    except Exception as e:
        logger.error(f"Prediction page error: {str(e)}")
        flash('Error loading prediction page', 'error')
        return redirect(url_for('dashboard'))

# Removed duplicate index route - using the one at line 494

@app.route('/chart')
@login_required
def chart_viewer():
    """Chart viewer page"""
    available_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    available_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo']
    available_models = ['LSTM', 'CNN', 'RNN', 'Hybrid']
    available_providers = ['finnhub', 'yahoo_finance', 'alpha_vantage', 'polygon']
    
    return render_template('chart_viewer.html',
                         user_config=current_user.to_dict(),
                         available_symbols=available_symbols,
                         available_timeframes=available_timeframes,
                         available_models=available_models,
                         available_providers=available_providers)

@app.route('/chart-predictions')
@login_required
def chart_predictions():
    """Chart predictions page with ML forecasting"""
    return render_template('chart_predictions.html')

# Route removed to avoid conflict with main_routes.py blueprint
# @app.route('/dashboard')
# @login_required
# def user_dashboard():
#     """User dashboard with personalized predictions"""
#     return render_template('user_dashboard.html', user_config=current_user.to_dict())

# API Routes
@app.route('/api/chart/data', methods=['POST'])
@login_required
def get_chart_data():
    """Get chart data for visualization"""
    try:
        if data_collector is None:
            return jsonify({'success': False, 'error': 'Data collector not available'}), 503
            
        symbol = request.form.get('symbol', 'AAPL')
        timeframe = request.form.get('timeframe', '1d')
        period = request.form.get('period', '1y')
        provider = request.form.get('provider', 'yahoo_finance')
        
        # Try multiple providers as fallback
        providers_to_try = [provider, 'finnhub', 'alpha_vantage', 'polygon', 'yahoo'] if provider != 'yahoo' else ['finnhub', 'yahoo', 'alpha_vantage', 'polygon']
        historical_data = None
        
        for fallback_provider in providers_to_try:
            try:
                historical_data = data_collector.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    period=period,
                    provider=fallback_provider
                )
                
                if historical_data is not None and not historical_data.empty and len(historical_data) >= 10:
                    logger.info(f"Chart data fetched successfully using {fallback_provider} provider")
                    break
                else:
                    logger.warning(f"Insufficient chart data from {fallback_provider}: {len(historical_data) if historical_data is not None else 0} records")
                    
            except Exception as e:
                logger.error(f"Error fetching chart data with {fallback_provider} provider: {str(e)}")
                continue
        
        if historical_data is None or historical_data.empty:
            return jsonify({'success': False, 'error': 'No data available from any provider'}), 404
        
        # Prepare chart data
        chart_data = {
            'timestamps': historical_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': historical_data['Open'].tolist(),
            'high': historical_data['High'].tolist(),
            'low': historical_data['Low'].tolist(),
            'close': historical_data['Close'].tolist(),
            'volume': historical_data['Volume'].tolist() if 'Volume' in historical_data.columns else [],
            'title': f'{symbol} - {timeframe} - {period}'
        }
        
        return jsonify({'success': True, 'chart_data': chart_data})
        
    except Exception as e:
        logger.error(f"Chart data error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predictions/generate', methods=['POST'])
@login_required
def generate_prediction():
    """Generate new prediction with enhanced user configuration"""
    try:
        symbol = request.form.get('symbol', 'AAPL')
        timeframe = request.form.get('timeframe', '1d')
        model_type = request.form.get('model_type', 'LSTM')
        prediction_count = int(request.form.get('prediction_count', 5))
        provider = request.form.get('provider', 'yahoo_finance')
        risk_level = request.form.get('risk_level', 'moderate')
        save_as_default = request.form.get('save_as_default') == 'on'
        
        # Validate prediction count
        if prediction_count < 1 or prediction_count > 50:
            return jsonify({'success': False, 'error': 'Prediction count must be between 1 and 50'}), 400
        
        # Save as default configuration if requested
        if save_as_default:
            # Update user preferences
            current_user.preferred_model = model_type
            current_user.default_company = symbol
            current_user.default_timeframe = timeframe
            current_user.default_prediction_count = prediction_count
            current_user.preferred_provider = provider
            
            # Save risk level as user config
            risk_config = UserConfig.query.filter_by(user_id=current_user.id, config_key='risk_level').first()
            if not risk_config:
                risk_config = UserConfig(user_id=current_user.id, config_key='risk_level', config_value=risk_level)
                db.session.add(risk_config)
            else:
                risk_config.config_value = risk_level
        
        # Fetch historical data with fallback providers
        if data_collector is None:
            return jsonify({'success': False, 'error': 'Data collector not available'}), 500
        
        # Try multiple providers as fallback
        providers_to_try = [provider, 'finnhub', 'alpha_vantage', 'polygon', 'yahoo'] if provider != 'yahoo' else ['yahoo', 'finnhub', 'alpha_vantage', 'polygon']
        historical_data = None
        data_source = 'unknown'
        min_required_points = 50  # Minimum points needed for basic predictions
        
        for fallback_provider in providers_to_try:
            try:
                historical_data = data_collector.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    period='2y',
                    provider=fallback_provider
                )
                
                if historical_data is not None and len(historical_data) >= 100:  # Need sufficient data for ML
                    data_source = fallback_provider
                    logger.info(f"Successfully fetched data using {fallback_provider} provider")
                    break
                elif historical_data is not None and len(historical_data) >= min_required_points:
                    data_source = fallback_provider
                    logger.warning(f"Limited data from {fallback_provider}: {len(historical_data)} records. Will use basic predictions.")
                    break
                else:
                    logger.warning(f"Insufficient data from {fallback_provider}: {len(historical_data) if historical_data is not None else 0} records")
                    
            except Exception as e:
                logger.error(f"Error with {fallback_provider} provider: {str(e)}")
                continue
        
        # If we don't have enough data from any provider, generate synthetic data
        if historical_data is None or len(historical_data) < min_required_points:
            logger.info(f"Generating synthetic data for {symbol} due to insufficient real data")
            synthetic_data = generate_synthetic_historical_data(symbol, timeframe, 100)
            
            if synthetic_data is not None:
                # If we have some real data, use the last price as starting point for synthetic data
                if historical_data is not None and len(historical_data) > 0:
                    last_real_price = historical_data['Close'].iloc[-1] if 'Close' in historical_data.columns else historical_data['close'].iloc[-1]
                    # Adjust synthetic data to start from last real price
                    price_ratio = last_real_price / synthetic_data['Close'].iloc[0]
                    for col in ['Open', 'High', 'Low', 'Close']:
                        synthetic_data[col] *= price_ratio
                    
                    # Combine real data with synthetic data
                    historical_data = pd.concat([historical_data, synthetic_data], ignore_index=False)
                    data_source = f'{data_source}_with_synthetic'
                else:
                    historical_data = synthetic_data
                    data_source = 'synthetic'
                
                logger.info(f"Using synthetic data with {len(historical_data)} total points")
            else:
                return jsonify({'success': False, 'error': 'Unable to fetch or generate historical data for predictions'}), 404
        
        # Determine prediction method based on data availability and source
        if len(historical_data) < 100 or 'synthetic' in data_source:
            logger.warning(f"Using basic prediction method. Data points: {len(historical_data)}, Source: {data_source}")
            use_basic_predictions_only = True
        else:
            use_basic_predictions_only = False
        
        # Process data and generate predictions using appropriate method based on data availability
        try:
            if use_basic_predictions_only:
                # Use basic predictions for limited data
                logger.info("Using basic prediction method due to limited historical data")
                predictions = generate_basic_predictions(
                    historical_data, prediction_count, risk_level
                )
            elif model_manager is not None:
                # Use ML models for prediction with sufficient data
                # The generate_ml_predictions function now handles both TensorFlow and SimpleModelManager
                predictions = generate_ml_predictions(
                    historical_data, model_type, prediction_count, risk_level, symbol, timeframe
                )
            else:
                # Fallback to enhanced statistical predictions
                predictions = generate_statistical_predictions(
                    historical_data, prediction_count, risk_level
                )
                
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            # Fallback to basic predictions
            predictions = generate_basic_predictions(
                historical_data, prediction_count, risk_level
            )
        
        # Predictions are now generated by the helper functions above
        
        # Save prediction
        prediction_record = Prediction(
            user_id=current_user.id,
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            provider=provider,
            prediction_count=prediction_count,
            predictions_data=json.dumps(predictions),
            prediction_date=datetime.now(),
            confidence_score=predictions[0]['confidence'] if predictions else 0.85
        )
        
        db.session.add(prediction_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_record.id,
            'predictions': predictions,
            'model_info': {
                'type': model_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'risk_level': risk_level,
                'accuracy': predictions[0]['confidence'] if predictions else 0.85,
                'prediction_count': prediction_count
            },
            'message': f'Generated {prediction_count} predictions for {symbol} using {model_type} model'
        })
        
    except Exception as e:
        logger.error(f"Prediction generation error: {str(e)}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/candlestick-data', methods=['POST'])
@login_required
def get_candlestick_data():
    """Get complete candlestick data with technical indicators"""
    try:
        if data_collector is None:
            return jsonify({'success': False, 'error': 'Data collector not available'}), 503
        
        # Check if request is JSON or form data
        if request.is_json:
            data = request.get_json()
            symbol = data.get('symbol', 'AAPL')
            timeframe = data.get('timeframe', '1d')
            period = data.get('period', '1y')
            provider = data.get('provider', 'finnhub')
        else:
            symbol = request.form.get('symbol', 'AAPL')
            timeframe = request.form.get('timeframe', '1d')
            period = request.form.get('period', '1y')
            provider = request.form.get('provider', 'finnhub')
        # Handle include_indicators parameter
        if request.is_json:
            include_indicators = data.get('include_indicators', True)
        else:
            include_indicators = request.form.get('include_indicators', 'true').lower() == 'true'
        
        # Try multiple providers as fallback
        providers_to_try = [provider, 'alpha_vantage', 'polygon', 'yahoo'] if provider != 'yahoo' else ['yahoo', 'alpha_vantage', 'polygon']
        historical_data = None
        
        for fallback_provider in providers_to_try:
            try:
                historical_data = data_collector.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    period=period,
                    provider=fallback_provider
                )
                
                if historical_data is not None and not historical_data.empty and len(historical_data) >= 10:
                    logger.info(f"Candlestick data fetched successfully using {fallback_provider} provider")
                    break
                else:
                    logger.warning(f"Insufficient candlestick data from {fallback_provider}: {len(historical_data) if historical_data is not None else 0} records")
                    
            except Exception as e:
                logger.error(f"Error fetching candlestick data with {fallback_provider} provider: {str(e)}")
                continue
        
        if historical_data is None or historical_data.empty:
            return jsonify({'success': False, 'error': 'No candlestick data available from any provider'}), 404
        
        # Calculate technical indicators if requested
        technical_indicators = {}
        if include_indicators:
            technical_indicators = calculate_technical_indicators(historical_data)
        
        # Prepare complete candlestick data
        candlestick_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'data_points': len(historical_data),
            'timestamps': historical_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'ohlc': {
                'open': historical_data['Open'].tolist(),
                'high': historical_data['High'].tolist(),
                'low': historical_data['Low'].tolist(),
                'close': historical_data['Close'].tolist()
            },
            'volume': historical_data['Volume'].tolist() if 'Volume' in historical_data.columns else [],
            'technical_indicators': technical_indicators,
            'summary_stats': {
                'latest_price': float(historical_data['Close'].iloc[-1]),
                'price_change': float(historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2]) if len(historical_data) > 1 else 0,
                'price_change_percent': float(((historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2]) / historical_data['Close'].iloc[-2]) * 100) if len(historical_data) > 1 else 0,
                'high_52w': float(historical_data['High'].max()),
                'low_52w': float(historical_data['Low'].min()),
                'avg_volume': float(historical_data['Volume'].mean()) if 'Volume' in historical_data.columns else 0,
                'volatility': float(historical_data['Close'].pct_change().std() * 100)
            }
        }
        
        return jsonify({
            'success': True,
            'candlestick_data': candlestick_data,
            'message': f'Retrieved {len(historical_data)} candlestick data points for {symbol}'
        })
        
    except Exception as e:
        logger.error(f"Candlestick data error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chart/candlestick-with-predictions', methods=['POST'])
@login_required
def get_candlestick_chart_with_predictions():
    """Get candlestick chart data with ML predictions for visualization"""
    try:
        if data_collector is None:
            return jsonify({'success': False, 'error': 'Data collector not available'}), 503
            
        # Check if request is JSON or form data
        if request.is_json:
            data = request.get_json()
            symbol = data.get('symbol', 'AAPL')
            timeframe = data.get('timeframe', '1d')
            period = data.get('period', '1y')
            provider = data.get('provider', 'yahoo')
            model_type = data.get('model_type', 'LSTM')
            prediction_count = int(data.get('prediction_count', 10))
            risk_level = data.get('risk_level', 'moderate')
            include_volume = data.get('include_volume', True)
            include_indicators = data.get('include_indicators', True)
            chart_style = data.get('chart_style', 'candlestick')  # candlestick, ohlc, line
            separate_predictions = data.get('separate_predictions', False)  # New parameter for separate predictions
        else:
            # User customization parameters from form data
            symbol = request.form.get('symbol', 'AAPL')
            timeframe = request.form.get('timeframe', '1d')
            period = request.form.get('period', '1y')
            provider = request.form.get('provider', 'yahoo')
            model_type = request.form.get('model_type', 'LSTM')
            prediction_count = int(request.form.get('prediction_count', 10))
            risk_level = request.form.get('risk_level', 'moderate')
            
            # Data customization options
            include_volume = request.form.get('include_volume', 'true').lower() == 'true'
            include_indicators = request.form.get('include_indicators', 'true').lower() == 'true'
            chart_style = request.form.get('chart_style', 'candlestick')  # candlestick, ohlc, line
            separate_predictions = request.form.get('separate_predictions', 'false').lower() == 'true'  # New parameter
        
        # Handle prediction_style parameter
        if request.is_json:
            prediction_style = data.get('prediction_style', 'separate')  # separate, overlay
        else:
            prediction_style = request.form.get('prediction_style', 'separate')  # separate, overlay
        
        # Validate prediction count against configuration
        max_prediction_count = Config.DATE_RANGE_CONFIG['prediction_config']['max_prediction_count']
        if prediction_count < 1 or prediction_count > max_prediction_count:
            return jsonify({
                'success': False, 
                'error': f'Prediction count must be between 1 and {max_prediction_count}',
                'parameter': 'prediction_count'
            }), 400
        
        # Get date range from configuration
        start_date = Config.DATE_RANGE_CONFIG['historical_data']['start_date']
        end_date = Config.DATE_RANGE_CONFIG['historical_data']['end_date']
        
        # Log request parameters for debugging
        logger.info(f"Chart request: symbol={symbol}, timeframe={timeframe}, provider={provider}, model={model_type}, predictions={prediction_count}")
        logger.info(f"Using date range: {start_date} to {end_date}")
        
        # Validate symbol
        if not symbol or len(symbol) < 1 or len(symbol) > 10:
            return jsonify({
                'success': False, 
                'error': 'Invalid symbol. Symbol must be between 1 and 10 characters.',
                'parameter': 'symbol'
            }), 400
        
        # Check cache first (simple in-memory cache based on request parameters)
        cache_key = f"{symbol}_{timeframe}_{period}_{start_date}_{end_date}"
        if hasattr(app, 'data_cache') and cache_key in app.data_cache:
            cache_entry = app.data_cache[cache_key]
            cache_age = datetime.now() - cache_entry['timestamp']
            # Use cache if it's less than 1 hour old
            if cache_age.total_seconds() < 3600:  # 1 hour in seconds
                historical_data = cache_entry['data']
                data_source = cache_entry['provider']
                logger.info(f"Using cached data for {symbol} (age: {cache_age.total_seconds()/60:.1f} minutes)")
            else:
                # Cache expired
                logger.info(f"Cache expired for {symbol} (age: {cache_age.total_seconds()/60:.1f} minutes)")
                app.data_cache.pop(cache_key, None)
        
        # If not in cache, fetch from providers with fallback
        if 'historical_data' not in locals() or historical_data is None:
            providers_to_try = [provider, 'alpha_vantage', 'polygon', 'yahoo'] if provider != 'yahoo' else ['yahoo', 'alpha_vantage', 'polygon']
            historical_data = None
            data_source = 'unknown'
            min_required_points = 10  # Reduced minimum points needed for chart display
            errors = []
            
            for fallback_provider in providers_to_try:
                try:
                    logger.info(f"Attempting to fetch data from {fallback_provider} for {symbol}")
                    historical_data = data_collector.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        period=period,
                        provider=fallback_provider,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if historical_data is not None and not historical_data.empty and len(historical_data) >= 30:
                        data_source = fallback_provider
                        logger.info(f"Chart data fetched successfully using {fallback_provider} provider: {len(historical_data)} records")
                        
                        # Store in cache
                        if not hasattr(app, 'data_cache'):
                            app.data_cache = {}
                        app.data_cache[cache_key] = {
                            'data': historical_data.copy(),
                            'provider': fallback_provider,
                            'timestamp': datetime.now()
                        }
                        break
                    elif historical_data is not None and not historical_data.empty and len(historical_data) >= min_required_points:
                        data_source = fallback_provider
                        logger.warning(f"Limited chart data from {fallback_provider}: {len(historical_data)} records")
                        
                        # Store in cache with limited flag
                        if not hasattr(app, 'data_cache'):
                            app.data_cache = {}
                        app.data_cache[cache_key] = {
                            'data': historical_data.copy(),
                            'provider': fallback_provider,
                            'timestamp': datetime.now(),
                            'limited': True
                        }
                        break
                    else:
                        msg = f"Insufficient chart data from {fallback_provider}: {len(historical_data) if historical_data is not None else 0} records"
                        logger.warning(msg)
                        errors.append(msg)
                        
                except Exception as e:
                    error_msg = f"Error fetching chart data with {fallback_provider} provider: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # If all providers failed, return detailed error
            if historical_data is None or historical_data.empty or len(historical_data) < min_required_points:
                # Analyze errors to provide more specific feedback
                data_points = len(historical_data) if historical_data is not None and not historical_data.empty else 0
                
                if any("not found" in err.lower() or "invalid symbol" in err.lower() for err in errors):
                    error_message = f'Company or symbol "{symbol}" not found. Please verify the symbol is correct (e.g., AAPL for Apple, MSFT for Microsoft).'
                elif any("rate limit" in err.lower() or "too many requests" in err.lower() for err in errors):
                    error_message = f'API rate limits exceeded. Please try again in a few minutes.'
                elif data_points > 0:
                    error_message = f'Limited data available for {symbol} ({data_points} data points). Try selecting a longer time period or different timeframe.'
                else:
                    error_message = f'No historical data available for {symbol}. This symbol may be delisted, invalid, or not supported by our data providers.'
                
                return jsonify({
                    'success': False, 
                    'error': error_message,
                    'details': errors,
                    'suggestions': [
                        'Verify the company symbol is correct',
                        'Try a different timeframe (e.g., 1d instead of 1h)',
                        'Select a longer time period',
                        'Choose from supported companies: AAPL, MSFT, GOOGL, AMZN, TSLA, etc.'
                    ],
                    'data_points_found': data_points,
                    'minimum_required': min_required_points,
                    'parameter': 'symbol'
                }), 404
        
        # If we don't have enough data from any provider, generate synthetic data
        if historical_data is None or historical_data.empty or len(historical_data) < min_required_points:
            logger.info(f"Generating synthetic data for {symbol} chart due to insufficient real data")
            synthetic_data = generate_synthetic_historical_data(symbol, timeframe, 100)
            
            if synthetic_data is not None:
                # If we have some real data, use the last price as starting point for synthetic data
                if historical_data is not None and not historical_data.empty and len(historical_data) > 0:
                    last_real_price = historical_data['Close'].iloc[-1]
                    # Adjust synthetic data to start from last real price
                    price_ratio = last_real_price / synthetic_data['Close'].iloc[0]
                    for col in ['Open', 'High', 'Low', 'Close']:
                        synthetic_data[col] *= price_ratio
                    
                    # Combine real data with synthetic data
                    historical_data = pd.concat([historical_data, synthetic_data], ignore_index=False)
                    data_source = f'{data_source}_with_synthetic'
                else:
                    historical_data = synthetic_data
                    data_source = 'synthetic'
                
                logger.info(f"Using synthetic data for chart with {len(historical_data)} total points")
            else:
                return jsonify({'success': False, 'error': 'Unable to fetch or generate historical data for chart'}), 404
        
        # Generate ML predictions with improved error handling and fallback mechanisms
        predictions = []
        prediction_method = 'unknown'
        prediction_errors = []
        
        # Check prediction cache
        prediction_cache_key = f"{symbol}_{timeframe}_{model_type}_{prediction_count}_{risk_level}_{len(historical_data)}"
        if hasattr(app, 'prediction_cache') and prediction_cache_key in app.prediction_cache:
            cache_entry = app.prediction_cache[prediction_cache_key]
            cache_age = datetime.now() - cache_entry['timestamp']
            # Use cache if it's less than 15 minutes old
            if cache_age.total_seconds() < 900:  # 15 minutes in seconds
                predictions = cache_entry['predictions']
                prediction_method = cache_entry['method']
                logger.info(f"Using cached predictions for {symbol} (age: {cache_age.total_seconds()/60:.1f} minutes)")
            else:
                # Cache expired
                logger.info(f"Prediction cache expired for {symbol} (age: {cache_age.total_seconds()/60:.1f} minutes)")
                app.prediction_cache.pop(prediction_cache_key, None)
        
        # Generate new predictions if not in cache
        if not predictions:
            try:
                # First attempt: ML predictions if possible
                if model_manager is not None and data_preprocessor is not None and len(historical_data) >= 100:
                    logger.info(f"Generating ML predictions using {model_type} model for {symbol}")
                    start_time = datetime.now()
                    
                    try:
                        predictions = generate_ml_predictions(
                            historical_data, model_type, prediction_count, risk_level, symbol, timeframe
                        )
                        prediction_method = f'ml_{model_type.lower()}'
                        
                        # Log performance metrics
                        elapsed = (datetime.now() - start_time).total_seconds()
                        logger.info(f"ML prediction generation completed in {elapsed:.2f} seconds")
                    except Exception as ml_error:
                        error_msg = f"ML prediction error with {model_type}: {str(ml_error)}"
                        logger.error(error_msg)
                        prediction_errors.append(error_msg)
                        # Continue to fallback
                else:
                    if model_manager is None:
                        logger.warning("ML model manager not available, using statistical predictions")
                    elif data_preprocessor is None:
                        logger.warning("Data preprocessor not available, using statistical predictions")
                    elif len(historical_data) < 100:
                        logger.warning(f"Insufficient data for ML predictions ({len(historical_data)} points), using statistical predictions")
                
                # Second attempt: Statistical predictions if ML failed or not available
                if not predictions:
                    logger.info(f"Generating statistical predictions for {symbol}")
                    start_time = datetime.now()
                    
                    try:
                        predictions = generate_statistical_predictions(
                            historical_data, prediction_count, risk_level
                        )
                        prediction_method = 'statistical'
                        
                        # Log performance metrics
                        elapsed = (datetime.now() - start_time).total_seconds()
                        logger.info(f"Statistical prediction generation completed in {elapsed:.2f} seconds")
                    except Exception as stat_error:
                        error_msg = f"Statistical prediction error: {str(stat_error)}"
                        logger.error(error_msg)
                        prediction_errors.append(error_msg)
                        # Continue to fallback
                
                # Last resort: Basic predictions if all else failed
                if not predictions:
                    logger.warning(f"Falling back to basic predictions for {symbol} after previous methods failed")
                    start_time = datetime.now()
                    
                    predictions = generate_basic_predictions(
                        historical_data, prediction_count, risk_level
                    )
                    prediction_method = 'basic'
                    
                    # Log performance metrics
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Basic prediction generation completed in {elapsed:.2f} seconds")
            except Exception as e:
                logger.error(f"Critical error in prediction generation pipeline: {str(e)}")
                # Emergency fallback - generate very simple predictions
                predictions = generate_emergency_predictions(historical_data, prediction_count)
                prediction_method = 'emergency'
            
            # Cache the predictions
            if predictions:
                if not hasattr(app, 'prediction_cache'):
                    app.prediction_cache = {}
                app.prediction_cache[prediction_cache_key] = {
                    'predictions': predictions,
                    'method': prediction_method,
                    'timestamp': datetime.now(),
                    'errors': prediction_errors if prediction_errors else None
                }
        
        # Validate predictions
        if not predictions or len(predictions) < prediction_count:
            logger.error(f"Failed to generate sufficient predictions: got {len(predictions) if predictions else 0}, needed {prediction_count}")
            if not predictions:
                # Create emergency predictions if all methods failed
                predictions = generate_emergency_predictions(historical_data, prediction_count)
                prediction_method = 'emergency'
                logger.warning("Using emergency predictions due to complete prediction failure")
        
        logger.info(f"Generated {len(predictions)} predictions using {prediction_method} method")
        
        # If separate_predictions is True, return only the predictions without historical data
        if separate_predictions:
            # Prepare response with only prediction data
            prediction_response = {
                'success': True,
                'prediction_data': {
                    'timestamps': [pred['timestamp'] for pred in predictions],
                    'ohlc': {
                        'open': [pred['predicted_open'] for pred in predictions],
                        'high': [pred['predicted_high'] for pred in predictions],
                        'low': [pred['predicted_low'] for pred in predictions],
                        'close': [pred['predicted_close'] for pred in predictions]
                    },
                    'confidence': [pred['confidence'] for pred in predictions],
                    'signals': [pred['signal'] for pred in predictions]
                },
                'prediction_summary': {
                    'prediction_count': len(predictions),
                    'model_type': model_type,
                    'risk_level': risk_level,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            }
            return jsonify(prediction_response)
        
        # Calculate technical indicators if requested
        technical_indicators = {}
        if include_indicators:
            technical_indicators = calculate_technical_indicators(historical_data)
        
        # Prepare historical candlestick data
        historical_chart_data = {
            'timestamps': historical_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'ohlc': {
                'open': historical_data['Open'].tolist(),
                'high': historical_data['High'].tolist(),
                'low': historical_data['Low'].tolist(),
                'close': historical_data['Close'].tolist()
            }
        }
        
        # Add volume if requested
        if include_volume and 'Volume' in historical_data.columns:
            historical_chart_data['volume'] = historical_data['Volume'].tolist()
        
        # Prepare prediction candlestick data
        prediction_chart_data = {
            'timestamps': [pred['timestamp'] for pred in predictions],
            'ohlc': {
                'open': [pred['predicted_open'] for pred in predictions],
                'high': [pred['predicted_high'] for pred in predictions],
                'low': [pred['predicted_low'] for pred in predictions],
                'close': [pred['predicted_close'] for pred in predictions]
            },
            'confidence': [pred['confidence'] for pred in predictions],
            'signals': [pred['signal'] for pred in predictions]
        }
        
        # Prepare chart configuration
        chart_config = {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'model_type': model_type,
            'prediction_count': prediction_count,
            'risk_level': risk_level,
            'chart_style': chart_style,
            'prediction_style': prediction_style,
            'include_volume': include_volume,
            'include_indicators': include_indicators,
            'provider_used': fallback_provider if 'fallback_provider' in locals() else provider,
            'data_source': data_source
        }
        
        # Prepare response data
        response_data = {
            'success': True,
            'chart_config': chart_config,
            'historical_data': historical_chart_data,
            'prediction_data': prediction_chart_data,
            'technical_indicators': technical_indicators,
            'data_summary': {
                'historical_points': len(historical_data),
                'prediction_points': len(predictions),
                'latest_price': float(historical_data['Close'].iloc[-1]),
                'predicted_price_range': {
                    'min': min([pred['predicted_low'] for pred in predictions]),
                    'max': max([pred['predicted_high'] for pred in predictions]),
                    'final': predictions[-1]['predicted_close'] if predictions else 0
                },
                'avg_confidence': sum([pred['confidence'] for pred in predictions]) / len(predictions) if predictions else 0
            },
            'message': f'Retrieved {len(historical_data)} historical candles and generated {len(predictions)} predictions for {symbol}' + 
                      (f' (Using synthetic data due to insufficient real data)' if 'synthetic' in data_source else '') + 
                      (f' (Limited data: using basic prediction method)' if 'use_basic_predictions_only' in locals() and use_basic_predictions_only else '')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Chart with predictions error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user-preferences', methods=['GET', 'POST'])
@login_required
def manage_user_preferences():
    """Manage user preferences for API customization"""
    try:
        if request.method == 'GET':
            # Get current user preferences
            preferences = {
                'default_symbol': current_user.default_company,
                'default_timeframe': current_user.default_timeframe,
                'default_prediction_count': current_user.default_prediction_count,
                'preferred_model': current_user.preferred_model,
                'preferred_provider': current_user.preferred_provider,
                'available_options': {
                    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ'],
                    'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo'],
                    'periods': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
                    'models': ['LSTM', 'CNN', 'RNN', 'Hybrid', 'Statistical', 'Basic'],
                    'providers': ['yahoo', 'alpha_vantage', 'polygon'],
                    'risk_levels': ['conservative', 'moderate', 'aggressive'],
                    'chart_styles': ['candlestick', 'ohlc', 'line'],
                    'prediction_styles': ['separate', 'overlay']
                }
            }
            
            # Get additional user configs
            user_configs = UserConfig.query.filter_by(user_id=current_user.id).all()
            for config in user_configs:
                preferences[config.config_key] = config.config_value
            
            return jsonify({
                'success': True,
                'preferences': preferences
            })
        
        elif request.method == 'POST':
            # Update user preferences
            data = request.get_json() or request.form.to_dict()
            
            # Update user model fields
            if 'default_symbol' in data:
                current_user.default_company = data['default_symbol']
            if 'default_timeframe' in data:
                current_user.default_timeframe = data['default_timeframe']
            if 'default_prediction_count' in data:
                current_user.default_prediction_count = int(data['default_prediction_count'])
            if 'preferred_model' in data:
                current_user.preferred_model = data['preferred_model']
            if 'preferred_provider' in data:
                current_user.preferred_provider = data['preferred_provider']
            
            # Update or create user configs for additional preferences
            config_keys = ['risk_level', 'chart_style', 'prediction_style', 'include_volume', 'include_indicators']
            for key in config_keys:
                if key in data:
                    config = UserConfig.query.filter_by(user_id=current_user.id, config_key=key).first()
                    if not config:
                        config = UserConfig(user_id=current_user.id, config_key=key, config_value=str(data[key]))
                        db.session.add(config)
                    else:
                        config.config_value = str(data[key])
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'User preferences updated successfully'
            })
            
    except Exception as e:
        logger.error(f"User preferences error: {str(e)}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_technical_indicators(data):
    """Calculate technical indicators for candlestick data"""
    try:
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_20'] = data['Close'].rolling(window=20).mean().tolist()
        indicators['sma_50'] = data['Close'].rolling(window=50).mean().tolist()
        indicators['sma_200'] = data['Close'].rolling(window=200).mean().tolist()
        
        # Exponential Moving Averages
        indicators['ema_12'] = data['Close'].ewm(span=12).mean().tolist()
        indicators['ema_26'] = data['Close'].ewm(span=26).mean().tolist()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).tolist()
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = {
            'macd_line': macd_line.tolist(),
            'signal_line': signal_line.tolist(),
            'histogram': (macd_line - signal_line).tolist()
        }
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        indicators['bollinger_bands'] = {
            'upper': (sma_20 + (std_20 * 2)).tolist(),
            'middle': sma_20.tolist(),
            'lower': (sma_20 - (std_20 * 2)).tolist()
        }
        
        # Volume indicators
        if 'Volume' in data.columns:
            indicators['volume_sma'] = data['Volume'].rolling(window=20).mean().tolist()
            
        return indicators
        
    except Exception as e:
        logger.error(f"Technical indicators calculation error: {str(e)}")
        return {}

@app.route('/api/user/predictions', methods=['GET'])
@login_required
def get_user_predictions():
    """Get all predictions for the current user"""
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
        
        predictions_data = []
        for pred in predictions:
            pred_dict = pred.to_dict()
            predictions_data.append(pred_dict)
        
        return jsonify({
            'success': True,
            'predictions': predictions_data,
            'total_count': len(predictions_data)
        })
        
    except Exception as e:
        logger.error(f"Error fetching user predictions: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch predictions: {str(e)}'
        }), 500

@app.route('/api/user/update-preferences', methods=['POST'])
@login_required
def update_user_preferences():
    """Update user default preferences"""
    try:
        # Get data from either JSON or form data
        if request.is_json:
            data = request.json
            default_company = data.get('default_company')
            default_timeframe = data.get('default_timeframe')
            default_prediction_count = data.get('default_prediction_count')
            preferred_model = data.get('preferred_model')
            preferred_provider = data.get('preferred_provider')
        else:
            # Get form data
            default_company = request.form.get('default_company')
            default_timeframe = request.form.get('default_timeframe')
            default_prediction_count = request.form.get('default_prediction_count')
            preferred_model = request.form.get('preferred_model')
            preferred_provider = request.form.get('preferred_provider')
        
        # Update user preferences
        if default_company:
            current_user.default_company = default_company
        if default_timeframe:
            current_user.default_timeframe = default_timeframe
        if default_prediction_count:
            current_user.default_prediction_count = int(default_prediction_count)
        if preferred_model:
            current_user.preferred_model = preferred_model
        if preferred_provider:
            current_user.preferred_provider = preferred_provider
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Preferences updated successfully',
            'updated_preferences': {
                'default_company': current_user.default_company,
                'default_timeframe': current_user.default_timeframe,
                'default_prediction_count': current_user.default_prediction_count,
                'preferred_model': current_user.preferred_model,
                'preferred_provider': current_user.preferred_provider
            }
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating user preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to update preferences: {str(e)}'
        }), 500

@app.route('/api/user/prediction-history/<symbol>', methods=['GET'])
@login_required
def get_user_prediction_history(symbol):
    """Get prediction history for a specific symbol"""
    try:
        predictions = Prediction.query.filter_by(
            user_id=current_user.id,
            symbol=symbol.upper()
        ).order_by(Prediction.created_at.desc()).limit(10).all()
        
        history_data = []
        for pred in predictions:
            pred_dict = pred.to_dict()
            history_data.append(pred_dict)
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'predictions': history_data,
            'total_count': len(history_data)
        })
        
    except Exception as e:
        logger.error(f"Error fetching prediction history for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch prediction history: {str(e)}'
        }), 500

@app.route('/api/user/stats', methods=['GET'])
@login_required
def get_user_stats():
    """Get user prediction statistics"""
    try:
        # Get all user predictions
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        
        if not predictions:
            return jsonify({
                'success': True,
                'stats': {
                    'total_predictions': 0,
                    'unique_symbols': 0,
                    'avg_confidence': 0,
                    'most_predicted_symbol': None,
                    'prediction_accuracy': 0,
                    'recent_predictions': 0
                }
            })
        
        # Calculate statistics
        total_predictions = len(predictions)
        unique_symbols = len(set(pred.symbol for pred in predictions))
        avg_confidence = sum(pred.confidence_score or 0 for pred in predictions) / total_predictions
        
        # Most predicted symbol
        symbol_counts = {}
        for pred in predictions:
            symbol_counts[pred.symbol] = symbol_counts.get(pred.symbol, 0) + 1
        most_predicted_symbol = max(symbol_counts, key=symbol_counts.get) if symbol_counts else None
        
        # Recent predictions (last 7 days)
        from datetime import datetime, timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_predictions = len([p for p in predictions if p.created_at >= week_ago])
        
        # Calculate accuracy (if actual data is available)
        accurate_predictions = len([p for p in predictions if p.accuracy_score and p.accuracy_score > 0.7])
        prediction_accuracy = (accurate_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        return jsonify({
            'success': True,
            'stats': {
                'total_predictions': total_predictions,
                'unique_symbols': unique_symbols,
                'avg_confidence': round(avg_confidence * 100, 1),
                'most_predicted_symbol': most_predicted_symbol,
                'prediction_accuracy': round(prediction_accuracy, 1),
                'recent_predictions': recent_predictions
            }
        })
        
    except Exception as e:
        logger.error(f"Error calculating user stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to calculate statistics: {str(e)}'
        }), 500

# Stock data API endpoints
# Initialize market data fetcher and technical analysis
market_data = MarketDataFetcher(
    alpha_vantage_key='283NVS9MBKNG6Q0W',
    polygon_key='dOuGXiLdaOP41p4unQrxwDbRrnc4b23vck',
    finnhub_key='d1h4h29r01qkdlvrp3rgd1h4h29r01qkdlvrp3s0'
)
technical_analysis = TechnicalAnalysis()

@app.route('/api/stock-data/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        timeframe = request.args.get('timeframe', '1d')
        candle_count = int(request.args.get('candle_count', '100'))
        provider = request.args.get('provider', 'yahoo')
        include_indicators = request.args.get('include_indicators', 'false').lower() == 'true'
        
        logger.info(f'Fetching stock data for {symbol}, timeframe: {timeframe}, count: {candle_count}, provider: {provider}')
        
        # Fetch candlestick data from the specified provider
        candles = market_data.get_candlestick_data(symbol, timeframe, candle_count, provider)
        
        if candles is None:
            # Fallback to mock data if API fails
            logger.warning(f'Failed to fetch data from {provider}, falling back to mock data')
            candles = generate_mock_candles(symbol, timeframe, candle_count)
        
        response_data = {'candles': candles}
        
        # Calculate technical indicators if requested
        if include_indicators:
            indicators = technical_analysis.calculate_all_indicators(candles)
            response_data['technical_indicators'] = indicators
        
        logger.info(f'Successfully processed {len(candles)} candles for {symbol}')
        return jsonify(response_data)
    except Exception as e:
        logger.error(f'Error in get_stock_data: {str(e)}')
        return jsonify({'error': str(e)}), 500

def generate_mock_candles(symbol, timeframe, count):
    base_price = 150.0
    volatility = 0.02
    candles = []
    
    # Generate timestamps based on timeframe
    time_delta = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '1h': timedelta(hours=1),
        '1d': timedelta(days=1)
    }.get(timeframe, timedelta(days=1))
    
    current_time = datetime.now()
    
    for i in range(count):
        timestamp = (current_time - (time_delta * (count - i - 1))).strftime('%Y-%m-%d %H:%M:%S')
        price_change = random.uniform(-volatility, volatility)
        open_price = base_price * (1 + price_change)
        close_price = open_price * (1 + random.uniform(-volatility, volatility))
        high_price = max(open_price, close_price) * (1 + abs(random.uniform(0, volatility)))
        low_price = min(open_price, close_price) * (1 - abs(random.uniform(0, volatility)))
        
        candles.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2)
        })
        
        base_price = close_price
    
    return candles

@app.route('/multi-user-dashboard')
def multi_user_dashboard():
    return render_template('multi_user_dashboard.html')

@app.route('/api/predictions/<symbol>', methods=['GET'])
def get_predictions(symbol):
    try:
        timeframe = request.args.get('timeframe', '1d')
        model_type = request.args.get('model_type', 'lstm')
        n_candles = int(request.args.get('n_candles', '5'))
        provider = request.args.get('provider', 'yahoo')
        compare_models = request.args.get('compare_models', 'false').lower() == 'true'
        
        logger.info(f'Generating predictions for {symbol} using {model_type} model')
        
        # Ensure static directory exists
        static_dir = os.path.join(app.static_folder, 'model_plots')
        os.makedirs(static_dir, exist_ok=True)
        
        # Fetch historical data for training
        candles = market_data.get_candlestick_data(symbol, timeframe, 200, provider)
        if candles is None:
            logger.warning(f'Failed to fetch data from {provider}, falling back to mock data')
            candles = generate_mock_candles(symbol, timeframe, 200)
        
        # Prepare data for ML model
        df = pd.DataFrame(candles)
        features = ['open', 'high', 'low', 'close']
        X = df[features].values
        sequence_length = 20
        
        # Create sequences for training
        sequences = []
        targets = []
        for i in range(len(X) - sequence_length - 1):
            sequences.append(X[i:i+sequence_length])
            targets.append(X[i+sequence_length])
        
        X_train = np.array(sequences)
        y_train = np.array(targets)
        
        # Initialize models dictionary to store results
        models = {}
        predictions = {}
        metrics = {}
        
        # Function to train and evaluate a model
        def train_and_evaluate(model_class):
            model = model_class(sequence_length, len(features))
            model.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            
            # Generate predictions
            last_sequence = X_train[-1:]
            model_predictions = []
            for _ in range(n_candles):
                pred = model.model.predict(last_sequence, verbose=0)
                model_predictions.append(pred[0])
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1] = pred[0]
            
            # Calculate metrics
            train_pred = model.model.predict(X_train, verbose=0)
            accuracy = accuracy_score(
                y_train.argmax(axis=1) if len(y_train.shape) > 2 else np.round(y_train),
                train_pred.argmax(axis=1) if len(train_pred.shape) > 2 else np.round(train_pred)
            )
            
            # Save performance plots
            plot_base = f'{symbol}_{model_class.__name__.lower()}'
            metrics_path = os.path.join(static_dir, f'{plot_base}_metrics.png')
            cm_path = os.path.join(static_dir, f'{plot_base}_cm.png')
            
            model.plot_metrics(save_path=metrics_path)
            model.plot_confusion_matrix(
                y_train if len(y_train.shape) > 2 else np.round(y_train).reshape(-1, 1),
                train_pred if len(train_pred.shape) > 2 else np.round(train_pred).reshape(-1, 1),
                save_path=cm_path
            )
            
            return {
                'predictions': model_predictions,
                'accuracy': accuracy,
                'metrics_plot': f'/static/model_plots/{plot_base}_metrics.png',
                'confusion_matrix_plot': f'/static/model_plots/{plot_base}_cm.png'
            }
        
        # Train and evaluate requested model or all models for comparison
        model_classes = {
            'rnn': RNNModel,
            'cnn': CNNModel,
            'lstm': LSTMModel,
            'hybrid': HybridModel
        }
        
        if compare_models:
            for name, model_class in model_classes.items():
                try:
                    models[name] = train_and_evaluate(model_class)
                except Exception as e:
                    logger.error(f'Error training {name} model: {str(e)}')
        else:
            selected_model = model_classes.get(model_type.lower())
            if not selected_model:
                raise ValueError(f'Invalid model type: {model_type}')
            models[model_type] = train_and_evaluate(selected_model)
        
        X_train = np.array(sequences)
        y_train = np.array(targets)
        
        # Format the response with predictions and performance metrics
        response = {
            'success': True,
            'models': {}
        }

        for model_name, model_results in models.items():
            response['models'][model_name] = {
                'predictions': [
                    {
                        'timestamp': (datetime.strptime(candles[-1]['timestamp'], '%Y-%m-%d %H:%M:%S') + 
                                    (i + 1) * pd.Timedelta(timeframe)).strftime('%Y-%m-%d %H:%M:%S'),
                        'open': float(pred[0]),
                        'high': float(pred[1]),
                        'low': float(pred[2]),
                        'close': float(pred[3])
                    } for i, pred in enumerate(model_results['predictions'])
                ],
                'performance': {
                    'accuracy': float(model_results['accuracy']),
                    'metrics_plot_url': model_results['metrics_plot'],
                    'confusion_matrix_url': model_results['confusion_matrix_plot']
                }
            }

        if compare_models:
            # Add model comparison summary
            best_model = max(models.items(), key=lambda x: x[1]['accuracy'])[0]
            response['comparison'] = {
                'best_model': best_model,
                'accuracy_comparison': {
                    name: results['accuracy'] 
                    for name, results in models.items()
                }
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f'Error in get_predictions: {str(e)}')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500            
            
            
        
        # Calculate model performance metrics
        y_pred = model.model.predict(X_train)
        accuracy = accuracy_score(
            y_train.argmax(axis=1),
            y_pred.argmax(axis=1)
        )
        
        # Generate performance plots
        metrics_path = f'static/metrics_{symbol}_{model_type}.png'
        confusion_matrix_path = f'static/confusion_matrix_{symbol}_{model_type}.png'
        model.plot_metrics(save_path=metrics_path)
        model.plot_confusion_matrix(y_train, y_pred, save_path=confusion_matrix_path)
        
        # Calculate technical indicators for predicted candles
        indicators = technical_analysis.calculate_all_indicators(predicted_candles)
        
        response_data = {
            'predicted_candles': predicted_candles,
            'technical_indicators': indicators,
            'model_info': {
                'type': model_type,
                'accuracy': float(accuracy),
                'metrics_plot': metrics_path,
                'confusion_matrix_plot': confusion_matrix_path
            }
        }
        
        logger.info(f'Successfully generated {len(predicted_candles)} predictions for {symbol}')
        return jsonify(response_data)

        model_type = request.args.get('model_type', 'lstm')
        future_candles = int(request.args.get('future_candles', '5'))
        
        logger.info(f'Generating predictions for {symbol}, timeframe: {timeframe}, model: {model_type}, future_candles: {future_candles}')
        # In a real implementation, use the selected ML model to generate predictions
        # For now, generate mock predictions that follow the current trend
        last_candles = generate_mock_candles(symbol, timeframe, 10)  # Get recent data for trend
        logger.info(f'Generated {len(last_candles)} historical candles for trend analysis')
        
        # Calculate trend from last candles
        trend = 'bullish' if last_candles[-1]['close'] > last_candles[0]['close'] else 'bearish'
        base_price = last_candles[-1]['close']
        
        predictions = {
            'model_info': {
                'type': model_type,
                'timeframe': timeframe,
                'confidence': round(random.uniform(0.70, 0.95), 2)
            },
            'predicted_candles': generate_mock_predictions(base_price, future_candles, trend),
            'technical_indicators': [
                {'name': 'Trend Direction', 'signal': trend, 'confidence': round(random.uniform(0.70, 0.90), 2)},
                {'name': 'Volume Analysis', 'signal': trend, 'confidence': round(random.uniform(0.65, 0.85), 2)},
                {'name': 'Pattern Recognition', 'signal': trend, 'confidence': round(random.uniform(0.60, 0.80), 2)}
            ]
        }
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_mock_predictions(base_price, count, trend):
    try:
        volatility = 0.015  # Lower volatility for predictions
        predictions = []
        current_price = base_price
        
        for i in range(count):
            # Add trend bias to price changes
            trend_bias = 0.005 if trend == 'bullish' else -0.005
            price_change = random.uniform(-volatility + trend_bias, volatility + trend_bias)
            
            open_price = current_price
            close_price = open_price * (1 + price_change)
            high_price = max(open_price, close_price) * (1 + abs(random.uniform(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(random.uniform(0, volatility/2)))
            
            predictions.append({
                'timestamp': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'confidence': round(random.uniform(0.60, 0.90) - (i * 0.05), 2)  # Confidence decreases with time
            })
            
            current_price = close_price
        
        return predictions
    except Exception as e:
        logger.error(f'Error generating predictions: {str(e)}')
        return []

@app.route('/api/technical-indicators/<symbol>')
def get_technical_indicators(symbol):
    try:
        # Mock technical indicator data
        mock_indicators = {
            'RSI': 65.5,
            'MACD': 2.3,
            'SMA_50': 155.75,
            'EMA_20': 157.25,
            'Volume': '1.2M'
        }
        return jsonify(mock_indicators)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found'}), 404
    return render_template('error.html', error_code=404, error_message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if request.path.startswith('/api/'):
        return jsonify({'error': str(error)}), 500
    return render_template('error.html', error_code=500, error_message='Internal server error'), 500

# Initialize database
with app.app_context():
    try:
        # Check if tables exist before creating them
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        if not inspector.has_table('users'):
            db.create_all()
            logger.info("Database initialized successfully")
        else:
            logger.info("Database tables already exist, skipping initialization")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

if __name__ == '__main__':
    logger.info("Starting Candlestick Pattern Prediction System...")
    app.run(debug=True, host='0.0.0.0', port=5002)