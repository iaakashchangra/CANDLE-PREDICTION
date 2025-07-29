import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the Candlestick Pattern Prediction System"""
    
    # Base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    STATIC_DIR = os.path.join(BASE_DIR, 'frontend', 'static')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'frontend', 'templates')
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Database Configuration
    DATABASE_CONFIG = {
        'type': 'sqlite',  # sqlite, postgresql, mysql
        'sqlite': {
            'path': os.path.join(BASE_DIR, 'candlestick_prediction.db')
        },
        'postgresql': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', 5432),
            'database': os.getenv('DB_NAME', 'candlestick_db'),
            'username': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
    }
    
    # API Configuration
    API_CONFIG = {
        'yahoo_finance': {
            'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart/',
            'rate_limit': 2000,  # requests per hour
            'timeout': 30
        },
        'alpha_vantage': {
            'base_url': 'https://www.alphavantage.co/query',
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            'rate_limit': 5,  # requests per minute
            'timeout': 30
        },
        'polygon': {
            'base_url': 'https://api.polygon.io/v2/aggs/ticker/',
            'api_key': os.getenv('POLYGON_API_KEY', ''),
            'rate_limit': 5,  # requests per minute
            'timeout': 30
        },
        'finnhub': {
            'base_url': 'https://finnhub.io/api/v1',
            'api_key': os.getenv('FINNHUB_API_KEY', 'd1h4h29r01qkdlvrp3rgd1h4h29r01qkdlvrp3s0'),
            'rate_limit': 60,  # requests per minute
            'timeout': 30
        }
    }
    
    # API Date Range Configuration
    DATE_RANGE_CONFIG = {
        'historical_data': {
            'start_date': '2024-07-01',  # Default start date for historical data
            'end_date': '2025-07-31',   # Default end date for historical data
        },
        'prediction_config': {
            'separate_predictions': False,  # Default setting for separate predictions
            'max_prediction_count': 50,   # Maximum number of predictions allowed
        }
    }
    
    # Data Processing Configuration
    DATA_CONFIG = {
        'sequence_length': 60,  # Number of timesteps to look back
        'prediction_horizon': 1,  # Number of timesteps to predict ahead
        'features': ['open', 'high', 'low', 'close', 'volume'],
        'technical_indicators': {
            'sma': [5, 10, 20, 50],  # Simple Moving Averages
            'sma_periods': [5, 10, 20, 50],  # Simple Moving Averages for indicators
            'ema': [12, 26],  # Exponential Moving Averages
            'ema_periods': [12, 26],  # Exponential Moving Averages for indicators
            'rsi': 14,  # RSI period
            'rsi_period': 14,  # RSI period
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_bands': {'period': 20, 'std': 2},
            'bollinger_period': 20,
            'bollinger_std': 2,
            'stochastic': {'k_period': 14, 'd_period': 3}
        },
        'normalization': {
            'method': 'minmax',  # 'minmax', 'standard', 'robust'
            'feature_range': (0, 1)
        },
        'train_test_split': {
            'train_ratio': 0.7,
            'validation_ratio': 0.15,
            'test_ratio': 0.15
        },
        'noise_reduction': {
            'ewma_alpha': 0.3,
            'savgol_window': 5,
            'savgol_polyorder': 2
        },
        'smooth_data': True,
        'smoothing_method': 'sma',
        'sma_window': 5,
        'ewma_span': 5,
        'gaussian_sigma': 1.0
    }
    
    # LSTM Model Configuration
    LSTM_CONFIG = {
        'sequence_length': 60,
        'units': [128, 64, 32],  # LSTM units for each layer
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae', 'mape'],
        'early_stopping': {
            'patience': 20,
            'restore_best_weights': True
        },
        'reduce_lr': {
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-7
        }
    }
    
    # CNN Model Configuration
    CNN_CONFIG = {
        'sequence_length': 60,
        'filters': [64, 128, 64],  # Number of filters for each conv layer
        'kernel_size': 3,
        'pool_size': 2,
        'dropout': 0.2,
        'epochs': 80,
        'batch_size': 32,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae', 'mape'],
        'early_stopping': {
            'patience': 15,
            'restore_best_weights': True
        },
        'reduce_lr': {
            'factor': 0.5,
            'patience': 8,
            'min_lr': 1e-7
        }
    }
    
    # RNN Model Configuration
    RNN_CONFIG = {
        'sequence_length': 60,
        'units': [64, 32],  # RNN units for each layer
        'dropout': 0.3,
        'recurrent_dropout': 0.3,
        'epochs': 80,
        'batch_size': 32,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae', 'mape'],
        'early_stopping': {
            'patience': 15,
            'restore_best_weights': True
        },
        'reduce_lr': {
            'factor': 0.5,
            'patience': 8,
            'min_lr': 1e-7
        }
    }
    
    # Hybrid Model Configuration
    HYBRID_CONFIG = {
        'sequence_length': 60,
        'lstm_units': [64, 32],
        'cnn_filters': [32, 64, 32],
        'rnn_units': [32, 16],
        'dropout': 0.2,
        'epochs': 120,
        'batch_size': 32,
        'validation_split': 0.2,
        'learning_rate': 0.0005,
        'ensemble_weights': [0.4, 0.3, 0.3],  # LSTM, CNN, RNN weights
        'meta_learner': {
            'hidden_units': [64, 32],
            'dropout': 0.2
        },
        'training_method': 'ensemble',  # 'ensemble', 'sequential', 'weighted'
        'early_stopping': {
            'patience': 25,
            'restore_best_weights': True
        },
        'reduce_lr': {
            'factor': 0.5,
            'patience': 12,
            'min_lr': 1e-8
        }
    }
    
    # Model Manager Configuration
    MODEL_MANAGER_CONFIG = {
        'model_dir': MODELS_DIR,
        'auto_save': True,
        'parallel_training': True,
        'max_workers': 4,
        'model_retention_days': 30,
        'performance_tracking': True,
        'export_format': 'json'
    }
    
    # Web Application Configuration
    WEB_CONFIG = {
        'host': '0.0.0.0',
        'port': 5000,
        'debug': os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        'secret_key': os.getenv('SECRET_KEY', 'your-secret-key-change-this'),
        'session_timeout': 3600,  # 1 hour
        'max_content_length': 16 * 1024 * 1024,  # 16MB
        'upload_folder': os.path.join(BASE_DIR, 'uploads'),
        'allowed_extensions': {'csv', 'json', 'xlsx'}
    }
    
    # Authentication Configuration
    AUTH_CONFIG = {
        'password_min_length': 8,
        'password_require_uppercase': True,
        'password_require_lowercase': True,
        'password_require_numbers': True,
        'password_require_special': True,
        'session_timeout': 3600,
        'max_login_attempts': 5,
        'lockout_duration': 900,  # 15 minutes
        'jwt_secret': os.getenv('JWT_SECRET', 'jwt-secret-key-change-this'),
        'jwt_expiration': 86400  # 24 hours
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            }
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'detailed',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.path.join(LOGS_DIR, 'app.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'level': 'ERROR',
                'formatter': 'detailed',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.path.join(LOGS_DIR, 'error.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'error': {
                'handlers': ['error_file'],
                'level': 'ERROR',
                'propagate': False
            }
        }
    }
    
    # Chart Configuration
    CHART_CONFIG = {
        'default_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'],
        'max_candles_display': 1000,
        'default_indicators': ['SMA', 'EMA', 'RSI', 'MACD'],
        'chart_themes': ['light', 'dark'],
        'default_theme': 'dark',
        'candlestick_colors': {
            'bullish': '#26a69a',
            'bearish': '#ef5350'
        },
        'prediction_colors': {
            'lstm': '#2196f3',
            'cnn': '#ff9800',
            'rnn': '#9c27b0',
            'hybrid': '#4caf50'
        }
    }
    
    # Performance Monitoring
    MONITORING_CONFIG = {
        'enable_metrics': True,
        'metrics_retention_days': 30,
        'alert_thresholds': {
            'model_accuracy_drop': 0.1,  # 10% drop
            'prediction_latency': 5.0,  # 5 seconds
            'api_error_rate': 0.05,  # 5% error rate
            'memory_usage': 0.8  # 80% memory usage
        },
        'health_check_interval': 300,  # 5 minutes
        'performance_report_interval': 3600  # 1 hour
    }
    
    # Cache Configuration
    CACHE_CONFIG = {
        'type': 'redis',  # 'redis', 'memory', 'file'
        'redis': {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': os.getenv('REDIS_PORT', 6379),
            'db': 0,
            'password': os.getenv('REDIS_PASSWORD', None)
        },
        'default_timeout': 3600,  # 1 hour
        'key_prefix': 'candlestick_',
        'cache_predictions': True,
        'cache_market_data': True,
        'cache_model_results': True
    }
    
    # Export Configuration
    EXPORT_CONFIG = {
        'formats': ['csv', 'json', 'xlsx', 'pdf'],
        'max_export_records': 10000,
        'include_metadata': True,
        'compression': True,
        'export_timeout': 300,  # 5 minutes
        'temp_dir': os.path.join(BASE_DIR, 'temp')
    }
    
    # Supported Companies and Timeframes
    MARKET_CONFIG = {
        'supported_companies': {
            # Tech Giants
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'TSLA': 'Tesla Inc.',
            'NFLX': 'Netflix Inc.',
            'AMD': 'Advanced Micro Devices',
            'INTC': 'Intel Corporation',
            'CRM': 'Salesforce Inc.',
            'ORCL': 'Oracle Corporation',
            'ADBE': 'Adobe Inc.',
            'PYPL': 'PayPal Holdings',
            'UBER': 'Uber Technologies',
            'SPOT': 'Spotify Technology',
            'ZOOM': 'Zoom Video Communications',
            'SQ': 'Block Inc.',
            'SHOP': 'Shopify Inc.',
            # Financial Services
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp.',
            'WFC': 'Wells Fargo & Company',
            'GS': 'Goldman Sachs Group Inc.',
            'MS': 'Morgan Stanley',
            'V': 'Visa Inc.',
            'MA': 'Mastercard Inc.',
            # Healthcare
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'ABBV': 'AbbVie Inc.',
            'MRK': 'Merck & Co. Inc.',
            # Consumer & Retail
            'WMT': 'Walmart Inc.',
            'HD': 'Home Depot Inc.',
            'PG': 'Procter & Gamble Co.',
            'KO': 'Coca-Cola Company',
            'PEP': 'PepsiCo Inc.',
            'MCD': "McDonald's Corp.",
            'NKE': 'Nike Inc.',
            'SBUX': 'Starbucks Corp.',
            # Energy & Utilities
            'XOM': 'Exxon Mobil Corp.',
            'CVX': 'Chevron Corp.',
            # Industrial
            'BA': 'Boeing Company',
            'CAT': 'Caterpillar Inc.',
            'GE': 'General Electric Co.',
            # ETFs
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'IWM': 'iShares Russell 2000 ETF'
        },
        'supported_timeframes': {
            '1m': {'name': '1 Minute', 'seconds': 60},
            '5m': {'name': '5 Minutes', 'seconds': 300},
            '15m': {'name': '15 Minutes', 'seconds': 900},
            '30m': {'name': '30 Minutes', 'seconds': 1800},
            '1h': {'name': '1 Hour', 'seconds': 3600},
            '4h': {'name': '4 Hours', 'seconds': 14400},
            '1d': {'name': '1 Day', 'seconds': 86400},
            '1w': {'name': '1 Week', 'seconds': 604800},
            '1M': {'name': '1 Month', 'seconds': 2592000}
        },
        'default_company': 'AAPL',
        'default_timeframe': '1h',
        'max_historical_days': 365,
        'min_data_points': 100
    }
    
    # Feature Engineering Configuration
    FEATURE_CONFIG = {
        'price_features': {
            'returns': True,
            'log_returns': True,
            'price_ratios': True,
            'price_differences': True
        },
        'volume_features': {
            'volume_sma': [5, 10, 20],
            'volume_ratio': True,
            'volume_price_trend': True
        },
        'volatility_features': {
            'historical_volatility': [5, 10, 20],
            'garch_volatility': True,
            'parkinson_volatility': True
        },
        'pattern_features': {
            'candlestick_patterns': True,
            'support_resistance': True,
            'trend_lines': True,
            'fibonacci_levels': True
        },
        'time_features': {
            'hour_of_day': True,
            'day_of_week': True,
            'month_of_year': True,
            'quarter': True
        }
    }
    
    # Model Validation Configuration
    VALIDATION_CONFIG = {
        'cross_validation': {
            'method': 'time_series_split',
            'n_splits': 5,
            'test_size': 0.2
        },
        'walk_forward': {
            'window_size': 252,  # 1 year of trading days
            'step_size': 21,  # 1 month
            'min_train_size': 126  # 6 months
        },
        'metrics': {
            'regression': ['mse', 'mae', 'rmse', 'mape', 'r2'],
            'classification': ['accuracy', 'precision', 'recall', 'f1'],
            'trading': ['sharpe_ratio', 'max_drawdown', 'win_rate']
        },
        'significance_tests': {
            'diebold_mariano': True,
            'model_confidence_set': True
        }
    }
    
    # Risk Management Configuration
    RISK_CONFIG = {
        'position_sizing': {
            'method': 'kelly_criterion',  # 'fixed', 'percent_risk', 'kelly_criterion'
            'max_position_size': 0.1,  # 10% of portfolio
            'risk_per_trade': 0.02  # 2% risk per trade
        },
        'stop_loss': {
            'method': 'atr',  # 'fixed', 'percent', 'atr'
            'atr_multiplier': 2.0,
            'max_stop_loss': 0.05  # 5%
        },
        'take_profit': {
            'method': 'risk_reward',  # 'fixed', 'percent', 'risk_reward'
            'risk_reward_ratio': 2.0,
            'max_take_profit': 0.1  # 10%
        },
        'portfolio_limits': {
            'max_correlation': 0.7,
            'max_sector_exposure': 0.3,
            'max_single_position': 0.1
        }
    }
    
    @classmethod
    def get_config(cls, section: str = None) -> Dict[str, Any]:
        """Get configuration for a specific section or all configurations"""
        if section:
            section_upper = section.upper() + '_CONFIG'
            return getattr(cls, section_upper, {})
        
        # Return all configurations
        configs = {}
        for attr_name in dir(cls):
            if attr_name.endswith('_CONFIG') and not attr_name.startswith('_'):
                configs[attr_name.replace('_CONFIG', '').lower()] = getattr(cls, attr_name)
        
        return configs
    
    @classmethod
    def update_config(cls, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for a specific section"""
        try:
            section_upper = section.upper() + '_CONFIG'
            if hasattr(cls, section_upper):
                current_config = getattr(cls, section_upper)
                current_config.update(updates)
                setattr(cls, section_upper, current_config)
                return True
            return False
        except Exception:
            return False
    
    @classmethod
    def validate_config(cls) -> Dict[str, List[str]]:
        """Validate all configurations and return any issues"""
        issues = {}
        
        # Validate API keys
        api_issues = []
        if not cls.API_CONFIG['alpha_vantage']['api_key']:
            api_issues.append('Alpha Vantage API key not set')
        if not cls.API_CONFIG['polygon']['api_key']:
            api_issues.append('Polygon API key not set')
        
        if api_issues:
            issues['api'] = api_issues
        
        # Validate directories
        dir_issues = []
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            if not os.path.exists(directory):
                dir_issues.append(f'Directory does not exist: {directory}')
        
        if dir_issues:
            issues['directories'] = dir_issues
        
        # Validate model configurations
        model_issues = []
        for config_name in ['LSTM_CONFIG', 'CNN_CONFIG', 'RNN_CONFIG', 'HYBRID_CONFIG']:
            config = getattr(cls, config_name)
            if config['epochs'] <= 0:
                model_issues.append(f'{config_name}: epochs must be positive')
            if config['batch_size'] <= 0:
                model_issues.append(f'{config_name}: batch_size must be positive')
        
        if model_issues:
            issues['models'] = model_issues
        
        return issues
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get current environment information"""
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'base_dir': cls.BASE_DIR,
            'config_validation': cls.validate_config()
        }