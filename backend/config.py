import os
from datetime import timedelta

class Config:
    """Application configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///candles.db'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
    POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY')
    
    # Data Collection Settings
    DEFAULT_DATA_PERIOD = '1y'
    MAX_DATA_POINTS = 10000
    CACHE_TIMEOUT = 300  # 5 minutes
    
    # Supported APIs
    SUPPORTED_APIS = ['yahoo', 'alpha_vantage', 'polygon']
    
    # Supported Timeframes
    SUPPORTED_TIMEFRAMES = {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1wk',
        '1mo': '1mo'
    }
    
    # ML Model Configuration
    MODEL_TYPES = ['lstm', 'cnn', 'rnn', 'hybrid']
    
    # LSTM Configuration
    LSTM_CONFIG = {
        'sequence_length': 60,
        'units': [50, 50, 50],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    # CNN Configuration
    CNN_CONFIG = {
        'sequence_length': 60,
        'filters': [32, 64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    # RNN Configuration
    RNN_CONFIG = {
        'sequence_length': 60,
        'units': [50, 50],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    # Hybrid Model Configuration
    HYBRID_CONFIG = {
        'sequence_length': 60,
        'lstm_units': [50, 50],
        'cnn_filters': [32, 64],
        'rnn_units': [50],
        'dropout': 0.2,
        'epochs': 150,
        'batch_size': 32,
        'validation_split': 0.2,
        'ensemble_weights': [0.4, 0.3, 0.3]  # LSTM, CNN, RNN weights
    }
    
    # Data Preprocessing Configuration
    PREPROCESSING_CONFIG = {
        'normalize': True,
        'scale_method': 'minmax',  # 'minmax', 'standard', 'robust'
        'remove_outliers': True,
        'outlier_method': 'iqr',  # 'iqr', 'zscore'
        'fill_missing': 'forward',  # 'forward', 'backward', 'interpolate'
        'smooth_data': True,
        'smoothing_method': 'ewma',  # 'ewma', 'sma', 'gaussian'
        'ewma_span': 10
    }
    
    # Technical Indicators Configuration
    INDICATORS_CONFIG = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'ema_periods': [9, 21, 50],
        'sma_periods': [20, 50, 200],
        'bollinger_period': 20,
        'bollinger_std': 2
    }
    
    # Prediction Configuration
    PREDICTION_CONFIG = {
        'min_prediction_count': 1,
        'max_prediction_count': 50,
        'confidence_threshold': 0.6,
        'signal_threshold': 0.05  # 5% change threshold for buy/sell signals
    }
    
    # Performance Evaluation Configuration
    EVALUATION_CONFIG = {
        'metrics': ['accuracy', 'mae', 'rmse', 'mape', 'sharpe_ratio'],
        'backtest_period': '6mo',
        'min_trades': 10,
        'transaction_cost': 0.001  # 0.1% transaction cost
    }
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'json'}
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Rate Limiting Configuration
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    RATELIMIT_DEFAULT = '100 per hour'
    
    # API Rate Limits (requests per minute)
    API_RATE_LIMITS = {
        'yahoo': 2000,  # No official limit, but be conservative
        'alpha_vantage': 5,  # 5 requests per minute for free tier
        'polygon': 5  # 5 requests per minute for free tier
    }
    
    # Cache Configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Email Configuration (for notifications)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Celery Configuration (for background tasks)
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Model Storage Configuration
    MODEL_STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    
    # Backup Configuration
    BACKUP_ENABLED = os.environ.get('BACKUP_ENABLED', 'False').lower() == 'true'
    BACKUP_INTERVAL = int(os.environ.get('BACKUP_INTERVAL', '24'))  # hours
    BACKUP_RETENTION = int(os.environ.get('BACKUP_RETENTION', '7'))  # days
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_STORAGE_PATH, exist_ok=True)
        
        # Validate required environment variables
        if not Config.ALPHA_VANTAGE_API_KEY:
            app.logger.warning('ALPHA_VANTAGE_API_KEY not set')
        
        if not Config.POLYGON_API_KEY:
            app.logger.warning('POLYGON_API_KEY not set')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])