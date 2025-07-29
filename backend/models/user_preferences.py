from backend.utils.database import db

class UserPreferences(db.Model):
    __tablename__ = 'user_preferences'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # UI Preferences
    theme = db.Column(db.String(20), default='light')
    notification_enabled = db.Column(db.Boolean, default=True)
    
    # Trading Preferences
    preferred_data_provider = db.Column(db.String(20))  # yahoo, alpha_vantage, polygon
    default_symbol = db.Column(db.String(10))  # Removed default='AAPL'
    default_timeframe = db.Column(db.String(10))  # Removed default='1d'
    default_prediction_count = db.Column(db.Integer, default=5)  # Number of future candlesticks
    preferred_model = db.Column(db.String(20))  # Removed default='lstm'
    
    # Risk Management
    prediction_threshold = db.Column(db.Float, default=0.7)
    risk_tolerance = db.Column(db.String(10), default='medium')  # low, medium, high
    
    # Advanced Settings
    auto_retrain = db.Column(db.Boolean, default=False)
    email_notifications = db.Column(db.Boolean, default=True)
    chart_interval = db.Column(db.String(10))  # Removed default='1d'
    
    # Timestamps
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    user = db.relationship('User', foreign_keys=[user_id])

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'theme': self.theme,
            'notification_enabled': self.notification_enabled,
            'preferred_data_provider': self.preferred_data_provider,
            'default_symbol': self.default_symbol,
            'default_timeframe': self.default_timeframe,
            'default_prediction_count': self.default_prediction_count,
            'preferred_model': self.preferred_model,
            'prediction_threshold': self.prediction_threshold,
            'risk_tolerance': self.risk_tolerance,
            'auto_retrain': self.auto_retrain,
            'email_notifications': self.email_notifications,
            'chart_interval': self.chart_interval,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @staticmethod
    def get_available_providers():
        """Get list of available data providers"""
        return [
            {'value': 'yahoo', 'name': 'Yahoo Finance', 'description': 'Free, reliable market data'},
            {'value': 'alpha_vantage', 'name': 'Alpha Vantage', 'description': 'Professional market data with API key'},
            {'value': 'polygon', 'name': 'Polygon.io', 'description': 'Real-time and historical market data'}
        ]
    
    @staticmethod
    def get_available_timeframes():
        """Get list of available timeframes"""
        return [
            {'value': '1m', 'name': '1 Minute'},
            {'value': '5m', 'name': '5 Minutes'},
            {'value': '15m', 'name': '15 Minutes'},
            {'value': '30m', 'name': '30 Minutes'},
            {'value': '1h', 'name': '1 Hour'},
            {'value': '4h', 'name': '4 Hours'},
            {'value': '1d', 'name': '1 Day'},
            {'value': '1w', 'name': '1 Week'},
            {'value': '1mo', 'name': '1 Month'}
        ]
    
    @staticmethod
    def get_available_models():
        """Get list of available prediction models"""
        return [
            {'value': 'lstm', 'name': 'LSTM', 'description': 'Long Short-Term Memory Neural Network'},
            {'value': 'cnn', 'name': 'CNN', 'description': 'Convolutional Neural Network'},
            {'value': 'rnn', 'name': 'RNN', 'description': 'Recurrent Neural Network'},
            {'value': 'hybrid', 'name': 'Hybrid', 'description': 'Combined Model Approach'}
        ]
    
    @staticmethod
    def get_available_companies():
        """Get list of available companies"""
        return [
            {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
            {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust'},
            {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust'}
        ]