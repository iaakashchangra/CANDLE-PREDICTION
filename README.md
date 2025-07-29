# Candlestick Pattern Prediction System Using Machine Learning

🔧 **Project Overview**

This is a multi-user stock market prediction system that uses machine learning to forecast the next n candlestick patterns (bullish/bearish) for different users. The system integrates with Yahoo Finance, Alpha Vantage, and Polygon.io APIs to fetch real-time and historical candlestick data.

## 🚀 Features

- **Multi-User Support**: Each user can configure their own prediction settings
- **Multiple Data Sources**: Yahoo Finance, Alpha Vantage, Polygon.io APIs
- **Flexible Timeframes**: 1min, 5min, 1h, 1d, 1w, 1mo
- **ML Models**: LSTM, CNN, RNN, and Hybrid models
- **Real-time Charts**: Interactive candlestick charts with technical indicators
- **Performance Metrics**: Accuracy, MAE, RMSE, MAPE tracking
- **Export Functionality**: CSV/PDF export of predictions

## 📁 Project Structure

```
NEW_CANDLES/
├── backend/
│   ├── app.py                 # Flask application entry point
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Python dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py      # LSTM implementation
│   │   ├── cnn_model.py       # CNN implementation
│   │   ├── rnn_model.py       # RNN implementation
│   │   └── hybrid_model.py    # Hybrid model combining all
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_collector.py  # API data collection
│   │   ├── preprocessor.py    # Data preprocessing
│   │   └── storage.py         # Database operations
│   ├── api/
│   │   ├── __init__.py
│   │   ├── yahoo_finance.py   # Yahoo Finance API
│   │   ├── alpha_vantage.py   # Alpha Vantage API
│   │   └── polygon_api.py     # Polygon.io API
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Model training pipeline
│   │   ├── predictor.py       # Prediction engine
│   │   └── evaluator.py       # Performance evaluation
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── user_manager.py    # User authentication
│   │   └── session_manager.py # Session management
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py         # Utility functions
│       └── validators.py      # Input validation
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── Auth/
│   │   │   ├── Dashboard/
│   │   │   ├── Charts/
│   │   │   ├── UserConfig/
│   │   │   └── Performance/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── package-lock.json
├── database/
│   ├── init.sql              # Database schema
│   └── migrations/
├── docs/
│   ├── API.md                # API documentation
│   ├── MODELS.md             # ML models documentation
│   └── DEPLOYMENT.md         # Deployment guide
└── tests/
    ├── backend/
    └── frontend/
```

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask (Python)
- **Database**: SQLite/PostgreSQL
- **ML Libraries**: TensorFlow/Keras, scikit-learn, pandas, numpy
- **APIs**: yfinance, alpha_vantage, polygon-api-client
- **Authentication**: Flask-Login, JWT

### Frontend
- **Framework**: React.js
- **Charts**: Chart.js, TradingView Lightweight Charts
- **UI Library**: Material-UI
- **State Management**: Redux/Context API
- **HTTP Client**: Axios

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 📊 ML Models

### 1. LSTM (Long Short-Term Memory)
- Best for sequential time series data
- Handles long-term dependencies
- Good for trend prediction

### 2. CNN (Convolutional Neural Network)
- Excellent for pattern recognition
- Identifies local features in candlestick patterns
- Fast training and inference

### 3. RNN (Recurrent Neural Network)
- Basic sequential model
- Good baseline for comparison
- Lightweight and fast

### 4. Hybrid Model
- Combines LSTM, CNN, and RNN
- Ensemble approach for better accuracy
- Weighted predictions from all models

## 🔧 Configuration

### API Keys Setup
Create a `.env` file in the backend directory:
```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
POLYGON_API_KEY=your_polygon_key
SECRET_KEY=your_flask_secret_key
DATABASE_URL=sqlite:///candles.db
```

### User Configuration Options
- **Data Provider**: Yahoo Finance, Alpha Vantage, Polygon.io
- **Company**: Any valid stock symbol (MSFT, AAPL, TSLA, etc.)
- **Timeframe**: 1min, 5min, 15min, 30min, 1h, 4h, 1d, 1w, 1mo
- **Prediction Count**: 1-50 future candlesticks
- **ML Model**: LSTM, CNN, RNN, or Hybrid

## 📈 Performance Metrics

- **Accuracy**: Percentage of correct directional predictions
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable predictions

## 🔄 Data Flow

1. **User Configuration**: User selects API, symbol, timeframe, and model
2. **Data Collection**: System fetches historical data from chosen API
3. **Preprocessing**: Data cleaning, normalization, and feature engineering
4. **Model Training**: Train selected ML model on historical data
5. **Prediction**: Generate next n candlestick predictions
6. **Visualization**: Display results on interactive charts
7. **Evaluation**: Calculate and display performance metrics

## 📱 Web Dashboard Features

### Authentication
- User registration and login
- Session management
- Password reset functionality

### User Management
- Multi-user support
- Individual user configurations
- User-specific model performance tracking

### Chart Viewer
- Interactive candlestick charts
- Zoom and pan functionality
- Technical indicators overlay (MACD, RSI, EMA)
- Predicted vs actual comparison

### Model Performance
- Real-time metrics dashboard
- Model comparison tables
- Performance history tracking
- Best model recommendations

### Export Features
- CSV export of predictions
- PDF reports generation
- Historical data download

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

## 📚 Documentation

- [API Documentation](docs/API.md)
- [ML Models Guide](docs/MODELS.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. Stock market predictions are inherently risky, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.