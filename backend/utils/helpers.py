from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values."""
    if not isinstance(old_value, (int, float, Decimal)) or not isinstance(new_value, (int, float, Decimal)):
        return 0.0
    
    if old_value == 0:
        return 100.0 if new_value > 0 else -100.0 if new_value < 0 else 0.0
    
    return ((new_value - old_value) / abs(old_value)) * 100.0

def format_currency(amount, currency='USD'):
    """Format a number as currency."""
    if not isinstance(amount, (int, float, Decimal)):
        return '0.00'
    
    amount = Decimal(str(amount))
    formatted = amount.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    
    if currency == 'USD':
        return f'${formatted:,.2f}'
    elif currency == 'EUR':
        return f'€{formatted:,.2f}'
    else:
        return f'{formatted:,.2f} {currency}'

def format_date(date_str):
    """Convert date string to datetime object."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset."""
    # Calculate moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def identify_candlestick_patterns(df):
    """Identify basic candlestick patterns."""
    df['Body'] = df['Close'] - df['Open']
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    patterns = []
    for i in range(len(df)):
        pattern = []
        
        # Doji
        if abs(df['Body'].iloc[i]) <= 0.1 * (df['High'].iloc[i] - df['Low'].iloc[i]):
            pattern.append('Doji')
        
        # Hammer
        if (df['Lower_Shadow'].iloc[i] > 2 * abs(df['Body'].iloc[i]) and
            df['Upper_Shadow'].iloc[i] <= abs(df['Body'].iloc[i])):
            pattern.append('Hammer')
        
        # Shooting Star
        if (df['Upper_Shadow'].iloc[i] > 2 * abs(df['Body'].iloc[i]) and
            df['Lower_Shadow'].iloc[i] <= abs(df['Body'].iloc[i])):
            pattern.append('Shooting Star')
        
        patterns.append(', '.join(pattern) if pattern else 'No Pattern')
    
    df['Patterns'] = patterns
    return df

def prepare_data_for_ml(df):
    """Prepare data for machine learning models."""
    # Calculate percentage changes
    df['Returns'] = df['Close'].pct_change()
    
    # Create features
    feature_columns = [
        'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Signal_Line',
        'Body', 'Upper_Shadow', 'Lower_Shadow', 'Returns'
    ]
    
    # Create target (1 if price goes up next day, 0 if down)
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df[feature_columns], df['Target']

def format_prediction_result(prediction, probability, pattern, indicators):
    """Format the prediction results for display."""
    return {
        'prediction': 'Up' if prediction == 1 else 'Down',
        'confidence': float(probability),
        'pattern': pattern,
        'technical_indicators': {
            'RSI': float(indicators['RSI']),
            'MACD': float(indicators['MACD']),
            'Signal_Line': float(indicators['Signal_Line'])
        }
    }