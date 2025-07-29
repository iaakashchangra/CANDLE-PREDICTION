import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
import json
from abc import ABC, abstractmethod
from config import Config
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
import yfinance as yf
import os

class BaseDataProvider(ABC):
    """Abstract base class for data providers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rate_limiter = RateLimiter(config.get('rate_limit', 60))
        
    @abstractmethod
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        pass
    
    @abstractmethod
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """Fetch real-time data for a symbol"""
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported - now more flexible"""
        # First check if it's in our predefined list
        if symbol.upper() in Config.MARKET_CONFIG['supported_companies']:
            return True
        
        # If not in predefined list, try to validate with Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            # Try to get basic info to validate symbol exists
            info = ticker.info
            return 'symbol' in info or 'shortName' in info or 'longName' in info
        except Exception as e:
            self.logger.warning(f"Symbol validation failed for {symbol}: {str(e)}")
            return False
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported"""
        return timeframe in Config.MARKET_CONFIG['supported_timeframes']

class RateLimiter:
    """Rate limiter to control API request frequency"""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request is more than 1 minute old
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            self.requests.append(now)

class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider"""
    
    def __init__(self):
        super().__init__(Config.API_CONFIG['yahoo_finance'])
        
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        try:
            self.rate_limiter.wait_if_needed()
            
            if not self.validate_symbol(symbol):
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            if not self.validate_timeframe(timeframe):
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Convert timeframe to Yahoo Finance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1M': '1mo'
            }
            
            yf_interval = interval_map.get(timeframe, '1h')
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No data found for {symbol} in the specified date range")
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Reset index to make datetime a column
            data.reset_index(inplace=True)
            data.rename(columns={'datetime': 'timestamp'}, inplace=True)
            
            # Add metadata
            data['symbol'] = symbol
            data['timeframe'] = timeframe
            data['provider'] = 'yahoo_finance'
            
            self.logger.info(f"Fetched {len(data)} records for {symbol} from Yahoo Finance")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
            return pd.DataFrame()
    
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """Fetch real-time data from Yahoo Finance"""
        try:
            self.rate_limiter.wait_if_needed()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price data
            current_data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now().isoformat(),
                'provider': 'yahoo_finance'
            }
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data from Yahoo Finance: {str(e)}")
            return {}

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage data provider"""
    
    def __init__(self):
        super().__init__(Config.API_CONFIG['alpha_vantage'])
        self.api_key = self.config['api_key']
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Alpha Vantage"""
        try:
            self.rate_limiter.wait_if_needed()
            
            if not self.validate_symbol(symbol):
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            # Map timeframes to Alpha Vantage functions
            function_map = {
                '1m': 'TIME_SERIES_INTRADAY',
                '5m': 'TIME_SERIES_INTRADAY',
                '15m': 'TIME_SERIES_INTRADAY',
                '30m': 'TIME_SERIES_INTRADAY',
                '1h': 'TIME_SERIES_INTRADAY',
                '1d': 'TIME_SERIES_DAILY',
                '1w': 'TIME_SERIES_WEEKLY',
                '1M': 'TIME_SERIES_MONTHLY'
            }
            
            interval_map = {
                '1m': '1min', '5m': '5min', '15m': '15min',
                '30m': '30min', '1h': '60min'
            }
            
            function = function_map.get(timeframe, 'TIME_SERIES_DAILY')
            
            # Build API parameters
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            # Add interval for intraday data
            if function == 'TIME_SERIES_INTRADAY':
                params['interval'] = interval_map.get(timeframe, '60min')
            
            # Make API request
            response = requests.get(
                self.config['base_url'],
                params=params,
                timeout=self.config['timeout']
            )
            
            response.raise_for_status()
            data_json = response.json()
            
            # Check for API errors
            if 'Error Message' in data_json:
                raise ValueError(f"Alpha Vantage API Error: {data_json['Error Message']}")
            
            if 'Note' in data_json:
                raise ValueError(f"Alpha Vantage API Note: {data_json['Note']}")
            
            # Extract time series data
            time_series_key = None
            for key in data_json.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                raise ValueError("No time series data found in response")
            
            time_series = data_json[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                row = {
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values.get('1. open', 0)),
                    'high': float(values.get('2. high', 0)),
                    'low': float(values.get('3. low', 0)),
                    'close': float(values.get('4. close', 0)),
                    'volume': int(values.get('5. volume', 0))
                }
                df_data.append(row)
            
            data = pd.DataFrame(df_data)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Sort by timestamp
            data.sort_values('timestamp', inplace=True)
            data.reset_index(drop=True, inplace=True)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data = data[(data['timestamp'] >= start_dt) & (data['timestamp'] <= end_dt)]
            
            # Add metadata
            data['symbol'] = symbol
            data['timeframe'] = timeframe
            data['provider'] = 'alpha_vantage'
            
            self.logger.info(f"Fetched {len(data)} records for {symbol} from Alpha Vantage")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Alpha Vantage: {str(e)}")
            return pd.DataFrame()
    
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """Fetch real-time data from Alpha Vantage"""
        try:
            self.rate_limiter.wait_if_needed()
            
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(
                self.config['base_url'],
                params=params,
                timeout=self.config['timeout']
            )
            
            response.raise_for_status()
            data_json = response.json()
            
            if 'Global Quote' not in data_json:
                raise ValueError("No global quote data found")
            
            quote = data_json['Global Quote']
            
            current_data = {
                'symbol': symbol,
                'current_price': float(quote.get('05. price', 0)),
                'previous_close': float(quote.get('08. previous close', 0)),
                'open': float(quote.get('02. open', 0)),
                'day_high': float(quote.get('03. high', 0)),
                'day_low': float(quote.get('04. low', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%'),
                'timestamp': datetime.now().isoformat(),
                'provider': 'alpha_vantage'
            }
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data from Alpha Vantage: {str(e)}")
            return {}

class PolygonProvider(BaseDataProvider):
    """Polygon.io data provider"""
    
    def __init__(self):
        super().__init__(Config.API_CONFIG['polygon'])
        self.api_key = self.config['api_key']
        
        if not self.api_key:
            raise ValueError("Polygon API key not configured")
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Polygon.io"""
        try:
            self.rate_limiter.wait_if_needed()
            
            if not self.validate_symbol(symbol):
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            # Map timeframes to Polygon format
            timespan_map = {
                '1m': ('1', 'minute'),
                '5m': ('5', 'minute'),
                '15m': ('15', 'minute'),
                '30m': ('30', 'minute'),
                '1h': ('1', 'hour'),
                '4h': ('4', 'hour'),
                '1d': ('1', 'day'),
                '1w': ('1', 'week'),
                '1M': ('1', 'month')
            }
            
            if timeframe not in timespan_map:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            multiplier, timespan = timespan_map[timeframe]
            
            # Format dates for Polygon API
            start_date_formatted = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_date_formatted = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            
            # Build API URL
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date_formatted}/{end_date_formatted}"
            
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=self.config['timeout'])
            response.raise_for_status()
            
            data_json = response.json()
            
            if data_json.get('status') != 'OK':
                raise ValueError(f"Polygon API Error: {data_json.get('error', 'Unknown error')}")
            
            if 'results' not in data_json or not data_json['results']:
                self.logger.warning(f"No data returned from Polygon for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Create DataFrame from response
            results = data_json['results']
            df = pd.DataFrame(results)
            
            # Rename columns to match expected format
            df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp'
            }, inplace=True)
            
            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Polygon: {str(e)}")
            return pd.DataFrame()
    
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """Fetch real-time data from Polygon.io"""
        try:
            self.rate_limiter.wait_if_needed()
            
            if not self.validate_symbol(symbol):
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            # Construct API URL for quote
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            
            params = {
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=self.config['timeout'])
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'OK' or 'ticker' not in data:
                raise ValueError(f"Polygon API Error: {data.get('error', 'Unknown error')}")
            
            ticker_data = data['ticker']
            day_data = ticker_data.get('day', {})
            prev_day_data = ticker_data.get('prevDay', {})
            
            # Format the response
            current_data = {
                'symbol': symbol,
                'current_price': ticker_data.get('lastTrade', {}).get('p', 0),
                'previous_close': prev_day_data.get('c', 0),
                'open': day_data.get('o', 0),
                'day_high': day_data.get('h', 0),
                'day_low': day_data.get('l', 0),
                'volume': day_data.get('v', 0),
                'change': day_data.get('c', 0) - prev_day_data.get('c', 0) if prev_day_data.get('c', 0) else 0,
                'change_percent': ((day_data.get('c', 0) - prev_day_data.get('c', 0)) / prev_day_data.get('c', 1) * 100) if prev_day_data.get('c', 0) else 0,
                'timestamp': datetime.now().isoformat(),
                'provider': 'polygon'
            }
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data from Polygon: {str(e)}")
            return {}

class FinnhubProvider(BaseDataProvider):
    """Finnhub data provider"""
    
    def __init__(self):
        super().__init__(Config.API_CONFIG['finnhub'])
        self.api_key = self.config['api_key']
        self.base_url = self.config['base_url']
        
        if not self.api_key:
            raise ValueError("Finnhub API key not configured")
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Finnhub"""
        try:
            self.rate_limiter.wait_if_needed()
            
            if not self.validate_symbol(symbol):
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            if not self.validate_timeframe(timeframe):
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Map timeframes to Finnhub resolution format
            resolution_map = {
                '1m': '1',
                '5m': '5',
                '15m': '15',
                '30m': '30',
                '1h': '60',
                '4h': '240',
                '1d': 'D',
                '1w': 'W'
            }
            
            resolution = resolution_map.get(timeframe, 'D')
            
            # Convert dates to Unix timestamps (seconds)
            start_timestamp = int(pd.to_datetime(start_date).timestamp())
            end_timestamp = int(pd.to_datetime(end_date).timestamp())
            
            # Construct API URL
            url = f"{self.base_url}/stock/candle"
            
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': start_timestamp,
                'to': end_timestamp,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=self.config['timeout'])
            response.raise_for_status()
            data = response.json()
            
            # Check if the API returned an error or no data
            if data.get('s') == 'no_data' or 'c' not in data:
                self.logger.warning(f"No data returned from Finnhub for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Create DataFrame from response
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Finnhub: {str(e)}")
            return pd.DataFrame()
    
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """Fetch real-time data from Finnhub"""
        try:
            self.rate_limiter.wait_if_needed()
            
            if not self.validate_symbol(symbol):
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            # Construct API URL for quote
            url = f"{self.base_url}/quote"
            
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=self.config['timeout'])
            response.raise_for_status()
            data = response.json()
            
            # Format the response
            current_data = {
                'symbol': symbol,
                'current_price': data.get('c', 0),
                'previous_close': data.get('pc', 0),
                'open': data.get('o', 0),
                'day_high': data.get('h', 0),
                'day_low': data.get('l', 0),
                'change': data.get('c', 0) - data.get('pc', 0),
                'change_percent': ((data.get('c', 0) - data.get('pc', 0)) / data.get('pc', 1) * 100) if data.get('pc', 0) else 0,
                'timestamp': datetime.now().isoformat(),
                'provider': 'finnhub'
            }
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data from Finnhub: {str(e)}")
            return {}

class DataCollector:
    """Main data collector that manages multiple providers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.providers = {
            'yahoo_finance': YahooFinanceProvider(),
            'alpha_vantage': None,
            'polygon': None,
            'finnhub': None
        }
        
        # Initialize Alpha Vantage if API key is available
        try:
            if Config.API_CONFIG['alpha_vantage']['api_key']:
                self.providers['alpha_vantage'] = AlphaVantageProvider()
        except Exception as e:
            self.logger.warning(f"Alpha Vantage provider not available: {str(e)}")
        
        # Initialize Polygon if API key is available
        try:
            if Config.API_CONFIG['polygon']['api_key']:
                self.providers['polygon'] = PolygonProvider()
        except Exception as e:
            self.logger.warning(f"Polygon provider not available: {str(e)}")
            
        # Initialize Finnhub if API key is available
        try:
            if Config.API_CONFIG['finnhub']['api_key']:
                self.providers['finnhub'] = FinnhubProvider()
        except Exception as e:
            self.logger.warning(f"Finnhub provider not available: {str(e)}")
        
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_available_providers(self) -> List[str]:
        """Get list of available data providers"""
        return [name for name, provider in self.providers.items() if provider is not None]
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str, 
                   provider: str = 'yahoo_finance') -> pd.DataFrame:
        """Fetch historical data using specified provider"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Unknown provider: {provider}")
            
            if self.providers[provider] is None:
                raise ValueError(f"Provider {provider} is not available (check API key configuration)")
            
            # Check cache first
            cache_key = f"{provider}_{symbol}_{timeframe}_{start_date}_{end_date}"
            if cache_key in self.cache:
                cache_data, cache_time = self.cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    self.logger.info(f"Returning cached data for {cache_key}")
                    return cache_data
            
            # Fetch data from provider
            data = self.providers[provider].fetch_data(symbol, timeframe, start_date, end_date)
            
            # Cache the result
            if not data.empty:
                self.cache[cache_key] = (data.copy(), time.time())
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_realtime_data(self, symbol: str, provider: str = 'yahoo_finance') -> Dict:
        """Fetch real-time data using specified provider"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Unknown provider: {provider}")
            
            if self.providers[provider] is None:
                raise ValueError(f"Provider {provider} is not available")
            
            return self.providers[provider].fetch_realtime_data(symbol)
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data: {str(e)}")
            return {}
    
    def fetch_multiple_symbols(self, symbols: List[str], timeframe: str, 
                              start_date: str, end_date: str, 
                              provider: str = 'yahoo_finance') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(
                    self.fetch_data, symbol, timeframe, start_date, end_date, provider
                ): symbol for symbol in symbols
            }
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    data = future.result(timeout=60)
                    results[symbol] = data
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    results[symbol] = pd.DataFrame()
        
        return results
    
    def fetch_with_fallback(self, symbol: str, timeframe: str, start_date: str, 
                           end_date: str, preferred_providers: List[str] = None) -> Tuple[pd.DataFrame, str]:
        """Fetch data with fallback to other providers if primary fails"""
        if preferred_providers is None:
            preferred_providers = ['yahoo_finance', 'alpha_vantage', 'polygon']
        
        for provider in preferred_providers:
            if provider in self.providers and self.providers[provider] is not None:
                try:
                    data = self.fetch_data(symbol, timeframe, start_date, end_date, provider)
                    if not data.empty:
                        self.logger.info(f"Successfully fetched data using {provider}")
                        return data, provider
                except Exception as e:
                    self.logger.warning(f"Provider {provider} failed: {str(e)}")
                    continue
        
        self.logger.error("All providers failed to fetch data")
        return pd.DataFrame(), 'none'
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate fetched data quality"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        if data.empty:
            validation_result['is_valid'] = False
            validation_result['issues'].append('No data available')
            return validation_result
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f'Missing columns: {missing_columns}')
        
        # Check for missing values
        missing_values = data[required_columns].isnull().sum()
        if missing_values.any():
            validation_result['issues'].append(f'Missing values found: {missing_values.to_dict()}')
        
        # Check for invalid OHLC relationships
        if 'high' in data.columns and 'low' in data.columns:
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                validation_result['issues'].append(f'Invalid high < low in {invalid_hl} records')
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = (data[col] < 0).sum()
                if negative_prices > 0:
                    validation_result['issues'].append(f'Negative prices in {col}: {negative_prices} records')
        
        # Calculate statistics
        if not data.empty:
            validation_result['statistics'] = {
                'total_records': len(data),
                'date_range': {
                    'start': data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                    'end': data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
                },
                'price_range': {
                    'min': float(data['low'].min()) if 'low' in data.columns else None,
                    'max': float(data['high'].max()) if 'high' in data.columns else None
                },
                'volume_stats': {
                    'mean': float(data['volume'].mean()) if 'volume' in data.columns else None,
                    'total': int(data['volume'].sum()) if 'volume' in data.columns else None
                }
            }
        
        if validation_result['issues']:
            validation_result['is_valid'] = False
        
        return validation_result
    
    def get_data_summary(self, symbol: str, timeframe: str, start_date: str, 
                        end_date: str, provider: str = 'yahoo_finance') -> Dict:
        """Get summary information about available data"""
        try:
            data = self.fetch_data(symbol, timeframe, start_date, end_date, provider)
            validation = self.validate_data(data)
            
            summary = {
                'symbol': symbol,
                'timeframe': timeframe,
                'provider': provider,
                'data_available': not data.empty,
                'validation': validation
            }
            
            if not data.empty:
                summary.update({
                    'first_timestamp': data['timestamp'].min().isoformat(),
                    'last_timestamp': data['timestamp'].max().isoformat(),
                    'total_records': len(data),
                    'completeness': len(data) / self._calculate_expected_records(start_date, end_date, timeframe) * 100
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_expected_records(self, start_date: str, end_date: str, timeframe: str) -> int:
        """Calculate expected number of records for a date range and timeframe"""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            total_seconds = (end_dt - start_dt).total_seconds()
            timeframe_seconds = Config.MARKET_CONFIG['supported_timeframes'][timeframe]['seconds']
            
            # Rough estimate (doesn't account for market hours/holidays)
            expected_records = int(total_seconds / timeframe_seconds)
            
            # Adjust for market hours if daily or intraday
            if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
                # Assume 6.5 hours of trading per day
                market_hours_ratio = 6.5 / 24
                expected_records = int(expected_records * market_hours_ratio)
            
            return max(1, expected_records)
            
        except Exception:
            return 1
    
    def get_historical_data(self, symbol: str, timeframe: str, period: str = '1y', 
                           provider: str = 'yahoo_finance') -> pd.DataFrame:
        """Get historical data - wrapper for fetch_data method"""
        try:
            # Convert period to start_date and end_date
            end_date = datetime.now()
            
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '1w':
                start_date = end_date - timedelta(weeks=1)
            elif period == '1m':
                start_date = end_date - timedelta(days=30)
            elif period == '3m':
                start_date = end_date - timedelta(days=90)
            elif period == '6m':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=730)
            elif period == '5y':
                start_date = end_date - timedelta(days=1825)
            else:
                # Default to 1 year if period not recognized
                start_date = end_date - timedelta(days=365)
            
            # Format dates as strings
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Use existing fetch_data method
            return self.fetch_data(symbol, timeframe, start_date_str, end_date_str, provider)
            
        except Exception as e:
            self.logger.error(f"Error in get_historical_data: {str(e)}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        cache_info = {
            'total_entries': len(self.cache),
            'cache_timeout': self.cache_timeout,
            'entries': []
        }
        
        current_time = time.time()
        for key, (data, cache_time) in self.cache.items():
            age = current_time - cache_time
            cache_info['entries'].append({
                'key': key,
                'age_seconds': age,
                'records': len(data),
                'expired': age > self.cache_timeout
            })
        
        return cache_info