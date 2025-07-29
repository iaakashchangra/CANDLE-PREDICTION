import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import talib
from scipy import signal
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')
from config import Config

class DataPreprocessor:
    """Data preprocessing pipeline for candlestick data"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or Config.DATA_CONFIG
        
        # Initialize scalers
        self.scalers = {}
        self.feature_columns = []
        self.is_fitted = False
        
        # Technical indicators configuration
        self.ta_config = self.config.get('technical_indicators', {})
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the input data"""
        try:
            if data.empty:
                self.logger.warning("Empty dataframe provided for cleaning")
                return data
            
            cleaned_data = data.copy()
            
            # Remove duplicates
            initial_count = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates(subset=['timestamp'], keep='last')
            duplicates_removed = initial_count - len(cleaned_data)
            
            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate records")
            
            # Sort by timestamp
            cleaned_data = cleaned_data.sort_values('timestamp').reset_index(drop=True)
            
            # Validate OHLC relationships
            cleaned_data = self._fix_ohlc_relationships(cleaned_data)
            
            # Handle missing values
            cleaned_data = self._handle_missing_values(cleaned_data)
            
            # Remove outliers
            cleaned_data = self._remove_outliers(cleaned_data)
            
            # Validate volume
            cleaned_data = self._validate_volume(cleaned_data)
            
            self.logger.info(f"Data cleaning completed. Records: {len(cleaned_data)}")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            return data
    
    def _fix_ohlc_relationships(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix invalid OHLC relationships"""
        try:
            # Ensure high >= max(open, close) and high >= low
            data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
            data['high'] = np.maximum(data['high'], data['low'])
            
            # Ensure low <= min(open, close) and low <= high
            data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
            data['low'] = np.minimum(data['low'], data['high'])
            
            # Count fixed relationships
            invalid_count = len(data) - len(data[
                (data['high'] >= data['low']) & 
                (data['high'] >= data['open']) & 
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) & 
                (data['low'] <= data['close'])
            ])
            
            if invalid_count > 0:
                self.logger.info(f"Fixed {invalid_count} invalid OHLC relationships")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fixing OHLC relationships: {str(e)}")
            return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Check for missing values
            missing_counts = data.isnull().sum()
            
            if missing_counts.sum() == 0:
                return data
            
            self.logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Forward fill for price data (carry last observation forward)
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill')
            
            # For volume, use median imputation
            if 'volume' in data.columns:
                volume_median = data['volume'].median()
                data['volume'] = data['volume'].fillna(volume_median)
            
            # Drop rows that still have missing values in critical columns
            critical_columns = ['open', 'high', 'low', 'close']
            before_drop = len(data)
            data = data.dropna(subset=critical_columns)
            after_drop = len(data)
            
            if before_drop != after_drop:
                self.logger.info(f"Dropped {before_drop - after_drop} rows with missing critical values")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            return data
    
    def _remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the dataset"""
        try:
            if method == 'iqr':
                return self._remove_outliers_iqr(data)
            elif method == 'zscore':
                return self._remove_outliers_zscore(data)
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"Error removing outliers: {str(e)}")
            return data
    
    def _remove_outliers_iqr(self, data: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        try:
            price_columns = ['open', 'high', 'low', 'close']
            initial_count = len(data)
            
            for col in price_columns:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    # Remove extreme outliers only
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            
            outliers_removed = initial_count - len(data)
            if outliers_removed > 0:
                self.logger.info(f"Removed {outliers_removed} outlier records using IQR method")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in IQR outlier removal: {str(e)}")
            return data
    
    def _remove_outliers_zscore(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        try:
            price_columns = ['open', 'high', 'low', 'close']
            initial_count = len(data)
            
            for col in price_columns:
                if col in data.columns:
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    data = data[z_scores < threshold]
            
            outliers_removed = initial_count - len(data)
            if outliers_removed > 0:
                self.logger.info(f"Removed {outliers_removed} outlier records using Z-score method")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in Z-score outlier removal: {str(e)}")
            return data
    
    def _validate_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean volume data"""
        try:
            if 'volume' not in data.columns:
                return data
            
            # Remove negative volumes
            negative_volumes = (data['volume'] < 0).sum()
            if negative_volumes > 0:
                data = data[data['volume'] >= 0]
                self.logger.info(f"Removed {negative_volumes} records with negative volume")
            
            # Handle zero volumes (replace with median)
            zero_volumes = (data['volume'] == 0).sum()
            if zero_volumes > 0:
                volume_median = data[data['volume'] > 0]['volume'].median()
                data.loc[data['volume'] == 0, 'volume'] = volume_median
                self.logger.info(f"Replaced {zero_volumes} zero volume records with median")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error validating volume: {str(e)}")
            return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        try:
            if data.empty:
                return data
            
            enhanced_data = data.copy()
            
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in enhanced_data.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for technical indicators: {missing_columns}")
                return enhanced_data
            
            # Convert to numpy arrays for TA-Lib
            open_prices = enhanced_data['open'].values.astype(float)
            high_prices = enhanced_data['high'].values.astype(float)
            low_prices = enhanced_data['low'].values.astype(float)
            close_prices = enhanced_data['close'].values.astype(float)
            volumes = enhanced_data['volume'].values.astype(float)
            
            # Simple Moving Averages
            if 'sma' in self.ta_config:
                for period in self.ta_config['sma']:
                    enhanced_data[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
            
            # Exponential Moving Averages
            if 'ema' in self.ta_config:
                for period in self.ta_config['ema']:
                    enhanced_data[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
            
            # RSI
            if 'rsi' in self.ta_config:
                rsi_period = self.ta_config['rsi']
                enhanced_data['rsi'] = talib.RSI(close_prices, timeperiod=rsi_period)
            
            # MACD
            if 'macd' in self.ta_config:
                macd_config = self.ta_config['macd']
                macd, macd_signal, macd_hist = talib.MACD(
                    close_prices,
                    fastperiod=macd_config['fast'],
                    slowperiod=macd_config['slow'],
                    signalperiod=macd_config['signal']
                )
                enhanced_data['macd'] = macd
                enhanced_data['macd_signal'] = macd_signal
                enhanced_data['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            if 'bollinger_bands' in self.ta_config:
                bb_config = self.ta_config['bollinger_bands']
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close_prices,
                    timeperiod=bb_config['period'],
                    nbdevup=bb_config['std'],
                    nbdevdn=bb_config['std']
                )
                enhanced_data['bb_upper'] = bb_upper
                enhanced_data['bb_middle'] = bb_middle
                enhanced_data['bb_lower'] = bb_lower
                enhanced_data['bb_width'] = (bb_upper - bb_lower) / bb_middle
                enhanced_data['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic Oscillator
            if 'stochastic' in self.ta_config:
                stoch_config = self.ta_config['stochastic']
                slowk, slowd = talib.STOCH(
                    high_prices, low_prices, close_prices,
                    fastk_period=stoch_config['k_period'],
                    slowk_period=stoch_config['d_period'],
                    slowd_period=stoch_config['d_period']
                )
                enhanced_data['stoch_k'] = slowk
                enhanced_data['stoch_d'] = slowd
            
            # Additional indicators
            enhanced_data['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            enhanced_data['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            enhanced_data['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            enhanced_data['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Volume indicators
            enhanced_data['obv'] = talib.OBV(close_prices, volumes)
            enhanced_data['ad'] = talib.AD(high_prices, low_prices, close_prices, volumes)
            
            # Price-based features
            enhanced_data['price_change'] = enhanced_data['close'].pct_change()
            enhanced_data['high_low_ratio'] = enhanced_data['high'] / enhanced_data['low']
            enhanced_data['close_open_ratio'] = enhanced_data['close'] / enhanced_data['open']
            
            # Volatility measures
            enhanced_data['volatility_5'] = enhanced_data['price_change'].rolling(5).std()
            enhanced_data['volatility_20'] = enhanced_data['price_change'].rolling(20).std()
            
            # Volume features
            enhanced_data['volume_sma_10'] = enhanced_data['volume'].rolling(10).mean()
            enhanced_data['volume_ratio'] = enhanced_data['volume'] / enhanced_data['volume_sma_10']
            
            self.logger.info(f"Added technical indicators. Total features: {len(enhanced_data.columns)}")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    def apply_noise_reduction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply noise reduction techniques"""
        try:
            if data.empty:
                return data
            
            denoised_data = data.copy()
            noise_config = self.config.get('noise_reduction', {})
            
            # EWMA smoothing
            if 'ewma_alpha' in noise_config:
                alpha = noise_config['ewma_alpha']
                price_columns = ['open', 'high', 'low', 'close']
                
                for col in price_columns:
                    if col in denoised_data.columns:
                        denoised_data[f'{col}_ewma'] = denoised_data[col].ewm(alpha=alpha).mean()
            
            # Savitzky-Golay filter
            if 'savgol_window' in noise_config and 'savgol_polyorder' in noise_config:
                window = noise_config['savgol_window']
                polyorder = noise_config['savgol_polyorder']
                
                if len(denoised_data) > window:
                    price_columns = ['open', 'high', 'low', 'close']
                    
                    for col in price_columns:
                        if col in denoised_data.columns:
                            denoised_data[f'{col}_savgol'] = savgol_filter(
                                denoised_data[col].values, window, polyorder
                            )
            
            self.logger.info("Applied noise reduction techniques")
            
            return denoised_data
            
        except Exception as e:
            self.logger.error(f"Error applying noise reduction: {str(e)}")
            return data
    
    def normalize_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using configured method"""
        try:
            if data.empty:
                return data
            
            normalized_data = data.copy()
            norm_config = self.config.get('normalization', {})
            method = norm_config.get('method', 'minmax')
            
            # Select numeric columns for normalization
            numeric_columns = normalized_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude timestamp and categorical columns
            exclude_columns = ['timestamp', 'symbol', 'timeframe', 'provider']
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            if not feature_columns:
                self.logger.warning("No numeric features found for normalization")
                return normalized_data
            
            if fit or not self.is_fitted:
                # Initialize scalers
                if method == 'minmax':
                    feature_range = norm_config.get('feature_range', (0, 1))
                    self.scalers['features'] = MinMaxScaler(feature_range=feature_range)
                elif method == 'standard':
                    self.scalers['features'] = StandardScaler()
                elif method == 'robust':
                    self.scalers['features'] = RobustScaler()
                else:
                    self.logger.warning(f"Unknown normalization method: {method}")
                    return normalized_data
                
                # Fit the scaler
                self.scalers['features'].fit(normalized_data[feature_columns])
                self.feature_columns = feature_columns
                self.is_fitted = True
                
                self.logger.info(f"Fitted {method} scaler on {len(feature_columns)} features")
            
            # Transform the data
            if 'features' in self.scalers and self.feature_columns:
                normalized_data[self.feature_columns] = self.scalers['features'].transform(
                    normalized_data[self.feature_columns]
                )
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
            return data
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = None, 
                        prediction_horizon: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling"""
        try:
            if data.empty:
                return np.array([]), np.array([])
            
            seq_length = sequence_length or self.config.get('sequence_length', 60)
            pred_horizon = prediction_horizon or self.config.get('prediction_horizon', 1)
            
            # Select feature columns (exclude metadata)
            exclude_columns = ['timestamp', 'symbol', 'timeframe', 'provider']
            feature_columns = [col for col in data.columns if col not in exclude_columns]
            
            if not feature_columns:
                self.logger.error("No feature columns found for sequence creation")
                return np.array([]), np.array([])
            
            # Convert to numpy array
            feature_data = data[feature_columns].values
            
            # Create sequences
            X, y = [], []
            
            for i in range(seq_length, len(feature_data) - pred_horizon + 1):
                # Input sequence
                X.append(feature_data[i - seq_length:i])
                
                # Target (next n values of OHLC)
                target_start = i
                target_end = i + pred_horizon
                
                # Assuming OHLC are the first 4 columns
                if len(feature_columns) >= 4:
                    y.append(feature_data[target_start:target_end, :4])  # OHLC only
                else:
                    y.append(feature_data[target_start:target_end])
            
            X = np.array(X)
            y = np.array(y)
            
            # Reshape y if prediction_horizon is 1
            if pred_horizon == 1:
                y = y.squeeze(axis=1)
            
            self.logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            return np.array([]), np.array([])
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = None, val_ratio: float = None, 
                   test_ratio: float = None) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets"""
        try:
            if len(X) == 0 or len(y) == 0:
                return tuple([np.array([])] * 6)
            
            split_config = self.config.get('train_test_split', {})
            train_ratio = train_ratio or split_config.get('train_ratio', 0.7)
            val_ratio = val_ratio or split_config.get('validation_ratio', 0.15)
            test_ratio = test_ratio or split_config.get('test_ratio', 0.15)
            
            # Ensure ratios sum to 1
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                self.logger.warning(f"Ratios don't sum to 1.0 ({total_ratio}), normalizing...")
                train_ratio /= total_ratio
                val_ratio /= total_ratio
                test_ratio /= total_ratio
            
            n_samples = len(X)
            
            # Calculate split indices
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            # Split the data
            X_train = X[:train_end]
            y_train = y[:train_end]
            
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            
            X_test = X[val_end:]
            y_test = y[val_end:]
            
            self.logger.info(
                f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            return tuple([np.array([])] * 6)
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform normalized predictions back to original scale"""
        try:
            if not self.is_fitted or 'features' not in self.scalers:
                self.logger.warning("Scaler not fitted, cannot inverse transform")
                return predictions
            
            # Create a dummy array with the same number of features
            n_features = len(self.feature_columns)
            dummy_data = np.zeros((len(predictions), n_features))
            
            # Assuming predictions are OHLC (first 4 features)
            if predictions.shape[1] <= 4:
                dummy_data[:, :predictions.shape[1]] = predictions
            else:
                dummy_data = predictions
            
            # Inverse transform
            inverse_transformed = self.scalers['features'].inverse_transform(dummy_data)
            
            # Return only the OHLC part
            return inverse_transformed[:, :predictions.shape[1]]
            
        except Exception as e:
            self.logger.error(f"Error inverse transforming predictions: {str(e)}")
            return predictions
    
    def preprocess_pipeline(self, data: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline"""
        try:
            self.logger.info("Starting preprocessing pipeline")
            
            # Step 1: Clean data
            cleaned_data = self.clean_data(data)
            
            if cleaned_data.empty:
                self.logger.error("No data remaining after cleaning")
                return np.array([]), np.array([])
            
            # Step 2: Add technical indicators
            enhanced_data = self.add_technical_indicators(cleaned_data)
            
            # Step 3: Apply noise reduction
            denoised_data = self.apply_noise_reduction(enhanced_data)
            
            # Step 4: Normalize features
            normalized_data = self.normalize_features(denoised_data, fit=fit)
            
            # Step 5: Create sequences
            X, y = self.create_sequences(normalized_data)
            
            if len(X) == 0:
                self.logger.error("No sequences created")
                return np.array([]), np.array([])
            
            self.logger.info(f"Preprocessing completed. Final shapes: X {X.shape}, y {y.shape}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return np.array([]), np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after preprocessing"""
        return self.feature_columns.copy() if self.feature_columns else []
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing configuration and state"""
        return {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_columns),
            'feature_names': self.feature_columns[:10] if self.feature_columns else [],  # First 10
            'scalers': list(self.scalers.keys()),
            'technical_indicators': list(self.ta_config.keys()) if self.ta_config else []
        }
    
    def save_preprocessor(self, filepath: str) -> bool:
        """Save the preprocessor state"""
        try:
            import joblib
            
            preprocessor_state = {
                'config': self.config,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'is_fitted': self.is_fitted,
                'ta_config': self.ta_config
            }
            
            joblib.dump(preprocessor_state, filepath)
            self.logger.info(f"Preprocessor saved to {filepath}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")
            return False
    
    def load_preprocessor(self, filepath: str) -> bool:
        """Load the preprocessor state"""
        try:
            import joblib
            
            preprocessor_state = joblib.load(filepath)
            
            self.config = preprocessor_state.get('config', self.config)
            self.scalers = preprocessor_state.get('scalers', {})
            self.feature_columns = preprocessor_state.get('feature_columns', [])
            self.is_fitted = preprocessor_state.get('is_fitted', False)
            self.ta_config = preprocessor_state.get('ta_config', {})
            
            self.logger.info(f"Preprocessor loaded from {filepath}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {str(e)}")
            return False
    
    def prepare_training_data(self, data: pd.DataFrame, sequence_length: int = 60, 
                                prediction_horizon: int = 1, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for ML models with proper error handling"""
        try:
            self.logger.info(f"Preparing training data with {len(data)} records")
            
            # Validate minimum data requirements
            min_required = sequence_length + prediction_horizon + 10  # Buffer for splitting
            if len(data) < min_required:
                self.logger.error(f"Insufficient data for training: {len(data)} records, minimum required: {min_required}")
                # Return empty arrays instead of raising exception
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Preprocess the data
            X, y = self.preprocess_pipeline(data, fit=True)
            
            if len(X) == 0 or len(y) == 0:
                self.logger.error("Preprocessing pipeline failed to create sequences")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Split data into train and validation sets
            split_idx = int(len(X) * (1 - test_size))
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            
            self.logger.info(f"Training data prepared - Train: {len(X_train)}, Val: {len(X_val)}")
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def prepare_data_for_n_candle_prediction(self, data: pd.DataFrame, n_candles: int = 5, 
                                           sequence_length: int = 60) -> Tuple[np.ndarray, bool]:
        """Prepare data specifically for n-candle prediction with enhanced error handling"""
        try:
            self.logger.info(f"Preparing data for {n_candles} candle prediction")
            
            # Validate input parameters
            if n_candles < 1 or n_candles > 50:
                self.logger.error(f"Invalid n_candles value: {n_candles}. Must be between 1 and 50")
                return np.array([]), False
            
            # Check minimum data requirements
            min_required = sequence_length + 20  # Buffer for indicators
            if len(data) < min_required:
                self.logger.warning(f"Limited data for prediction: {len(data)} records, recommended: {min_required}")
                # Adjust sequence length for limited data
                sequence_length = max(10, len(data) - 10)
                self.logger.info(f"Adjusted sequence length to {sequence_length}")
            
            # Preprocess data
            processed_data = self.clean_data(data)
            if processed_data.empty:
                self.logger.error("Data cleaning resulted in empty dataset")
                return np.array([]), False
            
            # Add technical indicators
            enhanced_data = self.add_technical_indicators(processed_data)
            
            # Apply noise reduction
            denoised_data = self.apply_noise_reduction(enhanced_data)
            
            # Normalize features
            normalized_data = self.normalize_features(denoised_data, fit=False)
            
            # Prepare the most recent sequence for prediction
            exclude_columns = ['timestamp', 'symbol', 'timeframe', 'provider']
            feature_columns = [col for col in normalized_data.columns if col not in exclude_columns]
            
            if len(normalized_data) >= sequence_length:
                # Use the most recent sequence_length records
                recent_sequence = normalized_data[feature_columns].tail(sequence_length).values
                recent_sequence = recent_sequence.reshape(1, sequence_length, -1)
                
                self.logger.info(f"Prepared sequence shape: {recent_sequence.shape} for {n_candles} candle prediction")
                return recent_sequence, True
            else:
                self.logger.error(f"Insufficient data for sequence creation: {len(normalized_data)} < {sequence_length}")
                return np.array([]), False
                
        except Exception as e:
            self.logger.error(f"Error preparing data for n-candle prediction: {str(e)}")
            return np.array([]), False
    
    def validate_preprocessing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data before preprocessing with enhanced checks"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'recommendations': [],
            'data_quality_score': 0.0
        }
        
        try:
            if data.empty:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Empty dataset provided")
                return validation_result
            
            quality_score = 100.0  # Start with perfect score
            
            # Check minimum data requirements
            min_records = self.config.get('sequence_length', 60) + 50  # Buffer for indicators
            if len(data) < min_records:
                if len(data) < 20:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(
                        f"Critically insufficient data: {len(data)} records, minimum required: 20"
                    )
                    quality_score -= 50
                else:
                    validation_result['recommendations'].append(
                        f"Limited data: {len(data)} records, recommended: {min_records}"
                    )
                    quality_score -= 20
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            optional_columns = ['volume', 'timestamp']
            
            missing_required = [col for col in required_columns if col not in data.columns]
            missing_optional = [col for col in optional_columns if col not in data.columns]
            
            if missing_required:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Missing critical columns: {missing_required}")
                quality_score -= 40
            
            if missing_optional:
                validation_result['recommendations'].append(f"Missing optional columns: {missing_optional}")
                quality_score -= 10
            
            # Check data quality
            if not data.empty and not missing_required:
                # Check for missing values
                missing_pct = data[required_columns].isnull().sum() / len(data) * 100
                high_missing = missing_pct[missing_pct > 10]
                
                if not high_missing.empty:
                    validation_result['recommendations'].append(
                        f"High missing values in columns: {high_missing.to_dict()}"
                    )
                    quality_score -= 15
                
                # Check OHLC relationships
                invalid_ohlc = (
                    (data['high'] < data['low']) |
                    (data['high'] < data['open']) |
                    (data['high'] < data['close']) |
                    (data['low'] > data['open']) |
                    (data['low'] > data['close'])
                ).sum()
                
                if invalid_ohlc > 0:
                    invalid_pct = (invalid_ohlc / len(data)) * 100
                    if invalid_pct > 5:
                        validation_result['issues'].append(
                            f"High percentage of invalid OHLC relationships: {invalid_pct:.1f}%"
                        )
                        quality_score -= 20
                    else:
                        validation_result['recommendations'].append(
                            f"Some invalid OHLC relationships found: {invalid_ohlc} records ({invalid_pct:.1f}%)"
                        )
                        quality_score -= 5
                
                # Check timestamp ordering
                if 'timestamp' in data.columns:
                    if not data['timestamp'].is_monotonic_increasing:
                        validation_result['recommendations'].append(
                            "Timestamps are not in ascending order - will be sorted"
                        )
                        quality_score -= 5
                
                # Check for extreme outliers
                for col in ['open', 'high', 'low', 'close']:
                    if col in data.columns:
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((data[col] < (Q1 - 3 * IQR)) | (data[col] > (Q3 + 3 * IQR))).sum()
                        
                        if outliers > len(data) * 0.05:  # More than 5% outliers
                            validation_result['recommendations'].append(
                                f"High number of outliers in {col}: {outliers} ({outliers/len(data)*100:.1f}%)"
                            )
                            quality_score -= 5
            
            # Ensure quality score is between 0 and 100
            validation_result['data_quality_score'] = max(0.0, min(100.0, quality_score))
            
            # Final validation based on quality score
            if validation_result['data_quality_score'] < 30:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Data quality too low: {validation_result['data_quality_score']:.1f}/100")
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            validation_result['data_quality_score'] = 0.0
            return validation_result