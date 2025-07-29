#!/usr/bin/env python3
"""
Test script to check if all dependencies are working properly
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    try:
        logger.info("Testing basic imports...")
        import numpy as np
        import pandas as pd
        logger.info("✓ NumPy and Pandas imported successfully")
        
        import talib
        logger.info("✓ TA-Lib imported successfully")
        
        import sklearn
        logger.info("✓ Scikit-learn imported successfully")
        
        logger.info("Testing TensorFlow...")
        import tensorflow as tf
        logger.info(f"✓ TensorFlow {tf.__version__} imported successfully")
        
        logger.info("Testing Keras...")
        from tensorflow import keras
        logger.info("✓ Keras imported successfully")
        
        logger.info("Testing model imports...")
        from backend.models.lstm_model import LSTMModel
        logger.info("✓ LSTMModel imported successfully")
        
        from backend.models.cnn_model import CNNModel
        logger.info("✓ CNNModel imported successfully")
        
        from backend.models.rnn_model import RNNModel
        logger.info("✓ RNNModel imported successfully")
        
        from backend.models.hybrid_model import HybridModel
        logger.info("✓ HybridModel imported successfully")
        
        from backend.models.model_manager import ModelManager
        logger.info("✓ ModelManager imported successfully")
        
        # Test ModelManager initialization
        model_manager = ModelManager()
        logger.info("✓ ModelManager initialized successfully")
        
        logger.info("Testing preprocessor...")
        from backend.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        logger.info("✓ DataPreprocessor imported and initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_talib_functionality():
    """Test TA-Lib functionality with sample data"""
    try:
        logger.info("Testing TA-Lib functionality...")
        import numpy as np
        import talib
        
        # Create sample data
        close_prices = np.random.random(100).astype(np.float64) * 100 + 50
        
        # Test RSI calculation
        rsi = talib.RSI(close_prices, timeperiod=14)
        logger.info("✓ TA-Lib RSI calculation successful")
        
        # Test SMA calculation
        sma = talib.SMA(close_prices, timeperiod=20)
        logger.info("✓ TA-Lib SMA calculation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TA-Lib test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting dependency tests...")
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_talib_functionality():
        success = False
    
    if success:
        logger.info("🎉 All tests passed! Dependencies are working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)