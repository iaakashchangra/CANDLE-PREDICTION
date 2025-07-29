#!/usr/bin/env python3
"""
Test script to diagnose AAPL data fetching issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.data_collection.data_collector import DataCollector
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_aapl_data():
    """Test AAPL data fetching with different providers"""
    
    # Initialize data collector
    data_collector = DataCollector()
    
    # Test parameters
    symbol = 'AAPL'
    timeframe = '1d'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Testing AAPL data fetching from {start_date_str} to {end_date_str}")
    print("=" * 60)
    
    # Test each provider individually
    providers = ['yahoo_finance', 'alpha_vantage']
    
    for provider in providers:
        print(f"\nTesting {provider}...")
        try:
            data = data_collector.fetch_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date_str,
                end_date=end_date_str,
                provider=provider
            )
            
            if data.empty:
                print(f"❌ {provider}: No data returned")
            else:
                print(f"✅ {provider}: Successfully fetched {len(data)} records")
                print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                print(f"   Sample data:")
                print(data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(3))
                
        except Exception as e:
            print(f"❌ {provider}: Error - {str(e)}")
    
    # Test fallback mechanism
    print(f"\n\nTesting fallback mechanism...")
    print("=" * 40)
    
    try:
        data, used_provider = data_collector.fetch_with_fallback(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date_str,
            end_date=end_date_str,
            preferred_providers=['yahoo_finance', 'alpha_vantage']
        )
        
        if data.empty:
            print(f"❌ Fallback: No data returned from any provider")
        else:
            print(f"✅ Fallback: Successfully fetched {len(data)} records using {used_provider}")
            print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            
    except Exception as e:
        print(f"❌ Fallback: Error - {str(e)}")

if __name__ == '__main__':
    test_aapl_data()