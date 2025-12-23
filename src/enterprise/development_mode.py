"""
Development mode helper for testing without live exchanges
"""
import ccxt
import pandas as pd
from typing import Dict, List, Optional

class DevelopmentMode:
    """Simulate exchanges for development when live APIs aren't available"""
    
    @staticmethod
    def get_simulated_exchange(name: str = 'gemini'):
        """Get a simulated exchange for testing"""
        exchange = ccxt.test()  # CCXT's built-in test exchange
        
        # Configure with realistic parameters
        exchange.enableRateLimit = True
        exchange.options['adjustForTimeDifference'] = True
        
        # Mock some data
        if hasattr(exchange, 'markets'):
            exchange.markets = {
                'BTC/USD': {'symbol': 'BTC/USD', 'type': 'spot', 'active': True},
                'ETH/USD': {'symbol': 'ETH/USD', 'type': 'spot', 'active': True},
            }
        
        return exchange
    
    @staticmethod
    def generate_mock_ohlcv(symbol: str = 'BTC/USD', limit: int = 100):
        """Generate mock OHLCV data for testing"""
        import numpy as np
        from datetime import datetime, timedelta
        
        base_price = 50000.0
        volatility = 0.02
        
        data = []
        current_time = datetime.now()
        
        for i in range(limit):
            time_offset = timedelta(minutes=i)
            timestamp = current_time - time_offset
            
            # Generate random walk prices
            if i == 0:
                price = base_price
            else:
                price = data[i-1][4] * (1 + np.random.normal(0, volatility))
            
            open_price = price * (1 + np.random.normal(0, volatility/10))
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, volatility/20)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, volatility/20)))
            close_price = price
            volume = np.random.uniform(10, 100)
            
            data.append([
                int(timestamp.timestamp() * 1000),
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ])
        
        return data
    
    @staticmethod
    def get_test_config():
        """Get a test configuration that won't hit real APIs"""
        return {
            'exchanges': {
                'gemini': {'enabled': False},
                'coinbase': {'enabled': False},
                'binance': {'enabled': False},
                'test': {'enabled': True}
            },
            'trading': {
                'default_symbol': 'BTC/USD',
                'risk_per_trade': 0.01,
                'paper_trading': True
            },
            'development': {
                'use_mock_data': True,
                'simulate_latency': True,
                'mock_fee_rate': 0.001
            }
        }