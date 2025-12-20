"""
Enterprise Data Layer for Gemini/Coinbase integration
Handles all cryptocurrency market data and exchange communication
"""
import ccxt
import pandas as pd
import time
from typing import Dict, List, Optional
import logging

class DataLayer:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data layer with exchange connections
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize exchanges
        self.exchanges = {}
        self.initialize_exchanges()
        
        # Cache for market data
        self.market_cache = {}
        self.cache_ttl = 60  # seconds
        
    def initialize_exchanges(self):
        """Initialize connections to supported exchanges"""
        try:
            # Gemini
            if self.config.get('gemini_api_key'):
                self.exchanges['gemini'] = ccxt.gemini({
                    'apiKey': self.config.get('gemini_api_key'),
                    'secret': self.config.get('gemini_api_secret'),
                    'enableRateLimit': True,
                })
                self.logger.info("Gemini exchange initialized")
            
            # Coinbase Pro
            if self.config.get('coinbase_api_key'):
                self.exchanges['coinbase'] = ccxt.coinbasepro({
                    'apiKey': self.config.get('coinbase_api_key'),
                    'secret': self.config.get('coinbase_api_secret'),
                    'password': self.config.get('coinbase_api_passphrase', ''),
                    'enableRateLimit': True,
                })
                self.logger.info("Coinbase Pro exchange initialized")
            
            # Binance (for additional liquidity)
            if self.config.get('binance_api_key'):
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.config.get('binance_api_key'),
                    'secret': self.config.get('binance_api_secret'),
                    'enableRateLimit': True,
                })
                self.logger.info("Binance exchange initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
    
    def get_market_data(self, symbol: str, exchange: str = 'gemini', 
                       timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            exchange: Exchange name
            timeframe: Timeframe for candles
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{exchange}:{symbol}:{timeframe}:{limit}"
        
        # Check cache
        if cache_key in self.market_cache:
            cached_time, cached_data = self.market_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data.copy()
        
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
            
            exchange_obj = self.exchanges[exchange]
            
            # Fetch OHLCV data
            ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Update cache
            self.market_cache[cache_key] = (time.time(), df.copy())
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to DataFrame"""
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def get_order_book(self, symbol: str, exchange: str = 'gemini', limit: int = 10) -> Dict:
        """Fetch order book for a symbol"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
            
            order_book = self.exchanges[exchange].fetch_order_book(symbol, limit=limit)
            return {
                'bids': order_book['bids'][:limit],
                'asks': order_book['asks'][:limit],
                'timestamp': order_book['timestamp'],
                'datetime': order_book['datetime']
            }
        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            return {'bids': [], 'asks': [], 'timestamp': None, 'datetime': None}
    
    def get_account_balance(self, exchange: str = 'gemini') -> Dict:
        """Fetch account balance for an exchange"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
            
            balance = self.exchanges[exchange].fetch_balance()
            return {
                'total': balance['total'],
                'free': balance['free'],
                'used': balance['used']
            }
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {'total': {}, 'free': {}, 'used': {}}
    
    def execute_order(self, symbol: str, order_type: str, side: str, 
                     amount: float, price: Optional[float] = None,
                     exchange: str = 'gemini') -> Dict:
        """
        Execute a trading order
        
        Args:
            symbol: Trading pair
            order_type: 'market', 'limit', 'stop'
            side: 'buy' or 'sell'
            amount: Amount to trade
            price: Price for limit/stop orders
            exchange: Exchange to use
            
        Returns:
            Order result dictionary
        """
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
            
            exchange_obj = self.exchanges[exchange]
            
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount
            }
            
            if price and order_type != 'market':
                order_params['price'] = price
            
            # Execute order
            order = exchange_obj.create_order(**order_params)
            
            self.logger.info(f"Order executed: {order['id']} - {side} {amount} {symbol} @ {price}")
            
            return {
                'id': order['id'],
                'status': order['status'],
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', amount),
                'average': order.get('average'),
                'cost': order.get('cost')
            }
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def get_best_price(self, symbol: str) -> Dict:
        """Get best bid/ask across all connected exchanges"""
        prices = {}
        
        for exchange_name, exchange_obj in self.exchanges.items():
            try:
                ticker = exchange_obj.fetch_ticker(symbol)
                prices[exchange_name] = {
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'last': ticker['last'],
                    'volume': ticker['quoteVolume']
                }
            except:
                continue
        
        if prices:
            # Find best bid (highest) and best ask (lowest)
            best_bid = max(prices.items(), key=lambda x: x[1]['bid'])
            best_ask = min(prices.items(), key=lambda x: x[1]['ask'])
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'all_prices': prices,
                'spread': best_ask[1]['ask'] - best_bid[1]['bid']
            }
        
        return {'best_bid': None, 'best_ask': None, 'all_prices': {}, 'spread': None}