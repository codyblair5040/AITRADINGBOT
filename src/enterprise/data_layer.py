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
            self.logger.info("Initializing exchanges...")
            
            # Check for exchanges in config
            exchanges_config = self.config.get('exchanges', {})
            
            # Initialize Gemini if enabled
            gemini_config = exchanges_config.get('gemini', {})
            if gemini_config.get('enabled', False):
                api_key = gemini_config.get('api_key')
                api_secret = gemini_config.get('api_secret')
                
                # Also check direct environment variables
                if not api_key:
                    api_key = self.config.get('GEMINI_API_KEY')
                if not api_secret:
                    api_secret = self.config.get('GEMINI_API_SECRET')
                
                if api_key and api_secret:
                    self.exchanges['gemini'] = ccxt.gemini({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True,
                    })
                    self.active_exchange = 'gemini'
                    self.logger.info("✅ Gemini exchange initialized")
                else:
                    self.logger.warning("Gemini enabled but no API keys found")
            
            # Initialize Coinbase if enabled
            coinbase_config = exchanges_config.get('coinbase', {})
            if coinbase_config.get('enabled', False):
                api_key = coinbase_config.get('api_key') or self.config.get('COINBASE_API_KEY')
                api_secret = coinbase_config.get('api_secret') or self.config.get('COINBASE_API_SECRET')
                
                if api_key and api_secret:
                    self.exchanges['coinbase'] = ccxt.coinbase({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True,
                    })
                    self.logger.info("Coinbase exchange initialized")
            
            # Fallback to mock if no exchanges initialized
            if not self.exchanges:
                if exchanges_config.get('mock', {}).get('enabled', True):
                    self.exchanges['mock'] = 'mock'
                    self.active_exchange = 'mock'
                    self.logger.info("📝 Using mock exchange")
            
            self.logger.info(f"Total exchanges: {len(self.exchanges)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
            self.exchanges['mock'] = 'mock'
            self.active_exchange = 'mock'
            
            # Coinbase Advanced Trade (PRODUCTION ONLY - no testnet)
            if self.config.get('coinbase_api_key'):
                self.exchanges['coinbase'] = ccxt.coinbase({
                    'apiKey': self.config.get('coinbase_api_key'),
                    'secret': self.config.get('coinbase_api_secret'),
                    'enableRateLimit': True,
                    # Coinbase Advanced Trade specific
                    'options': {
                        'fetchMarkets': ['spot'],  # Only spot markets
                    }
                })
                self.logger.info("Coinbase Advanced Trade initialized (PRODUCTION)")
                self.logger.warning("⚠️ Coinbase has no testnet - use with caution!")
            
            # Binance (with testnet option)
            if self.config.get('binance_api_key'):
                config = {
                    'apiKey': self.config.get('binance_api_key'),
                    'secret': self.config.get('binance_api_secret'),
                    'enableRateLimit': True,
                }
                
                # Binance testnet for development
                if self.config.get('binance_testnet', True):
                    config.setdefault('urls', {})['api'] = {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api',
                    }
                    self.logger.info("Binance Testnet initialized")
                else:
                    self.logger.info("Binance Production initialized")
                
                self.exchanges['binance'] = ccxt.binance(config)
            
            # Kraken (alternative for testing)
            if self.config.get('kraken_api_key'):
                self.exchanges['kraken'] = ccxt.kraken({
                    'apiKey': self.config.get('kraken_api_key'),
                    'secret': self.config.get('kraken_private_key'),
                    'enableRateLimit': True,
                })
                self.logger.info("Kraken exchange initialized")
            
            # CCXT built-in test exchange (for pure development)
            if self.config.get('use_ccxt_test', False):
                self.exchanges['test'] = ccxt.test()
                self.logger.info("CCXT Test Exchange initialized - for development only")
                    
            self.logger.info(f"Total exchanges initialized: {len(self.exchanges)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
    
    def get_market_data(self, symbol: str, exchange: Optional[str] = None, 
                       timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            exchange: Exchange name (optional, uses active exchange)
            timeframe: Timeframe for candles
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if not exchange:
            exchange = self.active_exchange
        
        cache_key = f"{exchange}:{symbol}:{timeframe}:{limit}"
        
        # Check cache
        if cache_key in self.market_cache:
            cached_time, cached_data = self.market_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data.copy()
        
        try:
            exchange_obj = self.exchanges.get(exchange)
            
            if exchange_obj == 'mock' or not exchange_obj:
                # Use mock data
                self.logger.debug(f"Using mock data for {symbol}")
                return self._get_mock_market_data(symbol, limit)
            
            # Fetch from real exchange
            self.logger.debug(f"Fetching {limit} {timeframe} candles for {symbol} from {exchange}")
            ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators if we have enough data
            if len(df) > 20:
                df = self.add_technical_indicators(df)
            
            # Update cache
            self.market_cache[cache_key] = (time.time(), df.copy())
            
            self.logger.debug(f"Fetched {len(df)} candles from {exchange}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data from {exchange}: {e}")
            # Fallback to mock data
            return self._get_mock_market_data(symbol, limit)
    
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