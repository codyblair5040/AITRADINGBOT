"""
Exchange Manager - Handles exchange-specific differences
"""
import ccxt
import logging
from typing import Dict, Any, Optional

class ExchangeManager:
    """Manages exchange connections and handles exchange-specific differences"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchanges = {}
        
    def initialize_exchange(self, exchange_name: str) -> Optional[ccxt.Exchange]:
        """Initialize a specific exchange with proper configuration"""
        exchange_configs = {
            'gemini_live': self._init_gemini_live,
            'coinbase_live': self._init_coinbase_live,
            'coinbase_test': self._init_coinbase_test,
            'binance_testnet': self._init_binance_testnet,
            'kraken': self._init_kraken,
            'mock': self._init_mock
        }
        
        if exchange_name in exchange_configs:
            try:
                exchange = exchange_configs[exchange_name]()
                if exchange:
                    self.exchanges[exchange_name] = exchange
                    self.logger.info(f"Initialized {exchange_name}")
                return exchange
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {e}")
        
        return None
    
    def _init_gemini_live(self) -> ccxt.Exchange:
        """Initialize Gemini Live Trading"""
        config = self.config.get('exchanges', {}).get('gemini_live', {})
        
        # Safety check: Ensure it's an account key
        api_key = config.get('api_key', '')
        if not api_key.startswith('account-'):
            self.logger.error("GEMINI ERROR: API key must start with 'account-'")
            self.logger.error("You have a Master key. Create an Account key in Gemini Settings.")
            raise ValueError("Invalid Gemini API key type")
        
        return ccxt.gemini({
            'apiKey': api_key,
            'secret': config.get('api_secret', ''),
            'enableRateLimit': True,
            'options': {
                'fetchMarkets': ['spot'],
            }
        })
    
    def _init_coinbase_live(self) -> ccxt.Exchange:
        """Initialize Coinbase Advanced Trade"""
        config = self.config.get('exchanges', {}).get('coinbase_live', {})
        
        # Coinbase Advanced Trade configuration
        exchange_config = {
            'apiKey': config.get('api_key', ''),
            'secret': config.get('api_secret', ''),
            'enableRateLimit': True,
            # Coinbase Advanced Trade doesn't use passphrase anymore
        }
        
        # Add sandbox URLs if in test mode
        if config.get('sandbox', False):
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://api-public.sandbox.exchange.coinbase.com',
                    'private': 'https://api-public.sandbox.exchange.coinbase.com',
                }
            }
        
        return ccxt.coinbase(exchange_config)
    
    def _init_coinbase_test(self) -> ccxt.Exchange:
        """Initialize Coinbase Test/Sandbox"""
        config = self.config.get('exchanges', {}).get('coinbase_test', {})
        
        return ccxt.coinbase({
            'apiKey': config.get('api_key', ''),
            'secret': config.get('api_secret', ''),
            'enableRateLimit': True,
            'urls': {
                'api': {
                    'public': 'https://api-public.sandbox.exchange.coinbase.com',
                    'private': 'https://api-public.sandbox.exchange.coinbase.com',
                }
            }
        })
    
    def _init_binance_testnet(self) -> ccxt.Exchange:
        """Initialize Binance Testnet (Recommended for testing)"""
        config = self.config.get('exchanges', {}).get('binance_testnet', {})
        
        return ccxt.binance({
            'apiKey': config.get('api_key', ''),
            'secret': config.get('api_secret', ''),
            'enableRateLimit': True,
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api/v3',
                    'private': 'https://testnet.binance.vision/api/v3',
                }
            },
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
    
    def _init_kraken(self) -> ccxt.Exchange:
        """Initialize Kraken"""
        config = self.config.get('exchanges', {}).get('kraken', {})
        
        return ccxt.kraken({
            'apiKey': config.get('api_key', ''),
            'secret': config.get('secret', ''),
            'enableRateLimit': True,
        })
    
    def _init_mock(self) -> ccxt.Exchange:
        """Initialize Mock Exchange for testing"""
        # Return a mock object that simulates exchange behavior
        class MockExchange:
            def fetch_ticker(self, symbol):
                return {'last': 50000, 'bid': 49900, 'ask': 50100}
            def fetch_balance(self):
                return {'total': {'USD': 10000, 'BTC': 0.1}}
            def create_order(self, *args, **kwargs):
                return {'id': 'mock_order_123', 'status': 'filled'}
        
        return MockExchange()
    
    def normalize_symbol(self, exchange_name: str, symbol: str) -> str:
        """Convert symbol to exchange-specific format"""
        symbol_mapping = {
            'gemini_live': {
                'BTC/USD': 'BTC/USD',
                'BTC/USDT': 'BTC/USD',  # Gemini uses USD, not USDT
                'ETH/USD': 'ETH/USD',
            },
            'coinbase_live': {
                'BTC/USD': 'BTC-USD',
                'BTC/USDT': 'BTC-USD',  # Coinbase uses USD
                'ETH/USD': 'ETH-USD',
            },
            'binance_testnet': {
                'BTC/USD': 'BTC/USDT',  # Binance uses USDT
                'BTC/USDT': 'BTC/USDT',
                'ETH/USD': 'ETH/USDT',
            }
        }
        
        return symbol_mapping.get(exchange_name, {}).get(symbol, symbol)
    
    def get_exchange(self, exchange_name: str) -> Optional[ccxt.Exchange]:
        """Get exchange instance by name"""
        return self.exchanges.get(exchange_name)