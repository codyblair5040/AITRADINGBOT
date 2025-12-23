"""
CONFIG_MANAGER.PY
Enterprise-grade configuration management system with environment variable support
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigManager:
    """Unified configuration management system with environment variable support"""
    
    _instance = None
    _configs = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Load environment variables
        self._load_environment_variables()
        
        self.config_dir = Path(__file__).parent
        self._ensure_config_directory()
        self._load_all_configs()
        self._apply_env_overrides()
        
        self._initialized = True
        logger.info("✅ ConfigManager initialized")
    
    def _load_environment_variables(self):
        """Load environment variables from .env file if exists"""
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=env_file)
                logger.info(f"Loaded environment variables from {env_file}")
            except ImportError:
                logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
            except Exception as e:
                logger.error(f"Failed to load .env file: {e}")
    
    def _ensure_config_directory(self):
        """Ensure config directory exists with default files"""
        self.config_dir.mkdir(exist_ok=True)
        
        # Create default config files if they don't exist
        default_configs = {
            'exchange_config.yaml': self._get_default_exchange_config(),
            'trading_config.yaml': self._get_default_trading_config(),
            'risk_config.yaml': self._get_default_risk_config(),
            'ai_config.yaml': self._get_default_ai_config(),
            'general_config.yaml': self._get_default_general_config()
        }
        
        for filename, content in default_configs.items():
            filepath = self.config_dir / filename
            if not filepath.exists():
                try:
                    with open(filepath, 'w') as f:
                        yaml.dump(content, f, default_flow_style=False)
                    logger.info(f"Created default config: {filename}")
                except Exception as e:
                    logger.error(f"Failed to create {filename}: {e}")
    
    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file with variable interpolation"""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Config file not found: {filename}, using defaults")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Replace environment variables in the content
            content = self._interpolate_env_variables(content)
            
            # Parse YAML
            config = yaml.safe_load(content)
            logger.info(f"Loaded config: {filename}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {}
    
    def _interpolate_env_variables(self, content: str) -> str:
        """Replace ${VAR_NAME} with environment variables"""
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            # Try to get from environment
            value = os.getenv(var_name)
            if value is not None:
                return value
            # Try nested config reference
            if '.' in var_name:
                parts = var_name.split('.')
                config = self._configs.get(parts[0], {})
                for part in parts[1:]:
                    if isinstance(config, dict):
                        config = config.get(part, '')
                    else:
                        return ''
                return str(config) if config is not None else ''
            return match.group(0)  # Return original if not found
        
        # Replace ${VAR_NAME} patterns
        pattern = r'\$\{([A-Za-z0-9_.]+)\}'
        return re.sub(pattern, replace_var, content)
    
    def _load_all_configs(self):
        """Load all configuration files with graceful handling"""
        config_files = {
            'exchange': 'exchange_config.yaml',
            'trading': 'trading_config.yaml', 
            'risk': 'risk_config.yaml',
            'ai': 'ai_config.yaml',           # Optional - will use defaults if missing
            'general': 'general_config.yaml'   # Optional - will use defaults if missing
        }
        
        for key, filename in config_files.items():
            loaded_config = self._load_config_file(filename)
            if loaded_config or key in ['exchange', 'trading', 'risk']:  # Required files
                self._configs[key] = loaded_config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configs"""
        logger.info("Applying environment variable overrides...")
        
        # Trading mode override
        trading_mode = os.getenv('TRADING_MODE')
        if trading_mode and trading_mode in ['paper', 'live', 'backtest']:
            if 'trading' not in self._configs:
                self._configs['trading'] = {}
            self._configs['trading']['mode'] = trading_mode
            logger.info(f"  Set trading mode: {trading_mode}")
        
        # Initial balance override
        initial_balance = os.getenv('INITIAL_BALANCE')
        if initial_balance:
            try:
                balance = float(initial_balance)
                if 'paper_trading' not in self._configs.get('trading', {}):
                    if 'trading' not in self._configs:
                        self._configs['trading'] = {}
                    self._configs['trading']['paper_trading'] = {}
                self._configs['trading']['paper_trading']['initial_balance'] = balance
                logger.info(f"  Set initial balance: ${balance}")
            except ValueError:
                logger.warning(f"Invalid INITIAL_BALANCE: {initial_balance}")
        
        # Symbols override - NEW: Allow overriding symbols via environment
        symbols_env = os.getenv('TRADING_SYMBOLS')
        if symbols_env:
            try:
                symbols = [s.strip() for s in symbols_env.split(',')]
                if 'trading' not in self._configs:
                    self._configs['trading'] = {}
                self._configs['trading']['symbols'] = symbols
                logger.info(f"  Set symbols from environment: {len(symbols)} symbols")
            except Exception as e:
                logger.warning(f"Failed to parse TRADING_SYMBOLS: {e}")
        
        # API Key overrides
        api_overrides = {
            'GEMINI_API_KEY': ('exchange', 'exchanges', 'gemini', 'api_key'),
            'GEMINI_API_SECRET': ('exchange', 'exchanges', 'gemini', 'api_secret'),
            'COINBASE_API_KEY': ('exchange', 'exchanges', 'coinbase', 'api_key'),
            'COINBASE_API_SECRET': ('exchange', 'exchanges', 'coinbase', 'api_secret')
        }
        
        for env_var, path in api_overrides.items():
            value = os.getenv(env_var)
            if value:
                section, *keys = path
                config_section = self._configs.get(section, {})
                current = config_section
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
                logger.info(f"  Set {env_var}")
    
    def _get_default_exchange_config(self) -> Dict[str, Any]:
        """Get default exchange configuration"""
        return {
            'exchanges': {
                'gemini': {
                    'api_key': '${GEMINI_API_KEY}',
                    'api_secret': '${GEMINI_API_SECRET}',
                    'sandbox': True,
                    'enable_rate_limit': True,
                    'timeout': 30000,
                    'verbose': False,
                    'rate_limit': True
                },
                'coinbase': {
                    'api_key': '${COINBASE_API_KEY}',
                    'api_secret': '${COINBASE_API_SECRET}',
                    'sandbox': True,
                    'enable_rate_limit': True,
                    'timeout': 30000,
                    'verbose': False
                }
            },
            'defaults': {
                'enable_rate_limit': True,
                'timeout': 30000,
                'sandbox': True,
                'verbose': False
            },
            'environments': {
                'development': {
                    'sandbox': True,
                    'verbose': True
                },
                'production': {
                    'sandbox': False,
                    'verbose': False
                }
            }
        }
    
    def _get_default_trading_config(self) -> Dict[str, Any]:
        """Get default trading configuration - UPDATED with all 10 symbols"""
        return {
            'trading': {
                'mode': 'paper',
                'symbols': [
                    'BTC/USD', 'ETH/USD', 'SOL/USD',
                    'ADA/USD', 'DOT/USD', 'MATIC/USD',
                    'AVAX/USD', 'LINK/USD', 'UNI/USD', 
                    'XRP/USD'
                ],
                'default_symbol': 'BTC/USD',
                'timeframe': '1m',
                'max_open_trades': 3,
                'risk_per_trade': 0.01,
                'stop_loss_percent': 2.0,
                'take_profit_percent': 3.0,
                'update_interval': 30
            },
            'paper_trading': {
                'initial_balance': 10000.00,
                'simulate_fees': True,
                'fee_rate': 0.001,
                'slippage': 0.0005,
                'enabled': True
            },
            'live_trading': {
                'max_open_orders': 5,
                'order_timeout': 30,
                'confirm_order': False,
                'min_order_size': 10.00,
                'enabled': False
            }
        }
    
    def _get_default_risk_config(self) -> Dict[str, Any]:
        """Get default risk management configuration"""
        return {
            'position_sizing': {
                'method': 'fixed_fractional',
                'max_position_size': 0.1,
                'max_portfolio_risk': 0.02,
                'kelly_fraction': 0.5
            },
            'stop_loss': {
                'enabled': True,
                'method': 'dynamic',
                'default_pct': 0.02,
                'max_pct': 0.05,
                'trailing_enabled': True,
                'trailing_distance': 0.01
            },
            'take_profit': {
                'enabled': True,
                'levels': [
                    {'pct': 0.01, 'size_pct': 0.33},
                    {'pct': 0.02, 'size_pct': 0.33},
                    {'pct': 0.03, 'size_pct': 0.34}
                ]
            },
            'circuit_breakers': {
                'daily_loss_limit': 0.05,
                'max_drawdown': 0.10,
                'consecutive_losses': 3,
                'volatility_limit': 0.15,
                'position_limit': 5
            }
        }
    
    def _get_default_ai_config(self) -> Dict[str, Any]:
        """Get default AI engine configuration"""
        return {
            'models': {
                'strategy_predictor': {
                    'type': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 10
                },
                'regime_classifier': {
                    'type': 'random_forest',
                    'n_estimators': 50,
                    'max_depth': 8
                },
                'profitability_predictor': {
                    'type': 'gradient_boosting',
                    'n_estimators': 100,
                    'max_depth': 7,
                    'learning_rate': 0.1
                }
            },
            'learning': {
                'retrain_interval': 50,
                'min_samples': 20,
                'validation_split': 0.2,
                'save_interval': 100
            },
            'strategies': {
                'quantitative_momentum': {'weight': 0.25},
                'statistical_arbitrage': {'weight': 0.20},
                'market_making': {'weight': 0.20},
                'ml_trend_prediction': {'weight': 0.20},
                'volatility_breakout': {'weight': 0.15}
            }
        }
    
    def _get_default_general_config(self) -> Dict[str, Any]:
        """Get default general configuration"""
        return {
            'logging': {
                'level': 'INFO',
                'file': 'logs/trading_bot.log',
                'max_size_mb': 10,
                'backup_count': 5
            },
            'performance': {
                'tracking_enabled': True,
                'save_interval': 10,
                'report_format': 'json',
                'metrics': ['sharpe_ratio', 'max_drawdown', 'win_rate']
            },
            'system': {
                'check_interval': 60,
                'health_check': True,
                'auto_restart': False,
                'memory_limit_mb': 1024
            },
            'notifications': {
                'enabled': False,
                'telegram': {
                    'bot_token': '${TELEGRAM_BOT_TOKEN}',
                    'chat_id': '${TELEGRAM_CHAT_ID}'
                },
                'email': {
                    'smtp_server': '${EMAIL_SMTP_SERVER}',
                    'username': '${EMAIL_USERNAME}',
                    'password': '${EMAIL_PASSWORD}'
                }
            }
        }
    
    # ============== PUBLIC API ==============
    
    def get_exchange_config(self, exchange_name: str = None) -> Dict[str, Any]:
        """Get exchange configuration"""
        exchange_configs = self._configs.get('exchange', {}).get('exchanges', {})
        
        if exchange_name:
            return exchange_configs.get(exchange_name, {})
        return exchange_configs
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        trading_config = self._configs.get('trading', {})
        
        # Ensure symbols exist - if not, add default 10 symbols
        if 'symbols' not in trading_config or not trading_config['symbols']:
            trading_config['symbols'] = [
                'BTC/USD', 'ETH/USD', 'SOL/USD',
                'ADA/USD', 'DOT/USD', 'MATIC/USD',
                'AVAX/USD', 'LINK/USD', 'UNI/USD', 
                'XRP/USD'
            ]
            logger.warning(f"Added default 10 symbols to trading config")
        
        return trading_config
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self._configs.get('risk', {})
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI engine configuration"""
        return self._configs.get('ai', {})
    
    def get_general_config(self) -> Dict[str, Any]:
        """Get general configuration"""
        return self._configs.get('general', {})
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get all configurations merged"""
        full_config = {}
        for key, config in self._configs.items():
            full_config[key] = config
        return full_config
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get a specific configuration value using dot notation"""
        parts = path.split('.')
        config = self._configs
        
        for part in parts:
            if isinstance(config, dict):
                config = config.get(part)
                if config is None:
                    return default
            else:
                return default
        
        return config if config is not None else default
    
    def update_config(self, section: str, updates: Dict[str, Any], save: bool = False):
        """Update configuration in memory"""
        if section not in self._configs:
            self._configs[section] = {}
        
        # Deep update
        import collections.abc
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._configs[section] = deep_update(self._configs[section], updates)
        
        if save:
            self.save_config(section)
    
    def save_config(self, section: str, filename: str = None):
        """Save configuration to file"""
        if not filename:
            filename = f"{section}_config.yaml"
        
        filepath = self.config_dir / filename
        config = self._configs.get(section, {})
        
        try:
            # Ensure directory exists
            filepath.parent.mkdir(exist_ok=True, parents=True)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved config: {filename}")
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
    
    def save_all_configs(self):
        """Save all configurations to files"""
        for section in self._configs.keys():
            self.save_config(section)
        logger.info("All configurations saved")
    
    def reload_configs(self):
        """Reload all configurations from files"""
        self._configs = {}
        self._load_all_configs()
        self._apply_env_overrides()
        logger.info("All configurations reloaded")
    
    def get_environment(self) -> str:
        """Get current environment (development/production)"""
        env = os.getenv('ENVIRONMENT', 'development')
        return env.lower()
    
    def get_mode(self) -> str:
        """Get trading mode"""
        return self.get_config_value('trading.mode', 'paper')
    
    def get_symbols(self) -> List[str]:
        """Get trading symbols - convenience method"""
        return self.get_trading_config().get('symbols', [])
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required fields
        required_paths = [
            'trading.mode',
            'trading.symbols',
            'trading.paper_trading.initial_balance'
        ]
        
        for path in required_paths:
            if self.get_config_value(path) is None:
                issues.append(f"Missing required config: {path}")
        
        # Validate trading mode
        mode = self.get_mode()
        if mode not in ['paper', 'live', 'backtest']:
            issues.append(f"Invalid trading mode: {mode}")
        
        # Validate initial balance
        balance = self.get_config_value('trading.paper_trading.initial_balance')
        if balance is not None and (not isinstance(balance, (int, float)) or balance <= 0):
            issues.append(f"Invalid initial balance: {balance}")
        
        # Validate symbols
        symbols = self.get_symbols()
        if not symbols:
            issues.append("No trading symbols configured")
        elif len(symbols) < 3:
            issues.append(f"Too few symbols configured: {len(symbols)}")
        
        return issues
    
    def export_config(self, format: str = 'yaml') -> str:
        """Export configuration in specified format"""
        full_config = self.get_full_config()
        
        if format.lower() == 'json':
            return json.dumps(full_config, indent=2, default=str)
        elif format.lower() == 'yaml':
            return yaml.dump(full_config, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Helper function for quick access
def get_config_manager() -> ConfigManager:
    """Get the singleton ConfigManager instance"""
    return ConfigManager()

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ConfigManager...")
    print("=" * 50)
    
    cm = ConfigManager()
    
    print(f"Trading Mode: {cm.get_mode()}")
    print(f"Environment: {cm.get_environment()}")
    
    print(f"\nTrading Symbols: {cm.get_symbols()}")
    print(f"Number of symbols: {len(cm.get_symbols())}")
    
    print("\nConfig Sections:")
    for section in cm._configs.keys():
        print(f"  - {section}")
    
    print("\nValidation Issues:")
    issues = cm.validate_config()
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ No issues found")
    
    print("\nSample Config Values:")
    print(f"  Initial Balance: ${cm.get_config_value('trading.paper_trading.initial_balance', 0)}")
    print(f"  Max Position Size: {cm.get_config_value('risk.position_sizing.max_position_size', 0)}")
    
    print("\n✅ ConfigManager test complete")