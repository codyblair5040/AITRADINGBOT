"""
Configuration loader for enterprise trading bot
Loads YAML config and environment variables
"""
import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv
import logging

class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML and environment variables"""
        try:
            # Load environment variables
            if os.path.exists(self.env_path):
                load_dotenv(self.env_path)
                self.logger.info(f"Loaded environment variables from {self.env_path}")
            else:
                self.logger.warning(f"No .env file found at {self.env_path}")
            
            # Load YAML configuration
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self.logger.error(f"Config file not found: {self.config_path}")
                self.config = {}
            
            # Substitute environment variables in config
            self._substitute_env_vars()
            
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _substitute_env_vars(self):
        """Recursively substitute ${VAR_NAME} with environment variables"""
        def _replace(obj):
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_replace(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        self.config = _replace(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, {})
            else:
                return default
        
        return value if value != {} else default
    
    def save_config_template(self):
        """Save a configuration template file"""
        template = {
            'exchanges': {
                'gemini': {
                    'api_key': '${GEMINI_API_KEY}',
                    'api_secret': '${GEMINI_API_SECRET}',
                    'enabled': True
                }
            },
            'trading': {
                'default_symbol': 'BTC/USD',
                'risk_per_trade': 0.02
            }
        }
        
        with open('config.template.yaml', 'w') as file:
            yaml.dump(template, file, default_flow_style=False)
        
        self.logger.info("Configuration template saved to config.template.yaml")

# Singleton instance
config_loader = ConfigLoader()
config = config_loader.load()