#!/usr/bin/env python3
"""
Development entry point - Runs without emojis and with mock data
"""
import sys
import os
import logging

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir)
sys.path.insert(0, src_dir)

# Configure logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dev_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

print("=" * 60)
print("AI TRADING BOT - ENTERPRISE EDITION")
print("DEVELOPMENT MODE")
print("=" * 60)
print("Running with mock data and paper trading")
print("Press Ctrl+C to stop")
print("=" * 60)

try:
    # Try to import the enterprise engine
    from enterprise.engine import EnterpriseTradingEngine
    logger.info("Enterprise engine imported successfully")
    
    # Create and run engine
    engine = EnterpriseTradingEngine()
    
    # Override config for development
    engine.config = {
        'development': {
            'use_mock_data': True,
            'paper_trading': True,
            'simulate_execution': True
        },
        'exchanges': {
            'test': {'enabled': True}
        },
        'trading': {
            'default_symbol': 'BTC/USD',
            'risk_per_trade': 0.02,
            'paper_trading': True,
            'update_interval': 30  # Faster for testing
        }
    }
    
    # Run the engine
    engine.run()
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    print(f"\nERROR: Could not import modules: {e}")
    print("\nTrying to diagnose the issue...")
    
    # Test individual imports
    print("\nTesting imports:")
    try:
        from enterprise import data_layer
        print("✅ enterprise.data_layer")
    except ImportError as e:
        print(f"❌ enterprise.data_layer: {e}")
    
    try:
        from enterprise import ai_engine
        print("✅ enterprise.ai_engine")
    except ImportError as e:
        print(f"❌ enterprise.ai_engine: {e}")
    
    try:
        from enterprise import risk_engine
        print("✅ enterprise.risk_engine")
    except ImportError as e:
        print(f"❌ enterprise.risk_engine: {e}")
    
    try:
        from enterprise import monitoring
        print("✅ enterprise.monitoring")
    except ImportError as e:
        print(f"❌ enterprise.monitoring: {e}")
    
    print("\nChecking file structure...")
    import os
    enterprise_dir = os.path.join('src', 'enterprise')
    if os.path.exists(enterprise_dir):
        print(f"Enterprise directory exists: {enterprise_dir}")
        files = os.listdir(enterprise_dir)
        print(f"Files in enterprise directory: {files}")
    else:
        print(f"Enterprise directory not found: {enterprise_dir}")
    
except KeyboardInterrupt:
    print("\n\nBot stopped by user")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()