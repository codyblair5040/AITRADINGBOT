#!/usr/bin/env python3
"""
Main entry point for AI Trading Bot Enterprise Edition
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enterprise.engine import create_engine
import logging

def main():
    """Main function"""
    print("=" * 60)
    print("🤖 AI TRADING BOT - ENTERPRISE EDITION")
    print("=" * 60)
    
    try:
        # Create and run engine
        engine = create_engine()
        engine.run()
        
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()