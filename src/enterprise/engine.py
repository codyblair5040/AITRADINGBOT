"""
Enterprise Trading Engine - Orchestrates all components
Main controller for the enterprise trading bot
"""
import logging
import time
import signal
import sys
from typing import Dict, Any, Optional
import yaml
import os
from datetime import datetime, timedelta

from .data_layer import DataLayer
from .ai_engine import AIEngine
from .risk_engine import RiskEngine
from .monitoring import MonitoringDashboard
from .config_loader import config

class EnterpriseTradingEngine:
    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        """
        Initialize Enterprise Trading Engine
        
        Args:
            config_path: Path to YAML configuration file
            env_path: Path to environment variables file
        """
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config_path = config_path
        self.env_path = env_path
        self.config = self.load_config()
        
        # Runtime state
        self.running = False
        self.components = {}
        self.trades = []
        self.performance_metrics = {}
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("Enterprise Trading Engine instance created")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_file = os.getenv('LOG_FILE', 'logs/trading_bot.log')
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML and environment variables
        
        Returns:
            Configuration dictionary
        """
        try:
            # Use the config_loader module
            self.logger.info(f"Loading configuration from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config files are missing"""
        return {
            'exchanges': {
                'gemini_sandbox': {
                    'enabled': True,
                    'sandbox': True
                }
            },
            'trading': {
                'default_symbol': 'BTC/USD',
                'risk_per_trade': 0.02,
                'paper_trading': True,
                'update_interval': 60  # seconds
            },
            'development': {
                'use_mock_data': True,
                'simulate_execution': True
            }
        }
    
    def initialize_components(self):
        """Initialize all enterprise components"""
        self.logger.info("Initializing Enterprise Trading Engine components...")
        
        try:
            # 1. Data Layer (Market Data & Exchange Communication)
            self.components['data'] = DataLayer(
                config=self.config.get('exchanges', {})
            )
            self.logger.info("[OK] Data Layer initialized")
            
            # 2. Risk Engine (Risk Management & Position Sizing)
            self.components['risk'] = RiskEngine(
                config=self.config.get('risk', {}),
                initial_capital=self.config.get('trading', {}).get('initial_capital', 10000)
            )
            self.logger.info("[OK] Risk Engine initialized")
            
            # 3. AI Engine (Predictive Models & Decision Making)
            self.components['ai'] = AIEngine(
                data_layer=self.components['data'],
                risk_engine=self.components['risk'],
                config=self.config.get('ai', {})
            )
            self.logger.info("[OK] AI Engine initialized")
            
            # 4. Monitoring Dashboard (Performance Tracking & Visualization)
            self.components['monitor'] = MonitoringDashboard(
                config=self.config.get('monitoring', {}),
                engine=self
            )
            self.logger.info("[OK] Monitoring Dashboard initialized")
            
            self.logger.info("[OK] All enterprise components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def validate_market_conditions(self) -> bool:
        """
        Validate current market conditions before trading
        
        Returns:
            bool: True if conditions are favorable for trading
        """
        try:
            # Check if markets are open (for traditional markets)
            # For crypto, always return True
            current_hour = datetime.now().hour
            
            # Basic validation - could be expanded
            validation_passed = True
            validation_errors = []
            
            # Example validations:
            # 1. Check if we have recent market data
            if self.components['data']:
                recent_data = self.components['data'].get_market_data(
                    symbol=self.config['trading']['default_symbol'],
                    limit=5
                )
                if len(recent_data) < 3:
                    validation_errors.append("Insufficient market data")
                    validation_passed = False
            
            # 2. Check risk limits
            if self.components['risk']:
                risk_status = self.components['risk'].get_current_risk_status()
                if risk_status.get('max_drawdown_exceeded', False):
                    validation_errors.append("Maximum drawdown exceeded")
                    validation_passed = False
            
            if not validation_passed:
                self.logger.warning(f"Market validation failed: {validation_errors}")
            
            return validation_passed
            
        except Exception as e:
            self.logger.error(f"Market validation error: {e}")
            return False
    
    def execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        cycle_start = time.time()
        self.logger.debug("Starting trading cycle")
        
        try:
            # 1. Fetch latest market data
            symbol = self.config['trading']['default_symbol']
            market_data = self.components['data'].get_market_data(
                symbol=symbol,
                limit=100
            )
            
            if len(market_data) < 10:
                self.logger.warning("Insufficient data for analysis")
                return
            
            # 2. Get AI prediction
            prediction = self.components['ai'].predict(
                market_data=market_data,
                symbol=symbol
            )
            
            if not prediction or prediction.get('confidence', 0) < self.config['ai'].get('confidence_threshold', 0.65):
                self.logger.debug("Low confidence prediction, skipping trade")
                return
            
            # 3. Calculate position size using risk engine
            position_size = self.components['risk'].calculate_position_size(
                prediction=prediction,
                account_balance=10000,  # This should come from exchange
                risk_per_trade=self.config['trading']['risk_per_trade']
            )
            
            if position_size <= 0:
                self.logger.debug("Zero or negative position size, skipping")
                return
            
            # 4. Validate market conditions
            if not self.validate_market_conditions():
                self.logger.debug("Market conditions unfavorable, skipping trade")
                return
            
            # 5. Execute trade (or simulate in paper trading)
            if self.config['trading'].get('paper_trading', True):
                # Paper trading - simulate execution
                simulated_trade = self.simulate_trade(
                    symbol=symbol,
                    side=prediction['action'],
                    amount=position_size,
                    price=market_data['close'].iloc[-1]
                )
                self.trades.append(simulated_trade)
                self.logger.info(f"📝 Paper Trade: {simulated_trade}")
            else:
                # Real trading - execute on exchange
                trade_result = self.components['data'].execute_order(
                    symbol=symbol,
                    order_type='limit',
                    side=prediction['action'],
                    amount=position_size,
                    price=market_data['close'].iloc[-1]
                )
                self.trades.append(trade_result)
                self.logger.info(f"💰 Real Trade Executed: {trade_result}")
            
            # 6. Update performance metrics
            self.update_performance_metrics()
            
            # 7. Update monitoring dashboard
            self.components['monitor'].update(
                trades=self.trades[-10:],  # Last 10 trades
                metrics=self.performance_metrics,
                market_data=market_data.tail(50)
            )
            
            cycle_time = time.time() - cycle_start
            self.logger.debug(f"Trading cycle completed in {cycle_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def simulate_trade(self, symbol: str, side: str, amount: float, price: float) -> Dict[str, Any]:
        """Simulate a trade for paper trading"""
        return {
            'id': f"sim_{int(time.time())}",
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'status': 'filled',
            'type': 'simulated',
            'pnl': 0.0,  # Would calculate based on exit
            'commission': 0.001 * amount * price  # 0.1% commission
        }
    
    def update_performance_metrics(self):
        """Update performance metrics based on recent trades"""
        if not self.trades:
            return
        
        recent_trades = self.trades[-20:]  # Last 20 trades
        
        # Calculate basic metrics
        winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in recent_trades if t.get('pnl', 0) <= 0]
        
        self.performance_metrics = {
            'total_trades': len(recent_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(recent_trades) if recent_trades else 0,
            'total_pnl': sum(t.get('pnl', 0) for t in recent_trades),
            'avg_win': sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(t.get('pnl', 0) for t in winning_trades) / 
                               sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades else float('inf'),
            'last_update': datetime.now().isoformat()
        }
    
    def run(self):
        """Main trading loop"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING ENTERPRISE TRADING ENGINE")
        self.logger.info("=" * 60)
        
        # Initialize components
        self.initialize_components()
        
        # Start monitoring dashboard
        self.components['monitor'].start()
        
        # Set running flag
        self.running = True
        
        # Get configuration
        update_interval = self.config['trading'].get('update_interval', 60)
        
        self.logger.info(f"Trading configuration:")
        self.logger.info(f"  Default Symbol: {self.config['trading'].get('default_symbol')}")
        self.logger.info(f"  Risk per Trade: {self.config['trading'].get('risk_per_trade') * 100}%")
        self.logger.info(f"  Update Interval: {update_interval} seconds")
        self.logger.info(f"  Paper Trading: {self.config['trading'].get('paper_trading', True)}")
        self.logger.info("-" * 40)
        
        cycle_count = 0
        
        try:
            while self.running:
                cycle_start = time.time()
                cycle_count += 1
                
                self.logger.info(f"--- Trading Cycle #{cycle_count} ---")
                
                # Execute trading cycle
                self.execute_trading_cycle()
                
                # Calculate sleep time
                cycle_time = time.time() - cycle_start
                sleep_time = max(1, update_interval - cycle_time)
                
                # Log cycle completion
                self.logger.info(f"Cycle completed in {cycle_time:.2f}s, sleeping for {sleep_time:.2f}s")
                
                # Sleep until next cycle
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    def shutdown(self):
        """Graceful shutdown procedure"""
        self.logger.info("Initiating graceful shutdown...")
        
        # Stop monitoring dashboard
        if 'monitor' in self.components:
            self.components['monitor'].stop()
        
        # Save state if needed
        self.save_state()
        
        # Generate final report
        self.generate_final_report()
        
        self.logger.info("=" * 60)
        self.logger.info("ENTERPRISE TRADING ENGINE SHUTDOWN COMPLETE")
        self.logger.info("=" * 60)
    
    def save_state(self):
        """Save engine state for recovery"""
        try:
            state = {
                'trades': self.trades[-100:],  # Last 100 trades
                'metrics': self.performance_metrics,
                'shutdown_time': datetime.now().isoformat()
            }
            
            os.makedirs('state', exist_ok=True)
            import json
            with open('state/engine_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info("Engine state saved")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def generate_final_report(self):
        """Generate final performance report"""
        self.logger.info("=" * 60)
        self.logger.info("FINAL PERFORMANCE REPORT")
        self.logger.info("=" * 60)
        
        if self.performance_metrics:
            for key, value in self.performance_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    self.logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        self.logger.info(f"Total Trading Cycles: {len(self.trades)}")
        self.logger.info("=" * 60)


# Factory function for easier instantiation
def create_engine(config_path: str = "config.yaml", env_path: str = ".env") -> EnterpriseTradingEngine:
    """Factory function to create and configure trading engine"""
    return EnterpriseTradingEngine(config_path=config_path, env_path=env_path)


if __name__ == "__main__":
    # Direct execution for testing
    engine = EnterpriseTradingEngine()
    engine.run()