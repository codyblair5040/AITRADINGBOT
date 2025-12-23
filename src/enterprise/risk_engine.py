"""
Risk Engine - Advanced risk management with Kelly Criterion
Handles position sizing, risk validation, and portfolio management
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import math

class RiskEngine:
    def __init__(self, config: Optional[Dict] = None, initial_capital: float = 10000):
        """
        Initialize Risk Engine
        
        Args:
            config: Configuration dictionary
            initial_capital: Starting capital for risk calculations
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.initial_capital = initial_capital
        
        # Portfolio state
        self.portfolio_value = initial_capital
        self.open_positions = []
        self.closed_positions = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_portfolio_value = initial_capital
        
        # Performance tracking
        self.returns_history = []
        self.risk_history = []
        
        # Risk limits
        self.max_position_size = self.config.get('max_position_size', 0.15)  # 15%
        self.min_position_size = self.config.get('min_position_size', 0.01)   # 1%
        self.max_drawdown_limit = self.config.get('max_drawdown', 0.25)       # 25%
        self.daily_loss_limit = self.config.get('daily_loss_limit', 0.10)     # 10%
        self.weekly_loss_limit = self.config.get('weekly_loss_limit', 0.20)   # 20%
        
        # Kelly Criterion settings
        self.use_kelly = self.config.get('use_kelly_criterion', True)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)  # Half-Kelly
        self.min_kelly_fraction = self.config.get('min_kelly_fraction', 0.1)
        self.max_kelly_fraction = self.config.get('max_kelly_fraction', 0.8)
        
        self.logger.info(f"Risk Engine initialized with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, prediction: Dict, account_balance: float, 
                               risk_per_trade: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            prediction: AI prediction with confidence and expected return
            account_balance: Current account balance
            risk_per_trade: Maximum risk per trade (e.g., 0.02 for 2%)
            
        Returns:
            Position size in base currency
        """
        try:
            # Extract prediction parameters
            confidence = prediction.get('confidence', 0.5)
            expected_return = abs(prediction.get('expected_return', 0.01)) / 100  # Convert % to decimal
            current_price = prediction.get('current_price', 1.0)
            
            # Validate inputs
            if confidence <= 0 or expected_return <= 0 or account_balance <= 0:
                self.logger.warning("Invalid inputs for position sizing")
                return 0.0
            
            # Method 1: Fixed fractional position sizing
            fixed_fraction = account_balance * risk_per_trade
            
            # Method 2: Kelly Criterion (if enabled)
            if self.use_kelly:
                kelly_size = self._calculate_kelly_position(
                    win_probability=confidence,
                    win_loss_ratio=expected_return / risk_per_trade,
                    account_balance=account_balance
                )
            else:
                kelly_size = fixed_fraction
            
            # Apply fractional Kelly (safer)
            fractional_kelly = kelly_size * self.kelly_fraction
            
            # Apply bounds
            min_size = account_balance * self.min_position_size
            max_size = account_balance * self.max_position_size
            
            position_size = max(min_size, min(fractional_kelly, max_size, fixed_fraction))
            
            # Convert to asset units if price is available
            if current_price > 0:
                position_units = position_size / current_price
                self.logger.debug(
                    f"Position sizing: Confidence={confidence:.2%}, "
                    f"Expected Return={expected_return:.2%}, "
                    f"Size=${position_size:,.2f} ({position_units:.6f} units)"
                )
            else:
                position_units = position_size
            
            return position_units
            
        except Exception as e:
            self.logger.error(f"Position sizing error: {e}")
            return account_balance * risk_per_trade * 0.5  # Fallback
    
    def _calculate_kelly_position(self, win_probability: float, win_loss_ratio: float, 
                                 account_balance: float) -> float:
        """
        Calculate Kelly Criterion position size
        
        Formula: f* = p - q/b
        Where:
          f* = fraction of capital to bet
          p = probability of winning
          q = probability of losing (1 - p)
          b = win/loss ratio (amount won per unit lost)
        
        Args:
            win_probability: Probability of winning (0 to 1)
            win_loss_ratio: Win amount divided by loss amount
            account_balance: Current account balance
            
        Returns:
            Optimal fraction of capital to risk
        """
        try:
            # Basic Kelly formula
            q = 1 - win_probability
            b = win_loss_ratio
            
            if b <= 0:
                self.logger.warning(f"Invalid win/loss ratio: {b}")
                return 0.0
            
            kelly_fraction = (win_probability * b - q) / b
            
            # Handle edge cases
            if kelly_fraction < 0:
                kelly_fraction = 0.0  # Don't bet if negative expectation
            elif kelly_fraction > 1:
                kelly_fraction = 1.0  # Cap at 100%
            
            # Apply safety bounds
            kelly_fraction = max(self.min_kelly_fraction, 
                               min(kelly_fraction, self.max_kelly_fraction))
            
            position_size = account_balance * kelly_fraction
            
            self.logger.debug(
                f"Kelly calculation: p={win_probability:.3f}, b={win_loss_ratio:.3f}, "
                f"f*={kelly_fraction:.3f}, size=${position_size:,.2f}"
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Kelly calculation error: {e}")
            return 0.0
    
    def validate_trade(self, symbol: str, side: str, size: float, price: float) -> Dict[str, Any]:
        """
        Validate a trade against risk limits
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Position size in units
            price: Entry price
            
        Returns:
            Validation result dictionary
        """
        validation = {
            'approved': True,
            'reasons': [],
            'warnings': [],
            'max_allowed_size': size,
            'recommended_size': size
        }
        
        try:
            trade_value = size * price
            
            # 1. Check position size limits
            position_pct = trade_value / self.portfolio_value if self.portfolio_value > 0 else 0
            
            if position_pct > self.max_position_size:
                validation['approved'] = False
                validation['reasons'].append(
                    f"Position size {position_pct:.1%} exceeds maximum {self.max_position_size:.1%}"
                )
                validation['max_allowed_size'] = self.portfolio_value * self.max_position_size / price
            
            if position_pct < self.min_position_size:
                validation['warnings'].append(
                    f"Position size {position_pct:.1%} below minimum {self.min_position_size:.1%}"
                )
            
            # 2. Check drawdown limits
            if self.current_drawdown > self.max_drawdown_limit:
                validation['approved'] = False
                validation['reasons'].append(
                    f"Current drawdown {self.current_drawdown:.1%} exceeds limit {self.max_drawdown_limit:.1%}"
                )
            
            # 3. Check daily loss limit
            daily_loss = self._calculate_daily_loss()
            if daily_loss > self.daily_loss_limit:
                validation['approved'] = False
                validation['reasons'].append(
                    f"Daily loss {daily_loss:.1%} exceeds limit {self.daily_loss_limit:.1%}"
                )
            
            # 4. Check weekly loss limit
            weekly_loss = self._calculate_weekly_loss()
            if weekly_loss > self.weekly_loss_limit:
                validation['approved'] = False
                validation['reasons'].append(
                    f"Weekly loss {weekly_loss:.1%} exceeds limit {self.weekly_loss_limit:.1%}"
                )
            
            # 5. Check correlation with existing positions
            correlation_risk = self._check_correlation_risk(symbol, side)
            if correlation_risk['high_risk']:
                validation['warnings'].append(
                    f"High correlation with existing positions: {correlation_risk['max_correlation']:.3f}"
                )
            
            # 6. Check concentration risk
            concentration = self._calculate_portfolio_concentration()
            if concentration > 0.3:  # 30% max concentration
                validation['warnings'].append(
                    f"Portfolio concentration {concentration:.1%} is high"
                )
            
            # Update recommended size based on validations
            if not validation['approved'] and validation.get('max_allowed_size'):
                validation['recommended_size'] = min(size, validation['max_allowed_size'])
            
            # Log validation result
            if validation['approved']:
                if validation['warnings']:
                    self.logger.warning(f"Trade approved with warnings: {validation['warnings']}")
                else:
                    self.logger.info(f"Trade approved: {side} {size} {symbol} @ {price}")
            else:
                self.logger.warning(f"Trade rejected: {validation['reasons']}")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {e}")
            validation['approved'] = False
            validation['reasons'].append(f"Validation error: {str(e)}")
            return validation
    
    def _calculate_daily_loss(self) -> float:
        """Calculate today's loss percentage"""
        try:
            # Get today's positions
            today = datetime.now().date()
            today_positions = [
                p for p in self.closed_positions 
                if p.get('close_time') and pd.Timestamp(p['close_time']).date() == today
            ]
            
            if not today_positions:
                return 0.0
            
            total_pnl = sum(p.get('pnl', 0) for p in today_positions)
            daily_loss_pct = abs(min(total_pnl, 0)) / self.portfolio_value if self.portfolio_value > 0 else 0
            
            return daily_loss_pct
            
        except Exception as e:
            self.logger.error(f"Daily loss calculation error: {e}")
            return 0.0
    
    def _calculate_weekly_loss(self) -> float:
        """Calculate this week's loss percentage"""
        try:
            # Get this week's positions
            week_start = datetime.now() - timedelta(days=datetime.now().weekday())
            week_positions = [
                p for p in self.closed_positions 
                if p.get('close_time') and pd.Timestamp(p['close_time']) >= week_start
            ]
            
            if not week_positions:
                return 0.0
            
            total_pnl = sum(p.get('pnl', 0) for p in week_positions)
            weekly_loss_pct = abs(min(total_pnl, 0)) / self.portfolio_value if self.portfolio_value > 0 else 0
            
            return weekly_loss_pct
            
        except Exception as e:
            self.logger.error(f"Weekly loss calculation error: {e}")
            return 0.0
    
    def _check_correlation_risk(self, symbol: str, side: str) -> Dict[str, Any]:
        """
        Check correlation risk with existing positions
        
        Returns:
            Dictionary with correlation analysis
        """
        # Simplified correlation check
        # In a real implementation, you would calculate actual correlations
        return {
            'high_risk': False,
            'max_correlation': 0.0,
            'correlated_positions': []
        }
    
    def _calculate_portfolio_concentration(self) -> float:
        """Calculate portfolio concentration (max asset percentage)"""
        if not self.open_positions or self.portfolio_value <= 0:
            return 0.0
        
        # Group positions by asset
        asset_values = {}
        for position in self.open_positions:
            asset = position.get('symbol', 'unknown').split('/')[0]
            value = position.get('value', 0)
            asset_values[asset] = asset_values.get(asset, 0) + value
        
        if not asset_values:
            return 0.0
        
        max_concentration = max(asset_values.values()) / self.portfolio_value
        return max_concentration
    
    def update_portfolio(self, position: Dict[str, Any]):
        """
        Update portfolio with new position
        
        Args:
            position: Position dictionary with details
        """
        try:
            # Add to open positions
            self.open_positions.append(position)
            
            # Update portfolio value (simplified)
            # In real implementation, you'd get actual market values
            
            # Update risk metrics
            self._update_risk_metrics()
            
            self.logger.debug(f"Portfolio updated: {len(self.open_positions)} open positions")
            
        except Exception as e:
            self.logger.error(f"Portfolio update error: {e}")
    
    def close_position(self, position_id: str, exit_price: float, exit_time: datetime = None):
        """
        Close a position and update portfolio
        
        Args:
            position_id: ID of position to close
            exit_price: Exit price
            exit_time: Exit time (defaults to now)
        """
        try:
            # Find position
            position_idx = None
            for i, pos in enumerate(self.open_positions):
                if pos.get('id') == position_id:
                    position_idx = i
                    break
            
            if position_idx is None:
                self.logger.warning(f"Position {position_id} not found")
                return
            
            position = self.open_positions.pop(position_idx)
            
            # Calculate P&L
            entry_price = position.get('entry_price', 0)
            size = position.get('size', 0)
            side = position.get('side', 'buy')
            
            if side == 'buy':
                pnl = size * (exit_price - entry_price)
            else:  # sell (short)
                pnl = size * (entry_price - exit_price)
            
            # Update position
            position['exit_price'] = exit_price
            position['exit_time'] = exit_time or datetime.now()
            position['pnl'] = pnl
            position['pnl_pct'] = pnl / (entry_price * size) * 100 if entry_price * size > 0 else 0
            
            # Add to closed positions
            self.closed_positions.append(position)
            
            # Update portfolio value
            self.portfolio_value += pnl
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Add to returns history
            self.returns_history.append(pnl / self.portfolio_value if self.portfolio_value > 0 else 0)
            
            self.logger.info(
                f"Position closed: {position_id}, P&L: ${pnl:,.2f} ({position['pnl_pct']:.2f}%)"
            )
            
        except Exception as e:
            self.logger.error(f"Close position error: {e}")
    
    def _update_risk_metrics(self):
        """Update risk metrics based on current portfolio"""
        try:
            # Update peak portfolio value
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
            
            # Calculate drawdown
            if self.peak_portfolio_value > 0:
                self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Update risk history
            self.risk_history.append({
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'open_positions': len(self.open_positions)
            })
            
            # Keep history limited
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-500:]
                
        except Exception as e:
            self.logger.error(f"Risk metrics update error: {e}")
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        return {
            'portfolio_value': self.portfolio_value,
            'open_positions': len(self.open_positions),
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_exceeded': self.current_drawdown > self.max_drawdown_limit,
            'daily_loss': self._calculate_daily_loss(),
            'weekly_loss': self._calculate_weekly_loss(),
            'portfolio_concentration': self._calculate_portfolio_concentration(),
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_drawdown': self.max_drawdown_limit,
                'daily_loss_limit': self.daily_loss_limit,
                'weekly_loss_limit': self.weekly_loss_limit
            }
        }
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Sharpe ratio
        """
        if not self.returns_history:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if len(excess_returns) < 2 or np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (only penalizes downside deviation)
        
        Args:
            risk_free_rate: Annual risk-free rate
            target_return: Target return (default 0%)
            
        Returns:
            Sortino ratio
        """
        if not self.returns_history:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - risk_free_rate / 252
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return sortino
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR as percentage of portfolio
        """
        if not self.returns_history:
            return 0.0
        
        returns = np.array(self.returns_history)
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        return abs(var)  # Return positive value
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()
        var_95 = self.calculate_var(0.95)
        
        total_pnl = sum(p.get('pnl', 0) for p in self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p.get('pnl', 0) > 0]
        losing_trades = [p for p in self.closed_positions if p.get('pnl', 0) < 0]
        
        return {
            'portfolio_summary': {
                'initial_capital': self.initial_capital,
                'current_value': self.portfolio_value,
                'total_pnl': total_pnl,
                'total_return_pct': (self.portfolio_value - self.initial_capital) / self.initial_capital * 100,
                'max_drawdown': self.max_drawdown * 100,
                'current_drawdown': self.current_drawdown * 100
            },
            'trade_statistics': {
                'total_trades': len(self.closed_positions),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(self.closed_positions) if self.closed_positions else 0,
                'avg_win': np.mean([p.get('pnl', 0) for p in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([p.get('pnl', 0) for p in losing_trades]) if losing_trades else 0,
                'profit_factor': abs(sum(p.get('pnl', 0) for p in winning_trades) / 
                                   sum(p.get('pnl', 0) for p in losing_trades)) if losing_trades else float('inf')
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'var_95': var_95 * 100,  # as percentage
                'portfolio_concentration': self._calculate_portfolio_concentration() * 100
            },
            'current_status': self.get_current_risk_status()
        }