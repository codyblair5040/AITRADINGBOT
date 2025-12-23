"""
Monitoring Dashboard - Real-time performance tracking and visualization
Provides web dashboard, logging, and alerting capabilities
"""
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
import asyncio
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Try to import FastAPI for web dashboard (optional)
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Monitoring dashboard will be limited.")

@dataclass
class TradeMetrics:
    """Data class for trade metrics"""
    timestamp: str
    symbol: str
    side: str
    size: float
    price: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = "open"
    confidence: float = 0.0

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    timestamp: str
    portfolio_value: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    open_positions: int = 0
    daily_return: float = 0.0

class MonitoringDashboard:
    def __init__(self, config: Optional[Dict] = None, engine=None):
        """
        Initialize Monitoring Dashboard
        
        Args:
            config: Configuration dictionary
            engine: Reference to main trading engine
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.engine = engine
        
        # Dashboard settings
        self.dashboard_enabled = self.config.get('dashboard_enabled', True)
        self.dashboard_port = self.config.get('dashboard_port', 8080)
        self.dashboard_host = self.config.get('dashboard_host', '0.0.0.0')
        
        # Data storage
        self.trades_history = []
        self.performance_history = []
        self.market_data_history = []
        self.alerts_history = []
        
        # WebSocket connections for real-time updates
        self.websocket_connections = []
        
        # Alert settings
        self.alerts_enabled = self.config.get('alerts_enabled', True)
        self.alert_thresholds = {
            'profit': self.config.get('alert_profit_percent', 5.0),
            'loss': self.config.get('alert_loss_percent', 3.0),
            'drawdown': self.config.get('alert_drawdown_percent', 10.0)
        }
        
        # Start background tasks
        self.running = False
        self.background_thread = None
        
        self.logger.info("Monitoring Dashboard initialized")
    
    def start(self):
        """Start monitoring dashboard and background tasks"""
        if self.running:
            self.logger.warning("Dashboard already running")
            return
        
        self.running = True
        
        # Start background metrics collection
        self.background_thread = threading.Thread(target=self._background_collector, daemon=True)
        self.background_thread.start()
        
        # Start web dashboard if enabled and FastAPI is available
        if self.dashboard_enabled and FASTAPI_AVAILABLE:
            self._start_web_dashboard()
        elif self.dashboard_enabled:
            self.logger.warning("FastAPI not installed. Web dashboard disabled.")
            self.logger.info("Install with: pip install fastapi uvicorn")
        
        self.logger.info(f"Monitoring Dashboard started on port {self.dashboard_port}")
    
    def stop(self):
        """Stop monitoring dashboard"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        self.logger.info("Monitoring Dashboard stopped")
    
    def _background_collector(self):
        """Background thread for collecting metrics"""
        self.logger.info("Background metrics collector started")
        
        while self.running:
            try:
                # Collect metrics every 10 seconds
                time.sleep(10)
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Broadcast updates to WebSocket clients
                self._broadcast_updates()
                
            except Exception as e:
                self.logger.error(f"Background collector error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system and performance metrics"""
        try:
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                portfolio_value=10000,  # Default, should come from engine
                total_pnl=0.0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                open_positions=0,
                daily_return=0.0
            )
            
            # Try to get metrics from engine if available
            if self.engine and hasattr(self.engine, 'performance_metrics'):
                engine_metrics = self.engine.performance_metrics
                if engine_metrics:
                    metrics.portfolio_value = engine_metrics.get('portfolio_value', 10000)
                    metrics.total_pnl = engine_metrics.get('total_pnl', 0.0)
                    metrics.win_rate = engine_metrics.get('win_rate', 0.0)
            
            # Add to history
            self.performance_history.append(asdict(metrics))
            
            # Keep history limited
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
                
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
    
    def update(self, trades: List[Dict], metrics: Dict, market_data: pd.DataFrame):
        """
        Update dashboard with latest data
        
        Args:
            trades: List of recent trades
            metrics: Performance metrics
            market_data: Recent market data
        """
        try:
            # Update trades history
            for trade in trades[-10:]:  # Last 10 trades
                if trade not in [t.get('id') for t in self.trades_history[-50:]]:
                    trade_metrics = TradeMetrics(
                        timestamp=trade.get('timestamp', datetime.now().isoformat()),
                        symbol=trade.get('symbol', 'unknown'),
                        side=trade.get('side', 'unknown'),
                        size=trade.get('amount', 0),
                        price=trade.get('price', 0),
                        pnl=trade.get('pnl', 0),
                        pnl_percent=trade.get('pnl_percent', 0),
                        status=trade.get('status', 'filled'),
                        confidence=trade.get('confidence', 0.0)
                    )
                    self.trades_history.append(asdict(trade_metrics))
            
            # Update performance metrics
            if metrics:
                perf_metrics = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    portfolio_value=metrics.get('portfolio_value', 10000),
                    total_pnl=metrics.get('total_pnl', 0.0),
                    win_rate=metrics.get('win_rate', 0.0),
                    sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                    sortino_ratio=metrics.get('sortino_ratio', 0.0),
                    max_drawdown=metrics.get('max_drawdown', 0.0),
                    current_drawdown=metrics.get('current_drawdown', 0.0),
                    open_positions=metrics.get('open_positions', 0),
                    daily_return=metrics.get('daily_return', 0.0)
                )
                self.performance_history.append(asdict(perf_metrics))
            
            # Update market data history
            if not market_data.empty:
                latest_data = market_data.iloc[-1:].copy()
                latest_data['timestamp'] = datetime.now().isoformat()
                self.market_data_history.append(latest_data.to_dict('records')[0])
            
            # Keep histories limited
            self.trades_history = self.trades_history[-100:]
            self.market_data_history = self.market_data_history[-100:]
            self.performance_history = self.performance_history[-100:]
            
            # Log update
            self.logger.debug(f"Dashboard updated: {len(trades)} trades, {len(metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Dashboard update error: {e}")
    
    def _check_alerts(self):
        """Check conditions and trigger alerts"""
        if not self.alerts_enabled:
            return
        
        try:
            # Check profit alerts
            if self.trades_history:
                latest_trade = self.trades_history[-1]
                pnl_percent = latest_trade.get('pnl_percent', 0)
                
                if pnl_percent >= self.alert_thresholds['profit']:
                    self._trigger_alert(
                        type="PROFIT",
                        message=f"Trade reached {pnl_percent:.1f}% profit!",
                        data=latest_trade
                    )
                
                elif pnl_percent <= -self.alert_thresholds['loss']:
                    self._trigger_alert(
                        type="LOSS",
                        message=f"Trade lost {abs(pnl_percent):.1f}%!",
                        data=latest_trade
                    )
            
            # Check drawdown alerts
            if self.performance_history:
                latest_perf = self.performance_history[-1]
                current_drawdown = latest_perf.get('current_drawdown', 0) * 100
                
                if current_drawdown >= self.alert_thresholds['drawdown']:
                    self._trigger_alert(
                        type="DRAWDOWN",
                        message=f"Portfolio drawdown reached {current_drawdown:.1f}%!",
                        data=latest_perf
                    )
                    
        except Exception as e:
            self.logger.error(f"Alert check error: {e}")
    
    def _trigger_alert(self, type: str, message: str, data: Dict):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': type,
            'message': message,
            'data': data,
            'acknowledged': False
        }
        
        self.alerts_history.append(alert)
        self.logger.warning(f"ALERT: {type} - {message}")
        
        # Keep alerts history limited
        self.alerts_history = self.alerts_history[-50:]
        
        # Send email alert if configured
        self._send_email_alert(alert)
        
        # Broadcast alert to WebSocket clients
        self._broadcast_alert(alert)
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert (if configured)"""
        # This would integrate with your email system
        # For now, just log it
        self.logger.info(f"Email alert would be sent: {alert['type']} - {alert['message']}")
    
    def _broadcast_alert(self, alert: Dict):
        """Broadcast alert to WebSocket clients"""
        try:
            for connection in self.websocket_connections:
                try:
                    asyncio.run(connection.send_json({
                        'type': 'alert',
                        'data': alert
                    }))
                except:
                    # Remove disconnected clients
                    self.websocket_connections.remove(connection)
        except Exception as e:
            self.logger.error(f"Alert broadcast error: {e}")
    
    def _broadcast_updates(self):
        """Broadcast updates to WebSocket clients"""
        try:
            if not self.websocket_connections:
                return
            
            update_data = {
                'type': 'update',
                'timestamp': datetime.now().isoformat(),
                'trades_count': len(self.trades_history),
                'performance_count': len(self.performance_history),
                'alerts_count': len([a for a in self.alerts_history if not a['acknowledged']])
            }
            
            for connection in self.websocket_connections:
                try:
                    asyncio.run(connection.send_json(update_data))
                except:
                    # Remove disconnected clients
                    self.websocket_connections.remove(connection)
                    
        except Exception as e:
            self.logger.error(f"Update broadcast error: {e}")
    
    def _start_web_dashboard(self):
        """Start FastAPI web dashboard in background"""
        if not FASTAPI_AVAILABLE:
            return
        
        app = FastAPI(title="Enterprise Trading Bot Dashboard")
        
        # Store app reference
        self.fastapi_app = app
        
        # HTML dashboard
        html_dashboard = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enterprise Trading Bot Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; background: #f9f9f9; }
                .card h3 { margin-top: 0; color: #333; }
                .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
                .metric { padding: 5px; background: white; border-radius: 3px; }
                .alert { color: red; font-weight: bold; }
                .profit { color: green; }
                .loss { color: red; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>🤖 Enterprise Trading Bot Dashboard</h1>
            <div id="dashboard" class="dashboard">
                <!-- Content will be loaded by JavaScript -->
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                let dashboardData = {};
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    dashboardData = { ...dashboardData, ...data };
                    updateDashboard();
                };
                
                function updateDashboard() {
                    const dashboard = document.getElementById('dashboard');
                    dashboard.innerHTML = `
                        <div class="card">
                            <h3>📊 Performance Overview</h3>
                            <div class="metrics">
                                <div class="metric">Portfolio Value: $${(dashboardData.portfolio_value || 10000).toFixed(2)}</div>
                                <div class="metric">Total P&L: $${(dashboardData.total_pnl || 0).toFixed(2)}</div>
                                <div class="metric">Win Rate: ${((dashboardData.win_rate || 0) * 100).toFixed(1)}%</div>
                                <div class="metric">Open Positions: ${dashboardData.open_positions || 0}</div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>⚠️ Alerts</h3>
                            <div id="alerts">
                                ${dashboardData.alerts_count ? 
                                    `<div class="alert">${dashboardData.alerts_count} unacknowledged alerts</div>` :
                                    '<div>No active alerts</div>'
                                }
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>📈 Recent Trades</h3>
                            <div id="trades">
                                Loading trades...
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>⚙️ System Status</h3>
                            <div class="metrics">
                                <div class="metric">Last Update: ${dashboardData.timestamp || 'N/A'}</div>
                                <div class="metric">Trades: ${dashboardData.trades_count || 0}</div>
                                <div class="metric">Uptime: Calculating...</div>
                                <div class="metric">Status: <span style="color: green;">● Running</span></div>
                            </div>
                        </div>
                    `;
                }
                
                // Initial update
                updateDashboard();
                
                // Request initial data
                ws.onopen = () => {
                    ws.send(JSON.stringify({ type: 'get_data' }));
                };
            </script>
        </body>
        </html>
        """
        
        @app.get("/")
        async def get_dashboard():
            return HTMLResponse(html_dashboard)
        
        @app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @app.get("/api/metrics")
        async def get_metrics():
            return {
                "trades": self.trades_history[-10:],
                "performance": self.performance_history[-5:],
                "alerts": [a for a in self.alerts_history[-5:] if not a['acknowledged']],
                "market_data": self.market_data_history[-5:]
            }
        
        @app.get("/api/trades")
        async def get_trades(limit: int = 20):
            return self.trades_history[-limit:]
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    data = await websocket.receive_json()
                    
                    if data.get('type') == 'get_data':
                        await websocket.send_json({
                            'trades_count': len(self.trades_history),
                            'performance_count': len(self.performance_history),
                            'alerts_count': len([a for a in self.alerts_history if not a['acknowledged']]),
                            'timestamp': datetime.now().isoformat()
                        })
                        
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
        
        # Start FastAPI server in background thread
        def run_server():
            uvicorn.run(app, host=self.dashboard_host, port=self.dashboard_port, log_level="warning")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        self.logger.info(f"Web dashboard available at http://{self.dashboard_host}:{self.dashboard_port}")
    
    def generate_report(self, report_type: str = "daily") -> Dict:
        """
        Generate performance report
        
        Args:
            report_type: 'daily', 'weekly', or 'monthly'
            
        Returns:
            Report dictionary
        """
        try:
            now = datetime.now()
            
            if report_type == "daily":
                start_time = now - timedelta(days=1)
                period_name = "Daily"
            elif report_type == "weekly":
                start_time = now - timedelta(days=7)
                period_name = "Weekly"
            elif report_type == "monthly":
                start_time = now - timedelta(days=30)
                period_name = "Monthly"
            else:
                start_time = now - timedelta(days=1)
                period_name = "Daily"
            
            # Filter data for period
            period_trades = [
                t for t in self.trades_history
                if datetime.fromisoformat(t['timestamp']) >= start_time
            ]
            
            period_performance = [
                p for p in self.performance_history
                if datetime.fromisoformat(p['timestamp']) >= start_time
            ]
            
            # Calculate metrics
            if period_trades:
                winning_trades = [t for t in period_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in period_trades if t.get('pnl', 0) < 0]
                
                win_rate = len(winning_trades) / len(period_trades) if period_trades else 0
                total_pnl = sum(t.get('pnl', 0) for t in period_trades)
                avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
                avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            else:
                win_rate = total_pnl = avg_win = avg_loss = 0
            
            report = {
                'report_type': report_type,
                'period_name': period_name,
                'generated_at': now.isoformat(),
                'period_start': start_time.isoformat(),
                'period_end': now.isoformat(),
                'summary': {
                    'total_trades': len(period_trades),
                    'winning_trades': len(winning_trades) if period_trades else 0,
                    'losing_trades': len(losing_trades) if period_trades else 0,
                    'win_rate': win_rate * 100,
                    'total_pnl': total_pnl,
                    'average_win': avg_win,
                    'average_loss': avg_loss,
                    'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
                },
                'recent_trades': period_trades[-5:],
                'alerts': self.alerts_history[-10:],
                'recommendations': self._generate_recommendations(period_trades, period_performance)
            }
            
            self.logger.info(f"{period_name} report generated: {len(period_trades)} trades, P&L: ${total_pnl:,.2f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, trades: List, performance: List) -> List[str]:
        """Generate trading recommendations based on performance"""
        recommendations = []
        
        try:
            if not trades:
                recommendations.append("No trading activity in this period.")
                return recommendations
            
            # Analyze win rate
            win_rate = len([t for t in trades if t.get('pnl', 0) > 0]) / len(trades)
            
            if win_rate < 0.4:
                recommendations.append("⚠️ Low win rate detected. Consider reviewing strategy or increasing risk management.")
            elif win_rate > 0.6:
                recommendations.append("✅ High win rate! Current strategy appears effective.")
            
            # Check average profit vs loss
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            if winning_trades and losing_trades:
                avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades)
                avg_loss = abs(sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades))
                
                if avg_win < avg_loss:
                    recommendations.append("⚠️ Average win smaller than average loss. Consider adjusting stop-loss/take-profit levels.")
            
            # Check trading frequency
            if len(trades) > 50:
                recommendations.append("⚠️ High trading frequency detected. Ensure trades are based on strong signals.")
            elif len(trades) < 5:
                recommendations.append("ℹ️ Low trading activity. Consider reviewing signal thresholds.")
            
            # Check drawdown
            if performance:
                latest_drawdown = performance[-1].get('current_drawdown', 0) * 100
                if latest_drawdown > 10:
                    recommendations.append(f"⚠️ High drawdown: {latest_drawdown:.1f}%. Consider reducing position sizes.")
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get current dashboard status"""
        return {
            'running': self.running,
            'dashboard_enabled': self.dashboard_enabled and FASTAPI_AVAILABLE,
            'data_counts': {
                'trades': len(self.trades_history),
                'performance': len(self.performance_history),
                'alerts': len(self.alerts_history),
                'market_data': len(self.market_data_history)
            },
            'active_alerts': len([a for a in self.alerts_history if not a['acknowledged']]),
            'websocket_clients': len(self.websocket_connections),
            'uptime': 'N/A',  # Would track actual uptime
            'last_update': datetime.now().isoformat()
        }