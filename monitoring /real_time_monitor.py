"""
Real-time Trading Monitor
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """Real-time system monitoring"""
    
    def __init__(self, engine, config: Dict):
        self.engine = engine
        self.config = config
        self.metrics_history = deque(maxlen=1000)
        self.alerts_triggered = []
        
        # Monitoring thresholds
        self.thresholds = config.get('monitoring', {}).get('alerts', {}).get('thresholds', {})
        
    async def start(self):
        """Start monitoring loop"""
        logger.info("Starting real-time monitor")
        
        while True:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                
                # Check alerts
                await self._check_alerts(metrics)
                
                # Log summary
                self._log_summary(metrics)
                
                # Sleep
                await asyncio.sleep(self.config.get('monitoring', {}).get('interval', 60))
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _collect_metrics(self) -> Dict:
        """Collect current system metrics"""
        
        # Get account info
        account = await self.engine.broker.get_account_info()
        
        # Calculate metrics
        metrics = {
            'equity': account.get('equity', 0),
            'cash': account.get('cash', 0),
            'positions_value': account.get('positions_value', 0),
            'open_positions': len(self.engine.positions),
            'pending_orders': len(self.engine.pending_orders),
            'daily_pnl': self._calculate_daily_pnl(),
            'current_drawdown': self._calculate_current_drawdown(),
            'win_rate': self._calculate_win_rate(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'system_health': self._check_system_health()
        }
        
        return metrics
        
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        if len(self.engine.equity_curve) < 2:
            return 0
            
        # Get today's starting equity
        today = datetime.now().date()
        today_start_idx = None
        
        for i, timestamp in enumerate(self.engine.timestamps):
            if timestamp.date() == today:
                today_start_idx = i
                break
                
        if today_start_idx is None or today_start_idx == 0:
            return 0
            
        start_equity = self.engine.equity_curve[today_start_idx - 1]
        current_equity = self.engine.equity_curve[-1]
        
        return current_equity - start_equity
        
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if len(self.engine.equity_curve) < 2:
            return 0
            
        peak = max(self.engine.equity_curve)
        current = self.engine.equity_curve[-1]
        
        return (current - peak) / peak if peak > 0 else 0
        
    def _calculate_win_rate(self) -> float:
        """Calculate recent win rate"""
        if not self.engine.completed_trades:
            return 0
            
        recent_trades = self.engine.completed_trades[-20:]  # Last 20 trades
        
        winning_trades = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        
        return winning_trades / len(recent_trades) if recent_trades else 0
        
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(self.engine.equity_curve) < 20:
            return 0
            
        returns = pd.Series(self.engine.equity_curve).pct_change().dropna()
        
        if len(returns) < 20:
            return 0
            
        # 20-period rolling Sharpe
        mean_return = returns.iloc[-20:].mean()
        std_return = returns.iloc[-20:].std()
        
        if std_return == 0:
            return 0
            
        return mean_return / std_return * np.sqrt(252 * 96)  # Annualized
        
    def _check_system_health(self) -> str:
        """Check overall system health"""
        
        # Check error rates
        # Check connection status
        # Check data freshness
        
        return "HEALTHY"  # Simplified
        
    async def _check_alerts(self, metrics: Dict):
        """Check and trigger alerts"""
        
        # Drawdown alert
        if metrics['current_drawdown'] < -self.thresholds.get('max_drawdown', 0.1):
            await self._trigger_alert(
                'DRAWDOWN',
                f"Drawdown exceeded: {metrics['current_drawdown']:.2%}"
            )
            
        # Daily loss alert
        daily_loss_pct = metrics['daily_pnl'] / metrics['equity'] if metrics['equity'] > 0 else 0
        if daily_loss_pct < -self.thresholds.get('daily_loss', 0.05):
            await self._trigger_alert(
                'DAILY_LOSS',
                f"Daily loss exceeded: {daily_loss_pct:.2%}"
            )
            
        # Position concentration alert
        if metrics['positions_value'] > 0:
            max_position_pct = max(
                abs(pos) / metrics['equity'] 
                for pos in self.engine.positions.values()
            ) if self.engine.positions else 0
            
            if max_position_pct > self.thresholds.get('position_concentration', 0.35):
                await self._trigger_alert(
                    'CONCENTRATION',
                    f"Position concentration high: {max_position_pct:.2%}"
                )
                
    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert"""
        
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now()
        }
        
        self.alerts_triggered.append(alert)
        
        logger.warning(f"ALERT: {alert_type} - {message}")
        
        # Send notifications (email, SMS, webhook)
        # Implementation depends on notification service
        
    def _log_summary(self, metrics: Dict):
        """Log metrics summary"""
        
        logger.info(
            f"Monitor Update - "
            f"Equity: ${metrics['equity']:,.2f} | "
            f"Daily P&L: ${metrics['daily_pnl']:+,.2f} | "
            f"DD: {metrics['current_drawdown']:.2%} | "
            f"Positions: {metrics['open_positions']} | "
            f"Win Rate: {metrics['win_rate']:.1%}"
        )
        
    def get_metrics_history(self) -> List[Dict]:
        """Get metrics history"""
        return list(self.metrics_history)
        
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alerts_triggered
            if alert['timestamp'] > cutoff
        ]