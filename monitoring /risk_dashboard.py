"""
Complete Risk Monitoring Dashboard
"""

import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import asyncio
from datetime import datetime, timedelta
import numpy as np

class RiskDashboard:
    """Real-time risk monitoring dashboard"""
    
    def __init__(self, trading_engine):
        self.engine = trading_engine
        self.refresh_interval = 5
        
    def render(self):
        """Render the complete dashboard"""
        st.set_page_config(
            page_title="Algorithmic Trading Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        # Header
        st.title("🚀 Algorithmic Trading Risk Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Controls")
            
            # Refresh button
            if st.button("🔄 Refresh Now"):
                st.experimental_rerun()
                
            # Auto-refresh
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            if auto_refresh:
                self.refresh_interval = st.slider(
                    "Refresh Interval (seconds)", 
                    min_value=1, 
                    max_value=60, 
                    value=5
                )
                
            # Strategy filter
            selected_strategies = st.multiselect(
                "Strategies",
                options=list(self.engine.strategies.keys()),
                default=list(self.engine.strategies.keys())
            )
            
            # Time range
            time_range = st.selectbox(
                "Time Range",
                options=["1H", "1D", "1W", "1M", "3M", "YTD", "ALL"],
                index=1
            )
        
        # Main content
        self._render_account_summary()
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            self._render_positions()
        with col2:
            self._render_pending_orders()
            
        st.markdown("---")
        
        self._render_performance_charts(time_range)
        st.markdown("---")
        
        self._render_risk_metrics()
        st.markdown("---")
        
        self._render_strategy_performance(selected_strategies)
        st.markdown("---")
        
        self._render_trade_log()
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(self.refresh_interval)
            st.experimental_rerun()
    
    def _render_account_summary(self):
        """Render account summary metrics"""
        st.header("💰 Account Summary")
        
        # Get account info
        account_info = asyncio.run(self.engine.broker.get_account_info())
        
        # Calculate metrics
        equity = account_info.get('equity', 0)
        initial_capital = self.engine.config.get('initial_capital', 100000)
        total_return = (equity / initial_capital - 1) * 100
        
        # Calculate daily P&L
        if len(self.engine.equity_curve) > 1:
            daily_pnl = equity - self.engine.equity_curve[-2]
            daily_return = (daily_pnl / self.engine.equity_curve[-2]) * 100
        else:
            daily_pnl = 0
            daily_return = 0
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${equity:,.2f}",
                f"{daily_pnl:+,.2f} ({daily_return:+.2f}%)"
            )
            
        with col2:
            st.metric(
                "Total Return",
                f"{total_return:.2f}%",
                f"{daily_return:+.2f}% today"
            )
            
        with col3:
            # Calculate Sharpe ratio
            if len(self.engine.equity_curve) > 2:
                returns = pd.Series(self.engine.equity_curve).pct_change().dropna()
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe = 0
            st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            
        with col4:
            # Calculate max drawdown
            if len(self.engine.equity_curve) > 1:
                peak = pd.Series(self.engine.equity_curve).expanding().max()
                dd = (pd.Series(self.engine.equity_curve) - peak) / peak * 100
                max_dd = dd.min()
            else:
                max_dd = 0
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
            
        with col5:
            st.metric(
                "Active Positions",
                len(self.engine.positions)
            )
    
    def _render_positions(self):
        """Render current positions"""
        st.subheader("📊 Current Positions")
        account_info = asyncio.run(self.engine.broker.get_account_info())
        
        if self.engine.positions:
            positions_data = []
            
            for symbol, quantity in self.engine.positions.items():
                # Get current price (simplified)
                current_price = 100  # In real implementation, get from market data
                
                positions_data.append({
                    "Symbol": symbol,
                    "Quantity": int(quantity),
                    "Current Price": f"${current_price:.2f}",
                    "Market Value": f"${quantity * current_price:,.2f}",
                    "Weight": f"{(quantity * current_price / account_info.get('equity', 1)) * 100:.1f}%"
                })
            
            df = pd.DataFrame(positions_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No open positions")
    
    def _render_pending_orders(self):
        """Render pending orders"""
        st.subheader("📝 Pending Orders")
        
        if self.engine.pending_orders:
            orders_data = []
            
            for order_id, order_info in self.engine.pending_orders.items():
                order = order_info['order']
                orders_data.append({
                    "Order ID": order_id[:8] + "...",
                    "Symbol": order.symbol,
                    "Side": order.side.value.upper(),
                    "Quantity": order.quantity,
                    "Type": order.order_type.value,
                    "Price": f"${order.price:.2f}" if order.price else "MARKET",
                    "Strategy": order_info['strategy']
                })
            
            df = pd.DataFrame(orders_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No pending orders")
    
    def _render_performance_charts(self, time_range: str):
        """Render performance charts"""
        st.header("📈 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity curve
            st.subheader("Equity Curve")
            
            if len(self.engine.equity_curve) > 1:
                fig = go.Figure()
                
                # Create time index
                end_time = datetime.now()
                time_index = pd.date_range(
                    end=end_time,
                    periods=len(self.engine.equity_curve),
                    freq='15min'
                )
                
                fig.add_trace(go.Scatter(
                    x=time_index,
                    y=self.engine.equity_curve,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for equity curve")
        
        with col2:
            # Drawdown chart
            st.subheader("Drawdown Analysis")
            
            if len(self.engine.equity_curve) > 1:
                equity_series = pd.Series(self.engine.equity_curve)
                peak = equity_series.expanding().max()
                drawdown = (equity_series - peak) / peak * 100
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=time_index,
                    y=drawdown,
                    fill='tozeroy',
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)'
                ))
                
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Drawdown (%)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for drawdown")
    
    def _render_risk_metrics(self):
        """Render risk metrics"""
        st.header("⚠️ Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Portfolio risk metrics
        portfolio_risk = self.engine.risk_manager.check_portfolio_risk(
            {symbol: {'size': abs(qty)} for symbol, qty in self.engine.positions.items()}
        )
        
        with col1:
            st.metric(
                "Total Exposure",
                f"{portfolio_risk['total_risk']:.1%}"
            )
            
        with col2:
            st.metric(
                "Concentration Risk",
                f"{portfolio_risk['concentration_risk']:.1%}"
            )
            
        with col3:
            # Current volatility
            if len(self.engine.equity_curve) > 20:
                returns = pd.Series(self.engine.equity_curve).pct_change().dropna()
                current_vol = returns.iloc[-20:].std() * np.sqrt(252 * 96)
            else:
                current_vol = 0
            st.metric(
                "Current Volatility",
                f"{current_vol*100:.1f}%"
            )
            
        with col4:
            st.metric(
                "Market Regime",
                self.engine.risk_manager.market_regime.upper()
            )
    
    def _render_strategy_performance(self, selected_strategies: List[str]):
        """Render strategy performance comparison"""
        st.header("🎯 Strategy Performance")
        
        if not selected_strategies:
            st.warning("No strategies selected")
            return
        
        # Create comparison data
        strategy_data = []
        
        for strategy_name in selected_strategies:
            if strategy_name in self.engine.strategies:
                strategy = self.engine.strategies[strategy_name]
                
                strategy_data.append({
                    "Strategy": strategy_name,
                    "Total Trades": strategy.performance_stats['total_trades'],
                    "Winning Trades": strategy.performance_stats['winning_trades'],
                    "Win Rate": f"{(strategy.performance_stats['winning_trades'] / max(strategy.performance_stats['total_trades'], 1)) * 100:.1f}%",
                    "Total P&L": f"${strategy.performance_stats['total_pnl']:,.2f}",
                    "Avg P&L": f"${strategy.performance_stats['total_pnl'] / max(strategy.performance_stats['total_trades'], 1):,.2f}"
                })
        
        if strategy_data:
            df = pd.DataFrame(strategy_data)
            st.dataframe(df, use_container_width=True)
            
            # Strategy comparison chart
            fig = px.bar(
                df,
                x='Strategy',
                y='Total Trades',
                title='Trades by Strategy',
                color='Strategy'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_trade_log(self):
        """Render recent trades"""
        st.header("📋 Recent Trades")
        
        if self.engine.completed_trades:
            # Get last 20 trades
            recent_trades = self.engine.completed_trades[-20:]
            
            trades_data = []
            for trade in recent_trades:
                trades_data.append({
                    "Time": trade.get('timestamp', datetime.now()).strftime("%Y-%m-%d %H:%M"),
                    "Symbol": trade.get('symbol', 'N/A'),
                    "Side": trade.get('side', 'N/A'),
                    "Quantity": trade.get('quantity', 0),
                    "Price": f"${trade.get('price', 0):.2f}",
                    "P&L": f"${trade.get('pnl', 0):,.2f}",
                    "Strategy": trade.get('strategy', 'N/A')
                })
            
            df = pd.DataFrame(trades_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No completed trades yet")

def run_dashboard(trading_engine):
    """Run the dashboard application"""
    dashboard = RiskDashboard(trading_engine)
    dashboard.render()

if __name__ == "__main__":
    st.write("This dashboard requires a running trading engine instance.")
    st.write("Please run from the main application.")