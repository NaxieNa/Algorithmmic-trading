"""
Complete Backtest Broker Implementation
"""

import asyncio
from .base_broker import IBrokerInterface
from core.data_structures import Order, OrderStatus, OrderSide, OrderType
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BacktestBroker(IBrokerInterface):
    """Complete backtest broker for strategy testing"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.commission_rate = commission
        
        # Order tracking
        self.orders = {}
        self.order_counter = 0
        self.pending_orders = []
        
        # Trade history
        self.trades = []
        self.trade_counter = 0
        
        # Market data
        self.current_prices = {}
        self.market_data = {}
        
        # Performance tracking
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]
        
        # Slippage model
        self.slippage_pct = 0.0005  # 0.05% slippage
        
        logger.info(f"Backtest broker initialized with ${initial_capital:,.2f}")
    
    async def connect(self):
        """Connect to backtest environment"""
        logger.info("Connected to backtest broker")
        return True
    
    async def disconnect(self):
        """Disconnect from backtest environment"""
        logger.info("Disconnected from backtest broker")
        return True
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        # Calculate current portfolio value
        portfolio_value = self.cash
        
        for symbol, quantity in self.positions.items():
            if symbol in self.current_prices:
                portfolio_value += quantity * self.current_prices[symbol]
        
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        
        return {
            'equity': portfolio_value,
            'cash': self.cash,
            'buying_power': self.cash,  # Simplified - no margin
            'positions_value': portfolio_value - self.cash,
            'initial_capital': self.initial_capital,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'open_positions': len(self.positions)
        }
    
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.positions.copy()
    
    async def place_order(self, order: Order) -> str:
        """Place an order in backtest"""
        # Generate order ID
        self.order_counter += 1
        order_id = f"BT_ORDER_{self.order_counter:06d}"
        
        # Validate order
        if order.quantity <= 0:
            raise ValueError(f"Invalid order quantity: {order.quantity}")
        
        # Check if we have price data
        if order.symbol not in self.current_prices:
            raise ValueError(f"No price data for {order.symbol}")
        
        current_price = self.current_prices[order.symbol]
        
        # Apply slippage
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                execution_price = current_price * (1 + self.slippage_pct)
            else:
                execution_price = current_price * (1 - self.slippage_pct)
        else:
            # For limit orders, check if executable
            if order.side == OrderSide.BUY and order.price < current_price:
                # Buy limit below market - will execute at limit price
                execution_price = order.price
            elif order.side == OrderSide.SELL and order.price > current_price:
                # Sell limit above market - will execute at limit price
                execution_price = order.price
            else:
                # Limit order not immediately executable
                self.pending_orders.append({
                    'order_id': order_id,
                    'order': order,
                    'status': OrderStatus.PENDING
                })
                
                self.orders[order_id] = {
                    'order': order,
                    'status': OrderStatus.PENDING,
                    'filled_quantity': 0,
                    'average_fill_price': 0,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Limit order {order_id} placed but not filled")
                return order_id
        
        # Execute order
        filled = await self._execute_order(order, execution_price)
        
        if filled:
            self.orders[order_id] = {
                'order': order,
                'status': OrderStatus.FILLED,
                'filled_quantity': order.quantity,
                'average_fill_price': execution_price,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Order {order_id} filled: {order.symbol} {order.side.value} "
                       f"{order.quantity} @ ${execution_price:.2f}")
        else:
            self.orders[order_id] = {
                'order': order,
                'status': OrderStatus.REJECTED,
                'filled_quantity': 0,
                'average_fill_price': 0,
                'timestamp': datetime.now()
            }
            
            logger.warning(f"Order {order_id} rejected")
        
        return order_id
    
    async def _execute_order(self, order: Order, price: float) -> bool:
        """Execute an order internally"""
        # Calculate order value
        order_value = order.quantity * price
        commission = order_value * self.commission_rate
        total_cost = order_value + commission
        
        # Check if we can execute
        if order.side == OrderSide.BUY:
            if self.cash < total_cost:
                logger.warning(f"Insufficient cash for buy order: "
                             f"${self.cash:.2f} < ${total_cost:.2f}")
                return False
            
            # Execute buy
            self.cash -= total_cost
            current_position = self.positions.get(order.symbol, 0)
            self.positions[order.symbol] = current_position + order.quantity
            
        else:  # SELL
            current_position = self.positions.get(order.symbol, 0)
            
            if current_position < order.quantity:
                logger.warning(f"Insufficient position for sell order: "
                             f"{current_position} < {order.quantity}")
                return False
            
            # Execute sell
            self.cash += order_value - commission
            self.positions[order.symbol] = current_position - order.quantity
            
            # Remove position if zero
            if self.positions[order.symbol] == 0:
                del self.positions[order.symbol]
        
        # Record trade
        self.trade_counter += 1
        trade = {
            'trade_id': f"BT_TRADE_{self.trade_counter:06d}",
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': price,
            'commission': commission,
            'timestamp': datetime.now(),
            'order_type': order.order_type.value
        }
        
        # Calculate P&L for closing trades
        if order.side == OrderSide.SELL:
            # Simple FIFO P&L calculation (simplified)
            trade['pnl'] = order_value - commission  # Simplified
            trade['return'] = 0  # Would need position tracking for accurate calculation
        else:
            trade['pnl'] = 0
            trade['return'] = 0
        
        self.trades.append(trade)
        
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        # Find in pending orders
        for i, pending in enumerate(self.pending_orders):
            if pending['order_id'] == order_id:
                self.pending_orders.pop(i)
                
                # Update order status
                if order_id in self.orders:
                    self.orders[order_id]['status'] = OrderStatus.CANCELLED
                
                logger.info(f"Order {order_id} cancelled")
                return True
        
        logger.warning(f"Order {order_id} not found or already filled")
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        if order_id in self.orders:
            order_info = self.orders[order_id]
            return {
                'status': order_info['status'],
                'filled_qty': order_info['filled_quantity'],
                'avg_fill_price': order_info['average_fill_price']
            }
        return None
    
    async def get_historical_data(self, symbol: str, period: str = "1 D", 
                                 bar_size: str = "15 mins") -> pd.DataFrame:
        """Get historical data from loaded market data"""
        if symbol in self.market_data:
            return self.market_data[symbol]
        
        # Return empty DataFrame if no data
        return pd.DataFrame()
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current market prices"""
        self.current_prices.update(prices)
        
        # Update equity curve
        account_info = asyncio.run(self.get_account_info())
        self.equity_curve.append(account_info['equity'])
        self.timestamps.append(datetime.now())
        
        # Check pending orders
        self._check_pending_orders()
    
    def _check_pending_orders(self):
        """Check if any pending orders can be filled"""
        filled_orders = []
        
        for i, pending in enumerate(self.pending_orders):
            order = pending['order']
            
            if order.symbol not in self.current_prices:
                continue
            
            current_price = self.current_prices[order.symbol]
            
            # Check if limit order can be filled
            can_fill = False
            
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    can_fill = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    can_fill = True
            
            if can_fill:
                # Execute order
                execution_price = order.price  # Limit orders execute at limit price
                
                if asyncio.run(self._execute_order(order, execution_price)):
                    # Update order status
                    order_id = pending['order_id']
                    if order_id in self.orders:
                        self.orders[order_id]['status'] = OrderStatus.FILLED
                        self.orders[order_id]['filled_quantity'] = order.quantity
                        self.orders[order_id]['average_fill_price'] = execution_price
                    
                    filled_orders.append(i)
                    
                    logger.info(f"Pending order filled: {order.symbol} "
                               f"{order.side.value} {order.quantity} @ ${execution_price:.2f}")
        
        # Remove filled orders from pending list
        for i in reversed(filled_orders):
            self.pending_orders.pop(i)
    
    def load_market_data(self, symbol: str, data: pd.DataFrame):
        """Load historical market data for backtesting"""
        self.market_data[symbol] = data
        
        # Set initial price
        if not data.empty and 'close' in data.columns:
            self.current_prices[symbol] = data['close'].iloc[-1]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if len(self.equity_curve) < 2:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Calculate metrics
        total_return = (equity[-1] / equity[0] - 1) * 100
        
        if len(returns) > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_drawdown = drawdown.min() * 100
        else:
            sharpe = 0
            max_drawdown = 0
        
        # Win rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate * 100,
            'final_equity': equity[-1],
            'trades': self.trades
        }
    
    def reset(self):
        """Reset broker state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
        self.pending_orders = []
        self.trades = []
        self.trade_counter = 0
        self.current_prices = {}
        self.equity_curve = [self.initial_capital]
        self.timestamps = [datetime.now()]
        
        logger.info("Backtest broker reset")