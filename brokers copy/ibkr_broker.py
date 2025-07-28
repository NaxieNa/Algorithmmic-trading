"""
Interactive Brokers Implementation
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
from collections import defaultdict
import numpy as np

from .base_broker import IBrokerInterface
from core.data_structures import Order, OrderStatus, OrderSide, OrderType

logger = logging.getLogger(__name__)

class IBKRBroker(IBrokerInterface):
    """Interactive Brokers implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497)  # 7497 for paper, 7496 for live
        self.client_id = config.get('client_id', 1)
        self.ib = IB()
        
        # Rate limiting
        self.rate_limits = config.get('rate_limits', {})
        self.call_counts = defaultdict(list)
        
        # Connection state
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Connect to TWS or IB Gateway"""
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id
            )
            
            self.is_connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            
            # Request market data type (delayed for paper trading)
            if self.config.get('paper', True):
                self.ib.reqMarketDataType(3)  # Delayed data
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.is_connected = False
            return False
            
    async def disconnect(self) -> bool:
        """Disconnect from IBKR"""
        try:
            if self.is_connected:
                self.ib.disconnect()
                self.is_connected = False
                logger.info("Disconnected from IBKR")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from IBKR: {e}")
            return False
            
    async def get_account_info(self) -> Dict:
        """Get account information"""
        await self._check_rate_limit('account', 1, 5)
        
        try:
            # Get account summary
            account_values = self.ib.accountSummary()
            
            # Parse values
            account_info = {}
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    account_info['equity'] = float(av.value)
                elif av.tag == 'TotalCashValue':
                    account_info['cash'] = float(av.value)
                elif av.tag == 'BuyingPower':
                    account_info['buying_power'] = float(av.value)
                elif av.tag == 'GrossPositionValue':
                    account_info['positions_value'] = float(av.value)
                elif av.tag == 'MaintMarginReq':
                    account_info['margin_used'] = float(av.value)
                    
            # Get positions count
            positions = self.ib.positions()
            account_info['open_positions'] = len(positions)
            
            # Calculate returns
            if 'initial_capital' in self.config:
                initial = self.config['initial_capital']
                current = account_info.get('equity', initial)
                account_info['total_return'] = (current / initial - 1) * 100
                
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
            
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        await self._check_rate_limit('positions', 1, 5)
        
        try:
            positions = self.ib.positions()
            
            position_dict = {}
            for pos in positions:
                symbol = pos.contract.symbol
                quantity = pos.position
                position_dict[symbol] = quantity
                
            return position_dict
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
            
    async def place_order(self, order: Order) -> str:
        """Place an order"""
        await self._check_rate_limit('orders', 50, 60)
        
        try:
            # Create contract
            contract = Stock(order.symbol, 'SMART', 'USD')
            
            # Create IB order
            if order.order_type == OrderType.MARKET:
                ib_order = MarketOrder(
                    'BUY' if order.side == OrderSide.BUY else 'SELL',
                    order.quantity
                )
            elif order.order_type == OrderType.LIMIT:
                ib_order = LimitOrder(
                    'BUY' if order.side == OrderSide.BUY else 'SELL',
                    order.quantity,
                    order.price
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
                
            # Set time in force
            ib_order.tif = order.time_in_force
            
            # Place order
            trade = self.ib.placeOrder(contract, ib_order)
            
            # Wait for order to be acknowledged
            await asyncio.sleep(0.1)
            
            # Log order
            logger.info(f"Placed order: {trade.order.orderId} - {order.symbol} "
                       f"{order.side.value} {order.quantity} @ "
                       f"{'MARKET' if order.order_type == OrderType.MARKET else order.price}")
            
            return str(trade.order.orderId)
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            # Find order
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order: {order_id}")
                    return True
                    
            logger.warning(f"Order not found: {order_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
            
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        try:
            # Find order in trades
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    # Map IBKR status to our status
                    status_map = {
                        'PendingSubmit': OrderStatus.PENDING,
                        'PendingCancel': OrderStatus.PENDING,
                        'PreSubmitted': OrderStatus.SUBMITTED,
                        'Submitted': OrderStatus.SUBMITTED,
                        'Filled': OrderStatus.FILLED,
                        'Cancelled': OrderStatus.CANCELLED,
                        'Inactive': OrderStatus.CANCELLED,
                    }
                    
                    return {
                        'status': status_map.get(trade.orderStatus.status, OrderStatus.PENDING),
                        'filled_qty': trade.orderStatus.filled,
                        'avg_fill_price': trade.orderStatus.avgFillPrice or 0
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
            
    async def get_historical_data(self, symbol: str, period: str = "1 D", 
                                 bar_size: str = "15 mins") -> pd.DataFrame:
        """Get historical market data"""
        await self._check_rate_limit('historical_data', 6, 60)
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Convert period to IB format
            duration = self._convert_period(period)
            
            # Convert bar size to IB format
            ib_bar_size = self._convert_bar_size(bar_size)
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=ib_bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])
            
            # Set index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Add synthetic bid/ask
            spread = 0.0005  # 0.05% spread
            df['bid'] = df['close'] * (1 - spread/2)
            df['ask'] = df['close'] * (1 + spread/2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
            
    async def stream_market_data(self, symbols: List[str], callback: Callable):
        """Stream real-time market data"""
        try:
            contracts = [Stock(symbol, 'SMART', 'USD') for symbol in symbols]
            
            for contract in contracts:
                # Request market data
                self.ib.reqMktData(contract, '', False, False)
                
                # Set up callback
                def on_tick(ticker):
                    data = {
                        'symbol': ticker.contract.symbol,
                        'bid': ticker.bid or 0,
                        'ask': ticker.ask or 0,
                        'last': ticker.last or 0,
                        'volume': ticker.volume or 0,
                        'timestamp': datetime.now()
                    }
                    callback(data)
                
                # Register callback
                ticker = self.ib.ticker(contract)
                ticker.updateEvent += on_tick
                
            logger.info(f"Streaming market data for {symbols}")
            
        except Exception as e:
            logger.error(f"Error streaming market data: {e}")
            
    async def _check_rate_limit(self, endpoint: str, max_calls: int, period: int):
        """Check and enforce rate limits"""
        if endpoint not in self.rate_limits:
            return
            
        now = datetime.now()
        cutoff = now - timedelta(seconds=period)
        
        # Remove old calls
        self.call_counts[endpoint] = [
            call_time for call_time in self.call_counts[endpoint]
            if call_time > cutoff
        ]
        
        # Check limit
        if len(self.call_counts[endpoint]) >= max_calls:
            wait_time = (self.call_counts[endpoint][0] + timedelta(seconds=period) - now).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {endpoint}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
        # Record call
        self.call_counts[endpoint].append(now)
        
    def _convert_period(self, period: str) -> str:
        """Convert period to IB format"""
        # Parse period like "30 D" or "1 M"
        parts = period.split()
        if len(parts) != 2:
            return "1 D"
            
        value, unit = parts
        
        # IB duration format
        unit_map = {
            'D': 'D',
            'W': 'W',
            'M': 'M',
            'Y': 'Y'
        }
        
        ib_unit = unit_map.get(unit.upper(), 'D')
        return f"{value} {ib_unit}"
        
    def _convert_bar_size(self, bar_size: str) -> str:
        """Convert bar size to IB format"""
        bar_size_map = {
            '1 min': '1 min',
            '5 mins': '5 mins',
            '15 mins': '15 mins',
            '30 mins': '30 mins',
            '1 hour': '1 hour',
            '1 day': '1 day'
        }
        
        return bar_size_map.get(bar_size, '15 mins')