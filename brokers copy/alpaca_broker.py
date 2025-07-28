"""
Alpaca Broker Implementation
"""

from .base_broker import IBrokerInterface
from typing import Dict, Optional, List
import pandas as pd
import aiohttp
import asyncio
import logging
from core.data_structures import Order, OrderStatus

logger = logging.getLogger(__name__)

class AlpacaBroker(IBrokerInterface):
    """Alpaca broker implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        self.data_url = config.get('data_url', 'https://data.alpaca.markets')
        self.session = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to Alpaca API"""
        self.session = aiohttp.ClientSession(headers={
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        })
        
        # Test connection
        async with self.session.get(f"{self.base_url}/v2/account") as resp:
            if resp.status == 200:
                self.is_connected = True
                logger.info("Connected to Alpaca API")
            else:
                raise ConnectionError(f"Failed to connect to Alpaca: {resp.status}")
                
    async def disconnect(self):
        """Disconnect from Alpaca"""
        if self.session:
            await self.session.close()
        self.is_connected = False
        logger.info("Disconnected from Alpaca")
        
    async def get_account_info(self) -> Dict:
        """Get account information"""
        async with self.session.get(f"{self.base_url}/v2/account") as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    'equity': float(data.get('equity', 0)),
                    'cash': float(data.get('cash', 0)),
                    'buying_power': float(data.get('buying_power', 0)),
                    'positions_value': float(data.get('long_market_value', 0)),
                    'day_trade_count': int(data.get('daytrade_count', 0)),
                    'pattern_day_trader': data.get('pattern_day_trader', False)
                }
            else:
                logger.error(f"Failed to get account info: {resp.status}")
                return {}
                
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        async with self.session.get(f"{self.base_url}/v2/positions") as resp:
            if resp.status == 200:
                positions_data = await resp.json()
                return {
                    pos['symbol']: float(pos['qty']) 
                    for pos in positions_data
                }
            else:
                logger.error(f"Failed to get positions: {resp.status}")
                return {}
                
    async def place_order(self, order: Order) -> str:
        """Place an order"""
        # Map order types
        alpaca_order_type = {
            'market': 'market',
            'limit': 'limit',
            'stop': 'stop',
            'stop_limit': 'stop_limit'
        }.get(order.order_type.value, 'market')
        
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": alpaca_order_type,
            "time_in_force": order.time_in_force
        }
        
        if order.price is not None:
            payload["limit_price"] = str(order.price)
        if order.stop_price is not None:
            payload["stop_price"] = str(order.stop_price)
            
        async with self.session.post(f"{self.base_url}/v2/orders", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data['id']
            else:
                error = await resp.text()
                raise Exception(f"Order failed: {error}")
                
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        async with self.session.delete(f"{self.base_url}/v2/orders/{order_id}") as resp:
            if resp.status in [200, 204]:
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}: {resp.status}")
                return False
                
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        async with self.session.get(f"{self.base_url}/v2/orders/{order_id}") as resp:
            if resp.status == 200:
                data = await resp.json()
                
                # Map Alpaca status to our status
                status_map = {
                    'new': OrderStatus.SUBMITTED,
                    'partially_filled': OrderStatus.PARTIAL_FILLED,
                    'filled': OrderStatus.FILLED,
                    'canceled': OrderStatus.CANCELLED,
                    'rejected': OrderStatus.REJECTED,
                    'expired': OrderStatus.EXPIRED
                }
                
                return {
                    'status': status_map.get(data['status'], OrderStatus.PENDING),
                    'filled_qty': float(data.get('filled_qty', 0)),
                    'avg_fill_price': float(data.get('filled_avg_price', 0))
                }
            else:
                return None
                
    async def get_historical_data(self, symbol: str, period: str = "1 D", 
                                 bar_size: str = "15 mins") -> pd.DataFrame:
        """Get historical data from Alpaca"""
        
        # Convert period to start/end times
        import datetime
        end_time = datetime.datetime.now()
        
        # Parse period
        if 'D' in period:
            days = int(period.split()[0])
            start_time = end_time - datetime.timedelta(days=days)
        elif 'M' in period:
            months = int(period.split()[0])
            start_time = end_time - datetime.timedelta(days=months*30)
        else:
            start_time = end_time - datetime.timedelta(days=1)
            
        # Convert bar size to Alpaca timeframe
        timeframe_map = {
            '1 min': '1Min',
            '5 mins': '5Min',
            '15 mins': '15Min',
            '30 mins': '30Min',
            '1 hour': '1Hour',
            '1 day': '1Day'
        }
        
        timeframe = timeframe_map.get(bar_size, '15Min')
        
        # Format times
        start_str = start_time.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')
        
        # Get data
        url = f"{self.data_url}/v2/stocks/{symbol}/bars"
        params = {
            'start': start_str,
            'end': end_str,
            'timeframe': timeframe,
            'limit': 10000
        }
        
        async with self.session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                
                if 'bars' in data:
                    df = pd.DataFrame(data['bars'])
                    
                    # Convert to standard format
                    df['datetime'] = pd.to_datetime(df['t'])
                    df.set_index('datetime', inplace=True)
                    
                    df.rename(columns={
                        'o': 'open',
                        'h': 'high',
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    }, inplace=True)
                    
                    return df[['open', 'high', 'low', 'close', 'volume']]
                else:
                    return pd.DataFrame()
            else:
                logger.error(f"Failed to get historical data: {resp.status}")
                return pd.DataFrame()
                
    async def stream_market_data(self, symbols: List[str], callback):
        """Stream real-time market data"""
        # This would require websocket implementation
        # For now, return a placeholder
        logger.warning("Streaming not implemented in this version")
        pass