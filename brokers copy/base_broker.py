"""
Base Broker Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
import pandas as pd
from core.data_structures import Order

class IBrokerInterface(ABC):
    """Abstract base class for all broker implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker"""
        pass
        
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass
        
    @abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        pass
        
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place an order"""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
        
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        pass
        
    @abstractmethod
    async def get_historical_data(self, symbol: str, period: str = "1 D", 
                                 bar_size: str = "15 mins") -> pd.DataFrame:
        """Get historical market data"""
        pass
        
    async def stream_market_data(self, symbols: List[str], callback: Callable):
        """Stream real-time market data (optional)"""
        raise NotImplementedError("Streaming not implemented for this broker")
        
    async def get_option_chain(self, symbol: str, expiry: str = None) -> pd.DataFrame:
        """Get options chain (optional)"""
        raise NotImplementedError("Options not implemented for this broker")
        
    async def get_market_hours(self, date: str = None) -> Dict:
        """Get market hours (optional)"""
        raise NotImplementedError("Market hours not implemented for this broker")