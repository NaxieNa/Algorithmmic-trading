"""
Broker Package Initialization
"""

from .base_broker import IBrokerInterface
from .ibkr_broker import IBKRBroker
from .alpaca_broker import AlpacaBroker
from .backtest_broker import BacktestBroker

# Broker factory
def create_broker(broker_type: str, config: dict) -> IBrokerInterface:
    """Factory function to create broker instances"""
    
    brokers = {
        'ibkr': IBKRBroker,
        'alpaca': AlpacaBroker,
        'backtest': BacktestBroker
    }
    
    if broker_type not in brokers:
        raise ValueError(f"Unknown broker type: {broker_type}")
    
    return brokers[broker_type](config)

__all__ = [
    'IBrokerInterface',
    'IBKRBroker', 
    'AlpacaBroker',
    'BacktestBroker',
    'create_broker'
]