"""
Base Strategy Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np
from core.data_structures import TradeSignal

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, strategy_type: str, config: Dict):
        self.strategy_type = strategy_type
        self.config = config
        self.positions = {}
        self.pending_orders = {}
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_losses': 0
        }
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        pass
        
    def _should_generate_signal(self, i: int, signals: pd.DataFrame) -> bool:
        """Check if we should generate a signal at this point"""
        if i > 0 and signals['signal'].iloc[i-1] != 0:
            return False
        
        last_signal_idx = signals[signals['signal'] != 0].index
        if len(last_signal_idx) > 0:
            last_signal_time = last_signal_idx[-1]
            current_time = signals.index[i]
            time_diff = (current_time - last_signal_time).total_seconds() / 60
            
            if time_diff < 60:  # Minimum 60 minutes between signals
                return False
        
        return True