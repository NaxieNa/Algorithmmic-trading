"""
Mean Reversion Strategy Implementation
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Optional
from .base_strategy import BaseStrategy
from core.data_structures import TradeSignal

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, config: Dict):
        super().__init__('mean_reversion', config)
        self.mr_entry_z = config.get('mr_entry_z', 1.5)
        self.mr_exit_z = config.get('mr_exit_z', 0.3)
        self.lookback = config.get('lookback', 20)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        
        features = self._calculate_indicators(df)
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['risk_score'] = 0.0
        
        for i in range(max(50, self.lookback), len(df)):
            if self._should_generate_signal(i, signals):
                signal = self._generate_mr_signal(df, features, i)
                
                if signal and signal.direction != 0:
                    signals.loc[df.index[i], 'signal'] = signal.direction
                    signals.loc[df.index[i], 'strength'] = signal.strength
                    
            self._check_exit_conditions(df, signals, i, features)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for mean reversion"""
        
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Price-based indicators
        features['returns'] = close.pct_change()
        
        # RSI
        features['rsi'] = talib.RSI(close, 14)
        features['rsi_fast'] = talib.RSI(close, 7)
        
        # Bollinger Bands
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(close, 20, 2, 2)
        
        # Z-scores for multiple periods
        for period in [10, 20, 30]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'z_score_{period}'] = (close - ma) / (std + 1e-8)
        
        # Volume indicators
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['obv'] = talib.OBV(close, volume)
        
        # ATR for risk management
        features['atr'] = talib.ATR(high, low, close, 14)
        
        # CCI
        features['cci'] = talib.CCI(high, low, close, 14)
        
        # Gap analysis
        features['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['close_open_ratio'] = df['close'] / df['open']
        
        return features
    
    def _generate_mr_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate mean reversion signal"""
        
        z_score = features['z_score_20'].iloc[i]
        rsi = features['rsi'].iloc[i]
        rsi_fast = features['rsi_fast'].iloc[i]
        
        # Bollinger Band position
        price = df['close'].iloc[i]
        bb_upper = features['bb_upper'].iloc[i]
        bb_lower = features['bb_lower'].iloc[i]
        bb_position = (price - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        volume_ratio = features['volume_ratio'].iloc[i]
        
        # Short-term momentum
        returns = features['returns'].iloc[max(0, i-5):i+1]
        short_term_momentum = returns.mean() if len(returns) > 0 else 0
        
        # Buy signal conditions
        if ((z_score < -self.mr_entry_z and rsi < 35) or
            (rsi < 25) or
            (bb_position < 0.1 and z_score < -1.0)):
            
            strength = min(1.0, 
                         (abs(z_score) - self.mr_entry_z) / self.mr_entry_z * 0.3 +
                         (25 - rsi) / 25 * 0.3 +
                         (20 - rsi_fast) / 20 * 0.2 +
                         (0.15 - bb_position) / 0.15 * 0.2)
            
            if volume_ratio > 1.5:
                strength = min(1.0, strength * 1.3)
            elif volume_ratio < 0.8:
                strength *= 0.7
            
            if strength > 0.6:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=1,
                    strength=strength,
                    strategy_type='mean_reversion',
                    indicators={'z_score': z_score, 'rsi': rsi, 'bb_position': bb_position}
                )
        
        # Sell signal conditions
        elif (z_score > self.mr_entry_z and 
              rsi > 75 and
              rsi_fast > 80 and
              bb_position > 0.85 and
              short_term_momentum > 0.001):
            
            strength = min(1.0,
                         (z_score - self.mr_entry_z) / self.mr_entry_z * 0.3 +
                         (rsi - 75) / 25 * 0.3 +
                         (rsi_fast - 80) / 20 * 0.2 +
                         (bb_position - 0.85) / 0.15 * 0.2)
            
            if volume_ratio > 1.5:
                strength = min(1.0, strength * 1.3)
            elif volume_ratio < 0.8:
                strength *= 0.7
            
            if strength > 0.6:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=-1,
                    strength=strength,
                    strategy_type='mean_reversion',
                    indicators={'z_score': z_score, 'rsi': rsi, 'bb_position': bb_position}
                )
        
        return None
    
    def _check_exit_conditions(self, df: pd.DataFrame, signals: pd.DataFrame, i: int, features: pd.DataFrame):
        """Check exit conditions for mean reversion"""
        
        if i == 0:
            return
        
        current_position = signals['signal'].iloc[i-1]
        if current_position == 0:
            return
        
        z_score = features['z_score_20'].iloc[i]
        rsi = features['rsi'].iloc[i]
        
        # Exit conditions for mean reversion
        if current_position == 1:  # Long position
            if z_score > self.mr_exit_z or rsi > 70:
                signals.loc[signals.index[i], 'signal'] = 0
        else:  # Short position
            if z_score < -self.mr_exit_z or rsi < 30:
                signals.loc[signals.index[i], 'signal'] = 0